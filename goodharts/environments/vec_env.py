"""
Vectorized Environment for parallel training.

Runs N environments in a single batched NumPy array for 10-50x speedup.
All operations are vectorized over the batch dimension.

Design:
- Derives all settings from config (no hardcoding)
- Channels derived from ObservationSpec
- Respects CellType enum for cell values
"""
import numpy as np
from typing import Tuple

from goodharts.modes import ObservationSpec
from goodharts.configs.default_config import CellType, get_config
from goodharts.config import get_training_config


# Action deltas: index -> (dx, dy)
# Build from action space for consistency
def _build_action_deltas() -> np.ndarray:
    """Build action deltas matching build_action_space(1) order."""
    from goodharts.behaviors.action_space import build_action_space
    action_space = build_action_space(1)
    return np.array([[dx, dy] for dx, dy in action_space], dtype=np.int32)


class VecEnv:
    """
    Vectorized environment running N parallel simulations.
    
    All state is stored in batched NumPy arrays:
    - grids: (n_envs, height, width) - cell types
    - agent positions and energy
    
    All parameters derived from config - no hardcoding.
    """
    
    def __init__(self, n_envs: int, obs_spec: ObservationSpec, config: dict = None, 
                 shared_grid: bool = False, agent_types: list[int] = None):
        """
        Initialize vectorized environment.
        
        Args:
            n_envs: Number of parallel environments (or agents if shared_grid=True)
            obs_spec: Observation specification (defines view size, channels)
            config: Optional config override (uses get_config() if None)
            shared_grid: If True, all agents inhabit the same world (n_envs becomes n_agents)
            agent_types: Optional list of CellType integer values for each agent (for visibility)
        """
        self.n_envs = n_envs  # batch size (number of agents)
        self.obs_spec = obs_spec
        self.shared_grid = shared_grid
        
        # Get config
        if config is None:
            config = get_config()
        self.config = config
        train_cfg = get_training_config()
        
        # Dimensions from config
        self.width = config.get('GRID_WIDTH', 100)
        self.height = config.get('GRID_HEIGHT', 100)
        self.loop = config.get('WORLD_LOOP', False)
        
        # View settings from obs_spec
        self.view_radius = obs_spec.view_size // 2
        self.view_size = obs_spec.view_size
        self.n_channels = obs_spec.num_channels
        self.channel_names = obs_spec.channel_names
        
        # Agent settings from config (legacy keys)
        self.initial_energy = config.get('ENERGY_START', 50.0)
        self.energy_move_cost = config.get('ENERGY_MOVE_COST', 0.1)
        
        # Training settings
        self.initial_food = config.get('GRID_FOOD_INIT', 500)
        self.poison_count = config.get('GRID_POISON_INIT', 30)
        self.max_steps = train_cfg.get('steps_per_episode', 500)
        
        # Get CellType for rewards
        self.CellType = config['CellType']
        self.food_reward = self.CellType.FOOD.energy_reward
        self.poison_penalty = self.CellType.POISON.energy_penalty
        
        # Agent types for visibility
        if agent_types is None:
            # Default to PREY if not specified
            agent_types = [self.CellType.PREY.value] * n_envs
        self.agent_types = np.array(agent_types, dtype=np.int32)
        
        # Action deltas
        self.action_deltas = _build_action_deltas()
        
        # Batched state arrays
        # If shared_grid, we strictly have 1 grid. If not, we have n_envs grids.
        self.n_grids = 1 if shared_grid else n_envs
        self.grid_indices = np.zeros(n_envs, dtype=np.int32) if shared_grid else np.arange(n_envs, dtype=np.int32)
        
        self.grids = np.zeros((self.n_grids, self.height, self.width), dtype=np.int8)
        self.agent_x = np.zeros(n_envs, dtype=np.int32)
        self.agent_y = np.zeros(n_envs, dtype=np.int32)
        self.agent_energy = np.ones(n_envs, dtype=np.float32) * self.initial_energy
        self.agent_steps = np.zeros(n_envs, dtype=np.int32)
        
        # Pre-allocated view buffer
        self._view_buffer = np.zeros(
            (n_envs, self.n_channels, self.view_size, self.view_size), 
            dtype=np.float32
        )
        
        # Done states
        self.dones = np.zeros(n_envs, dtype=bool)
        
        # Stats tracking
        self.current_episode_food = np.zeros(n_envs, dtype=np.int32)
        self.last_episode_food = np.zeros(n_envs, dtype=np.int32)
        self.current_episode_poison = np.zeros(n_envs, dtype=np.int32)
        self.last_episode_poison = np.zeros(n_envs, dtype=np.int32)
        
        # Reset all environments
        self.reset()
    
    def reset(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Reset environments and return observations."""
        if env_ids is None:
            env_ids = np.arange(self.n_envs)
        
        # If shared grid, we only reset the grid ONCE if any agent resets?
        # Or we respawn agents?
        # Policy: 
        # - Independent grids: Reset grid and agent.
        # - Shared grid: Resetting an agent just respawns the agent. 
        # - BUT: If ALL agents reset (start of sim), we assume we clear grid.
        # - Implementation: We track 'grid_resets'. 
        
        unique_grids = np.unique(self.grid_indices[env_ids])
        
        # Reset grids (clearing food/poison)
        # Note: In shared grid simulation, usually we DON'T clear grid when one agent dies.
        # We only clear grid if explicitly requested or at start.
        # For PPO training (independent), this works fine.
        # For Shared Grid partial resets (agent death): Just respawn agent.
        # For Shared Grid full reset: Reset grid + all agents.
        
        # Heuristic: If we are resetting ALL agents in a shared grid, reset the world.
        if self.shared_grid and len(env_ids) == self.n_envs:
             self._reset_grid(0)
        elif not self.shared_grid:
             for gid in unique_grids:
                 self._reset_grid(gid)

        # Reset agents
        for i in env_ids:
            self._reset_agent(i)
        
        return self._get_observations()
    
    def _reset_grid(self, grid_id: int):
        """Clear and repopulate a specific grid."""
        CellType = self.CellType
        self.grids[grid_id] = CellType.EMPTY.value
        self._place_items(grid_id, CellType.FOOD.value, self.initial_food)
        self._place_items(grid_id, CellType.POISON.value, self.poison_count)

    def _reset_agent(self, env_id: int):
        """Reset a single agent's state."""
        self.last_episode_food[env_id] = self.current_episode_food[env_id]
        self.current_episode_food[env_id] = 0
        self.last_episode_poison[env_id] = self.current_episode_poison[env_id]
        self.current_episode_poison[env_id] = 0
        
        # Random spawn (avoid walls if we had them, but we don't yet)
        self.agent_x[env_id] = np.random.randint(0, self.width)
        self.agent_y[env_id] = np.random.randint(0, self.height)
        self.agent_energy[env_id] = self.initial_energy
        self.agent_steps[env_id] = 0
        self.dones[env_id] = False
    
    def _place_items(self, grid_id: int, cell_type: int, count: int):
        """Place items at random empty positions on specific grid (vectorized)."""
        grid = self.grids[grid_id]
        
        # Find all empty cell indices
        empty_mask = (grid == self.CellType.EMPTY.value)
        empty_indices = np.argwhere(empty_mask)
        
        if len(empty_indices) == 0:
            return
        
        # Sample min(count, available) positions
        n_to_place = min(count, len(empty_indices))
        chosen_idx = np.random.choice(len(empty_indices), size=n_to_place, replace=False)
        chosen_positions = empty_indices[chosen_idx]
        
        # Place all items at once
        grid[chosen_positions[:, 0], chosen_positions[:, 1]] = cell_type
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute batched actions."""
        dx = self.action_deltas[actions, 0]
        dy = self.action_deltas[actions, 1]
        
        if self.loop:
            self.agent_x = (self.agent_x + dx) % self.width
            self.agent_y = (self.agent_y + dy) % self.height
        else:
            self.agent_x = np.clip(self.agent_x + dx, 0, self.width - 1)
            self.agent_y = np.clip(self.agent_y + dy, 0, self.height - 1)
        
        self.agent_energy -= self.energy_move_cost
        
        rewards = self._eat_batch()
        
        self.agent_steps += 1
        self.dones = (self.agent_energy <= 0) | (self.agent_steps >= self.max_steps)
        rewards = np.where(self.dones, rewards - 10.0, rewards)
        
        # Save dones before reset (reset clears the flag)
        dones_to_return = self.dones.copy()
        
        # Auto-reset done agents
        if self.dones.any():
            done_indices = np.where(self.dones)[0]
            for i in done_indices:
                self._reset_agent(i)
        
        return self._get_observations(), rewards, dones_to_return
    
    def _eat_batch(self) -> np.ndarray:
        """Batched eating logic with shared grid support."""
        CellType = self.CellType
        # Determine which cells each agent is on
        # Use grid_indices[i] to pick the correct grid for agent i
        # Since advanced indexing (grids[indices, y, x]) acts like zip, this works for both modes
        cells = self.grids[self.grid_indices, self.agent_y, self.agent_x]
        
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        
        # Identify food/poison
        food_mask = cells == CellType.FOOD.value
        poison_mask = cells == CellType.POISON.value
        
        # Handling Competition in Shared Grid:
        # If multiple agents eat the same food, only one should get it.
        # We can detect duplicates in (grid, y, x)
        if self.shared_grid and (food_mask.any() or poison_mask.any()):
            # Construct scalar IDs for positions: grid*H*W + y*W + x
            pos_ids = self.grid_indices * (self.height * self.width) + self.agent_y * self.width + self.agent_x
            
            # For agents on food, find unique pos_ids. 
            # np.unique with return_index gives index of FIRST occurrence.
            # Only first agent arriving (lowest index) eats.
            # This is a simple tie-breaker.
            
            # Filter for food only
            eaters = np.where(food_mask)[0]
            if len(eaters) > 0:
                food_pos = pos_ids[eaters]
                _, unique_idx = np.unique(food_pos, return_index=True)
                # Filter 'eaters' to keeps only the unique winners
                winners = eaters[unique_idx]
                
                # Update masks: only winners eat
                new_food_mask = np.zeros_like(food_mask)
                new_food_mask[winners] = True
                food_mask = new_food_mask
            
            # Poison is not consumed (usually), so multiple agents can step on it.
            # But earlier code implied poison adds penalty.
            # If poison is consumed (CellType.EMPTY assignment below), then we need same logic.
            # World.eat() implementation shows poison IS consumed ("Respawn consumed resource").
            # So same logic for poison.
            sufferers = np.where(poison_mask)[0]
            if len(sufferers) > 0:
                poison_pos = pos_ids[sufferers]
                _, unique_idx = np.unique(poison_pos, return_index=True)
                winners = sufferers[unique_idx]
                new_poison_mask = np.zeros_like(poison_mask)
                new_poison_mask[winners] = True
                poison_mask = new_poison_mask
                
        # Apply rewards
        rewards[food_mask] = self.food_reward
        self.agent_energy[food_mask] += self.food_reward
        self.current_episode_food[food_mask] += 1
        
        rewards[poison_mask] = -self.poison_penalty
        self.agent_energy[poison_mask] -= self.poison_penalty
        self.current_episode_poison[poison_mask] += 1
        
        # Consumed items
        consumed_mask = food_mask | poison_mask
        
        # Clear consumed cells
        self.grids[self.grid_indices[consumed_mask], 
                   self.agent_y[consumed_mask], 
                   self.agent_x[consumed_mask]] = CellType.EMPTY.value
        
        # Respawn logic
        # For shared grid, we just place item back on THAT grid.
        # PPO default was: 1 item eaten -> 1 item spawned.
        for i in np.where(food_mask)[0]:
            self._place_items(self.grid_indices[i], CellType.FOOD.value, 1)
        for i in np.where(poison_mask)[0]:
            self._place_items(self.grid_indices[i], CellType.POISON.value, 1)
        
        return rewards
    
    def _get_observations(self) -> np.ndarray:
        """
        Get batched observations using fully vectorized extraction.
        """
        r = self.view_radius
        CellType = self.CellType
        
        # MARK AGENTS on the grid so they are visible to each other
        # This overwrites whatever was there (even food), but logic priority is Agent > Item
        # EXCEPT: Agent stands ON food.
        # If we overwrite, the obs shows Agent at (0,0), not Food.
        # But Center cell is always blanked later anyway.
        # Problem: Agent B looks at Agent A. Agent A is on Food.
        # Agent B should see Agent A (Predator/Prey).
        # We should overwrite only EMPTY cells?
        # Simulation logic: "Only mark if cell is empty".
        # Vectorized way:
        # Check cell content at agent pos. Only write if Empty?
        # But for Predator/Prey, knowing there IS an agent is more important than the food under it.
        # So we overwrite.
        
        # We save original values to restore them later
        # Actually, since agent movement doesn't permanently change grid type to Agent,
        # we can just write, extract, and restore.
        # To restore efficiently without saving everything:
        # We know we only wrote to (agent_x, agent_y).
        # But we need to know WHAT was there.
        # So we save `original_cells` at agent positions.
        
        current_vals = self.grids[self.grid_indices, self.agent_y, self.agent_x].copy()
        
        # Write agent types
        self.grids[self.grid_indices, self.agent_y, self.agent_x] = self.agent_types
        
        # Pad all grids
        # Shape: (n_grids, H + 2r, W + 2r)
        pad_mode = 'wrap' if self.loop else 'constant'
        pad_kwargs = {'constant_values': CellType.WALL.value} if not self.loop else {}
        
        padded_grids = np.pad(
            self.grids,
            ((0, 0), (r, r), (r, r)),
            mode=pad_mode,
            **pad_kwargs
        )
        
        # Build index arrays
        env_idx = self.grid_indices[:, None, None] # Use grid indices mapping
        
        # Offset indices
        y_offsets = np.arange(self.view_size)[None, :, None]
        x_offsets = np.arange(self.view_size)[None, None, :]
        
        agent_y_padded = self.agent_y[:, None, None]
        agent_x_padded = self.agent_x[:, None, None]
        
        y_indices = agent_y_padded + y_offsets
        x_indices = agent_x_padded + x_offsets
        
        # Extract views
        grid_views = padded_grids[
            env_idx,
            y_indices,
            x_indices
        ]
        
        # Convert to channels
        # Supports two channel name formats:
        # 1. 'cell_<name>' - one-hot encoding for a specific CellType (e.g. 'cell_food')
        # 2. '<property>' - continuous value from CellTypeInfo (e.g. 'interestingness')
        # This extensible design supports adding new cell types or properties without code changes.
        property_names = CellType.all_types()[0].property_names()  # Get all valid property names
        
        for c, name in enumerate(self.channel_names):
            if name.startswith('cell_'):
                # One-hot encoding for specific cell type
                cell_name = name[5:].upper()
                try:
                    cell_type = getattr(CellType, cell_name)
                    self._view_buffer[:, c] = (grid_views == cell_type.value).astype(np.float32)
                except AttributeError:
                    self._view_buffer[:, c] = 0.0
            elif name in property_names:
                # Continuous property value (e.g. 'interestingness', 'energy_reward')
                # Build a lookup table: cell_value -> property_value
                prop_values = np.zeros_like(grid_views, dtype=np.float32)
                for ct in CellType.all_types():
                    mask = (grid_views == ct.value)
                    prop_values[mask] = ct.get_property(name)
                self._view_buffer[:, c] = prop_values
            else:
                self._view_buffer[:, c] = 0.0
        
        # Blank center cell
        self._view_buffer[:, :, r, r] = 0.0
        
        # Restore grid state (un-mark agents)
        self.grids[self.grid_indices, self.agent_y, self.agent_x] = current_vals
        
        return self._view_buffer.copy()


def create_vec_env(n_envs: int, obs_spec: ObservationSpec, config: dict = None, 
                   shared_grid: bool = False, agent_types: list[int] = None) -> VecEnv:
    """
    Factory function to create a vectorized environment.
    """
    return VecEnv(n_envs, obs_spec, config, shared_grid, agent_types)
