"""
GPU-Native Vectorized Environment using PyTorch tensors.

This is a drop-in replacement for VecEnv that keeps all state on GPU,
eliminating CPU-GPU transfer overhead during training.

All operations are vectorized over the batch dimension using PyTorch ops.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from goodharts.modes import ObservationSpec
from goodharts.configs.default_config import CellType, get_config
from goodharts.config import get_training_config
from goodharts.utils.device import get_device


# Build action deltas as a torch tensor
def _build_action_deltas(device: torch.device) -> torch.Tensor:
    """Build action delta tensor matching build_action_space(1) order."""
    from goodharts.behaviors.action_space import build_action_space
    actions = build_action_space(1)  # Get action list
    deltas = []
    for action in actions:
        if isinstance(action, (tuple, list)) and len(action) >= 2:
            deltas.append([action[0], action[1]])
        elif hasattr(action, 'dx') and hasattr(action, 'dy'):
            deltas.append([action.dx, action.dy])
        else:
            deltas.append([0, 0])  # No-op or unknown
    return torch.tensor(deltas, dtype=torch.int32, device=device)


class TorchVecEnv:
    """
    GPU-native vectorized environment.
    
    All state is stored in PyTorch tensors on the specified device.
    Observations are returned as GPU tensors (no CPU transfer).
    
    API matches VecEnv for drop-in replacement.
    """
    
    def __init__(
        self, 
        n_envs: int, 
        obs_spec: ObservationSpec, 
        config: dict = None,
        device: Optional[torch.device] = None,
        shared_grid: bool = False,
        agent_types: list[int] = None
    ):
        """
        Initialize GPU-native vectorized environment.
        
        Args:
            n_envs: Number of parallel environments
            obs_spec: Observation specification
            config: Optional config override
            device: Torch device (auto-detect if None)
            shared_grid: If True, all agents share one grid
            agent_types: Optional list of agent types
        """
        self.n_envs = n_envs
        self.obs_spec = obs_spec
        self.shared_grid = shared_grid
        
        # Device selection - use centralized get_device()
        if device is None:
            device = get_device()
        self.device = device

        # Get config - all values required from TOML
        if config is None:
            config = get_config()
        self.config = config
        train_cfg = get_training_config()

        # Dimensions from config (required)
        self.width = config['GRID_WIDTH']
        self.height = config['GRID_HEIGHT']
        self.loop = config['WORLD_LOOP']
        
        # View settings from obs_spec
        self.view_radius = obs_spec.view_size // 2
        self.view_size = obs_spec.view_size
        self.n_channels = obs_spec.num_channels
        self.channel_names = obs_spec.channel_names
        
        # Mode-aware observation encoding
        # Detect proxy mode: uses 'interestingness' instead of cell-type one-hot
        self.is_proxy_mode = 'interestingness' in self.channel_names
        
        # Build interestingness lookup table: cell_value -> interestingness
        # Used for proxy mode observations
        cell_types = CellType.all_types()
        n_cell_types = len(cell_types)
        self._interestingness_lut = torch.zeros(n_cell_types, dtype=torch.float32, device=device)
        for ct in cell_types:
            self._interestingness_lut[ct.value] = ct.interestingness
        
        # Agent settings (required)
        self.initial_energy = config['ENERGY_START']
        self.energy_move_cost = config['ENERGY_MOVE_COST']

        # Training settings (required)
        default_food = config['GRID_FOOD_INIT']
        default_poison = config['GRID_POISON_INIT']
        self.max_steps = train_cfg.get('steps_per_episode', 500)
        
        # Curriculum ranges
        self.food_range = (default_food, default_food)
        self.poison_range = (default_poison, default_poison)
        self._default_food = default_food
        self._default_poison = default_poison
        
        # CellType enum - all cell values accessed dynamically from this
        self.CellType = config['CellType']
        self.food_reward = self.CellType.FOOD.energy_reward
        self.poison_penalty = self.CellType.POISON.energy_penalty
        
        # Agent types
        if agent_types is None:
            agent_types = [self.CellType.PREY.value] * n_envs
        self.agent_types = torch.tensor(agent_types, dtype=torch.int32, device=device)
        
        # Action deltas
        self.action_deltas = _build_action_deltas(device)
        
        # Grid setup
        self.n_grids = 1 if shared_grid else n_envs
        self.grid_indices = torch.zeros(n_envs, dtype=torch.long, device=device) if shared_grid else torch.arange(n_envs, device=device)
        
        # Per-grid counts
        self.grid_food_counts = torch.full((self.n_grids,), self._default_food, dtype=torch.int32, device=device)
        self.grid_poison_counts = torch.full((self.n_grids,), self._default_poison, dtype=torch.int32, device=device)
        
        # State tensors (all on GPU)
        # Grid uses float32 for faster F.pad (avoids dtype conversion overhead)
        self.grids = torch.zeros((self.n_grids, self.height, self.width), dtype=torch.float32, device=device)
        self.agent_x = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self.agent_y = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self.agent_energy = torch.full((n_envs,), self.initial_energy, dtype=torch.float32, device=device)
        self.agent_steps = torch.zeros(n_envs, dtype=torch.int32, device=device)
        
        # Pre-allocated view buffer
        self._view_buffer = torch.zeros(
            (n_envs, self.n_channels, self.view_size, self.view_size),
            dtype=torch.float32, device=device
        )
        
        # Done states
        self.dones = torch.zeros(n_envs, dtype=torch.bool, device=device)
        
        # Stats tracking
        self.current_episode_food = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self.last_episode_food = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self.current_episode_poison = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self.last_episode_poison = torch.zeros(n_envs, dtype=torch.int32, device=device)
        
        # Initialize
        self.reset()
    
    def set_curriculum_ranges(self, food_min: int, food_max: int, 
                               poison_min: int, poison_max: int):
        """Set curriculum ranges for per-environment randomization."""
        self.food_range = (food_min, food_max)
        self.poison_range = (poison_min, poison_max)
    
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset environments and return observations."""
        if env_ids is None:
            # Reset all
            for grid_id in range(self.n_grids):
                self._reset_grid(grid_id)
            for env_id in range(self.n_envs):
                self._reset_agent(env_id)
        else:
            # Reset specific envs
            done_grids = set()
            for env_id in env_ids.tolist():
                grid_id = self.grid_indices[env_id].item()
                if grid_id not in done_grids and not self.shared_grid:
                    self._reset_grid(grid_id)
                    done_grids.add(grid_id)
                self._reset_agent(env_id)
        
        return self._get_observations()
    
    def _reset_grid(self, grid_id: int):
        """Clear and repopulate a specific grid."""
        self.grids[grid_id].fill_(self.CellType.EMPTY.value)
        
        # Randomize counts - use int32 for XLA/TPU compatibility
        food_count = torch.randint(
            self.food_range[0], self.food_range[1] + 1, (1,),
            dtype=torch.int32, device=self.device
        ).item()
        poison_count = torch.randint(
            self.poison_range[0], self.poison_range[1] + 1, (1,),
            dtype=torch.int32, device=self.device
        ).item()
        
        self.grid_food_counts[grid_id] = food_count
        self.grid_poison_counts[grid_id] = poison_count
        
        # Place items
        self._place_items(grid_id, self.CellType.FOOD.value, food_count)
        self._place_items(grid_id, self.CellType.POISON.value, poison_count)
    
    def _reset_agent(self, env_id: int):
        """Reset a single agent's state."""
        grid_id = self.grid_indices[env_id].item()
        
        # Find empty positions - use as_tuple for XLA compatibility
        empty_mask = (self.grids[grid_id] == self.CellType.EMPTY.value)
        empty_y, empty_x = empty_mask.nonzero(as_tuple=True)
        n_empty = empty_y.shape[0]
        
        if n_empty > 0:
            idx = torch.randint(n_empty, (1,), dtype=torch.int32, device=self.device).item()
            self.agent_y[env_id] = empty_y[idx]
            self.agent_x[env_id] = empty_x[idx]
        
        self.agent_energy[env_id] = self.initial_energy
        self.agent_steps[env_id] = 0
        self.dones[env_id] = False
        
        # Update stats
        self.last_episode_food[env_id] = self.current_episode_food[env_id]
        self.last_episode_poison[env_id] = self.current_episode_poison[env_id]
        self.current_episode_food[env_id] = 0
        self.current_episode_poison[env_id] = 0
    
    def _reset_agents_batch(self, env_ids: torch.Tensor):
        """Vectorized batch reset for multiple agents - minimal GPU syncs."""
        if len(env_ids) == 0:
            return
        
        n_reset = len(env_ids)
        
        # Update stats (vectorized)
        self.last_episode_food[env_ids] = self.current_episode_food[env_ids]
        self.last_episode_poison[env_ids] = self.current_episode_poison[env_ids]
        self.current_episode_food[env_ids] = 0
        self.current_episode_poison[env_ids] = 0
        
        # Reset energy, steps, dones (vectorized)
        self.agent_energy[env_ids] = self.initial_energy
        self.agent_steps[env_ids] = 0
        self.dones[env_ids] = False
        
        # Random spawn positions - sample uniformly and check for empty
        # This is approximate but avoids per-agent loops
        grid_ids = self.grid_indices[env_ids]
        
        # Generate random positions (int32 to match agent_y/x dtype)
        rand_y = torch.randint(0, self.height, (n_reset,), dtype=torch.int32, device=self.device)
        rand_x = torch.randint(0, self.width, (n_reset,), dtype=torch.int32, device=self.device)
        
        # Assign positions (if occupied, agent spawns there anyway - rare edge case)
        self.agent_y[env_ids] = rand_y
        self.agent_x[env_ids] = rand_x
    
    def _place_items(self, grid_id: int, cell_type: int, count: int):
        """Place items at random empty positions."""
        if count <= 0:
            return
        
        empty_mask = (self.grids[grid_id] == self.CellType.EMPTY.value)
        # Use as_tuple for XLA compatibility (returns tuple of 1D tensors)
        empty_y, empty_x = empty_mask.nonzero(as_tuple=True)
        n_empty = empty_y.shape[0]
        
        if n_empty == 0:
            return
        
        # Randomly select positions - use int32 for XLA/TPU compatibility
        n_to_place = min(count, n_empty)
        # Generate random indices (with replacement to avoid int64 randperm)
        # For small counts vs large pool, duplicates are rare and handled by overwriting
        indices = torch.randint(n_empty, (n_to_place,), dtype=torch.int32, device=self.device)
        
        # Index and place items
        chosen_y = empty_y[indices.long()]  # Index with long for PyTorch compat
        chosen_x = empty_x[indices.long()]
        self.grids[grid_id, chosen_y, chosen_x] = cell_type
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute batched actions."""
        # Ensure actions are on device
        if actions.device != self.device:
            actions = actions.to(self.device)
        
        # Get movement deltas
        dx = self.action_deltas[actions, 0]
        dy = self.action_deltas[actions, 1]
        
        # Move agents
        if self.loop:
            self.agent_x = (self.agent_x + dx) % self.width
            self.agent_y = (self.agent_y + dy) % self.height
        else:
            self.agent_x = torch.clamp(self.agent_x + dx, 0, self.width - 1)
            self.agent_y = torch.clamp(self.agent_y + dy, 0, self.height - 1)
        
        # Energy cost
        self.agent_energy -= self.energy_move_cost
        
        # Eating
        rewards = self._eat_batch()
        
        # Step count and done check
        self.agent_steps += 1
        self.dones = (self.agent_energy <= 0) | (self.agent_steps >= self.max_steps)
        
        # Death penalty
        # Death penalty (ONLY for starvation, not timeout)
        rewards = torch.where(self.agent_energy <= 0, rewards - 10.0, rewards)
        
        # Movement cost (living penalty) - crucial for sparse reward learning!
        rewards -= self.energy_move_cost
        
        # Save dones before reset
        dones_to_return = self.dones.clone()
        
        # Auto-reset done agents (vectorized - no Python loop)
        if self.dones.any():
            done_indices = self.dones.nonzero(as_tuple=True)[0]
            self._reset_agents_batch(done_indices)
        
        return self._get_observations(), rewards, dones_to_return
    
    def _eat_batch(self) -> torch.Tensor:
        """Fully vectorized eating logic - minimized syncs (no .any() checks)."""
        rewards = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
        
        # Get cell values at agent positions (vectorized)
        agent_y_long = self.agent_y.long()
        agent_x_long = self.agent_x.long()
        cell_values = self.grids[self.grid_indices, agent_y_long, agent_x_long]
        
        # Food - unconditional execution with masking (avoids .any() sync)
        food_mask = (cell_values == self.CellType.FOOD.value)
        
        # Apply food effects masked
        rewards = torch.where(food_mask, rewards + self.food_reward, rewards)
        
        # Only update energy/count where food was eaten
        # We can use index_addish logic or just masked update since we have per-agent state
        self.agent_energy = torch.where(food_mask, self.agent_energy + self.food_reward, self.agent_energy)
        self.current_episode_food = torch.where(food_mask, self.current_episode_food + 1, self.current_episode_food)
        
        # Clear eaten food (mask write)
        # We need grid_indices for scatter
        # scatter_ or index_put_ works with masks?
        # grids[ind, y, x] = val works with boolean mask indexing, but that causes sync for indices?
        # No, grids[...] = val is purely GPU if indices are GPU tensors.
        # But `grids[grid_indices[food_mask], ...]` creates dynamic shape -> sync?
        # YES. `grid_indices[food_mask]` is dynamic shape.
        # FIX: Use `torch.where` on the WHOLE grid update? Too expensive (copy whole grid).
        # Better: masked_scatter?
        # Actually, `self.grids[self.grid_indices, agent_y_long, agent_x_long] = selected_vals`
        # We can construct the "new value" for the cell:
        # new_val = where(food_mask, EMPTY, old_val)
        # Then write back to ALL agent positions.
        # (Overwriting non-food cells with themselves is fine/identity).
        
        new_cell_values = torch.where(food_mask, torch.tensor(self.CellType.EMPTY.value, device=self.device, dtype=torch.float32), cell_values)
        self.grids[self.grid_indices, agent_y_long, agent_x_long] = new_cell_values
        
        # Respawn food - pass mask, internal logic must be sync-free
        self._respawn_items_vectorized(food_mask, self.CellType.FOOD.value)
        
        # Poison - same pattern
        # Re-read cell values? No, we just updated them to empty if food.
        # If it was poison, it wasn't food, so it's still poison in 'new_cell_values'?
        # Wait, if we overwrote with new_cell_values, we need to base poison check on ORIGINAL cell_values.
        
        poison_mask = (cell_values == self.CellType.POISON.value)
        
        rewards = torch.where(poison_mask, rewards - self.poison_penalty, rewards)
        self.agent_energy = torch.where(poison_mask, self.agent_energy - self.poison_penalty, self.agent_energy)
        self.current_episode_poison = torch.where(poison_mask, self.current_episode_poison + 1, self.current_episode_poison)
        
        # Clear poison (if eaten)
        # Update our local 'new_cell_values' to reflect poison removal too
        # If it was poison, become EMPTY. Else keep what it was (EMPTY or WALL etc)
        # Note: 'new_cell_values' currently has EMPTY where food was, and ORIGINAL where food wasn't.
        # We want to apply poison clearing on top.
        
        final_cell_values = torch.where(poison_mask, torch.tensor(self.CellType.EMPTY.value, device=self.device, dtype=torch.float32), new_cell_values)
        
        # Write back (updates both food and poison removals in one go)
        self.grids[self.grid_indices, agent_y_long, agent_x_long] = final_cell_values
        
        self._respawn_items_vectorized(poison_mask, self.CellType.POISON.value)
        
        return rewards
    
    def _respawn_items_vectorized(self, eaten_mask: torch.Tensor, cell_type: int):
        """
        Fully vectorized respawn - shape-static-ish (masked) to avoid ghosts.
        Uses nonzero() for writing but avoids CPU syncs (no shape checks).
        """
        # grid_ids: We use self.grid_indices directly (static shape N_envs)
        N = self.n_envs
        K = 20 # Reduced from 100 (sufficient for 99% success)
        
        # Generate K candidates for ALL envs: (N, K)
        rand_y = torch.randint(0, self.height, (N, K), device=self.device)
        rand_x = torch.randint(0, self.width, (N, K), device=self.device)
        
        # Check candidates
        grid_read_ids = self.grid_indices.unsqueeze(1)
        vals = self.grids[grid_read_ids, rand_y, rand_x]
        
        # Find valid (empty) spots
        is_empty = (vals == self.CellType.EMPTY.value)  # (N, K) bool
        
        # Select first valid candidate (argmax returns index of first True)
        first_valid_idx = is_empty.int().argmax(dim=1)  # (N,) indices in [0, K-1]
        
        # Check validity (did we find ANY?)
        # Efficient check: gather the value at the chosen index
        range_n = torch.arange(N, device=self.device)
        
        # Extract chosen coords
        chosen_y = rand_y[range_n, first_valid_idx]
        chosen_x = rand_x[range_n, first_valid_idx]
        
        # Verify if the chosen candidate is valid
        # We need to re-check 'is_empty' at the chosen index
        # Or implicitly: has_valid = is_empty.any(dim=1)
        has_valid = is_empty.any(dim=1) # (N,) bool
        
        # FINAL WRITE MASK
        # We write IF:
        # 1. It was eaten (eaten_mask)
        # 2. We found a valid spot (has_valid)
        do_write = eaten_mask & has_valid
        
        # Sync-free Write:
        # Use nonzero() to get indices, but DO NOT READ size on CPU.
        # Just use the result for advanced indexing.
        # This keeps execution on GPU.
        
        write_indices = do_write.nonzero(as_tuple=True)[0]
        
        # Advanced indexing with tensors filter writes to only valid items
        # grids[grid_ids[idx], y[idx], x[idx]] = val
        
        target_grids = self.grid_indices[write_indices]
        target_y = chosen_y[write_indices]
        target_x = chosen_x[write_indices]
        
        self.grids[target_grids, target_y, target_x] = float(cell_type)
    
    def _get_observations(self) -> torch.Tensor:
        """Get batched observations (fully vectorized on GPU).
        
        Mode-aware encoding:
        - Ground truth: One-hot encoding of cell types (channels = cell types)
        - Proxy modes: Channels 0-1 are empty/wall, channels 2+ are interestingness
        """
        r = self.view_radius
        vs = self.view_size
        
        # Mark agents on grid (save original values)
        agent_y_long = self.agent_y.long()
        agent_x_long = self.agent_x.long()
        current_vals = self.grids[self.grid_indices, agent_y_long, agent_x_long].clone()
        self.grids[self.grid_indices, agent_y_long, agent_x_long] = self.agent_types.float()
        
        # Pad grids (grid is float32, no conversion needed)
        if self.loop:
            padded_grids = F.pad(self.grids, (r, r, r, r), mode='circular')
        else:
            padded_grids = F.pad(
                self.grids, (r, r, r, r), 
                mode='constant', value=float(self.CellType.WALL.value)
            )
        
        # VECTORIZED VIEW EXTRACTION using unfold
        windows = padded_grids.unfold(1, vs, 1).unfold(2, vs, 1)
        
        # Index by agent positions to get each agent's view
        views = windows[self.grid_indices, agent_y_long, agent_x_long]  # (n_envs, vs, vs)
        
        if self.is_proxy_mode:
            # PROXY MODE: Hide ground truth, show interestingness
            # Channel 0: is_empty (binary)
            # Channel 1: is_wall (binary)  
            # Channels 2+: interestingness value (same across all these channels)
            
            views_long = views.long()
            
            # Binary channels for empty/wall
            self._view_buffer[:, 0, :, :] = (views_long == self.CellType.EMPTY.value).float()
            self._view_buffer[:, 1, :, :] = (views_long == self.CellType.WALL.value).float()
            
            # Interestingness channel (lookup from cell values)
            # Clamp to valid indices to avoid index errors
            clamped_views = views_long.clamp(0, len(self._interestingness_lut) - 1)
            interestingness = self._interestingness_lut[clamped_views]  # (n_envs, vs, vs)
            
            # Fill channels 2+ with interestingness (single vectorized op, no loop)
            n_interest_channels = self.n_channels - 2
            # Expand interestingness to (n_envs, n_interest_channels, vs, vs) and write
            self._view_buffer[:, 2:, :, :] = interestingness.unsqueeze(1).expand(-1, n_interest_channels, -1, -1)
        else:
            # GROUND TRUTH MODE: One-hot encoding of cell types
            one_hot = F.one_hot(views.long(), num_classes=self.n_channels)
            self._view_buffer[:] = one_hot.permute(0, 3, 1, 2).float()
        
        # Blank center (agent's own cell)
        self._view_buffer[:, :, r, r] = 0.0
        
        # Restore grid
        self.grids[self.grid_indices, agent_y_long, agent_x_long] = current_vals
        
        return self._view_buffer


def create_torch_vec_env(
    n_envs: int, 
    obs_spec: ObservationSpec, 
    config: dict = None,
    device: Optional[torch.device] = None,
    shared_grid: bool = False,
    agent_types: list[int] = None
) -> TorchVecEnv:
    """Factory function to create a GPU-native vectorized environment."""
    return TorchVecEnv(
        n_envs=n_envs,
        obs_spec=obs_spec,
        config=config,
        device=device,
        shared_grid=shared_grid,
        agent_types=agent_types
    )
