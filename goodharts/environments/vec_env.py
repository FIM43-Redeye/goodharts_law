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

from goodharts.configs.observation_spec import ObservationSpec
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
    
    def __init__(self, n_envs: int, obs_spec: ObservationSpec, config: dict = None):
        """
        Initialize vectorized environment.
        
        Args:
            n_envs: Number of parallel environments
            obs_spec: Observation specification (defines view size, channels)
            config: Optional config override (uses get_config() if None)
        """
        self.n_envs = n_envs
        self.obs_spec = obs_spec
        
        # Get config
        if config is None:
            config = get_config()
        self.config = config
        train_cfg = get_training_config()
        
        # Dimensions from config
        world_cfg = config.get('world', {})
        self.width = world_cfg.get('width', 100)
        self.height = world_cfg.get('height', 100)
        
        # View settings from obs_spec
        self.view_radius = obs_spec.view_size // 2
        self.view_size = obs_spec.view_size
        self.n_channels = obs_spec.num_channels
        self.channel_names = obs_spec.channel_names
        
        # Agent settings from config
        agent_cfg = config.get('agent', {})
        self.initial_energy = agent_cfg.get('starting_energy', 100.0)
        self.energy_move_cost = agent_cfg.get('move_cost', 0.1)
        
        # Training settings
        self.initial_food = train_cfg.get('initial_food', 500)
        self.poison_count = train_cfg.get('poison_count', 30)
        
        # Get CellType for rewards
        self.CellType = config['CellType']
        self.food_reward = self.CellType.FOOD.energy_reward
        self.poison_penalty = self.CellType.POISON.energy_penalty
        
        # Action deltas
        self.action_deltas = _build_action_deltas()
        
        # Batched state arrays
        self.grids = np.zeros((n_envs, self.height, self.width), dtype=np.int8)
        self.agent_x = np.zeros(n_envs, dtype=np.int32)
        self.agent_y = np.zeros(n_envs, dtype=np.int32)
        self.agent_energy = np.ones(n_envs, dtype=np.float32) * self.initial_energy
        
        # Pre-allocated view buffer
        self._view_buffer = np.zeros(
            (n_envs, self.n_channels, self.view_size, self.view_size), 
            dtype=np.float32
        )
        
        # Done states
        self.dones = np.zeros(n_envs, dtype=bool)
        
        # Reset all environments
        self.reset()
    
    def reset(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Reset environments and return observations."""
        if env_ids is None:
            env_ids = np.arange(self.n_envs)
        
        for i in env_ids:
            self._reset_single(i)
        
        return self._get_observations()
    
    def _reset_single(self, env_id: int):
        """Reset a single environment."""
        CellType = self.CellType
        
        # Clear grid
        self.grids[env_id] = CellType.EMPTY.value
        
        # Place food
        self._place_items(env_id, CellType.FOOD.value, self.initial_food)
        
        # Place poison
        self._place_items(env_id, CellType.POISON.value, self.poison_count)
        
        # Reset agent to random position
        self.agent_x[env_id] = np.random.randint(0, self.width)
        self.agent_y[env_id] = np.random.randint(0, self.height)
        self.agent_energy[env_id] = self.initial_energy
        self.dones[env_id] = False
    
    def _place_items(self, env_id: int, cell_type: int, count: int):
        """Place items at random empty positions."""
        placed = 0
        attempts = 0
        max_attempts = count * 3
        
        while placed < count and attempts < max_attempts:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.grids[env_id, y, x] == self.CellType.EMPTY.value:
                self.grids[env_id, y, x] = cell_type
                placed += 1
            attempts += 1
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute batched actions and return (observations, rewards, dones)."""
        # Get movement deltas
        dx = self.action_deltas[actions, 0]
        dy = self.action_deltas[actions, 1]
        
        # Move agents (bounded)
        self.agent_x = np.clip(self.agent_x + dx, 0, self.width - 1)
        self.agent_y = np.clip(self.agent_y + dy, 0, self.height - 1)
        
        # Apply movement cost
        self.agent_energy -= self.energy_move_cost
        
        # Batched eating
        rewards = self._eat_batch()
        
        # Check done
        self.dones = self.agent_energy <= 0
        
        # Death penalty
        rewards = np.where(self.dones, rewards - 10.0, rewards)
        
        return self._get_observations(), rewards, self.dones.copy()
    
    def _eat_batch(self) -> np.ndarray:
        """Batched eating at agent positions."""
        CellType = self.CellType
        cells = self.grids[np.arange(self.n_envs), self.agent_y, self.agent_x]
        
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        
        # Food
        food_mask = cells == CellType.FOOD.value
        rewards[food_mask] = self.food_reward
        self.agent_energy[food_mask] += self.food_reward
        
        # Poison
        poison_mask = cells == CellType.POISON.value
        rewards[poison_mask] = -self.poison_penalty
        self.agent_energy[poison_mask] -= self.poison_penalty
        
        # Clear consumed cells
        consumed_mask = food_mask | poison_mask
        self.grids[np.arange(self.n_envs)[consumed_mask], 
                   self.agent_y[consumed_mask], 
                   self.agent_x[consumed_mask]] = CellType.EMPTY.value
        
        # Respawn consumed items
        for i in np.where(food_mask)[0]:
            self._place_items(i, CellType.FOOD.value, 1)
        for i in np.where(poison_mask)[0]:
            self._place_items(i, CellType.POISON.value, 1)
        
        return rewards
    
    def _get_observations(self) -> np.ndarray:
        """
        Get batched observations using fully vectorized extraction.
        
        Uses padded grids to avoid boundary handling per-environment.
        """
        r = self.view_radius
        CellType = self.CellType
        
        # Pad all grids with WALL value for boundary handling
        # Shape: (n_envs, H + 2r, W + 2r)
        padded_grids = np.pad(
            self.grids,
            ((0, 0), (r, r), (r, r)),
            mode='constant',
            constant_values=CellType.WALL.value
        )
        
        # Compute extraction indices for each environment
        # After padding, agent position (x, y) becomes (x+r, y+r)
        # Extract view centered at that position
        
        # Build index arrays for all environments at once
        env_idx = np.arange(self.n_envs)[:, None, None]  # (n_envs, 1, 1)
        
        # Offset indices for view extraction
        y_offsets = np.arange(self.view_size)[None, :, None]  # (1, view_size, 1)
        x_offsets = np.arange(self.view_size)[None, None, :]  # (1, 1, view_size)
        
        # Agent positions (shifted by padding radius)
        agent_y_padded = self.agent_y[:, None, None]  # (n_envs, 1, 1)
        agent_x_padded = self.agent_x[:, None, None]  # (n_envs, 1, 1)
        
        # Absolute indices in padded grid
        y_indices = agent_y_padded + y_offsets  # (n_envs, view_size, 1)
        x_indices = agent_x_padded + x_offsets  # (n_envs, 1, view_size)
        
        # Extract views: (n_envs, view_size, view_size)
        grid_views = padded_grids[
            env_idx.squeeze(-1).squeeze(-1)[:, None, None],
            y_indices,
            x_indices
        ]
        
        # Convert to one-hot channels for all envs at once
        # Shape: (n_envs, n_channels, view_size, view_size)
        for c, name in enumerate(self.channel_names):
            if name.startswith('cell_'):
                cell_name = name[5:].upper()
                try:
                    cell_type = getattr(CellType, cell_name)
                    self._view_buffer[:, c] = (grid_views == cell_type.value).astype(np.float32)
                except AttributeError:
                    self._view_buffer[:, c] = 0.0
            else:
                self._view_buffer[:, c] = 0.0
        
        # Blank center cell for all environments
        self._view_buffer[:, :, r, r] = 0.0
        
        return self._view_buffer.copy()


def create_vec_env(n_envs: int, obs_spec: ObservationSpec, config: dict = None) -> VecEnv:
    """
    Factory function to create a vectorized environment.
    
    Args:
        n_envs: Number of parallel environments
        obs_spec: Observation specification
        config: Optional config override
    
    Returns:
        Configured VecEnv instance
    """
    return VecEnv(n_envs, obs_spec, config)
