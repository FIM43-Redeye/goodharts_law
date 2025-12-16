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


# Build action deltas as a torch tensor
def _build_action_deltas(device: torch.device) -> torch.Tensor:
    """Build action delta tensor matching build_action_space(1) order."""
    from goodharts.behaviors.action_space import build_action_space
    actions = build_action_space(1)  # Get action list
    deltas = []
    for action in actions:
        if hasattr(action, 'dx') and hasattr(action, 'dy'):
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
        
        # Device selection
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
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
        
        # Agent settings
        self.initial_energy = config.get('ENERGY_START', 50.0)
        self.energy_move_cost = config.get('ENERGY_MOVE_COST', 0.1)
        
        # Training settings
        default_food = config.get('GRID_FOOD_INIT', 500)
        default_poison = config.get('GRID_POISON_INIT', 30)
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
        self.grids = torch.zeros((self.n_grids, self.height, self.width), dtype=torch.int8, device=device)
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
        
        # Randomize counts
        food_count = torch.randint(
            self.food_range[0], self.food_range[1] + 1, (1,),
            device=self.device
        ).item()
        poison_count = torch.randint(
            self.poison_range[0], self.poison_range[1] + 1, (1,),
            device=self.device
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
            idx = torch.randint(n_empty, (1,), device=self.device).item()
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
        # randperm doesn't support int32 directly, so we generate and cast
        perm = torch.randperm(n_empty, device=self.device)[:n_to_place]
        
        # Index and place items
        chosen_y = empty_y[perm]
        chosen_x = empty_x[perm]
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
        rewards = torch.where(self.dones, rewards - 10.0, rewards)
        
        # Save dones before reset
        dones_to_return = self.dones.clone()
        
        # Auto-reset done agents
        if self.dones.any():
            done_indices = self.dones.nonzero(as_tuple=True)[0]
            for i in done_indices.tolist():
                self._reset_agent(i)
        
        return self._get_observations(), rewards, dones_to_return
    
    def _eat_batch(self) -> torch.Tensor:
        """Fully vectorized eating logic - no Python loops."""
        rewards = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
        
        # Get cell values at agent positions
        agent_y_long = self.agent_y.long()
        agent_x_long = self.agent_x.long()
        cell_values = self.grids[self.grid_indices, agent_y_long, agent_x_long]
        
        # Food - fully vectorized
        food_mask = (cell_values == self.CellType.FOOD.value)
        n_food = food_mask.sum().item()
        if n_food > 0:
            rewards[food_mask] += self.food_reward
            self.agent_energy[food_mask] += self.food_reward
            self.current_episode_food[food_mask] += 1
            
            # Clear eaten food (vectorized)
            self.grids[self.grid_indices[food_mask], agent_y_long[food_mask], agent_x_long[food_mask]] = self.CellType.EMPTY.value
            
            # Respawn food (simplified - respawn to random positions)
            self._respawn_items_vectorized(food_mask, self.CellType.FOOD.value)
        
        # Poison - fully vectorized
        poison_mask = (cell_values == self.CellType.POISON.value)
        n_poison = poison_mask.sum().item()
        if n_poison > 0:
            rewards[poison_mask] -= self.poison_penalty
            self.agent_energy[poison_mask] -= self.poison_penalty
            self.current_episode_poison[poison_mask] += 1
            
            # Clear eaten poison (vectorized)
            self.grids[self.grid_indices[poison_mask], agent_y_long[poison_mask], agent_x_long[poison_mask]] = self.CellType.EMPTY.value
            
            # Respawn poison
            self._respawn_items_vectorized(poison_mask, self.CellType.POISON.value)
        
        return rewards
    
    def _respawn_items_vectorized(self, eaten_mask: torch.Tensor, cell_type: int):
        """Respawn items - simplified vectorized version."""
        # For each grid, count how many items were eaten and respawn that many
        if not eaten_mask.any():
            return
        
        # Group by grid using bincount
        grid_ids = self.grid_indices[eaten_mask]
        counts = torch.bincount(grid_ids, minlength=self.n_grids)
        
        # Respawn for each grid that needs it
        for grid_id in counts.nonzero(as_tuple=True)[0].tolist():
            count = counts[grid_id].item()
            self._place_items(grid_id, cell_type, count)
    
    def _respawn_items_batched(self, eaten_mask: torch.Tensor, cell_type: int):
        """Respawn items for grids that had items eaten."""
        # Group by grid
        grid_counts = {}
        for i in eaten_mask.nonzero(as_tuple=True)[0].tolist():
            grid_id = self.grid_indices[i].item()
            grid_counts[grid_id] = grid_counts.get(grid_id, 0) + 1
        
        # Respawn for each grid
        for grid_id, count in grid_counts.items():
            self._place_items(grid_id, cell_type, count)
    
    def _get_observations(self) -> torch.Tensor:
        """Get batched observations (fully vectorized on GPU)."""
        r = self.view_radius
        vs = self.view_size
        
        # Mark agents on grid (save original values)
        agent_y_long = self.agent_y.long()
        agent_x_long = self.agent_x.long()
        current_vals = self.grids[self.grid_indices, agent_y_long, agent_x_long].clone()
        self.grids[self.grid_indices, agent_y_long, agent_x_long] = self.agent_types.to(torch.int8)
        
        # Pad grids
        if self.loop:
            padded_grids = F.pad(self.grids.float(), (r, r, r, r), mode='circular').to(torch.int8)
        else:
            padded_grids = F.pad(
                self.grids.float(), (r, r, r, r), 
                mode='constant', value=float(self.CellType.WALL.value)
            ).to(torch.int8)
        
        # VECTORIZED VIEW EXTRACTION using unfold
        # padded_grids shape: (n_grids, H+2r, W+2r)
        # We'll use unfold to create all possible view windows, then index by agent position
        
        # Create all possible windows: (n_grids, num_y_positions, num_x_positions, view_size, view_size)
        # Using unfold on last two dimensions
        windows = padded_grids.unfold(1, vs, 1).unfold(2, vs, 1)
        # windows shape: (n_grids, H, W, view_size, view_size)
        
        # Index by agent positions to get each agent's view
        # Shape: (n_envs, view_size, view_size)
        views = windows[self.grid_indices, agent_y_long, agent_x_long]
        
        # Convert to channels - vectorized
        for c, name in enumerate(self.channel_names):
            if name.startswith('cell_'):
                cell_name = name[5:].upper()
                try:
                    cell_type = getattr(self.CellType, cell_name)
                    self._view_buffer[:, c] = (views == cell_type.value).float()
                except AttributeError:
                    self._view_buffer[:, c] = 0.0
            else:
                self._view_buffer[:, c] = 0.0
        
        # Blank center
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
