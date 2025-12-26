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
        # Max items for fixed-size topk in _place_items (avoids sync)
        self._max_items_per_grid = max(default_food, default_poison, 1)
        
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
        # Use int32 for positions (TPU-friendly; PyTorch accepts int32 for indexing)
        self.agent_x = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self.agent_y = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self.agent_energy = torch.full((n_envs,), self.initial_energy, dtype=torch.float32, device=device)
        self.agent_steps = torch.zeros(n_envs, dtype=torch.int32, device=device)

        # Track what cell type is "under" each agent (for restoration when moving)
        # Agents are permanently marked on grid - this tracks the original cell value
        self.agent_underlying_cell = torch.zeros(n_envs, dtype=torch.float32, device=device)
        
        # Pre-allocated view buffer
        self._view_buffer = torch.zeros(
            (n_envs, self.n_channels, self.view_size, self.view_size),
            dtype=torch.float32, device=device
        )
        
        # Done states
        self.dones = torch.zeros(n_envs, dtype=torch.bool, device=device)

        # Pre-allocated temporaries for step() to avoid per-step allocations
        self._old_y = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self._old_x = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self._dones_return = torch.zeros(n_envs, dtype=torch.bool, device=device)
        self._step_rewards = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self._empty_value = torch.tensor(self.CellType.EMPTY.value, device=device, dtype=torch.float32)
        self._neg_inf = torch.tensor(-float('inf'), device=device, dtype=torch.float32)

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
        # Update max for fixed-size topk in _place_items
        self._max_items_per_grid = max(food_max, poison_max, 1)
    
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset environments and return observations."""
        if env_ids is None:
            # Reset all - this path is only at initialization
            for grid_id in range(self.n_grids):
                self._reset_grid(grid_id)
            for env_id in range(self.n_envs):
                self._reset_agent(env_id)
        else:
            # Reset specific envs - called during training when episodes end
            # Single sync to get Python-iterable env IDs
            env_id_list = env_ids.tolist()

            if self.shared_grid:
                # Shared grid mode: reset grid 0 once, then all agents
                self._reset_grid(0)
                for env_id in env_id_list:
                    self._reset_agent(env_id)
            else:
                # Independent mode: grid_id == env_id (no lookup needed)
                done_grids = set()
                for env_id in env_id_list:
                    # Each env has its own grid with matching index
                    if env_id not in done_grids:
                        self._reset_grid(env_id)
                        done_grids.add(env_id)
                    self._reset_agent(env_id)

        return self._get_observations()
    
    def _reset_grid(self, grid_id: int):
        """Clear and repopulate a specific grid."""
        self.grids[grid_id].fill_(self.CellType.EMPTY.value)

        # Randomize counts on GPU (stays on device, no sync)
        food_count = torch.randint(
            self.food_range[0], self.food_range[1] + 1, (1,),
            dtype=torch.int32, device=self.device
        ).squeeze()
        poison_count = torch.randint(
            self.poison_range[0], self.poison_range[1] + 1, (1,),
            dtype=torch.int32, device=self.device
        ).squeeze()

        self.grid_food_counts[grid_id] = food_count
        self.grid_poison_counts[grid_id] = poison_count

        # Place items (accepts tensor counts)
        self._place_items(grid_id, self.CellType.FOOD.value, food_count)
        self._place_items(grid_id, self.CellType.POISON.value, poison_count)
    
    def _reset_agent(self, env_id: int):
        """Reset a single agent's state and mark on grid.

        Uses tensor operations throughout to avoid CPU-GPU sync points.
        """
        grid_id = self.grid_indices[env_id]  # Keep as tensor, no .item()

        # Clear old position (restore underlying cell) - only if agent is actually there
        old_y = self.agent_y[env_id]
        old_x = self.agent_x[env_id]
        old_cell_value = self.grids[grid_id, old_y, old_x]
        agent_type = self.agent_types[env_id].float()

        # Conditional restore using torch.where (no Python branching on GPU values)
        is_agent_marked = (old_cell_value == agent_type)
        self.grids[grid_id, old_y, old_x] = torch.where(
            is_agent_marked,
            self.agent_underlying_cell[env_id],
            old_cell_value
        )

        # Noise-based random selection (avoids .item() on randint result)
        noise = torch.rand(self.height, self.width, device=self.device)
        empty_mask = (self.grids[grid_id] == self.CellType.EMPTY.value)
        masked_noise = torch.where(empty_mask, noise, self._neg_inf)

        # Argmax finds the random empty cell
        flat_idx = masked_noise.view(-1).argmax()
        new_y = (flat_idx // self.width).int()
        new_x = (flat_idx % self.width).int()

        # Update position and mark on grid
        self.agent_y[env_id] = new_y
        self.agent_x[env_id] = new_x
        self.agent_underlying_cell[env_id] = self.CellType.EMPTY.value
        self.grids[grid_id, new_y, new_x] = self.agent_types[env_id].float()

        self.agent_energy[env_id] = self.initial_energy
        self.agent_steps[env_id] = 0
        self.dones[env_id] = False

        # Update stats
        self.last_episode_food[env_id] = self.current_episode_food[env_id]
        self.last_episode_poison[env_id] = self.current_episode_poison[env_id]
        self.current_episode_food[env_id] = 0
        self.current_episode_poison[env_id] = 0
    
    def _reset_agents_batch(self, env_ids: torch.Tensor):
        """Batch reset for multiple agents with proper grid marking."""
        if len(env_ids) == 0:
            return

        # Update stats (vectorized)
        self.last_episode_food[env_ids] = self.current_episode_food[env_ids]
        self.last_episode_poison[env_ids] = self.current_episode_poison[env_ids]
        self.current_episode_food[env_ids] = 0
        self.current_episode_poison[env_ids] = 0

        # Restore old cells (clear agent markers) - only where agents are actually marked
        old_y = self.agent_y[env_ids]
        old_x = self.agent_x[env_ids]
        grid_ids = self.grid_indices[env_ids]
        old_cell_values = self.grids[grid_ids, old_y, old_x]
        agent_types_for_reset = self.agent_types[env_ids].float()
        # Only restore where the agent is actually marked
        actually_marked = (old_cell_values == agent_types_for_reset)
        restore_values = torch.where(
            actually_marked,
            self.agent_underlying_cell[env_ids],
            old_cell_values  # Keep existing if agent wasn't marked here
        )
        self.grids[grid_ids, old_y, old_x] = restore_values

        # Reset energy, steps, dones (vectorized)
        self.agent_energy[env_ids] = self.initial_energy
        self.agent_steps[env_ids] = 0
        self.dones[env_ids] = False

        if self.shared_grid:
            # Shared grid: sequential spawning for correctness (avoids collisions)
            # Use range to avoid .tolist() sync; _spawn_agent_on_empty handles tensor indexing
            for i in range(len(env_ids)):
                self._spawn_agent_on_empty(env_ids[i])
        else:
            # Independent grids: vectorized spawning (each agent has own grid)
            self._spawn_agents_vectorized(env_ids)

    def _spawn_agent_on_empty(self, env_id):
        """Spawn a single agent on an empty cell using noise-based selection.

        Args:
            env_id: Environment index (int or 0-d tensor)

        Uses argmax on masked noise to avoid CPU-GPU sync. Still called
        sequentially for shared_grid mode to prevent collisions.
        """
        grid_id = self.grid_indices[env_id]  # Keep as tensor, no .item()

        # Noise-based random selection (avoids .item() on randint result)
        noise = torch.rand(self.height, self.width, device=self.device)
        empty_mask = (self.grids[grid_id] == self.CellType.EMPTY.value)
        masked_noise = torch.where(empty_mask, noise, self._neg_inf)

        # Argmax finds the random empty cell
        flat_idx = masked_noise.view(-1).argmax()
        new_y = (flat_idx // self.width).int()
        new_x = (flat_idx % self.width).int()

        # Update position and mark on grid
        self.agent_y[env_id] = new_y
        self.agent_x[env_id] = new_x
        self.agent_underlying_cell[env_id] = self.CellType.EMPTY.value
        self.grids[grid_id, new_y, new_x] = self.agent_types[env_id].float()

    def _spawn_agents_vectorized(self, env_ids: torch.Tensor):
        """Fully vectorized spawn using noise-based random selection.

        Uses argmax on masked noise to select random empty cells without
        any CPU-GPU synchronization. Each agent has its own grid, so
        spawns are independent and can be fully parallelized.
        """
        n_reset = len(env_ids)
        if n_reset == 0:
            return

        grid_ids = self.grid_indices[env_ids]

        # Generate random noise for all cells across all grids
        noise = torch.rand(n_reset, self.height, self.width, device=self.device)

        # Get empty masks for all grids at once
        grids = self.grids[grid_ids]  # (n_reset, height, width)
        empty_mask = (grids == self.CellType.EMPTY.value)

        # Mask non-empty cells to -inf so they can't be chosen
        masked_noise = torch.where(empty_mask, noise, self._neg_inf)

        # Find argmax per grid (random empty cell due to uniform noise)
        flat_noise = masked_noise.view(n_reset, -1)
        flat_idx = flat_noise.argmax(dim=1)

        # Convert flat index to (y, x) coordinates
        new_y = (flat_idx // self.width).int()
        new_x = (flat_idx % self.width).int()

        # Update agent positions (vectorized)
        self.agent_y[env_ids] = new_y
        self.agent_x[env_ids] = new_x
        self.agent_underlying_cell[env_ids] = self.CellType.EMPTY.value

        # Mark agents on grids (vectorized advanced indexing)
        self.grids[grid_ids, new_y, new_x] = self.agent_types[env_ids].float()

    def _reset_agents_masked(self, done_mask: torch.Tensor):
        """
        Fully masked reset - NO GPU sync required.

        Uses torch.where everywhere instead of indexing, so we never need
        to know which agents are done (no .any(), .nonzero(), or len()).

        This runs operations on ALL agents but only applies changes where
        done_mask is True. Slightly more compute but zero sync overhead.

        Args:
            done_mask: Boolean tensor (n_envs,) indicating which agents are done
        """
        # Expand mask for broadcasting where needed
        done_f = done_mask.float()  # For numeric operations
        done_int = done_mask.int()

        # Update episode stats (only where done)
        self.last_episode_food = torch.where(
            done_mask, self.current_episode_food, self.last_episode_food
        )
        self.last_episode_poison = torch.where(
            done_mask, self.current_episode_poison, self.last_episode_poison
        )
        self.current_episode_food = torch.where(
            done_mask, torch.zeros_like(self.current_episode_food), self.current_episode_food
        )
        self.current_episode_poison = torch.where(
            done_mask, torch.zeros_like(self.current_episode_poison), self.current_episode_poison
        )

        # Restore old cells where done (clear agent markers)
        old_y = self.agent_y
        old_x = self.agent_x
        grid_ids = self.grid_indices
        old_cell_values = self.grids[grid_ids, old_y, old_x]
        agent_types_f = self.agent_types.float()

        # Only restore where agent is marked AND agent is done
        actually_marked = (old_cell_values == agent_types_f)
        should_restore = done_mask & actually_marked
        restore_values = torch.where(
            should_restore,
            self.agent_underlying_cell,
            old_cell_values
        )
        self.grids[grid_ids, old_y, old_x] = restore_values

        # Reset energy, steps, dones where done
        self.agent_energy = torch.where(
            done_mask, self.initial_energy, self.agent_energy
        )
        self.agent_steps = torch.where(
            done_mask,
            torch.zeros_like(self.agent_steps),
            self.agent_steps
        )
        self.dones = torch.where(
            done_mask,
            torch.zeros_like(self.dones),
            self.dones
        )

        # Spawn new positions for ALL agents (cheap), only apply where done
        # Generate random noise for all agents' grids
        noise = torch.rand(self.n_envs, self.height, self.width, device=self.device)

        # Get empty masks for all grids
        grids = self.grids[grid_ids]  # (n_envs, height, width)
        empty_mask = (grids == self.CellType.EMPTY.value)

        # Mask non-empty cells
        masked_noise = torch.where(empty_mask, noise, self._neg_inf)

        # Find random empty cell for each agent
        flat_noise = masked_noise.view(self.n_envs, -1)
        flat_idx = flat_noise.argmax(dim=1)
        new_y = (flat_idx // self.width).int()
        new_x = (flat_idx % self.width).int()

        # Only update positions where done
        self.agent_y = torch.where(done_mask, new_y, self.agent_y)
        self.agent_x = torch.where(done_mask, new_x, self.agent_x)
        self.agent_underlying_cell = torch.where(
            done_mask,
            torch.full_like(self.agent_underlying_cell, self.CellType.EMPTY.value),
            self.agent_underlying_cell
        )

        # Mark agents on grids where done
        # Need to use scatter or advanced indexing carefully
        # The new positions for done agents need to be marked
        new_positions_y = torch.where(done_mask, new_y, old_y)
        new_positions_x = torch.where(done_mask, new_x, old_x)

        # Get current values at new positions
        current_at_new = self.grids[grid_ids, new_positions_y, new_positions_x]
        # Only mark where done (set to agent type)
        new_values = torch.where(done_mask, agent_types_f, current_at_new)
        self.grids[grid_ids, new_positions_y, new_positions_x] = new_values
    
    def _place_items(self, grid_id: int, cell_type: int, count):
        """Place items at random empty positions using noise-based selection.

        Fully GPU-native: no .item(), .nonzero(), or .shape[] calls.
        Uses topk on masked random noise to select random empty cells.

        Args:
            grid_id: Grid index
            cell_type: Cell type value to place
            count: Number to place (int or scalar tensor)
        """
        # Determine max items we might place (for fixed-size topk)
        max_k = self._max_items_per_grid

        # Generate random noise for all cells
        noise = torch.rand(self.height, self.width, device=self.device)

        # Mask non-empty cells to -inf so they can't be selected
        empty_mask = (self.grids[grid_id] == self.CellType.EMPTY.value)
        masked_noise = torch.where(empty_mask, noise, self._neg_inf)

        # Get top max_k candidates (fixed k avoids sync)
        flat_noise = masked_noise.view(-1)
        _, top_indices = torch.topk(flat_noise, k=max_k, largest=True)

        # Convert to grid coordinates
        chosen_y = top_indices // self.width
        chosen_x = top_indices % self.width

        # Create position mask: only first `count` positions are valid
        # This works with tensor count without sync (element-wise comparison)
        position_idx = torch.arange(max_k, device=self.device, dtype=torch.int32)
        if isinstance(count, torch.Tensor):
            valid_mask = position_idx < count
        else:
            valid_mask = position_idx < count

        # Use scatter to place items: write cell_type where valid, EMPTY where not
        # Since we're targeting empty cells, writing EMPTY is a no-op
        flat_grid = self.grids[grid_id].view(-1)
        scatter_values = torch.where(
            valid_mask,
            torch.tensor(cell_type, device=self.device, dtype=flat_grid.dtype),
            torch.tensor(self.CellType.EMPTY.value, device=self.device, dtype=flat_grid.dtype)
        ).expand(max_k)
        flat_grid.scatter_(0, top_indices, scatter_values)
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute batched actions with proper grid updates."""
        # Ensure actions are on device
        if actions.device != self.device:
            actions = actions.to(self.device)

        # Store old positions (reuse pre-allocated buffers)
        self._old_y.copy_(self.agent_y)
        self._old_x.copy_(self.agent_x)

        # Restore old cells (clear agent markers before moving)
        self.grids[self.grid_indices, self._old_y, self._old_x] = self.agent_underlying_cell

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

        # Eating (updates underlying_cell and clears eaten items)
        rewards = self._eat_batch()

        # Mark agents at new positions
        self.grids[self.grid_indices, self.agent_y, self.agent_x] = self.agent_types.float()

        # Step count and done check
        self.agent_steps += 1
        self.dones = (self.agent_energy <= 0) | (self.agent_steps >= self.max_steps)

        # Death penalty (ONLY for starvation, not timeout)
        rewards = torch.where(self.agent_energy <= 0, rewards - 10.0, rewards)

        # Movement cost (living penalty) - crucial for sparse reward learning!
        rewards -= self.energy_move_cost

        # Save dones before reset (reuse pre-allocated buffer)
        self._dones_return.copy_(self.dones)

        # Auto-reset done agents
        # Use masked reset for independent grids (training) - no GPU sync
        # Use indexed reset for shared grid (visualization) - handles collisions
        if self.shared_grid:
            # Shared grid: need sequential spawning to avoid collisions
            # This path still syncs but is only used for visualization
            if self.dones.any():
                done_indices = self.dones.nonzero(as_tuple=True)[0]
                self._reset_agents_batch(done_indices)
        else:
            # Independent grids: fully masked reset - ZERO GPU sync
            self._reset_agents_masked(self.dones)

        return self._get_observations(), rewards, self._dones_return
    
    def _eat_batch(self) -> torch.Tensor:
        """Vectorized eating logic with underlying cell tracking."""
        # Reuse pre-allocated rewards buffer
        self._step_rewards.zero_()

        # Get cell values at agent positions (vectorized)
        cell_values = self.grids[self.grid_indices, self.agent_y, self.agent_x]

        # Food eating
        food_mask = (cell_values == self.CellType.FOOD.value)
        self._step_rewards = torch.where(food_mask, self._step_rewards + self.food_reward, self._step_rewards)
        self.agent_energy = torch.where(food_mask, self.agent_energy + self.food_reward, self.agent_energy)
        self.current_episode_food = torch.where(food_mask, self.current_episode_food + 1, self.current_episode_food)

        # Poison eating (check against original cell_values before any modifications)
        poison_mask = (cell_values == self.CellType.POISON.value)
        self._step_rewards = torch.where(poison_mask, self._step_rewards - self.poison_penalty, self._step_rewards)
        self.agent_energy = torch.where(poison_mask, self.agent_energy - self.poison_penalty, self.agent_energy)
        self.current_episode_poison = torch.where(poison_mask, self.current_episode_poison + 1, self.current_episode_poison)

        # Determine what the cell becomes after eating
        # If ate food or poison -> EMPTY, otherwise keep original
        ate_something = food_mask | poison_mask
        final_cell_values = torch.where(ate_something, self._empty_value, cell_values)

        # Update underlying cell (what's "under" the agent after this step)
        self.agent_underlying_cell = final_cell_values

        # Clear eaten items from grid
        self.grids[self.grid_indices, self.agent_y, self.agent_x] = final_cell_values

        # Respawn eaten items
        self._respawn_items_vectorized(food_mask, self.CellType.FOOD.value)
        self._respawn_items_vectorized(poison_mask, self.CellType.POISON.value)

        return self._step_rewards
    
    def _respawn_items_vectorized(self, eaten_mask: torch.Tensor, cell_type: int):
        """
        Fully vectorized respawn - shape-static-ish (masked) to avoid ghosts.
        Uses nonzero() for writing but avoids CPU syncs (no shape checks).
        """
        # grid_ids: We use self.grid_indices directly (static shape N_envs)
        N = self.n_envs
        K = 50  # Increased for 99.5%+ success rate even on dense grids
        
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

        Note: Agents are permanently marked on the grid, so no temp mark/restore needed.
        """
        r = self.view_radius
        vs = self.view_size

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
        views = windows[self.grid_indices, self.agent_y, self.agent_x]  # (n_envs, vs, vs)

        if self.is_proxy_mode:
            # PROXY MODE: Hide ground truth, show interestingness
            # Channel 0: is_empty (binary)
            # Channel 1: is_wall (binary)
            # Channels 2+: interestingness value (same across all these channels)

            views_int = views.int()  # int32 for TPU-friendly indexing

            # Binary channels for empty/wall
            self._view_buffer[:, 0, :, :] = (views_int == self.CellType.EMPTY.value).float()
            self._view_buffer[:, 1, :, :] = (views_int == self.CellType.WALL.value).float()

            # Interestingness channel (lookup from cell values)
            # Clamp to valid indices to avoid index errors
            clamped_views = views_int.clamp(0, len(self._interestingness_lut) - 1)
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
