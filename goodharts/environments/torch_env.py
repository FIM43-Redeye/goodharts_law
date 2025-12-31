"""
GPU-Native Vectorized Environment using PyTorch tensors.

This is a drop-in replacement for VecEnv that keeps all state on GPU,
eliminating CPU-GPU transfer overhead during training.

All operations are vectorized over the batch dimension using PyTorch ops.

NOTE: This class inherits from nn.Module so that mutable state tensors can be
registered as buffers. This is required for CUDA graph compatibility - PyTorch's
CUDA graph system allows in-place mutations of nn.Module buffers but not of
regular instance attributes ("eager inputs").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from goodharts.modes import ObservationSpec
from goodharts.configs.default_config import CellType, get_simulation_config
from goodharts.config import get_training_config
from goodharts.utils.device import get_device


# Build action deltas as a torch tensor
def _build_action_deltas(device: torch.device, max_move_distance: int = 1) -> torch.Tensor:
    """Build action delta tensor matching build_action_space order."""
    from goodharts.behaviors.action_space import build_action_space
    actions = build_action_space(max_move_distance)
    deltas = []
    for action in actions:
        if isinstance(action, (tuple, list)) and len(action) >= 2:
            deltas.append([action[0], action[1]])
        elif hasattr(action, 'dx') and hasattr(action, 'dy'):
            deltas.append([action.dx, action.dy])
        else:
            deltas.append([0, 0])  # No-op or unknown
    return torch.tensor(deltas, dtype=torch.int32, device=device)


class TorchVecEnv(nn.Module):
    """
    GPU-native vectorized environment.

    All state is stored in PyTorch tensors on the specified device.
    Observations are returned as GPU tensors (no CPU transfer).

    API matches VecEnv for drop-in replacement.

    Inherits from nn.Module to enable CUDA graph compatibility.
    All mutable tensor state is registered as buffers (persistent=False),
    which allows in-place mutations without triggering "mutated inputs"
    warnings that would disable CUDA graph capture.
    """

    def __init__(
        self,
        n_envs: int,
        obs_spec: ObservationSpec,
        config: dict = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize GPU-native vectorized environment.

        Args:
            n_envs: Number of parallel environments
            obs_spec: Observation specification
            config: Optional config override
            device: Torch device (auto-detect if None)
        """
        super().__init__()

        self.n_envs = n_envs
        self.obs_spec = obs_spec

        # Device selection - use centralized get_device()
        if device is None:
            device = get_device()
        self.device = device

        # Get config - all values required from TOML
        if config is None:
            config = get_simulation_config()
        self.config = config
        train_cfg = get_training_config()

        # Dimensions from config (required)
        self.width = config['GRID_WIDTH']
        self.height = config['GRID_HEIGHT']
        
        # View settings from obs_spec
        self.view_radius = obs_spec.view_size // 2
        self.view_size = obs_spec.view_size
        self.n_channels = obs_spec.num_channels
        self.channel_names = obs_spec.channel_names
        
        # Mode-aware observation encoding
        # Detect proxy mode: uses 'interestingness' instead of cell-type one-hot
        self.is_proxy_mode = any('interestingness' in ch for ch in self.channel_names)

        # Freeze energy during training: agents don't die, enabling exploration
        # Used for proxy mode where agents can't learn from energy consequences
        # Use private attribute + property so setting freeze_energy updates multiplier
        self._freeze_energy = obs_spec.freeze_energy_in_training
        # Branchless multiplier: 0.0 when frozen (skip energy updates), 1.0 otherwise
        self._energy_enabled = 0.0 if self._freeze_energy else 1.0

        # Build interestingness lookup table: cell_value -> interestingness
        # Used for proxy mode observations
        cell_types = CellType.all_types()
        n_cell_types = len(cell_types)
        interestingness_lut = torch.zeros(n_cell_types, dtype=torch.float32, device=device)
        for ct in cell_types:
            interestingness_lut[ct.value] = ct.interestingness
        self.register_buffer('_interestingness_lut', interestingness_lut, persistent=False)
        
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
        
        # CellType enum - cache values as plain ints/floats for torch.compile compatibility
        # (Descriptors cause graph breaks, plain values don't)
        self.CellType = config['CellType']
        self._cell_empty = self.CellType.EMPTY.value
        self._cell_food = self.CellType.FOOD.value
        self._cell_poison = self.CellType.POISON.value
        self.food_reward = self.CellType.FOOD.energy_reward
        self.poison_penalty = self.CellType.POISON.energy_penalty
        # Note: interestingness values are handled via _interestingness_lut (line 116)

        # Action deltas - must match policy's action space (constant)
        self.max_move_distance = config.get('MAX_MOVE_DISTANCE', 1)
        self.register_buffer('action_deltas', _build_action_deltas(device, self.max_move_distance), persistent=False)

        # Grid setup - each env has its own independent grid
        self.n_grids = n_envs
        self.register_buffer('grid_indices', torch.arange(n_envs, device=device), persistent=False)

        # Per-grid counts (mutable)
        self.register_buffer('grid_food_counts', torch.full((self.n_grids,), self._default_food, dtype=torch.int32, device=device), persistent=False)
        self.register_buffer('grid_poison_counts', torch.full((self.n_grids,), self._default_poison, dtype=torch.int32, device=device), persistent=False)

        # State tensors (all on GPU, all mutable)
        # Grid uses float32 for faster F.pad (avoids dtype conversion overhead)
        self.register_buffer('grids', torch.zeros((self.n_grids, self.height, self.width), dtype=torch.float32, device=device), persistent=False)
        # Use int32 for positions (TPU-friendly; PyTorch accepts int32 for indexing)
        self.register_buffer('agent_x', torch.zeros(n_envs, dtype=torch.int32, device=device), persistent=False)
        self.register_buffer('agent_y', torch.zeros(n_envs, dtype=torch.int32, device=device), persistent=False)
        self.register_buffer('agent_energy', torch.full((n_envs,), self.initial_energy, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer('agent_steps', torch.zeros(n_envs, dtype=torch.int32, device=device), persistent=False)

        # Pre-allocated view buffer (mutable - written each step)
        self.register_buffer('_view_buffer', torch.zeros(
            (n_envs, self.n_channels, self.view_size, self.view_size),
            dtype=torch.float32, device=device
        ), persistent=False)

        # Done states - separate terminated (real death) from truncated (time limit)
        self.register_buffer('terminated', torch.zeros(n_envs, dtype=torch.bool, device=device), persistent=False)
        self.register_buffer('truncated', torch.zeros(n_envs, dtype=torch.bool, device=device), persistent=False)
        self.register_buffer('dones', torch.zeros(n_envs, dtype=torch.bool, device=device), persistent=False)

        # Pre-allocated temporaries for step() to avoid per-step allocations
        self.register_buffer('_dones_return', torch.zeros(n_envs, dtype=torch.bool, device=device), persistent=False)
        self.register_buffer('_terminated_return', torch.zeros(n_envs, dtype=torch.bool, device=device), persistent=False)
        self.register_buffer('_truncated_return', torch.zeros(n_envs, dtype=torch.bool, device=device), persistent=False)
        self.register_buffer('_step_rewards', torch.zeros(n_envs, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer('_empty_value', torch.tensor(self._cell_empty, device=device, dtype=torch.float32), persistent=False)
        self.register_buffer('_neg_inf', torch.tensor(-float('inf'), device=device, dtype=torch.float32), persistent=False)

        # Pre-allocated constants for branchless torch.where in _reset_agents_masked
        # These avoid per-reset tensor allocations (constant, but buffer for address stability)
        self.register_buffer('_zero_int32', torch.zeros(n_envs, dtype=torch.int32, device=device), persistent=False)
        self.register_buffer('_zero_bool', torch.zeros(n_envs, dtype=torch.bool, device=device), persistent=False)
        self.register_buffer('_initial_energy_tensor', torch.full((n_envs,), self.initial_energy, dtype=torch.float32, device=device), persistent=False)

        # Stats tracking (mutable)
        self.register_buffer('current_episode_food', torch.zeros(n_envs, dtype=torch.int32, device=device), persistent=False)
        self.register_buffer('last_episode_food', torch.zeros(n_envs, dtype=torch.int32, device=device), persistent=False)
        self.register_buffer('current_episode_poison', torch.zeros(n_envs, dtype=torch.int32, device=device), persistent=False)
        self.register_buffer('last_episode_poison', torch.zeros(n_envs, dtype=torch.int32, device=device), persistent=False)

        # Episode rewards accumulator (for CUDA graph-compatible tracking)
        # Lives here because TorchVecEnv inherits from nn.Module, enabling stable buffer addresses
        self.register_buffer('episode_rewards', torch.zeros(n_envs, dtype=torch.float32, device=device), persistent=False)

        # Pre-allocated density buffer and cached normalization constants
        # Eliminates ~384 tensor allocations per update in privileged critic mode
        self.register_buffer('_density_buffer', torch.zeros((n_envs, 2), dtype=torch.float32, device=device), persistent=False)
        self._density_food_mid = (default_food + default_food) / 2.0
        self._density_food_scale = 1.0  # Updated in set_curriculum_ranges
        self._density_poison_mid = (default_poison + default_poison) / 2.0
        self._density_poison_scale = 1.0

        # Privileged view buffer: ground truth encoding for value function
        # Only used when is_proxy_mode=True (ground_truth_blinded, proxy)
        # For ground_truth modes, privileged view == regular view (no extra buffer needed)
        if self.is_proxy_mode:
            self.register_buffer('_privileged_view_buffer', torch.zeros(
                (n_envs, self.n_channels, self.view_size, self.view_size),
                dtype=torch.float32, device=device
            ), persistent=False)
        else:
            self._privileged_view_buffer = None

        # Initialize
        self.reset()

    @property
    def freeze_energy(self) -> bool:
        """Whether energy updates are frozen (agents can't die)."""
        return self._freeze_energy

    @freeze_energy.setter
    def freeze_energy(self, value: bool):
        """Set freeze_energy and update the branchless multiplier."""
        self._freeze_energy = value
        self._energy_enabled = 0.0 if value else 1.0

    def set_curriculum_ranges(self, food_min: int, food_max: int,
                               poison_min: int, poison_max: int):
        """Set curriculum ranges for per-environment randomization."""
        self.food_range = (food_min, food_max)
        self.poison_range = (poison_min, poison_max)
        # Update max for fixed-size topk in _place_items
        self._max_items_per_grid = max(food_max, poison_max, 1)
        # Cache normalization constants for get_density_info() (avoids per-call arithmetic)
        self._density_food_mid = (food_min + food_max) / 2.0
        self._density_food_scale = (food_max - food_min) / 2.0 if food_max > food_min else 1.0
        self._density_poison_mid = (poison_min + poison_max) / 2.0
        self._density_poison_scale = (poison_max - poison_min) / 2.0 if poison_max > poison_min else 1.0

    def _grid_scatter(self, grid_ids: torch.Tensor, y: torch.Tensor,
                      x: torch.Tensor, values: torch.Tensor):
        """
        Scatter values into grids at specified positions.

        Uses scatter_ instead of indexed assignment to avoid CUDA graph issues.
        Indexed assignment like `grids[ids, y, x] = vals` causes "mutated inputs"
        warnings because the indices change each step. scatter_ is explicitly
        designed for this and works with CUDA graphs.

        Args:
            grid_ids: (n,) grid indices
            y: (n,) y coordinates
            x: (n,) x coordinates
            values: (n,) values to write (must be float for grid dtype)
        """
        # Convert 3D indices to flat 1D indices
        flat_idx = grid_ids * (self.height * self.width) + y * self.width + x
        self.grids.view(-1).scatter_(0, flat_idx.long(), values)

    def compile_step(self, mode: str = 'max-autotune', fullgraph: bool = True):
        """
        Compile the step method using torch.compile for maximum performance.

        This fuses the environment step + observation extraction into optimized kernels,
        reducing CPU dispatch overhead from ~8,300 ops/100ms to a single fused kernel.

        Args:
            mode: torch.compile mode. Options:
                - 'max-autotune': Full optimization with CUDA graphs (default)
                - 'reduce-overhead': Faster compile, enables CUDA graphs
                - 'max-autotune-no-cudagraphs': For debugging CUDA graph issues
            fullgraph: If True, requires entire step to compile as one graph.
                       Falls back to fullgraph=False if compilation fails.

        Note: Call this AFTER set_curriculum_ranges but BEFORE training starts.
        """
        import logging

        try:
            self._step_uncompiled = self.step
            compiled = torch.compile(self._step_uncompiled, mode=mode, fullgraph=fullgraph)
            self.step = compiled
            logging.info(f"[TorchVecEnv] Compiled step with mode={mode}, fullgraph={fullgraph}")
        except Exception as e:
            if fullgraph:
                logging.warning(f"[TorchVecEnv] fullgraph=True failed: {e}, retrying with fullgraph=False")
                try:
                    compiled = torch.compile(self._step_uncompiled, mode=mode, fullgraph=False)
                    self.step = compiled
                    logging.info(f"[TorchVecEnv] Compiled step with mode={mode}, fullgraph=False")
                except Exception as e2:
                    logging.warning(f"[TorchVecEnv] Compilation failed entirely: {e2}")
            else:
                logging.warning(f"[TorchVecEnv] Compilation failed: {e}")

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset environments and return observations.

        When env_ids is None, resets all environments using vectorized ops (ZERO sync).
        When env_ids is provided, resets only specified envs (used for partial resets).
        """
        if env_ids is None:
            # Reset all - use fully vectorized path (ZERO GPU sync)
            self._reset_all_grids_vectorized()
            self._reset_all_agents_vectorized()
        else:
            # Reset specific envs - called during training when episodes end
            # Single sync to get Python-iterable env IDs
            env_id_list = env_ids.tolist()

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
        self.grids[grid_id].fill_(self._cell_empty)

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
        self._place_items(grid_id, self._cell_food, food_count)
        self._place_items(grid_id, self._cell_poison, poison_count)

    def _reset_all_grids_vectorized(self):
        """Reset all grids in a single batched operation - ZERO GPU sync.

        Clears all grids, generates random item counts, and places items
        on all grids simultaneously using vectorized operations.
        """
        N = self.n_grids

        # Clear all grids at once
        self.grids.fill_(self._cell_empty)

        # Generate random counts for all grids at once
        # Use copy_() to preserve buffer memory address for CUDA graph compatibility
        self.grid_food_counts.copy_(torch.randint(
            self.food_range[0], self.food_range[1] + 1,
            (N,), dtype=torch.int32, device=self.device
        ))
        self.grid_poison_counts.copy_(torch.randint(
            self.poison_range[0], self.poison_range[1] + 1,
            (N,), dtype=torch.int32, device=self.device
        ))

        # Place food on all grids
        self._place_items_all_grids(self._cell_food, self.grid_food_counts)

        # Place poison on all grids
        self._place_items_all_grids(self._cell_poison, self.grid_poison_counts)

    def _place_items_all_grids(self, cell_type: int, counts: torch.Tensor):
        """Place items on all grids simultaneously - ZERO GPU sync.

        Uses batched topk on masked random noise to select random empty cells
        across all grids in a single operation.

        Args:
            cell_type: Cell type value to place
            counts: (N,) tensor of item counts per grid
        """
        N = self.n_grids
        H, W = self.height, self.width
        max_k = self._max_items_per_grid

        # Generate random noise for all grids: (N, H, W)
        noise = torch.rand(N, H, W, device=self.device)

        # Mask non-empty cells to -inf so they can't be selected
        empty_mask = (self.grids == self._cell_empty)
        masked_noise = torch.where(empty_mask, noise, self._neg_inf)

        # Get top max_k candidates for each grid: (N, max_k)
        flat_noise = masked_noise.view(N, -1)
        _, top_indices = torch.topk(flat_noise, k=max_k, largest=True, dim=1)

        # Convert to grid coordinates
        chosen_y = top_indices // W  # (N, max_k)
        chosen_x = top_indices % W   # (N, max_k)

        # Create position mask: only first count[i] positions valid for grid i
        position_idx = torch.arange(max_k, device=self.device).unsqueeze(0)  # (1, max_k)
        valid_mask = position_idx < counts.unsqueeze(1)  # (N, max_k)

        # Prepare grid indices for advanced indexing
        grid_ids = torch.arange(N, device=self.device).unsqueeze(1).expand(-1, max_k)

        # Read current values, compute new, write back (masked write pattern)
        current_vals = self.grids[grid_ids, chosen_y, chosen_x]
        new_vals = torch.where(valid_mask, float(cell_type), current_vals)
        self.grids[grid_ids, chosen_y, chosen_x] = new_vals

    def _reset_all_agents_vectorized(self):
        """Reset all agents using the masked reset path - ZERO GPU sync.

        Calls _reset_agents_masked with an all-True mask to reset every agent.
        """
        all_done = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        self._reset_agents_masked(all_done)

    def _random_empty_positions(
        self,
        grids: torch.Tensor,
        k: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select k random empty positions from each grid using noise-based selection.

        This is the core GPU-native pattern for random position selection without
        CPU-GPU synchronization. Uses argmax/topk on masked random noise.

        Args:
            grids: Grid tensor, shape (H, W) for single grid or (N, H, W) for batch
            k: Number of positions to select per grid

        Returns:
            (y, x): Position tensors
                - Single grid, k=1: scalar tensors
                - Single grid, k>1: shape (k,)
                - Batch, k=1: shape (N,)
                - Batch, k>1: shape (N, k)
        """
        single_grid = grids.dim() == 2
        if single_grid:
            grids = grids.unsqueeze(0)

        N, H, W = grids.shape

        # Generate random noise for all cells
        noise = torch.rand(N, H, W, device=self.device)

        # Mask non-empty cells to -inf so they can't be chosen
        empty_mask = (grids == self._cell_empty)
        masked_noise = torch.where(empty_mask, noise, self._neg_inf)

        # Flatten spatial dimensions for selection
        flat_noise = masked_noise.view(N, -1)

        if k == 1:
            # Single position per grid: use argmax
            flat_idx = flat_noise.argmax(dim=1)
            y = (flat_idx // W).int()
            x = (flat_idx % W).int()
        else:
            # Multiple positions per grid: use topk
            _, top_indices = torch.topk(flat_noise, k=k, largest=True, dim=1)
            y = (top_indices // W).int()
            x = (top_indices % W).int()

        # Squeeze back if single grid
        if single_grid:
            y = y.squeeze(0)
            x = x.squeeze(0)

        return y, x

    def _reset_agent(self, env_id: int):
        """Reset a single agent's state.

        Agents are tracked by coordinates only (not marked on grid).
        Uses tensor operations throughout to avoid CPU-GPU sync points.
        """
        grid_id = self.grid_indices[env_id]  # Keep as tensor, no .item()

        # Select random empty position
        new_y, new_x = self._random_empty_positions(self.grids[grid_id])

        # Update position
        self.agent_y[env_id] = new_y
        self.agent_x[env_id] = new_x

        self.agent_energy[env_id] = self.initial_energy
        self.agent_steps[env_id] = 0
        self.dones[env_id] = False

        # Update stats
        self.last_episode_food[env_id] = self.current_episode_food[env_id]
        self.last_episode_poison[env_id] = self.current_episode_poison[env_id]
        self.current_episode_food[env_id] = 0
        self.current_episode_poison[env_id] = 0
    
    def _spawn_agents_vectorized(self, env_ids: torch.Tensor):
        """Fully vectorized spawn using noise-based random selection.

        Uses _random_empty_positions to select random empty cells without
        any CPU-GPU synchronization. Each agent has its own grid, so
        spawns are independent and can be fully parallelized.

        Agents are tracked by coordinates only (not marked on grid).
        """
        n_reset = len(env_ids)
        if n_reset == 0:
            return

        grid_ids = self.grid_indices[env_ids]

        # Select random empty positions for all agents
        new_y, new_x = self._random_empty_positions(self.grids[grid_ids])

        # Update agent positions (vectorized)
        self.agent_y[env_ids] = new_y
        self.agent_x[env_ids] = new_x

    def _reset_agents_masked(self, done_mask: torch.Tensor):
        """
        Fully masked reset - NO GPU sync required.

        Uses torch.where everywhere instead of indexing, so we never need
        to know which agents are done (no .any(), .nonzero(), or len()).

        This runs operations on ALL agents but only applies changes where
        done_mask is True. Slightly more compute but zero sync overhead.

        Agents are tracked by coordinates only (not marked on grid).

        Args:
            done_mask: Boolean tensor (n_envs,) indicating which agents are done
        """
        # Update episode stats (only where done)
        # Use copy_() to preserve buffer memory address for CUDA graph compatibility
        self.last_episode_food.copy_(torch.where(
            done_mask, self.current_episode_food, self.last_episode_food
        ))
        self.last_episode_poison.copy_(torch.where(
            done_mask, self.current_episode_poison, self.last_episode_poison
        ))
        self.current_episode_food.copy_(torch.where(
            done_mask, self._zero_int32, self.current_episode_food
        ))
        self.current_episode_poison.copy_(torch.where(
            done_mask, self._zero_int32, self.current_episode_poison
        ))

        # Reset energy, steps, dones where done (using pre-allocated constants)
        # Use copy_() to preserve buffer memory address for CUDA graph compatibility
        self.agent_energy.copy_(torch.where(
            done_mask, self._initial_energy_tensor, self.agent_energy
        ))
        self.agent_steps.copy_(torch.where(
            done_mask, self._zero_int32, self.agent_steps
        ))
        self.dones.copy_(torch.where(
            done_mask, self._zero_bool, self.dones
        ))

        # Spawn new positions for ALL agents (cheap), only apply where done
        grid_ids = self.grid_indices
        new_y, new_x = self._random_empty_positions(self.grids[grid_ids])

        # Only update positions where done
        # Use copy_() to preserve buffer memory address for CUDA graph compatibility
        self.agent_y.copy_(torch.where(done_mask, new_y, self.agent_y))
        self.agent_x.copy_(torch.where(done_mask, new_x, self.agent_x))
    
    def _place_items(self, grid_id: int, cell_type: int, count):
        """Place items at random empty positions using noise-based selection.

        Fully GPU-native: no .item(), .nonzero(), or .shape[] calls.
        Uses _random_empty_positions with topk to select random empty cells.

        Args:
            grid_id: Grid index
            cell_type: Cell type value to place
            count: Number to place (int or scalar tensor)
        """
        # Determine max items we might place (for fixed-size topk)
        max_k = self._max_items_per_grid

        # Select random empty positions
        chosen_y, chosen_x = self._random_empty_positions(self.grids[grid_id], k=max_k)

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
        flat_indices = chosen_y * self.width + chosen_x
        scatter_values = torch.where(
            valid_mask,
            torch.tensor(cell_type, device=self.device, dtype=flat_grid.dtype),
            torch.tensor(self._cell_empty, device=self.device, dtype=flat_grid.dtype)
        ).expand(max_k)
        flat_grid.scatter_(0, flat_indices, scatter_values)
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Execute batched actions with proper grid updates.

        Returns:
            observations: (n_envs, channels, height, width) observation tensor
            eating_info: (food_mask, poison_mask, starved_mask) - what happened this step
            terminated: (n_envs,) boolean - True when agent died (starvation)
            truncated: (n_envs,) boolean - True when episode hit max_steps (time limit)
        """
        # Ensure actions are on device
        if actions.device != self.device:
            actions = actions.to(self.device)

        # Get movement deltas
        dx = self.action_deltas[actions, 0]
        dy = self.action_deltas[actions, 1]

        # Move agents (toroidal wrapping)
        # Use copy_() to preserve buffer memory address for CUDA graph compatibility
        self.agent_x.copy_((self.agent_x + dx) % self.width)
        self.agent_y.copy_((self.agent_y + dy) % self.height)

        # Energy cost (branchless: multiplier is 0.0 when frozen)
        # -= is in-place so address is preserved
        self.agent_energy -= self.energy_move_cost * self._energy_enabled

        # Eating (updates energy unless frozen, clears consumed items from grid)
        food_mask, poison_mask = self._eat_batch()

        # Step count and done check
        # Separate terminated (real death) from truncated (time limit)
        # Use in-place ops on pre-allocated buffers to avoid tensor allocation
        self.agent_steps += 1

        # Truncated: hit time limit (in-place comparison)
        torch.ge(self.agent_steps, self.max_steps, out=self.truncated)

        # Terminated: agent died (branchless - when freeze_energy, energy never drops so always False)
        torch.le(self.agent_energy, 0, out=self.terminated)

        # Dones: combined for reset logic (in-place OR)
        torch.bitwise_or(self.terminated, self.truncated, out=self.dones)

        # Save signals before reset (reuse pre-allocated buffers)
        self._dones_return.copy_(self.dones)
        self._terminated_return.copy_(self.terminated)
        self._truncated_return.copy_(self.truncated)

        # Auto-reset done agents - fully masked reset (ZERO GPU sync)
        self._reset_agents_masked(self.dones)

        eating_info = (food_mask, poison_mask, self.terminated)
        return self._get_observations(), eating_info, self._terminated_return, self._truncated_return
    
    def _eat_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized eating logic.

        Returns:
            (food_mask, poison_mask): Boolean tensors indicating what each agent ate.
            The reward computer uses these to compute rewards based on mode.
        """
        # Get cell values at agent positions (vectorized)
        cell_values = self.grids[self.grid_indices, self.agent_y, self.agent_x]

        # Detect what was eaten
        food_mask = (cell_values == self._cell_food)
        poison_mask = (cell_values == self._cell_poison)

        # Update agent energy (branchless: multiplier is 0.0 when frozen, so no change)
        # Use copy_() to preserve buffer memory address for CUDA graph compatibility
        self.agent_energy.copy_(torch.where(
            food_mask, self.agent_energy + self.food_reward * self._energy_enabled, self.agent_energy
        ))
        self.agent_energy.copy_(torch.where(
            poison_mask, self.agent_energy - self.poison_penalty * self._energy_enabled, self.agent_energy
        ))

        # Update episode counters
        self.current_episode_food.copy_(torch.where(food_mask, self.current_episode_food + 1, self.current_episode_food))
        self.current_episode_poison.copy_(torch.where(poison_mask, self.current_episode_poison + 1, self.current_episode_poison))

        # Clear eaten items from grid (set to EMPTY)
        ate_something = food_mask | poison_mask
        current_grid_vals = self.grids[self.grid_indices, self.agent_y, self.agent_x]
        new_grid_vals = torch.where(ate_something, self._empty_value, current_grid_vals)
        self._grid_scatter(self.grid_indices, self.agent_y, self.agent_x, new_grid_vals)

        # Respawn eaten items
        self._respawn_items_vectorized(food_mask, self._cell_food)
        self._respawn_items_vectorized(poison_mask, self._cell_poison)

        return food_mask, poison_mask
    
    def _respawn_items_vectorized(self, eaten_mask: torch.Tensor, cell_type: int):
        """
        Fully masked respawn - ZERO GPU sync.

        Uses torch.where everywhere instead of nonzero indexing.
        Runs operations on ALL envs but only applies changes where mask is True.
        No .any(), .nonzero(), or data-dependent shapes.
        """
        N = self.n_envs
        K = 50  # Candidates per env for 99.5%+ success rate

        # Generate K random candidates for ALL envs: (N, K)
        rand_y = torch.randint(0, self.height, (N, K), device=self.device)
        rand_x = torch.randint(0, self.width, (N, K), device=self.device)

        # Check which candidates are empty AND not on agent positions
        grid_read_ids = self.grid_indices.unsqueeze(1)  # (N, 1)
        vals = self.grids[grid_read_ids, rand_y, rand_x]  # (N, K)
        is_empty = (vals == self._cell_empty)  # (N, K) bool

        # Exclude agent positions (agents aren't marked on grid anymore)
        is_agent_pos = (rand_y == self.agent_y.unsqueeze(1)) & (rand_x == self.agent_x.unsqueeze(1))
        is_empty = is_empty & ~is_agent_pos

        # Select first valid candidate (argmax returns index of first True)
        first_valid_idx = is_empty.int().argmax(dim=1)  # (N,)

        # Extract chosen coords
        range_n = torch.arange(N, device=self.device)
        chosen_y = rand_y[range_n, first_valid_idx]  # (N,)
        chosen_x = rand_x[range_n, first_valid_idx]  # (N,)

        # Check if the chosen spot is actually valid (avoids .any() which can sync)
        chosen_is_valid = is_empty[range_n, first_valid_idx]  # (N,) bool

        # Final write mask: item was eaten AND we found a valid respawn spot
        do_write = eaten_mask & chosen_is_valid  # (N,) bool

        # MASKED WRITE: read current values, compute new, write back
        # This writes to ALL positions but only changes where do_write=True
        current_vals = self.grids[self.grid_indices, chosen_y, chosen_x]
        new_vals = torch.where(do_write, float(cell_type), current_vals)
        self._grid_scatter(self.grid_indices, chosen_y, chosen_x, new_vals)
    
    def _get_observations(self) -> torch.Tensor:
        """Get batched observations (fully vectorized on GPU).

        2-channel encoding for both modes (same CNN architecture):
        - Ground truth: Food=[1,0], Poison=[0,1], Empty=[0,0]
        - Proxy: Food=[i,i], Poison=[i,i], Empty=[0,0]
                 where i = cell's interestingness value

        The proxy agent sees the same value in both channels, making
        food and poison indistinguishable - only magnitude matters.
        Poison has higher interestingness (1.0 > 0.5), creating the Goodhart trap.

        For proxy modes, also computes ground truth encoding into _privileged_view_buffer
        for the value function (asymmetric actor-critic). This reuses the same
        views_int tensor, so only the encoding step is duplicated (trivial cost).
        """
        r = self.view_radius
        vs = self.view_size

        # Pad grids with circular wrapping (grid is float32, no conversion needed)
        padded_grids = F.pad(self.grids, (r, r, r, r), mode='circular')

        # VECTORIZED VIEW EXTRACTION using unfold (expensive - done once)
        windows = padded_grids.unfold(1, vs, 1).unfold(2, vs, 1)

        # Index by agent positions to get each agent's view
        views = windows[self.grid_indices, self.agent_y, self.agent_x]  # (n_envs, vs, vs)
        views_int = views.int()

        if self.is_proxy_mode:
            # PROXY MODE: Both channels get the same interestingness value
            # Agent cannot distinguish food from poison, only sees magnitude
            clamped_views = views_int.clamp(0, len(self._interestingness_lut) - 1)
            interestingness = self._interestingness_lut[clamped_views]  # (n_envs, vs, vs)

            # Both channels get the same value
            self._view_buffer[:, 0, :, :] = interestingness
            self._view_buffer[:, 1, :, :] = interestingness

            # PRIVILEGED VIEW: Ground truth encoding for value function
            # Computed from same views_int - only cost is 2 comparisons (trivial)
            self._privileged_view_buffer[:, 0, :, :] = (views_int == self._cell_food).float()
            self._privileged_view_buffer[:, 1, :, :] = (views_int == self._cell_poison).float()
            self._privileged_view_buffer[:, :, r, r] = 0.0
        else:
            # GROUND TRUTH MODE: Binary encoding
            # Channel 0: is_food, Channel 1: is_poison
            self._view_buffer[:, 0, :, :] = (views_int == self._cell_food).float()
            self._view_buffer[:, 1, :, :] = (views_int == self._cell_poison).float()
            # Privileged view == regular view (no separate buffer needed)

        # Blank center (agent's own cell)
        self._view_buffer[:, :, r, r] = 0.0

        return self._view_buffer

    def get_density_info(self) -> torch.Tensor:
        """
        Get normalized density info for privileged critic.

        Returns density as (n_envs, 2) tensor with [food_density, poison_density].
        Values are normalized to roughly [-1, 1] based on curriculum ranges.

        This info helps the value function predict returns more accurately
        without affecting the policy (which only sees the grid).

        Performance: Uses pre-allocated buffer and cached normalization constants
        to eliminate ~384 tensor allocations per update.
        """
        # Compute normalized density values (no in-place ops for CUDA graph compatibility)
        # Column 0: normalized food density
        # Column 1: normalized poison density
        food_counts = self.grid_food_counts[self.grid_indices].float()
        poison_counts = self.grid_poison_counts[self.grid_indices].float()

        food_normalized = (food_counts - self._density_food_mid) / self._density_food_scale
        poison_normalized = (poison_counts - self._density_poison_mid) / self._density_poison_scale

        # Stack into buffer (returns new tensor, doesn't mutate)
        return torch.stack([food_normalized, poison_normalized], dim=1)

    def get_privileged_view(self) -> torch.Tensor:
        """
        Get ground truth observation for privileged critic (asymmetric actor-critic).

        Returns the GROUND TRUTH one-hot encoding for value function use:
        - Channel 0: is_food (1.0 where cell is food, 0.0 elsewhere)
        - Channel 1: is_poison (1.0 where cell is poison, 0.0 elsewhere)

        Returns:
            (n_envs, 2, view_size, view_size) tensor with ground truth encoding

        For proxy modes (ground_truth_blinded, proxy), returns the pre-computed
        _privileged_view_buffer that was populated during _get_observations().
        For ground truth modes, returns the regular view buffer (same content).

        IMPORTANT: Must be called AFTER _get_observations() in the same step,
        as it returns the cached buffer computed there.
        """
        if self._privileged_view_buffer is not None:
            # Proxy mode: return pre-computed ground truth view
            return self._privileged_view_buffer
        else:
            # Ground truth mode: privileged view == regular view
            return self._view_buffer

    def get_critic_aux(self) -> torch.Tensor:
        """
        Get combined auxiliary inputs for privileged critic.

        Returns flattened tensor combining:
        - Normalized density info: [food_density, poison_density] (2 values)
        - Flattened ground truth view: is_food and is_poison channels (2 * H * W values)

        Total size: 2 + 2 * view_size^2

        For ground truth modes, the privileged view is the same as the policy view.
        For proxy modes, the privileged view reveals food vs poison that the policy can't see.

        IMPORTANT: Must be called AFTER _get_observations() in the same step.
        """
        density = self.get_density_info()  # (n_envs, 2)
        priv_view = self.get_privileged_view()  # (n_envs, 2, H, W)
        priv_flat = priv_view.reshape(self.n_envs, -1)  # (n_envs, 2*H*W)
        return torch.cat([density, priv_flat], dim=-1)  # (n_envs, 2 + 2*H*W)

    @property
    def num_critic_aux(self) -> int:
        """Number of auxiliary inputs for privileged critic."""
        return 2 + 2 * self.view_size * self.view_size


def create_torch_vec_env(
    n_envs: int,
    obs_spec: ObservationSpec,
    config: dict = None,
    device: Optional[torch.device] = None,
) -> TorchVecEnv:
    """Factory function to create a GPU-native vectorized environment."""
    return TorchVecEnv(
        n_envs=n_envs,
        obs_spec=obs_spec,
        config=config,
        device=device,
    )
