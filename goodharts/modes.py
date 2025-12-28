"""
ModeSpec: Central registry for training/behavior modes.

Defines observation format, reward type, and special training flags per mode.
This enables automatic configuration without hardcoding mode names throughout.
"""
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Type
import torch

from goodharts.utils.device import get_device


@dataclass
class ObservationSpec:
    """
    Defines the observation format for a behavior mode.
    
    All model architectures and training code should derive input dimensions
    from this spec, ensuring changes to observation format propagate automatically.
    """
    mode: str
    channel_names: list[str]
    view_range: int
    reward_type: str = 'energy_delta'
    reward_strategy: Type['RewardComputer'] = None
    freeze_energy_in_training: bool = False
    
    @property
    def num_channels(self) -> int:
        return len(self.channel_names)
    
    @property
    def view_size(self) -> int:
        return 2 * self.view_range + 1
    
    @property
    def input_shape(self) -> tuple[int, int]:
        return (self.view_size, self.view_size)
    
    @classmethod
    def for_mode(cls, mode: str, config: dict) -> 'ObservationSpec':
        """Factory: create observation spec for a given mode."""
        modes = _get_modes(config)
        
        if mode not in modes:
            valid = ', '.join(modes.keys())
            raise ValueError(f"Unknown mode: {mode}. Must be one of: {valid}")
        
        spec = modes[mode]
        view_range = config['AGENT_VIEW_RANGE']
        
        return cls(
            mode=mode,
            channel_names=spec.observation_channels,
            view_range=view_range,
            reward_type=spec.reward_type,
            reward_strategy=spec.reward_strategy,
            freeze_energy_in_training=spec.freeze_energy_in_training,
        )
    
    def __repr__(self) -> str:
        extra = f", reward={self.reward_type}"
        if self.freeze_energy_in_training:
            extra += ", frozen_energy"
        return f"ObservationSpec(mode='{self.mode}', channels={self.num_channels}, view={self.view_size}x{self.view_size}{extra})"


# Cache for torch distance maps per (h, w, device) tuple
_torch_distance_map_cache: dict[tuple[int, int, str], torch.Tensor] = {}

def _get_distance_map_torch(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Get cached distance map as torch tensor on specified device."""
    cache_key = (h, w, str(device))
    if cache_key not in _torch_distance_map_cache:
        y = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1)
        x = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(0)
        center_y, center_x = h // 2, w // 2
        dist = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
        dist[center_y, center_x] = 1e-6  # Avoid division by zero
        _torch_distance_map_cache[cache_key] = dist
    return _torch_distance_map_cache[cache_key]


class RewardComputer(ABC):
    """
    Base class for computing shaped rewards for training.

    The environment reports what happened (eating_info), and the RewardComputer
    assigns value based on the training mode. This cleanly separates physical
    simulation from reward signal design.

    Uses torch tensors exclusively. All inputs/outputs are torch tensors.

    Subclasses define their own shaping behavior - the base class does not
    accept shaping coefficients from config. This keeps reward shaping as
    an implementation detail of each training mode's pedagogical design.

    All penalties are computed relative to food_reward, ensuring the reward
    signal scales properly regardless of the absolute values in config.
    """

    # Death penalty as a multiple of food_reward (dying costs N food worth)
    DEATH_PENALTY_RATIO = 2.0

    # Reference food reward for scaling movement cost
    # Movement cost ratio is computed relative to this baseline
    REFERENCE_FOOD_REWARD = 5.0

    def __init__(self, mode: str, spec: 'ObservationSpec', config: dict,
                 gamma: float = 0.99, device: torch.device = None):
        """
        Initialize reward computer.

        Args:
            mode: Training mode name
            spec: Observation specification
            config: Simulation config dict (contains CellType, ENERGY_MOVE_COST, etc.)
            gamma: Discount factor (for potential-based shaping)
            device: Torch device (auto-detected if None)
        """
        self.mode = mode
        self.spec = spec
        self.gamma = gamma
        self.device = device or get_device()
        self.prev_potentials: torch.Tensor = None

        # Extract reward values from config (cell type properties from TOML)
        CellType = config['CellType']
        self.food_reward = CellType.FOOD.energy_reward
        self.poison_penalty = CellType.POISON.energy_penalty
        self.food_interestingness = CellType.FOOD.interestingness
        self.poison_interestingness = CellType.POISON.interestingness
        self._raw_movement_cost = config['ENERGY_MOVE_COST']

        # Compute scaled penalties (relative to food reward)
        # This preserves the original ratio when food_reward was REFERENCE_FOOD_REWARD
        self.death_penalty = self.food_reward * self.DEATH_PENALTY_RATIO
        self.movement_cost = self._raw_movement_cost * (self.food_reward / self.REFERENCE_FOOD_REWARD)

        # Cache constant tensor to avoid per-call allocation
        self._inf_tensor = torch.tensor(float('inf'), device=self.device)

    @classmethod
    def create(cls, mode: str, spec: 'ObservationSpec', config: dict,
               gamma: float = 0.99, device: torch.device = None) -> 'RewardComputer':
        """Factory method to create the appropriate RewardComputer instance."""
        if spec.reward_strategy is None:
            raise ValueError(f"No reward strategy defined for mode: {mode}")
        return spec.reward_strategy(mode, spec, config, gamma, device)

    def initialize(self, states: torch.Tensor):
        """Initialize potentials for first step."""
        self.prev_potentials = self._compute_potentials(states)

    def compute(
        self,
        eating_info: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        states: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute shaped rewards for training.

        Args:
            eating_info: (food_mask, poison_mask, starved_mask) from environment
            states: Current observations
            next_states: Next observations
            terminated: True when agent truly died (starvation). Do NOT pass
                        truncated (time limits) - those episodes are still "alive".

        Returns:
            Shaped reward tensor for PPO training
        """
        food_mask, poison_mask, starved_mask = eating_info

        # 1. Compute base rewards from eating info (mode-specific)
        base_rewards = self._compute_base_rewards(food_mask, poison_mask, starved_mask)

        # 2. Add potential-based shaping
        # Only zero potential on TRUE termination (death), not on truncation (time limit)
        next_potentials = self._compute_potentials(next_states)
        target_potentials = torch.where(terminated.bool(),
                                         torch.zeros_like(next_potentials),
                                         next_potentials)

        shaping = (target_potentials * self.gamma) - self.prev_potentials

        # Update for next step
        self.prev_potentials = next_potentials

        # 3. Combine and clip
        total_rewards = base_rewards + shaping
        total_rewards = torch.clamp(total_rewards, -20.0, 20.0)

        return total_rewards

    @abstractmethod
    def _compute_base_rewards(
        self,
        food_mask: torch.Tensor,
        poison_mask: torch.Tensor,
        starved_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute base rewards from eating info. Mode-specific.

        Args:
            food_mask: Boolean tensor, True where agent ate food
            poison_mask: Boolean tensor, True where agent ate poison
            starved_mask: Boolean tensor, True where agent died from starvation

        Returns:
            Base reward tensor before potential shaping
        """
        pass

    @abstractmethod
    def _compute_potentials(self, states: torch.Tensor) -> torch.Tensor:
        """Calculate potential-based shaping values."""
        pass

    def _calculate_potential_from_target(self, states: torch.Tensor,
                                          target_mask: torch.Tensor,
                                          shaping_coef: float) -> torch.Tensor:
        """
        Helper to calculate inverse distance potential to targets (vectorized).

        Args:
            states: Observation tensor (N, C, H, W)
            target_mask: Boolean mask of target cells (N, H, W)
            shaping_coef: Magnitude of the potential (mode-specific)

        Returns:
            Potential values per environment (N,)
        """
        n, c, h, w = states.shape
        dist_map = _get_distance_map_torch(h, w, states.device)

        # Broadcast dist_map to (n, h, w) and mask non-targets with inf
        dist_map_expanded = dist_map.unsqueeze(0).expand(n, -1, -1)  # (n, h, w)

        # Where target_mask is True, use distance; else use inf
        masked_dist = torch.where(target_mask, dist_map_expanded, self._inf_tensor)

        # Compute min distance per env (flatten spatial dims, then min)
        min_dist = masked_dist.view(n, -1).min(dim=1).values  # (n,)

        # Inverse potential: coef / (dist + 0.5), handle inf as 0
        potentials = shaping_coef / (min_dist + 0.5)
        potentials = torch.where(torch.isinf(min_dist),
                                  torch.zeros_like(potentials),
                                  potentials)

        return potentials


class GroundTruthRewards(RewardComputer):
    """
    Standard ground truth rewards - full energy model.

    Reward = food_reward - poison_penalty - movement_cost - death_penalty
    Agent learns the true survival dynamics.
    """
    def _compute_base_rewards(
        self,
        food_mask: torch.Tensor,
        poison_mask: torch.Tensor,
        starved_mask: torch.Tensor
    ) -> torch.Tensor:
        n_envs = food_mask.shape[0]
        rewards = torch.zeros(n_envs, device=food_mask.device)

        # Eating rewards
        rewards = torch.where(food_mask, rewards + self.food_reward, rewards)
        rewards = torch.where(poison_mask, rewards - self.poison_penalty, rewards)

        # Movement cost (every step costs energy)
        rewards = rewards - self.movement_cost

        # Death penalty for starvation (scaled relative to food reward)
        rewards = torch.where(starved_mask, rewards - self.death_penalty, rewards)

        return rewards

    def _compute_potentials(self, states: torch.Tensor) -> torch.Tensor:
        # No potential-based shaping for ground truth mode
        return torch.zeros(states.shape[0], device=states.device)


class HandholdRewards(RewardComputer):
    """
    Handhold rewards for easier learning.

    Same as ground truth but scaled to [-1, 1] range for more consistent
    gradient signal. Also includes potential-based shaping toward food.

    All costs are fixed ratios in the normalized space, independent of
    the absolute values in config. This ensures stable learning regardless
    of how reward magnitudes are configured.
    """
    # Shaping coefficient for potential-based reward toward food
    FOOD_SHAPING_COEF = 0.5

    # Movement cost as fraction of normalized food reward (+1.0)
    # 0.01 means 100 steps of movement costs the same as one food
    MOVEMENT_COST_RATIO = 0.01

    # Death penalty in normalized space (dying costs 2 food worth)
    NORMALIZED_DEATH_PENALTY = 2.0

    def _compute_base_rewards(
        self,
        food_mask: torch.Tensor,
        poison_mask: torch.Tensor,
        starved_mask: torch.Tensor
    ) -> torch.Tensor:
        n_envs = food_mask.shape[0]
        rewards = torch.zeros(n_envs, device=food_mask.device)

        # Eating rewards (normalized to +1/-1)
        rewards = torch.where(food_mask, rewards + 1.0, rewards)
        rewards = torch.where(poison_mask, rewards - 1.0, rewards)

        # Movement cost (fixed ratio, independent of config values)
        rewards = rewards - self.MOVEMENT_COST_RATIO

        # Death penalty (fixed in normalized space)
        rewards = torch.where(starved_mask, rewards - self.NORMALIZED_DEATH_PENALTY, rewards)

        return rewards

    def _compute_potentials(self, states: torch.Tensor) -> torch.Tensor:
        # Potential-based shaping toward food (channel 2 in ground truth obs)
        target = states[:, 2, :, :] > 0.5
        return self._calculate_potential_from_target(states, target, self.FOOD_SHAPING_COEF)


class ProxyJammedRewards(RewardComputer):
    """
    Rewards for proxy_jammed mode (information asymmetry).

    Agent sees interestingness observations but is rewarded for true energy.
    This tests whether agents can learn to avoid poison when they can't
    distinguish it from food visually but feel the energy consequences.
    """
    def _compute_base_rewards(
        self,
        food_mask: torch.Tensor,
        poison_mask: torch.Tensor,
        starved_mask: torch.Tensor
    ) -> torch.Tensor:
        # Same as ground truth - full energy model
        n_envs = food_mask.shape[0]
        rewards = torch.zeros(n_envs, device=food_mask.device)

        rewards = torch.where(food_mask, rewards + self.food_reward, rewards)
        rewards = torch.where(poison_mask, rewards - self.poison_penalty, rewards)
        rewards = rewards - self.movement_cost
        rewards = torch.where(starved_mask, rewards - self.death_penalty, rewards)

        return rewards

    def _compute_potentials(self, states: torch.Tensor) -> torch.Tensor:
        # No potential-based shaping for proxy_jammed mode
        return torch.zeros(states.shape[0], device=states.device)


class ProxyRewards(RewardComputer):
    """
    Rewards for proxy mode (main Goodhart failure case).

    Agent sees interestingness and is rewarded for interestingness consumption.
    No movement cost, no death penalty - pure interestingness optimization.

    This demonstrates Goodhart's Law: the agent eagerly pursues both food
    and poison (both have high interestingness), eventually dying from
    poison accumulation despite "succeeding" at its proxy objective.

    Interestingness values come from config (cell_types), not hardcoded here.
    """
    def _compute_base_rewards(
        self,
        food_mask: torch.Tensor,
        poison_mask: torch.Tensor,
        starved_mask: torch.Tensor
    ) -> torch.Tensor:
        n_envs = food_mask.shape[0]
        rewards = torch.zeros(n_envs, device=food_mask.device)

        # Reward = interestingness of consumed cell (from config)
        rewards = torch.where(food_mask, rewards + self.food_interestingness, rewards)
        rewards = torch.where(poison_mask, rewards + self.poison_interestingness, rewards)

        # No movement cost - moving doesn't consume interestingness
        # No death penalty - the agent doesn't "know" about energy

        return rewards

    def _compute_potentials(self, states: torch.Tensor) -> torch.Tensor:
        # No potential-based shaping for proxy mode
        return torch.zeros(states.shape[0], device=states.device)

@dataclass
class ModeSpec:
    """
    Complete specification for a training/behavior mode.
    """
    name: str
    observation_channels: list[str]
    reward_type: str  # 'energy_delta' or 'interestingness' 
    reward_strategy: Type[RewardComputer]  # Class of RewardComputer strategy
    freeze_energy_in_training: bool = False
    behavior_requirement: str = 'ground_truth'
    
    @property
    def num_channels(self) -> int:
        return len(self.observation_channels)


# =============================================================================
# MODE REGISTRY
# =============================================================================

def _get_modes(config: dict) -> dict[str, ModeSpec]:
    """Build mode registry from config. Called once at startup."""
    CellType = config['CellType']
    gt_channels = [f"cell_{ct.name.lower()}" for ct in CellType.all_types()]
    gt_count = len(gt_channels)
    
    # Proxy channels: empty, wall, then interestingness repeated to match count
    proxy_channels = ['cell_empty', 'cell_wall'] + ['interestingness'] * (gt_count - 2)
    
    return {
        'ground_truth': ModeSpec(
            name='ground_truth',
            observation_channels=gt_channels,
            reward_type='energy_delta',
            reward_strategy=GroundTruthRewards,
            freeze_energy_in_training=True,
            behavior_requirement='ground_truth',
        ),
        'ground_truth_handhold': ModeSpec(
            name='ground_truth_handhold',
            observation_channels=gt_channels,
            reward_type='shaped',
            reward_strategy=HandholdRewards,
            freeze_energy_in_training=True,
            behavior_requirement='ground_truth',
        ),
        'proxy_jammed': ModeSpec(
            name='proxy_jammed',
            observation_channels=proxy_channels,
            reward_type='energy_delta',
            reward_strategy=ProxyJammedRewards,
            freeze_energy_in_training=True,
            behavior_requirement='proxy_metric',
        ),
        'proxy': ModeSpec(
            name='proxy',
            observation_channels=proxy_channels,
            reward_type='interestingness',
            reward_strategy=ProxyRewards,
            freeze_energy_in_training=True,
            behavior_requirement='proxy_metric',
        ),
    }


def get_all_mode_names(config: dict) -> list[str]:
    """Get list of all available mode names (for CLI choices)."""
    return list(_get_modes(config).keys())


def get_mode_for_requirement(requirement: str, config: dict) -> str:
    """Get mode name from a behavior requirement."""
    modes = _get_modes(config)
    
    for mode_name, spec in modes.items():
        if spec.behavior_requirement == requirement:
            return mode_name
    
    return 'ground_truth'
