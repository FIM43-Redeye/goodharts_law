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
    
    Uses torch tensors exclusively. All inputs/outputs are torch tensors.
    """
    
    def __init__(self, mode: str, spec: 'ObservationSpec', gamma: float = 0.99,
                 shaping_coef: float = 0.5, device: torch.device = None):
        """
        Initialize reward computer.

        Args:
            mode: Training mode name
            spec: Observation specification
            gamma: Discount factor (for potential-based shaping)
            shaping_coef: Magnitude of the potential (e.g. 0.5 for food)
            device: Torch device (auto-detected if None)
        """
        self.mode = mode
        self.spec = spec
        self.gamma = gamma
        self.shaping_coef = shaping_coef
        self.device = device or get_device()
        self.prev_potentials: torch.Tensor = None
        # Cache constant tensor to avoid per-call allocation
        self._inf_tensor = torch.tensor(float('inf'), device=self.device)
        
    @classmethod
    def create(cls, mode: str, spec: 'ObservationSpec', gamma: float = 0.99,
               shaping_coef: float = 0.5, device: torch.device = None) -> 'RewardComputer':
        """Factory method to create the appropriate RewardComputer instance."""
        if spec.reward_strategy is None:
             raise ValueError(f"No reward strategy defined for mode: {mode}")
        return spec.reward_strategy(mode, spec, gamma, shaping_coef, device)
    
    def initialize(self, states: torch.Tensor):
        """Initialize potentials for first step."""
        self.prev_potentials = self._compute_potentials(states)
    
    def compute(
        self,
        raw_rewards: torch.Tensor,
        states: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute shaped rewards for training."""
        # 1. Scale raw rewards based on reward type
        scaled_rewards = self._scale_rewards(raw_rewards.float())
        
        # 2. Add potential-based shaping
        next_potentials = self._compute_potentials(next_states)
        target_potentials = torch.where(dones.bool(), 
                                         torch.zeros_like(next_potentials), 
                                         next_potentials)
        
        shaping = (target_potentials * self.gamma) - self.prev_potentials
        
        # Update for next step
        self.prev_potentials = next_potentials
        
        # 3. Combine and clip
        total_rewards = scaled_rewards + shaping
        total_rewards = torch.clamp(total_rewards, -20.0, 20.0)
        
        return total_rewards
    
    @abstractmethod
    def _compute_potentials(self, states: torch.Tensor) -> torch.Tensor:
        """Calculate potential-based shaping values."""
        pass
    
    def _scale_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Scale raw rewards. Default is 1.0 (pass-through).
        Override for custom scaling (e.g. +1/-1).
        """
        return rewards
        
    def _calculate_potential_from_target(self, states: torch.Tensor, 
                                          target_mask: torch.Tensor) -> torch.Tensor:
        """Helper to calculate inverse distance potential to targets (vectorized)."""
        n, c, h, w = states.shape
        dist_map = _get_distance_map_torch(h, w, states.device)
        
        # Broadcast dist_map to (n, h, w) and mask non-targets with inf
        dist_map_expanded = dist_map.unsqueeze(0).expand(n, -1, -1)  # (n, h, w)
        
        # Where target_mask is True, use distance; else use inf
        masked_dist = torch.where(target_mask, dist_map_expanded, self._inf_tensor)
        
        # Compute min distance per env (flatten spatial dims, then min)
        min_dist = masked_dist.view(n, -1).min(dim=1).values  # (n,)
        
        # Inverse potential: coef / (dist + 0.5), handle inf as 0
        potentials = self.shaping_coef / (min_dist + 0.5)
        potentials = torch.where(torch.isinf(min_dist), 
                                  torch.zeros_like(potentials), 
                                  potentials)
        
        return potentials


class GroundTruthRewards(RewardComputer):
    """Standard ground truth rewards - no shaping, just raw energy delta."""
    def _compute_potentials(self, states: torch.Tensor) -> torch.Tensor:
        # No potential-based shaping for ground truth mode
        return torch.zeros(states.shape[0], device=states.device)


class HandholdRewards(RewardComputer):
    """
    Handhold rewards for easier learning.
    
    Scales rewards linearly to [-1, 1] based on the max possible energy 
    values from the config (food/poison energy). This makes the reward 
    signal more consistent and easier for the agent to learn from.
    """
    # Scaling constants derived from CellType config
    # Food: +5.0 energy, Poison: -3.0 energy
    # These are computed once at import time from the config
    MAX_POSITIVE = 5.0   # CellType.FOOD.energy_reward
    MAX_NEGATIVE = -3.0  # -CellType.POISON.energy_penalty
    
    def _scale_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        # Scale linearly: positive rewards scaled by MAX_POSITIVE, negative by MAX_NEGATIVE
        # This maps food (+5) to +1, poison (-3) to -1, movement costs (~-0.05) to ~-0.017
        scaled = torch.zeros_like(rewards)
        
        # Positive rewards: divide by MAX_POSITIVE
        pos_mask = rewards > 0
        scaled[pos_mask] = rewards[pos_mask] / self.MAX_POSITIVE
        
        # Negative rewards: divide by abs(MAX_NEGATIVE)
        neg_mask = rewards < 0
        scaled[neg_mask] = rewards[neg_mask] / abs(self.MAX_NEGATIVE)
        
        # Clamp to [-1, 1] for safety
        return torch.clamp(scaled, -1.0, 1.0)

    def _compute_potentials(self, states: torch.Tensor) -> torch.Tensor:
        target = states[:, 2, :, :] > 0.5
        return self._calculate_potential_from_target(states, target)


class ProxyJammedRewards(RewardComputer):
    """
    Rewards for proxy_jammed mode (information asymmetry).
    Agent sees interestingness, rewarded for energy delta.
    No potential-based shaping - agent must learn from raw energy rewards.
    """
    def _compute_potentials(self, states: torch.Tensor) -> torch.Tensor:
        # No potential-based shaping for proxy_jammed mode
        return torch.zeros(states.shape[0], device=states.device)


class ProxyRewards(RewardComputer):
    """
    Rewards for proxy mode (main Goodhart failure case).
    Agent sees interestingness, rewarded for interestingness consumption.
    No potential-based shaping - demonstrates pure Goodhart failure.
    """
    def _scale_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        scaled = torch.zeros_like(rewards)
        scaled[rewards > 0] = 1.0
        scaled[rewards < 0] = 0.9
        return scaled

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
