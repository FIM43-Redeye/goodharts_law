"""
ModeSpec: Central registry for training/behavior modes.

Defines observation format, reward type, and special training flags per mode.
This enables automatic configuration without hardcoding mode names throughout.
"""
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Type
import numpy as np


@lru_cache(maxsize=4)
def _get_distance_map(h: int, w: int) -> np.ndarray:
    """Get cached distance map for given dimensions."""
    y, x = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    dist = np.sqrt((y - center[0])**2 + (x - center[1])**2)
    dist[center] = 1e-6  # Avoid division by zero
    return dist


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


class RewardComputer(ABC):
    """
    Base class for computing shaped rewards for training.
    """
    
    def __init__(self, mode: str, spec: 'ObservationSpec', gamma: float = 0.99):
        """
        Initialize reward computer.
        
        Args:
            mode: Training mode name
            spec: Observation specification
            gamma: Discount factor (for potential-based shaping)
        """
        self.mode = mode
        self.spec = spec
        self.gamma = gamma
        
    @classmethod
    def create(cls, mode: str, spec: 'ObservationSpec', gamma: float = 0.99) -> 'RewardComputer':
        """
        Factory method to create the appropriate RewardComputer instance.
        """
        if spec.reward_strategy is None:
             raise ValueError(f"No reward strategy defined for mode: {mode}")
             
        # Instantiate the strategy class defined in the spec
        return spec.reward_strategy(mode, spec, gamma)
    
    def initialize(self, states: np.ndarray):

        """Initialize potentials for first step."""
        self.prev_potentials = self._compute_potentials(states)
    
    def compute(
        self,
        raw_rewards: np.ndarray,
        states: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> np.ndarray:
        """Compute shaped rewards for training."""
        # 1. Scale raw rewards based on reward type
        scaled_rewards = self._scale_rewards(raw_rewards)
        
        # 2. Add potential-based shaping
        next_potentials = self._compute_potentials(next_states)
        target_potentials = np.where(dones, 0.0, next_potentials)
        
        shaping = (target_potentials * self.gamma) - self.prev_potentials
        
        # Update for next step
        self.prev_potentials = next_potentials
        
        # 3. Combine and clip
        total_rewards = scaled_rewards + shaping
        total_rewards = np.clip(total_rewards, -5.0, 5.0)
        
        return total_rewards
    
    @abstractmethod
    def _compute_potentials(self, states: np.ndarray) -> np.ndarray:
        """Calculate potential-based shaping values."""
        pass
    
    def _scale_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """
        Scale raw rewards. Default is to scale by max magnitude (energy delta).
        Override for custom scaling (e.g. +1/-1).
        """
        from goodharts.configs.default_config import TRAINING_DEFAULTS
        # Use centralized default scaling
        scale = TRAINING_DEFAULTS.get('reward_scale', 0.1)
        return rewards * scale
        
    def _calculate_potential_from_target(self, states: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
        """Helper to calculate inverse distance potential to targets."""
        n, c, h, w = states.shape
        dist_map = _get_distance_map(h, w)
        
        # Vectorized min distance
        potentials = np.zeros(n, dtype=np.float32)
        
        # Find envs with targets
        has_target = target_mask.any(axis=(1, 2))
        if has_target.any():
            # Mask distance map where target is present
            masked_dist = np.where(target_mask[has_target], dist_map, np.inf)
            min_dists = masked_dist.reshape(masked_dist.shape[0], -1).min(axis=1)
            
            # Inverse potential: 0.1 / (dist + 0.5)
            # Max value (at dist=0) = 0.2
            # Value at dist=1 = 0.067
            # This ensures shaping is weak (<10% of consumption reward) and tapers hard.
            potentials[has_target] = 0.1 / (min_dists + 0.5)
            
        return potentials


class GroundTruthRewards(RewardComputer):
    """
    Standard ground truth rewards.
    """
    def _compute_potentials(self, states: np.ndarray) -> np.ndarray:
        target = states[:, 2, :, :] > 0.5
        return self._calculate_potential_from_target(states, target)


class HandholdRewards(RewardComputer):
    """
    Handhold rewards for easier learning.
    """
    def _scale_rewards(self, rewards: np.ndarray) -> np.ndarray:
        scaled = np.zeros_like(rewards)
        scaled[rewards > 0] = 1.0   # Food
        scaled[rewards < 0] = -1.0  # Poison
        return scaled

    def _compute_potentials(self, states: np.ndarray) -> np.ndarray:
        target = states[:, 2, :, :] > 0.5
        return self._calculate_potential_from_target(states, target)


class ProxyRewards(RewardComputer):
    """
    Proxy rewards based on 'interestingness'.
    """
    def _compute_potentials(self, states: np.ndarray) -> np.ndarray:
        interestingness = states[:, 2:, :, :].max(axis=1)
        target = interestingness > 0.5
        return self._calculate_potential_from_target(states, target)


class IllAdjustedRewards(RewardComputer):
    """
    Ill-adjusted proxy rewards.
    """
    def _scale_rewards(self, rewards: np.ndarray) -> np.ndarray:
        scaled = np.zeros_like(rewards)
        scaled[rewards > 0] = 1.0
        scaled[rewards < 0] = 0.9
        return scaled
        
    def _compute_potentials(self, states: np.ndarray) -> np.ndarray:
        interestingness = states[:, 2:, :, :].max(axis=1)
        target = interestingness > 0.5
        return self._calculate_potential_from_target(states, target)


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
        'proxy': ModeSpec(
            name='proxy',
            observation_channels=proxy_channels,
            reward_type='energy_delta',
            reward_strategy=ProxyRewards,
            freeze_energy_in_training=True,
            behavior_requirement='proxy_metric',
        ),
        'proxy_ill_adjusted': ModeSpec(
            name='proxy_ill_adjusted',
            observation_channels=proxy_channels,
            reward_type='interestingness',
            reward_strategy=IllAdjustedRewards,
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
