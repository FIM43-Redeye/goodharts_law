"""
ModeSpec: Central registry for training/behavior modes.

Defines observation format, reward type, and special training flags per mode.
This enables automatic configuration without hardcoding mode names throughout.
"""
from dataclasses import dataclass, field


@dataclass
class ModeSpec:
    """
    Complete specification for a training/behavior mode.
    
    This is the single source of truth for mode behavior.
    Training code and behaviors derive all parameters from this.
    """
    name: str
    observation_channels: list[str]
    reward_type: str  # 'energy_delta' or 'interestingness' 
    freeze_energy_in_training: bool = False  # If True, energy doesn't change during training
    behavior_requirement: str = 'ground_truth'  # What the behavior.requirements returns
    
    @property
    def num_channels(self) -> int:
        return len(self.observation_channels)


# =============================================================================
# MODE REGISTRY - Add new modes here
# =============================================================================

def _get_modes(config: dict) -> dict[str, ModeSpec]:
    """Build mode registry from config. Called once at startup."""
    CellType = config['CellType']
    gt_channels = [f"cell_{ct.name.lower()}" for ct in CellType.all_types()]
    gt_count = len(gt_channels)
    
    # Proxy channels: empty, wall, then interestingness repeated to match count
    proxy_channels = ['cell_empty', 'cell_wall'] + ['interestingness'] * (gt_count - 2)
    # We actually want to freeze energy for EVERYBODY so training always runs to time limit and reward accrues nicely
    return {
        'ground_truth': ModeSpec(
            name='ground_truth',
            observation_channels=gt_channels,
            reward_type='energy_delta',
            freeze_energy_in_training=True,
            behavior_requirement='ground_truth',
        ),
        'proxy': ModeSpec(
            name='proxy',
            observation_channels=proxy_channels,
            reward_type='energy_delta',
            freeze_energy_in_training=True,
            behavior_requirement='proxy_metric',
        ),
        'proxy_ill_adjusted': ModeSpec(
            name='proxy_ill_adjusted',
            observation_channels=proxy_channels,
            reward_type='interestingness',
            freeze_energy_in_training=True,
            behavior_requirement='proxy_metric',
        ),
    }


def get_all_mode_names(config: dict) -> list[str]:
    """Get list of all available mode names (for CLI choices)."""
    return list(_get_modes(config).keys())


def get_mode_for_requirement(requirement: str, config: dict) -> str:
    """
    Get mode name from a behavior requirement.
    
    This allows organisms to determine their observation mode from their
    behavior's requirements without hardcoding mode names.
    
    Args:
        requirement: A behavior requirement like 'ground_truth' or 'proxy_metric'
        config: Runtime config
        
    Returns:
        Mode name to use (e.g., 'ground_truth', 'proxy')
    """
    modes = _get_modes(config)
    
    for mode_name, spec in modes.items():
        if spec.behavior_requirement == requirement:
            return mode_name
    
    # Default to ground_truth if unknown requirement
    return 'ground_truth'


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
            freeze_energy_in_training=spec.freeze_energy_in_training,
        )
    
    def __repr__(self) -> str:
        extra = f", reward={self.reward_type}"
        if self.freeze_energy_in_training:
            extra += ", frozen_energy"
        return f"ObservationSpec(mode='{self.mode}', channels={self.num_channels}, view={self.view_size}x{self.view_size}{extra})"
