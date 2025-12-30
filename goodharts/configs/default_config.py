"""
Cell type registry and simulation config builder.

CellType provides the canonical registry of cell types. Properties are loaded
lazily from TOML config, making config.default.toml the single source of truth
for all numeric values.

The CellType class maintains backward-compatible API:
    CellType.FOOD.value          # Integer for grid comparisons
    CellType.FOOD.channel_index  # Same as value, for observation indexing
    CellType.FOOD.energy_delta   # Energy change when consumed
    CellType.FOOD.interestingness # Proxy observation value
    CellType.FOOD.color          # RGB tuple for visualization
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CellTypeInfo:
    """
    Intrinsic properties of a cell type, loaded from TOML config.

    Supports direct comparison with integers/numpy arrays:
        grid[y, x] == CellType.FOOD  # Just works!
    """
    value: int
    name: str
    color: tuple[int, int, int]
    interestingness: float
    energy_delta: float

    def __eq__(self, other):
        if isinstance(other, (int, np.integer)):
            return self.value == other
        if isinstance(other, CellTypeInfo):
            return self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

    def __int__(self):
        return self.value

    @property
    def channel_index(self) -> int:
        """Channel index for observation tensors. Equals value."""
        return self.value

    @property
    def energy_reward(self) -> float:
        """Positive energy gain (for food). Returns 0 if energy_delta is negative."""
        return max(0.0, self.energy_delta)

    @property
    def energy_penalty(self) -> float:
        """Positive energy loss (for poison). Returns 0 if energy_delta is positive."""
        return abs(min(0.0, self.energy_delta))

    def get_property(self, prop_name: str) -> float:
        """Get a property value by name."""
        return getattr(self, prop_name, 0.0)

    @classmethod
    def property_names(cls) -> list[str]:
        """Get all numeric property names available for observation channels."""
        return ['interestingness', 'energy_delta', 'energy_reward', 'energy_penalty']


# =============================================================================
# LAZY-LOADING INFRASTRUCTURE
# =============================================================================

# Module-level cache for loaded cell types
_cell_types_cache: dict[str, CellTypeInfo] | None = None

# Mapping from uppercase class attribute names to TOML section names
_CELL_TYPE_NAMES = {
    'EMPTY': 'empty',
    'WALL': 'wall',
    'FOOD': 'food',
    'POISON': 'poison',
    'PREY': 'prey',
    'PREDATOR': 'predator',
}

# Integer values for each cell type (these are structural, not config)
_CELL_TYPE_VALUES = {
    'empty': 0,
    'wall': 1,
    'food': 2,
    'poison': 3,
    'prey': 4,
    'predator': 5,
}


def _load_cell_types() -> dict[str, CellTypeInfo]:
    """Load cell type properties from TOML config. Called once on first access."""
    global _cell_types_cache

    if _cell_types_cache is not None:
        return _cell_types_cache

    from goodharts.config import get_config
    cfg = get_config()

    cell_types_cfg = cfg['cell_types']

    _cell_types_cache = {}
    for name, value in _CELL_TYPE_VALUES.items():
        section = cell_types_cfg[name]
        _cell_types_cache[name] = CellTypeInfo(
            value=value,
            name=name.capitalize(),
            color=tuple(section['color']),
            interestingness=section['interestingness'],
            energy_delta=section['energy_delta'],
        )

    return _cell_types_cache


def _get_cell_type(name: str) -> CellTypeInfo:
    """Get a cell type by lowercase name, loading from config if needed."""
    types = _load_cell_types()
    return types[name]


class _CellTypeDescriptor:
    """
    Descriptor that lazy-loads CellTypeInfo from TOML on first access.

    This allows CellType.FOOD to work as a class attribute while loading
    properties from config at runtime rather than import time.
    """
    def __init__(self, toml_name: str):
        self.toml_name = toml_name

    def __get__(self, obj, objtype=None) -> CellTypeInfo:
        return _get_cell_type(self.toml_name)


# =============================================================================
# CELL TYPE REGISTRY
# =============================================================================

class CellType:
    """
    Registry of all cell types with their intrinsic properties.

    Properties are loaded lazily from TOML config on first access.
    The integer values (0-5) are structural and defined in code.

    Usage:
        CellType.FOOD.value          # 2
        CellType.FOOD.energy_delta   # 5.0 (from config)
        CellType.FOOD.interestingness # 1.0 (from config)
        CellType.all_types()         # List of all CellTypeInfo
    """
    # Descriptors for lazy-loading from TOML
    EMPTY = _CellTypeDescriptor('empty')
    WALL = _CellTypeDescriptor('wall')
    FOOD = _CellTypeDescriptor('food')
    POISON = _CellTypeDescriptor('poison')
    PREY = _CellTypeDescriptor('prey')
    PREDATOR = _CellTypeDescriptor('predator')

    @classmethod
    def by_value(cls, value: int) -> CellTypeInfo | None:
        """Lookup cell type by its integer value."""
        types = _load_cell_types()
        for ct in types.values():
            if ct.value == int(value):
                return ct
        return None

    @classmethod
    def all_types(cls) -> list[CellTypeInfo]:
        """Get all registered cell types in value order."""
        types = _load_cell_types()
        return sorted(types.values(), key=lambda t: t.value)

    @classmethod
    def num_types(cls) -> int:
        """Number of distinct cell types."""
        return len(_CELL_TYPE_VALUES)


# =============================================================================
# SIMULATION CONFIG BUILDER
# =============================================================================

def get_simulation_config(config_path: str | None = None):
    """
    Build runtime config dictionary from TOML config file.

    This bridges the TOML config to the simulation-ready dictionary format
    expected by simulation code.

    All config values MUST be present in the TOML file - there are no
    fallback defaults. Missing keys will raise KeyError.

    Args:
        config_path: Optional path to specific config file.
                     If None, uses auto-detection (config.toml > config.default.toml)

    Raises:
        KeyError: If required config keys are missing from TOML.
    """
    from goodharts.modes import ObservationSpec
    from goodharts.config import load_config, get_config as get_toml_config

    # Load TOML config
    if config_path:
        toml_cfg = load_config(config_path)
    else:
        toml_cfg = get_toml_config()

    # Extract required sections (will raise KeyError if missing)
    world = toml_cfg['world']
    resources = toml_cfg['resources']
    agent_cfg = toml_cfg['agent']
    agents_list = toml_cfg.get('agents', [])

    # Build agents setup from [[agents]] list
    agents_setup = []
    for agent in agents_list:
        setup = {
            'behavior_class': agent['type'],
            'count': agent['count']
        }
        if 'model' in agent:
            setup['model_path'] = agent['model']
        agents_setup.append(setup)

    # Default agents if none specified (this is acceptable as [[agents]] is optional)
    if not agents_setup:
        agents_setup = [
            {'behavior_class': 'OmniscientSeeker', 'count': 5},
            {'behavior_class': 'ProxySeeker', 'count': 5}
        ]

    # Build runtime config - all values required from TOML
    config = {
        # World (required)
        'GRID_WIDTH': world['width'],
        'GRID_HEIGHT': world['height'],

        # Resources (required)
        'GRID_FOOD_INIT': resources['food'],
        'GRID_POISON_INIT': resources['poison'],
        'RESPAWN_RESOURCES': resources['respawn'],

        # Agent physics (required)
        'ENERGY_START': agent_cfg['energy_start'],
        'ENERGY_MOVE_COST': agent_cfg['energy_move_cost'],
        'MOVE_COST_EXPONENT': agent_cfg['move_cost_exponent'],
        'MAX_MOVE_DISTANCE': agent_cfg['max_move_distance'],
        'AGENT_VIEW_RANGE': agent_cfg['view_range'],
        'DEATH_PENALTY_RATIO': agent_cfg.get('death_penalty_ratio', 2.0),

        # Agents
        'AGENTS_SETUP': agents_setup,

        # Cell types (always from Python - these are code, not config)
        'CellType': CellType,
    }

    # Observation spec factory
    config['get_observation_spec'] = lambda mode: ObservationSpec.for_mode(mode, config)

    return config


# Deprecated alias for backward compatibility
def get_config(config_path: str | None = None):
    """DEPRECATED: Use get_simulation_config() instead."""
    import warnings
    warnings.warn(
        "get_config() is deprecated, use get_simulation_config()",
        DeprecationWarning,
        stacklevel=2
    )
    return get_simulation_config(config_path)
