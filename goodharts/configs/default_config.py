from dataclasses import dataclass, fields
import numpy as np


@dataclass(frozen=True)
class CellTypeInfo:
    """
    Intrinsic properties of a cell type.
    
    Extensible: Add new properties here and they become available
    as observation channels for learned behaviors.
    
    Supports direct comparison with integers/numpy arrays:
        grid[y, x] == CellType.FOOD  # Just works!
    """
    value: int
    name: str
    color: tuple[int, int, int] = (128, 128, 128)  # RGB visualization color
    # Observable properties (can be exposed to agents)
    interestingness: float = 0.0
    energy_reward: float = 0.0
    energy_penalty: float = 0.0
    # Add more properties here for future cell types:
    # toxicity: float = 0.0
    # nutrition: float = 0.0  
    # visibility: float = 1.0
    
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
        """Channel index for observation tensors. Equals value for now."""
        return self.value
    
    def get_property(self, prop_name: str) -> float:
        """Get a property value by name."""
        return getattr(self, prop_name, 0.0)
    
    @classmethod
    def property_names(cls) -> list[str]:
        """Get all numeric property names (excludes value, name, color)."""
        return [f.name for f in fields(cls) if f.name not in ('value', 'name', 'color')]


class CellType:
    """
    Registry of all cell types with their intrinsic properties.
    
    To add new cell types:
    1. Add a class attribute like: NEW_TYPE = CellTypeInfo(N, "NewType", ...)
    2. That's it! all_types() auto-discovers via introspection.
    """
    EMPTY = CellTypeInfo(0, "Empty", color=(26, 26, 46))
    WALL = CellTypeInfo(1, "Wall", color=(74, 74, 74))
    FOOD = CellTypeInfo(2, "Food", color=(22, 199, 154), interestingness=1.0, energy_reward=15.0)
    POISON = CellTypeInfo(3, "Poison", color=(255, 107, 107), interestingness=0.9, energy_penalty=50.0)
    # Agent types (visible in observations)
    PREY = CellTypeInfo(4, "Prey", color=(0, 255, 255), interestingness=0.3)
    PREDATOR = CellTypeInfo(5, "Predator", color=(255, 0, 0), interestingness=1.0, energy_reward=25.0)  
    
    _BY_VALUE: dict[int, CellTypeInfo] | None = None
    _ALL_TYPES: list[CellTypeInfo] | None = None
    
    @classmethod
    def by_value(cls, value: int) -> CellTypeInfo | None:
        """Lookup cell type by its integer value."""
        if cls._BY_VALUE is None:
            cls._BY_VALUE = {ct.value: ct for ct in cls.all_types()}
        return cls._BY_VALUE.get(int(value))
    
    @classmethod
    def all_types(cls) -> list[CellTypeInfo]:
        """
        Get all registered cell types in value order.
        
        Auto-discovers all CellTypeInfo class attributes via introspection.
        """
        if cls._ALL_TYPES is None:
            types = []
            for name in dir(cls):
                if name.startswith('_'):
                    continue
                attr = getattr(cls, name)
                if isinstance(attr, CellTypeInfo):
                    types.append(attr)
            cls._ALL_TYPES = sorted(types, key=lambda t: t.value)
        return cls._ALL_TYPES
    
    @classmethod
    def num_types(cls) -> int:
        """Number of distinct cell types."""
        return len(cls.all_types())

# Simulation Physics / Hyperparameters
ENERGY_START = 50.0
ENERGY_MOVE_COST = 0.1  # Base cost per unit distance
MOVE_COST_EXPONENT = 1.5  # Nonlinear scaling: cost = distance^exponent * base_cost
MAX_MOVE_DISTANCE = 3  # Speed cap: organisms can't move more than this per step

# Grid Settings
GRID_WIDTH = 100
GRID_HEIGHT = 100
GRID_FOOD_INIT = 50
GRID_POISON_INIT = 10
AGENTS_SETUP = [
    {'behavior_class': 'OmniscientSeeker', 'count': 5},
    {'behavior_class': 'ProxySeeker', 'count': 5}
]

# Agent Properties
AGENT_VIEW_RANGE = 5

# World Topology
WORLD_LOOP = False  # If True, world wraps (toroidal); if False, has edges


def get_config(config_path: str | None = None):
    """
    Build runtime config dictionary from TOML config file.
    
    This bridges the TOML config to the legacy dictionary format
    expected by simulation code.
    
    Args:
        config_path: Optional path to specific config file.
                     If None, uses auto-detection (config.toml > config.default.toml)
    """
    from .observation_spec import ObservationSpec
    from goodharts.config import load_config, get_config as get_toml_config
    
    # Load TOML config
    if config_path:
        toml_cfg = load_config(config_path)
    else:
        toml_cfg = get_toml_config()
    
    # Extract sections with defaults
    world = toml_cfg.get('world', {})
    resources = toml_cfg.get('resources', {})
    agent_cfg = toml_cfg.get('agent', {})
    agents_list = toml_cfg.get('agents', [])
    
    # Build agents setup from [[agents]] list
    agents_setup = []
    for agent in agents_list:
        setup = {
            'behavior_class': agent.get('type', 'OmniscientSeeker'),
            'count': agent.get('count', 1)
        }
        if 'model' in agent:
            setup['model_path'] = agent['model']
        agents_setup.append(setup)
    
    # Default agents if none specified
    if not agents_setup:
        agents_setup = [
            {'behavior_class': 'OmniscientSeeker', 'count': 5},
            {'behavior_class': 'ProxySeeker', 'count': 5}
        ]
    
    # Build runtime config
    config = {
        # World
        'GRID_WIDTH': world.get('width', 100),
        'GRID_HEIGHT': world.get('height', 100),
        'WORLD_LOOP': world.get('loop', False),
        
        # Resources
        'GRID_FOOD_INIT': resources.get('food', 50),
        'GRID_POISON_INIT': resources.get('poison', 10),
        'RESPAWN_RESOURCES': resources.get('respawn', True),
        
        # Agent physics
        'ENERGY_START': agent_cfg.get('energy_start', 50.0),
        'ENERGY_MOVE_COST': agent_cfg.get('energy_move_cost', 0.1),
        'MOVE_COST_EXPONENT': agent_cfg.get('move_cost_exponent', 1.5),
        'MAX_MOVE_DISTANCE': agent_cfg.get('max_move_distance', 3),
        'AGENT_VIEW_RANGE': agent_cfg.get('view_range', 5),
        
        # Agents
        'AGENTS_SETUP': agents_setup,
        
        # Cell types (always from Python - these are code, not config)
        'CellType': CellType,
    }
    
    # Observation spec factory
    config['get_observation_spec'] = lambda mode: ObservationSpec.for_mode(mode, config)
    
    return config


