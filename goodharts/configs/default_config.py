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
    
    def get_property(self, prop_name: str) -> float:
        """Get a property value by name."""
        return getattr(self, prop_name, 0.0)
    
    @classmethod
    def property_names(cls) -> list[str]:
        """Get all numeric property names (excludes value and name)."""
        return [f.name for f in fields(cls) if f.name not in ('value', 'name')]


class CellType:
    """
    Registry of all cell types with their intrinsic properties.
    
    To add new cell types:
    1. Add a class attribute like: NEW_TYPE = CellTypeInfo(4, "NewType", ...)
    2. Add it to _BY_VALUE in by_value()
    3. Update all_types() if needed
    """
    EMPTY = CellTypeInfo(0, "Empty")
    WALL = CellTypeInfo(1, "Wall")
    FOOD = CellTypeInfo(2, "Food", interestingness=1.0, energy_reward=15.0)
    POISON = CellTypeInfo(3, "Poison", interestingness=0.9, energy_penalty=50.0)
    
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
        """Get all registered cell types in value order."""
        if cls._ALL_TYPES is None:
            cls._ALL_TYPES = [cls.EMPTY, cls.WALL, cls.FOOD, cls.POISON]
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


def get_config():
    from .observation_spec import ObservationSpec
    
    config = {
        'ENERGY_START': ENERGY_START,
        'ENERGY_MOVE_COST': ENERGY_MOVE_COST,
        'MOVE_COST_EXPONENT': MOVE_COST_EXPONENT,
        'MAX_MOVE_DISTANCE': MAX_MOVE_DISTANCE,
        'GRID_WIDTH': GRID_WIDTH,
        'GRID_HEIGHT': GRID_HEIGHT,
        'GRID_FOOD_INIT': GRID_FOOD_INIT,
        'GRID_POISON_INIT': GRID_POISON_INIT,
        'AGENTS_SETUP': AGENTS_SETUP,
        'AGENT_VIEW_RANGE': AGENT_VIEW_RANGE,
        'CellType': CellType
    }
    
    # Central observation spec factory - derives from CellType and view range
    config['get_observation_spec'] = lambda mode: ObservationSpec.for_mode(mode, config)
    
    return config

