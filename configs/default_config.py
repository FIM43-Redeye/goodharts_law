from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CellTypeInfo:
    """
    Intrinsic properties of a cell type.
    
    Supports direct comparison with integers/numpy arrays:
        grid[y, x] == CellType.FOOD  # Just works!
    """
    value: int
    name: str
    interestingness: float = 0.0
    energy_reward: float = 0.0
    energy_penalty: float = 0.0
    
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


class CellType:
    """Registry of all cell types with their intrinsic properties."""
    EMPTY = CellTypeInfo(0, "Empty")
    WALL = CellTypeInfo(1, "Wall")
    FOOD = CellTypeInfo(2, "Food", interestingness=1.0, energy_reward=15.0)
    POISON = CellTypeInfo(3, "Poison", interestingness=0.9, energy_penalty=50.0)
    
    _BY_VALUE = None  # Lazy-loaded lookup table
    
    @classmethod
    def by_value(cls, value: int) -> CellTypeInfo | None:
        """Lookup cell type by its integer value."""
        if cls._BY_VALUE is None:
            cls._BY_VALUE = {
                0: cls.EMPTY,
                1: cls.WALL, 
                2: cls.FOOD,
                3: cls.POISON,
            }
        return cls._BY_VALUE.get(int(value))

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
    return {
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
