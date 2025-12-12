from enum import IntEnum

class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    FOOD = 2
    POISON = 3

# Simulation Physics / Hyperparameters
ENERGY_START = 50.0
ENERGY_MOVE_COST = 0.1
ENERGY_FOOD_REWARD = 15.0
ENERGY_POISON_PENALTY = 50.0

# Grid Settings
GRID_WIDTH = 100
GRID_HEIGHT = 100
GRID_FOOD_INIT = 50
GRID_POISON_INIT = 10
GRID_AGENTS_INIT = 5

# Agent Properties
AGENT_VIEW_RANGE = 5

CELL_PROPERTIES = {
    CellType.FOOD: {'energy_reward': ENERGY_FOOD_REWARD},
    CellType.POISON: {'energy_penalty': ENERGY_POISON_PENALTY},
}

def get_config():
    return {
        'ENERGY_START': ENERGY_START,
        'ENERGY_MOVE_COST': ENERGY_MOVE_COST,
        'GRID_WIDTH': GRID_WIDTH,
        'GRID_HEIGHT': GRID_HEIGHT,
        'GRID_FOOD_INIT': GRID_FOOD_INIT,
        'GRID_POISON_INIT': GRID_POISON_INIT,
        'GRID_AGENTS_INIT': GRID_AGENTS_INIT,
        'AGENT_VIEW_RANGE': AGENT_VIEW_RANGE,
        'CELL_PROPERTIES': CELL_PROPERTIES,
        'CellType': CellType
    }
