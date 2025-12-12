from enum import IntEnum

# The "Vocabulary" of the grid
# IntEnum allows these to be used directly in numpy arrays
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