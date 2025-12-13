import pytest
import numpy as np
from environments.world import World
from configs.default_config import get_config

@pytest.fixture
def config():
    return get_config()

def test_world_init(config):
    w = World(20, 15, config)
    assert w.width == 20
    assert w.height == 15
    assert w.grid.shape == (15, 20)

def test_place_food(config):
    w = World(10, 10, config)
    w.place_food(5)
    
    food_count = np.sum(w.grid == config['CellType'].FOOD)
    assert food_count == 5

def test_proxy_grid_sync(config):
    w = World(10, 10, config)
    w.place_food(1)
    
    # Find where food is
    y, x = np.where(w.grid == config['CellType'].FOOD)
    
    # Check proxy grid has signal there
    assert w.proxy_grid[y[0], x[0]] > 0
