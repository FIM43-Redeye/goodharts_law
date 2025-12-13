import pytest
import numpy as np
from goodharts.environments.world import World
from goodharts.configs.default_config import get_config

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


class TestLoopMode:
    """Tests for looping (toroidal) world mode."""
    
    @pytest.fixture
    def loop_config(self, config):
        """Config with loop mode enabled."""
        config['WORLD_LOOP'] = True
        return config
    
    def test_world_loop_flag(self, loop_config):
        """World should have loop flag set from config."""
        w = World(10, 10, loop_config)
        assert w.loop is True
    
    def test_wrap_position_bounded(self, config):
        """Bounded world should clamp coordinates."""
        config['WORLD_LOOP'] = False
        w = World(10, 10, config)
        
        # At edge, moving right should clamp
        x, y = w.wrap_position(15, 5)
        assert x == 9  # Clamped to width-1
        assert y == 5
        
        # Negative should clamp to 0
        x, y = w.wrap_position(-5, -3)
        assert x == 0
        assert y == 0
    
    def test_wrap_position_looped(self, loop_config):
        """Looping world should wrap coordinates."""
        w = World(10, 10, loop_config)
        
        # Past edge should wrap
        x, y = w.wrap_position(15, 5)
        assert x == 5  # 15 % 10 = 5
        assert y == 5
        
        # Negative should wrap
        x, y = w.wrap_position(-3, 5)
        assert x == 7  # -3 % 10 = 7
    
    def test_get_view_bounded_pads(self, config):
        """Bounded world should pad view at edges."""
        config['WORLD_LOOP'] = False
        w = World(10, 10, config)
        w.grid[0, 0] = 1  # Mark a cell
        
        # View at corner with radius 2 should be 5x5
        view = w.get_view(w.grid, 0, 0, 2, fill_value=99)
        
        assert view.shape == (5, 5)
        # Top-left should be padded
        assert view[0, 0] == 99
        assert view[1, 1] == 99
        # Center should be the actual cell
        assert view[2, 2] == 1
    
    def test_get_view_looped_wraps(self, loop_config):
        """Looping world should wrap view at edges."""
        w = World(10, 10, loop_config)
        w.grid[0, 0] = 1  # Mark origin
        w.grid[9, 9] = 2  # Mark opposite corner
        
        # View at origin should see wrapped cells
        view = w.get_view(w.grid, 0, 0, 2)
        
        assert view.shape == (5, 5)
        # Center should be origin
        assert view[2, 2] == 1
        # Top-left should wrap to (8,8), (8,9), (9,8), (9,9)
        # (9,9) is at relative (-1,-1) = view[1,1]
        assert view[1, 1] == 2

