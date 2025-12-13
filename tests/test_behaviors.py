"""Tests for behavior implementations."""
import pytest
import numpy as np
from goodharts.configs.default_config import get_config
from goodharts.environments.world import World
from goodharts.agents.organism import Organism


@pytest.fixture
def config():
    return get_config()


@pytest.fixture
def small_world(config):
    """Create a small 10x10 world."""
    return World(10, 10, config)


class TestOmniscientSeeker:
    """Tests for OmniscientSeeker behavior."""
    
    def test_moves_toward_food(self, small_world, config):
        """OmniscientSeeker should move toward visible food."""
        from goodharts.behaviors import OmniscientSeeker
        
        # Place food at (5, 5)
        small_world.grid[5, 5] = config['CellType'].FOOD
        
        # Agent at (3, 5) should move right toward food
        behavior = OmniscientSeeker()
        agent = Organism(3, 5, 100, 5, small_world, behavior, config)
        
        view = agent.get_local_view()
        dx, dy = behavior.decide_action(agent, view)
        
        # Should move toward food (positive x direction)
        assert dx > 0 or dy != 0  # Moving toward food
    
    def test_avoids_poison(self, small_world, config):
        """OmniscientSeeker should not move toward poison."""
        from goodharts.behaviors import OmniscientSeeker
        
        # Only poison visible, no food
        small_world.grid[5, 5] = config['CellType'].POISON
        
        behavior = OmniscientSeeker()
        agent = Organism(4, 5, 100, 5, small_world, behavior, config)
        
        view = agent.get_local_view()
        dx, dy = behavior.decide_action(agent, view)
        
        # Should not move toward poison (would be dx=1, dy=0)
        # Any other move is acceptable
        assert not (dx == 1 and dy == 0)


class TestProxySeeker:
    """Tests for ProxySeeker behavior."""
    
    def test_moves_toward_interestingness(self, small_world, config):
        """ProxySeeker should move toward high interestingness."""
        from goodharts.behaviors import ProxySeeker
        
        # Place something interesting (both food and poison have interestingness)
        small_world.grid[5, 5] = config['CellType'].FOOD
        small_world.proxy_grid[5, 5] = 1.0  # High interestingness
        
        behavior = ProxySeeker()
        agent = Organism(3, 5, 100, 5, small_world, behavior, config)
        
        view = agent.get_local_view()
        dx, dy = behavior.decide_action(agent, view)
        
        # Should move toward the interesting thing
        assert dx > 0 or dy != 0


class TestActionSpace:
    """Tests for action space utilities."""
    
    def test_build_action_space(self):
        """build_action_space should return 8 actions for distance 1."""
        from goodharts.behaviors.action_space import build_action_space
        
        actions = build_action_space(1)
        assert len(actions) == 8  # 8 directional (no stay)
        
        # Check all actions are unique
        assert len(set(actions)) == 8
        
        # No (0,0) - no stay action
        assert (0, 0) not in actions
    
    def test_action_index_roundtrip(self):
        """action_to_index and index_to_action should be inverses."""
        from goodharts.behaviors.action_space import (
            action_to_index,
            index_to_action,
            build_action_space
        )
        
        actions = build_action_space(1)
        
        for idx, action in enumerate(actions):
            # index -> action -> index
            recovered_action = index_to_action(idx)
            assert recovered_action == action
            
            # action -> index -> action
            dx, dy = action
            recovered_idx = action_to_index(dx, dy)
            assert recovered_idx == idx
