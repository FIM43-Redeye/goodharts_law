"""Tests for behavior implementations."""
import pytest
import numpy as np
from goodharts.configs.default_config import get_config


@pytest.fixture
def config():
    return get_config()



class MockAgent:
    def __init__(self, x=0, y=0, sight_radius=5):
        self.x = x
        self.y = y
        self.sight_radius = sight_radius


class TestOmniscientSeeker:
    """Tests for OmniscientSeeker behavior."""
    
    def test_moves_toward_food(self, config):
        """OmniscientSeeker should move toward visible food."""
        from goodharts.behaviors import OmniscientSeeker
        from goodharts.configs.default_config import CellType
        
        behavior = OmniscientSeeker()
        agent = MockAgent(sight_radius=5)
        
        # Create a view (num_channels, 2r+1, 2r+1)
        # Channels: empty, wall, food, poison... (depends on implementation)
        # OmniscientSeeker expects channels mapped by CellType indices
        # We need to ensure we construct a view valid for the logic
        
        r = 5
        size = 2*r + 1
        num_channels = len(CellType.all_types()) # Roughly
        # Actually OmniscientSeeker uses view[CellType.FOOD.channel_index]
        # We must ensure the view has enough channels
        
        max_channel = max(ct.channel_index for ct in CellType.all_types())
        view = np.zeros((max_channel + 1, size, size), dtype=np.float32)
        
        # Place food relative to center (r, r)
        # Agent at (3, 5), Food at (5, 5) -> dx=2, dy=0
        # In local view: center is (r, r). Target is at (r, r+2)
        view[CellType.FOOD.channel_index, r, r+2] = 1.0
        
        dx, dy = behavior.decide_action(agent, view)
        
        # Should move toward food (positive x direction)
        assert dx == 1 and dy == 0
    
    def test_avoids_poison(self, config):
        """OmniscientSeeker should not move toward poison."""
        from goodharts.behaviors import OmniscientSeeker
        from goodharts.configs.default_config import CellType
        
        behavior = OmniscientSeeker()
        agent = MockAgent(sight_radius=5)
        
        r = 5
        size = 2*r + 1
        max_channel = max(ct.channel_index for ct in CellType.all_types())
        view = np.zeros((max_channel + 1, size, size), dtype=np.float32)
        
        # Place poison directly to the right (r, r+1)
        view[CellType.POISON.channel_index, r, r+1] = 1.0
        
        dx, dy = behavior.decide_action(agent, view)
        
        # Should not move right
        assert not (dx == 1 and dy == 0)


class TestProxySeeker:
    """Tests for ProxySeeker behavior."""
    
    def test_moves_toward_interestingness(self, config):
        """ProxySeeker should move toward high interestingness."""
        from goodharts.behaviors import ProxySeeker
        
        behavior = ProxySeeker()
        agent = MockAgent(sight_radius=5)
        
        # ProxySeeker expects a view with 'interestingness' channel?
        # Actually ProxySeeker requirements=['proxy_metric']
        # which usually maps to a specific index or channel.
        # But wait, ProxySeeker logic is: view[0] is the proxy metric (usually)
        # Let's check ProxySeeker implementation? 
        # Requirement is 'proxy_metric'. 
        # Standard implementation expects 1 channel (if requirements=['proxy_metric'])
        
        r = 5
        size = 2*r + 1
        view = np.zeros((1, size, size), dtype=np.float32)
        
        # High interestingness to the right
        view[0, r, r+2] = 1.0
        
        dx, dy = behavior.decide_action(agent, view)
        
        # Should move toward it
        assert dx == 1 and dy == 0


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
