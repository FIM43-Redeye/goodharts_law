"""Tests for behavior implementations."""
import pytest
import torch
from goodharts.configs.default_config import get_simulation_config


@pytest.fixture
def config():
    return get_simulation_config()



class MockAgent:
    def __init__(self, x=0, y=0, sight_radius=5):
        self.x = x
        self.y = y
        self.sight_radius = sight_radius


class TestOmniscientSeeker:
    """Tests for OmniscientSeeker behavior.

    Uses 2-channel encoding: channel 0 = food, channel 1 = poison.
    """

    def test_moves_toward_food(self, config):
        """OmniscientSeeker should move toward visible food."""
        from goodharts.behaviors import OmniscientSeeker

        behavior = OmniscientSeeker()
        agent = MockAgent(sight_radius=5)

        # 2-channel view: channel 0 = food, channel 1 = poison
        r = 5
        size = 2 * r + 1
        view = torch.zeros((2, size, size), dtype=torch.float32)

        # Place food to the right of center (channel 0)
        view[0, r, r + 2] = 1.0

        dx, dy = behavior.decide_action(agent, view)

        # Should move toward food (positive x direction)
        assert dx == 1 and dy == 0

    def test_avoids_poison(self, config):
        """OmniscientSeeker should not move toward poison."""
        from goodharts.behaviors import OmniscientSeeker

        behavior = OmniscientSeeker()
        agent = MockAgent(sight_radius=5)

        # 2-channel view: channel 0 = food, channel 1 = poison
        r = 5
        size = 2 * r + 1
        view = torch.zeros((2, size, size), dtype=torch.float32)

        # Place poison directly to the right (channel 1)
        view[1, r, r + 1] = 1.0

        dx, dy = behavior.decide_action(agent, view)

        # Should not move right into poison
        assert not (dx == 1 and dy == 0)


class TestProxySeeker:
    """Tests for ProxySeeker behavior.

    Uses 2-channel encoding with identical interestingness in both channels.
    """

    def test_moves_toward_interestingness(self, config):
        """ProxySeeker should move toward high interestingness."""
        from goodharts.behaviors import ProxySeeker

        behavior = ProxySeeker()
        agent = MockAgent(sight_radius=5)

        # 2-channel view with identical interestingness
        r = 5
        size = 2 * r + 1
        view = torch.zeros((2, size, size), dtype=torch.float32)

        # High interestingness to the right (same value in both channels)
        view[0, r, r + 2] = 1.0
        view[1, r, r + 2] = 1.0

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
