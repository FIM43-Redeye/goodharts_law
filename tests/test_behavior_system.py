"""Tests for behavior system - registry, discovery, and decision making.

These tests verify that the behavior system components work correctly:
- Auto-discovery of behavior classes
- Registry lookup and instantiation
- Action space conversions
- Behavior decision interface
"""
import pytest
import torch

from goodharts.behaviors.base import BehaviorStrategy
from goodharts.behaviors.registry import (
    get_behavior,
    get_all_behaviors,
    list_behavior_names,
)
from goodharts.behaviors.action_space import (
    build_action_space,
    action_to_index,
    index_to_action,
    num_actions,
)
from goodharts.configs.default_config import CellType

# Note: config fixture is provided by conftest.py


class TestBehaviorRegistry:
    """Tests for behavior auto-discovery and registry."""

    def test_registry_discovers_omniscient_seeker(self):
        """OmniscientSeeker should be auto-discovered."""
        behaviors = list_behavior_names()
        assert 'OmniscientSeeker' in behaviors, \
            f"OmniscientSeeker not found. Available: {behaviors}"

    def test_registry_discovers_proxy_seeker(self):
        """ProxySeeker should be auto-discovered."""
        behaviors = list_behavior_names()
        assert 'ProxySeeker' in behaviors, \
            f"ProxySeeker not found. Available: {behaviors}"

    def test_registry_discovers_learned_behavior(self):
        """LearnedBehavior should be auto-discovered."""
        behaviors = list_behavior_names()
        assert 'LearnedBehavior' in behaviors, \
            f"LearnedBehavior not found. Available: {behaviors}"

    def test_get_behavior_returns_class(self):
        """get_behavior should return a class, not an instance."""
        cls = get_behavior('OmniscientSeeker')
        assert isinstance(cls, type), "get_behavior should return a class"
        assert issubclass(cls, BehaviorStrategy), \
            "Returned class should be a BehaviorStrategy subclass"

    def test_get_behavior_invalid_raises(self):
        """get_behavior with invalid name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown behavior"):
            get_behavior('NonExistentBehavior')

    def test_get_all_behaviors_returns_dict(self):
        """get_all_behaviors should return a dict of name -> class."""
        behaviors = get_all_behaviors()
        assert isinstance(behaviors, dict)
        assert len(behaviors) >= 3  # At least OmniscientSeeker, ProxySeeker, LearnedBehavior

        for name, cls in behaviors.items():
            assert isinstance(name, str)
            assert isinstance(cls, type)
            assert issubclass(cls, BehaviorStrategy)

    def test_behavior_instantiation(self):
        """Discovered behaviors should be instantiable."""
        from goodharts.behaviors import OmniscientSeeker, ProxySeeker

        omni = OmniscientSeeker()
        proxy = ProxySeeker()

        assert isinstance(omni, BehaviorStrategy)
        assert isinstance(proxy, BehaviorStrategy)


class TestBehaviorInterface:
    """Tests for BehaviorStrategy interface."""

    def test_behavior_has_requirements(self):
        """All behaviors should declare their requirements."""
        from goodharts.behaviors import OmniscientSeeker, ProxySeeker

        omni = OmniscientSeeker()
        proxy = ProxySeeker()

        assert hasattr(omni, 'requirements')
        assert hasattr(proxy, 'requirements')
        assert isinstance(omni.requirements, list)
        assert isinstance(proxy.requirements, list)

    def test_behavior_has_color(self):
        """All behaviors should have a color property."""
        from goodharts.behaviors import OmniscientSeeker, ProxySeeker

        omni = OmniscientSeeker()
        proxy = ProxySeeker()

        assert hasattr(omni, 'color')
        assert hasattr(proxy, 'color')

        # Color should be RGB tuple
        assert len(omni.color) == 3
        assert len(proxy.color) == 3
        assert all(0 <= c <= 255 for c in omni.color)
        assert all(0 <= c <= 255 for c in proxy.color)

    def test_behavior_has_decide_action(self):
        """All behaviors should have decide_action method."""
        from goodharts.behaviors import OmniscientSeeker, ProxySeeker

        omni = OmniscientSeeker()
        proxy = ProxySeeker()

        assert hasattr(omni, 'decide_action')
        assert hasattr(proxy, 'decide_action')
        assert callable(omni.decide_action)
        assert callable(proxy.decide_action)


class TestActionSpace:
    """Tests for action space utilities."""

    def test_action_space_size(self):
        """Action space should have 8 directions for distance 1."""
        actions = build_action_space(1)
        assert len(actions) == 8, f"Expected 8 actions, got {len(actions)}"

    def test_num_actions_returns_correct_count(self):
        """num_actions should return action count."""
        n = num_actions(1)
        assert n == 8

    def test_action_space_no_stay(self):
        """Action space should not include (0, 0) stay action."""
        actions = build_action_space(1)
        assert (0, 0) not in actions, "Stay action (0,0) should not be included"

    def test_action_space_covers_directions(self):
        """Action space should cover all 8 cardinal and diagonal directions."""
        actions = build_action_space(1)

        expected_directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        for direction in expected_directions:
            assert direction in actions, f"Missing direction: {direction}"

    def test_action_index_roundtrip(self):
        """action_to_index and index_to_action should be inverses."""
        for idx in range(8):
            dx, dy = index_to_action(idx)
            recovered_idx = action_to_index(dx, dy)
            assert recovered_idx == idx, \
                f"Roundtrip failed: {idx} -> ({dx}, {dy}) -> {recovered_idx}"

    def test_index_to_action_returns_tuple(self):
        """index_to_action should return (dx, dy) tuple."""
        for idx in range(8):
            result = index_to_action(idx)
            assert isinstance(result, tuple)
            assert len(result) == 2
            dx, dy = result
            assert -1 <= dx <= 1
            assert -1 <= dy <= 1


class TestOmniscientSeekerBehavior:
    """Tests for OmniscientSeeker decision making."""

    def test_moves_toward_visible_food(self, config):
        """OmniscientSeeker should move toward food when visible."""
        from goodharts.behaviors import OmniscientSeeker

        behavior = OmniscientSeeker()

        # Create mock 2-channel view with food to the right
        # Channel 0 = food, Channel 1 = poison
        r = 5
        size = 2 * r + 1
        view = torch.zeros((2, size, size), dtype=torch.float32)

        # Place food 2 cells to the right of center (channel 0)
        view[0, r, r + 2] = 1.0

        class MockAgent:
            x = 0
            y = 0
            sight_radius = 5

        dx, dy = behavior.decide_action(MockAgent(), view)

        # Should move right toward food
        assert dx == 1, f"Should move right (dx=1), got dx={dx}"
        assert dy == 0, f"Should not move vertically, got dy={dy}"

    def test_avoids_poison_when_no_food(self, config):
        """OmniscientSeeker should not move directly toward poison."""
        from goodharts.behaviors import OmniscientSeeker

        behavior = OmniscientSeeker()

        # 2-channel view: Channel 0 = food, Channel 1 = poison
        r = 5
        size = 2 * r + 1
        view = torch.zeros((2, size, size), dtype=torch.float32)

        # Place poison 1 cell to the right (channel 1)
        view[1, r, r + 1] = 1.0

        class MockAgent:
            x = 0
            y = 0
            sight_radius = 5

        dx, dy = behavior.decide_action(MockAgent(), view)

        # Should NOT move right into poison
        assert not (dx == 1 and dy == 0), \
            f"Should not move into poison, got ({dx}, {dy})"

    def test_prefers_food_over_poison(self, config):
        """OmniscientSeeker should prefer food when both are visible."""
        from goodharts.behaviors import OmniscientSeeker

        behavior = OmniscientSeeker()

        # 2-channel view: Channel 0 = food, Channel 1 = poison
        r = 5
        size = 2 * r + 1
        view = torch.zeros((2, size, size), dtype=torch.float32)

        # Food to the right (channel 0), poison to the left (channel 1)
        view[0, r, r + 2] = 1.0
        view[1, r, r - 2] = 1.0

        class MockAgent:
            x = 0
            y = 0
            sight_radius = 5

        dx, dy = behavior.decide_action(MockAgent(), view)

        # Should move toward food (right), not poison (left)
        assert dx >= 0, f"Should move toward food (right), got dx={dx}"


class TestProxySeekerBehavior:
    """Tests for ProxySeeker decision making."""

    def test_moves_toward_high_interestingness(self, config):
        """ProxySeeker should move toward high interestingness."""
        from goodharts.behaviors import ProxySeeker

        behavior = ProxySeeker()

        # 2-channel view with identical interestingness values in both channels
        r = 5
        size = 2 * r + 1
        view = torch.zeros((2, size, size), dtype=torch.float32)

        # High interestingness to the right (both channels same value)
        view[0, r, r + 2] = 1.0
        view[1, r, r + 2] = 1.0

        class MockAgent:
            x = 0
            y = 0
            sight_radius = 5

        dx, dy = behavior.decide_action(MockAgent(), view)

        # Should move toward high interestingness
        assert dx == 1, f"Should move right toward interestingness, got dx={dx}"

    def test_cannot_distinguish_food_from_poison(self, config):
        """ProxySeeker requirements should not include ground truth."""
        from goodharts.behaviors import ProxySeeker

        behavior = ProxySeeker()

        # Should require proxy metric, not ground truth
        assert 'proxy_metric' in behavior.requirements or 'interestingness' in behavior.requirements
        assert 'ground_truth' not in behavior.requirements


class TestLearnedBehavior:
    """Tests for LearnedBehavior system."""

    def test_learned_behavior_presets_exist(self):
        """Learned behavior presets should be registered."""
        from goodharts.behaviors.learned import create_learned_behavior, LEARNED_PRESETS

        # Should be able to create all registered presets without errors
        for preset in LEARNED_PRESETS.keys():
            behavior = create_learned_behavior(preset)
            assert behavior is not None
            assert isinstance(behavior, BehaviorStrategy)

    def test_learned_behavior_has_brain(self):
        """LearnedBehavior should have a brain attribute after initialization."""
        from goodharts.behaviors.learned import create_learned_behavior

        behavior = create_learned_behavior('ground_truth')

        # Brain is lazily initialized, so we need to trigger it
        # This might require a mock observation
        assert hasattr(behavior, '_brain') or hasattr(behavior, 'brain')
