"""Tests for the ActionSpace abstraction.

These tests verify:
- Action space creation via registry
- Serialization/deserialization roundtrip
- Correct action decoding for each implementation
- Batch decoding
"""
import pytest
import torch

from goodharts.behaviors.action_space import (
    ActionSpace,
    DiscreteGridActionSpace,
    ContinuousActionSpace,
    FactoredActionSpace,
    create_action_space,
    load_action_space,
    get_action_space_types,
)


class TestActionSpaceRegistry:
    """Tests for action space registry and factory."""

    def test_get_action_space_types(self):
        """Registry should list all available types."""
        types = get_action_space_types()
        assert 'discrete_grid' in types
        assert 'continuous' in types
        assert 'factored' in types

    def test_create_discrete_grid(self):
        """Factory should create discrete grid action space."""
        space = create_action_space('discrete_grid', max_move_distance=1)
        assert isinstance(space, DiscreteGridActionSpace)
        assert space.n_outputs == 8
        assert space.output_mode == 'discrete'

    def test_create_continuous(self):
        """Factory should create continuous action space."""
        space = create_action_space('continuous', max_move_distance=2)
        assert isinstance(space, ContinuousActionSpace)
        assert space.n_outputs == 2
        assert space.output_mode == 'continuous'

    def test_create_factored(self):
        """Factory should create factored action space."""
        space = create_action_space('factored', max_move_distance=3)
        assert isinstance(space, FactoredActionSpace)
        assert space.n_outputs == 8 + 3  # 8 directions + 3 magnitudes
        assert space.output_mode == 'discrete'

    def test_create_unknown_raises(self):
        """Factory should raise on unknown type."""
        with pytest.raises(ValueError, match="Unknown action space"):
            create_action_space('unknown_type')


class TestActionSpaceSerialization:
    """Tests for action space serialization roundtrip."""

    @pytest.mark.parametrize("space_type,max_dist", [
        ('discrete_grid', 1),
        ('discrete_grid', 2),
        ('continuous', 1),
        ('continuous', 3),
        ('factored', 2),
    ])
    def test_serialization_roundtrip(self, space_type, max_dist):
        """Action space should survive serialization roundtrip."""
        original = create_action_space(space_type, max_move_distance=max_dist)
        config = original.get_config()
        restored = load_action_space(config)

        assert type(restored) == type(original)
        assert restored.n_outputs == original.n_outputs
        assert restored.output_mode == original.output_mode
        assert restored.max_move_distance == original.max_move_distance

    def test_config_contains_type(self):
        """Config should contain type key."""
        space = create_action_space('discrete_grid')
        config = space.get_config()
        assert 'type' in config
        assert config['type'] == 'discrete_grid'


class TestDiscreteGridActionSpace:
    """Tests for DiscreteGridActionSpace implementation."""

    def test_n_outputs_for_distance_1(self):
        """Distance 1 should give 8 actions."""
        space = DiscreteGridActionSpace(max_move_distance=1)
        assert space.n_outputs == 8

    def test_n_outputs_for_distance_2(self):
        """Distance 2 should give 24 actions (5x5 - 1)."""
        space = DiscreteGridActionSpace(max_move_distance=2)
        assert space.n_outputs == 24

    def test_decode_returns_valid_movement(self):
        """Decode should return (dx, dy) within range."""
        space = DiscreteGridActionSpace(max_move_distance=1)
        logits = torch.randn(1, 8)

        for _ in range(10):
            dx, dy = space.decode(logits, sample=True)
            assert -1 <= dx <= 1
            assert -1 <= dy <= 1
            assert not (dx == 0 and dy == 0), "Should not stay in place"

    def test_decode_deterministic_returns_argmax(self):
        """Decode with sample=False should return argmax action."""
        space = DiscreteGridActionSpace(max_move_distance=1)

        # Make one action clearly dominant
        logits = torch.zeros(1, 8)
        logits[0, 3] = 10.0  # Strong preference for action 3

        dx, dy = space.decode(logits, sample=False)
        expected = space.index_to_action(3)
        assert (dx, dy) == expected

    def test_decode_batch(self):
        """Batch decode should work correctly."""
        space = DiscreteGridActionSpace(max_move_distance=1)
        batch_size = 16
        logits = torch.randn(batch_size, 8)

        dx, dy = space.decode_batch(logits, sample=True)

        assert dx.shape == (batch_size,)
        assert dy.shape == (batch_size,)
        assert ((dx >= -1) & (dx <= 1)).all()
        assert ((dy >= -1) & (dy <= 1)).all()

    def test_action_to_index_roundtrip(self):
        """action_to_index and index_to_action should be inverses."""
        space = DiscreteGridActionSpace(max_move_distance=1)

        for idx in range(space.n_outputs):
            dx, dy = space.index_to_action(idx)
            recovered_idx = space.action_to_index(dx, dy)
            assert recovered_idx == idx


class TestContinuousActionSpace:
    """Tests for ContinuousActionSpace implementation."""

    def test_n_outputs_is_2(self):
        """Continuous space always outputs 2 values."""
        for dist in [1, 2, 3]:
            space = ContinuousActionSpace(max_move_distance=dist)
            assert space.n_outputs == 2

    def test_decode_scales_by_distance(self):
        """Decode should scale by max_move_distance."""
        space = ContinuousActionSpace(max_move_distance=3)

        # Output in range [-1, 1] should scale to [-3, 3]
        output = torch.tensor([[1.0, -1.0]])
        dx, dy = space.decode(output)
        assert dx == 3
        assert dy == -3

    def test_decode_clamps_output(self):
        """Decode should clamp to valid range."""
        space = ContinuousActionSpace(max_move_distance=2)

        # Extreme values should be clamped
        output = torch.tensor([[5.0, -5.0]])
        dx, dy = space.decode(output)
        assert dx == 2
        assert dy == -2

    def test_decode_batch(self):
        """Batch decode should work correctly."""
        space = ContinuousActionSpace(max_move_distance=2)
        batch_size = 16
        outputs = torch.randn(batch_size, 2)

        dx, dy = space.decode_batch(outputs)

        assert dx.shape == (batch_size,)
        assert dy.shape == (batch_size,)
        assert ((dx >= -2) & (dx <= 2)).all()
        assert ((dy >= -2) & (dy <= 2)).all()


class TestFactoredActionSpace:
    """Tests for FactoredActionSpace implementation."""

    def test_n_outputs_is_8_plus_magnitude(self):
        """Factored space outputs 8 + max_move_distance."""
        for dist in [1, 2, 3]:
            space = FactoredActionSpace(max_move_distance=dist)
            assert space.n_outputs == 8 + dist

    def test_decode_returns_valid_movement(self):
        """Decode should return valid (dx, dy)."""
        space = FactoredActionSpace(max_move_distance=3)
        logits = torch.randn(1, space.n_outputs)

        for _ in range(10):
            dx, dy = space.decode(logits, sample=True)
            assert -3 <= dx <= 3
            assert -3 <= dy <= 3
            # At least one of dx, dy must be non-zero
            assert dx != 0 or dy != 0

    def test_decode_respects_direction_and_magnitude(self):
        """Strong preferences should be respected."""
        space = FactoredActionSpace(max_move_distance=2)

        # Force direction 0 (-1, -1) and magnitude 1 (index 0 = mag 1)
        logits = torch.zeros(1, 10)  # 8 + 2
        logits[0, 0] = 10.0  # Direction 0: (-1, -1)
        logits[0, 8] = 10.0  # Magnitude 0: 1

        dx, dy = space.decode(logits, sample=False)
        assert dx == -1
        assert dy == -1

    def test_decode_batch(self):
        """Batch decode should work correctly."""
        space = FactoredActionSpace(max_move_distance=2)
        batch_size = 16
        logits = torch.randn(batch_size, space.n_outputs)

        dx, dy = space.decode_batch(logits, sample=True)

        assert dx.shape == (batch_size,)
        assert dy.shape == (batch_size,)


class TestActionSpaceWithBrain:
    """Integration tests with neural network."""

    def test_brain_output_matches_action_space(self):
        """Brain output size should match action space n_outputs."""
        from goodharts.behaviors.brains.base_cnn import BaseCNN

        for space_type in ['discrete_grid', 'continuous', 'factored']:
            space = create_action_space(space_type, max_move_distance=1)

            brain = BaseCNN(
                input_shape=(11, 11),
                input_channels=6,
                output_size=space.n_outputs,
                action_mode=space.output_mode,
            )

            x = torch.randn(4, 6, 11, 11)
            output = brain(x)

            assert output.shape == (4, space.n_outputs)

    def test_decode_brain_output(self):
        """Should be able to decode brain output directly."""
        from goodharts.behaviors.brains.base_cnn import BaseCNN

        space = create_action_space('discrete_grid', max_move_distance=1)

        brain = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=space.n_outputs,
        )
        brain.eval()

        x = torch.randn(1, 6, 11, 11)
        with torch.no_grad():
            logits = brain(x)
            dx, dy = space.decode(logits)

        assert isinstance(dx, int)
        assert isinstance(dy, int)
        assert -1 <= dx <= 1
        assert -1 <= dy <= 1
