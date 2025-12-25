"""Tests for neural network architecture and behavior.

These tests verify that the CNN and related components work correctly:
- Shape handling for various input sizes
- Gradient flow (no dead neurons)
- Weight initialization
- Feature extraction and action head
"""
import pytest
import torch
import torch.nn as nn

from goodharts.behaviors.brains.base_cnn import BaseCNN
from goodharts.behaviors.action_space import num_actions
from goodharts.configs.default_config import get_config
from goodharts.modes import ObservationSpec


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def config():
    return get_config()


class TestCNNArchitecture:
    """Tests for BaseCNN architecture."""

    def test_cnn_accepts_various_input_sizes(self, device):
        """CNN should handle various spatial input sizes."""
        n_channels = 6
        n_actions = 8

        for size in [(5, 5), (11, 11), (21, 21)]:
            model = BaseCNN(
                input_shape=size,
                input_channels=n_channels,
                output_size=n_actions
            ).to(device)

            x = torch.randn(2, n_channels, *size, device=device)
            output = model(x)

            assert output.shape == (2, n_actions), \
                f"Wrong output shape for input size {size}: {output.shape}"

    def test_cnn_output_is_logits(self, device):
        """CNN output should be unnormalized logits."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        x = torch.randn(4, 6, 11, 11, device=device)
        output = model(x)

        # Logits should NOT sum to 1 (that would be probabilities)
        sums = output.sum(dim=1)
        assert not torch.allclose(sums, torch.ones_like(sums)), \
            "Output looks like probabilities, not logits"

    def test_cnn_handles_batch_size_one(self, device):
        """CNN should handle batch size of 1."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        x = torch.randn(1, 6, 11, 11, device=device)
        output = model(x)

        assert output.shape == (1, 8)


class TestGradientFlow:
    """Tests for gradient flow through the network."""

    def test_gradients_flow_to_all_layers(self, device):
        """Gradients should reach all trainable parameters."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        x = torch.randn(4, 6, 11, 11, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, \
                    f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, \
                    f"Zero gradient for {name} - potential dead layer"

    def test_no_nan_in_forward_pass(self, device):
        """Forward pass should not produce NaN values."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        # Test with various input distributions
        for _ in range(10):
            x = torch.randn(4, 6, 11, 11, device=device)
            output = model(x)

            assert not torch.isnan(output).any(), "NaN in output"
            assert not torch.isinf(output).any(), "Inf in output"

    def test_no_nan_in_backward_pass(self, device):
        """Backward pass should not produce NaN gradients."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        x = torch.randn(4, 6, 11, 11, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), \
                    f"NaN gradient in {name}"


class TestFeatureExtraction:
    """Tests for feature extraction interface."""

    def test_get_features_returns_tensor(self, device):
        """get_features should return a feature tensor."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        x = torch.randn(4, 6, 11, 11, device=device)
        features = model.get_features(x)

        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == 4  # Batch size preserved

    def test_logits_from_features_produces_actions(self, device):
        """logits_from_features should produce action logits."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        x = torch.randn(4, 6, 11, 11, device=device)
        features = model.get_features(x)
        logits = model.logits_from_features(features)

        assert logits.shape == (4, 8)

    def test_two_stage_equals_forward(self, device):
        """get_features + logits_from_features should equal forward."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        x = torch.randn(4, 6, 11, 11, device=device)

        # Single forward pass
        direct_output = model(x)

        # Two-stage
        features = model.get_features(x)
        staged_output = model.logits_from_features(features)

        assert torch.allclose(direct_output, staged_output, atol=1e-5), \
            "Two-stage output doesn't match direct forward"


class TestWeightInitialization:
    """Tests for weight initialization."""

    def test_weights_are_initialized(self, device):
        """Weights should not be all zeros after initialization."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        for name, param in model.named_parameters():
            if 'weight' in name:
                assert param.abs().sum() > 0, \
                    f"Weights are all zeros in {name}"

    def test_weights_are_finite(self, device):
        """All weights should be finite."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), \
                f"Non-finite values in {name}"


class TestActionSampling:
    """Tests for action sampling from the network."""

    def test_action_space_decodes_valid_movement(self, device):
        """ActionSpace should decode logits to valid (dx, dy) movement."""
        from goodharts.behaviors.action_space import DiscreteGridActionSpace

        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)
        model.eval()

        action_space = DiscreteGridActionSpace(max_move_distance=1)
        x = torch.randn(1, 6, 11, 11, device=device)

        with torch.no_grad():
            logits = model(x)
            dx, dy = action_space.decode(logits, sample=True)

        # dx and dy should be valid movement values
        assert isinstance(dx, int), f"dx should be int, got {type(dx)}"
        assert isinstance(dy, int), f"dy should be int, got {type(dy)}"
        assert -1 <= dx <= 1, f"dx {dx} out of valid range [-1, 1]"
        assert -1 <= dy <= 1, f"dy {dy} out of valid range [-1, 1]"

    def test_forward_returns_logits(self, device):
        """forward() should return valid action logits."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)
        model.eval()

        x = torch.randn(1, 6, 11, 11, device=device)

        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (1, 8), f"Expected (1, 8) logits, got {logits.shape}"
        # Logits can be any real value (not necessarily in [0, 1])
        assert not torch.isnan(logits).any(), "Logits should not contain NaN"


class TestWithRealConfig:
    """Tests using actual project configuration."""

    def test_cnn_with_ground_truth_spec(self, config, device):
        """CNN should work with ground truth observation spec."""
        spec = ObservationSpec.for_mode('ground_truth', config)

        model = BaseCNN(
            input_shape=spec.input_shape,
            input_channels=spec.num_channels,
            output_size=num_actions(1)
        ).to(device)

        x = torch.randn(4, spec.num_channels, *spec.input_shape, device=device)
        output = model(x)

        assert output.shape == (4, 8)

    def test_cnn_with_proxy_spec(self, config, device):
        """CNN should work with proxy observation spec."""
        spec = ObservationSpec.for_mode('proxy', config)

        model = BaseCNN(
            input_shape=spec.input_shape,
            input_channels=spec.num_channels,
            output_size=num_actions(1)
        ).to(device)

        x = torch.randn(4, spec.num_channels, *spec.input_shape, device=device)
        output = model(x)

        assert output.shape == (4, 8)


class TestDevicePlacement:
    """Tests for correct device placement."""

    def test_model_on_specified_device(self, device):
        """Model parameters should be on specified device."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        for name, param in model.named_parameters():
            # Compare device type (cuda vs cpu), not exact device index
            assert param.device.type == device.type, \
                f"Parameter {name} on wrong device type: {param.device.type} != {device.type}"

    def test_output_on_same_device_as_input(self, device):
        """Output should be on same device as input."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        x = torch.randn(4, 6, 11, 11, device=device)
        output = model(x)

        # Compare device type (cuda vs cpu)
        assert output.device.type == device.type, \
            f"Output on wrong device type: {output.device.type} != {device.type}"
