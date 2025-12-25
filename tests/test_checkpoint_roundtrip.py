"""Tests for model checkpoint saving and loading.

These tests verify that:
- Models can be saved and loaded correctly
- Loaded models produce same outputs as original
- Checkpoint format is stable
"""
import pytest
import torch
import tempfile
import os

from goodharts.behaviors.brains.base_cnn import BaseCNN
from goodharts.modes import ObservationSpec
from goodharts.configs.default_config import get_config


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def config():
    return get_config()


@pytest.fixture
def temp_model_path():
    """Create a temporary file path for model saving."""
    fd, path = tempfile.mkstemp(suffix='.pth')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


class TestModelSaveLoad:
    """Tests for saving and loading individual models."""

    def test_save_and_load_state_dict(self, device, temp_model_path):
        """Saving and loading state_dict should preserve weights."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        # Save
        torch.save(model.state_dict(), temp_model_path)

        # Load into new model
        model2 = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)
        model2.load_state_dict(torch.load(temp_model_path, map_location=device))

        # Compare weights
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.equal(p1, p2), f"Parameter {n1} differs after load"

    def test_loaded_model_same_output(self, device, temp_model_path):
        """Loaded model should produce same outputs as original."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)
        model.eval()

        # Test input
        x = torch.randn(4, 6, 11, 11, device=device)

        # Original output
        with torch.no_grad():
            orig_output = model(x)

        # Save and load
        torch.save(model.state_dict(), temp_model_path)

        model2 = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)
        model2.load_state_dict(torch.load(temp_model_path, map_location=device))
        model2.eval()

        # Loaded output
        with torch.no_grad():
            loaded_output = model2(x)

        assert torch.equal(orig_output, loaded_output), \
            "Loaded model produces different output"

    def test_file_exists_after_save(self, device, temp_model_path):
        """Saved file should exist and have content."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        torch.save(model.state_dict(), temp_model_path)

        assert os.path.exists(temp_model_path), "Model file not created"
        assert os.path.getsize(temp_model_path) > 0, "Model file is empty"


class TestCheckpointFormat:
    """Tests for checkpoint file format stability."""

    def test_state_dict_keys_match(self, device):
        """State dict keys should be consistent."""
        model1 = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        model2 = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        keys1 = set(model1.state_dict().keys())
        keys2 = set(model2.state_dict().keys())

        assert keys1 == keys2, "State dict keys inconsistent between models"

    def test_load_from_different_device(self, temp_model_path):
        """Model saved on GPU should load on CPU and vice versa."""
        # Save on available device
        save_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(save_device)

        torch.save(model.state_dict(), temp_model_path)

        # Load on CPU explicitly
        model2 = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        )
        model2.load_state_dict(torch.load(temp_model_path, map_location='cpu'))

        # Should work without errors
        x = torch.randn(2, 6, 11, 11)
        with torch.no_grad():
            output = model2(x)

        assert output.shape == (2, 8)


class TestTrainerCheckpoint:
    """Tests for full trainer checkpoint (policy + optimizer)."""

    def test_save_with_optimizer_state(self, device, temp_model_path):
        """Should be able to save model with optimizer state."""
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Do one update step
        x = torch.randn(4, 6, 11, 11, device=device)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Save both
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, temp_model_path)

        # Load
        checkpoint = torch.load(temp_model_path, map_location=device)

        model2 = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)
        model2.load_state_dict(checkpoint['model_state_dict'])

        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])

        # Should have loaded correctly
        assert len(optimizer2.state) > 0, "Optimizer state not loaded"

    def test_continued_training_produces_same_result(self, device, temp_model_path):
        """Continuing training from checkpoint should be reproducible."""
        torch.manual_seed(42)

        # Initial model and optimizer
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train for a bit
        for _ in range(5):
            x = torch.randn(4, 6, 11, 11, device=device)
            loss = model(x).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, temp_model_path)

        # Continue training
        torch.manual_seed(123)
        for _ in range(5):
            x = torch.randn(4, 6, 11, 11, device=device)
            loss = model(x).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_weights_1 = {n: p.clone() for n, p in model.named_parameters()}

        # Load checkpoint and continue with same seed
        model2 = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)
        checkpoint = torch.load(temp_model_path, map_location=device)
        model2.load_state_dict(checkpoint['model_state_dict'])
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])

        torch.manual_seed(123)
        for _ in range(5):
            x = torch.randn(4, 6, 11, 11, device=device)
            loss = model2(x).sum()
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

        # Should have same final weights
        for name, param in model2.named_parameters():
            assert torch.equal(param, final_weights_1[name]), \
                f"Parameter {name} differs after resumed training"


class TestLearnedBehaviorLoading:
    """Tests for loading trained behaviors."""

    def test_learned_behavior_loads_preset(self, config):
        """LearnedBehavior should load preset models without error."""
        from goodharts.behaviors.learned import create_learned_behavior

        # These should not raise errors
        for preset in ['ground_truth', 'proxy', 'proxy_jammed']:
            try:
                behavior = create_learned_behavior(preset)
                assert behavior is not None
            except FileNotFoundError:
                # Model file might not exist in test environment
                pytest.skip(f"Model file for {preset} not found")

    def test_loaded_behavior_produces_valid_actions(self, config, device):
        """Loaded behavior should produce valid action indices."""
        from goodharts.behaviors.learned import create_learned_behavior

        try:
            behavior = create_learned_behavior('ground_truth')
        except FileNotFoundError:
            pytest.skip("Ground truth model not found")

        # Create a mock view
        spec = ObservationSpec.for_mode('ground_truth', config)
        view = torch.randn(spec.num_channels, spec.view_size, spec.view_size)

        class MockAgent:
            x = 10
            y = 10
            sight_radius = spec.view_size // 2

        # Should return valid dx, dy
        result = behavior.decide_action(MockAgent(), view)

        assert len(result) == 2, "decide_action should return (dx, dy)"
        dx, dy = result
        assert -1 <= dx <= 1, f"dx out of range: {dx}"
        assert -1 <= dy <= 1, f"dy out of range: {dy}"
