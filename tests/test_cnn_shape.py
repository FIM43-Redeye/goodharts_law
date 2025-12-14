"""Tests for CNN shape compatibility with various input sizes."""
import torch

from goodharts.configs.default_config import get_config
from goodharts.behaviors.brains.base_cnn import BaseCNN
from goodharts.behaviors.action_space import num_actions


def test_shapes():
    """Test that BaseCNN works with various input shapes."""
    config = get_config()
    obs_spec = config['get_observation_spec']('ground_truth')
    
    # Get dynamic values from config
    n_channels = obs_spec.num_channels
    n_actions = num_actions(1)  # 8 for max_move_distance=1
    
    shapes = [(5, 5), (11, 11), (21, 21)]
    
    for shape in shapes:
        print(f"Testing input shape: {shape} with {n_channels} channels")
        model = BaseCNN(input_shape=shape, input_channels=n_channels, output_size=n_actions)
        
        # Create dummy input (Batch=1, Channels=n_channels, H, W)
        x = torch.randn(1, n_channels, *shape)
        
        try:
            output = model(x)
            print(f"  Output shape: {output.shape}")
            expected_shape = (1, n_actions)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            print("  Success!")
        except Exception as e:
            print(f"  Failed! {e}")
            raise e


if __name__ == "__main__":
    test_shapes()
