"""
Directional accuracy tests for trained models.

Tests whether models correctly move toward food placed in different directions.
"""
import torch

from goodharts.configs.default_config import get_simulation_config, CellType
from goodharts.behaviors.brains.base_cnn import BaseCNN


def test_directional_accuracy(model_path: str, model_name: str) -> float:
    """
    Test if model goes toward food in each direction.
    
    Creates synthetic observations with food placed at specific positions
    and checks if the model's chosen action moves toward the food.
    
    Args:
        model_path: Path to saved model weights
        model_name: Display name for output
        
    Returns:
        Accuracy as float 0.0-1.0
    """
    print(f"Testing {model_name}...")
    
    config = get_simulation_config()
    obs_spec = config['get_observation_spec']('ground_truth')
    
    # Dynamic dimensions from observation spec
    num_channels = obs_spec.num_channels
    view_size = obs_spec.input_shape[0]  # e.g., 11
    center = view_size // 2
    
    try:
        model = BaseCNN(
            input_shape=obs_spec.input_shape,
            input_channels=num_channels,
            output_size=8
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()
    except FileNotFoundError:
        print(f"Model not found: {model_path}")
        return 0.0
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0.0
    
    # Test cases: food position -> expected direction indices
    # Positions are relative to center, expected actions are direction indices
    # Action indices: 0=up-left, 1=up, 2=up-right, 3=left, 4=right, 5=down-left, 6=down, 7=down-right
    test_cases = [
        ((center, center - 2), {3, 0, 5}),     # Left of center -> should go left-ish
        ((center, center + 2), {4, 2, 7}),     # Right of center -> should go right-ish
        ((center - 2, center), {1, 0, 2}),     # Above center -> should go up-ish
        ((center + 2, center), {6, 5, 7}),     # Below center -> should go down-ish
        ((center - 2, center - 2), {0}),       # Up-left -> should go up-left
        ((center + 2, center + 2), {7}),       # Down-right -> should go down-right
        ((center - 2, center + 2), {2}),       # Up-right -> should go up-right
        ((center + 2, center - 2), {5}),       # Down-left -> should go down-left
    ]
    
    correct = 0
    for food_pos, expected_actions in test_cases:
        # Create observation tensor
        view = torch.zeros(1, num_channels, view_size, view_size)
        
        # Set empty channel to 1.0 everywhere
        view[0, CellType.EMPTY.channel_index, :, :] = 1.0
        
        # Place food at target position
        food_y, food_x = food_pos
        view[0, CellType.EMPTY.channel_index, food_y, food_x] = 0.0
        view[0, CellType.FOOD.channel_index, food_y, food_x] = 1.0
        
        # Blank center (agent's own position)
        view[0, :, center, center] = 0.0
        
        with torch.no_grad():
            logits = model(view)
            action_idx = logits.argmax(dim=1).item()
        
        if action_idx in expected_actions:
            correct += 1
    
    accuracy = correct / len(test_cases)
    print(f"Directional accuracy: {accuracy:.0%} ({correct}/{len(test_cases)})")
    
    return accuracy
