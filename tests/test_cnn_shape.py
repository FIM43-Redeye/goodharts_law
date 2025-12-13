import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from behaviors.brains.tiny_cnn import TinyCNN

def test_shapes():
    shapes = [(5, 5), (11, 11), (21, 21)]
    for shape in shapes:
        print(f"Testing input shape: {shape}")
        model = TinyCNN(input_shape=shape)
        
        # Create dummy input (Batch=1, Channels=1, H, W)
        x = torch.randn(1, 1, *shape)
        
        try:
            output = model(x)
            print(f"  Output shape: {output.shape}")
            assert output.shape == (1, 8), f"Expected (1, 8), got {output.shape}"
            print("  Success!")
        except Exception as e:
            print(f"  Failed! {e}")
            raise e

if __name__ == "__main__":
    test_shapes()
