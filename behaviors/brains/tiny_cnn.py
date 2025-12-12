import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    """
    A small Convolutional Neural Network (CNN) for the agent's brain.
    
    This class inherits from torch.nn.Module, which is the base class for all
    neural network modules in PyTorch.
    
    The goal is to take a 2D local view of the world (like an image) and output
    an action (move direction).
    """
    def __init__(self, input_channels=1, output_size=4):
        super(TinyCNN, self).__init__()
        
        # TODO: Define your layers here!
        # A common pattern for small images is:
        # 1. Convolutional Layer (extract features)
        # 2. Activation Function (ReLU)
        # 3. Pooling Layer (reduce size) - sometimes optional for very small inputs
        # 4. Fully Connected (Linear) Layer (decision making)
        
        # Example definition (commented out):
        # self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=3)
        # self.fc1 = nn.Linear(in_features=..., out_features=output_size)
        pass

    def forward(self, x):
        """
        Defines the forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor representing the local view.
                              Shape: (batch_size, channels, height, width)
                              
        Returns:
            torch.Tensor: Output tensor representing action scores/logits.
                          Shape: (batch_size, output_size)
        """
        
        # ---------------------------------------------------------------------
        # EDUCATIONAL NOTE: The "Forward" Pass
        # 
        # In PyTorch, the forward method defines how data flows through the network.
        # You pass the input 'x' through the layers you defined in __init__.
        #
        # Example flow:
        # x = self.conv1(x)       # Apply convolution
        # x = F.relu(x)           # Apply activation function
        # x = torch.flatten(x, 1) # Flatten 2D features to 1D vector
        # x = self.fc1(x)         # Apply linear layer to get logits
        # ---------------------------------------------------------------------
        
        # TODO: Implement the forward pass
        raise NotImplementedError("You need to implement the forward method!")
