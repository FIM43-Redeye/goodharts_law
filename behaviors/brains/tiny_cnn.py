import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TinyCNN(nn.Module):
    """
    A small Convolutional Neural Network (CNN) for the agent's brain.
    
    This class inherits from torch.nn.Module, which is the base class for all
    neural network modules in PyTorch.
    
    The goal is to take a 2D local view of the world (like an image) and output
    an action (move direction).
    """
    def __init__(self, input_shape: tuple[int, int], input_channels: int = 1, output_size: int = 4):
        """
        Args:
            input_shape (tuple): (height, width) of the input view.
            input_channels (int): Number of channels (1 for grayscale/grid values).
            output_size (int): Number of possible actions.
        """
        super(TinyCNN, self).__init__()
        
        self.input_shape = input_shape
        self.output_size = output_size
        
        # ---------------------------------------------------------------------
        # EDUCATIONAL NOTE: Convolutional Layers (Conv2d)
        # 
        # Conv2d layers slide small "filters" (kernels) over the input to detect features.
        # - in_channels: Number of input signal planes (1 for a simple grid).
        # - out_channels: Number of features to detect (e.g., 8).
        # - kernel_size: Size of the sliding window (3x3 is common).
        # ---------------------------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=3, padding=1)
        
        # ---------------------------------------------------------------------
        # EDUCATIONAL NOTE: Calculating Flattened Size
        # 
        # After convolutions, we have a 3D tensor (Channels, Height, Width).
        # To feed this into a linear (fully connected) layer, we must "flatten" it
        # into a 1D vector. We need to calculate how many numbers that is.
        # 
        # Since we used padding=1 and kernel_size=3 with stride=1, the spatial 
        # dimensions (H, W) remain the same after conv1.
        # ---------------------------------------------------------------------
        flattened_size = 8 * input_shape[0] * input_shape[1]
        
        # ---------------------------------------------------------------------
        # EDUCATIONAL NOTE: Linear Layers (Fully Connected)
        # 
        # Linear layers connect every input to every output. They are the "decision making"
        # part of the network, combining the features detected by the conv layers.
        # ---------------------------------------------------------------------
        self.fc1 = nn.Linear(in_features=flattened_size, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=output_size)

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
        # EDUCATIONAL NOTE: Activation Functions (ReLU)
        # 
        # ReLU (Rectified Linear Unit) introduces non-linearity.
        # It simply turns negative numbers into 0. This is crucial for the network
        # to learn complex patterns, not just linear combinations.
        # ---------------------------------------------------------------------
        
        # 1. Convolution + Activation
        x = self.conv1(x)
        x = F.relu(x)
        
        # 2. Flatten
        # x.size(0) is the batch size. We flatten dimensions 1 onwards.
        x = x.view(x.size(0), -1) 
        
        # 3. Dense Layers + Activation
        x = self.fc1(x)
        x = F.relu(x)
        
        # 4. Output Layer
        # No activation here usually! We want raw scores (logits).
        x = self.fc2(x)
        
        return x

