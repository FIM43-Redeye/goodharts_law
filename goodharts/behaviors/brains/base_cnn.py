"""
Neural network architectures for learned behaviors.

BaseCNN is a flexible architecture that supports:
- Arbitrary input shapes and channel counts
- Optional auxiliary scalar inputs (energy, step count, etc.)
- Both discrete and continuous action outputs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseCNN(nn.Module):
    """
    A robust Convolutional Neural Network for agent decision-making.
    
    Takes a local view of the world (like an image) and outputs
    either discrete action logits or continuous action values.
    
    Supports optional auxiliary scalar inputs that get concatenated
    after the CNN feature extraction.
    """
    
    def __init__(
        self, 
        input_shape: tuple[int, int], 
        input_channels: int = 1, 
        output_size: int = 8,
        num_aux_inputs: int = 0,
        action_mode: str = 'discrete',
        hidden_size: int = 512,  # Increased from 64
    ):
        """
        Args:
            input_shape: (height, width) of the input view
            input_channels: Number of grid channels (e.g., 4 for one-hot cell types)
            output_size: Number of discrete actions (ignored if action_mode='continuous')
            num_aux_inputs: Number of scalar auxiliary inputs (energy, etc.)
            action_mode: 'discrete' for classification, 'continuous' for dx/dy regression
            hidden_size: Size of hidden fully-connected layer
        """
        super(BaseCNN, self).__init__()
        
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.output_size = output_size
        self.num_aux_inputs = num_aux_inputs
        self.action_mode = action_mode
        self.hidden_size = hidden_size
        
        # Convolutional layers
        # Using padding=1 with kernel_size=3 preserves spatial dimensions
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        # Calculate flattened size after convolutions
        # With padding=1, kernel=3, stride=1: output_size = input_size
        conv_output_size = 64 * input_shape[0] * input_shape[1]
        
        # Fully connected layers
        # Input: CNN features + auxiliary scalars
        fc_input_size = conv_output_size + num_aux_inputs
        self.fc1 = nn.Linear(fc_input_size, hidden_size)

        
        # Output layer depends on action mode
        if action_mode == 'discrete':
            self.fc_out = nn.Linear(hidden_size, output_size)
        elif action_mode == 'continuous':
            # Output (dx, dy) as two continuous values
            self.fc_out = nn.Linear(hidden_size, 2)
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")
        
        self._init_weights()

    def _init_weights(self):
        # Orthogonal init for better PPO stability
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Scale the final action layer to be nearly 0 (uniform random policy at start)
        nn.init.orthogonal_(self.fc_out.weight, gain=0.01)

    @classmethod
    def from_spec(cls, spec: 'ObservationSpec', output_size: int, **kwargs) -> 'BaseCNN':
        """
        Factory: create BaseCNN from an ObservationSpec.
        
        This ensures model architecture is derived from the centralized spec,
        not hardcoded values. Any changes to observation format will
        automatically propagate to models.
        
        Args:
            spec: ObservationSpec defining input dimensions
            output_size: Number of output actions
            **kwargs: Additional args passed to BaseCNN.__init__
        
        Returns:
            BaseCNN configured for the spec
        
        Example:
            spec = config['get_observation_spec']('ground_truth')
            model = BaseCNN.from_spec(spec, output_size=8)
        """
        return cls(
            input_shape=spec.input_shape,
            input_channels=spec.num_channels,
            output_size=output_size,
            **kwargs
        )

    def get_features(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        """
        Extract feature representation (for value head in PPO).
        
        This is the intermediate representation AFTER conv layers and fc1,
        but BEFORE the final action layer (fc_out).
        
        Args:
            x: Grid input of shape (batch, channels, height, width)
            aux: Optional auxiliary scalars of shape (batch, num_aux_inputs)
        
        Returns:
            Feature tensor of shape (batch, hidden_size)
        """
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Concatenate auxiliary inputs if provided
        if aux is not None:
            if aux.dim() == 1:
                aux = aux.unsqueeze(0)
            x = torch.cat([x, aux], dim=1)
        elif self.num_aux_inputs > 0:
            zeros = torch.zeros(x.size(0), self.num_aux_inputs, device=x.device)
            x = torch.cat([x, zeros], dim=1)
        
        # Final feature layer
        features = F.relu(self.fc1(x))
        return features

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass: observation → action logits.
        
        Args:
            x: Grid input of shape (batch, channels, height, width)
            aux: Optional auxiliary scalars of shape (batch, num_aux_inputs)
            
        Returns:
            For discrete mode: logits of shape (batch, output_size)
            For continuous mode: (dx, dy) of shape (batch, 2), values in [-1, 1]
        """
        # Get features (shared computation with value head)
        features = self.get_features(x, aux)
        
        # Action output
        output = self.fc_out(features)
        
        if self.action_mode == 'continuous':
            output = torch.tanh(output)
        
        return output
    
    def get_action(self, x: torch.Tensor, aux: torch.Tensor | None = None, 
                   max_move_distance: int = 1) -> tuple[int, int]:
        """
        Convenience method to get (dx, dy) action from network output.
        
        Args:
            x: Grid input
            aux: Optional auxiliary scalars
            max_move_distance: Scale factor for continuous mode
            
        Returns:
            (dx, dy) movement tuple
        """
        with torch.no_grad():
            output = self(x, aux)
            
            if self.action_mode == 'discrete':
                action_idx = torch.argmax(output, dim=1).item()
                # Map index to (dx, dy) - this should match LearnedBehavior's action space
                return self._index_to_action(action_idx)
            else:
                # Continuous: scale tanh output by max_move_distance
                dx = output[0, 0].item() * max_move_distance
                dy = output[0, 1].item() * max_move_distance
                return (int(round(dx)), int(round(dy)))
    
    def _index_to_action(self, idx: int) -> tuple[int, int]:
        """Map action index to (dx, dy) for 8-directional movement."""
        # Standard mapping for max_move_distance=1
        actions = [
            (0, -1),   # 0: Up
            (0, 1),    # 1: Down
            (-1, 0),   # 2: Left
            (1, 0),    # 3: Right
            (-1, -1),  # 4: Up-Left
            (1, -1),   # 5: Up-Right
            (-1, 1),   # 6: Down-Left
            (1, 1),    # 7: Down-Right
        ]
        if 0 <= idx < len(actions):
            return actions[idx]
        return (0, 0)


class BaseCNNWithMemory(nn.Module):
    """
    CNN with LSTM for temporal memory.
    
    STUB for Phase 3 - allows agents to develop behavior 
    that evolves over time.
    """
    
    def __init__(
        self,
        input_shape: tuple[int, int],
        input_channels: int = 1,
        output_size: int = 8,
        hidden_size: int = 32,
        memory_size: int = 16,
    ):
        super().__init__()
        raise NotImplementedError("BaseCNNWithMemory is a Phase 3 stub")
