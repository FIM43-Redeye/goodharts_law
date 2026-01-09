"""
BrainVisualizer: Architecture-agnostic neural network introspection.

Uses PyTorch forward hooks to capture layer activations in real-time.
Works with any nn.Module - CNNs, MLPs, transformers, etc.

Usage:
    visualizer = BrainVisualizer(model)
    output = model(input)  # Triggers hooks
    for name in visualizer.get_displayable_layers():
        img = visualizer.get_activation_display(name)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable


class BrainVisualizer:
    """
    Auto-discovers and visualizes activations from any nn.Module.
    
    Registers forward hooks on interesting layers (Conv2d, Linear, etc.)
    and captures their outputs for visualization.
    
    Attributes:
        model: The neural network being visualized
        activations: Dict mapping layer names to captured tensors
        layer_info: Dict with metadata about each layer (type, shape, etc.)
    """
    
    # Layer types we want to visualize
    VISUALIZABLE_LAYERS = (nn.Conv2d, nn.Linear)
    
    # Layer types to skip (internal/uninteresting)
    SKIP_PATTERNS = ('bias', 'weight')
    
    def __init__(self, model: nn.Module, layer_filter: Callable[[str, nn.Module], bool] | None = None):
        """
        Args:
            model: Neural network to visualize
            layer_filter: Optional function(name, module) -> bool to filter layers.
                          If None, all VISUALIZABLE_LAYERS are included.
                          Use this to show only specific layers in larger networks.
        """
        self.model = model
        self.activations: dict[str, torch.Tensor] = {}
        self.layer_info: dict[str, dict] = {}
        self._hooks: list = []
        self._layer_filter = layer_filter
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on all visualizable layers."""
        for name, module in self.model.named_modules():
            # Skip root module
            if name == '':
                continue
            
            # Check if this is a type we want to visualize
            if not isinstance(module, self.VISUALIZABLE_LAYERS):
                continue
            
            # Apply custom filter if provided
            if self._layer_filter is not None and not self._layer_filter(name, module):
                continue
            
            # Store layer metadata
            self.layer_info[name] = {
                'type': type(module).__name__,
                'module': module,
            }
            
            # Register hook
            hook = module.register_forward_hook(
                lambda m, inp, out, n=name: self._save_activation(n, out)
            )
            self._hooks.append(hook)
    
    def _save_activation(self, name: str, output: torch.Tensor):
        """Hook callback to save layer output."""
        # Detach and move to CPU to avoid GPU memory issues
        self.activations[name] = output.detach().cpu()
        
        # Update shape info on first capture
        if 'shape' not in self.layer_info[name]:
            self.layer_info[name]['shape'] = tuple(output.shape)
    
    def get_displayable_layers(self) -> list[str]:
        """Return list of layer names that have been captured."""
        return list(self.layer_info.keys())
    
    def get_layer_type(self, name: str) -> str:
        """Get the type of a layer (e.g., 'Conv2d', 'Linear')."""
        return self.layer_info.get(name, {}).get('type', 'Unknown')
    
    def get_activation_display(self, name: str, max_channels: int = 16) -> np.ndarray | None:
        """
        Get activation as a displayable 2D numpy array.
        
        Args:
            name: Layer name
            max_channels: For multi-channel outputs, limit to this many
        
        Returns:
            2D numpy array suitable for imshow(), or None if not available
        """
        if name not in self.activations:
            return None
        
        act = self.activations[name]
        
        # Remove batch dimension if present
        if act.dim() >= 1 and act.size(0) == 1:
            act = act.squeeze(0)
        
        layer_type = self.get_layer_type(name)
        
        if layer_type == 'Conv2d':
            return self._conv_to_display(act, max_channels)
        elif layer_type == 'Linear':
            return self._linear_to_display(act)
        else:
            # Generic fallback: try to make it 2D somehow
            return self._generic_to_display(act)
    
    def _conv_to_display(self, act: torch.Tensor, max_channels: int) -> np.ndarray:
        """
        Convert conv activation (C, H, W) to grid of feature maps.
        
        Creates a grid layout like:
        [ch0][ch1][ch2][ch3]
        [ch4][ch5][ch6][ch7]
        ...
        """
        if act.dim() != 3:
            return act.numpy() if act.dim() == 2 else act.view(-1).numpy().reshape(-1, 1)
        
        c, h, w = act.shape
        n_show = min(c, max_channels)
        
        # Calculate grid dimensions
        n_cols = int(np.ceil(np.sqrt(n_show)))
        n_rows = int(np.ceil(n_show / n_cols))
        
        # Create grid image
        grid = np.zeros((n_rows * h, n_cols * w))
        
        for i in range(n_show):
            row = i // n_cols
            col = i % n_cols
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = act[i].numpy()
        
        return grid
    
    def _linear_to_display(self, act: torch.Tensor) -> np.ndarray:
        """
        Convert linear activation to 2D display.

        Special cases:
        - 8 elements: Compass layout for 8-directional actions
        - Other: Square-ish tile with padding
        """
        flat = act.view(-1).numpy()
        n = len(flat)

        # Special case: 8-directional action space as compass
        if n == 8:
            return self._action_to_compass(flat)

        # Generic square layout
        side = int(np.ceil(np.sqrt(n)))

        if side * side == n:
            return flat.reshape(side, side)

        padded = np.zeros(side * side)
        padded[:n] = flat
        return padded.reshape(side, side)

    def _action_to_compass(self, values: np.ndarray) -> np.ndarray:
        """
        Arrange 8-directional action values as a compass rose.

        Input indices: [N, NE, E, SE, S, SW, W, NW] (standard 8-action space)

        Output 3x3 masked array:
            NW  N  NE
            W   X  E
            SW  S  SE

        Center cell is masked (not rendered at all).
        """
        # Map from action index to (row, col) in 3x3 grid
        # Actions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
        compass = np.zeros((3, 3))
        compass[0, 1] = values[0]  # N
        compass[0, 2] = values[1]  # NE
        compass[1, 2] = values[2]  # E
        compass[2, 2] = values[3]  # SE
        compass[2, 1] = values[4]  # S
        compass[2, 0] = values[5]  # SW
        compass[1, 0] = values[6]  # W
        compass[0, 0] = values[7]  # NW

        # Mask the center cell so it's not rendered
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        return np.ma.array(compass, mask=mask)
    
    def _generic_to_display(self, act: torch.Tensor) -> np.ndarray:
        """Fallback for unknown layer types."""
        flat = act.view(-1).numpy()
        # Try to make it roughly square
        n = len(flat)
        side = int(np.ceil(np.sqrt(n)))
        padded = np.zeros(side * side)
        padded[:n] = flat
        return padded.reshape(side, side)
    
    def get_action_probs(self, temperature: float = 1.0) -> np.ndarray | None:
        """
        Get action probabilities from the final output layer.
        
        Returns softmax probabilities if the last layer looks like action logits.
        """
        # Find the last layer (likely the output)
        layers = self.get_displayable_layers()
        if not layers:
            return None
        
        last_layer = layers[-1]
        if last_layer not in self.activations:
            return None
        
        logits = self.activations[last_layer]
        if logits.dim() > 1:
            logits = logits.squeeze()
        
        probs = F.softmax(logits / temperature, dim=0)
        return probs.numpy()
    
    def remove_hooks(self):
        """Clean up hooks when done."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def __del__(self):
        self.remove_hooks()


# Re-export from action_space for convenience
from goodharts.behaviors.action_space import (
    get_action_labels,
    DISCRETE_8,
)


def get_action_label(idx: int, n_actions: int) -> str:
    """
    Get human-readable label for action index.

    For standard 8-action discrete grid, returns direction arrows.
    Falls back to numeric index for unknown action counts.
    """
    if n_actions == 8:
        labels = get_action_labels(DISCRETE_8)
        if idx < len(labels):
            return labels[idx]
    return str(idx)
