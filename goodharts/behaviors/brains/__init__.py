"""
Brain: Abstract interface for neural network architectures.

All brain implementations must follow this protocol to be used with:
- LearnedBehavior (inference)
- train_ppo.py (training)

The BRAIN_REGISTRY allows plug-and-play swapping of architectures.
"""
from typing import Protocol, runtime_checkable, TYPE_CHECKING
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from goodharts.configs.observation_spec import ObservationSpec


@runtime_checkable
class Brain(Protocol):
    """
    Protocol defining the interface all brain architectures must implement.
    
    This enables plug-and-play swapping of neural network architectures
    while keeping training code (PPO) and behavior code (LearnedBehavior) generic.
    """
    
    hidden_size: int  # Size of feature vector (for value head)
    
    @classmethod
    def from_spec(cls, spec: 'ObservationSpec', output_size: int, **kwargs) -> 'Brain':
        """
        Factory: create brain from observation spec.
        
        Args:
            spec: ObservationSpec defining input dimensions
            output_size: Number of output actions
            **kwargs: Architecture-specific parameters
        
        Returns:
            Configured brain instance
        """
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: observation → action logits.
        
        Args:
            x: Observation tensor of shape (batch, channels, height, width)
        
        Returns:
            Action logits of shape (batch, output_size)
        """
        ...
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representation (for value head in PPO).
        
        This is the intermediate representation BEFORE the final action layer.
        The value head attaches here to share feature extraction.
        
        Args:
            x: Observation tensor of shape (batch, channels, height, width)
        
        Returns:
            Feature tensor of shape (batch, hidden_size)
        """
        ...


# =============================================================================
# BRAIN REGISTRY - Add new architectures here
# =============================================================================

def _get_brain_registry() -> dict[str, type]:
    """
    Lazy import to avoid circular dependencies.
    
    Add new brain architectures here:
    """
    from .tiny_cnn import TinyCNN
    # Future: from .mlp import MLP
    # Future: from .transformer import TransformerBrain
    
    return {
        'tiny_cnn': TinyCNN,
        # 'mlp': MLP,
        # 'transformer': TransformerBrain,
    }


def get_brain_names() -> list[str]:
    """Get list of available brain architecture names."""
    return list(_get_brain_registry().keys())


def create_brain(name: str, spec: 'ObservationSpec', output_size: int, **kwargs) -> nn.Module:
    """
    Factory: create a brain by name.
    
    Args:
        name: Brain architecture name (e.g., 'tiny_cnn')
        spec: ObservationSpec for input dimensions
        output_size: Number of output actions
        **kwargs: Architecture-specific parameters
    
    Returns:
        Configured brain instance (nn.Module)
    
    Raises:
        ValueError: If brain name not found in registry
    
    Example:
        brain = create_brain('tiny_cnn', spec, output_size=8)
    """
    registry = _get_brain_registry()
    
    if name not in registry:
        valid = ', '.join(registry.keys())
        raise ValueError(f"Unknown brain: {name}. Available: {valid}")
    
    brain_class = registry[name]
    return brain_class.from_spec(spec, output_size, **kwargs)
