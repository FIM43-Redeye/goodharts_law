"""
Brain: Abstract interface for neural network architectures.

All brain implementations must follow this protocol to be used with:
- LearnedBehavior (inference)
- train_ppo.py (training)
- visualize_saliency.py (interpretability)

The BRAIN_REGISTRY allows plug-and-play swapping of architectures.
Serialization methods enable architecture-agnostic model saving/loading.
"""
from typing import Protocol, runtime_checkable, TYPE_CHECKING
from pathlib import Path
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from goodharts.modes import ObservationSpec


@runtime_checkable
class Brain(Protocol):
    """
    Protocol defining the interface all brain architectures must implement.

    This enables plug-and-play swapping of neural network architectures
    while keeping training code (PPO) and behavior code (LearnedBehavior) generic.

    Implementations must provide:
    - forward(): observation -> action logits
    - get_features(): observation -> intermediate features (for value head)
    - logits_from_features(): features -> action logits (avoid recomputation)
    - get_architecture_info(): serialization metadata
    - from_architecture_info(): deserialization factory
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
        Forward pass: observation -> action logits.

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

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits from pre-computed features.

        This enables PPO to compute features once and reuse them for both
        action logits and value estimation, avoiding duplicate forward passes.

        Args:
            features: Feature tensor from get_features(), shape (batch, hidden_size)

        Returns:
            Action logits of shape (batch, output_size)
        """
        ...

    def get_architecture_info(self) -> dict:
        """
        Return architecture parameters needed for reconstruction.

        This enables saving models with full metadata so they can be
        loaded without knowing the architecture in advance.

        Returns:
            Dict with all __init__ parameters needed to recreate this brain

        Example:
            {'input_shape': (11, 11), 'input_channels': 6, 'output_size': 8, ...}
        """
        ...

    @classmethod
    def from_architecture_info(cls, info: dict) -> 'Brain':
        """
        Reconstruct brain from architecture info.

        Args:
            info: Dict from get_architecture_info()

        Returns:
            New brain instance with same architecture (random weights)
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
    from .base_cnn import BaseCNN
    # Future: from .mlp import MLP
    # Future: from .transformer import TransformerBrain
    
    return {
        'base_cnn': BaseCNN,
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
        name: Brain architecture name (e.g., 'base_cnn')
        spec: ObservationSpec for input dimensions
        output_size: Number of output actions
        **kwargs: Architecture-specific parameters

    Returns:
        Configured brain instance (nn.Module)

    Raises:
        ValueError: If brain name not found in registry

    Example:
        brain = create_brain('base_cnn', spec, output_size=8)
    """
    registry = _get_brain_registry()

    if name not in registry:
        valid = ', '.join(registry.keys())
        raise ValueError(f"Unknown brain: {name}. Available: {valid}")

    brain_class = registry[name]
    return brain_class.from_spec(spec, output_size, **kwargs)


# =============================================================================
# SERIALIZATION - Save and load brains with full metadata
# =============================================================================

def _clean_state_dict(state_dict: dict) -> dict:
    """
    Clean state dict keys for compatibility.

    Handles:
    - torch.compile() prefix: '_orig_mod.conv1.weight' -> 'conv1.weight'
    """
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}


def _infer_brain_type_from_state_dict(state_dict: dict) -> str:
    """
    Infer brain type from state dict keys (backward compatibility).

    Returns:
        Brain type name for registry lookup
    """
    keys = set(state_dict.keys())

    # BaseCNN signature: conv1, conv2, conv3, fc1, fc_out
    if any('conv1' in k for k in keys) and any('fc_out' in k for k in keys):
        return 'base_cnn'

    # Future: Add detection for other architectures
    # if any('transformer' in k for k in keys):
    #     return 'transformer'

    # Default fallback
    return 'base_cnn'


def _infer_architecture_from_state_dict(state_dict: dict, brain_type: str) -> dict:
    """
    Infer architecture parameters from weight shapes (backward compatibility).

    This enables loading old models that were saved without metadata.
    """
    if brain_type == 'base_cnn':
        # conv1.weight: (out_channels, in_channels, kH, kW)
        conv1_weight = state_dict.get('conv1.weight')
        if conv1_weight is None:
            raise ValueError("Cannot infer architecture: missing conv1.weight")

        input_channels = conv1_weight.shape[1]

        # fc1.weight: (hidden_size, flatten_size)
        fc1_weight = state_dict.get('fc1.weight')
        if fc1_weight is None:
            raise ValueError("Cannot infer architecture: missing fc1.weight")

        hidden_size = fc1_weight.shape[0]
        flatten_size = fc1_weight.shape[1]

        # fc_out.weight: (output_size, hidden_size)
        fc_out_weight = state_dict.get('fc_out.weight')
        if fc_out_weight is None:
            raise ValueError("Cannot infer architecture: missing fc_out.weight")

        output_size = fc_out_weight.shape[0]

        # Infer spatial size: flatten_size = 64 * H * W (after conv3)
        spatial_total = flatten_size // 64
        side = int(spatial_total ** 0.5)
        input_shape = (side, side)

        return {
            'input_shape': input_shape,
            'input_channels': input_channels,
            'output_size': output_size,
            'hidden_size': hidden_size,
        }

    raise ValueError(f"Cannot infer architecture for brain type: {brain_type}")


def save_brain(
    brain: nn.Module,
    path: str | Path,
    brain_type: str = 'base_cnn',
    mode: str | None = None,
    training_steps: int | None = None,
    extra_metadata: dict | None = None,
) -> None:
    """
    Save brain with full metadata for architecture-agnostic loading.

    Args:
        brain: Neural network to save
        path: Output file path
        brain_type: Registry key for this brain type
        mode: Training mode (e.g., 'ground_truth', 'proxy')
        training_steps: Number of training steps completed
        extra_metadata: Additional metadata to include

    The saved checkpoint contains:
    - brain_type: Registry key for reconstruction
    - architecture: Parameters for from_architecture_info()
    - state_dict: Model weights
    - mode, training_steps: Training context
    """
    checkpoint = {
        'brain_type': brain_type,
        'architecture': brain.get_architecture_info(),
        'state_dict': brain.state_dict(),
    }

    if mode is not None:
        checkpoint['mode'] = mode
    if training_steps is not None:
        checkpoint['training_steps'] = training_steps
    if extra_metadata:
        checkpoint.update(extra_metadata)

    torch.save(checkpoint, path)


def load_brain(
    path: str | Path,
    device: torch.device | str | None = None,
    strict: bool = True,
) -> tuple[nn.Module, dict]:
    """
    Load brain with automatic architecture detection.

    Handles both:
    - New format: checkpoint with 'brain_type' and 'architecture' keys
    - Legacy format: raw state_dict (infers architecture from weights)

    Args:
        path: Path to saved model
        device: Device to load onto (None = CPU)
        strict: Whether to require exact state_dict match

    Returns:
        (brain, metadata) where metadata contains mode, training_steps, etc.

    Example:
        brain, meta = load_brain('models/ppo_ground_truth.pth', device='cuda')
        print(f"Loaded {meta.get('brain_type')} trained for {meta.get('training_steps')} steps")
    """
    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Detect format
    if isinstance(checkpoint, dict) and 'brain_type' in checkpoint:
        # New format with metadata
        brain_type = checkpoint['brain_type']
        architecture = checkpoint['architecture']
        state_dict = _clean_state_dict(checkpoint['state_dict'])

        metadata = {k: v for k, v in checkpoint.items()
                    if k not in ('state_dict', 'architecture')}
    else:
        # Legacy format: raw state_dict
        if isinstance(checkpoint, dict):
            state_dict = _clean_state_dict(checkpoint)
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

        brain_type = _infer_brain_type_from_state_dict(state_dict)
        architecture = _infer_architecture_from_state_dict(state_dict, brain_type)
        metadata = {'brain_type': brain_type, 'legacy_format': True}

    # Reconstruct brain
    registry = _get_brain_registry()
    if brain_type not in registry:
        valid = ', '.join(registry.keys())
        raise ValueError(f"Unknown brain type: {brain_type}. Available: {valid}")

    brain_class = registry[brain_type]
    brain = brain_class.from_architecture_info(architecture)
    brain.load_state_dict(state_dict, strict=strict)
    brain.to(device)
    brain.eval()

    return brain, metadata
