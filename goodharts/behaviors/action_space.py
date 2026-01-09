"""
Action space definitions and decoders.

This module defines the interface between neural network outputs and
environment actions. It provides:

1. ActionSpace ABC - interface all action spaces implement
2. Implementations - DiscreteGrid, Continuous, Factored
3. Registry - pluggable action space creation
4. Serialization - save/load action space config with models

The brain outputs raw values; the ActionSpace interprets them as movement.

Usage:
    # Create action space
    action_space = create_action_space('discrete_grid', max_move_distance=1)

    # Configure brain to match
    brain = create_brain('base_cnn', spec,
        output_size=action_space.n_outputs,
        action_mode=action_space.output_mode)

    # Decode during inference
    logits = brain(observation)
    dx, dy = action_space.decode(logits)
"""
from abc import ABC, abstractmethod
from functools import lru_cache

import torch
import torch.nn.functional as F


# =============================================================================
# ACTION SPACE PROTOCOL
# =============================================================================

class ActionSpace(ABC):
    """
    Abstract base class for action spaces.

    Defines the interface between neural network outputs and (dx, dy) actions.
    Implementations handle different output formats (discrete, continuous, etc.)
    """

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        """
        Number of outputs the brain should produce.

        For discrete: number of action logits
        For continuous: number of continuous values (typically 2 for dx, dy)
        """
        pass

    @property
    @abstractmethod
    def output_mode(self) -> str:
        """
        Output mode for the brain.

        Returns:
            'discrete' - brain outputs logits for classification
            'continuous' - brain outputs continuous values (e.g., tanh)
        """
        pass

    @property
    @abstractmethod
    def max_move_distance(self) -> int:
        """Maximum cells the agent can move per step."""
        pass

    @abstractmethod
    def decode(
        self,
        output: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> tuple[int, int]:
        """
        Convert brain output to (dx, dy) movement.

        Args:
            output: Brain output tensor (shape depends on implementation)
            sample: If True, sample from distribution; if False, take argmax/mean
            temperature: Sampling temperature (higher = more random)

        Returns:
            (dx, dy) movement tuple
        """
        pass

    @abstractmethod
    def decode_batch(
        self,
        output: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Batch decode brain outputs to (dx, dy) tensors.

        Args:
            output: Brain output tensor, shape (batch, n_outputs)
            sample: If True, sample; if False, deterministic
            temperature: Sampling temperature

        Returns:
            (dx_tensor, dy_tensor) each shape (batch,)
        """
        pass

    def get_action_index(
        self,
        output: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> int:
        """
        Get action index (for discrete spaces).

        For continuous spaces, raises NotImplementedError.

        Returns:
            Action index
        """
        raise NotImplementedError("Action index only available for discrete spaces")

    @abstractmethod
    def get_config(self) -> dict:
        """
        Return configuration for serialization.

        Returns:
            Dict with 'type' and all parameters needed to reconstruct
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> 'ActionSpace':
        """
        Reconstruct action space from config dict.

        Args:
            config: Dict from get_config()

        Returns:
            ActionSpace instance
        """
        pass


# =============================================================================
# HELPER FUNCTIONS (kept for backward compatibility)
# =============================================================================

@lru_cache(maxsize=10)
def build_action_list(max_move_distance: int = 1) -> tuple[tuple[int, int], ...]:
    """
    Build the list of possible (dx, dy) actions for a given movement range.

    Actions are ordered by iterating:
        for dx in range(-max_dist, max_dist + 1):
            for dy in range(-max_dist, max_dist + 1):
                if dx == 0 and dy == 0: skip

    Returns tuple for hashability (cached).

    Examples:
        max_move_distance=1 gives 8 actions (8-directional)
        max_move_distance=2 gives 24 actions (5x5 grid minus center)
    """
    actions = []
    for dx in range(-max_move_distance, max_move_distance + 1):
        for dy in range(-max_move_distance, max_move_distance + 1):
            if dx == 0 and dy == 0:
                continue
            actions.append((dx, dy))
    return tuple(actions)


# Legacy alias
def build_action_space(max_move_distance: int = 1) -> list[tuple[int, int]]:
    """Legacy wrapper - returns list instead of tuple."""
    return list(build_action_list(max_move_distance))


def num_actions(max_move_distance: int = 1) -> int:
    """Get number of discrete actions for a movement range."""
    return len(build_action_list(max_move_distance))


# =============================================================================
# IMPLEMENTATIONS
# =============================================================================

class DiscreteGridActionSpace(ActionSpace):
    """
    Discrete grid action space.

    Each action is a specific (dx, dy) pair. For max_move_distance=1,
    this gives 8 actions (8-directional movement).

    Pros: Simple, works well with standard RL algorithms
    Cons: Action count grows as (2*max_dist+1)^2 - 1
    """

    def __init__(self, max_move_distance: int = 1):
        self._max_move_distance = max_move_distance
        self._actions = build_action_list(max_move_distance)
        # Pre-compute tensors for batch decode
        self._dx_tensor = torch.tensor([a[0] for a in self._actions])
        self._dy_tensor = torch.tensor([a[1] for a in self._actions])

    @property
    def n_outputs(self) -> int:
        return len(self._actions)

    @property
    def output_mode(self) -> str:
        return 'discrete'

    @property
    def max_move_distance(self) -> int:
        return self._max_move_distance

    @property
    def actions(self) -> tuple[tuple[int, int], ...]:
        """All possible actions as (dx, dy) tuples."""
        return self._actions

    def index_to_action(self, idx: int) -> tuple[int, int]:
        """Convert action index to (dx, dy)."""
        if 0 <= idx < len(self._actions):
            return self._actions[idx]
        return (0, 0)

    def action_to_index(self, dx: int, dy: int) -> int:
        """Convert (dx, dy) to action index."""
        try:
            return self._actions.index((dx, dy))
        except ValueError:
            return 0

    def get_action_index(
        self,
        output: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> int:
        """Sample or argmax action index from logits."""
        logits = output.view(-1)

        if sample and temperature > 0:
            scaled = logits / temperature
            probs = F.softmax(scaled, dim=0)
            idx = torch.multinomial(probs, 1).item()
        else:
            idx = logits.argmax().item()

        return idx

    def decode(
        self,
        output: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> tuple[int, int]:
        """Convert logits to (dx, dy)."""
        idx = self.get_action_index(output, sample, temperature)
        return self._actions[idx]

    def decode_batch(
        self,
        output: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch decode logits to dx, dy tensors."""
        device = output.device

        # Ensure lookup tensors are on correct device
        if self._dx_tensor.device != device:
            self._dx_tensor = self._dx_tensor.to(device)
            self._dy_tensor = self._dy_tensor.to(device)

        if sample and temperature > 0:
            scaled = output / temperature
            probs = F.softmax(scaled, dim=-1)
            indices = torch.multinomial(probs, 1).squeeze(-1)
        else:
            indices = output.argmax(dim=-1)

        dx = self._dx_tensor[indices]
        dy = self._dy_tensor[indices]

        return dx, dy

    def get_config(self) -> dict:
        return {
            'type': 'discrete_grid',
            'max_move_distance': self._max_move_distance,
        }

    @classmethod
    def from_config(cls, config: dict) -> 'DiscreteGridActionSpace':
        return cls(max_move_distance=config.get('max_move_distance', 1))


class ContinuousActionSpace(ActionSpace):
    """
    Continuous action space.

    Brain outputs 2 values (dx, dy) in [-1, 1], scaled by max_move_distance.

    Pros: Compact output (always 2), natural for variable-distance movement
    Cons: Requires continuous RL algorithms or careful discretization
    """

    def __init__(self, max_move_distance: int = 1):
        self._max_move_distance = max_move_distance

    @property
    def n_outputs(self) -> int:
        return 2  # Always dx, dy

    @property
    def output_mode(self) -> str:
        return 'continuous'

    @property
    def max_move_distance(self) -> int:
        return self._max_move_distance

    def decode(
        self,
        output: torch.Tensor,
        sample: bool = True,  # Ignored for continuous
        temperature: float = 1.0,  # Ignored for continuous
    ) -> tuple[int, int]:
        """Convert continuous (dx, dy) to integer movement."""
        output = output.view(-1)
        dx = round(output[0].item() * self._max_move_distance)
        dy = round(output[1].item() * self._max_move_distance)

        # Clamp to valid range
        dx = max(-self._max_move_distance, min(self._max_move_distance, dx))
        dy = max(-self._max_move_distance, min(self._max_move_distance, dy))

        return (dx, dy)

    def decode_batch(
        self,
        output: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch decode continuous outputs."""
        # output shape: (batch, 2)
        scaled = output * self._max_move_distance
        dx = scaled[:, 0].round().clamp(-self._max_move_distance, self._max_move_distance).int()
        dy = scaled[:, 1].round().clamp(-self._max_move_distance, self._max_move_distance).int()
        return dx, dy

    def get_config(self) -> dict:
        return {
            'type': 'continuous',
            'max_move_distance': self._max_move_distance,
        }

    @classmethod
    def from_config(cls, config: dict) -> 'ContinuousActionSpace':
        return cls(max_move_distance=config.get('max_move_distance', 1))


class FactoredActionSpace(ActionSpace):
    """
    Factored action space: Direction (8) + Magnitude (M) as separate outputs.

    Brain outputs 8 + M values:
    - First 8: logits for direction (8 cardinal/diagonal)
    - Last M: logits for magnitude (1 to max_move_distance)

    Pros: Grows as 8 + M instead of (2M+1)^2
    Cons: Slightly more complex decoding, direction and magnitude are independent
    """

    # 8 unit directions
    DIRECTIONS = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    def __init__(self, max_move_distance: int = 1):
        self._max_move_distance = max_move_distance
        self._n_magnitudes = max_move_distance  # 1, 2, ..., max

    @property
    def n_outputs(self) -> int:
        return 8 + self._n_magnitudes  # 8 directions + M magnitudes

    @property
    def output_mode(self) -> str:
        return 'discrete'

    @property
    def max_move_distance(self) -> int:
        return self._max_move_distance

    def decode(
        self,
        output: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> tuple[int, int]:
        """Decode factored output to (dx, dy)."""
        output = output.view(-1)

        dir_logits = output[:8]
        mag_logits = output[8:]

        if sample and temperature > 0:
            dir_probs = F.softmax(dir_logits / temperature, dim=0)
            dir_idx = torch.multinomial(dir_probs, 1).item()

            mag_probs = F.softmax(mag_logits / temperature, dim=0)
            mag_idx = torch.multinomial(mag_probs, 1).item()
        else:
            dir_idx = dir_logits.argmax().item()
            mag_idx = mag_logits.argmax().item()

        # Get unit direction and scale by magnitude
        unit_dx, unit_dy = self.DIRECTIONS[dir_idx]
        magnitude = mag_idx + 1  # 0-indexed to 1-indexed

        return (unit_dx * magnitude, unit_dy * magnitude)

    def decode_batch(
        self,
        output: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch decode factored outputs."""
        device = output.device

        dir_logits = output[:, :8]
        mag_logits = output[:, 8:]

        if sample and temperature > 0:
            dir_probs = F.softmax(dir_logits / temperature, dim=-1)
            dir_idx = torch.multinomial(dir_probs, 1).squeeze(-1)

            mag_probs = F.softmax(mag_logits / temperature, dim=-1)
            mag_idx = torch.multinomial(mag_probs, 1).squeeze(-1)
        else:
            dir_idx = dir_logits.argmax(dim=-1)
            mag_idx = mag_logits.argmax(dim=-1)

        # Build direction lookup tensors
        dir_tensor = torch.tensor(self.DIRECTIONS, device=device)
        unit_dx = dir_tensor[dir_idx, 0]
        unit_dy = dir_tensor[dir_idx, 1]

        magnitude = mag_idx + 1

        return unit_dx * magnitude, unit_dy * magnitude

    def get_config(self) -> dict:
        return {
            'type': 'factored',
            'max_move_distance': self._max_move_distance,
        }

    @classmethod
    def from_config(cls, config: dict) -> 'FactoredActionSpace':
        return cls(max_move_distance=config.get('max_move_distance', 1))


# =============================================================================
# REGISTRY
# =============================================================================

ACTION_SPACE_REGISTRY: dict[str, type[ActionSpace]] = {
    'discrete_grid': DiscreteGridActionSpace,
    'continuous': ContinuousActionSpace,
    'factored': FactoredActionSpace,
}


def get_action_space_types() -> list[str]:
    """Get list of available action space types."""
    return list(ACTION_SPACE_REGISTRY.keys())


def create_action_space(
    space_type: str = 'discrete_grid',
    max_move_distance: int = 1,
    **kwargs,
) -> ActionSpace:
    """
    Factory: create an action space by type.

    Args:
        space_type: 'discrete_grid', 'continuous', or 'factored'
        max_move_distance: Maximum cells agent can move per step
        **kwargs: Additional type-specific parameters

    Returns:
        ActionSpace instance
    """
    if space_type not in ACTION_SPACE_REGISTRY:
        valid = ', '.join(ACTION_SPACE_REGISTRY.keys())
        raise ValueError(f"Unknown action space: {space_type}. Available: {valid}")

    space_class = ACTION_SPACE_REGISTRY[space_type]
    return space_class(max_move_distance=max_move_distance, **kwargs)


def load_action_space(config: dict) -> ActionSpace:
    """
    Load action space from config dict.

    Args:
        config: Dict with 'type' and parameters

    Returns:
        ActionSpace instance
    """
    space_type = config.get('type', 'discrete_grid')

    if space_type not in ACTION_SPACE_REGISTRY:
        valid = ', '.join(ACTION_SPACE_REGISTRY.keys())
        raise ValueError(f"Unknown action space: {space_type}. Available: {valid}")

    space_class = ACTION_SPACE_REGISTRY[space_type]
    return space_class.from_config(config)


# =============================================================================
# CONVENIENCE
# =============================================================================

# Pre-built common action spaces
DISCRETE_8 = DiscreteGridActionSpace(max_move_distance=1)

# Legacy compatibility
ACTIONS_8 = list(DISCRETE_8.actions)
ACTION_LABELS_8 = ['↖', '←', '↙', '↑', '↓', '↗', '→', '↘']


def get_direction_arrow(dx: int, dy: int) -> str:
    """
    Get arrow character for a direction.

    Uses a lookup table for unit directions, falls back to
    a composite representation for larger magnitudes.

    Args:
        dx, dy: Movement delta

    Returns:
        Arrow string like '→' or '2→' for magnitude 2
    """
    # Unit direction arrows (dy is inverted: -1 = up visually)
    ARROWS = {
        (-1, -1): '↖', (-1, 0): '←', (-1, 1): '↙',
        (0, -1): '↑',               (0, 1): '↓',
        (1, -1): '↗',  (1, 0): '→', (1, 1): '↘',
        (0, 0): '·',
    }

    # For unit vectors, just return the arrow
    if (dx, dy) in ARROWS:
        return ARROWS[(dx, dy)]

    # For larger magnitudes, normalize and prefix with magnitude
    magnitude = max(abs(dx), abs(dy))
    if magnitude == 0:
        return '·'

    unit_dx = dx // magnitude if dx != 0 else 0
    unit_dy = dy // magnitude if dy != 0 else 0

    arrow = ARROWS.get((unit_dx, unit_dy), '?')
    return f"{magnitude}{arrow}"


def get_action_labels(action_space: ActionSpace) -> list[str]:
    """
    Generate human-readable labels for all actions in an action space.

    Works with any ActionSpace type:
    - DiscreteGrid: labels each (dx, dy) action
    - Factored: labels 8 directions + magnitude levels
    - Continuous: returns ['dx', 'dy']

    Args:
        action_space: ActionSpace instance

    Returns:
        List of label strings, one per output
    """
    if isinstance(action_space, DiscreteGridActionSpace):
        return [get_direction_arrow(dx, dy) for dx, dy in action_space.actions]

    elif isinstance(action_space, FactoredActionSpace):
        # 8 directions + M magnitudes
        dir_labels = [get_direction_arrow(dx, dy) for dx, dy in FactoredActionSpace.DIRECTIONS]
        mag_labels = [f"x{i+1}" for i in range(action_space.max_move_distance)]
        return dir_labels + mag_labels

    elif isinstance(action_space, ContinuousActionSpace):
        return ['dx', 'dy']

    else:
        # Fallback: numeric indices
        return [str(i) for i in range(action_space.n_outputs)]


def index_to_action(idx: int, max_move_distance: int = 1) -> tuple[int, int]:
    """Legacy wrapper for DiscreteGridActionSpace.index_to_action."""
    space = DiscreteGridActionSpace(max_move_distance)
    return space.index_to_action(idx)


def action_to_index(dx: int, dy: int, max_move_distance: int = 1) -> int:
    """Legacy wrapper for DiscreteGridActionSpace.action_to_index."""
    space = DiscreteGridActionSpace(max_move_distance)
    return space.action_to_index(dx, dy)


# =============================================================================
# Developer Utilities
# =============================================================================

def print_action_space(space: ActionSpace | str = 'discrete_grid', max_move_distance: int = 1) -> None:
    """Print action space details for inspection."""
    if isinstance(space, str):
        space = create_action_space(space, max_move_distance)

    print(f"ActionSpace: {space.__class__.__name__}")
    print(f"  Type: {space.get_config()['type']}")
    print(f"  Max move distance: {space.max_move_distance}")
    print(f"  Output mode: {space.output_mode}")
    print(f"  N outputs: {space.n_outputs}")

    if isinstance(space, DiscreteGridActionSpace):
        print(f"  Actions:")
        for i, (dx, dy) in enumerate(space.actions):
            print(f"    {i}: ({dx:+d}, {dy:+d})")


if __name__ == "__main__":
    print("=== Discrete Grid (max_move_distance=1) ===")
    print_action_space('discrete_grid', 1)
    print()

    print("=== Discrete Grid (max_move_distance=2) ===")
    print_action_space('discrete_grid', 2)
    print()

    print("=== Continuous (max_move_distance=2) ===")
    print_action_space('continuous', 2)
    print()

    print("=== Factored (max_move_distance=3) ===")
    print_action_space('factored', 3)
