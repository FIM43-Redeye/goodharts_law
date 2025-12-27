"""
Utility functions and constants for behavior handling.

This module provides shared utilities used by multiple behavior implementations,
reducing duplication between hardcoded behaviors like OmniscientSeeker and ProxySeeker.
"""

import torch


# 8-directional movement vectors (cardinals + diagonals)
# Used for random walks when no goal is visible
RANDOM_WALK_MOVES: list[tuple[int, int]] = [
    (0, 1), (0, -1), (1, 0), (-1, 0),   # Cardinals
    (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonals
]


def sign_scalar(x: int | float) -> int:
    """
    Return the sign of a scalar value.

    Unlike torch.sign(), this works on Python scalars without tensor overhead.

    Args:
        x: Numeric value

    Returns:
        1 if x > 0, -1 if x < 0, 0 if x == 0
    """
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def create_circular_mask(
    size: int,
    center: int,
    radius: int,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create grid coordinates and circular visibility mask.

    Used to mask out cells beyond the agent's sight radius when
    computing distances or searching for targets.

    Args:
        size: View dimension (typically 2*radius + 1)
        center: Center index (typically radius)
        radius: Sight radius in cells
        device: Torch device for tensor creation

    Returns:
        (y_grid, x_grid, visible_mask, dist_sq) - all (size, size) tensors
        - y_grid, x_grid: coordinate grids
        - visible_mask: True for cells within the circular radius
        - dist_sq: squared distance from center (useful for finding closest targets)
    """
    y_grid, x_grid = torch.meshgrid(
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        indexing='ij'
    )
    dist_sq = (x_grid - center)**2 + (y_grid - center)**2
    visible_mask = dist_sq <= radius**2
    return y_grid, x_grid, visible_mask, dist_sq


def get_behavior_name(behavior) -> str:
    """
    Get a human-readable name for a behavior instance.

    Handles both named behaviors (like 'OmniscientSeeker') and
    generic Python object representations (like '<LearnedBehavior ...>').

    Args:
        behavior: A behavior instance (any object with a string representation)

    Returns:
        A clean, human-readable name string

    Examples:
        >>> get_behavior_name(OmniscientSeeker())
        'OmniscientSeeker'
        >>> get_behavior_name(some_learned_behavior)
        'LearnedBehavior'
    """
    name = str(behavior)
    if name.startswith('<'):
        return type(behavior).__name__
    # Handle case like "BehaviorName<extra info>"
    return name.split('<')[0] if '<' in name else name
