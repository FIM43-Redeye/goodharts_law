"""
Pluggable reward shaping system.

Provides distance-based reward signals to guide agents toward or away from
specific cell types. Designed to be extensible for predator/prey dynamics.

Key design:
- Only considers VISIBLE targets (within agent's view range)
- Distance weighting: closer targets have more influence (1/distance)
- Configurable attract/repel weights per cell type
"""
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from goodharts.configs.default_config import CellType


@dataclass(frozen=True)
class ShapingTarget:
    """
    Defines attraction/repulsion toward a cell type.
    
    Attributes:
        cell_type: What to attract/repel from
        weight: Positive = attract, negative = repel
        distance_decay: If True, closer targets influence more (1/distance)
    """
    cell_type: CellType
    weight: float
    distance_decay: bool = True


# Pre-built target configurations
PREY_TARGETS = (
    ShapingTarget(CellType.FOOD, weight=0.5, distance_decay=True),
    ShapingTarget(CellType.POISON, weight=-0.3, distance_decay=True),
)

PREDATOR_TARGETS = (
    ShapingTarget(CellType.PREY, weight=1.0, distance_decay=True),
)


def compute_shaping_score(
    view: np.ndarray,
    agent_pos: tuple[int, int],
    targets: Sequence[ShapingTarget],
) -> float:
    """
    Compute the shaping score for a given view.
    
    The score is a weighted sum of inverse distances to all visible targets.
    Higher score = better position (closer to attractors, farther from repellers).
    
    Args:
        view: Agent's local view, shape (channels, H, W) or (H, W) for single channel
        agent_pos: Agent's position within the view (usually center)
        targets: Sequence of ShapingTarget to consider
    
    Returns:
        Float shaping score
    """
    if view.ndim == 2:
        # Single channel view - need to know what channel it represents
        # This case shouldn't happen in normal usage
        return 0.0
    
    total_score = 0.0
    view_h, view_w = view.shape[1], view.shape[2]
    agent_y, agent_x = agent_pos
    
    for target in targets:
        # Get the channel for this cell type
        channel_idx = target.cell_type.channel_index
        if channel_idx >= view.shape[0]:
            continue  # Cell type not in observation
        
        channel = view[channel_idx]
        
        # Find all positions with this cell type
        positions = np.argwhere(channel > 0)  # (N, 2) array of (y, x) positions
        
        if len(positions) == 0:
            continue
        
        # Compute distances from agent to each target
        distances = np.sqrt(
            (positions[:, 0] - agent_y) ** 2 + 
            (positions[:, 1] - agent_x) ** 2
        )
        
        # Avoid division by zero for targets at agent position
        distances = np.maximum(distances, 0.1)
        
        if target.distance_decay:
            # Weighted inverse distance: closer = stronger influence
            # Sum of 1/dist gives natural gradient toward clusters while preferring closer
            influence = np.sum(1.0 / distances)
        else:
            # Simple count-based influence
            influence = len(positions)
        
        total_score += target.weight * influence
    
    return total_score


def compute_shaping_reward(
    view_before: np.ndarray,
    view_after: np.ndarray,
    agent_pos_before: tuple[int, int],
    agent_pos_after: tuple[int, int],
    targets: Sequence[ShapingTarget],
) -> float:
    """
    Compute reward for a state transition based on shaping improvement.
    
    Reward is positive if the agent moved to a better position (higher score).
    
    Args:
        view_before: Agent's view before the action
        view_after: Agent's view after the action
        agent_pos_before: Agent position in view before (usually center)
        agent_pos_after: Agent position in view after (usually center)
        targets: Shaping targets to consider
    
    Returns:
        Float reward (positive = improvement, negative = worse)
    """
    score_before = compute_shaping_score(view_before, agent_pos_before, targets)
    score_after = compute_shaping_score(view_after, agent_pos_after, targets)
    
    return score_after - score_before


def get_shaping_targets(mode: str) -> tuple[ShapingTarget, ...]:
    """
    Get shaping targets for a training mode.
    
    Args:
        mode: Training mode name
    
    Returns:
        Tuple of ShapingTarget for this mode
    """
    # For now, all modes use prey-style targets
    # Future: predator mode would use PREDATOR_TARGETS
    return PREY_TARGETS


def visualize_shaping_field(
    view: np.ndarray,
    targets: Sequence[ShapingTarget],
) -> np.ndarray:
    """
    Create a heatmap showing the shaping gradient across the view.
    
    Useful for debugging - shows what the shaping system "sees".
    
    Args:
        view: Agent's view, shape (channels, H, W)
        targets: Shaping targets
    
    Returns:
        2D array of shaping scores at each position
    """
    h, w = view.shape[1], view.shape[2]
    field = np.zeros((h, w))
    
    for y in range(h):
        for x in range(w):
            field[y, x] = compute_shaping_score(view, (y, x), targets)
    
    return field
