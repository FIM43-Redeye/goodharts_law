"""
Centralized action space definitions.

This is the SINGLE SOURCE OF TRUTH for action indexing.
All components (LearnedBehavior, data collection, BaseCNN, etc.) 
should import from here to avoid mismatches.
"""
from functools import lru_cache


@lru_cache(maxsize=10)
def build_action_space(max_move_distance: int = 1) -> list[tuple[int, int]]:
    """
    Build the list of possible actions for a given movement range.
    
    Actions are (dx, dy) tuples ordered by iterating:
        for dx in range(-max_dist, max_dist + 1):
            for dy in range(-max_dist, max_dist + 1):
                if dx == 0 and dy == 0: skip
    
    Args:
        max_move_distance: Maximum cells the agent can move per step
        
    Returns:
        List of (dx, dy) tuples in deterministic order
        
    Examples:
        max_move_distance=1 gives 8 actions:
            0: (-1, -1)  ↖ Up-Left
            1: (-1,  0)  ← Left  
            2: (-1,  1)  ↙ Down-Left
            3: ( 0, -1)  ↑ Up
            4: ( 0,  1)  ↓ Down
            5: ( 1, -1)  ↗ Up-Right
            6: ( 1,  0)  → Right
            7: ( 1,  1)  ↘ Down-Right
    """
    actions = []
    for dx in range(-max_move_distance, max_move_distance + 1):
        for dy in range(-max_move_distance, max_move_distance + 1):
            if dx == 0 and dy == 0:
                continue  # No "stay in place" action
            actions.append((dx, dy))
    return actions


def action_to_index(dx: int, dy: int, max_move_distance: int = 1) -> int:
    """
    Convert (dx, dy) action to its index in the action space.
    
    Args:
        dx, dy: Movement delta
        max_move_distance: Movement range (must match what was used for action space)
        
    Returns:
        Index into the action list, or 0 if action not found
    """
    actions = build_action_space(max_move_distance)
    try:
        return actions.index((dx, dy))
    except ValueError:
        # Action not in valid range - return 0 (first action)
        return 0


def index_to_action(idx: int, max_move_distance: int = 1) -> tuple[int, int]:
    """
    Convert action index to (dx, dy) tuple.
    
    Args:
        idx: Action index
        max_move_distance: Movement range
        
    Returns:
        (dx, dy) movement tuple
    """
    actions = build_action_space(max_move_distance)
    if 0 <= idx < len(actions):
        return actions[idx]
    return (0, 0)  # Invalid index - no movement


def num_actions(max_move_distance: int = 1) -> int:
    """Get number of actions for given movement range."""
    return len(build_action_space(max_move_distance))


# Convenience: pre-computed for common case
ACTIONS_8 = build_action_space(1)  # 8 directional
ACTION_LABELS_8 = ['↖', '←', '↙', '↑', '↓', '↗', '→', '↘']


# Debug helper
def print_action_space(max_move_distance: int = 1):
    """Print the action space for debugging."""
    actions = build_action_space(max_move_distance)
    labels_8 = ['↖', '←', '↙', '↑', '↓', '↗', '→', '↘']
    print(f"Action space for max_move_distance={max_move_distance} ({len(actions)} actions):")
    for i, (dx, dy) in enumerate(actions):
        label = labels_8[i] if max_move_distance == 1 and i < 8 else ''
        print(f"  {i}: ({dx:+d}, {dy:+d})  {label}")


if __name__ == "__main__":
    print_action_space(1)
    print()
    print_action_space(2)
