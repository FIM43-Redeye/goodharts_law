"""
Utility functions for behavior handling.
"""


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
