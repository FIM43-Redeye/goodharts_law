"""
Behavior strategies for agents.

This package provides both hardcoded baseline behaviors and learned (neural network)
behaviors. Use the registry functions to discover available behaviors dynamically.

Example usage:
    # Get a behavior by name
    from goodharts.behaviors import get_behavior
    BehaviorClass = get_behavior('OmniscientSeeker')
    
    # List all available behaviors
    from goodharts.behaviors import list_behavior_names
    print(list_behavior_names())
    
    # Create learned behavior from preset
    from goodharts.behaviors import create_learned_behavior
    behavior = create_learned_behavior('ground_truth', model_path='models/my_model.pth')
"""
from goodharts.behaviors.base import BehaviorStrategy, ROLE_COLORS
from goodharts.behaviors.registry import get_behavior, get_all_behaviors, list_behavior_names

# Explicit imports for convenience and backwards compatibility
from goodharts.behaviors.hardcoded import OmniscientSeeker, ProxySeeker
from goodharts.behaviors.learned import (
    LearnedBehavior, 
    create_learned_behavior,
    LEARNED_PRESETS,
)

__all__ = [
    # Base class
    'BehaviorStrategy', 
    'ROLE_COLORS',
    # Registry functions
    'get_behavior', 
    'get_all_behaviors', 
    'list_behavior_names',
    # Hardcoded behaviors
    'OmniscientSeeker', 
    'ProxySeeker',
    # Learned behaviors
    'LearnedBehavior', 
    'create_learned_behavior',
    'LEARNED_PRESETS',
]