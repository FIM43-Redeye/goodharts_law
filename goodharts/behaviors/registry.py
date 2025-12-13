"""
Behavior auto-discovery registry.

Automatically discovers all BehaviorStrategy subclasses in the behaviors package.
No manual registration needed - just create a class that inherits from BehaviorStrategy
and it will be available via get_behavior().
"""
import importlib
import pkgutil
from pathlib import Path
from typing import Type

from goodharts.behaviors.base import BehaviorStrategy


_REGISTRY: dict[str, Type[BehaviorStrategy]] = {}
_DISCOVERED = False


def _register_from_module(module) -> None:
    """Register all BehaviorStrategy subclasses from a module."""
    for name in dir(module):
        if name.startswith('_'):
            continue
        obj = getattr(module, name)
        if (isinstance(obj, type) and 
            issubclass(obj, BehaviorStrategy) and 
            obj is not BehaviorStrategy):
            _REGISTRY[name] = obj


def _discover_behaviors() -> None:
    """Scan behaviors package for BehaviorStrategy subclasses."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    
    behaviors_path = Path(__file__).parent
    
    # Scan immediate submodules (skip base, registry, action_space)
    skip_modules = {'base', 'registry', 'action_space'}
    
    for finder, name, ispkg in pkgutil.iter_modules([str(behaviors_path)]):
        if name.startswith('_') or name in skip_modules:
            continue
        
        try:
            module = importlib.import_module(f'goodharts.behaviors.{name}')
            _register_from_module(module)
        except ImportError as e:
            # Silently skip modules that fail to import
            pass
        
        # Also scan subpackages (hardcoded/, learned/, brains/)
        if ispkg:
            subpkg_path = behaviors_path / name
            for _, subname, _ in pkgutil.iter_modules([str(subpkg_path)]):
                if subname.startswith('_'):
                    continue
                try:
                    submodule = importlib.import_module(f'goodharts.behaviors.{name}.{subname}')
                    _register_from_module(submodule)
                except ImportError:
                    pass
    
    _DISCOVERED = True


def get_behavior(name: str) -> Type[BehaviorStrategy]:
    """
    Get a behavior class by name.
    
    Args:
        name: Class name of the behavior (e.g., 'OmniscientSeeker')
        
    Returns:
        The behavior class
        
    Raises:
        ValueError: If behavior name is not found in registry
    """
    _discover_behaviors()
    if name not in _REGISTRY:
        available = ', '.join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown behavior: '{name}'. Available: [{available}]")
    return _REGISTRY[name]


def get_all_behaviors() -> dict[str, Type[BehaviorStrategy]]:
    """Get all registered behavior classes as a dict."""
    _discover_behaviors()
    return _REGISTRY.copy()


def list_behavior_names() -> list[str]:
    """List all registered behavior names."""
    _discover_behaviors()
    return sorted(_REGISTRY.keys())
