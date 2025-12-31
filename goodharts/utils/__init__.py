"""
Utility functions for the Goodhart's Law simulation.

Provides device management, logging, seeding, and visualization helpers.
"""

from .device import get_device, apply_system_optimizations, is_tpu
from .logging_config import get_logger
from .seed import set_seed, get_random_seed

__all__ = [
    # Device management
    'get_device',
    'apply_system_optimizations',
    'is_tpu',
    # Logging
    'get_logger',
    # Reproducibility
    'set_seed',
    'get_random_seed',
]
