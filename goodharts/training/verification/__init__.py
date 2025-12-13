"""
Model verification suite.

Run headless tests on trained models to verify behavior and fitness
before running visual demos.

Usage:
    python -m goodharts.training.verification
    python -m goodharts.training.verification --steps 500 --verbose
"""
from .directional import test_directional_accuracy
from .survival import test_simulation_survival, compare_behaviors

__all__ = [
    'test_directional_accuracy',
    'test_simulation_survival',
    'compare_behaviors',
]
