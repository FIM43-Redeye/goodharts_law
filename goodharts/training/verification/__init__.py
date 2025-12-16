"""
Model verification and testing suite.

Provides tools to validate trained models before deployment:

- **Survival tests**: Run headless simulations to compare agent types
- **Directional tests**: Verify models move toward food correctly

See Also
--------
goodharts.training.verification.survival : Full survival statistics
goodharts.training.verification.directional : Directional accuracy tests

Example
-------
Run the full verification suite from CLI::

    python -m goodharts.training.verification --steps 500 --verbose
"""
from .survival import run_survival_test, run_comparative_verification

__all__ = [
    'run_survival_test',
    'run_comparative_verification',
]
