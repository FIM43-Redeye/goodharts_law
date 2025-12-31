"""
Pytest configuration and shared fixtures.

Sets a consistent random seed before each test for reproducibility.
Disables torch.compile and cuDNN benchmark by default for fast test execution.

IMPORTANT: Environment variables are set at module load time (before torch import)
to ensure they take effect. The fixture approach doesn't work for unittest-style
tests or when torch is imported at module level.
"""
import os

# Set BEFORE any torch imports happen (module-level, not in fixture)
# - torch.compile adds 30-60s of startup time per test file
# - cuDNN benchmark adds 60-300s on AMD/ROCm (MIOpen doesn't cache across processes)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("GOODHARTS_CUDNN_BENCHMARK", "0")

import pytest

from goodharts.utils.seed import set_seed


@pytest.fixture(autouse=True)
def seed_tests():
    """Set a consistent seed before each test for reproducibility."""
    set_seed(42)
    yield
