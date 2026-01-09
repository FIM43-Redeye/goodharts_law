"""
Pytest configuration and shared fixtures.

Sets a consistent random seed before each test for reproducibility.
Disables torch.compile and cuDNN benchmark by default for fast test execution.

IMPORTANT: Environment variables are set at module load time (before torch import)
to ensure they take effect. The fixture approach doesn't work for unittest-style
tests or when torch is imported at module level.

Shared Fixtures:
    config: Base simulation configuration (unmodified defaults)
    small_config: 20x20 grid for faster integration tests
    tiny_config: 10x10 grid for minimal tests
    device: PyTorch device (CUDA if available, else CPU)
"""
import os

# Set BEFORE any torch imports happen (module-level, not in fixture)
# - torch.compile adds 30-60s of startup time per test file
# - cuDNN benchmark adds 60-300s on AMD/ROCm (MIOpen doesn't cache across processes)
# - CUBLAS_WORKSPACE_CONFIG is required for deterministic cuBLAS operations
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("GOODHARTS_CUDNN_BENCHMARK", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from goodharts.utils.seed import set_seed
from goodharts.configs.default_config import get_simulation_config


@pytest.fixture(autouse=True)
def seed_tests():
    """Set a consistent seed before each test for reproducibility."""
    set_seed(42)
    yield


@pytest.fixture
def device():
    """PyTorch device for test execution.

    Uses CUDA if available, otherwise CPU. Tests should work on both.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def config():
    """Base simulation configuration with default values.

    Returns an unmodified configuration. Tests that need specific settings
    should either:
    1. Use small_config/tiny_config for common cases
    2. Modify the config dict directly in the test
    3. Create a local fixture if the modifications are complex
    """
    return get_simulation_config()


@pytest.fixture
def small_config():
    """20x20 grid configuration for faster integration tests.

    Use this when you need a working environment but don't care about
    grid size. Good for testing mechanics, observations, and training loops.
    """
    cfg = get_simulation_config()
    cfg['GRID_WIDTH'] = 20
    cfg['GRID_HEIGHT'] = 20
    cfg['GRID_FOOD_INIT'] = 20
    cfg['GRID_POISON_INIT'] = 10
    return cfg


@pytest.fixture
def tiny_config():
    """10x10 grid configuration for minimal tests.

    Use this for the fastest possible test execution. Good for testing
    basic functionality where environment complexity doesn't matter.
    """
    cfg = get_simulation_config()
    cfg['GRID_WIDTH'] = 10
    cfg['GRID_HEIGHT'] = 10
    cfg['GRID_FOOD_INIT'] = 5
    cfg['GRID_POISON_INIT'] = 5
    return cfg
