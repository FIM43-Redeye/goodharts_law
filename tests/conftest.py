"""
Pytest configuration and shared fixtures.

Sets a consistent random seed before each test for reproducibility.
Disables torch.compile by default for fast test execution.
"""
import os
import pytest

from goodharts.utils.seed import set_seed


@pytest.fixture(autouse=True)
def seed_tests():
    """Set a consistent seed before each test for reproducibility."""
    set_seed(42)
    yield


@pytest.fixture(autouse=True)
def disable_compilation(monkeypatch):
    """
    Disable torch.compile and cuDNN benchmark for tests by default.

    - torch.compile adds 30-60s of startup time per test file
    - cuDNN benchmark adds 60-300s on AMD/ROCm (MIOpen doesn't cache across processes)

    Tests that specifically need these features can enable them via config overrides.
    """
    # Disable Dynamo compilation entirely
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    # Disable cuDNN benchmark (expensive on AMD/ROCm)
    monkeypatch.setenv("GOODHARTS_CUDNN_BENCHMARK", "0")
    yield
