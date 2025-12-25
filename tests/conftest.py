"""
Pytest configuration and shared fixtures.

Sets a consistent random seed before each test for reproducibility.
"""
import pytest

from goodharts.utils.seed import set_seed


@pytest.fixture(autouse=True)
def seed_tests():
    """Set a consistent seed before each test for reproducibility."""
    set_seed(42)
    yield
