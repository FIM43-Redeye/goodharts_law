"""Tests for centralized TOML config loader."""
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config cache before each test."""
    from goodharts import config
    config._config = None
    config._config_path = None
    yield


def test_load_default_config():
    """Should load config.default.toml when no config.toml exists."""
    from goodharts.config import get_config, get_config_path
    
    cfg = get_config()
    
    assert cfg is not None
    assert 'world' in cfg
    assert 'resources' in cfg
    assert get_config_path().name in ('config.toml', 'config.default.toml')


def test_config_has_required_sections():
    """Config should have all expected sections."""
    from goodharts.config import get_config
    
    cfg = get_config()
    
    # Check all required sections exist
    assert 'world' in cfg
    assert 'resources' in cfg
    assert 'agent' in cfg
    assert 'agents' in cfg
    assert 'visualization' in cfg
    assert 'brain_view' in cfg
    assert 'training' in cfg  # Training hyperparameters


def test_config_caching():
    """get_config should return cached value."""
    from goodharts.config import get_config
    
    cfg1 = get_config()
    cfg2 = get_config()
    
    assert cfg1 is cfg2  # Same object (cached)


def test_reload_config():
    """reload_config should refresh the cache."""
    from goodharts.config import get_config, reload_config
    
    cfg1 = get_config()
    cfg2 = reload_config()
    
    assert cfg1 is not cfg2  # Different objects


def test_convenience_accessors():
    """Convenience functions should return correct sections."""
    from goodharts.config import (
        get_world_config,
        get_agent_config,
        get_resources_config,
        get_visualization_config,
        get_brain_view_config,
        get_agents_list,
        get_training_config,
    )
    
    assert 'width' in get_world_config()
    assert 'view_range' in get_agent_config()
    assert 'food' in get_resources_config()
    assert 'speed' in get_visualization_config()
    assert 'enabled' in get_brain_view_config()
    assert isinstance(get_agents_list(), list)
    assert 'min_food' in get_training_config()  # Environment randomization


def test_default_config_values():
    """Default config should have sensible values."""
    from goodharts.config import get_world_config, get_resources_config
    
    world = get_world_config()
    resources = get_resources_config()
    
    assert world.get('width', 0) > 0
    assert world.get('height', 0) > 0
    assert resources.get('food', 0) > 0
