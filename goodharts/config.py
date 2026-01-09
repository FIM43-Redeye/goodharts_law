"""
Centralized configuration loader.

Loads config from TOML file and provides a cached get_config() function
that all modules can use.

Config precedence:
1. config.toml (user's custom config, in .gitignore)
2. config.default.toml (shipped defaults)

Usage:
    from goodharts.config import get_config
    config = get_config()
    print(config['world']['width'])
"""
import tomllib
from pathlib import Path


# Cached config
_config: dict | None = None
_config_path: Path | None = None

# Project root (where config files live)
PROJECT_ROOT = Path(__file__).parent.parent


def find_config_file() -> Path:
    """
    Find the config file to use.
    
    Priority:
    1. config.toml (user customizations)
    2. config.default.toml (shipped defaults)
    """
    user_config = PROJECT_ROOT / 'config.toml'
    default_config = PROJECT_ROOT / 'config.default.toml'
    
    if user_config.exists():
        return user_config
    elif default_config.exists():
        return default_config
    else:
        raise FileNotFoundError(
            f"No config file found. Expected one of:\n"
            f"  {user_config}\n"
            f"  {default_config}\n"
            f"Copy config.default.toml to config.toml to customize."
        )


def load_config(path: Path | str | None = None) -> dict:
    """
    Load configuration from a TOML file.
    
    Args:
        path: Path to config file. If None, auto-detects.
    
    Returns:
        Parsed config dictionary
    """
    global _config, _config_path
    
    if path is not None:
        config_path = Path(path)
    else:
        config_path = find_config_file()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    # Apply runtime overrides (must happen early)
    if 'runtime' in config:
        if config['runtime'].get('hsa_override_gfx_version'):
            import os
            # Only set if not already set by user env
            if 'HSA_OVERRIDE_GFX_VERSION' not in os.environ:
                val = config['runtime']['hsa_override_gfx_version']
                os.environ['HSA_OVERRIDE_GFX_VERSION'] = str(val)
                print(f"Applied HSA_OVERRIDE_GFX_VERSION={val} (from config)")

    _config = config
    _config_path = config_path
    
    return config


def get_config() -> dict:
    """
    Get the current configuration (loads if not already loaded).
    
    This is the main entry point for accessing config throughout the codebase.
    """
    global _config
    if _config is None:
        load_config()
    return _config


def get_config_path() -> Path | None:
    """Get the path to the currently loaded config file."""
    return _config_path


def reload_config(path: Path | str | None = None) -> dict:
    """Force reload of configuration."""
    global _config
    _config = None
    return load_config(path)


# Convenience accessors for common config sections
def get_world_config() -> dict:
    """Get [world] config section."""
    return get_config().get('world', {})


def get_agent_config() -> dict:
    """Get [agent] config section."""
    return get_config().get('agent', {})


def get_resources_config() -> dict:
    """Get [resources] config section."""
    return get_config().get('resources', {})


def get_visualization_config() -> dict:
    """Get [visualization] config section."""
    return get_config().get('visualization', {})


def get_brain_view_config() -> dict:
    """Get [brain_view] config section."""
    return get_config().get('brain_view', {})


def get_agents_list() -> list[dict]:
    """Get [[agents]] list."""
    return get_config().get('agents', [])


def get_training_config() -> dict:
    """
    Get [training] config section.
    
    Contains curriculum learning and PPO hyperparameters.
    """
    return get_config().get('training', {})


def get_runtime_config() -> dict:
    """
    Get [runtime] config section.

    Contains device selection and other runtime settings.
    """
    return get_config().get('runtime', {})


def get_evaluation_config() -> dict:
    """
    Get [evaluation] config section.

    Contains settings for model evaluation: n_envs, steps_per_env, runs, base_seed.
    """
    defaults = {
        'n_envs': 8192,
        'steps_per_env': 4096,
        'runs': 3,
        'base_seed': 42,
    }
    cfg = get_config().get('evaluation', {})
    return {**defaults, **cfg}

