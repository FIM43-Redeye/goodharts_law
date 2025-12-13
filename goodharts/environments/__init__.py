from goodharts.environments.base import Environment
from goodharts.environments.world import World

def create_world(config: dict) -> World:
    """Factory to create a World instance from config."""
    return World(config['GRID_WIDTH'], config['GRID_HEIGHT'], config)
