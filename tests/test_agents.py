import pytest
import numpy as np
from goodharts.agents.organism import Organism
from goodharts.environments.world import World
from goodharts.behaviors import OmniscientSeeker

class MockBehavior(OmniscientSeeker):
    def decide_action(self, agent, view):
        return 1, 0  # Always move right

@pytest.fixture
def world(config):
    return World(10, 10, config)

@pytest.fixture
def config():
    from goodharts.configs.default_config import get_config
    return get_config()

def test_organism_move(world, config):
    agent = Organism(0, 0, 100, 5, world, MockBehavior(), config)
    initial_x = agent.x
    agent.update()
    assert agent.x == initial_x + 1

def test_organism_boundary(world, config):
    # Place at edge
    agent = Organism(world.width - 1, 0, 100, 5, world, MockBehavior(), config)
    # Try to move right
    agent.update()
    # Should stay at boundary
    assert agent.x == world.width - 1

def test_organism_eat_food(world, config):
    # Place food at (1,0)
    world.grid[0, 1] = config['CellType'].FOOD
    
    # Agent at (0,0) will move to (1,0)
    agent = Organism(0, 0, 100, 5, world, MockBehavior(), config)
    initial_energy = agent.energy
    
    agent.update()
    
    # Should have gained energy (reward - move_cost)
    # We know it gained reward because reward > move cost usually
    assert agent.energy > initial_energy - config['ENERGY_MOVE_COST']
    assert world.grid[0, 1] == config['CellType'].EMPTY

def test_organism_starvation(world, config):
    agent = Organism(0, 0, 0.05, 5, world, MockBehavior(), config) # Very low energy, less than move cost (0.1)
    agent.update() # Move costs energy
    assert not agent.alive
    assert agent.death_reason == "Starvation"
