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


class TestObservationChannels:
    """Tests for observation channel consistency."""
    
    def test_ground_truth_channel_count(self, world, config):
        """Ground truth view should have one channel per CellType."""
        from goodharts.configs.observation_spec import ObservationSpec
        
        agent = Organism(5, 5, 100, 5, world, MockBehavior(), config)
        
        view = agent.get_local_view(mode='ground_truth')
        spec = ObservationSpec.for_mode('ground_truth', config)
        
        assert view.shape[0] == spec.num_channels, \
            f"ground_truth view has {view.shape[0]} channels but spec says {spec.num_channels}"
        assert view.shape[0] == len(config['CellType'].all_types()), \
            f"ground_truth should have one channel per CellType"
    
    def test_proxy_channel_count_matches_ground_truth(self, world, config):
        """Proxy view must have same channel count as ground truth for architecture compat."""
        agent = Organism(5, 5, 100, 5, world, MockBehavior(), config)
        
        gt_view = agent.get_local_view(mode='ground_truth')
        proxy_view = agent.get_local_view(mode='proxy')
        
        assert proxy_view.shape == gt_view.shape, \
            f"proxy view {proxy_view.shape} must match ground_truth {gt_view.shape}"
    
    def test_observation_spec_matches_actual_view(self, world, config):
        """ObservationSpec channel count should match actual get_local_view output."""
        from goodharts.configs.observation_spec import ObservationSpec
        
        agent = Organism(5, 5, 100, 5, world, MockBehavior(), config)
        
        for mode in ['ground_truth', 'proxy', 'proxy_ill_adjusted']:
            spec = ObservationSpec.for_mode(mode, config)
            # proxy_ill_adjusted uses proxy observation
            actual_mode = 'proxy' if 'proxy' in mode else mode
            view = agent.get_local_view(mode=actual_mode)
            
            assert view.shape[0] == spec.num_channels, \
                f"{mode}: view has {view.shape[0]} channels but spec says {spec.num_channels}"

