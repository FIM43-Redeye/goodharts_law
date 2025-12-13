import pytest
import numpy as np
from goodharts.simulation import Simulation
from goodharts.configs.default_config import get_config

@pytest.fixture
def config():
    cfg = get_config()
    # Reduce size for faster tests
    cfg['GRID_WIDTH'] = 10
    cfg['GRID_HEIGHT'] = 10
    cfg['AGENTS_SETUP'] = [
        {'behavior_class': 'OmniscientSeeker', 'count': 1},
        {'behavior_class': 'ProxySeeker', 'count': 1}
    ]
    cfg['GRID_FOOD_INIT'] = 5
    cfg['GRID_POISON_INIT'] = 5
    return cfg

def test_simulation_init(config):
    sim = Simulation(config)
    assert sim.step_count == 0
    assert len(sim.agents) == 2
    assert sim.world.width == 10
    assert sim.world.height == 10

def test_simulation_step(config):
    sim = Simulation(config)
    initial_step = sim.step_count
    sim.step()
    assert sim.step_count == initial_step + 1

def test_agent_death_cleanup(config):
    sim = Simulation(config)
    # Kill an agent manually
    agent = sim.agents[0]
    agent.alive = False
    agent.death_reason = "Test"
    
    sim.step()
    
    # Should be removed
    assert agent not in sim.agents
    assert len(sim.stats['deaths']) == 1
    assert sim.stats['deaths'][0]['id'] == agent.id

def test_heatmap_update(config):
    sim = Simulation(config)
    sim.step()
    assert np.sum(sim.stats['heatmap']) > 0
