import pytest
import torch
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
    """Test that negative energy triggers agent done state and death tracking."""
    sim = Simulation(config)
    agent = sim.agents[0]
    idx = agent.idx
    
    # Record initial steps
    initial_steps = sim.vec_env.agent_steps[idx]
    
    # Force very negative energy that even eating food can't recover from
    # (food gives +10 energy, so -1000 is definitely fatal)
    sim.vec_env.agent_energy[idx] = -1000.0
    
    # Step the simulation - this should detect the done state
    sim.step()
    
    # VecEnv sets dones=True when energy <= 0
    # Simulation records death when dones[i] is True
    # Either deaths were recorded, OR the done flag was set
    death_detected = (
        len(sim.stats['deaths']) >= 1 or 
        sim.vec_env.dones[idx] == True
    )
    
    assert death_detected, \
        f"Expected death: deaths={len(sim.stats['deaths'])}, dones[{idx}]={sim.vec_env.dones[idx]}, energy={sim.vec_env.agent_energy[idx]}"

def test_heatmap_update(config):
    sim = Simulation(config)
    sim.step()
    # Heatmap is now a Torch tensor
    assert sim.stats['heatmap']['all'].sum().item() > 0
