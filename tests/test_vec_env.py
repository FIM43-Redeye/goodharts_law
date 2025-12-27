"""Tests for the vectorized environment."""
import pytest
import torch
from goodharts.environments.torch_env import create_torch_vec_env
from goodharts.modes import ObservationSpec
from goodharts.configs.default_config import get_simulation_config


@pytest.fixture
def config():
    cfg = get_simulation_config()
    cfg['GRID_WIDTH'] = 20
    cfg['GRID_HEIGHT'] = 20
    cfg['GRID_FOOD_INIT'] = 50
    cfg['GRID_POISON_INIT'] = 10
    return cfg


def test_vec_env_creation(config):
    """VecEnv should initialize with correct shapes."""
    spec = ObservationSpec.for_mode('ground_truth', config)
    env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config)
    
    assert env.n_envs == 4
    assert env.grids.shape == (4, 20, 20)
    assert env.agent_x.shape == (4,)
    assert env.agent_y.shape == (4,)


def test_vec_env_step(config):
    """VecEnv step should return correct shapes."""
    spec = ObservationSpec.for_mode('ground_truth', config)
    env = create_torch_vec_env(n_envs=8, obs_spec=spec, config=config)
    
    # Random actions (8 possible directions)
    actions = torch.randint(0, 8, (8,), device=env.device)
    
    obs, rewards, dones = env.step(actions)
    
    assert obs.shape[0] == 8  # batch size
    assert obs.shape[1] == spec.num_channels
    assert rewards.shape == (8,)
    assert dones.shape == (8,)


def test_vec_env_reset(config):
    """VecEnv reset should reinitialize environments."""
    spec = ObservationSpec.for_mode('ground_truth', config)
    env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config)
    
    # Run some steps
    for _ in range(10):
        actions = torch.randint(0, 8, (4,), device=env.device)
        env.step(actions)
    
    # Reset specific environments
    env.reset(torch.tensor([0, 2], device=env.device))
    
    # Check energy is restored for reset envs
    assert env.agent_energy[0] == env.initial_energy
    assert env.agent_energy[2] == env.initial_energy


def test_vec_env_food_placement(config):
    """Items should be placed correctly on grid reset."""
    spec = ObservationSpec.for_mode('ground_truth', config)
    env = create_torch_vec_env(n_envs=2, obs_spec=spec, config=config)
    
    CellType = config['CellType']
    
    for grid_id in range(2):
        food_count = (env.grids[grid_id] == CellType.FOOD.value).sum().item()
        poison_count = (env.grids[grid_id] == CellType.POISON.value).sum().item()

        # Food may be less if agents spawned on food cells (up to 10 agents)
        # Allow for variance in random placement as well
        assert food_count >= 35, f"Expected ~50 food, got {food_count}"
        assert poison_count >= 5, f"Expected ~10 poison, got {poison_count}"


def test_vec_env_observation_shape(config):
    """Observations should match spec dimensions."""
    spec = ObservationSpec.for_mode('ground_truth', config)
    env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config)
    
    obs = env.reset()
    
    expected_shape = (4, spec.num_channels, spec.view_size, spec.view_size)
    assert obs.shape == torch.Size(expected_shape), f"Expected {expected_shape}, got {obs.shape}"
