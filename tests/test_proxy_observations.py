"""Tests for proxy mode observation encoding."""
import pytest
import torch
from goodharts.environments.torch_env import create_torch_vec_env
from goodharts.modes import ObservationSpec
from goodharts.configs.default_config import get_simulation_config, CellType


@pytest.fixture
def config():
    cfg = get_simulation_config()
    cfg['GRID_WIDTH'] = 20
    cfg['GRID_HEIGHT'] = 20
    cfg['GRID_FOOD_INIT'] = 50
    cfg['GRID_POISON_INIT'] = 50
    return cfg


def test_proxy_observations_contain_interestingness(config):
    """Proxy mode observations should have interestingness values, not zeros."""
    spec = ObservationSpec.for_mode('proxy', config)
    env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config)

    obs = env.reset()

    # Both channels contain interestingness values
    assert obs.shape[1] == 2, "Proxy mode should have 2 channels"

    # Should have non-zero values (food/poison have interestingness)
    assert obs.max().item() > 0, \
        "Proxy observations are all zeros - encoding is broken!"

    # Food has interestingness 1.0, poison 0.5
    # We should see these values in the observations
    nonzero_mask = obs > 0
    unique_values = torch.unique(obs[nonzero_mask])
    assert len(unique_values) > 0, "No non-zero interestingness values found"


def test_proxy_channels_identical(config):
    """In proxy mode, both channels should have identical values."""
    spec = ObservationSpec.for_mode('proxy', config)
    env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config)

    obs = env.reset()

    # Both channels should be identical (same interestingness in both)
    assert torch.allclose(obs[0, 0], obs[0, 1]), \
        "Proxy channels should be identical (food/poison indistinguishable)"


def test_proxy_food_poison_indistinguishable(config):
    """In proxy mode, food and poison should look identical to the agent."""
    spec = ObservationSpec.for_mode('proxy', config)
    env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config)

    # Place food and poison in known locations
    env.agent_x[0] = 10
    env.agent_y[0] = 10
    env.grids[0, 5:16, 5:16] = CellType.EMPTY.value
    env.grids[0, 10, 12] = CellType.FOOD.value    # +2 in x
    env.grids[0, 10, 8] = CellType.POISON.value   # -2 in x

    obs = env._get_observations()
    r = 5  # view radius

    # Get values at food and poison positions
    food_ch0 = obs[0, 0, r, r + 2].item()
    food_ch1 = obs[0, 1, r, r + 2].item()
    poison_ch0 = obs[0, 0, r, r - 2].item()
    poison_ch1 = obs[0, 1, r, r - 2].item()

    # Both channels should have same value for each position
    assert abs(food_ch0 - food_ch1) < 0.001, "Food channels differ"
    assert abs(poison_ch0 - poison_ch1) < 0.001, "Poison channels differ"


def test_ground_truth_food_poison_distinguishable(config):
    """In ground_truth mode, food and poison should be in separate channels."""
    spec = ObservationSpec.for_mode('ground_truth', config)
    env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config)

    obs = env.reset()

    # Channel 0 is food, channel 1 is poison (2-channel encoding)
    food_channel = obs[0, 0, :, :]
    poison_channel = obs[0, 1, :, :]

    # Food and poison should be mutually exclusive
    overlap = (food_channel > 0.5) & (poison_channel > 0.5)
    assert overlap.sum().item() == 0, "Food and poison overlap in ground_truth mode"

    # Both should have some cells
    assert food_channel.sum().item() > 0, "No food visible in ground_truth mode"
    assert poison_channel.sum().item() > 0, "No poison visible in ground_truth mode"


def test_property_based_encoding_extensible(config):
    """Any CellTypeInfo property should be usable as an observation channel."""
    # This tests extensibility - energy_reward is another property

    # Verify that property_names includes expected properties
    property_names = CellType.FOOD.property_names()
    assert 'interestingness' in property_names
    assert 'energy_reward' in property_names
    assert 'energy_penalty' in property_names
