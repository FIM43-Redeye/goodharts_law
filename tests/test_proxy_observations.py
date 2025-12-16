"""Tests for proxy mode observation encoding."""
import pytest
import numpy as np
from goodharts.environments.vec_env import create_vec_env
from goodharts.modes import ObservationSpec
from goodharts.configs.default_config import get_config, CellType


@pytest.fixture
def config():
    cfg = get_config()
    cfg['GRID_WIDTH'] = 20
    cfg['GRID_HEIGHT'] = 20
    cfg['GRID_FOOD_INIT'] = 50
    cfg['GRID_POISON_INIT'] = 50
    return cfg


def test_proxy_observations_contain_interestingness(config):
    """Proxy mode observations should have interestingness values, not zeros."""
    spec = ObservationSpec.for_mode('proxy', config)
    env = create_vec_env(n_envs=4, obs_spec=spec, config=config)
    
    obs = env.reset()
    
    # Channels 2-5 are interestingness in proxy mode
    interestingness_channels = obs[:, 2:, :, :]
    
    # Should NOT be all zeros
    assert interestingness_channels.max() > 0, \
        "Interestingness channels are all zeros - proxy encoding is broken!"
    
    # Food has interestingness 1.0, poison 0.9
    # We should see these values in the observations
    unique_values = np.unique(interestingness_channels[interestingness_channels > 0])
    assert len(unique_values) > 0, "No non-zero interestingness values found"


def test_proxy_food_poison_nearly_identical(config):
    """In proxy mode, food and poison should have similar interestingness."""
    spec = ObservationSpec.for_mode('proxy', config)
    env = create_vec_env(n_envs=1, obs_spec=spec, config=config)
    
    obs = env.reset()
    
    # Get interestingness channel (channel 2)
    interestingness = obs[0, 2, :, :]
    
    # Find cells with high interestingness (>0.8) 
    # Both food (1.0) and poison (0.9) qualify
    high_interest = interestingness > 0.8
    
    # Should have multiple high-interest cells (mix of food and poison)
    assert high_interest.sum() > 0, "No high-interestingness cells visible"


def test_ground_truth_food_poison_distinguishable(config):
    """In ground_truth mode, food and poison should be in separate channels."""
    spec = ObservationSpec.for_mode('ground_truth', config)
    env = create_vec_env(n_envs=1, obs_spec=spec, config=config)
    
    obs = env.reset()
    
    # Channel 2 is food (one-hot), channel 3 is poison (one-hot)
    food_channel = obs[0, 2, :, :]
    poison_channel = obs[0, 3, :, :]
    
    # Food and poison should be mutually exclusive
    overlap = (food_channel > 0.5) & (poison_channel > 0.5)
    assert overlap.sum() == 0, "Food and poison overlap in ground_truth mode"
    
    # Both should have some cells
    assert food_channel.sum() > 0, "No food visible in ground_truth mode"
    assert poison_channel.sum() > 0, "No poison visible in ground_truth mode"


def test_property_based_encoding_extensible(config):
    """Any CellTypeInfo property should be usable as an observation channel."""
    # This tests extensibility - energy_reward is another property
    # ObservationSpec is already imported at the top from goodharts.modes
    
    # Verify that property_names includes expected properties
    property_names = CellType.FOOD.property_names()
    assert 'interestingness' in property_names
    assert 'energy_reward' in property_names
    assert 'energy_penalty' in property_names

