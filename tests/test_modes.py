"""Tests for the mode system (ObservationSpec, ModeSpec, RewardComputer).

These tests verify the central mode registry that configures how agents
observe the world and receive rewards.
"""
import pytest
import numpy as np

from goodharts.modes import (
    ObservationSpec, 
    ModeSpec, 
    RewardComputer,
    GroundTruthRewards,
    ProxyRewards,
    ProxyJammedRewards,
    HandholdRewards,
    get_all_mode_names,
    get_mode_for_requirement,
    _get_modes,
)
from goodharts.configs.default_config import get_config, CellType


@pytest.fixture
def config():
    return get_config()


@pytest.fixture
def mode_registry(config):
    """Get the mode registry for testing."""
    return _get_modes(config)


class TestObservationSpec:
    """Tests for ObservationSpec configuration."""
    
    def test_ground_truth_spec_channels(self, config):
        """Ground truth mode should have one-hot channels."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        
        # Should have channels for each cell type
        assert spec.num_channels == CellType.num_types()
    
    def test_proxy_spec_channels(self, config):
        """Proxy mode should have interestingness channels."""
        spec = ObservationSpec.for_mode('proxy', config)
        
        # Proxy uses: empty, wall, then interestingness (replicated)
        # Exact count depends on implementation, but should match ground_truth
        assert spec.num_channels >= 2
    
    def test_spec_view_size_from_config(self, config):
        """View size should be derived from agent view_range."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        
        view_range = config.get('AGENT_VIEW_RANGE', 5)
        expected_size = 2 * view_range + 1
        
        assert spec.view_size == expected_size
    
    def test_spec_input_shape(self, config):
        """input_shape should return (view_size, view_size)."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        
        assert spec.input_shape == (spec.view_size, spec.view_size)
    
    def test_all_modes_have_spec(self, config, mode_registry):
        """All registered modes should produce valid specs."""
        for mode_name in mode_registry:
            spec = ObservationSpec.for_mode(mode_name, config)
            assert spec is not None
            assert spec.num_channels > 0
            assert spec.view_size > 0


class TestModeSpec:
    """Tests for ModeSpec registry."""
    
    def test_ground_truth_registered(self, mode_registry):
        """Ground truth mode should be in registry."""
        assert 'ground_truth' in mode_registry
    
    def test_proxy_registered(self, mode_registry):
        """Proxy mode should be in registry."""
        assert 'proxy' in mode_registry
    
    def test_proxy_jammed_registered(self, mode_registry):
        """Proxy jammed mode should be in registry."""
        assert 'proxy_jammed' in mode_registry
    
    def test_get_all_mode_names(self, config):
        """get_all_mode_names should return list of mode names."""
        names = get_all_mode_names(config)
        
        assert isinstance(names, list)
        assert 'ground_truth' in names
        assert 'proxy' in names
    
    def test_mode_spec_has_reward_strategy(self, mode_registry):
        """Each mode should define a reward strategy class."""
        for mode_name, spec in mode_registry.items():
            assert spec.reward_strategy is not None
            assert issubclass(spec.reward_strategy, RewardComputer)


class TestRewardComputer:
    """Tests for reward computation classes."""
    
    @pytest.fixture
    def gt_spec(self, config):
        return ObservationSpec.for_mode('ground_truth', config)
    
    def test_reward_computer_factory(self, config, gt_spec):
        """RewardComputer.create should return correct subclass."""
        computer = RewardComputer.create('ground_truth', gt_spec, gamma=0.99)
        assert isinstance(computer, GroundTruthRewards)
        
        # proxy mode uses ProxyRewards (main Goodhart failure case)
        proxy_spec = ObservationSpec.for_mode('proxy', config)
        computer = RewardComputer.create('proxy', proxy_spec, gamma=0.99)
        assert isinstance(computer, ProxyRewards)
        
        # proxy_jammed uses ProxyJammedRewards (information asymmetry case)
        jammed_spec = ObservationSpec.for_mode('proxy_jammed', config)
        computer = RewardComputer.create('proxy_jammed', jammed_spec, gamma=0.99)
        assert isinstance(computer, ProxyJammedRewards)
    
    def test_reward_computer_compute_returns_array(self, config, gt_spec):
        """compute() should return shaped rewards as array."""
        computer = RewardComputer.create('ground_truth', gt_spec, gamma=0.99)
        
        n_envs = 4
        view_size = gt_spec.view_size
        n_channels = gt_spec.num_channels
        
        # Create dummy observations
        states = np.random.randn(n_envs, n_channels, view_size, view_size).astype(np.float32)
        next_states = np.random.randn(n_envs, n_channels, view_size, view_size).astype(np.float32)
        raw_rewards = np.random.randn(n_envs).astype(np.float32)
        dones = np.zeros(n_envs, dtype=np.float32)
        
        # Initialize potentials
        computer.initialize(states)
        
        # Compute shaped rewards
        shaped = computer.compute(raw_rewards, states, next_states, dones)
        
        assert shaped.shape == (n_envs,)
        assert shaped.dtype == np.float32
    
    def test_reward_scaling_normalizes(self, config, gt_spec):
        """Reward scaling should reduce magnitude of raw rewards."""
        computer = RewardComputer.create('ground_truth', gt_spec, gamma=0.99)
        
        # Large raw rewards (typical energy deltas: +15 food, -10 poison)
        raw_rewards = np.array([15.0, -10.0, 0.0, 1.0], dtype=np.float32)
        
        scaled = computer._scale_rewards(raw_rewards)
        
        # Scaled should have smaller magnitude than raw (divides by 10)
        assert np.abs(scaled).max() < np.abs(raw_rewards).max(), \
            f"Scaled rewards should have smaller magnitude, raw max={np.abs(raw_rewards).max()}, scaled max={np.abs(scaled).max()}"
