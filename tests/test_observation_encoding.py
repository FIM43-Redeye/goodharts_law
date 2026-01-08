"""Tests for observation encoding correctness.

These tests verify that observations are correctly encoded for each mode,
ensuring the neural network receives the expected input format.
"""
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
    cfg['GRID_FOOD_INIT'] = 30
    cfg['GRID_POISON_INIT'] = 20
    cfg['AGENT_VIEW_RANGE'] = 5
    return cfg


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestObservationShape:
    """Tests for observation tensor shapes."""

    def test_observation_has_correct_batch_dimension(self, config, device):
        """Observations should have correct batch size."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        n_envs = 8
        env = create_torch_vec_env(n_envs=n_envs, obs_spec=spec, config=config, device=device)

        obs = env.reset()

        assert obs.shape[0] == n_envs, f"Batch dim should be {n_envs}, got {obs.shape[0]}"

    def test_observation_has_correct_channels(self, config, device):
        """Observations should have correct number of channels."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)

        obs = env.reset()

        assert obs.shape[1] == spec.num_channels, \
            f"Channels should be {spec.num_channels}, got {obs.shape[1]}"

    def test_observation_view_size_matches_config(self, config, device):
        """Observation spatial dimensions should match view range."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)

        obs = env.reset()

        expected_size = 2 * config['AGENT_VIEW_RANGE'] + 1

        assert obs.shape[2] == expected_size, f"Height should be {expected_size}"
        assert obs.shape[3] == expected_size, f"Width should be {expected_size}"


class TestGroundTruthEncoding:
    """Tests for ground truth mode observation encoding.

    Ground truth uses 2-channel binary encoding:
    - Channel 0: is_food (1.0 where food, 0.0 elsewhere)
    - Channel 1: is_poison (1.0 where poison, 0.0 elsewhere)
    - Empty cells: [0, 0]
    - Food cells: [1, 0]
    - Poison cells: [0, 1]
    """

    def test_ground_truth_is_binary_encoding(self, config, device):
        """Ground truth observations should be binary (0 or 1) per channel."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        obs = env.reset()

        # All values should be 0 or 1
        assert obs.min().item() >= 0.0, "Observation values should be >= 0"
        assert obs.max().item() <= 1.0, "Observation values should be <= 1"

        # Check that values are binary
        unique_vals = torch.unique(obs)
        assert len(unique_vals) <= 2, f"Expected binary values, got {unique_vals}"

    def test_ground_truth_channels_match_cell_types(self, config, device):
        """Channel 0=food, Channel 1=poison in 2-channel encoding."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        # Manually place known cells in agent's view
        env.agent_x[0] = 10
        env.agent_y[0] = 10

        # Clear area and place specific items
        env.grids[0, 5:16, 5:16] = CellType.EMPTY.value
        env.grids[0, 10, 12] = CellType.FOOD.value  # 2 cells right of agent
        env.grids[0, 10, 8] = CellType.POISON.value  # 2 cells left of agent

        obs = env._get_observations()

        r = spec.view_size // 2

        # Check food position: channel 0 = 1, channel 1 = 0
        assert obs[0, 0, r, r + 2].item() == 1.0, "Food not detected in channel 0"
        assert obs[0, 1, r, r + 2].item() == 0.0, "Food shouldn't be in channel 1"

        # Check poison position: channel 0 = 0, channel 1 = 1
        assert obs[0, 0, r, r - 2].item() == 0.0, "Poison shouldn't be in channel 0"
        assert obs[0, 1, r, r - 2].item() == 1.0, "Poison not detected in channel 1"

    def test_ground_truth_center_is_blank(self, config, device):
        """Center of observation (agent's position) should be zeroed."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)

        obs = env.reset()

        r = spec.view_size // 2

        for env_id in range(4):
            center_values = obs[env_id, :, r, r]
            assert center_values.sum().item() == 0.0, \
                f"Center should be blank, got {center_values}"


class TestProxyEncoding:
    """Tests for proxy mode observation encoding.

    Proxy mode uses 2-channel encoding with identical interestingness values:
    - Both channels contain the same interestingness value for each cell
    - Agent cannot distinguish food from poison, only sees magnitude
    - Empty: [0, 0], Food: [i, i], Poison: [i, i] where i = interestingness
    """

    def test_proxy_has_interestingness_values(self, config, device):
        """Proxy mode should have interestingness values in both channels."""
        spec = ObservationSpec.for_mode('proxy', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        obs = env.reset()

        # Should have 2 channels
        assert obs.shape[1] == 2, f"Expected 2 channels, got {obs.shape[1]}"

        # Should have non-zero values (food/poison have interestingness)
        assert obs.max().item() > 0, \
            "Proxy observations should contain interestingness values"

    def test_proxy_both_channels_identical(self, config, device):
        """Both proxy channels should contain identical values."""
        spec = ObservationSpec.for_mode('proxy', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        # Place food in known location
        env.agent_x[0] = 10
        env.agent_y[0] = 10
        env.grids[0, 5:16, 5:16] = CellType.EMPTY.value
        env.grids[0, 10, 12] = CellType.FOOD.value
        env.grids[0, 10, 8] = CellType.POISON.value

        obs = env._get_observations()

        # Both channels should be identical
        assert torch.allclose(obs[0, 0], obs[0, 1]), \
            "Proxy channels should be identical (same interestingness in both)"

    def test_proxy_food_interestingness(self, config, device):
        """Food should have correct interestingness in proxy mode."""
        spec = ObservationSpec.for_mode('proxy', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        # Place food in known location
        env.agent_x[0] = 10
        env.agent_y[0] = 10
        env.grids[0, 5:16, 5:16] = CellType.EMPTY.value
        env.grids[0, 10, 12] = CellType.FOOD.value

        obs = env._get_observations()

        r = spec.view_size // 2
        # Check channel 0 (both channels have same value)
        food_interestingness = obs[0, 0, r, r + 2].item()

        assert abs(food_interestingness - CellType.FOOD.interestingness) < 0.001, \
            f"Food interestingness wrong: {food_interestingness} != {CellType.FOOD.interestingness}"

    def test_proxy_poison_interestingness(self, config, device):
        """Poison should have correct interestingness in proxy mode."""
        spec = ObservationSpec.for_mode('proxy', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        # Place poison in known location
        env.agent_x[0] = 10
        env.agent_y[0] = 10
        env.grids[0, 5:16, 5:16] = CellType.EMPTY.value
        env.grids[0, 10, 12] = CellType.POISON.value

        obs = env._get_observations()

        r = spec.view_size // 2
        # Check channel 0 (both channels have same value)
        poison_interestingness = obs[0, 0, r, r + 2].item()

        assert abs(poison_interestingness - CellType.POISON.interestingness) < 0.001, \
            f"Poison interestingness wrong: {poison_interestingness} != {CellType.POISON.interestingness}"

    def test_proxy_metric_incomplete_not_adversarial(self, config, device):
        """
        The proxy metric is INCOMPLETE, not adversarial.

        Food is MORE interesting than poison - the metric isn't designed to cause
        harm. But poison still has positive interestingness, so agents consume it.

        Why this is a more honest Goodhart demonstration:
        - Food is MORE interesting than poison - the metric isn't designed to harm
        - But poison still has positive interestingness (0.5 > 0)
        - Agents are rewarded for consuming interesting things
        - The metric doesn't encode harm, so agents eat poison anyway
        - Result: correct optimization, catastrophic outcomes
        """
        food_i = CellType.FOOD.interestingness
        poison_i = CellType.POISON.interestingness

        # Food should be more interesting than poison (the honest setup)
        assert food_i > poison_i, (
            f"Food should be MORE interesting than poison for honest Goodhart demo. "
            f"Got food={food_i}, poison={poison_i}"
        )

        # BOTH should be non-zero - this is the key!
        # Poison is "interesting enough" to consume, even if food is more interesting
        assert food_i > 0, f"Food should have positive interestingness: {food_i}"
        assert poison_i > 0, f"Poison should have positive interestingness: {poison_i}"

        # Empty cells should be boring (so agents move toward items)
        empty_i = CellType.EMPTY.interestingness
        assert empty_i < food_i, f"Empty should be less interesting than food"
        assert empty_i < poison_i, f"Empty should be less interesting than poison"

    def test_proxy_empty_has_zero_interestingness(self, config, device):
        """Empty cells should have zero interestingness."""
        assert CellType.EMPTY.interestingness == 0.0, \
            f"Empty should have 0 interestingness: {CellType.EMPTY.interestingness}"


class TestViewExtraction:
    """Tests for view extraction at various positions."""

    def test_view_centered_on_agent(self, config, device):
        """Agent should be at center of observation view."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        # Place agent and mark surrounding cells distinctly
        env.agent_x[0] = 10
        env.agent_y[0] = 10

        # Clear area
        env.grids[0, :, :] = CellType.EMPTY.value

        # Place food at specific relative positions
        env.grids[0, 10, 11] = CellType.FOOD.value  # +1 in x
        env.grids[0, 10, 9] = CellType.FOOD.value   # -1 in x
        env.grids[0, 11, 10] = CellType.FOOD.value  # +1 in y
        env.grids[0, 9, 10] = CellType.FOOD.value   # -1 in y

        obs = env._get_observations()

        r = spec.view_size // 2
        # Channel 0 is food in 2-channel encoding
        food_channel = 0

        # Check positions relative to center
        assert obs[0, food_channel, r, r + 1].item() == 1.0, "Food at +x not at expected position"
        assert obs[0, food_channel, r, r - 1].item() == 1.0, "Food at -x not at expected position"
        assert obs[0, food_channel, r + 1, r].item() == 1.0, "Food at +y not at expected position"
        assert obs[0, food_channel, r - 1, r].item() == 1.0, "Food at -y not at expected position"

    def test_view_at_edge_with_wrapping(self, config, device):
        """View at grid edge should wrap (toroidal world)."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        # Clear grid and place agent at edge
        env.grids[0, :, :] = CellType.EMPTY.value
        env.agent_x[0] = 0
        env.agent_y[0] = 0

        # Place food on opposite edge (should be visible due to wrapping)
        env.grids[0, 0, config['GRID_WIDTH'] - 1] = CellType.FOOD.value

        obs = env._get_observations()

        r = spec.view_size // 2
        # Channel 0 is food in 2-channel encoding
        food_channel = 0

        # Food at (0, width-1) should appear at (r, r-1) in view
        # (one cell left of agent in wrapped space)
        assert obs[0, food_channel, r, r - 1].item() == 1.0, \
            "Wrapped food not visible at expected position"


class TestMultiEnvConsistency:
    """Tests for consistency across multiple environments."""

    def test_observations_independent_per_env(self, config, device):
        """Each environment should have independent observations."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)

        # Reset and get observations
        obs = env.reset()

        # Observations should generally be different (different spawn positions)
        # Check that at least some envs have different observations
        diffs = []
        for i in range(4):
            for j in range(i + 1, 4):
                diff = (obs[i] - obs[j]).abs().sum().item()
                diffs.append(diff)

        # At least some pairs should be different
        assert max(diffs) > 0, "All environments have identical observations (suspicious)"

    def test_observations_on_correct_device(self, config, device):
        """Observations should be on the specified device."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)

        obs = env.reset()

        # Compare device type, not exact index
        assert obs.device.type == device.type, \
            f"Observations on wrong device type: {obs.device.type} != {device.type}"

    def test_observation_dtype_is_float32(self, config, device):
        """Observations should be float32 for neural network input."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)

        obs = env.reset()

        assert obs.dtype == torch.float32, f"Observations should be float32, got {obs.dtype}"
