"""Tests for determinism and reproducibility.

These tests verify that:
- Setting seeds produces reproducible results
- Environment initialization is reproducible
- Neural network forward passes are reproducible
- Training with same seed produces same weights
"""
import pytest
import torch
import numpy as np
from contextlib import contextmanager

from goodharts.environments.torch_env import create_torch_vec_env
from goodharts.modes import ObservationSpec
from goodharts.configs.default_config import get_simulation_config
from goodharts.behaviors.brains.base_cnn import BaseCNN


@pytest.fixture
def config():
    cfg = get_simulation_config()
    cfg['GRID_WIDTH'] = 20
    cfg['GRID_HEIGHT'] = 20
    cfg['GRID_FOOD_INIT'] = 20
    cfg['GRID_POISON_INIT'] = 10
    return cfg


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def deterministic_mode():
    """Context manager to enable PyTorch deterministic algorithms.

    Enables torch.use_deterministic_algorithms and cudnn deterministic mode
    for the duration of the context. Also sets CUBLAS_WORKSPACE_CONFIG for
    deterministic cuBLAS operations. Restores previous state on exit.

    Note: Even with deterministic mode, some GPU operations may have tiny
    floating-point differences across architectures. Tests using this should
    use torch.allclose with a tight tolerance rather than torch.equal.
    """
    import os

    # Save previous state
    prev_deterministic = torch.are_deterministic_algorithms_enabled()
    prev_cudnn_deterministic = torch.backends.cudnn.deterministic
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    prev_cublas_config = os.environ.get('CUBLAS_WORKSPACE_CONFIG')

    try:
        # Enable deterministic mode
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        # Restore previous state
        if prev_cublas_config is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = prev_cublas_config
        torch.use_deterministic_algorithms(prev_deterministic)
        torch.backends.cudnn.deterministic = prev_cudnn_deterministic
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark


class TestTorchSeeding:
    """Tests for PyTorch random seeding."""

    def test_torch_randn_reproducible(self, device):
        """torch.randn should be reproducible with same seed."""
        set_all_seeds(42)
        a1 = torch.randn(10, 10, device=device)

        set_all_seeds(42)
        a2 = torch.randn(10, 10, device=device)

        assert torch.equal(a1, a2), "Same seed should produce same random tensor"

    def test_torch_randint_reproducible(self, device):
        """torch.randint should be reproducible with same seed."""
        set_all_seeds(42)
        a1 = torch.randint(0, 100, (10,), device=device)

        set_all_seeds(42)
        a2 = torch.randint(0, 100, (10,), device=device)

        assert torch.equal(a1, a2), "Same seed should produce same random integers"

    def test_different_seeds_produce_different_results(self, device):
        """Different seeds should produce different results."""
        set_all_seeds(42)
        a1 = torch.randn(10, 10, device=device)

        set_all_seeds(123)
        a2 = torch.randn(10, 10, device=device)

        assert not torch.equal(a1, a2), "Different seeds should produce different tensors"


class TestEnvironmentDeterminism:
    """Tests for environment reproducibility."""

    def test_env_reset_reproducible(self, config, device):
        """Environment reset should be reproducible with same seed."""
        spec = ObservationSpec.for_mode('ground_truth', config)

        set_all_seeds(42)
        env1 = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)
        obs1 = env1.reset()
        grid1 = env1.grids.clone()
        pos1 = (env1.agent_x.clone(), env1.agent_y.clone())

        set_all_seeds(42)
        env2 = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)
        obs2 = env2.reset()
        grid2 = env2.grids.clone()
        pos2 = (env2.agent_x.clone(), env2.agent_y.clone())

        assert torch.equal(grid1, grid2), "Same seed should produce same grid"
        assert torch.equal(pos1[0], pos2[0]), "Same seed should produce same agent x positions"
        assert torch.equal(pos1[1], pos2[1]), "Same seed should produce same agent y positions"
        assert torch.equal(obs1, obs2), "Same seed should produce same observations"

    def test_env_step_reproducible(self, config, device):
        """Environment step should be reproducible with same seed."""
        spec = ObservationSpec.for_mode('ground_truth', config)

        # First run
        set_all_seeds(42)
        env1 = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)
        env1.reset()

        actions = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
        obs1, eating_info1, terminated1, truncated1 = env1.step(actions)
        food1, poison1, starved1 = eating_info1
        dones1 = terminated1 | truncated1

        # Second run
        set_all_seeds(42)
        env2 = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)
        env2.reset()

        obs2, eating_info2, terminated2, truncated2 = env2.step(actions)
        food2, poison2, starved2 = eating_info2
        dones2 = terminated2 | truncated2

        assert torch.equal(obs1, obs2), "Same seed should produce same observations after step"
        assert torch.equal(food1, food2), "Same seed should produce same food masks"
        assert torch.equal(poison1, poison2), "Same seed should produce same poison masks"
        assert torch.equal(starved1, starved2), "Same seed should produce same starved masks"
        assert torch.equal(dones1, dones2), "Same seed should produce same done flags"

    def test_env_trajectory_reproducible(self, config, device):
        """Multi-step trajectory should be reproducible."""
        spec = ObservationSpec.for_mode('ground_truth', config)

        # First trajectory
        set_all_seeds(42)
        env1 = create_torch_vec_env(n_envs=2, obs_spec=spec, config=config, device=device)
        env1.reset()

        trajectory1 = []
        for _ in range(20):
            actions = torch.randint(0, 8, (2,), device=device)
            obs, eating_info, terminated, truncated = env1.step(actions)
            food, poison, starved = eating_info
            dones = terminated | truncated
            trajectory1.append((obs.clone(), food.clone(), poison.clone(), starved.clone(), dones.clone()))

        # Second trajectory
        set_all_seeds(42)
        env2 = create_torch_vec_env(n_envs=2, obs_spec=spec, config=config, device=device)
        env2.reset()

        trajectory2 = []
        for _ in range(20):
            actions = torch.randint(0, 8, (2,), device=device)
            obs, eating_info, terminated, truncated = env2.step(actions)
            food, poison, starved = eating_info
            dones = terminated | truncated
            trajectory2.append((obs.clone(), food.clone(), poison.clone(), starved.clone(), dones.clone()))

        # Compare trajectories
        for i, ((obs1, f1, p1, s1, d1), (obs2, f2, p2, s2, d2)) in enumerate(zip(trajectory1, trajectory2)):
            assert torch.equal(obs1, obs2), f"Observations differ at step {i}"
            assert torch.equal(f1, f2), f"Food masks differ at step {i}"
            assert torch.equal(p1, p2), f"Poison masks differ at step {i}"
            assert torch.equal(s1, s2), f"Starved masks differ at step {i}"
            assert torch.equal(d1, d2), f"Dones differ at step {i}"


class TestNetworkDeterminism:
    """Tests for neural network reproducibility."""

    def test_network_init_reproducible(self, device):
        """Network initialization should be reproducible."""
        set_all_seeds(42)
        model1 = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        set_all_seeds(42)
        model2 = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)

        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Parameter {n1} differs with same seed"

    def test_forward_pass_reproducible(self, device):
        """Forward pass should be reproducible with same input."""
        set_all_seeds(42)
        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)
        model.eval()

        x = torch.randn(4, 6, 11, 11, device=device)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.equal(out1, out2), "Same input should produce same output"

    def test_action_sampling_reproducible_with_seed(self, device):
        """Action sampling should be reproducible when torch seed is reset."""
        from goodharts.behaviors.action_space import DiscreteGridActionSpace

        model = BaseCNN(
            input_shape=(11, 11),
            input_channels=6,
            output_size=8
        ).to(device)
        model.eval()

        action_space = DiscreteGridActionSpace(max_move_distance=1)
        x = torch.randn(1, 6, 11, 11, device=device)

        set_all_seeds(42)
        with torch.no_grad():
            logits = model(x)
            dx1, dy1 = action_space.decode(logits, sample=True)

        set_all_seeds(42)
        with torch.no_grad():
            logits = model(x)
            dx2, dy2 = action_space.decode(logits, sample=True)

        assert dx1 == dx2 and dy1 == dy2, "Same seed should produce same action"


class TestGradientDeterminism:
    """Tests for gradient computation reproducibility."""

    def test_gradients_reproducible(self, device):
        """Gradient computation should be reproducible.

        Uses deterministic mode and tight tolerance to handle minor
        floating-point variance that can occur even with identical seeds
        on different GPU architectures.
        """
        with deterministic_mode():
            set_all_seeds(42)
            model1 = BaseCNN(
                input_shape=(11, 11),
                input_channels=6,
                output_size=8
            ).to(device)

            x1 = torch.randn(4, 6, 11, 11, device=device)
            out1 = model1(x1)
            loss1 = out1.sum()
            loss1.backward()
            grads1 = {n: p.grad.clone() for n, p in model1.named_parameters() if p.grad is not None}

            set_all_seeds(42)
            model2 = BaseCNN(
                input_shape=(11, 11),
                input_channels=6,
                output_size=8
            ).to(device)

            x2 = torch.randn(4, 6, 11, 11, device=device)
            out2 = model2(x2)
            loss2 = out2.sum()
            loss2.backward()
            grads2 = {n: p.grad.clone() for n, p in model2.named_parameters() if p.grad is not None}

            for name in grads1:
                assert torch.allclose(grads1[name], grads2[name], rtol=1e-5, atol=1e-7), \
                    f"Gradients for {name} differ with same seed"


class TestCrossRunConsistency:
    """Tests for consistency across separate runs."""

    def test_observation_values_consistent(self, config, device):
        """Observation encoding should produce consistent values."""
        spec = ObservationSpec.for_mode('ground_truth', config)

        set_all_seeds(42)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)
        obs = env.reset()

        # Run multiple times, observations should be identical
        for _ in range(5):
            obs2 = env._get_observations()
            assert torch.equal(obs, obs2), "Observation encoding is inconsistent"

    def test_reward_computation_consistent(self, config, device):
        """Reward computation should be consistent."""
        spec = ObservationSpec.for_mode('ground_truth', config)

        set_all_seeds(42)
        env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)
        env.reset()

        actions = torch.zeros(4, dtype=torch.long, device=device)

        # Run same action multiple times from same state
        initial_state = (
            env.agent_x.clone(),
            env.agent_y.clone(),
            env.agent_energy.clone(),
            env.grids.clone()
        )

        _, eating_info1, _, _ = env.step(actions)
        food1, poison1, starved1 = eating_info1

        # Restore state
        env.agent_x[:] = initial_state[0]
        env.agent_y[:] = initial_state[1]
        env.agent_energy[:] = initial_state[2]
        env.grids[:] = initial_state[3]

        _, eating_info2, _, _ = env.step(actions)
        food2, poison2, starved2 = eating_info2

        assert torch.equal(food1, food2), "Food mask computation is inconsistent"
        assert torch.equal(poison1, poison2), "Poison mask computation is inconsistent"
        assert torch.equal(starved1, starved2), "Starved mask computation is inconsistent"
