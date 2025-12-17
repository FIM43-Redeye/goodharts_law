"""Tests for PPO algorithm implementations.

Tests the core RL algorithms (GAE, PPO update) independently of the training loop.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn

from goodharts.training.ppo.algorithms import compute_gae, ppo_update


class SimplePolicy(nn.Module):
    """Minimal policy network for testing - matches Brain interface."""
    
    def __init__(self, obs_dim=4, n_actions=8, hidden=32):
        super().__init__()
        self.fc = nn.Linear(obs_dim, hidden)
        self.fc_out = nn.Linear(hidden, n_actions)
        self._features = None
    
    def forward(self, x):
        self._features = torch.relu(self.fc(x))
        return self.fc_out(self._features)
    
    def get_features(self, x):
        self._features = torch.relu(self.fc(x))
        return self._features
    
    def logits_from_features(self, features):
        """Matches Brain interface - compute logits from pre-computed features."""
        return self.fc_out(features)


class SimpleValueHead(nn.Module):
    """Minimal value head for testing."""
    
    def __init__(self, hidden=32):
        super().__init__()
        self.fc = nn.Linear(hidden, 1)
    
    def forward(self, features):
        return self.fc(features)


class TestComputeGAE:
    """Tests for Generalized Advantage Estimation."""
    
    def test_gae_shapes(self):
        """GAE should return correct shapes."""
        n_steps = 10
        n_envs = 4
        device = torch.device('cpu')
        
        rewards = [torch.ones(n_envs, device=device) for _ in range(n_steps)]
        values = [torch.zeros(n_envs, device=device) for _ in range(n_steps)]
        dones = [torch.zeros(n_envs, device=device) for _ in range(n_steps)]
        next_value = torch.zeros(n_envs, device=device)
        
        advantages, returns = compute_gae(rewards, values, dones, next_value)
        
        assert advantages.shape == (n_steps, n_envs)
        assert returns.shape == (n_steps, n_envs)
    
    def test_gae_positive_rewards_give_positive_advantage(self):
        """Positive rewards with zero values should give positive advantages."""
        n_steps = 5
        n_envs = 2
        device = torch.device('cpu')
        
        rewards = [torch.ones(n_envs, device=device) for _ in range(n_steps)]
        values = [torch.zeros(n_envs, device=device) for _ in range(n_steps)]
        dones = [torch.zeros(n_envs, device=device) for _ in range(n_steps)]
        next_value = torch.zeros(n_envs, device=device)
        
        advantages, returns = compute_gae(rewards, values, dones, next_value)
        
        # All advantages should be positive (rewards > values everywhere)
        assert torch.all(advantages > 0), "Positive rewards should yield positive advantages"
    
    def test_gae_done_resets_advantage(self):
        """Done flag should reset advantage accumulation."""
        n_steps = 4
        n_envs = 1
        device = torch.device('cpu')
        
        # Reward pattern: [1, 1, 1, 1] but done after step 1
        rewards = [torch.tensor([1.0], device=device) for _ in range(n_steps)]
        values = [torch.tensor([0.0], device=device) for _ in range(n_steps)]
        dones = [torch.tensor([0.0], device=device), torch.tensor([1.0], device=device), 
                 torch.tensor([0.0], device=device), torch.tensor([0.0], device=device)]
        next_value = torch.tensor([0.0], device=device)
        
        advantages, returns = compute_gae(rewards, values, dones, next_value, gamma=0.99)
        
        # After done, advantage for step 1 should only consider immediate reward
        assert advantages[1, 0] <= advantages[0, 0], \
            "Advantage after done should not accumulate from future"
    
    def test_gae_returns_equal_advantages_plus_values(self):
        """Returns should equal advantages plus values."""
        n_steps = 5
        n_envs = 2
        device = torch.device('cpu')
        
        rewards = [torch.randn(n_envs, device=device) for _ in range(n_steps)]
        values = [torch.randn(n_envs, device=device) for _ in range(n_steps)]
        dones = [torch.zeros(n_envs, device=device) for _ in range(n_steps)]
        next_value = torch.randn(n_envs, device=device)
        
        advantages, returns = compute_gae(rewards, values, dones, next_value)
        
        expected_returns = advantages + torch.stack(values)
        assert torch.allclose(returns, expected_returns, rtol=1e-5)


class TestPPOUpdate:
    """Tests for PPO clipped objective update."""
    
    @pytest.fixture
    def setup(self):
        """Create policy, value head, and optimizer for testing."""
        policy = SimplePolicy(obs_dim=4, n_actions=8)
        value_head = SimpleValueHead(hidden=32)
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_head.parameters()),
            lr=1e-3
        )
        return policy, value_head, optimizer
    
    def test_ppo_update_returns_losses(self, setup):
        """PPO update should return loss metrics."""
        policy, value_head, optimizer = setup
        device = torch.device('cpu')
        
        batch_size = 32
        states = torch.randn(batch_size, 4, device=device)
        actions = torch.randint(0, 8, (batch_size,), device=device)
        log_probs = torch.randn(batch_size, device=device)
        returns = torch.randn(batch_size, device=device)
        advantages = torch.randn(batch_size, device=device)
        old_values = torch.randn(batch_size, device=device)
        
        p_loss, v_loss, entropy, ev = ppo_update(
            policy, value_head, optimizer,
            states, actions, log_probs, returns, advantages, old_values,
            device=device,
            k_epochs=1,
            n_minibatches=1
        )
        
        # ppo_update now returns tensors (for async logging)
        # .item() calls happen in AsyncLogger background thread
        assert isinstance(p_loss, torch.Tensor)
        assert isinstance(v_loss, torch.Tensor)
        assert isinstance(entropy, torch.Tensor)
        assert isinstance(ev, torch.Tensor)
    
    def test_ppo_update_modifies_weights(self, setup):
        """PPO update should modify network weights."""
        policy, value_head, optimizer = setup
        device = torch.device('cpu')
        
        # Record initial weights
        initial_weights = policy.fc.weight.clone().detach()
        
        batch_size = 64
        states = torch.randn(batch_size, 4, device=device)
        actions = torch.randint(0, 8, (batch_size,), device=device)
        log_probs = torch.zeros(batch_size, device=device)  # Force ratio = 1
        returns = torch.ones(batch_size, device=device)
        advantages = torch.ones(batch_size, device=device)  # Positive advantage
        old_values = torch.zeros(batch_size, device=device)
        
        ppo_update(
            policy, value_head, optimizer,
            states, actions, log_probs, returns, advantages, old_values,
            device=device,
            k_epochs=2,
            n_minibatches=1
        )
        
        # Weights should have changed
        assert not torch.allclose(policy.fc.weight, initial_weights), \
            "PPO update should modify policy weights"
    
    def test_ppo_update_entropy_is_nonnegative(self, setup):
        """Entropy should always be non-negative."""
        policy, value_head, optimizer = setup
        device = torch.device('cpu')
        
        batch_size = 32
        states = torch.randn(batch_size, 4, device=device)
        actions = torch.randint(0, 8, (batch_size,), device=device)
        log_probs = torch.randn(batch_size, device=device)
        returns = torch.randn(batch_size, device=device)
        advantages = torch.randn(batch_size, device=device)
        old_values = torch.randn(batch_size, device=device)
        
        _, _, entropy, _ = ppo_update(
            policy, value_head, optimizer,
            states, actions, log_probs, returns, advantages, old_values,
            device=device,
            k_epochs=1,
            n_minibatches=1
        )
        
        assert entropy >= 0, "Entropy should be non-negative"
