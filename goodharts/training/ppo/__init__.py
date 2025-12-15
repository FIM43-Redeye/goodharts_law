"""
PPO Training Module.

Modular implementation of Proximal Policy Optimization for training
learned behaviors. Designed for extensibility to support future
predator/prey and multi-agent scenarios.

Usage:
    from goodharts.training.ppo import PPOTrainer, PPOConfig
    
    config = PPOConfig(mode='ground_truth', total_timesteps=100_000)
    trainer = PPOTrainer(config)
    results = trainer.train()
"""

from .models import ValueHead, Profiler
from goodharts.modes import RewardComputer
from .algorithms import compute_gae, ppo_update
from .trainer import PPOTrainer, PPOConfig

__all__ = [
    # Models
    'ValueHead',
    'Profiler',
    # Rewards
    'RewardComputer',
    # Algorithms
    'compute_gae',
    'ppo_update',
    # Trainer
    'PPOTrainer',
    'PPOConfig',
]
