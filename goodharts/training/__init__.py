"""
Training pipeline for learned behaviors.

Provides PPO training with live visualization, structured logging, and model
verification tools. The modular design supports future multi-agent extensions.

Submodules
----------
ppo
    Proximal Policy Optimization implementation (PPOTrainer, algorithms).
verification
    Model fitness testing (survival tests, directional accuracy).

Key CLI Commands
----------------
Train ground truth agent::

    python -m goodharts.training.train_ppo --mode ground_truth --timesteps 100000

Train all modes with dashboard::

    python -m goodharts.training.train_ppo --mode all --dashboard

Verify trained models::

    python -m goodharts.training.verification --steps 500 --verbose

See training/README.md for detailed documentation.
"""
