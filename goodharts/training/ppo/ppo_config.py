"""
PPO training configuration.

Provides PPOConfig dataclass with factory method to load from config.toml.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOConfig:
    """
    Configuration for PPO training.

    Use PPOConfig.from_config() to load defaults from config.toml,
    with CLI arguments as optional overrides.
    """
    mode: str = 'ground_truth'
    brain_type: str = 'base_cnn'
    value_head_type: str = 'popart'  # 'simple' or 'popart'
    action_space_type: str = 'discrete_grid'
    max_move_distance: int = 1
    n_envs: int = 192  # Larger batches = less GPU burstiness
    total_timesteps: int = 100_000
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    k_epochs: int = 4
    steps_per_env: int = 128
    n_minibatches: int = 1
    value_coef: float = 0.5

    # Entropy scheduling: prevents premature collapse while allowing full convergence
    entropy_initial: float = 0.1      # Strong exploration early
    entropy_final: float = 0.001      # Minimal when near-optimal
    entropy_decay_fraction: float = 0.7  # Decay over 70% of training
    entropy_floor: float = 0.5        # Min entropy during learning phase
    entropy_floor_penalty: float = 0.05  # Penalty coefficient for floor violation

    # Learning rate decay: reduces LR over training for fine-tuning
    lr_decay: bool = False            # Enable LR decay
    lr_final: float = 3e-5            # Final LR (10x lower than initial)

    # Clip decay: tightens trust region over training to reduce late oscillation
    eps_clip_final: float = 0.1       # Final clip (tighter than initial 0.2)
    output_path: str = 'models/ppo_agent.pth'
    log_to_file: bool = True
    log_dir: str = 'generated/logs'
    use_amp: bool = False
    compile_models: bool = True
    compile_mode: str = 'max-autotune'  # reduce-overhead, max-autotune, max-autotune-no-cudagraphs
    compile_env: bool = True  # torch.compile the environment step for better GPU utilization
    # TensorBoard is always enabled (unified logging)
    skip_warmup: bool = False
    use_torch_env: bool = True
    hyper_verbose: bool = False
    clean_cache: bool = False
    profile_enabled: bool = True  # Disable with --no-profile for production
    benchmark_mode: bool = False  # Skip saving, just measure throughput
    gpu_log_interval_ms: int = 0  # GPU utilization logging interval (0 = disabled)
    cuda_graphs: bool = False     # Use CUDA/HIP graphs for inference (experimental)

    # Reproducibility
    seed: Optional[int] = 42  # Default 42 for reproducibility; None = random

    # Validation episodes (periodic eval without exploration)
    validation_interval: int = 0     # Every N updates (0 = disabled)
    validation_episodes: int = 16     # Episodes per validation
    validation_mode: str = "training" # "training" or "fixed"
    validation_food: int = 100        # Fixed mode: food count
    validation_poison: int = 50       # Fixed mode: poison count

    # Privileged critic: value function sees episode density (food/poison counts)
    # This helps explain variance from episode difficulty without affecting policy
    privileged_critic: bool = True   # Enable density info for value head

    # PopArt configuration
    # beta_min: minimum EMA decay rate (smaller = slower adaptation, more stable)
    # rescale_weights: when True, rescale fc weights when stats change to preserve outputs
    popart_beta_min: float = 0.001   # 10x slower than original 0.01 default
    popart_rescale_weights: bool = True

    @classmethod
    def from_config(cls, mode: str = 'ground_truth', **overrides) -> 'PPOConfig':
        """
        Create PPOConfig from config.toml with optional CLI overrides.

        TOML provides all defaults; explicit kwargs override them.
        Missing TOML keys will raise KeyError - no silent fallbacks.

        Args:
            mode: Training mode (ground_truth, proxy, etc.)
            **overrides: Any PPOConfig fields to override

        Returns:
            PPOConfig with values from config file + overrides

        Raises:
            KeyError: If required config keys are missing from TOML
        """
        from goodharts.config import get_agent_config, get_training_config

        train_cfg = get_training_config()
        agent_cfg = get_agent_config()

        # Build config from TOML - no fallbacks, missing keys will crash
        config_values = {
            'mode': mode,
            'brain_type': train_cfg['brain_type'],
            'value_head_type': train_cfg['value_head_type'],
            'action_space_type': train_cfg['action_space_type'],
            'max_move_distance': agent_cfg['max_move_distance'],
            'n_envs': train_cfg['n_envs'],
            'lr': train_cfg['learning_rate'],
            'gamma': train_cfg['gamma'],
            'gae_lambda': train_cfg['gae_lambda'],
            'eps_clip': train_cfg['eps_clip'],
            'k_epochs': train_cfg['k_epochs'],
            'steps_per_env': train_cfg['steps_per_env'],
            'n_minibatches': train_cfg['n_minibatches'],
            'value_coef': train_cfg['value_coef'],
            # Entropy scheduling
            'entropy_initial': train_cfg['entropy_initial'],
            'entropy_final': train_cfg['entropy_final'],
            'entropy_decay_fraction': train_cfg['entropy_decay_fraction'],
            'entropy_floor': train_cfg['entropy_floor'],
            'entropy_floor_penalty': train_cfg['entropy_floor_penalty'],
            # LR decay
            'lr_decay': train_cfg.get('lr_decay', False),
            'lr_final': train_cfg.get('lr_final', 3e-5),
            # Clip decay
            'eps_clip_final': train_cfg.get('eps_clip_final', 0.1),
            'use_amp': train_cfg['use_amp'],
            'compile_models': train_cfg['compile_models'],
            'compile_mode': train_cfg.get('compile_mode', 'max-autotune'),
            'compile_env': train_cfg.get('compile_env', True),  # Compile env.step() by default
            # Validation
            'validation_interval': train_cfg['validation_interval'],
            'validation_episodes': train_cfg['validation_episodes'],
            'validation_mode': train_cfg['validation_mode'],
            'validation_food': train_cfg['validation_food'],
            'validation_poison': train_cfg['validation_poison'],
            # PopArt options
            'popart_beta_min': train_cfg.get('popart_beta_min', 0.001),
            'popart_rescale_weights': train_cfg.get('popart_rescale_weights', True),
        }

        # Apply overrides (CLI args take precedence)
        for key, value in overrides.items():
            if value is not None:  # Only override if explicitly set
                config_values[key] = value

        return cls(**config_values)
