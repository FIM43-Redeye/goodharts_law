#!/usr/bin/env python3
"""
PPO Training CLI.

This is a thin wrapper around the PPOTrainer class.
All core logic is in the goodharts.training.ppo module.

Usage:
    python -m goodharts.training.train_ppo --mode ground_truth --timesteps 100000
    python -m goodharts.training.train_ppo --mode all --dashboard
"""
# CRITICAL: Set cache dir BEFORE any torch imports
# torch.compile reads this env var at import time, not at compile time
import os
_CACHE_DIR = os.path.expanduser("~/.cache/torch_inductor")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _CACHE_DIR)

import argparse
import threading

from goodharts.modes import get_all_mode_names
from goodharts.configs.default_config import get_config
from goodharts.config import get_training_config
from goodharts.behaviors.brains import get_brain_names
from goodharts.training.ppo import PPOTrainer, PPOConfig


# Global synchronization for stop signal (multi-threaded training)
_TRAINING_LOCK = threading.Lock()
_TRAINING_COUNTER = 0


def train_ppo(
    mode: str = 'ground_truth',
    dashboard = None,
    **overrides
) -> dict:
    """
    Train a PPO agent.
    
    Uses config.toml for defaults; explicit kwargs override them.
    This is a convenience wrapper around PPOTrainer.
    
    Args:
        mode: Training mode (ground_truth, proxy, etc.)
        dashboard: Optional training dashboard
        **overrides: Any PPOConfig fields to override
        
    Returns:
        Summary dict with training results
    """
    global _TRAINING_COUNTER
    
    with _TRAINING_LOCK:
        _TRAINING_COUNTER += 1
    
    try:
        # Use factory method - config.toml provides defaults, overrides take precedence
        config = PPOConfig.from_config(mode=mode, **overrides)
        
        trainer = PPOTrainer(config, dashboard=dashboard)
        return trainer.train()
        
    finally:
        with _TRAINING_LOCK:
            _TRAINING_COUNTER -= 1
        # Note: Stop signal cleanup is handled by the sequential training loops,
        # NOT here. This allows the signal to persist across trainer boundaries.


def main():
    """Main CLI entry point."""
    # Archive existing logs to keep current run clean
    from goodharts.training.train_log import TrainingLogger
    TrainingLogger.archive_existing_logs()

    config = get_config()
    train_cfg = get_training_config()
    all_modes = get_all_mode_names(config)
    brain_names = get_brain_names()

    # Get defaults from config for help text
    default_envs = train_cfg.get('n_envs', 64)
    default_minibatches = train_cfg.get('n_minibatches', 4)
    default_brain = train_cfg.get('brain_type', 'base_cnn')

    parser = argparse.ArgumentParser(
        description='PPO training for Goodhart agents.',
        epilog='Config precedence: CLI args > config.toml > config.default.toml',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Training options
    training = parser.add_argument_group('Training')
    training.add_argument('-m', '--mode', default='ground_truth', choices=all_modes + ['all'],
                          metavar='MODE',
                          help=f'Training mode: {", ".join(all_modes)}, or "all" [default: ground_truth]')
    training.add_argument('-t', '--timesteps', type=int, default=None, metavar='N',
                          help='Total environment steps')
    training.add_argument('-u', '--updates', type=int, default=None, metavar='N',
                          help='PPO updates (alternative to --timesteps)')
    training.add_argument('-s', '--sequential', action='store_true',
                          help='Train modes sequentially (saves VRAM)')

    # Performance tuning
    perf = parser.add_argument_group('Performance')
    perf.add_argument('-e', '--n-envs', type=int, default=None, metavar='N',
                      help=f'Parallel environments [default: {default_envs}]')
    perf.add_argument('--minibatches', type=int, default=None, metavar='N',
                      help=f'Minibatches per epoch [default: {default_minibatches}]')
    perf.add_argument('--no-amp', action='store_true',
                      help='Disable mixed precision (AMP)')
    perf.add_argument('--no-warmup', action='store_true',
                      help='Skip JIT warmup (faster startup, slower training)')
    perf.add_argument('--no-profile', action='store_true',
                      help='Disable profiling (faster for production)')

    # Monitoring
    monitor = parser.add_argument_group('Monitoring')
    monitor.add_argument('-d', '--dashboard', action='store_true',
                         help='Show live training dashboard')
    monitor.add_argument('-tb', '--tensorboard', action='store_true',
                         help='Enable TensorBoard logging')
    monitor.add_argument('-v', '--verbose', action='store_true', dest='hyper_verbose',
                         help='Debug mode: print at every major step')

    # Experiment settings
    experiment = parser.add_argument_group('Experiment')
    experiment.add_argument('--brain', default=None, choices=brain_names, metavar='ARCH',
                            help=f'Neural network architecture [default: {default_brain}]')
    experiment.add_argument('--entropy', type=float, default=None, metavar='COEF',
                            help='Entropy coefficient for exploration')
    experiment.add_argument('--seed', type=int, default=None, metavar='N',
                            help='Random seed for reproducibility')
    experiment.add_argument('--deterministic', action='store_true',
                            help='Full determinism (slower, for debugging)')

    # Utility
    utility = parser.add_argument_group('Utility')
    utility.add_argument('-b', '--benchmark', action='store_true',
                         help='Benchmark mode: measure throughput, discard model')
    utility.add_argument('--clean-cache', action='store_true',
                         help='Delete compilation cache before starting')

    args = parser.parse_args()
    
    # Build overrides dict from CLI args (only non-None values)
    overrides = {}
    
    if args.brain:
        overrides['brain_type'] = args.brain
    if args.n_envs:
        overrides['n_envs'] = args.n_envs
    if args.entropy:
        overrides['entropy_coef'] = args.entropy
    if args.minibatches:
        overrides['n_minibatches'] = args.minibatches
    if args.tensorboard:
        overrides['tensorboard'] = True
    if args.hyper_verbose:
        overrides['hyper_verbose'] = True
    if args.clean_cache:
        overrides['clean_cache'] = True
    if args.benchmark:
        overrides['benchmark_mode'] = True
        overrides['validation_interval'] = 0  # Skip validation in benchmarks
        # Default to 50k steps if not specified
        if args.timesteps is None and args.updates is None:
            overrides['total_timesteps'] = 50_000

    # Handle timesteps (--updates is a convenience conversion)
    n_envs = args.n_envs or train_cfg.get('n_envs', 64)
    steps_per_env = train_cfg.get('steps_per_env', 128)
    if args.updates is not None:
        overrides['total_timesteps'] = args.updates * n_envs * steps_per_env
        print(f"   {args.updates} updates x {n_envs} envs x {steps_per_env} steps = {overrides['total_timesteps']:,} timesteps")
    elif args.timesteps is not None:
        overrides['total_timesteps'] = args.timesteps
    
    # Handle AMP
    if args.no_amp:
        overrides['use_amp'] = False
    
    # Handle --no-warmup
    if args.no_warmup:
        overrides['compile_models'] = False
        overrides['skip_warmup'] = True
        os.environ['GOODHARTS_CUDNN_BENCHMARK'] = '0'
        import torch
        torch.backends.cudnn.benchmark = False
        print("   Warmup disabled: torch.compile and cuDNN benchmark OFF")
    
    # Handle --no-profile (critical for high steps_per_env configs)
    if args.no_profile:
        overrides['profile_enabled'] = False

    # Handle seed and deterministic mode
    if args.seed is not None:
        overrides['seed'] = args.seed
    if args.deterministic:
        overrides['deterministic'] = True

    modes_to_train = all_modes if args.mode == 'all' else [args.mode]
    
    # Training execution
    if args.dashboard:
        _run_with_dashboard(modes_to_train, overrides, args.sequential)
    else:
        _run_without_dashboard(modes_to_train, overrides, args.sequential)
 
 
def _run_with_dashboard(modes_to_train: list, overrides: dict, sequential: bool):
    """Run training with live dashboard in separate process."""
    from goodharts.training.train_dashboard import create_dashboard_process
    from goodharts.behaviors.action_space import num_actions

    print(f"\nTraining with dashboard (process-isolated)")

    # Create dashboard in separate process
    dashboard = create_dashboard_process(modes_to_train, n_actions=num_actions(1))
    dashboard.start()

    try:
        if sequential:
            # Sequential: run training in main thread (better GPU perf)
            for mode in modes_to_train:
                # Check if stop was requested or dashboard closed
                if dashboard.should_stop():
                    print(f"[Sequential] Stop signal detected, skipping remaining modes")
                    break
                if not dashboard.is_alive():
                    print("[Sequential] Dashboard closed, stopping training")
                    break
                    
                train_ppo(
                    mode=mode,
                    dashboard=dashboard,
                    output_path=f'models/ppo_{mode}.pth',
                    log_to_file=True,
                    **overrides
                )
        else:
            # Parallel training (first trainer warms up, others skip via global flag)
            threads = []
            for mode in modes_to_train:
                t = threading.Thread(
                    target=train_ppo,
                    kwargs={
                        'mode': mode,
                        'dashboard': dashboard,
                        'output_path': f'models/ppo_{mode}.pth',
                        'log_to_file': True,
                        **overrides
                    },
                    daemon=True
                )
                threads.append(t)
                t.start()

            # Wait for all threads to complete
            for t in threads:
                t.join()
                
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    
    # Don't auto-close dashboard - let user take screenshots
    # Dashboard will close when user closes the window
    if dashboard.is_alive():
        print("\n[Dashboard] Training complete. Close the dashboard window when done.")


def _run_without_dashboard(modes_to_train: list, overrides: dict, sequential: bool):
    """Run training without dashboard."""
    if len(modes_to_train) > 1 and not sequential:
        # Parallel training (first trainer warms up, others skip via global flag)
        print(f"\nParallel training: {len(modes_to_train)} modes")

        threads = []
        for mode in modes_to_train:
            t = threading.Thread(
                target=train_ppo,
                kwargs={
                    'mode': mode,
                    'output_path': f'models/ppo_{mode}.pth',
                    'log_to_file': True,
                    **overrides
                }
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
    else:
        # Sequential training
        if len(modes_to_train) > 1:
            print(f"\nSequential training: {len(modes_to_train)} modes")
        for mode in modes_to_train:
            # Check if stop was requested before starting next run
            if os.path.exists('.training_stop_signal'):
                print(f"[Sequential] Stop signal detected, skipping remaining modes")
                try:
                    os.remove('.training_stop_signal')
                except OSError:
                    pass
                break
            train_ppo(
                mode=mode,
                output_path=f'models/ppo_{mode}.pth',
                log_to_file=len(modes_to_train) > 1,
                **overrides
            )


if __name__ == '__main__':
    main()

