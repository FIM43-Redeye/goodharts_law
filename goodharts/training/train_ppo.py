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


def _warmup_compile(brain_type: str, n_envs: int, n_minibatches: int):
    """
    Pre-compile kernels by running a single-threaded warmup.
    This populates the cache so parallel threads can reuse compiled kernels.
    """
    print("\n[Warmup] Pre-compiling kernels for parallel training...")
    
    # Create a minimal config just for warmup
    config = PPOConfig(
        mode='ground_truth',  # Any mode works, they all use the same model
        brain_type=brain_type,
        n_envs=n_envs,
        total_timesteps=0,  # No actual training
        compile_models=True,
        n_minibatches=n_minibatches,
        log_to_file=False,
    )
    
    # Create trainer and just run setup (which includes JIT warmup)
    trainer = PPOTrainer(config)
    trainer._setup()
    
    # Cleanup
    del trainer
    import gc
    gc.collect()
    
    # Force CUDA cache cleanup
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("[Warmup] Pre-compilation complete. Starting parallel training...\n")


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
    # Cleanup stale stop signal
    if os.path.exists('.training_stop_signal'):
        try:
            os.remove('.training_stop_signal')
            print("Removed stale stop signal.")
        except OSError:
            pass

    # Archive existing logs to keep current run clean
    from goodharts.training.train_log import TrainingLogger
    TrainingLogger.archive_existing_logs()

    config = get_config()
    train_cfg = get_training_config()
    all_modes = get_all_mode_names(config)
    brain_names = get_brain_names()
    
    parser = argparse.ArgumentParser(
        description='PPO training for Goodhart agents. Config file provides defaults; CLI args override.',
        epilog='Most settings can be changed in config.toml instead of using CLI args.'
    )
    parser.add_argument('--mode', default='ground_truth', choices=all_modes + ['all'],
                        help='Training mode (or "all" for parallel training)')
    parser.add_argument('--brain', default=None, choices=brain_names,
                        help='Override neural network architecture')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Override total environment steps')
    parser.add_argument('--updates', type=int, default=None,
                        help='Number of PPO updates (alternative to --timesteps)')
    parser.add_argument('--entropy', type=float, default=None,
                        help='Override entropy coefficient')
    parser.add_argument('--dashboard', '-d', action='store_true',
                        help='Show live training dashboard')
    parser.add_argument('--sequential', '-s', action='store_true',
                        help='Train modes sequentially (saves VRAM)')
    parser.add_argument('--n-envs', type=int, default=None,
                        help='Override number of parallel environments')
    parser.add_argument('--use-amp', action='store_true', default=None,
                        help='Force enable automatic mixed precision')
    parser.add_argument('--no-amp', action='store_true',
                        help='Force disable automatic mixed precision')
    parser.add_argument('--minibatches', type=int, default=None,
                        help='Override minibatches per epoch')
    parser.add_argument('--no-warmup', action='store_true',
                        help='Skip warmup (faster startup but slower training)')
    parser.add_argument('--no-profile', action='store_true',
                        help='Disable profiling (removes GPU sync overhead, faster for production)')
    parser.add_argument('--tensorboard', '-tb', action='store_true',
                        help='Enable TensorBoard logging')
    parser.add_argument('--hyper-verbose', action='store_true',
                        help='Debug mode: print at every major step')
    parser.add_argument('--clean-cache', action='store_true',
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
    
    # Handle timesteps (--updates is a convenience conversion)
    n_envs = args.n_envs or train_cfg.get('n_envs', 64)
    steps_per_env = train_cfg.get('steps_per_env', 128)
    if args.updates is not None:
        overrides['total_timesteps'] = args.updates * n_envs * steps_per_env
        print(f"   {args.updates} updates x {n_envs} envs x {steps_per_env} steps = {overrides['total_timesteps']:,} timesteps")
    elif args.timesteps is not None:
        overrides['total_timesteps'] = args.timesteps
    
    # Handle AMP: --no-amp > --use-amp > config
    if args.no_amp:
        overrides['use_amp'] = False
    elif args.use_amp:
        overrides['use_amp'] = True
    # else: use config default
    
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
    
    train_cfg = get_training_config()
    
    print(f"\nTraining with dashboard (process-isolated)")
    
    # Create dashboard in separate process
    dashboard = create_dashboard_process(modes_to_train, n_actions=num_actions(1))
    dashboard.start()
    
    # For parallel training, do warmup first
    compile_models = overrides.get('compile_models', train_cfg.get('compile_models', True))
    n_envs = overrides.get('n_envs', train_cfg.get('n_envs', 64))
    n_minibatches = overrides.get('n_minibatches', train_cfg.get('n_minibatches', 4))
    brain_type = overrides.get('brain_type', train_cfg.get('brain_type', 'base_cnn'))
    
    try:
        if sequential:
            # Sequential: run training in main thread (better GPU perf)
            for mode in modes_to_train:
                # Check if stop was requested or dashboard closed
                if os.path.exists('.training_stop_signal'):
                    print(f"[Sequential] Stop signal detected, skipping remaining modes")
                    try:
                        os.remove('.training_stop_signal')
                    except OSError:
                        pass
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
            # Parallel: warmup first if compiling
            if compile_models:
                _warmup_compile(brain_type, n_envs, n_minibatches)
            
            threads = []
            for mode in modes_to_train:
                t = threading.Thread(
                    target=train_ppo,
                    kwargs={
                        'mode': mode,
                        'dashboard': dashboard,
                        'output_path': f'models/ppo_{mode}.pth',
                        'log_to_file': True,
                        'skip_warmup': True,  # Already warmed up
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
    train_cfg = get_training_config()
    
    compile_models = overrides.get('compile_models', train_cfg.get('compile_models', True))
    n_envs = overrides.get('n_envs', train_cfg.get('n_envs', 64))
    n_minibatches = overrides.get('n_minibatches', train_cfg.get('n_minibatches', 4))
    brain_type = overrides.get('brain_type', train_cfg.get('brain_type', 'base_cnn'))
    
    if len(modes_to_train) > 1 and not sequential:
        # Parallel training
        print(f"\nParallel training: {len(modes_to_train)} modes")
        
        if compile_models:
            _warmup_compile(brain_type, n_envs, n_minibatches)
        
        threads = []
        for mode in modes_to_train:
            t = threading.Thread(
                target=train_ppo,
                kwargs={
                    'mode': mode,
                    'output_path': f'models/ppo_{mode}.pth',
                    'log_to_file': True,
                    'skip_warmup': True,
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

