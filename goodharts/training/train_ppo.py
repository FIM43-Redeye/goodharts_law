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
    brain_type: str = 'base_cnn',
    n_envs: int = 64,
    total_timesteps: int = 100_000,
    entropy_coef: float = 0.02,
    output_path: str = 'models/ppo_agent.pth',
    dashboard = None,
    log_to_file: bool = True,
    use_amp: bool = False,
    compile_models: bool = True,
    **kwargs
) -> dict:
    """
    Train a PPO agent.
    
    This is a convenience wrapper around PPOTrainer for backwards compatibility
    and multi-threaded training coordination.
    
    Args:
        mode: Training mode (ground_truth, proxy, etc.)
        brain_type: Neural network architecture
        n_envs: Number of parallel environments
        total_timesteps: Total environment steps
        entropy_coef: Entropy bonus coefficient
        output_path: Where to save the trained model
        dashboard: Optional training dashboard
        log_to_file: Whether to log training metrics
        use_amp: Enable automatic mixed precision
        compile_models: Whether to use torch.compile (disable for parallel training)
        
    Returns:
        Summary dict with training results
    """
    global _TRAINING_COUNTER
    
    with _TRAINING_LOCK:
        _TRAINING_COUNTER += 1
    
    try:
        config = PPOConfig(
            mode=mode,
            brain_type=brain_type,
            n_envs=n_envs,
            total_timesteps=total_timesteps,
            entropy_coef=entropy_coef,
            output_path=output_path,
            log_to_file=log_to_file,
            use_amp=use_amp,
            compile_models=compile_models,
            n_minibatches=kwargs.get('n_minibatches', 4),
            tensorboard=kwargs.get('tensorboard', False),
            skip_warmup=kwargs.get('skip_warmup', False),
            use_torch_env=kwargs.get('use_torch_env', False),
            hyper_verbose=kwargs.get('hyper_verbose', False),
            clean_cache=kwargs.get('clean_cache', False),
        )
        
        trainer = PPOTrainer(config, dashboard=dashboard)
        return trainer.train()
        
    finally:
        with _TRAINING_LOCK:
            _TRAINING_COUNTER -= 1
            # Last thread cleans up stop signal
            if _TRAINING_COUNTER == 0 and os.path.exists('.training_stop_signal'):
                try:
                    os.remove('.training_stop_signal')
                except OSError:
                    pass


def main():
    """Main CLI entry point."""
    # Cleanup stale stop signal
    if os.path.exists('.training_stop_signal'):
        try:
            os.remove('.training_stop_signal')
            print("Removed stale stop signal.")
        except OSError:
            pass

    config = get_config()
    train_cfg = get_training_config()
    all_modes = get_all_mode_names(config)
    brain_names = get_brain_names()
    
    parser = argparse.ArgumentParser(description='PPO training for Goodhart agents')
    parser.add_argument('--mode', default='ground_truth', choices=all_modes + ['all'],
                        help='Training mode (or "all" for parallel training)')
    parser.add_argument('--brain', default='base_cnn', choices=brain_names,
                        help='Neural network architecture')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total environment steps (default: 100,000)')
    parser.add_argument('--updates', type=int, default=None,
                        help='Number of PPO updates (alternative to --timesteps)')
    parser.add_argument('--entropy', type=float, default=None,
                        help='Entropy coefficient (default: from config)')
    parser.add_argument('--dashboard', '-d', action='store_true',
                        help='Show live training dashboard')
    parser.add_argument('--sequential', '-s', action='store_true',
                        help='Train modes sequentially (saves VRAM)')
    parser.add_argument('--n-envs', type=int, default=64,
                        help='Number of parallel environments')
    parser.add_argument('--use-amp', action='store_true', default=None,
                        help='Enable automatic mixed precision (FP16)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--minibatches', type=int, default=None,
                        help='Number of minibatches per epoch (default: from config)')
    parser.add_argument('--no-warmup', action='store_true',
                        help='Skip warmup (disables torch.compile and cuDNN benchmark for faster startup but slower training)')
    parser.add_argument('--tensorboard', '-tb', action='store_true',
                        help='Enable TensorBoard logging (works on Colab)')
    parser.add_argument('--torch-env', action='store_true',
                        help='Use GPU-native TorchVecEnv (faster on powerful GPUs)')
    parser.add_argument('--hyper-verbose', action='store_true',
                        help='Debug mode: print at every major step (for TPU debugging)')
    parser.add_argument('--clean-cache', action='store_true',
                        help='Delete existing compilation cache before starting')
    args = parser.parse_args()
    
    # Determine steps_per_env for update calculation
    steps_per_env = train_cfg.get('steps_per_env', 128)
    
    # Determine AMP setting: CLI overrides config
    if args.no_amp:
        use_amp = False
    elif args.use_amp:
        use_amp = True
    else:
        amp_cfg = train_cfg.get('use_amp', 'auto')
        if amp_cfg == 'auto':
            from goodharts.utils.device import get_amp_support
            use_amp = get_amp_support(verbose=True)
        else:
            use_amp = bool(amp_cfg)
    
    # Determine timesteps
    if args.updates is not None:
        total_timesteps = args.updates * args.n_envs * steps_per_env
        print(f"   {args.updates} updates x {args.n_envs} envs x {steps_per_env} steps = {total_timesteps:,} timesteps")
    elif args.timesteps is not None:
        total_timesteps = args.timesteps
    else:
        total_timesteps = 100_000
    
    modes_to_train = all_modes if args.mode == 'all' else [args.mode]
    entropy_coef = args.entropy if args.entropy is not None else train_cfg.get('entropy_coef', 0.02)
    n_minibatches = args.minibatches if args.minibatches is not None else train_cfg.get('n_minibatches', 4)
    
    # Determine warmup/compile settings
    # --no-warmup disables torch.compile AND cuDNN benchmark for fastest startup (but slower training)
    if args.no_warmup:
        compile_models = False
        # Set env var to prevent device.py from re-enabling cuDNN benchmark
        os.environ['GOODHARTS_CUDNN_BENCHMARK'] = '0'
        import torch
        torch.backends.cudnn.benchmark = False
        print("   Warmup disabled: torch.compile and cuDNN benchmark OFF")
    else:
        compile_models = True
    
    print(f"   Entropy coefficient: {entropy_coef}")
    
    # Training execution
    if args.dashboard:
        _run_with_dashboard(modes_to_train, args, total_timesteps, entropy_coef, use_amp, n_minibatches, compile_models)
    else:
        _run_without_dashboard(modes_to_train, args, total_timesteps, entropy_coef, use_amp, n_minibatches, compile_models)
 
 
def _run_with_dashboard(modes_to_train, args, total_timesteps, entropy_coef, use_amp, n_minibatches, compile_models):
    """Run training with live dashboard."""
    from goodharts.training.train_dashboard import create_dashboard
    from goodharts.behaviors.action_space import num_actions
    
    print(f"\nTraining with dashboard")
    dashboard = create_dashboard(modes_to_train, n_actions=num_actions(1))
    
    if args.sequential:
        # Sequential: train one at a time in background
        def sequential_training():
            for mode in modes_to_train:
                output_path = f'models/ppo_{mode}.pth'
                train_ppo(
                    mode=mode,
                    brain_type=args.brain,
                    n_envs=args.n_envs,
                    total_timesteps=total_timesteps,
                    entropy_coef=entropy_coef,
                    output_path=output_path,
                    dashboard=dashboard,
                    log_to_file=True,
                    use_amp=use_amp,
                    compile_models=compile_models,
                    n_minibatches=n_minibatches,
                    tensorboard=args.tensorboard,
                    use_torch_env=args.torch_env,
                    hyper_verbose=args.hyper_verbose,
                    skip_warmup=not compile_models,  # Skip warmup when --no-warmup
                    clean_cache=args.clean_cache,
                )
        
        t = threading.Thread(target=sequential_training, daemon=True)
        t.start()
        
        try:
            dashboard.run()
        except KeyboardInterrupt:
            print("\nTraining interrupted")
        
        t.join(timeout=1.0)
    else:
        # Parallel: all modes at once
        # First, do single-threaded warmup to populate kernel cache
        if compile_models:
            _warmup_compile(args.brain, args.n_envs, n_minibatches)
        
        threads = []
        for mode in modes_to_train:
            output_path = f'models/ppo_{mode}.pth'
            t = threading.Thread(
                target=train_ppo,
                kwargs={
                    'mode': mode,
                    'brain_type': args.brain,
                    'n_envs': args.n_envs,
                    'total_timesteps': total_timesteps,
                    'entropy_coef': entropy_coef,
                    'output_path': output_path,
                    'dashboard': dashboard,
                    'log_to_file': True,
                    'use_amp': use_amp,
                    'compile_models': compile_models,
                    'n_minibatches': n_minibatches,
                    'tensorboard': args.tensorboard,
                    'skip_warmup': True,
                    'clean_cache': args.clean_cache,
                    'use_torch_env': args.torch_env,
                    'hyper_verbose': args.hyper_verbose,
                },
                daemon=True
            )
            threads.append(t)
            t.start()
        
        try:
            dashboard.run()
        except KeyboardInterrupt:
            print("\nTraining interrupted")
        
        for t in threads:
            t.join(timeout=1.0)



def _run_without_dashboard(modes_to_train, args, total_timesteps, entropy_coef, use_amp, n_minibatches, compile_models):
    """Run training without dashboard."""
    if len(modes_to_train) > 1 and not args.sequential:
        # Parallel training
        print(f"\nParallel training: {len(modes_to_train)} modes")
        
        # First, do single-threaded warmup to populate kernel cache
        if compile_models:
            _warmup_compile(args.brain, args.n_envs, n_minibatches)
        
        threads = []
        for mode in modes_to_train:
            output_path = f'models/ppo_{mode}.pth'
            t = threading.Thread(
                target=train_ppo,
                kwargs={
                    'mode': mode,
                    'brain_type': args.brain,
                    'n_envs': args.n_envs,
                    'total_timesteps': total_timesteps,
                    'entropy_coef': entropy_coef,
                    'output_path': output_path,
                    'log_to_file': True,
                    'use_amp': use_amp,
                    'compile_models': compile_models,
                    'n_minibatches': n_minibatches,
                    'tensorboard': args.tensorboard,
                    'skip_warmup': True,
                    'clean_cache': args.clean_cache,
                    'use_torch_env': args.torch_env,
                    'hyper_verbose': args.hyper_verbose,
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
            output_path = f'models/ppo_{mode}.pth'
            train_ppo(
                mode=mode,
                brain_type=args.brain,
                n_envs=args.n_envs,
                total_timesteps=total_timesteps,
                entropy_coef=entropy_coef,
                output_path=output_path,
                log_to_file=len(modes_to_train) > 1,
                use_amp=use_amp,
                compile_models=compile_models,
                n_minibatches=n_minibatches,
                tensorboard=args.tensorboard,
                use_torch_env=args.torch_env,
                hyper_verbose=args.hyper_verbose,
                skip_warmup=not compile_models,  # Skip warmup when --no-warmup
                clean_cache=args.clean_cache,
            )


if __name__ == '__main__':
    main()
