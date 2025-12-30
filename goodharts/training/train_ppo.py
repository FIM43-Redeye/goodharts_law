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
import signal
import sys
import threading
from pathlib import Path

from goodharts.modes import get_all_mode_names
from goodharts.configs.default_config import get_simulation_config
from goodharts.config import get_training_config
from goodharts.behaviors.brains import get_brain_names
from goodharts.training.ppo import PPOTrainer, PPOConfig
from goodharts.training.ppo.trainer import (
    request_abort, clear_abort, is_abort_requested, reset_training_state
)


# Global synchronization for stop signal (multi-threaded training)
_TRAINING_LOCK = threading.Lock()
_TRAINING_COUNTER = 0

# Track if we're currently handling a signal (prevent re-entrancy)
_SIGNAL_RECEIVED = False


def _signal_handler(signum, frame):
    """
    Handle termination signals (SIGINT, SIGTERM) for graceful shutdown.

    This signals all trainers to abort and exit cleanly, ensuring:
    - Async loggers flush remaining data
    - GPU memory is released
    - No partial/corrupted model saves
    """
    global _SIGNAL_RECEIVED

    # Prevent re-entrancy (user pressing Ctrl+C multiple times)
    if _SIGNAL_RECEIVED:
        print("\n[Signal] Forced exit (second signal received)")
        sys.exit(1)

    _SIGNAL_RECEIVED = True
    sig_name = signal.Signals(signum).name
    print(f"\n[Signal] {sig_name} received - initiating graceful shutdown...")

    # Signal all trainers to abort
    request_abort()


def _install_signal_handlers():
    """Install signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _reset_signal_state():
    """Reset signal state for new training run."""
    global _SIGNAL_RECEIVED
    _SIGNAL_RECEIVED = False
    clear_abort()


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
    config = get_simulation_config()
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
    training.add_argument('-m', '--mode', default='ground_truth',
                          metavar='MODE',
                          help=f'Training mode(s): {", ".join(all_modes)}, "all", or comma-separated list [default: ground_truth]')
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
    perf.add_argument('--compile-mode', type=str, default=None,
                      choices=['reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
                      help='torch.compile mode (reduce-overhead enables CUDA graphs)')

    # Monitoring
    monitor = parser.add_argument_group('Monitoring')
    monitor.add_argument('-d', '--dashboard', action='store_true',
                         help='Show live training dashboard')
    monitor.add_argument('-v', '--verbose', action='store_true', dest='hyper_verbose',
                         help='Debug mode: print at every major step')
    monitor.add_argument('--log', action='store_true', dest='force_log',
                         help='Force file logging (for debugging)')
    monitor.add_argument('--no-log', action='store_true', dest='no_log',
                         help='Disable file logging even in multi-mode')

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
    experiment.add_argument('--analyze', action='store_true',
                            help='Run evaluation after training and generate comparison stats')

    # Utility
    utility = parser.add_argument_group('Utility')
    utility.add_argument('-b', '--benchmark', action='store_true',
                         help='Benchmark mode: measure throughput, discard model')
    utility.add_argument('--profile-trace', type=int, metavar='N',
                         help='Profile N updates and save trace to ./profile_trace/')
    utility.add_argument('--clean-cache', action='store_true',
                         help='Delete compilation cache before starting')
    utility.add_argument('--gpu-log-interval-ms', type=int, metavar='MS',
                         help='Log GPU utilization every MS milliseconds (0 = disabled)')

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
    if args.gpu_log_interval_ms is not None:
        overrides['gpu_log_interval_ms'] = args.gpu_log_interval_ms

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

    # Handle compile mode
    if args.compile_mode:
        overrides['compile_mode'] = args.compile_mode

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

    # Parse mode(s) - supports "all", single mode, or comma-separated list
    if args.mode == 'all':
        modes_to_train = all_modes
    elif ',' in args.mode:
        modes_to_train = [m.strip() for m in args.mode.split(',')]
        invalid = [m for m in modes_to_train if m not in all_modes]
        if invalid:
            parser.error(f"Invalid mode(s): {', '.join(invalid)}. Valid: {', '.join(all_modes)}")
    else:
        if args.mode not in all_modes:
            parser.error(f"Invalid mode: {args.mode}. Valid: {', '.join(all_modes)}, or 'all'")
        modes_to_train = [args.mode]

    # Install signal handlers for graceful shutdown
    _install_signal_handlers()
    _reset_signal_state()
    reset_training_state()  # Clear any stale global state from previous runs

    # Determine file logging: --log enables, --no-log disables
    if args.force_log:
        log_to_file = True
    elif args.no_log:
        log_to_file = False
    else:
        # Default: logging disabled (use --log to enable TensorBoard logging)
        log_to_file = False

    # Handle --profile-trace separately (requires special profiler setup)
    if args.profile_trace:
        if len(modes_to_train) > 1:
            print("ERROR: --profile-trace only supports single mode. Use --mode <mode>")
            sys.exit(1)
        _run_with_profiling(modes_to_train[0], overrides, args.profile_trace)
        return

    # Training execution
    try:
        if args.dashboard:
            _run_with_dashboard(modes_to_train, overrides, args.sequential, log_to_file)
        else:
            _run_without_dashboard(modes_to_train, overrides, args.sequential, log_to_file)

        # Post-training analysis if requested
        if args.analyze and not is_abort_requested():
            _run_post_training_analysis(modes_to_train)
    finally:
        # Always reset state on exit (clean or aborted)
        reset_training_state()
 
 
def _run_with_profiling(mode: str, overrides: dict, n_updates: int):
    """
    Run training with PyTorch profiler to capture GPU trace.

    Runs warmup first, then profiles exactly n_updates and stops cleanly.
    """
    import torch
    from torch.profiler import profile, schedule, ProfilerActivity
    import shutil

    # Clear old profile data
    profile_dir = './profile_trace'
    if os.path.exists(profile_dir):
        shutil.rmtree(profile_dir)
    os.makedirs(profile_dir, exist_ok=True)

    # Configure for profiling: warmup + profiled updates
    n_envs = overrides.get('n_envs', 192)
    steps_per_env = 128
    warmup_updates = 3  # Let JIT compile and stabilize
    total_updates = warmup_updates + n_updates + 2  # +2 buffer for schedule
    total_timesteps = total_updates * n_envs * steps_per_env

    profile_overrides = {
        **overrides,
        'total_timesteps': total_timesteps,
        'validation_interval': 0,  # Skip validation during profiling
        'profile_enabled': False,  # Disable internal profiler (we use our own)
        # Disable CUDA graphs during profiling: profiler can't see inside graphs,
        # and graph capture overhead skews timing measurements
    }

    print(f"\n[Profile] Configuration:")
    print(f"   Warmup: {warmup_updates} updates (not profiled)")
    print(f"   Profile: {n_updates} updates")
    print(f"   Output: {profile_dir}/")

    # Create trainer
    config = PPOConfig.from_config(mode=mode, **profile_overrides)
    trainer = PPOTrainer(config, dashboard=None)

    # Trace export path
    trace_path = os.path.join(profile_dir, 'trace.json')

    # Profiler state
    profiler_state = {
        'profiler': None,
        'done': False,
    }

    def on_trace_ready(prof):
        """Called when trace is ready - export it."""
        print(f"   [Exporting trace to {trace_path}]")
        prof.export_chrome_trace(trace_path)

        # Print summary
        print("\n" + "=" * 70)
        print("KERNEL TIME SUMMARY (sorted by CUDA time)")
        print("=" * 70)
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=30
        ))

        print(f"\n[Profile] Trace saved to {trace_path}")
        print("[Profile] View with: Open chrome://tracing or ui.perfetto.dev and load trace.json")
        profiler_state['done'] = True

    # Create profiler with schedule:
    # - wait: skip first N updates (our warmup)
    # - warmup: profiler's own warmup (1 update)
    # - active: actually record N updates
    # - repeat: only do this once
    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=warmup_updates,
            warmup=1,
            active=n_updates,
            repeat=1,
        ),
        on_trace_ready=on_trace_ready,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
    )
    profiler_state['profiler'] = profiler

    def profiler_callback(update_count: int) -> bool:
        """Called after each update. Returns False to stop training."""
        prof = profiler_state['profiler']

        # Step the profiler (advances through wait -> warmup -> active -> done)
        prof.step()

        if update_count < warmup_updates:
            print(f"   [Warmup {update_count}/{warmup_updates}]")
        elif update_count < warmup_updates + 1:
            print(f"   [Profiler warmup]")
        elif not profiler_state['done']:
            active_step = update_count - warmup_updates
            print(f"   [Profiling {active_step}/{n_updates}]")

        # Stop once trace is exported
        return not profiler_state['done']

    # Install callback and run
    trainer._profiler_callback = profiler_callback

    print("\n[Profile] Starting training...")
    profiler.__enter__()
    try:
        trainer.train()
    finally:
        # Clean up profiler
        try:
            profiler.__exit__(None, None, None)
        except Exception:
            pass

    print("\n[Profile] Done!")


def _run_with_dashboard(modes_to_train: list, overrides: dict, sequential: bool, log_to_file: bool):
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
                # Check if abort was requested (signal handler)
                if is_abort_requested():
                    print(f"[Sequential] Abort signal received, stopping")
                    break
                # Check if stop was requested or dashboard closed
                if dashboard.should_stop():
                    print(f"[Sequential] Dashboard stop requested, skipping remaining modes")
                    break
                if not dashboard.is_alive():
                    print("[Sequential] Dashboard closed, stopping training")
                    break

                train_ppo(
                    mode=mode,
                    dashboard=dashboard,
                    output_path=f'models/ppo_{mode}.pth',
                    log_to_file=log_to_file,
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
                        'log_to_file': log_to_file,
                        **overrides
                    },
                    daemon=True
                )
                threads.append(t)
                t.start()

            # Wait for all threads to complete (with abort check)
            for t in threads:
                while t.is_alive():
                    t.join(timeout=0.5)
                    if is_abort_requested():
                        # Signal received - threads will see it and exit
                        break

    except KeyboardInterrupt:
        # KeyboardInterrupt may still happen if signal handler hasn't run yet
        request_abort()
        print("\n[Dashboard] Training interrupted")

    # Don't auto-close dashboard - let user take screenshots
    # Dashboard will close when user closes the window
    if dashboard.is_alive():
        if is_abort_requested():
            print("\n[Dashboard] Training aborted. Close the dashboard window when done.")
        else:
            print("\n[Dashboard] Training complete. Close the dashboard window when done.")


def _run_without_dashboard(modes_to_train: list, overrides: dict, sequential: bool, log_to_file: bool):
    """Run training without dashboard."""
    try:
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
                        'log_to_file': log_to_file,
                        **overrides
                    },
                    daemon=True
                )
                threads.append(t)
                t.start()

            # Wait for all threads to complete (with abort check)
            for t in threads:
                while t.is_alive():
                    t.join(timeout=0.5)
                    if is_abort_requested():
                        # Signal received - threads will see it and exit
                        break
        else:
            # Sequential training
            if len(modes_to_train) > 1:
                print(f"\nSequential training: {len(modes_to_train)} modes")
            for mode in modes_to_train:
                # Check if abort was requested (signal handler)
                if is_abort_requested():
                    print(f"[Sequential] Abort signal received, stopping")
                    break
                # Legacy file-based stop signal (for compatibility)
                if os.path.exists('.training_stop_signal'):
                    print(f"[Sequential] Stop signal file detected, skipping remaining modes")
                    try:
                        os.remove('.training_stop_signal')
                    except OSError:
                        pass
                    break
                train_ppo(
                    mode=mode,
                    output_path=f'models/ppo_{mode}.pth',
                    log_to_file=log_to_file,
                    **overrides
                )

    except KeyboardInterrupt:
        # KeyboardInterrupt may still happen if signal handler hasn't run yet
        request_abort()
        print("\nTraining interrupted")


def _run_post_training_analysis(modes_trained: list, n_episodes: int = 100):
    """Run evaluation and comparison on trained models."""
    print("\n" + "="*60)
    print("POST-TRAINING ANALYSIS")
    print("="*60)

    try:
        from goodharts.analysis.evaluate import evaluate_model
        from goodharts.analysis.compare import compare_modes

        all_results = {}
        for mode in modes_trained:
            model_path = f'models/ppo_{mode}.pth'
            if Path(model_path).exists():
                print(f"\n[Evaluate] Running {mode} ({n_episodes} episodes)...")
                results = evaluate_model(mode, n_episodes=n_episodes, verbose=False)
                all_results[mode] = results

                # Quick summary
                avg_food = sum(r.food_eaten for r in results) / len(results)
                avg_poison = sum(r.poison_eaten for r in results) / len(results)
                avg_efficiency = sum(r.efficiency for r in results) / len(results)
                print(f"         Food: {avg_food:.1f}, Poison: {avg_poison:.1f}, Efficiency: {avg_efficiency:.2%}")
            else:
                print(f"\n[Evaluate] Skipping {mode} - model not found")

        if len(all_results) >= 2:
            print("\n[Compare] Generating comparison statistics...")
            comparison = compare_modes(all_results)
            print(f"         Goodhart Failure Index: {comparison.get('goodhart_failure_index', 'N/A')}")

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)

    except Exception as e:
        print(f"\n[Analysis] Error during post-training analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

