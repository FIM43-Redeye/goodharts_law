#!/usr/bin/env python3
"""
Comprehensive testing for trained Goodhart agents.

Uses continuous survival paradigm: agents run until they die (starvation),
then auto-respawn. Tracks death events and survival times, not "episodes".

Collected metrics:
- Survival time (steps lived before each death)
- Food/poison consumed per death
- Deaths per 1000 steps (population death rate)
- Efficiency (food / total consumed) - the key Goodhart metric

Supports parallel mode testing, real-time dashboard, and structured JSON output.

Usage:
    python scripts/evaluate.py --mode ground_truth --timesteps 100000
    python scripts/evaluate.py --mode all --dashboard
    python scripts/evaluate.py --mode all --deterministic --seed 42
"""

import argparse
import json
import os
import sys
import threading
import queue
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from goodharts.utils.device import get_device
from goodharts.utils.seed import set_seed
from goodharts.configs.default_config import get_simulation_config
from goodharts.modes import get_all_mode_names
from goodharts.config import get_training_config
from goodharts.evaluation import EvaluationConfig, ModelTester


def discover_models(models_dir: Path) -> dict[str, Path]:
    """Find all trained models in directory."""
    models = {}
    for f in models_dir.glob("ppo_*.pth"):
        name = f.stem.replace("ppo_", "")
        models[name] = f
    return models


def run_single_mode(mode: str, overrides: dict, dashboard=None) -> dict:
    """Run testing for a single mode."""
    config = EvaluationConfig.from_config(mode=mode, **overrides)
    tester = ModelTester(config, dashboard=dashboard)
    return tester.run()


def run_parallel(modes: list[str], overrides: dict, dashboard=None) -> dict:
    """Run testing for multiple modes in parallel.

    Uses a startup lock to serialize CUDA context initialization,
    preventing device contention errors on AMD/ROCm.
    """
    import torch
    from goodharts.utils.device import get_device

    # Pre-initialize CUDA context on main thread to avoid race conditions
    device = get_device()
    if device.type == 'cuda':
        # Warm up CUDA context
        _ = torch.zeros(1, device=device)

    results = {}
    results_lock = threading.Lock()
    startup_lock = threading.Lock()  # Serialize initial setup

    def test_mode(mode: str):
        try:
            # Serialize the initial CUDA setup phase
            with startup_lock:
                config = EvaluationConfig.from_config(mode=mode, **overrides)
                tester = ModelTester(config, dashboard=dashboard)
                tester._setup()

            # Run the actual testing loop (can run in parallel after setup)
            try:
                tester._testing_loop()
            except KeyboardInterrupt:
                print(f"\n[{mode}] Testing interrupted")

            result = tester._finalize()

            with results_lock:
                results[mode] = result
        except Exception as e:
            print(f"[{mode}] Error: {e}")
            import traceback
            traceback.print_exc()
            with results_lock:
                results[mode] = {'error': str(e)}

    threads = []
    for mode in modes:
        t = threading.Thread(target=test_mode, args=(mode,), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return results


def run_sequential(modes: list[str], overrides: dict, dashboard=None) -> dict:
    """Run testing for multiple modes sequentially."""
    results = {}
    
    for mode in modes:
        if dashboard and dashboard.should_stop():
            print("\n[Test] Stopping due to dashboard request")
            break
        
        try:
            results[mode] = run_single_mode(mode, overrides, dashboard)
        except FileNotFoundError as e:
            print(f"[{mode}] Skipped: {e}")
            results[mode] = {'error': str(e)}
        except Exception as e:
            print(f"[{mode}] Error: {e}")
            results[mode] = {'error': str(e)}
    
    return results


def merge_results(results: dict, output_path: Path):
    """Merge multi-mode results into single JSON."""
    merged = {
        'timestamp': datetime.now().isoformat(),
        'modes': list(results.keys()),
        'results': results,
    }

    # Compute cross-mode comparison using survival-based metrics
    aggregates = {}
    for mode, result in results.items():
        if 'aggregates' in result and result['aggregates']:
            agg = result['aggregates']
            aggregates[mode] = {
                'efficiency': agg['overall_efficiency'],
                'survival': agg['survival_mean'],
                'deaths_per_1k': agg['deaths_per_1k_steps'],
                'food_per_1k': agg['food_per_1k_steps'],
                'poison_per_1k': agg['poison_per_1k_steps'],
            }

    if aggregates:
        merged['comparison'] = aggregates

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"\n[Test] Combined results saved to: {output_path}")


def print_comparison(results: dict):
    """Print cross-mode comparison table using survival-based metrics."""
    valid_results = {
        mode: r for mode, r in results.items()
        if 'aggregates' in r and r['aggregates']
    }

    if not valid_results:
        return

    print(f"\n{'='*75}")
    print("CROSS-MODE SURVIVAL COMPARISON")
    print(f"{'='*75}")
    print(f"{'Mode':<20} {'Efficiency':>12} {'Survival':>10} {'Deaths/1k':>10} {'Food/1k':>10} {'Poison/1k':>10}")
    print(f"-"*75)

    for mode, result in sorted(valid_results.items()):
        agg = result['aggregates']
        print(f"{mode:<20} {agg['overall_efficiency']:>12.1%} {agg['survival_mean']:>10.1f} "
              f"{agg['deaths_per_1k_steps']:>10.2f} {agg['food_per_1k_steps']:>10.1f} "
              f"{agg['poison_per_1k_steps']:>10.1f}")

    print(f"{'='*75}")

    # Highlight Goodhart's Law demonstration
    if 'ground_truth' in valid_results and 'proxy' in valid_results:
        gt = valid_results['ground_truth']['aggregates']
        px = valid_results['proxy']['aggregates']

        efficiency_gap = gt['overall_efficiency'] - px['overall_efficiency']
        survival_gap = gt['survival_mean'] - px['survival_mean']

        print(f"\nGoodhart's Law Effect:")
        print(f"  Ground truth efficiency: {gt['overall_efficiency']:.1%}")
        print(f"  Proxy efficiency:        {px['overall_efficiency']:.1%}")
        if efficiency_gap > 0.1:
            print(f"  Efficiency gap: {efficiency_gap:.1%} (proxy fails to distinguish food from poison)")
        if survival_gap > 0:
            print(f"  Survival gap: {survival_gap:.0f} steps (proxy dies faster)")
        elif survival_gap < 0:
            print(f"  Note: Proxy survives longer ({-survival_gap:.0f} steps), but consumes poison")


def main():
    config = get_simulation_config()
    train_cfg = get_training_config()
    all_modes = get_all_mode_names(config)
    
    parser = argparse.ArgumentParser(
        description='Testing for trained Goodhart agents.',
        epilog='Config precedence: CLI args > config.toml',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Scope
    scope = parser.add_argument_group('Scope')
    scope.add_argument('-m', '--mode', default='ground_truth', metavar='MODE',
                       help=f'Mode(s): {", ".join(all_modes)}, "all", or comma-separated')
    scope.add_argument('-t', '--timesteps', type=int, default=1000, metavar='N',
                       help='Steps per environment (multiplied by n_envs for total, default: 1000)')
    scope.add_argument('-e', '--n-envs', type=int, default=None, metavar='N',
                       help='Parallel environments per mode')
    scope.add_argument('-s', '--sequential', action='store_true',
                       help='Test modes sequentially (saves VRAM)')
    
    # Determinism
    determ = parser.add_argument_group('Determinism')
    determ.add_argument('--deterministic', action='store_true',
                        help='Deterministic mode: argmax actions, fixed seed')
    determ.add_argument('--seed', type=int, default=None, metavar='N',
                        help='Random seed for reproducibility')
    determ.add_argument('--temperature', type=float, default=1.0, metavar='T',
                        help='Softmax temperature (default: 1.0, ignored if deterministic)')
    
    # Environment
    env = parser.add_argument_group('Environment')
    env.add_argument('--food', type=int, default=None, metavar='N',
                     help='Fixed food count (default: training ranges)')
    env.add_argument('--poison', type=int, default=None, metavar='N',
                     help='Fixed poison count (default: training ranges)')
    env.add_argument('--move-cost', type=float, default=None, metavar='C',
                     help='Energy cost per move (default: 0.01, try 0.2 for more deaths)')
    
    # Output
    output = parser.add_argument_group('Output')
    output.add_argument('-o', '--output', default='generated/eval_results.json',
                        help='JSON output path (default: generated/eval_results.json)')
    output.add_argument('-d', '--dashboard', action='store_true',
                        help='Show live testing dashboard')
    
    # Models
    models = parser.add_argument_group('Models')
    models.add_argument('--model', default=None, metavar='PATH',
                        help='Model path override (default: auto-detect from mode)')
    models.add_argument('--models-dir', type=Path, default=Path('models'),
                        help='Directory containing trained models')
    
    # Legacy compatibility
    legacy = parser.add_argument_group('Legacy (for backwards compatibility)')
    legacy.add_argument('--episodes', '-n', type=int, default=None, metavar='N',
                        help='[DEPRECATED] Use --timesteps instead. Episodes to run.')
    
    args = parser.parse_args()
    
    # Handle legacy --episodes flag
    if args.episodes is not None:
        print("Warning: --episodes is deprecated. Use --timesteps instead.")
        print(f"         Converting {args.episodes} episodes to ~{args.episodes * 100} timesteps")
        args.timesteps = args.episodes * 100
    
    # Parse modes
    if args.mode == 'all':
        available = discover_models(args.models_dir)
        if not available:
            print(f"No models found in {args.models_dir}")
            print(f"Train models first with: python -m goodharts.training.train_ppo --mode all")
            return
        modes_to_test = [m for m in all_modes if m in available]
        missing = set(all_modes) - set(available.keys())
        if missing:
            print(f"Note: Skipping modes without trained models: {', '.join(missing)}")
    elif ',' in args.mode:
        modes_to_test = [m.strip() for m in args.mode.split(',')]
        invalid = [m for m in modes_to_test if m not in all_modes]
        if invalid:
            parser.error(f"Invalid mode(s): {', '.join(invalid)}")
    else:
        if args.mode not in all_modes:
            parser.error(f"Invalid mode: {args.mode}. Valid: {', '.join(all_modes)}, or 'all'")
        modes_to_test = [args.mode]
    
    # Build overrides
    overrides = {
        'total_timesteps': args.timesteps,
        'deterministic': args.deterministic,
        'seed': args.seed,
        'temperature': args.temperature,
        'food_count': args.food,
        'poison_count': args.poison,
        'move_cost': args.move_cost,
        'output_path': args.output,
        'model_path': args.model,
    }
    
    if args.n_envs:
        overrides['n_envs'] = args.n_envs
    
    # Remove None values
    overrides = {k: v for k, v in overrides.items() if v is not None}
    
    print(f"\nTesting: {len(modes_to_test)} mode(s)")
    print(f"Modes: {', '.join(modes_to_test)}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Deterministic: {args.deterministic}")
    if args.seed:
        print(f"Seed: {args.seed}")
    
    # Run with or without dashboard
    dashboard = None

    if args.dashboard:
        from goodharts.evaluation.eval_dashboard import create_testing_dashboard
        # Total timesteps = per-env steps * n_envs
        n_envs = overrides.get('n_envs', train_cfg.get('n_envs', 64))
        total_timesteps = args.timesteps * n_envs
        dashboard = create_testing_dashboard(modes_to_test, total_timesteps)
        dashboard.start()
    
    try:
        if len(modes_to_test) == 1 or args.sequential:
            results = run_sequential(modes_to_test, overrides, dashboard)
        else:
            results = run_parallel(modes_to_test, overrides, dashboard)
        
        # Save combined results
        if len(results) > 1:
            output_path = Path(args.output)
            merge_results(results, output_path)
            print_comparison(results)
        
    finally:
        if dashboard:
            if dashboard.is_alive():
                print("\n[Dashboard] Testing complete. Close window when done.")
                dashboard.wait()
            else:
                dashboard.stop()


if __name__ == '__main__':
    main()
