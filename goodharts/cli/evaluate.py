"""
Comprehensive testing for trained Goodhart agents.

Uses continuous survival paradigm: agents run until they die (starvation),
then auto-respawn. Tracks death events and survival times, not "episodes".

Collected metrics:
- Survival time (steps lived before each death)
- Food/poison consumed per death
- Deaths per 1000 steps (population death rate)
- Efficiency (food / total consumed) - the key Goodhart metric

Supports parallel mode testing, real-time dashboard, multi-run aggregation,
and unified report generation.

Usage:
    # Single evaluation
    python main.py evaluate --mode ground_truth --timesteps 100000
    python main.py evaluate --mode all --dashboard

    # Multi-run with statistical aggregation
    python main.py evaluate --mode all --runs 5 --base-seed 42

    # Full report with figures and markdown
    python main.py evaluate --full-report --runs 5 --timesteps 50000
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

import torch

from goodharts.utils.device import get_device
from goodharts.utils.seed import set_seed
from goodharts.configs.default_config import get_simulation_config
from goodharts.modes import get_all_mode_names
from goodharts.config import get_training_config, get_evaluation_config
from goodharts.evaluation import (
    EvaluationConfig, ModelTester,
    generate_seeds, aggregate_runs, RunResult, MultiRunAggregates,
)


def discover_models(models_dir: Path) -> dict[str, Path]:
    """Find all trained models in directory."""
    models = {}
    for f in models_dir.glob("ppo_*.pth"):
        name = f.stem.replace("ppo_", "")
        models[name] = f
    return models


def prompt_for_seed(is_multi_run: bool = False) -> int | None:
    """
    Interactively prompt user for seed choice.

    Args:
        is_multi_run: If True, prompts for base_seed (multi-run mode)

    Returns:
        Seed value (42, custom int, or None for random)
    """
    seed_type = "base seed" if is_multi_run else "seed"
    print(f"\nNo {seed_type} specified. For reproducibility, choose a seed:")
    print("  [y] Use seed 42 (recommended for reproducibility)")
    print("  [n] Use random seed (results will vary between runs)")
    print("  [number] Use a custom seed")

    while True:
        try:
            response = input(f"Seed choice [y/n/number]: ").strip().lower()

            if response in ('y', 'yes', ''):
                print(f"Using {seed_type} 42")
                return 42
            elif response in ('n', 'no', 'random'):
                print(f"Using random {seed_type}")
                return None
            else:
                # Try to parse as integer
                seed = int(response)
                print(f"Using {seed_type} {seed}")
                return seed
        except ValueError:
            print(f"Invalid input. Enter 'y' for 42, 'n' for random, or a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nUsing random seed (interrupted)")
            return None


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


def run_multi_seed(
    modes: list[str],
    overrides: dict,
    n_runs: int,
    base_seed: int,
    sequential: bool = False,
) -> dict[str, list[dict]]:
    """
    Run evaluation multiple times with different seeds.

    Returns:
        Dict mapping mode -> list of per-run results
    """
    seeds = generate_seeds(n_runs, base_seed)
    all_results = {mode: [] for mode in modes}

    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{n_runs} (seed={seed})")
        print(f"{'='*60}")

        # Update overrides with this run's seed
        run_overrides = overrides.copy()
        run_overrides['seed'] = seed

        # Run evaluation
        if sequential or len(modes) == 1:
            results = run_sequential(modes, run_overrides)
        else:
            results = run_parallel(modes, run_overrides)

        # Collect results
        for mode, result in results.items():
            if 'error' not in result:
                all_results[mode].append(result)

    return all_results


def aggregate_multi_run_results(
    multi_results: dict[str, list[dict]],
    output_path: Path,
) -> dict:
    """
    Aggregate results from multiple runs and save to JSON.

    Returns:
        Aggregated results dict
    """
    aggregated = {
        'timestamp': datetime.now().isoformat(),
        'n_runs': max(len(runs) for runs in multi_results.values()),
        'modes': list(multi_results.keys()),
        'results': {},
    }

    for mode, runs in multi_results.items():
        if not runs:
            continue

        # Convert to RunResult format
        run_results = []
        for i, result in enumerate(runs):
            agg = result.get('aggregates', {})
            if agg:
                run_results.append(RunResult(
                    run_id=i,
                    seed=result.get('seed', 0),
                    n_deaths=agg.get('n_deaths', 0),
                    total_timesteps=agg.get('total_timesteps', 0),
                    overall_efficiency=agg.get('overall_efficiency', 0),
                    survival_mean=agg.get('survival_mean', 0),
                    survival_std=agg.get('survival_std', 0),
                    deaths_per_1k_steps=agg.get('deaths_per_1k_steps', 0),
                    food_per_1k_steps=agg.get('food_per_1k_steps', 0),
                    poison_per_1k_steps=agg.get('poison_per_1k_steps', 0),
                    food_per_death_mean=agg.get('food_per_death_mean', 0),
                    poison_per_death_mean=agg.get('poison_per_death_mean', 0),
                    reward_mean=agg.get('reward_mean', 0),
                ))

        # Aggregate
        if run_results:
            agg = aggregate_runs(mode, run_results)
            aggregated['results'][mode] = {
                'aggregates': {
                    'n_runs': agg.n_runs,
                    'seeds': agg.seeds,
                    'overall_efficiency': agg.efficiency_mean,
                    'efficiency_ci': [agg.efficiency_ci_low, agg.efficiency_ci_high],
                    'survival_mean': agg.survival_mean_of_means,
                    'survival_ci': [agg.survival_ci_low, agg.survival_ci_high],
                    'deaths_per_1k_steps': agg.death_rate_mean,
                    'food_per_1k_steps': agg.food_rate_mean,
                    'poison_per_1k_steps': agg.poison_rate_mean,
                    'total_deaths': agg.total_deaths,
                    'total_timesteps': agg.total_timesteps,
                },
                # Flatten all deaths from all runs for distribution plots
                'deaths': [
                    death
                    for result in runs
                    for death in result.get('deaths', [])
                ],
                # Flatten all survivors from all runs (critical for ground_truth)
                'survivors': [
                    survivor
                    for result in runs
                    for survivor in result.get('survivors', [])
                ],
            }

    # Save aggregated results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n[Multi-Run] Aggregated results saved to: {output_path}")
    return aggregated


def run_full_report(
    modes: list[str],
    overrides: dict,
    n_runs: int,
    base_seed: int,
    output_dir: Path,
    sequential: bool = False,
):
    """
    Run complete evaluation pipeline with report generation.

    Steps:
    1. Run multi-seed evaluation
    2. Aggregate results
    3. Generate visualizations
    4. Generate markdown report
    5. Print console summary
    """
    from goodharts.analysis.report import ReportGenerator, ReportConfig

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = output_dir / timestamp

    print("\n" + "=" * 60)
    print("FULL REPORT PIPELINE")
    print("=" * 60)
    print(f"Modes: {', '.join(modes)}")
    print(f"Runs: {n_runs}")
    print(f"Base seed: {base_seed}")
    print(f"Output: {report_dir}")

    # Step 1: Multi-seed evaluation
    print("\n[1/4] Running multi-seed evaluation...")
    multi_results = run_multi_seed(modes, overrides, n_runs, base_seed, sequential)

    # Step 2: Aggregate results
    print("\n[2/4] Aggregating results...")
    json_path = report_dir / 'results.json'
    aggregated = aggregate_multi_run_results(multi_results, json_path)

    # Step 3 & 4: Generate report (includes figures)
    print("\n[3/4] Generating report and figures...")
    config = ReportConfig(
        title="Goodhart's Law Experiment Results",
        output_dir=output_dir,
        timestamp=timestamp,
        include_figures=True,
        include_power_analysis=True,
    )
    generator = ReportGenerator(config)
    generator.add_data(str(json_path))
    report_path = generator.generate()

    # Step 5: Console summary
    print("\n[4/4] Results summary:")
    print(generator.generate_console_summary())

    print(f"\nFull report generated at: {report_path}")
    print(f"Figures directory: {config.figures_dir}")

    return report_path


def main():
    config = get_simulation_config()
    train_cfg = get_training_config()
    eval_cfg = get_evaluation_config()
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
    scope.add_argument('-t', '--timesteps', type=int, default=eval_cfg['steps_per_env'], metavar='N',
                       help=f'Steps per environment (default: {eval_cfg["steps_per_env"]})')
    scope.add_argument('-e', '--n-envs', type=int, default=eval_cfg['n_envs'], metavar='N',
                       help=f'Parallel environments per mode (default: {eval_cfg["n_envs"]})')
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
    
    # Multi-run
    multi = parser.add_argument_group('Multi-run evaluation')
    multi.add_argument('-r', '--runs', type=int, default=eval_cfg['runs'], metavar='N',
                       help=f'Number of evaluation runs with different seeds (default: {eval_cfg["runs"]})')
    multi.add_argument('--base-seed', type=int, default=None, metavar='N',
                       help='Base seed for reproducible multi-run (prompted if not specified)')

    # Full report
    report = parser.add_argument_group('Report generation')
    report.add_argument('--full-report', action='store_true',
                        help='Run full pipeline: evaluate -> aggregate -> visualize -> report')
    report.add_argument('--report-dir', type=Path, default=Path('generated/reports'),
                        help='Output directory for reports (default: generated/reports)')

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
    
    # Handle seed interactively if not specified
    # Priority: explicit --seed/--base-seed > --deterministic (uses 42) > prompt user
    is_multi_run = args.runs > 1

    if is_multi_run:
        # Multi-run mode: use base_seed
        if args.base_seed is not None:
            base_seed = args.base_seed
        elif args.deterministic:
            base_seed = 42
            print(f"Note: Using base seed {base_seed} for deterministic multi-run evaluation")
        else:
            base_seed = prompt_for_seed(is_multi_run=True)
        # Store for later use
        args.base_seed = base_seed
        seed = args.seed  # Single-run seed not used in multi-run
    else:
        # Single-run mode: use seed
        if args.seed is not None:
            seed = args.seed
        elif args.deterministic:
            seed = 42
            print(f"Note: Using seed {seed} for deterministic evaluation")
        else:
            seed = prompt_for_seed(is_multi_run=False)

    # Build overrides
    overrides = {
        'total_timesteps': args.timesteps,
        'n_envs': args.n_envs,
        'deterministic': args.deterministic,
        'seed': seed,
        'temperature': args.temperature,
        'food_count': args.food,
        'poison_count': args.poison,
        'move_cost': args.move_cost,
        'output_path': args.output,
        'model_path': args.model,
    }

    # Remove None values
    overrides = {k: v for k, v in overrides.items() if v is not None}

    total_steps = args.timesteps * args.n_envs
    print(f"\nTesting: {len(modes_to_test)} mode(s)")
    print(f"Modes: {', '.join(modes_to_test)}")
    print(f"Envs: {args.n_envs:,} x {args.timesteps:,} steps = {total_steps:,} total steps")
    print(f"Deterministic: {args.deterministic}")
    if is_multi_run:
        seed_desc = str(args.base_seed) if args.base_seed is not None else "random"
        print(f"Runs: {args.runs} (base seed: {seed_desc})")
    else:
        seed_desc = str(seed) if seed is not None else "random"
        print(f"Seed: {seed_desc}")

    # Handle --full-report mode
    if args.full_report:
        # Full report mode: multi-run -> aggregate -> visualize -> report
        run_full_report(
            modes=modes_to_test,
            overrides=overrides,
            n_runs=args.runs if args.runs > 1 else 3,  # Default to 3 runs for reports
            base_seed=args.base_seed,
            output_dir=args.report_dir,
            sequential=args.sequential,
        )
        return

    # Handle multi-run mode (without full report)
    if args.runs > 1:
        multi_results = run_multi_seed(
            modes=modes_to_test,
            overrides=overrides,
            n_runs=args.runs,
            base_seed=args.base_seed,
            sequential=args.sequential,
        )
        output_path = Path(args.output)
        aggregate_multi_run_results(multi_results, output_path)
        return

    # Standard single-run mode
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
