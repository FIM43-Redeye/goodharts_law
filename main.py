#!/usr/bin/env python3
"""
Goodhart's Law Simulation - Unified CLI

Entry point for all simulation tools: training, evaluation, visualization, and reporting.

Usage:
    python main.py train --mode ground_truth --updates 128
    python main.py evaluate --mode all --runs 5
    python main.py brain-view -m ground_truth
    python main.py parallel-stats
    python main.py report
    python main.py --help
"""
import sys


def main():
    # Handle no arguments
    if len(sys.argv) < 2:
        _print_banner()
        return 0

    # Extract command (first positional arg)
    command = sys.argv[1]

    # Handle global --help
    if command in ('-h', '--help'):
        _print_help()
        return 0

    # Pass remaining args (including --help) to subcommand
    remaining = sys.argv[2:]

    # Dispatch to appropriate module
    if command == 'train':
        sys.argv = ['train_ppo'] + remaining
        from goodharts.training.train_ppo import main as train_main
        return train_main()

    elif command == 'evaluate':
        sys.argv = ['evaluate'] + remaining
        from goodharts.cli.evaluate import main as evaluate_main
        return evaluate_main()

    elif command == 'brain-view':
        sys.argv = ['brain-view'] + remaining
        from goodharts.cli.brain_view import main as brain_view_main
        return brain_view_main()

    elif command == 'parallel-stats':
        sys.argv = ['parallel-stats'] + remaining
        from goodharts.cli.parallel_stats import main as parallel_stats_main
        return parallel_stats_main()

    elif command == 'report':
        sys.argv = ['report'] + remaining
        from goodharts.cli.report import main as report_main
        return report_main()

    else:
        print(f"Unknown command: {command}")
        print("Use 'python main.py --help' for available commands.")
        return 1


def _print_banner():
    """Print welcome banner with quick-start guide."""
    print()
    print("=" * 64)
    print("  GOODHART'S LAW SIMULATION")
    print("  'When a measure becomes a target,")
    print("   it ceases to be a good measure.'")
    print("=" * 64)
    print()
    print("Quick Start:")
    print()
    print("  1. Train agents on different modes:")
    print("     python main.py train --mode ground_truth --updates 128")
    print("     python main.py train --mode proxy --updates 128")
    print()
    print("  2. Evaluate and compare:")
    print("     python main.py evaluate --mode all --runs 5 --full-report")
    print()
    print("  3. Visualize (requires trained models):")
    print("     python main.py brain-view -m ground_truth")
    print("     python main.py parallel-stats")
    print()
    print("Commands:")
    print("  train           Train PPO agents")
    print("  evaluate        Evaluate with statistics")
    print("  brain-view      Neural network visualization")
    print("  parallel-stats  Multi-mode statistics")
    print("  report          Regenerate reports")
    print()
    print("Use 'python main.py <command> --help' for command options.")
    print()


def _print_help():
    """Print detailed help message."""
    print("usage: python main.py <command> [options]")
    print()
    print("Goodhart's Law Simulation - AI Safety Demonstration")
    print()
    print("Commands:")
    print("  train           Train PPO agents on different observation modes")
    print("  evaluate        Evaluate trained models with statistical analysis")
    print("  brain-view      Single-agent neural network visualization")
    print("  parallel-stats  Multi-environment statistics dashboard")
    print("  report          Regenerate report from saved evaluation results")
    print()
    print("Examples:")
    print("  python main.py train --mode ground_truth --updates 128")
    print("  python main.py train --mode all --dashboard")
    print("  python main.py evaluate --mode all --runs 5 --full-report")
    print("  python main.py brain-view -m proxy --speed 100")
    print("  python main.py parallel-stats --modes ground_truth,proxy --envs 256")
    print("  python main.py report --input generated/reports/latest/results.json")
    print()
    print("For command-specific help:")
    print("  python main.py <command> --help")
    print()


if __name__ == '__main__':
    sys.exit(main())
