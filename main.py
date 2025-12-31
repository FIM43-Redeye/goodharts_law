#!/usr/bin/env python3
"""
Goodhart's Law Simulation - Entry Point

Dispatcher to visualization dashboards.

Usage:
    python main.py --brain-view -m ground_truth    # Neural network visualization
    python main.py --parallel-stats                 # Multi-mode comparison
    python main.py --help                           # Show all options
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Goodhart's Law Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Visualization Modes:
  --brain-view        Single-agent neural network visualization
                      Shows: full grid, agent view, layer activations, actions

  --parallel-stats    Multi-environment statistics dashboard
                      Shows: survival curves, food ratio, deaths per mode

Examples:
  %(prog)s --brain-view -m ground_truth
  %(prog)s --brain-view -m proxy --speed 100
  %(prog)s --parallel-stats --modes ground_truth,proxy --envs 256
  %(prog)s --parallel-stats --timesteps 50000
        ''',
    )

    # Mode selection
    mode_group = parser.add_argument_group('Visualization Mode')
    mode_group.add_argument('--brain-view', action='store_true',
                            help='Launch brain view dashboard')
    mode_group.add_argument('--parallel-stats', action='store_true',
                            help='Launch parallel stats dashboard')

    # Brain view options
    bv_group = parser.add_argument_group('Brain View Options')
    bv_group.add_argument('-m', '--mode', type=str, default='ground_truth',
                          help='Training mode (default: ground_truth)')
    bv_group.add_argument('--model', type=str, default=None,
                          help='Custom model path')
    bv_group.add_argument('--speed', type=int, default=50,
                          help='Step interval in ms (default: 50)')
    bv_group.add_argument('--steps', type=int, default=None,
                          help='Run for N steps then exit')

    # Parallel stats options
    ps_group = parser.add_argument_group('Parallel Stats Options')
    ps_group.add_argument('--modes', type=str, default='all',
                          help='Comma-separated modes or "all"')
    ps_group.add_argument('--envs', type=int, default=192,
                          help='Environments per mode (default: 192)')
    ps_group.add_argument('--timesteps', type=int, default=10000,
                          help='Steps per environment (default: 10000)')

    args = parser.parse_args()

    scripts_dir = Path(__file__).parent / 'scripts'

    if args.brain_view:
        # Launch brain view
        cmd = [sys.executable, str(scripts_dir / 'brain_view.py'),
               '-m', args.mode,
               '--speed', str(args.speed)]
        if args.model:
            cmd.extend(['--model', args.model])
        if args.steps:
            cmd.extend(['--steps', str(args.steps)])

        print(f"Launching brain view: {args.mode}")
        return subprocess.call(cmd)

    elif args.parallel_stats:
        # Launch parallel stats
        cmd = [sys.executable, str(scripts_dir / 'parallel_stats.py'),
               '--modes', args.modes,
               '--envs', str(args.envs),
               '--timesteps', str(args.timesteps)]

        print(f"Launching parallel stats")
        return subprocess.call(cmd)

    else:
        # No mode selected - show help
        print()
        print("=" * 60)
        print("  GOODHART'S LAW SIMULATION")
        print("  'When a measure becomes a target,")
        print("   it ceases to be a good measure.'")
        print("=" * 60)
        print()
        print("Choose a visualization mode:")
        print()
        print("  --brain-view        Single-agent neural network viz")
        print("  --parallel-stats    Multi-mode statistics comparison")
        print()
        print("Examples:")
        print("  python main.py --brain-view -m ground_truth")
        print("  python main.py --brain-view -m proxy")
        print("  python main.py --parallel-stats")
        print("  python main.py --parallel-stats --modes ground_truth,proxy")
        print()
        print("Use --help for all options.")
        print()
        return 0


if __name__ == '__main__':
    sys.exit(main())
