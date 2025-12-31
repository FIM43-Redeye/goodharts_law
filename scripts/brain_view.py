#!/usr/bin/env python3
"""
Brain View: Single-agent neural network visualization.

Shows what a trained agent sees, how its neural network processes
the observation, and what actions it chooses.

Usage:
    python scripts/brain_view.py --mode ground_truth
    python scripts/brain_view.py --mode proxy --model models/ppo_proxy.pth
    python scripts/brain_view.py --mode ground_truth --speed 100 --steps 1000
"""
import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from goodharts.utils.device import get_device
from goodharts.configs.default_config import get_simulation_config
from goodharts.modes import ObservationSpec, get_all_mode_names
from goodharts.environments import create_vec_env
from goodharts.behaviors.brains import load_brain
from goodharts.visualization import create_brain_view_app


def main():
    config = get_simulation_config()
    all_modes = get_all_mode_names(config)

    parser = argparse.ArgumentParser(
        description='Neural network visualization for single agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-m', '--mode', choices=all_modes, default='ground_truth',
                        help='Training mode to visualize (default: ground_truth)')
    parser.add_argument('--model', type=str, default=None, metavar='PATH',
                        help='Model path (default: models/ppo_{mode}.pth)')
    parser.add_argument('--speed', type=int, default=50, metavar='MS',
                        help='Step interval in milliseconds (default: 50)')
    parser.add_argument('--steps', type=int, default=None, metavar='N',
                        help='Run for N steps then exit (default: forever)')
    parser.add_argument('--freeze-energy', action='store_true', default=True,
                        help='Freeze agent energy for clean visualization (default: True)')
    parser.add_argument('--no-freeze-energy', action='store_false', dest='freeze_energy',
                        help='Allow agent to die normally')
    parser.add_argument('--port', type=int, default=8050,
                        help='Dashboard server port (default: 8050)')
    args = parser.parse_args()

    device = get_device()

    # Determine model path
    model_path = args.model or f'models/ppo_{args.mode}.pth'
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print(f"Train first with: python -m goodharts.training.train_ppo --mode {args.mode}")
        return 1

    # Load model
    print(f"Loading model: {model_path}")
    brain, metadata = load_brain(model_path, device=device)
    brain.eval()

    # Create single-agent environment
    spec = ObservationSpec.for_mode(args.mode, config)
    vec_env = create_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

    if args.freeze_energy:
        # Disable energy consumption for visualization
        vec_env._energy_enabled = 0.0

    grid_size = (config['GRID_HEIGHT'], config['GRID_WIDTH'])

    # Start visualization dashboard
    app = create_brain_view_app(args.mode, brain, grid_size=grid_size, port=args.port)
    app.start()

    # Give dashboard time to start
    time.sleep(1.0)

    print(f"\nBrain View: {args.mode}")
    print(f"Speed: {args.speed}ms/step")
    if args.steps:
        print(f"Steps: {args.steps}")
    print(f"Dashboard: http://localhost:{args.port}")
    print("\nPress Ctrl+C to stop\n")

    # Run simulation loop
    obs = vec_env.reset()
    step = 0

    try:
        while args.steps is None or step < args.steps:
            # Get action from model
            with torch.no_grad():
                logits = brain(obs.float())
                action = logits.argmax(dim=-1)

            # Update visualization
            app.update(
                grid=vec_env.grids[0],
                agent_x=vec_env.agent_x[0].item(),
                agent_y=vec_env.agent_y[0].item(),
                agent_energy=vec_env.agent_energy[0].item(),
                obs=obs[0],
                step_count=step,
            )

            # Step environment
            obs, _, _, _ = vec_env.step(action)
            step += 1

            # Control speed
            time.sleep(args.speed / 1000.0)

            # Check if dashboard closed
            if not app.is_running():
                print("\nDashboard closed")
                break

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print(f"Completed {step} steps")
        app.stop()

    return 0


if __name__ == '__main__':
    sys.exit(main())
