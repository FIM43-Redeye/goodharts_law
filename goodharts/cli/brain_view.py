"""
Brain View: Single-agent neural network visualization.

Shows what a trained agent sees, how its neural network processes
the observation, and what actions it chooses.

Usage:
    python main.py brain-view --mode ground_truth
    python main.py brain-view --mode proxy --model models/ppo_proxy.pth
    python main.py brain-view --mode ground_truth --speed 100 --steps 1000
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from goodharts.utils.device import get_device
from goodharts.configs.default_config import get_simulation_config
from goodharts.modes import ObservationSpec, get_all_mode_names
from goodharts.environments import create_vec_env
from goodharts.behaviors.brains import load_brain
from goodharts.visualization.brain_view import create_brain_view


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
        vec_env._energy_enabled = 0.0

    grid_size = (config['GRID_HEIGHT'], config['GRID_WIDTH'])

    # Create visualization (matplotlib-based, runs in main thread)
    view = create_brain_view(args.mode, brain, grid_size=grid_size)

    print(f"\nBrain View: {args.mode}")
    print(f"Speed: {args.speed}ms/step")
    if args.steps:
        print(f"Steps: {args.steps}")
    print("\nClose the window or press Ctrl+C to stop\n")

    # Run simulation loop
    obs = vec_env.reset()
    step = 0
    interval_sec = args.speed / 1000.0

    try:
        while args.steps is None or step < args.steps:
            if not view.is_open():
                print("\nWindow closed")
                break

            # Single forward pass - BrainVisualizer hooks capture activations
            with torch.no_grad():
                logits = brain(obs.float())
                probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
                action = logits.argmax(dim=-1)

            # Update visualization with current state
            view.update(
                grid=vec_env.grids[0],
                agent_x=vec_env.agent_x[0].item(),
                agent_y=vec_env.agent_y[0].item(),
                agent_energy=vec_env.agent_energy[0].item(),
                obs=obs[0],
                action_probs=probs,
                step_count=step,
            )

            # Step environment
            obs, _, _, _ = vec_env.step(action)
            step += 1

            # Control speed and process matplotlib events
            view.pause(interval_sec)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print(f"Completed {step} steps")
        view.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
