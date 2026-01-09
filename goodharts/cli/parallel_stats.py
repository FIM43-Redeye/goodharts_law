"""
Parallel Stats: Multi-environment statistics dashboard.

Runs many environments in parallel to compare training modes side-by-side.
Demonstrates Goodhart's Law through statistical divergence.

Usage:
    python main.py parallel-stats                    # All modes, default envs
    python main.py parallel-stats --modes ground_truth,proxy --envs 256
    python main.py parallel-stats --timesteps 50000 --checkpoint-interval 100
"""
import argparse
import threading
from pathlib import Path

import torch

from goodharts.utils.device import get_device
from goodharts.configs.default_config import get_simulation_config
from goodharts.modes import ObservationSpec, get_all_mode_names
from goodharts.environments import create_vec_env
from goodharts.behaviors.brains import load_brain
from goodharts.visualization import create_parallel_stats_app


def run_mode(
    mode: str,
    n_envs: int,
    timesteps: int,
    checkpoint_interval: int,
    dashboard,
    device: torch.device,
    config: dict,
):
    """
    Run simulation for a single mode and send updates to dashboard.
    
    Runs in a separate thread to allow parallel mode execution.
    """
    # Load model
    model_path = Path(f'models/ppo_{mode}.pth')
    if not model_path.exists():
        print(f"[{mode}] Model not found, skipping")
        dashboard.send_complete(mode)
        return

    brain, _ = load_brain(model_path, device=device)
    brain.eval()

    # Create environment
    spec = ObservationSpec.for_mode(mode, config)
    vec_env = create_vec_env(n_envs=n_envs, obs_spec=spec, config=config, device=device)

    # Disable energy freezing - agents must be able to die during evaluation
    # (freeze_energy_in_training exists only for training exploration)
    vec_env.freeze_energy = False

    # No truncation - agents run until they die
    vec_env.max_steps = 1_000_000

    # Tracking
    total_food = 0
    total_poison = 0
    total_deaths = 0
    recent_survivals = []
    survival_times = torch.zeros(n_envs, dtype=torch.int32, device=device)

    # Run loop
    obs = vec_env.reset()
    step = 0
    total_steps = timesteps * n_envs

    print(f"[{mode}] Starting: {n_envs} envs x {timesteps} steps = {total_steps:,} total", flush=True)

    with torch.no_grad():
        while step < total_steps:
            logits = brain(obs.float())
            actions = logits.argmax(dim=-1)

            # step returns (obs, eating_info, terminated, truncated)
            # eating_info is (food_mask, poison_mask, starved_mask)
            obs, eating_info, terminated, truncated = vec_env.step(actions)
            food_mask, poison_mask, _ = eating_info
            
            # Track consumption from masks
            food_this_step = food_mask.sum().item()
            poison_this_step = poison_mask.sum().item()
            total_food += int(food_this_step)
            total_poison += int(poison_this_step)
            
            survival_times += 1
            step += n_envs

            # Process deaths
            dones = terminated | truncated
            if dones.any():
                death_indices = dones.nonzero(as_tuple=True)[0]
                for idx in death_indices:
                    recent_survivals.append(survival_times[idx].item())
                    survival_times[idx] = 0
                total_deaths += dones.sum().item()

            # Send checkpoint
            if step % (checkpoint_interval * n_envs) == 0:
                # Include current ages of all living agents for survival metric
                current_ages = survival_times.tolist()
                dashboard.send_checkpoint(
                    mode=mode,
                    timesteps=step,
                    food=total_food,
                    poison=total_poison,
                    deaths=total_deaths,
                    death_times=recent_survivals,
                    current_ages=current_ages,
                )
                recent_survivals = []

    # Final update
    current_ages = survival_times.tolist()
    dashboard.send_checkpoint(mode, step, total_food, total_poison, total_deaths, recent_survivals, current_ages)
    dashboard.send_complete(mode)
    print(f"[{mode}] Complete: {step:,} steps, {total_deaths:,} deaths, "
          f"food ratio: {total_food/(total_food+total_poison+1):.1%}", flush=True)


def main():
    config = get_simulation_config()
    # Batch modes: used for --modes all (excludes experimental modes)
    batch_modes = get_all_mode_names(config, include_manual=False)
    # All modes: used for validation
    all_modes = get_all_mode_names(config, include_manual=True)

    parser = argparse.ArgumentParser(
        description='Parallel environment statistics dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-m', '--modes', type=str, default='all',
                        help=f'Comma-separated modes or "all" (default: all). Available: {", ".join(all_modes)}')
    parser.add_argument('-e', '--envs', type=int, default=192,
                        help='Environments per mode (default: 192)')
    parser.add_argument('-t', '--timesteps', type=int, default=10000,
                        help='Steps per environment (default: 10000)')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Steps between dashboard updates (default: 100)')
    parser.add_argument('--port', type=int, default=8051,
                        help='Dashboard server port (default: 8051)')
    args = parser.parse_args()

    # Parse modes
    if args.modes == 'all':
        # Only use batch_modes that have trained models (excludes manual_only)
        modes = [m for m in batch_modes if Path(f'models/ppo_{m}.pth').exists()]
    else:
        modes = [m.strip() for m in args.modes.split(',')]
        invalid = [m for m in modes if m not in all_modes]
        if invalid:
            print(f"Invalid mode(s): {', '.join(invalid)}. Valid: {', '.join(all_modes)}")
            return 1

    if not modes:
        print("No trained models found. Train first with:")
        print("  python -m goodharts.training.train_ppo --mode all")
        return 1

    device = get_device()
    total_steps = args.timesteps * args.envs * len(modes)

    print(f"\nParallel Stats Dashboard")
    print(f"========================")
    print(f"Modes: {', '.join(modes)}")
    print(f"Envs per mode: {args.envs}")
    print(f"Steps per env: {args.timesteps:,}")
    print(f"Total steps: {total_steps:,}")
    print(f"Dashboard: http://localhost:{args.port}")
    print()

    # Start dashboard
    dashboard = create_parallel_stats_app(
        modes=modes,
        total_timesteps=args.timesteps * args.envs,
        port=args.port,
    )
    dashboard.start()

    # Run modes in parallel threads
    threads = []
    for mode in modes:
        t = threading.Thread(
            target=run_mode,
            args=(mode, args.envs, args.timesteps, args.checkpoint_interval,
                  dashboard, device, config),
            daemon=True,
        )
        threads.append(t)
        t.start()

    try:
        # Wait for all threads to complete
        for t in threads:
            t.join()

        print("\nAll modes complete. Close dashboard window when done.")
        dashboard.wait()

    except KeyboardInterrupt:
        print("\nInterrupted")
        dashboard.stop()

    return 0


if __name__ == '__main__':
    sys.exit(main())
