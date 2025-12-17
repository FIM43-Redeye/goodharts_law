#!/usr/bin/env python3
"""
Evaluate trained models across all modes.

Runs deterministic (argmax) episodes and exports comparison data to CSV.
Uses batched environments for speed.

Usage:
    python scripts/evaluate.py                    # All models, 100 episodes each
    python scripts/evaluate.py --episodes 500     # More episodes
    python scripts/evaluate.py --models ground_truth proxy  # Specific models
    python scripts/evaluate.py --n-envs 256       # More parallel envs
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from goodharts.utils.device import get_device
from goodharts.configs.default_config import get_config
from goodharts.modes import ObservationSpec
from goodharts.behaviors.brains import create_brain
from goodharts.behaviors.action_space import num_actions
from goodharts.environments.torch_env import create_torch_vec_env


def discover_models(models_dir: Path) -> dict[str, Path]:
    """Find all trained models in directory."""
    models = {}
    for f in models_dir.glob("ppo_*.pth"):
        # Extract mode name from filename (ppo_ground_truth.pth -> ground_truth)
        name = f.stem.replace("ppo_", "")
        models[name] = f
    return models


def evaluate_model(
    model_path: Path,
    mode: str,
    n_episodes: int,
    n_envs: int,
    device: torch.device,
) -> list[dict]:
    """
    Run deterministic evaluation episodes for a model.
    
    Returns list of episode results.
    """
    config = get_config()
    spec = ObservationSpec.for_mode(mode, config)
    n_actions = num_actions(1)
    
    # Create environment
    env = create_torch_vec_env(n_envs=n_envs, obs_spec=spec, device=device)
    
    # Load model
    policy = create_brain("base_cnn", spec, output_size=n_actions).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Handle compiled model state dicts
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    policy.load_state_dict(state_dict)
    policy.eval()
    
    # Run evaluation
    results = []
    episodes_done = 0
    
    states = env.reset()
    episode_rewards = torch.zeros(n_envs, device=device)
    episode_steps = torch.zeros(n_envs, dtype=torch.long, device=device)
    
    print(f"  Evaluating {mode}...", end=" ", flush=True)
    start_time = time.perf_counter()
    
    with torch.no_grad():
        while episodes_done < n_episodes:
            # Deterministic action
            logits = policy(states.float())
            actions = logits.argmax(dim=-1)
            
            next_states, rewards, dones = env.step(actions)
            episode_rewards += rewards
            episode_steps += 1
            
            # Process completed episodes
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_indices.numel() > 0:
                for idx in done_indices:
                    if episodes_done >= n_episodes:
                        break
                    
                    results.append({
                        "mode": mode,
                        "episode": episodes_done,
                        "reward": episode_rewards[idx].item(),
                        "steps": episode_steps[idx].item(),
                        "food_eaten": env.last_episode_food[idx].item(),
                        "poison_eaten": env.last_episode_poison[idx].item(),
                    })
                    
                    episode_rewards[idx] = 0.0
                    episode_steps[idx] = 0
                    episodes_done += 1
            
            states = next_states
    
    elapsed = time.perf_counter() - start_time
    eps_per_sec = n_episodes / elapsed
    print(f"{n_episodes} episodes in {elapsed:.1f}s ({eps_per_sec:.1f} eps/s)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--episodes", "-e", type=int, default=100,
                        help="Episodes per model (default: 100)")
    parser.add_argument("--n-envs", type=int, default=192,
                        help="Parallel environments (default: 192)")
    parser.add_argument("--models", "-m", nargs="+", default=None,
                        help="Specific models to evaluate (default: all)")
    parser.add_argument("--models-dir", type=Path, default=Path("models"),
                        help="Directory containing trained models")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations"),
                        help="Directory for output files")
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    args.output_dir.mkdir(exist_ok=True)
    
    # Discover models
    all_models = discover_models(args.models_dir)
    if not all_models:
        print(f"No models found in {args.models_dir}")
        return
    
    # Filter if specific models requested
    if args.models:
        models = {k: v for k, v in all_models.items() if k in args.models}
        missing = set(args.models) - set(models.keys())
        if missing:
            print(f"Warning: models not found: {missing}")
    else:
        models = all_models
    
    print(f"\nEvaluation: {len(models)} models, {args.episodes} episodes each")
    print(f"Device: {device}, Envs: {args.n_envs}")
    print(f"Models: {', '.join(sorted(models.keys()))}\n")
    
    # Run evaluations sequentially
    all_results = []
    for mode, model_path in sorted(models.items()):
        results = evaluate_model(
            model_path=model_path,
            mode=mode,
            n_episodes=args.episodes,
            n_envs=args.n_envs,
            device=device,
        )
        all_results.extend(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    
    # CSV with all episodes
    csv_path = args.output_dir / f"{timestamp}_episodes.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "episode", "reward", "steps", "food_eaten", "poison_eaten"])
        writer.writeheader()
        writer.writerows(all_results)
    
    # Summary JSON
    summary = {}
    for mode in models.keys():
        mode_results = [r for r in all_results if r["mode"] == mode]
        if mode_results:
            summary[mode] = {
                "episodes": len(mode_results),
                "mean_reward": sum(r["reward"] for r in mode_results) / len(mode_results),
                "mean_food": sum(r["food_eaten"] for r in mode_results) / len(mode_results),
                "mean_poison": sum(r["poison_eaten"] for r in mode_results) / len(mode_results),
                "mean_steps": sum(r["steps"] for r in mode_results) / len(mode_results),
            }
    
    summary_path = args.output_dir / f"{timestamp}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<25} {'Reward':>10} {'Food':>8} {'Poison':>8}")
    print("-" * 60)
    for mode, stats in sorted(summary.items()):
        print(f"{mode:<25} {stats['mean_reward']:>10.1f} {stats['mean_food']:>8.1f} {stats['mean_poison']:>8.1f}")
    print(f"{'='*60}")
    print(f"\nSaved: {csv_path}")
    print(f"       {summary_path}")


if __name__ == "__main__":
    main()
