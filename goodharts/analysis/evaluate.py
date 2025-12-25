"""
Model Evaluation for Goodhart's Law Experiments.

Run trained models and collect per-episode metrics for analysis.
Designed to be deterministic (argmax actions) for reproducible results.

Usage:
    python -m goodharts.analysis.evaluate --mode all --episodes 100
    python -m goodharts.analysis.evaluate --mode ground_truth --episodes 1024 --batch-size 128
"""
import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch

from goodharts.configs.default_config import get_config
from goodharts.modes import get_all_mode_names, ObservationSpec
from goodharts.environments.torch_env import create_torch_vec_env
from goodharts.behaviors.brains import create_brain


@dataclass
class EpisodeResult:
    """Metrics from a single evaluation episode."""
    mode: str
    episode: int
    total_reward: float
    food_eaten: int
    poison_eaten: int
    survival_steps: int
    efficiency: float  # food / (food + poison), 1.0 if no consumption


def evaluate_model(
    mode: str,
    n_episodes: int = 100,
    batch_size: int = 64,
    model_path: Optional[str] = None,
    device: torch.device = None,
    verbose: bool = True
) -> list[EpisodeResult]:
    """
    Run a trained model and collect metrics in parallel batches.
    
    Args:
        mode: Training mode (ground_truth, proxy, etc.)
        n_episodes: Total number of episodes to run
        batch_size: Number of parallel environments (higher = faster)
        model_path: Path to model file (defaults to models/ppo_{mode}.pth)
        device: Torch device
        verbose: Print progress
        
    Returns:
        List of EpisodeResult for each episode
    """
    config = get_config()
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    if model_path is None:
        model_path = f"models/ppo_{mode}.pth"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    spec = ObservationSpec.for_mode(mode, config)
    brain = create_brain('base_cnn', spec, output_size=8)
    brain = brain.to(device)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    # Handle torch.compile prefix
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    brain.load_state_dict(cleaned, strict=False)
    brain.eval()
    
    # Create parallel environments
    env = create_torch_vec_env(batch_size, spec, config, device=device)
    
    results = []
    episode_id = 0
    
    # Per-environment tracking
    episode_rewards = torch.zeros(batch_size, device=device)
    episode_food = torch.zeros(batch_size, dtype=torch.int32, device=device)
    episode_poison = torch.zeros(batch_size, dtype=torch.int32, device=device)
    episode_steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    state = env.reset()
    
    while episode_id < n_episodes:
        with torch.no_grad():
            logits = brain(state)
            actions = logits.argmax(dim=1)  # Deterministic: argmax
        
        next_state, rewards, dones = env.step(actions)
        
        # Update per-environment trackers
        episode_rewards += rewards
        episode_steps += 1
        
        # Count food/poison from reward signature
        episode_food += (rewards > 1.0).int()
        episode_poison += (rewards < -1.0).int()
        
        # Process completed episodes
        done_mask = dones.bool()
        if done_mask.any():
            done_indices = done_mask.nonzero(as_tuple=True)[0]
            
            for idx in done_indices:
                if episode_id >= n_episodes:
                    break
                
                i = idx.item()
                food = episode_food[i].item()
                poison = episode_poison[i].item()
                total_consumed = food + poison
                efficiency = food / total_consumed if total_consumed > 0 else 1.0
                
                results.append(EpisodeResult(
                    mode=mode,
                    episode=episode_id,
                    total_reward=episode_rewards[i].item(),
                    food_eaten=food,
                    poison_eaten=poison,
                    survival_steps=episode_steps[i].item(),
                    efficiency=efficiency
                ))
                episode_id += 1
                
                # Reset trackers for this environment
                episode_rewards[i] = 0
                episode_food[i] = 0
                episode_poison[i] = 0
                episode_steps[i] = 0
        
        if verbose and episode_id > 0 and episode_id % 100 == 0:
            print(f"  [{mode}] {episode_id}/{n_episodes} episodes completed")
        
        state = next_state
    
    if verbose:
        avg_reward = sum(r.total_reward for r in results) / len(results)
        avg_food = sum(r.food_eaten for r in results) / len(results)
        print(f"  [{mode}] Done: {len(results)} episodes, avg R={avg_reward:.1f}, avg Food={avg_food:.1f}")
    
    return results


def save_results(results: list[EpisodeResult], output_path: str):
    """Save evaluation results to CSV."""
    if not results:
        return
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    
    print(f"Saved {len(results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--mode', default='all', 
                        help='Mode to evaluate (or "all")')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes per mode')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Parallel environments (higher = faster)')
    parser.add_argument('--output', default='analysis/evaluation_results.csv',
                        help='Output CSV path')
    args = parser.parse_args()
    
    config = get_config()
    all_modes = get_all_mode_names(config)
    
    modes = all_modes if args.mode == 'all' else [args.mode]
    
    print(f"\nEvaluating {len(modes)} mode(s), {args.episodes} episodes each "
          f"(batch_size={args.batch_size})\n")
    
    all_results = []
    
    for mode in modes:
        try:
            print(f"[{mode}] Starting evaluation...")
            results = evaluate_model(
                mode, 
                n_episodes=args.episodes,
                batch_size=args.batch_size
            )
            all_results.extend(results)
            
            # Summary stats
            avg_reward = sum(r.total_reward for r in results) / len(results)
            avg_food = sum(r.food_eaten for r in results) / len(results)
            avg_poison = sum(r.poison_eaten for r in results) / len(results)
            avg_eff = sum(r.efficiency for r in results) / len(results)
            
            print(f"[{mode}] Summary: R={avg_reward:.1f}, "
                  f"Food={avg_food:.1f}, Poison={avg_poison:.1f}, Eff={avg_eff:.2%}\n")
            
        except FileNotFoundError as e:
            print(f"[{mode}] Skipped: {e}\n")
    
    if all_results:
        save_results(all_results, args.output)
        print(f"\nTotal: {len(all_results)} episodes evaluated")


if __name__ == '__main__':
    main()
