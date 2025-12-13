#!/usr/bin/env python3
"""
Headless model verification script.

Runs comprehensive tests on trained models without GUI.
Use this to verify model fitness before running visual demos.

Usage:
    python training/verify_models.py
    python training/verify_models.py --steps 500 --verbose
"""
import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from goodharts.configs.default_config import get_config
from goodharts.environments.world import World
from goodharts.agents.organism import Organism
from goodharts.behaviors import OmniscientSeeker, ProxySeeker, LearnedGroundTruth, LearnedProxy
from goodharts.behaviors.brains.tiny_cnn import TinyCNN
from goodharts.behaviors.action_space import ACTION_LABELS_8, index_to_action
from goodharts.simulation import Simulation


def check_gpu():
    """Check GPU availability and current setup."""
    print("=" * 60)
    print("GPU STATUS")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Current device: {torch.cuda.current_device()}")
    else:
        print("✗ CUDA not available")
        if hasattr(torch.version, 'hip') and torch.version.hip:
            print(f"  (ROCm detected: {torch.version.hip})")
        print("  Training will use CPU")
    print()


def test_model_directional_accuracy(model_path: str, model_name: str) -> float:
    """Test if model goes toward food in each direction."""
    print(f"Testing {model_name}...")
    
    try:
        model = TinyCNN(input_shape=(11, 11), input_channels=4, output_size=8)
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()
    except FileNotFoundError:
        print(f"  ✗ Model not found: {model_path}")
        return 0.0
    
    # Test cases: food position -> expected directions
    test_cases = [
        ((5, 3), {1, 0, 2}),   # Left of center -> should go left-ish
        ((5, 7), {6, 5, 7}),   # Right of center -> should go right-ish
        ((3, 5), {3, 0, 5}),   # Above center -> should go up-ish
        ((7, 5), {4, 2, 7}),   # Below center -> should go down-ish
        ((3, 3), {0}),         # Up-left -> should go up-left
        ((7, 7), {7}),         # Down-right -> should go down-right
        ((3, 7), {5}),         # Up-right -> should go up-right
        ((7, 3), {2}),         # Down-left -> should go down-left
    ]
    
    correct = 0
    for food_pos, expected_actions in test_cases:
        view = torch.zeros(1, 4, 11, 11)
        view[0, 0, :, :] = 1.0  # All empty
        view[0, 0, food_pos[0], food_pos[1]] = 0.0  # Not empty at food
        view[0, 2, food_pos[0], food_pos[1]] = 1.0  # Food here
        
        with torch.no_grad():
            logits = model(view)
            action_idx = logits.argmax(dim=1).item()
        
        if action_idx in expected_actions:
            correct += 1
    
    accuracy = correct / len(test_cases)
    status = "✓" if accuracy >= 0.75 else "✗"
    print(f"  {status} Directional accuracy: {accuracy:.0%} ({correct}/{len(test_cases)})")
    
    return accuracy


def test_simulation_survival(behavior_class, behavior_name: str, steps: int = 500, 
                             num_runs: int = 3, verbose: bool = False) -> dict:
    """Run headless simulation and collect survival statistics."""
    print(f"\nTesting {behavior_name} survival ({num_runs} runs, {steps} steps each)...")
    
    all_stats = {
        'final_alive': [],
        'food_eaten': [],
        'poison_eaten': [],
        'survival_rate': [],
    }
    
    for run in range(num_runs):
        config = get_config()
        config['AGENTS_SETUP'] = [{'behavior_class': behavior_class.__name__, 'count': 10}]
        
        sim = Simulation(config)
        initial_count = len(sim.agents)
        
        food_eaten = 0
        poison_eaten = 0
        
        for step in range(steps):
            # Track deaths this step
            alive_before = len(sim.agents)
            sim.step()
            
            # Count deaths by reason
            for death in sim.stats['deaths']:
                if death['step'] == sim.step_count:
                    if death['reason'] == 'Poison':
                        poison_eaten += 1
                    elif death['reason'] == 'Starvation':
                        pass  # Starved, not relevant to food-finding
        
        final_alive = len(sim.agents)
        survival_rate = final_alive / initial_count
        
        all_stats['final_alive'].append(final_alive)
        all_stats['survival_rate'].append(survival_rate)
        all_stats['poison_eaten'].append(poison_eaten)
        
        if verbose:
            print(f"  Run {run+1}: {final_alive}/{initial_count} survived, {poison_eaten} poison deaths")
    
    avg_survival = np.mean(all_stats['survival_rate'])
    avg_poison = np.mean(all_stats['poison_eaten'])
    
    status = "✓" if avg_survival > 0.1 else "⚠"
    print(f"  {status} Avg survival: {avg_survival:.0%}, Avg poison deaths: {avg_poison:.1f}")
    
    return {
        'avg_survival': avg_survival,
        'avg_poison_deaths': avg_poison,
        'runs': all_stats,
    }


def compare_behaviors(steps: int = 500, verbose: bool = False):
    """Compare ground-truth vs proxy behaviors (both hardcoded and learned)."""
    print("\n" + "=" * 60)
    print("BEHAVIOR COMPARISON")
    print("=" * 60)
    
    results = {}
    
    # Test each behavior type
    behaviors = [
        (OmniscientSeeker, "OmniscientSeeker (hardcoded)"),
        (ProxySeeker, "ProxySeeker (hardcoded)"),
        (LearnedGroundTruth, "LearnedGroundTruth (CNN)"),
        (LearnedProxy, "LearnedProxy (CNN)"),
    ]
    
    for behavior_class, name in behaviors:
        try:
            results[name] = test_simulation_survival(
                behavior_class, name, steps=steps, num_runs=3, verbose=verbose
            )
        except Exception as e:
            print(f"  ✗ Error testing {name}: {e}")
            results[name] = None
    
    # Summary comparison
    print("\n" + "-" * 60)
    print("SUMMARY: Goodhart's Law Effect")
    print("-" * 60)
    
    gt_results = results.get("LearnedGroundTruth (CNN)")
    proxy_results = results.get("LearnedProxy (CNN)")
    
    if gt_results and proxy_results:
        gt_survival = gt_results['avg_survival']
        proxy_survival = proxy_results['avg_survival']
        gt_poison = gt_results['avg_poison_deaths']
        proxy_poison = proxy_results['avg_poison_deaths']
        
        print(f"Ground-Truth CNN: {gt_survival:.0%} survival, {gt_poison:.1f} poison deaths")
        print(f"Proxy CNN:        {proxy_survival:.0%} survival, {proxy_poison:.1f} poison deaths")
        
        if proxy_poison > gt_poison:
            print(f"\n🎯 Goodhart's Law DEMONSTRATED!")
            print(f"   Proxy agents ate {proxy_poison - gt_poison:.1f} more poison on average")
        else:
            print(f"\n⚠ Goodhart's Law effect not visible in this run")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify trained model fitness")
    parser.add_argument('--steps', type=int, default=500, help='Simulation steps per run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--skip-gpu', action='store_true', help='Skip GPU check')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  MODEL VERIFICATION SUITE")
    print("=" * 60 + "\n")
    
    # GPU status
    if not args.skip_gpu:
        check_gpu()
    
    # Test model directional accuracy
    print("=" * 60)
    print("DIRECTIONAL ACCURACY TESTS")
    print("=" * 60)
    
    gt_acc = test_model_directional_accuracy('models/ground_truth.pth', 'Ground Truth Model')
    proxy_acc = test_model_directional_accuracy('models/proxy_trained.pth', 'Proxy Model')
    
    # Behavior comparison
    compare_behaviors(steps=args.steps, verbose=args.verbose)
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    
    # Final verdict
    if gt_acc >= 0.75:
        print("✓ Models appear to be trained correctly")
        print("  Run 'python main.py --learned' for visual demo")
    else:
        print("⚠ Models may need retraining")
        print("  Run 'python training/train.py --mode both --epochs 100'")


if __name__ == "__main__":
    main()
