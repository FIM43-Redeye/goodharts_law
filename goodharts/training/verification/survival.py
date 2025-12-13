"""
Survival simulation tests for trained models.

Runs headless simulations to test model fitness in realistic scenarios.
"""
import numpy as np

from goodharts.configs.default_config import get_config
from goodharts.simulation import Simulation
from goodharts.behaviors import (
    OmniscientSeeker, 
    ProxySeeker, 
    LearnedGroundTruth, 
    LearnedProxy
)


def test_simulation_survival(behavior_class, behavior_name: str, 
                             steps: int = 500, num_runs: int = 3, 
                             verbose: bool = False) -> dict:
    """
    Run headless simulation and collect survival statistics.
    
    Args:
        behavior_class: Behavior class to test
        behavior_name: Display name  
        steps: Steps per simulation run
        num_runs: Number of runs to average over
        verbose: Print per-run details
        
    Returns:
        Dict with avg_survival, avg_poison_deaths, and raw runs data
    """
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
        
        poison_eaten = 0
        
        for step in range(steps):
            sim.step()
            
            # Count poison deaths this step
            for death in sim.stats['deaths']:
                if death['step'] == sim.step_count:
                    if death['reason'] == 'Poison':
                        poison_eaten += 1
        
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


def compare_behaviors(steps: int = 500, verbose: bool = False) -> dict:
    """
    Compare ground-truth vs proxy behaviors (both hardcoded and learned).
    
    Returns:
        Dict mapping behavior names to their test results
    """
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
