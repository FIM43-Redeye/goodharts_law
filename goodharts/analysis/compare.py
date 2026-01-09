"""
Statistical Comparison for Goodhart's Law Experiments.

Analyze evaluation results and compute statistical significance.

Usage:
    python -m goodharts.analysis.compare
    python -m goodharts.analysis.compare --input analysis/evaluation_results.csv
"""
import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy import stats


@dataclass
class ModeStats:
    """Aggregate statistics for a single mode."""
    mode: str
    n_episodes: int
    
    # Means
    mean_reward: float
    mean_food: float
    mean_poison: float
    mean_survival: float
    mean_efficiency: float
    
    # Standard deviations
    std_reward: float
    std_food: float
    std_poison: float
    std_efficiency: float
    
    # 95% confidence intervals (as half-widths)
    ci_reward: float
    ci_food: float
    ci_poison: float
    ci_efficiency: float


def load_results(path: str) -> dict[str, list[dict]]:
    """Load evaluation results grouped by mode."""
    results = defaultdict(list)
    
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row['mode']
            results[mode].append({
                'total_reward': float(row['total_reward']),
                'food_eaten': int(row['food_eaten']),
                'poison_eaten': int(row['poison_eaten']),
                'survival_steps': int(row['survival_steps']),
                'efficiency': float(row['efficiency']),
            })
    
    return dict(results)


def compute_stats(mode: str, episodes: list[dict]) -> ModeStats:
    """Compute aggregate statistics for a mode."""
    n = len(episodes)
    
    rewards = [e['total_reward'] for e in episodes]
    foods = [e['food_eaten'] for e in episodes]
    poisons = [e['poison_eaten'] for e in episodes]
    survivals = [e['survival_steps'] for e in episodes]
    efficiencies = [e['efficiency'] for e in episodes]
    
    # 95% CI = 1.96 * std / sqrt(n)
    ci_factor = 1.96 / np.sqrt(n)
    
    return ModeStats(
        mode=mode,
        n_episodes=n,
        mean_reward=np.mean(rewards),
        mean_food=np.mean(foods),
        mean_poison=np.mean(poisons),
        mean_survival=np.mean(survivals),
        mean_efficiency=np.mean(efficiencies),
        std_reward=np.std(rewards),
        std_food=np.std(foods),
        std_poison=np.std(poisons),
        std_efficiency=np.std(efficiencies),
        ci_reward=np.std(rewards) * ci_factor,
        ci_food=np.std(foods) * ci_factor,
        ci_poison=np.std(poisons) * ci_factor,
        ci_efficiency=np.std(efficiencies) * ci_factor,
    )


def compare_modes(
    stats_a: ModeStats, 
    stats_b: ModeStats, 
    data_a: list[dict], 
    data_b: list[dict]
) -> dict:
    """
    Statistical comparison between two modes.
    
    Returns dict with t-test and effect size for each metric.
    """
    results = {}
    
    metrics = ['total_reward', 'food_eaten', 'poison_eaten', 'efficiency']
    
    for metric in metrics:
        values_a = [e[metric] for e in data_a]
        values_b = [e[metric] for e in data_b]
        
        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
        
        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(values_a) + np.var(values_b)) / 2)
        cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std if pooled_std > 0 else 0
        
        results[metric] = {
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'mean_diff': np.mean(values_a) - np.mean(values_b),
            'significant': p_value < 0.05
        }
    
    return results


def compute_goodhart_index(gt_stats: ModeStats, proxy_stats: ModeStats) -> float:
    """
    Compute Goodhart Failure Index.
    
    GFI = (gt_efficiency - proxy_efficiency) / gt_efficiency
    
    Higher = more severe Goodhart failure.
    0 = No difference, 1 = proxy has 0 efficiency.
    """
    if gt_stats.mean_efficiency == 0:
        return 0.0
    return (gt_stats.mean_efficiency - proxy_stats.mean_efficiency) / gt_stats.mean_efficiency


def print_summary(all_stats: dict[str, ModeStats], all_data: dict[str, list[dict]]):
    """Print formatted summary of results."""
    print("\n" + "="*80)
    print("GOODHART'S LAW EXPERIMENT RESULTS")
    print("="*80)
    
    # Summary table
    print("\n{:<25} {:>10} {:>10} {:>10} {:>12} {:>12}".format(
        "Mode", "Episodes", "Reward", "Food", "Poison", "Efficiency"
    ))
    print("-"*80)
    
    for mode, s in all_stats.items():
        print("{:<25} {:>10} {:>10.1f} {:>10.1f} {:>10.1f} {:>11.1%}".format(
            mode, s.n_episodes, s.mean_reward, s.mean_food, s.mean_poison, s.mean_efficiency
        ))
    
    print("-"*80)
    
    # Goodhart Failure Index
    if 'ground_truth' in all_stats and 'proxy' in all_stats:
        gfi = compute_goodhart_index(all_stats['ground_truth'], all_stats['proxy'])
        print(f"\nGoodhart Failure Index (proxy vs ground_truth): {gfi:.1%}")
        
        if gfi > 0.1:
            print("  -> Proxy agent shows significant Goodhart failure")
        elif gfi > 0:
            print("  -> Proxy agent shows mild Goodhart tendency")
        else:
            print("  -> No Goodhart failure detected (unusual)")
    
    # Statistical comparisons
    if 'ground_truth' in all_stats and 'proxy' in all_stats:
        print("\n" + "-"*80)
        print("Statistical Comparison: ground_truth vs proxy")
        print("-"*80)
        
        comparison = compare_modes(
            all_stats['ground_truth'], all_stats['proxy'],
            all_data['ground_truth'], all_data['proxy']
        )
        
        for metric, result in comparison.items():
            sig = "*" if result['significant'] else ""
            print(f"  {metric:<15}: diff={result['mean_diff']:+.2f}, "
                  f"p={result['p_value']:.4f}{sig}, d={result['cohens_d']:.2f}")
        
        print("\n  * = statistically significant at p<0.05")
    
    print("\n" + "="*80)


def save_summary(all_stats: dict[str, ModeStats], output_path: str):
    """Save summary statistics to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['mode', 'n_episodes', 'mean_reward', 'mean_food', 'mean_poison',
                      'mean_survival', 'mean_efficiency', 'std_reward', 'std_efficiency',
                      'ci_reward', 'ci_efficiency']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for mode, s in all_stats.items():
            writer.writerow({
                'mode': s.mode,
                'n_episodes': s.n_episodes,
                'mean_reward': s.mean_reward,
                'mean_food': s.mean_food,
                'mean_poison': s.mean_poison,
                'mean_survival': s.mean_survival,
                'mean_efficiency': s.mean_efficiency,
                'std_reward': s.std_reward,
                'std_efficiency': s.std_efficiency,
                'ci_reward': s.ci_reward,
                'ci_efficiency': s.ci_efficiency,
            })
    
    print(f"\nSaved summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare evaluation results')
    parser.add_argument('--input', default='analysis/evaluation_results.csv',
                        help='Input CSV from evaluate.py')
    parser.add_argument('--output', default='analysis/comparison_summary.csv',
                        help='Output summary CSV')
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run evaluate.py first to generate results.")
        return
    
    # Load and analyze
    all_data = load_results(args.input)
    all_stats = {mode: compute_stats(mode, episodes) 
                 for mode, episodes in all_data.items()}
    
    # Print and save
    print_summary(all_stats, all_data)
    save_summary(all_stats, args.output)


if __name__ == '__main__':
    main()
