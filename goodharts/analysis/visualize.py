"""
Visualization for Goodhart's Law Experiments.

Generate publication-ready figures from evaluation results.

Usage:
    python -m goodharts.analysis.visualize
    python -m goodharts.analysis.visualize --input analysis/evaluation_results.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# Color scheme for modes
MODE_COLORS = {
    'ground_truth': '#22c79a',       # Green - correct behavior
    'ground_truth_handhold': '#4ecdc4',  # Teal - guided learning
    'proxy_jammed': '#ffc107',       # Yellow/amber - imperfect info
    'proxy': '#ff6b6b',              # Red - Goodhart failure
}

# Friendly display names
MODE_NAMES = {
    'ground_truth': 'Ground Truth',
    'ground_truth_handhold': 'Ground Truth (Guided)',
    'proxy_jammed': 'Proxy (Jammed)',
    'proxy': 'Proxy (Goodhart)',
}


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


def plot_reward_comparison(data: dict[str, list[dict]], output_dir: Path):
    """Bar chart comparing average rewards across modes."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modes = list(data.keys())
    means = [np.mean([e['total_reward'] for e in data[m]]) for m in modes]
    stds = [np.std([e['total_reward'] for e in data[m]]) for m in modes]
    colors = [MODE_COLORS.get(m, '#888888') for m in modes]
    labels = [MODE_NAMES.get(m, m) for m in modes]
    
    x = np.arange(len(modes))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='white')
    
    ax.set_ylabel('Average Episode Reward', fontsize=12)
    ax.set_title('Agent Performance by Training Mode', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.annotate(f'{mean:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'reward_comparison.png'
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")


def plot_consumption_comparison(data: dict[str, list[dict]], output_dir: Path):
    """Grouped bar chart comparing food vs poison consumption."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modes = list(data.keys())
    food_means = [np.mean([e['food_eaten'] for e in data[m]]) for m in modes]
    poison_means = [np.mean([e['poison_eaten'] for e in data[m]]) for m in modes]
    labels = [MODE_NAMES.get(m, m) for m in modes]
    
    x = np.arange(len(modes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, food_means, width, label='Food', color='#22c79a', alpha=0.8)
    bars2 = ax.bar(x + width/2, poison_means, width, label='Poison', color='#ff6b6b', alpha=0.8)
    
    ax.set_ylabel('Items Consumed per Episode', fontsize=12)
    ax.set_title('Food vs Poison Consumption by Mode', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'consumption_comparison.png'
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")


def plot_efficiency_comparison(data: dict[str, list[dict]], output_dir: Path):
    """Bar chart comparing efficiency (food / total consumed) across modes."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modes = list(data.keys())
    means = [np.mean([e['efficiency'] for e in data[m]]) * 100 for m in modes]
    stds = [np.std([e['efficiency'] for e in data[m]]) * 100 for m in modes]
    colors = [MODE_COLORS.get(m, '#888888') for m in modes]
    labels = [MODE_NAMES.get(m, m) for m in modes]
    
    x = np.arange(len(modes))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='white')
    
    ax.set_ylabel('Efficiency (%)', fontsize=12)
    ax.set_title('Consumption Efficiency by Mode\n(Food / Total Consumed)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='Random (50%)')
    
    # Add value labels
    for bar, mean in zip(bars, means):
        ax.annotate(f'{mean:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'efficiency_comparison.png'
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")


def plot_goodhart_summary(data: dict[str, list[dict]], output_dir: Path):
    """
    Summary figure showing Goodhart's Law effect.
    
    Side-by-side comparison of ground_truth vs proxy on key metrics.
    """
    if 'ground_truth' not in data or 'proxy' not in data:
        print("Skipping Goodhart summary: need both ground_truth and proxy modes")
        return
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    gt_data = data['ground_truth']
    proxy_data = data['proxy']
    
    metrics = [
        ('total_reward', 'Reward', 'Average Episode Reward'),
        ('efficiency', 'Efficiency', 'Consumption Efficiency'),
        ('poison_eaten', 'Poison', 'Poison Consumed'),
    ]
    
    for ax, (key, short_name, title) in zip(axes, metrics):
        gt_vals = [e[key] for e in gt_data]
        proxy_vals = [e[key] for e in proxy_data]
        
        if key == 'efficiency':
            gt_vals = [v * 100 for v in gt_vals]
            proxy_vals = [v * 100 for v in proxy_vals]
        
        gt_mean, gt_std = np.mean(gt_vals), np.std(gt_vals)
        proxy_mean, proxy_std = np.mean(proxy_vals), np.std(proxy_vals)
        
        x = [0, 1]
        bars = ax.bar(x, [gt_mean, proxy_mean], yerr=[gt_std, proxy_std], 
                      capsize=5, color=['#22c79a', '#ff6b6b'], alpha=0.8, edgecolor='white')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Ground Truth', 'Proxy'])
        
        # Value labels
        for bar in bars:
            height = bar.get_height()
            suffix = '%' if key == 'efficiency' else ''
            ax.annotate(f'{height:.1f}{suffix}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    fig.suptitle("Goodhart's Law: Optimizing Proxy Metrics Leads to Failure", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = output_dir / 'goodhart_summary.png'
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations')
    parser.add_argument('--input', default='analysis/evaluation_results.csv',
                        help='Input CSV from evaluate.py')
    parser.add_argument('--output-dir', default='analysis/figures',
                        help='Output directory for figures')
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run evaluate.py first to generate results.")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading results from {args.input}")
    data = load_results(args.input)
    print(f"Found {len(data)} modes: {list(data.keys())}\n")
    
    # Generate all plots
    plot_reward_comparison(data, output_dir)
    plot_consumption_comparison(data, output_dir)
    plot_efficiency_comparison(data, output_dir)
    plot_goodhart_summary(data, output_dir)
    
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == '__main__':
    main()
