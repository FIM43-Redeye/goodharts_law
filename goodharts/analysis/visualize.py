"""
Visualization for Goodhart's Law Experiments.

Generate publication-ready figures from post-hoc analysis results.

Usage:
    python -m goodharts.analysis.visualize
    python -m goodharts.analysis.visualize --input analysis/results.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Color scheme for modes
MODE_COLORS = {
    'ground_truth': '#22c79a',       # Green - correct behavior
    'ground_truth_handhold': '#4ecdc4',  # Teal - guided learning
    'ground_truth_blinded': '#ffc107',   # Yellow/amber - blinded control
    'proxy': '#ff6b6b',              # Red - Goodhart failure
}

# Friendly display names
MODE_NAMES = {
    'ground_truth': 'Ground Truth',
    'ground_truth_handhold': 'Ground Truth (Guided)',
    'ground_truth_blinded': 'Ground Truth (Blinded)',
    'proxy': 'Proxy (Goodhart)',
}

# Dark theme colors
THEME = {
    'background': '#1a1a2e',
    'paper': '#16213e',
    'text': '#e0e0e0',
    'grid': 'rgba(128, 128, 128, 0.15)',
}


def load_results(path: str) -> dict[str, list[dict]]:
    """Load post-hoc analysis results grouped by mode."""
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


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply dark theme to figure."""
    fig.update_layout(
        plot_bgcolor=THEME['paper'],
        paper_bgcolor=THEME['background'],
        font=dict(color=THEME['text']),
    )
    fig.update_xaxes(gridcolor=THEME['grid'])
    fig.update_yaxes(gridcolor=THEME['grid'])
    return fig


def plot_reward_comparison(data: dict[str, list[dict]], output_dir: Path):
    """Bar chart comparing average rewards across modes."""
    modes = list(data.keys())
    means = [np.mean([e['total_reward'] for e in data[m]]) for m in modes]
    stds = [np.std([e['total_reward'] for e in data[m]]) for m in modes]
    colors = [MODE_COLORS.get(m, '#888888') for m in modes]
    labels = [MODE_NAMES.get(m, m) for m in modes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=means,
        error_y=dict(type='data', array=stds, visible=True),
        marker_color=colors,
        text=[f'{m:.0f}' for m in means],
        textposition='outside',
    ))

    fig.update_layout(
        title=dict(text='Agent Performance by Training Mode', x=0.5,
                  font=dict(size=16)),
        yaxis_title='Average Episode Reward',
        showlegend=False,
        height=500,
        width=800,
    )

    # Reference line at y=0
    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.3)

    _apply_theme(fig)

    output_path = output_dir / 'reward_comparison.png'
    fig.write_image(str(output_path), scale=2)
    print(f"Saved: {output_path}")


def plot_consumption_comparison(data: dict[str, list[dict]], output_dir: Path):
    """Grouped bar chart comparing food vs poison consumption."""
    modes = list(data.keys())
    food_means = [np.mean([e['food_eaten'] for e in data[m]]) for m in modes]
    poison_means = [np.mean([e['poison_eaten'] for e in data[m]]) for m in modes]
    labels = [MODE_NAMES.get(m, m) for m in modes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Food',
        x=labels,
        y=food_means,
        marker_color='#22c79a',
        text=[f'{m:.1f}' for m in food_means],
        textposition='outside',
    ))

    fig.add_trace(go.Bar(
        name='Poison',
        x=labels,
        y=poison_means,
        marker_color='#ff6b6b',
        text=[f'{m:.1f}' for m in poison_means],
        textposition='outside',
    ))

    fig.update_layout(
        title=dict(text='Food vs Poison Consumption by Mode', x=0.5,
                  font=dict(size=16)),
        yaxis_title='Items Consumed per Episode',
        barmode='group',
        legend=dict(x=0.85, y=0.95),
        height=500,
        width=800,
    )

    _apply_theme(fig)

    output_path = output_dir / 'consumption_comparison.png'
    fig.write_image(str(output_path), scale=2)
    print(f"Saved: {output_path}")


def plot_efficiency_comparison(data: dict[str, list[dict]], output_dir: Path):
    """Bar chart comparing efficiency (food / total consumed) across modes."""
    modes = list(data.keys())
    means = [np.mean([e['efficiency'] for e in data[m]]) * 100 for m in modes]
    stds = [np.std([e['efficiency'] for e in data[m]]) * 100 for m in modes]
    colors = [MODE_COLORS.get(m, '#888888') for m in modes]
    labels = [MODE_NAMES.get(m, m) for m in modes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=means,
        error_y=dict(type='data', array=stds, visible=True),
        marker_color=colors,
        text=[f'{m:.0f}%' for m in means],
        textposition='outside',
    ))

    fig.update_layout(
        title=dict(text='Consumption Efficiency by Mode<br>(Food / Total Consumed)',
                  x=0.5, font=dict(size=16)),
        yaxis_title='Efficiency (%)',
        yaxis_range=[0, 105],
        showlegend=False,
        height=500,
        width=800,
    )

    # Reference line at 50% (random baseline)
    fig.add_hline(y=50, line_dash='dash', line_color='gray', opacity=0.3,
                  annotation_text='Random (50%)', annotation_position='bottom right')

    _apply_theme(fig)

    output_path = output_dir / 'efficiency_comparison.png'
    fig.write_image(str(output_path), scale=2)
    print(f"Saved: {output_path}")


def plot_goodhart_summary(data: dict[str, list[dict]], output_dir: Path):
    """
    Summary figure showing Goodhart's Law effect.

    Side-by-side comparison of ground_truth vs proxy on key metrics.
    """
    if 'ground_truth' not in data or 'proxy' not in data:
        print("Skipping Goodhart summary: need both ground_truth and proxy modes")
        return

    gt_data = data['ground_truth']
    proxy_data = data['proxy']

    metrics = [
        ('total_reward', 'Reward', 'Average Episode Reward'),
        ('efficiency', 'Efficiency', 'Consumption Efficiency'),
        ('poison_eaten', 'Poison', 'Poison Consumed'),
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[m[2] for m in metrics],
        horizontal_spacing=0.1,
    )

    for i, (key, short_name, title) in enumerate(metrics, 1):
        gt_vals = [e[key] for e in gt_data]
        proxy_vals = [e[key] for e in proxy_data]

        if key == 'efficiency':
            gt_vals = [v * 100 for v in gt_vals]
            proxy_vals = [v * 100 for v in proxy_vals]

        gt_mean, gt_std = np.mean(gt_vals), np.std(gt_vals)
        proxy_mean, proxy_std = np.mean(proxy_vals), np.std(proxy_vals)

        suffix = '%' if key == 'efficiency' else ''

        fig.add_trace(go.Bar(
            x=['Ground Truth', 'Proxy'],
            y=[gt_mean, proxy_mean],
            error_y=dict(type='data', array=[gt_std, proxy_std], visible=True),
            marker_color=['#22c79a', '#ff6b6b'],
            text=[f'{gt_mean:.1f}{suffix}', f'{proxy_mean:.1f}{suffix}'],
            textposition='outside',
            showlegend=False,
        ), row=1, col=i)

    fig.update_layout(
        title=dict(text="Goodhart's Law: Optimizing Proxy Metrics Leads to Failure",
                  x=0.5, font=dict(size=16)),
        height=450,
        width=1200,
    )

    _apply_theme(fig)

    output_path = output_dir / 'goodhart_summary.png'
    fig.write_image(str(output_path), scale=2)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations')
    parser.add_argument('--input', default='analysis/results.csv',
                        help='Input CSV from analysis scripts')
    parser.add_argument('--output-dir', default='analysis/figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run analysis scripts first to generate results.")
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
