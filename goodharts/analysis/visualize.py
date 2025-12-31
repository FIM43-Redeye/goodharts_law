"""
Visualization for Goodhart's Law Experiments.

Generate publication-ready figures from post-hoc analysis results.

Provides two categories of plots:
1. Basic comparison plots (bar charts with std error bars)
2. Distribution plots (violin, box, histogram)
3. Annotated plots (with p-values, CIs, effect sizes)

Usage:
    python -m goodharts.analysis.visualize
    python -m goodharts.analysis.visualize --input analysis/results.csv
    python -m goodharts.analysis.visualize --annotated --distributions
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

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


# -----------------------------------------------------------------------------
# Distribution plots
# -----------------------------------------------------------------------------

def plot_efficiency_distribution(
    data: dict[str, list[dict]],
    output_dir: Path,
    plot_type: str = 'violin',
) -> Path:
    """
    Distribution visualization for efficiency across modes.

    Shows the full distribution of efficiency values, not just mean/std.
    This reveals important information about the spread and shape of outcomes.

    Args:
        data: Mode -> list of episode dicts with 'efficiency' key
        output_dir: Directory for output file
        plot_type: 'violin' (default), 'box', or 'histogram'

    Returns:
        Path to saved figure
    """
    modes = list(data.keys())
    labels = [MODE_NAMES.get(m, m) for m in modes]
    colors = [MODE_COLORS.get(m, '#888888') for m in modes]

    fig = go.Figure()

    for mode, label, color in zip(modes, labels, colors):
        efficiencies = [e['efficiency'] * 100 for e in data[mode]]

        if plot_type == 'violin':
            fig.add_trace(go.Violin(
                y=efficiencies,
                name=label,
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                line_color=color,
                opacity=0.7,
            ))
        elif plot_type == 'box':
            fig.add_trace(go.Box(
                y=efficiencies,
                name=label,
                marker_color=color,
                boxmean='sd',  # Show mean and std
            ))
        else:  # histogram
            fig.add_trace(go.Histogram(
                x=efficiencies,
                name=label,
                marker_color=color,
                opacity=0.6,
            ))

    if plot_type == 'histogram':
        fig.update_layout(barmode='overlay')
        xaxis_title = 'Efficiency (%)'
        yaxis_title = 'Frequency'
    else:
        xaxis_title = None
        yaxis_title = 'Efficiency (%)'
        fig.update_layout(yaxis_range=[0, 105])

    fig.update_layout(
        title=dict(
            text=f'Efficiency Distribution by Mode ({plot_type.title()} Plot)',
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=500,
        width=900,
        legend=dict(x=0.85, y=0.95),
    )

    # Reference line at 50% (random baseline)
    if plot_type != 'histogram':
        fig.add_hline(y=50, line_dash='dash', line_color='gray', opacity=0.3,
                      annotation_text='Random (50%)', annotation_position='left')

    _apply_theme(fig)

    output_path = output_dir / f'efficiency_distribution_{plot_type}.png'
    fig.write_image(str(output_path), scale=2)
    print(f"Saved: {output_path}")
    return output_path


def plot_survival_distribution(
    data: dict[str, list[dict]],
    output_dir: Path,
    plot_type: str = 'violin',
) -> Path:
    """
    Distribution visualization for survival steps across modes.

    Args:
        data: Mode -> list of episode dicts with 'survival_steps' key
        output_dir: Directory for output file
        plot_type: 'violin' (default), 'box', or 'histogram'

    Returns:
        Path to saved figure
    """
    modes = list(data.keys())
    labels = [MODE_NAMES.get(m, m) for m in modes]
    colors = [MODE_COLORS.get(m, '#888888') for m in modes]

    fig = go.Figure()

    for mode, label, color in zip(modes, labels, colors):
        # Handle both 'survival_steps' and 'survival_time' keys
        survivals = []
        for e in data[mode]:
            if 'survival_steps' in e:
                survivals.append(e['survival_steps'])
            elif 'survival_time' in e:
                survivals.append(e['survival_time'])

        if not survivals:
            continue

        if plot_type == 'violin':
            fig.add_trace(go.Violin(
                y=survivals,
                name=label,
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                line_color=color,
                opacity=0.7,
            ))
        elif plot_type == 'box':
            fig.add_trace(go.Box(
                y=survivals,
                name=label,
                marker_color=color,
                boxmean='sd',
            ))
        else:  # histogram
            fig.add_trace(go.Histogram(
                x=survivals,
                name=label,
                marker_color=color,
                opacity=0.6,
            ))

    if plot_type == 'histogram':
        fig.update_layout(barmode='overlay')
        xaxis_title = 'Survival (steps)'
        yaxis_title = 'Frequency'
    else:
        xaxis_title = None
        yaxis_title = 'Survival (steps)'

    fig.update_layout(
        title=dict(
            text=f'Survival Distribution by Mode ({plot_type.title()} Plot)',
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=500,
        width=900,
        legend=dict(x=0.85, y=0.95),
    )

    _apply_theme(fig)

    output_path = output_dir / f'survival_distribution_{plot_type}.png'
    fig.write_image(str(output_path), scale=2)
    print(f"Saved: {output_path}")
    return output_path


def plot_multi_distribution(
    data: dict[str, list[dict]],
    output_dir: Path,
    metrics: list[str] = None,
    plot_type: str = 'violin',
) -> Path:
    """
    Multi-panel distribution plot for several metrics.

    Creates a subplot grid with one distribution per metric, allowing
    side-by-side comparison of all key metrics at once.

    Args:
        data: Mode -> list of episode dicts
        output_dir: Directory for output file
        metrics: List of metric keys (default: ['efficiency', 'survival_steps', 'total_reward'])
        plot_type: 'violin' (default) or 'box'

    Returns:
        Path to saved figure
    """
    if metrics is None:
        metrics = ['efficiency', 'survival_steps', 'total_reward']

    metric_titles = {
        'efficiency': 'Efficiency',
        'survival_steps': 'Survival (steps)',
        'survival_time': 'Survival (steps)',
        'total_reward': 'Reward',
        'food_eaten': 'Food Eaten',
        'poison_eaten': 'Poison Eaten',
    }

    modes = list(data.keys())
    labels = [MODE_NAMES.get(m, m) for m in modes]
    colors = [MODE_COLORS.get(m, '#888888') for m in modes]

    n_metrics = len(metrics)
    fig = make_subplots(
        rows=1, cols=n_metrics,
        subplot_titles=[metric_titles.get(m, m) for m in metrics],
        horizontal_spacing=0.08,
    )

    for col, metric in enumerate(metrics, 1):
        for mode, label, color in zip(modes, labels, colors):
            # Handle metric aliases
            key = metric
            if key not in data[mode][0] and key == 'survival_steps':
                key = 'survival_time'

            values = [e.get(key, 0) for e in data[mode]]

            # Scale efficiency to percentage
            if metric == 'efficiency':
                values = [v * 100 for v in values]

            if plot_type == 'violin':
                fig.add_trace(go.Violin(
                    y=values,
                    name=label,
                    legendgroup=mode,
                    showlegend=(col == 1),  # Only show legend for first column
                    fillcolor=color,
                    line_color=color,
                    opacity=0.7,
                    box_visible=True,
                    meanline_visible=True,
                ), row=1, col=col)
            else:  # box
                fig.add_trace(go.Box(
                    y=values,
                    name=label,
                    legendgroup=mode,
                    showlegend=(col == 1),
                    marker_color=color,
                    boxmean='sd',
                ), row=1, col=col)

    fig.update_layout(
        title=dict(
            text='Distribution Comparison Across Metrics',
            x=0.5,
            font=dict(size=16)
        ),
        height=500,
        width=400 * n_metrics,
        legend=dict(x=0.85, y=0.95),
    )

    _apply_theme(fig)

    output_path = output_dir / f'multi_distribution_{plot_type}.png'
    fig.write_image(str(output_path), scale=2)
    print(f"Saved: {output_path}")
    return output_path


# -----------------------------------------------------------------------------
# Annotated comparison plots (with p-values, CIs, effect sizes)
# -----------------------------------------------------------------------------

def plot_efficiency_comparison_annotated(
    data: dict[str, list[dict]],
    output_dir: Path,
    show_ci: bool = True,
    show_pvalue: bool = True,
    show_effect_size: bool = True,
) -> Path:
    """
    Efficiency bar chart with statistical annotations.

    Extends basic efficiency comparison with:
    - 95% confidence intervals as error bars
    - P-value annotation between ground_truth and proxy
    - Cohen's d effect size label
    - Significance stars

    Args:
        data: Mode -> list of episode dicts with 'efficiency' key
        output_dir: Directory for output file
        show_ci: Show confidence intervals (default True)
        show_pvalue: Show p-value annotation (default True)
        show_effect_size: Show Cohen's d (default True)

    Returns:
        Path to saved figure
    """
    from goodharts.analysis.stats_helpers import compute_comparison, format_p_value

    modes = list(data.keys())
    labels = [MODE_NAMES.get(m, m) for m in modes]
    colors = [MODE_COLORS.get(m, '#888888') for m in modes]

    # Compute statistics
    efficiencies = {m: [e['efficiency'] * 100 for e in data[m]] for m in modes}
    means = [np.mean(efficiencies[m]) for m in modes]

    # Compute CIs
    if show_ci:
        from goodharts.analysis.stats_helpers import compute_confidence_interval
        cis = [compute_confidence_interval(efficiencies[m]) for m in modes]
        errors = [means[i] - cis[i][0] for i in range(len(modes))]
    else:
        errors = [np.std(efficiencies[m]) for m in modes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=means,
        error_y=dict(type='data', array=errors, visible=True),
        marker_color=colors,
        text=[f'{m:.1f}%' for m in means],
        textposition='outside',
    ))

    # Add statistical annotations if comparing ground_truth and proxy
    annotations = []
    if 'ground_truth' in modes and 'proxy' in modes and (show_pvalue or show_effect_size):
        gt_vals = efficiencies['ground_truth']
        proxy_vals = efficiencies['proxy']
        comparison = compute_comparison(
            gt_vals, proxy_vals,
            metric='efficiency',
            group_a='ground_truth',
            group_b='proxy'
        )

        gt_idx = modes.index('ground_truth')
        proxy_idx = modes.index('proxy')

        # Build annotation text
        annotation_parts = []
        if show_pvalue:
            p_str = format_p_value(comparison.p_value)
            annotation_parts.append(f"p = {p_str} {comparison.significance_stars}")
        if show_effect_size:
            annotation_parts.append(f"d = {comparison.cohens_d:.2f} ({comparison.effect_magnitude})")

        annotation_text = '<br>'.join(annotation_parts)

        # Position annotation between the two bars
        x_mid = (gt_idx + proxy_idx) / 2
        y_pos = max(means[gt_idx], means[proxy_idx]) + 10

        annotations.append(dict(
            x=x_mid,
            y=y_pos,
            text=annotation_text,
            showarrow=False,
            font=dict(size=12, color=THEME['text']),
            bgcolor='rgba(0,0,0,0.5)',
            borderpad=4,
        ))

        # Add bracket line connecting the bars
        fig.add_shape(
            type='line',
            x0=gt_idx, y0=means[gt_idx] + errors[gt_idx] + 2,
            x1=gt_idx, y1=y_pos - 3,
            line=dict(color=THEME['text'], width=1),
        )
        fig.add_shape(
            type='line',
            x0=proxy_idx, y0=means[proxy_idx] + errors[proxy_idx] + 2,
            x1=proxy_idx, y1=y_pos - 3,
            line=dict(color=THEME['text'], width=1),
        )
        fig.add_shape(
            type='line',
            x0=gt_idx, y0=y_pos - 3,
            x1=proxy_idx, y1=y_pos - 3,
            line=dict(color=THEME['text'], width=1),
        )

    error_type = '95% CI' if show_ci else 'Std Dev'
    fig.update_layout(
        title=dict(
            text=f'Consumption Efficiency by Mode<br><sub>Error bars: {error_type}</sub>',
            x=0.5,
            font=dict(size=16)
        ),
        yaxis_title='Efficiency (%)',
        yaxis_range=[0, 115],
        showlegend=False,
        height=550,
        width=800,
        annotations=annotations,
    )

    # Reference line at 50% (random baseline)
    fig.add_hline(y=50, line_dash='dash', line_color='gray', opacity=0.3,
                  annotation_text='Random (50%)', annotation_position='left')

    _apply_theme(fig)

    output_path = output_dir / 'efficiency_comparison_annotated.png'
    fig.write_image(str(output_path), scale=2)
    print(f"Saved: {output_path}")
    return output_path


def plot_goodhart_summary_annotated(
    data: dict[str, list[dict]],
    output_dir: Path,
) -> Path:
    """
    Enhanced Goodhart summary with full statistical context.

    Shows side-by-side GT vs Proxy with:
    - CI error bars
    - P-value brackets
    - Effect size labels
    - Goodhart Failure Index prominently displayed

    Args:
        data: Mode -> list of episode dicts
        output_dir: Directory for output file

    Returns:
        Path to saved figure
    """
    from goodharts.analysis.stats_helpers import (
        compute_comparison, format_p_value, compute_goodhart_failure_index
    )

    if 'ground_truth' not in data or 'proxy' not in data:
        print("Skipping annotated Goodhart summary: need both ground_truth and proxy modes")
        return None

    gt_data = data['ground_truth']
    proxy_data = data['proxy']

    # Compute GFI
    gt_eff = np.mean([e['efficiency'] for e in gt_data])
    proxy_eff = np.mean([e['efficiency'] for e in proxy_data])
    gfi = compute_goodhart_failure_index(gt_eff, proxy_eff)

    metrics = [
        ('efficiency', 'Efficiency (%)', True),  # (key, title, scale_to_percent)
        ('poison_eaten', 'Poison Consumed', False),
        ('total_reward', 'Reward', False),
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[m[1] for m in metrics],
        horizontal_spacing=0.12,
    )

    for col, (key, title, scale) in enumerate(metrics, 1):
        gt_vals = [e[key] for e in gt_data]
        proxy_vals = [e[key] for e in proxy_data]

        if scale:
            gt_vals = [v * 100 for v in gt_vals]
            proxy_vals = [v * 100 for v in proxy_vals]

        # Compute comparison
        comparison = compute_comparison(gt_vals, proxy_vals, metric=key)

        gt_mean = np.mean(gt_vals)
        proxy_mean = np.mean(proxy_vals)

        # Use CI half-width for error bars
        gt_ci = comparison.ci_a
        proxy_ci = comparison.ci_b
        gt_err = gt_mean - gt_ci[0]
        proxy_err = proxy_mean - proxy_ci[0]

        suffix = '%' if scale else ''

        fig.add_trace(go.Bar(
            x=['Ground Truth', 'Proxy'],
            y=[gt_mean, proxy_mean],
            error_y=dict(type='data', array=[gt_err, proxy_err], visible=True),
            marker_color=['#22c79a', '#ff6b6b'],
            text=[f'{gt_mean:.1f}{suffix}', f'{proxy_mean:.1f}{suffix}'],
            textposition='outside',
            showlegend=False,
        ), row=1, col=col)

        # Add p-value annotation above the bars
        y_max = max(gt_mean + gt_err, proxy_mean + proxy_err)
        p_str = format_p_value(comparison.p_value)
        fig.add_annotation(
            x=0.5, y=y_max * 1.15,
            text=f"p={p_str} {comparison.significance_stars}<br>d={comparison.cohens_d:.2f}",
            showarrow=False,
            font=dict(size=10, color=THEME['text']),
            row=1, col=col,
        )

    # Add GFI as a prominent title annotation
    fig.update_layout(
        title=dict(
            text=(f"Goodhart's Law: Optimizing Proxy Metrics Leads to Failure<br>"
                  f"<b>Goodhart Failure Index: {gfi:.1%}</b>"),
            x=0.5,
            font=dict(size=16)
        ),
        height=500,
        width=1200,
    )

    _apply_theme(fig)

    output_path = output_dir / 'goodhart_summary_annotated.png'
    fig.write_image(str(output_path), scale=2)
    print(f"Saved: {output_path}")
    return output_path


# -----------------------------------------------------------------------------
# JSON data loading (for multi-run results)
# -----------------------------------------------------------------------------

def load_json_results(path: str) -> dict[str, list[dict]]:
    """
    Load evaluation results from JSON format.

    Handles both single-run and multi-run JSON structures.
    Converts death events to the same format as CSV results.

    Args:
        path: Path to JSON file

    Returns:
        Mode -> list of episode/death dicts
    """
    with open(path, 'r') as f:
        data = json.load(f)

    results = defaultdict(list)

    # Handle multi-mode structure (from scripts/evaluate.py)
    if 'results' in data:
        for mode, mode_data in data['results'].items():
            if 'deaths' in mode_data:
                for death in mode_data['deaths']:
                    results[mode].append({
                        'total_reward': death.get('total_reward', 0),
                        'food_eaten': death.get('food_eaten', 0),
                        'poison_eaten': death.get('poison_eaten', 0),
                        'survival_steps': death.get('survival_time', 0),
                        'efficiency': death.get('efficiency', 0),
                    })
    # Handle single-mode structure
    elif 'deaths' in data:
        mode = data.get('mode', 'unknown')
        for death in data['deaths']:
            results[mode].append({
                'total_reward': death.get('total_reward', 0),
                'food_eaten': death.get('food_eaten', 0),
                'poison_eaten': death.get('poison_eaten', 0),
                'survival_steps': death.get('survival_time', 0),
                'efficiency': death.get('efficiency', 0),
            })

    return dict(results)


def generate_all_figures(
    data: dict[str, list[dict]],
    output_dir: Path,
    annotated: bool = True,
    distributions: bool = True,
) -> list[Path]:
    """
    Generate all standard figures for a Goodhart experiment.

    Args:
        data: Mode -> list of episode dicts
        output_dir: Directory for figures
        annotated: Include annotated plots with statistics
        distributions: Include distribution plots

    Returns:
        List of paths to generated figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Basic plots
    plot_reward_comparison(data, output_dir)
    paths.append(output_dir / 'reward_comparison.png')

    plot_consumption_comparison(data, output_dir)
    paths.append(output_dir / 'consumption_comparison.png')

    plot_efficiency_comparison(data, output_dir)
    paths.append(output_dir / 'efficiency_comparison.png')

    plot_goodhart_summary(data, output_dir)
    paths.append(output_dir / 'goodhart_summary.png')

    # Annotated plots
    if annotated:
        path = plot_efficiency_comparison_annotated(data, output_dir)
        if path:
            paths.append(path)

        path = plot_goodhart_summary_annotated(data, output_dir)
        if path:
            paths.append(path)

    # Distribution plots
    if distributions:
        for plot_type in ['violin', 'box']:
            path = plot_efficiency_distribution(data, output_dir, plot_type)
            paths.append(path)

            path = plot_survival_distribution(data, output_dir, plot_type)
            paths.append(path)

        path = plot_multi_distribution(data, output_dir)
        paths.append(path)

    return paths


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations')
    parser.add_argument('--input', default='analysis/results.csv',
                        help='Input CSV or JSON from analysis scripts')
    parser.add_argument('--output-dir', default='analysis/figures',
                        help='Output directory for figures')
    parser.add_argument('--annotated', action='store_true',
                        help='Include annotated plots with p-values and effect sizes')
    parser.add_argument('--distributions', action='store_true',
                        help='Include distribution plots (violin, box)')
    parser.add_argument('--all', action='store_true',
                        help='Generate all plot types (same as --annotated --distributions)')
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run analysis scripts first to generate results.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results from {args.input}")

    # Detect file format
    if args.input.endswith('.json'):
        data = load_json_results(args.input)
    else:
        data = load_results(args.input)

    print(f"Found {len(data)} modes: {list(data.keys())}")
    for mode, episodes in data.items():
        print(f"  {mode}: {len(episodes)} episodes/deaths")
    print()

    # Determine what to generate
    annotated = args.annotated or args.all
    distributions = args.distributions or args.all

    # Generate figures
    paths = generate_all_figures(data, output_dir, annotated, distributions)

    print(f"\nGenerated {len(paths)} figures in {output_dir}/")


if __name__ == '__main__':
    main()
