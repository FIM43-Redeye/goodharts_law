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
import signal
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Default timeout for figure rendering (seconds)
FIGURE_TIMEOUT = 300  # 5 minutes should be plenty even for large datasets

# Maximum points to render in distribution plots (violin/box)
# Beyond this, we downsample to preserve distribution shape without killing kaleido
MAX_DISTRIBUTION_POINTS = 10000


def _downsample_for_distribution(values: list, max_points: int = MAX_DISTRIBUTION_POINTS) -> list:
    """
    Downsample large datasets for distribution plots.

    Violin and box plots don't benefit from millions of points - the visual
    distribution is essentially identical with 10k points. This prevents
    kaleido from hanging on large datasets.

    Uses random sampling which preserves distribution shape for large n.

    Args:
        values: List of numeric values
        max_points: Maximum points to keep (default: 10000)

    Returns:
        Original list if small enough, otherwise random sample
    """
    if len(values) <= max_points:
        return values

    # Use numpy for efficient random sampling
    indices = np.random.choice(len(values), size=max_points, replace=False)
    return [values[i] for i in indices]


class FigureTimeoutError(Exception):
    """Raised when figure rendering exceeds timeout."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for figure timeout."""
    raise FigureTimeoutError("Figure rendering timed out")


def save_figure(fig: go.Figure, output_path: Path, timeout: int = FIGURE_TIMEOUT) -> bool:
    """
    Save figure with timeout protection.

    Uses SIGALRM on Unix to prevent kaleido hangs from blocking indefinitely.
    Falls back to direct write_image on Windows or if signal fails.

    Args:
        fig: Plotly figure to save
        output_path: Path for output PNG
        timeout: Maximum seconds to wait (default: 300)

    Returns:
        True if saved successfully, False if timed out or failed
    """
    try:
        # Set up timeout (Unix only)
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

        try:
            fig.write_image(str(output_path), scale=2)
            signal.alarm(0)  # Cancel the alarm
            print(f"Saved: {output_path}")
            return True
        except FigureTimeoutError:
            print(f"TIMEOUT: {output_path} (exceeded {timeout}s, skipping)")
            return False
        finally:
            signal.signal(signal.SIGALRM, old_handler)
            signal.alarm(0)
    except (AttributeError, ValueError):
        # Windows or signal not available - fall back to direct call
        fig.write_image(str(output_path), scale=2)
        print(f"Saved: {output_path}")
        return True


# Import canonical colors and theme from single source of truth
from goodharts.visualization.components import MODE_COLORS, THEME

# Friendly display names for report output
MODE_NAMES = {
    'ground_truth': 'Ground Truth',
    'ground_truth_handhold': 'Ground Truth (Guided)',
    'ground_truth_blinded': 'Ground Truth (Blinded)',
    'proxy_mortal': 'Proxy (Mortal)',
    'proxy': 'Proxy (Immortal)',
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


def plot_survival_comparison(
    data: dict[str, list[dict]],
    output_dir: Path,
    aggregates: dict[str, dict] = None,
):
    """Bar chart comparing mean survival time across modes."""
    modes = list(data.keys())
    colors = [MODE_COLORS.get(m, '#888888') for m in modes]
    labels = [MODE_NAMES.get(m, m) for m in modes]

    # Use aggregate survival_mean if available (more accurate)
    if aggregates:
        means = [aggregates.get(m, {}).get('survival_mean', 0) for m in modes]
        # Use CI from aggregates if available
        errors = []
        for m in modes:
            ci = aggregates.get(m, {}).get('survival_ci')
            if ci and len(ci) == 2:
                mean_val = aggregates.get(m, {}).get('survival_mean', 0)
                errors.append(mean_val - ci[0])
            else:
                errors.append(0)
    else:
        # Fallback to per-death mean
        means = [np.mean([e.get('survival_steps', e.get('survival_time', 0)) for e in data[m]]) for m in modes]
        stds = [np.std([e.get('survival_steps', e.get('survival_time', 0)) for e in data[m]]) for m in modes]
        errors = stds

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=means,
        error_y=dict(type='data', array=errors, visible=True) if any(e > 0 for e in errors) else None,
        marker_color=colors,
        text=[f'{m:.0f}' for m in means],
        textposition='outside',
    ))

    fig.update_layout(
        title=dict(text='Mean Survival Time by Mode', x=0.5,
                  font=dict(size=16)),
        yaxis_title='Steps Until Death',
        showlegend=False,
        height=500,
        width=800,
    )

    _apply_theme(fig)

    output_path = output_dir / 'survival_comparison.png'
    save_figure(fig, output_path)


def plot_consumption_comparison(
    data: dict[str, list[dict]],
    output_dir: Path,
    aggregates: dict[str, dict] = None,
):
    """Grouped bar chart comparing food vs poison consumption rates.

    Uses per-1000-steps rates from aggregates when available, which properly
    accounts for different death rates across modes. Per-death averages are
    misleading because ground_truth agents live ~4000 steps per death while
    proxy agents live ~25 steps per death.
    """
    modes = list(data.keys())
    labels = [MODE_NAMES.get(m, m) for m in modes]

    # Use aggregate rates if available (much more accurate)
    if aggregates:
        food_rates = [aggregates.get(m, {}).get('food_per_1k_steps', 0) for m in modes]
        poison_rates = [aggregates.get(m, {}).get('poison_per_1k_steps', 0) for m in modes]
        y_label = 'Items Consumed per 1000 Steps'
    else:
        # Fallback to per-death averages (less accurate but better than nothing)
        food_rates = [np.mean([e['food_eaten'] for e in data[m]]) for m in modes]
        poison_rates = [np.mean([e['poison_eaten'] for e in data[m]]) for m in modes]
        y_label = 'Items Consumed per Episode (biased - see aggregates)'

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Food',
        x=labels,
        y=food_rates,
        marker_color='#22c79a',
        text=[f'{m:.1f}' for m in food_rates],
        textposition='outside',
    ))

    fig.add_trace(go.Bar(
        name='Poison',
        x=labels,
        y=poison_rates,
        marker_color='#ff6b6b',
        text=[f'{m:.1f}' for m in poison_rates],
        textposition='outside',
    ))

    fig.update_layout(
        title=dict(text='Food vs Poison Consumption Rates by Mode', x=0.5,
                  font=dict(size=16)),
        yaxis_title=y_label,
        barmode='group',
        legend=dict(x=0.85, y=0.95),
        height=500,
        width=800,
    )

    _apply_theme(fig)

    output_path = output_dir / 'consumption_comparison.png'
    save_figure(fig, output_path)


def plot_efficiency_comparison(
    data: dict[str, list[dict]],
    output_dir: Path,
    aggregates: dict[str, dict] = None,
):
    """Bar chart comparing efficiency (food / total consumed) across modes."""
    modes = list(data.keys())

    # Use aggregate efficiency if available (more accurate than per-death mean)
    if aggregates:
        means = [aggregates.get(m, {}).get('overall_efficiency', 0) * 100 for m in modes]
        # No std for aggregate efficiency - it's a ratio, not a sample mean
        stds = [0 for _ in modes]
    else:
        # Fallback to per-death mean (less accurate due to short-lived deaths bias)
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
    save_figure(fig, output_path)


def plot_goodhart_summary(data: dict[str, list[dict]], output_dir: Path, aggregates: dict[str, dict] = None):
    """
    Summary figure showing Goodhart's Law effect.

    Side-by-side comparison of ground_truth vs proxy on key metrics.
    Uses aggregate statistics (not per-death averages) for accurate comparison.
    """
    if 'ground_truth' not in data or 'proxy' not in data:
        print("Skipping Goodhart summary: need both ground_truth and proxy modes")
        return

    # Define metrics to show - all use aggregates for accuracy
    # (per-death averages are misleading due to different death rates)
    metrics = [
        ('efficiency', 'Efficiency (%)', True),      # Key Goodhart metric
        ('survival', 'Survival (steps)', False),     # How long agents live
        ('poison_rate', 'Poison per 1k Steps', False),  # Poison consumption rate
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[m[1] for m in metrics],
        horizontal_spacing=0.1,
    )

    for i, (key, title, is_percent) in enumerate(metrics, 1):
        if aggregates:
            gt_agg = aggregates.get('ground_truth', {})
            proxy_agg = aggregates.get('proxy', {})

            if key == 'efficiency':
                gt_mean = gt_agg.get('overall_efficiency', 0) * 100
                proxy_mean = proxy_agg.get('overall_efficiency', 0) * 100
            elif key == 'survival':
                gt_mean = gt_agg.get('survival_mean', 0)
                proxy_mean = proxy_agg.get('survival_mean', 0)
            elif key == 'poison_rate':
                gt_mean = gt_agg.get('poison_per_1k_steps', 0)
                proxy_mean = proxy_agg.get('poison_per_1k_steps', 0)
            else:
                gt_mean, proxy_mean = 0, 0
        else:
            # Fallback to per-death (less accurate)
            gt_data_list = data['ground_truth']
            proxy_data_list = data['proxy']

            if key == 'efficiency':
                gt_mean = np.mean([e['efficiency'] for e in gt_data_list]) * 100
                proxy_mean = np.mean([e['efficiency'] for e in proxy_data_list]) * 100
            elif key == 'survival':
                gt_mean = np.mean([e.get('survival_steps', e.get('survival_time', 0)) for e in gt_data_list])
                proxy_mean = np.mean([e.get('survival_steps', e.get('survival_time', 0)) for e in proxy_data_list])
            elif key == 'poison_rate':
                gt_mean = np.mean([e['poison_eaten'] for e in gt_data_list])
                proxy_mean = np.mean([e['poison_eaten'] for e in proxy_data_list])
            else:
                gt_mean, proxy_mean = 0, 0

        suffix = '%' if is_percent else ''

        # Format numbers appropriately
        if gt_mean >= 100 or proxy_mean >= 100:
            text_fmt = [f'{gt_mean:.0f}{suffix}', f'{proxy_mean:.0f}{suffix}']
        elif key == 'poison_rate':
            text_fmt = [f'{gt_mean:.2f}', f'{proxy_mean:.1f}']
        else:
            text_fmt = [f'{gt_mean:.1f}{suffix}', f'{proxy_mean:.1f}{suffix}']

        fig.add_trace(go.Bar(
            x=['Ground Truth', 'Proxy'],
            y=[gt_mean, proxy_mean],
            marker_color=['#22c79a', '#ff6b6b'],
            text=text_fmt,
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
    save_figure(fig, output_path)


# -----------------------------------------------------------------------------
# Distribution plots
# -----------------------------------------------------------------------------

def plot_efficiency_distribution(
    data: dict[str, list[dict]],
    output_dir: Path,
    plot_type: str = 'violin',
    survivors: dict[str, list[dict]] = None,
) -> Path:
    """
    Distribution visualization for efficiency across modes.

    Shows the full distribution of efficiency values, not just mean/std.
    Includes both deaths AND survivors for a complete picture.

    Args:
        data: Mode -> list of episode dicts with 'efficiency' key
        output_dir: Directory for output file
        plot_type: 'violin' (default), 'box', or 'histogram'
        survivors: Optional survivor snapshots (right-censored data)

    Returns:
        Path to saved figure
    """
    modes = list(data.keys())
    labels = [MODE_NAMES.get(m, m) for m in modes]
    colors = [MODE_COLORS.get(m, '#888888') for m in modes]

    fig = go.Figure()

    for mode, label, color in zip(modes, labels, colors):
        # Combine deaths and survivors for complete distribution
        efficiencies = [e['efficiency'] * 100 for e in data[mode]]
        if survivors and mode in survivors:
            survivor_effs = [s['efficiency'] * 100 for s in survivors[mode]]
            efficiencies.extend(survivor_effs)

        # Downsample for rendering performance (distribution shape preserved)
        efficiencies = _downsample_for_distribution(efficiencies)

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

    # Note if survivors included
    title_suffix = ' (incl. survivors)' if survivors else ''
    fig.update_layout(
        title=dict(
            text=f'Efficiency Distribution by Mode ({plot_type.title()} Plot){title_suffix}',
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
    save_figure(fig, output_path)
    return output_path


def plot_survival_distribution(
    data: dict[str, list[dict]],
    output_dir: Path,
    plot_type: str = 'violin',
    survivors: dict[str, list[dict]] = None,
) -> Path:
    """
    Distribution visualization for survival steps across modes.

    CRITICAL: Includes survivors (agents still alive at evaluation end).
    Without survivors, ground_truth would show only the rare deaths,
    completely missing the typical ~4000 step lifespans.

    Args:
        data: Mode -> list of episode dicts with 'survival_steps' key
        output_dir: Directory for output file
        plot_type: 'violin' (default), 'box', or 'histogram'
        survivors: Optional survivor snapshots (right-censored data)

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

        # Add survivors (they lived AT LEAST this long - right-censored)
        if survivors and mode in survivors:
            for s in survivors[mode]:
                if 'survival_steps' in s:
                    survivals.append(s['survival_steps'])
                elif 'survival_time' in s:
                    survivals.append(s['survival_time'])

        if not survivals:
            continue

        # Downsample for rendering performance (distribution shape preserved)
        survivals = _downsample_for_distribution(survivals)

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

    # Note if survivors included
    title_suffix = ' (incl. survivors)' if survivors else ''
    fig.update_layout(
        title=dict(
            text=f'Survival Distribution by Mode ({plot_type.title()} Plot){title_suffix}',
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
    save_figure(fig, output_path)
    return output_path


def plot_multi_distribution(
    data: dict[str, list[dict]],
    output_dir: Path,
    metrics: list[str] = None,
    plot_type: str = 'violin',
    survivors: dict[str, list[dict]] = None,
) -> Path:
    """
    Multi-panel distribution plot for several metrics.

    Creates a subplot grid with one distribution per metric, allowing
    side-by-side comparison of all key metrics at once. Includes survivors
    for efficiency and survival metrics.

    Args:
        data: Mode -> list of episode dicts
        output_dir: Directory for output file
        metrics: List of metric keys (default: ['efficiency', 'survival_steps'])
        plot_type: 'violin' (default) or 'box'
        survivors: Optional survivor snapshots (right-censored data)

    Returns:
        Path to saved figure
    """
    # Don't include total_reward in multi-distribution - it's mode-specific
    if metrics is None:
        metrics = ['efficiency', 'survival_steps']

    metric_titles = {
        'efficiency': 'Efficiency',
        'survival_steps': 'Survival (steps)',
        'survival_time': 'Survival (steps)',
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
            if data[mode] and key not in data[mode][0] and key == 'survival_steps':
                key = 'survival_time'

            values = [e.get(key, 0) for e in data[mode]]

            # Add survivors for efficiency and survival metrics
            if survivors and mode in survivors and metric in ('efficiency', 'survival_steps', 'survival_time'):
                surv_key = key
                if survivors[mode] and surv_key not in survivors[mode][0] and surv_key == 'survival_steps':
                    surv_key = 'survival_time'
                survivor_vals = [s.get(surv_key, 0) for s in survivors[mode]]
                values.extend(survivor_vals)

            # Scale efficiency to percentage
            if metric == 'efficiency':
                values = [v * 100 for v in values]

            # Downsample for rendering performance (distribution shape preserved)
            values = _downsample_for_distribution(values)

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
    save_figure(fig, output_path)
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
    aggregates: dict[str, dict] = None,
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
        aggregates: Optional aggregate stats per mode

    Returns:
        Path to saved figure
    """
    from goodharts.analysis.stats_helpers import compute_comparison, format_p_value

    modes = list(data.keys())
    labels = [MODE_NAMES.get(m, m) for m in modes]
    colors = [MODE_COLORS.get(m, '#888888') for m in modes]

    # Use aggregate efficiency if available (more accurate than per-death mean)
    if aggregates:
        means = [aggregates.get(m, {}).get('overall_efficiency', 0) * 100 for m in modes]
        # Use CI from aggregates if available, otherwise no error bars
        errors = []
        for m in modes:
            ci = aggregates.get(m, {}).get('efficiency_ci')
            if ci and len(ci) == 2:
                errors.append((means[modes.index(m)] - ci[0] * 100))
            else:
                errors.append(0)
    else:
        # Fallback to per-death statistics
        efficiencies = {m: [e['efficiency'] * 100 for e in data[m]] for m in modes}
        means = [np.mean(efficiencies[m]) for m in modes]
        if show_ci:
            from goodharts.analysis.stats_helpers import compute_confidence_interval
            cis = [compute_confidence_interval(efficiencies[m]) for m in modes]
            errors = [means[i] - cis[i][0] for i in range(len(modes))]
        else:
            errors = [np.std(efficiencies[m]) for m in modes]

    # For statistical comparison, still use per-death data (need samples for t-test)
    efficiencies = {m: [e['efficiency'] * 100 for e in data[m]] for m in modes}

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
    save_figure(fig, output_path)
    return output_path


def plot_goodhart_summary_annotated(
    data: dict[str, list[dict]],
    output_dir: Path,
    aggregates: dict[str, dict] = None,
) -> Path:
    """
    Enhanced Goodhart summary with full statistical context.

    Shows side-by-side GT vs Proxy with:
    - Aggregate statistics (not per-death averages)
    - CI error bars where available
    - Survival Collapse Ratio prominently displayed

    Args:
        data: Mode -> list of episode dicts
        output_dir: Directory for output file
        aggregates: Optional aggregate stats per mode

    Returns:
        Path to saved figure
    """
    if 'ground_truth' not in data or 'proxy' not in data:
        print("Skipping annotated Goodhart summary: need both ground_truth and proxy modes")
        return None

    # Compute Survival Collapse Ratio from aggregates or fallback to per-death
    if aggregates:
        gt_agg = aggregates.get('ground_truth', {})
        px_agg = aggregates.get('proxy', {})
        gt_survival = gt_agg.get('survival_mean', 0)
        proxy_survival = px_agg.get('survival_mean', 0)
    else:
        gt_survival = np.mean([e.get('survival_steps', e.get('survival_time', 0))
                              for e in data['ground_truth']])
        proxy_survival = np.mean([e.get('survival_steps', e.get('survival_time', 0))
                                 for e in data['proxy']])
    # SCR: how many times longer ground truth survives vs proxy
    scr = gt_survival / proxy_survival if proxy_survival > 0 else float('inf')

    # Metrics to display - all use aggregates for accuracy
    metrics = [
        ('efficiency', 'Efficiency (%)', True),
        ('survival', 'Survival (steps)', False),
        ('poison_rate', 'Poison per 1k Steps', False),
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[m[1] for m in metrics],
        horizontal_spacing=0.12,
    )

    for col, (key, title, is_percent) in enumerate(metrics, 1):
        # Get values and CIs from aggregates if available
        if aggregates:
            gt_agg = aggregates.get('ground_truth', {})
            px_agg = aggregates.get('proxy', {})

            if key == 'efficiency':
                gt_mean = gt_agg.get('overall_efficiency', 0) * 100
                proxy_mean = px_agg.get('overall_efficiency', 0) * 100
                gt_ci = gt_agg.get('efficiency_ci')
                px_ci = px_agg.get('efficiency_ci')
                if gt_ci:
                    gt_err = gt_mean - gt_ci[0] * 100
                    px_err = proxy_mean - px_ci[0] * 100
                else:
                    gt_err, px_err = 0, 0
            elif key == 'survival':
                gt_mean = gt_agg.get('survival_mean', 0)
                proxy_mean = px_agg.get('survival_mean', 0)
                gt_ci = gt_agg.get('survival_ci')
                px_ci = px_agg.get('survival_ci')
                if gt_ci:
                    gt_err = gt_mean - gt_ci[0]
                    px_err = proxy_mean - px_ci[0]
                else:
                    gt_err, px_err = 0, 0
            elif key == 'poison_rate':
                gt_mean = gt_agg.get('poison_per_1k_steps', 0)
                proxy_mean = px_agg.get('poison_per_1k_steps', 0)
                gt_err, px_err = 0, 0  # No CI for rates yet
            else:
                gt_mean, proxy_mean = 0, 0
                gt_err, px_err = 0, 0
        else:
            # Fallback to per-death averages (less accurate)
            gt_data_list = data['ground_truth']
            px_data_list = data['proxy']

            if key == 'efficiency':
                gt_mean = np.mean([e['efficiency'] for e in gt_data_list]) * 100
                proxy_mean = np.mean([e['efficiency'] for e in px_data_list]) * 100
            elif key == 'survival':
                gt_mean = np.mean([e.get('survival_steps', e.get('survival_time', 0)) for e in gt_data_list])
                proxy_mean = np.mean([e.get('survival_steps', e.get('survival_time', 0)) for e in px_data_list])
            elif key == 'poison_rate':
                gt_mean = np.mean([e['poison_eaten'] for e in gt_data_list])
                proxy_mean = np.mean([e['poison_eaten'] for e in px_data_list])
            else:
                gt_mean, proxy_mean = 0, 0
            gt_err, px_err = 0, 0

        suffix = '%' if is_percent else ''

        # Format numbers appropriately
        if gt_mean >= 100 or proxy_mean >= 100:
            text_fmt = [f'{gt_mean:.0f}{suffix}', f'{proxy_mean:.0f}{suffix}']
        elif key == 'poison_rate':
            text_fmt = [f'{gt_mean:.2f}', f'{proxy_mean:.1f}']
        else:
            text_fmt = [f'{gt_mean:.1f}{suffix}', f'{proxy_mean:.1f}{suffix}']

        fig.add_trace(go.Bar(
            x=['Ground Truth', 'Proxy'],
            y=[gt_mean, proxy_mean],
            error_y=dict(type='data', array=[gt_err, px_err], visible=True) if gt_err > 0 else None,
            marker_color=['#22c79a', '#ff6b6b'],
            text=text_fmt,
            textposition='outside',
            showlegend=False,
        ), row=1, col=col)

    # Add SCR as a prominent title annotation
    fig.update_layout(
        title=dict(
            text=(f"Goodhart's Law: Optimizing Proxy Metrics Leads to Failure<br>"
                  f"<b>Survival Collapse Ratio: {scr:.0f}x</b>"),
            x=0.5,
            font=dict(size=16)
        ),
        height=500,
        width=1200,
    )

    _apply_theme(fig)

    output_path = output_dir / 'goodhart_summary_annotated.png'
    save_figure(fig, output_path)
    return output_path


# -----------------------------------------------------------------------------
# JSON data loading (for multi-run results)
# -----------------------------------------------------------------------------

def load_json_results(path: str) -> tuple[dict[str, list[dict]], dict[str, dict], dict[str, list[dict]]]:
    """
    Load evaluation results from JSON format.

    Handles both single-run and multi-run JSON structures.
    Converts death events and survivors to the same format as CSV results.

    Args:
        path: Path to JSON file

    Returns:
        Tuple of (deaths_by_mode, aggregates_by_mode, survivors_by_mode)
        - deaths_by_mode: Mode -> list of per-death dicts
        - aggregates_by_mode: Mode -> aggregate stats dict
        - survivors_by_mode: Mode -> list of survivor snapshot dicts (censored data)
    """
    with open(path, 'r') as f:
        data = json.load(f)

    results = defaultdict(list)
    aggregates = {}
    survivors = defaultdict(list)

    def extract_event(event: dict, is_survivor: bool = False) -> dict:
        """Extract death/survivor data, computing efficiency from food/poison."""
        food = event.get('food_eaten', 0)
        poison = event.get('poison_eaten', 0)
        total = food + poison
        # Compute efficiency (property not serialized to JSON)
        efficiency = food / total if total > 0 else 1.0
        return {
            'total_reward': event.get('total_reward', 0),
            'food_eaten': food,
            'poison_eaten': poison,
            'survival_steps': event.get('survival_time', 0),
            'efficiency': efficiency,
            'censored': is_survivor,  # True if still alive (lower bound data)
        }

    # Handle multi-mode structure (from scripts/evaluate.py)
    if 'results' in data:
        for mode, mode_data in data['results'].items():
            # Always create a results entry for the mode (even if 0 deaths)
            # This ensures modes with only survivors still appear
            if mode not in results:
                results[mode] = []
            if 'deaths' in mode_data:
                for death in mode_data['deaths']:
                    results[mode].append(extract_event(death, is_survivor=False))
            if 'survivors' in mode_data:
                for survivor in mode_data['survivors']:
                    survivors[mode].append(extract_event(survivor, is_survivor=True))
            if 'aggregates' in mode_data:
                aggregates[mode] = mode_data['aggregates']
    # Handle single-mode structure
    elif 'deaths' in data:
        mode = data.get('mode', 'unknown')
        for death in data['deaths']:
            results[mode].append(extract_event(death, is_survivor=False))
        if 'survivors' in data:
            for survivor in data['survivors']:
                survivors[mode].append(extract_event(survivor, is_survivor=True))
        if 'aggregates' in data:
            aggregates[mode] = data['aggregates']

    return dict(results), aggregates, dict(survivors)


def generate_all_figures(
    data: dict[str, list[dict]],
    output_dir: Path,
    annotated: bool = True,
    distributions: bool = True,
    aggregates: dict[str, dict] = None,
    survivors: dict[str, list[dict]] = None,
) -> list[Path]:
    """
    Generate all standard figures for a Goodhart experiment.

    Args:
        data: Mode -> list of episode/death dicts
        output_dir: Directory for figures
        annotated: Include annotated plots with statistics
        distributions: Include distribution plots
        aggregates: Optional aggregate stats per mode
        survivors: Optional survivor snapshot dicts per mode (for distributions)

    Returns:
        List of paths to generated figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Basic plots
    plot_survival_comparison(data, output_dir, aggregates=aggregates)
    paths.append(output_dir / 'survival_comparison.png')

    plot_consumption_comparison(data, output_dir, aggregates=aggregates)
    paths.append(output_dir / 'consumption_comparison.png')

    plot_efficiency_comparison(data, output_dir, aggregates=aggregates)
    paths.append(output_dir / 'efficiency_comparison.png')

    plot_goodhart_summary(data, output_dir, aggregates=aggregates)
    paths.append(output_dir / 'goodhart_summary.png')

    # Annotated plots
    if annotated:
        path = plot_efficiency_comparison_annotated(data, output_dir, aggregates=aggregates)
        if path:
            paths.append(path)

        path = plot_goodhart_summary_annotated(data, output_dir, aggregates=aggregates)
        if path:
            paths.append(path)

    # Distribution plots (include survivors for complete picture)
    # Note: efficiency violin and multi-distribution violin produce unfortunate
    # shapes that aren't visually useful - box plots communicate the same info better
    if distributions:
        for plot_type in ['violin', 'box']:
            # Skip efficiency violin - produces unfortunate shapes
            if plot_type != 'violin':
                path = plot_efficiency_distribution(data, output_dir, plot_type, survivors=survivors)
                paths.append(path)

            path = plot_survival_distribution(data, output_dir, plot_type, survivors=survivors)
            paths.append(path)

        # Skip multi_distribution violin - same issue as efficiency violin

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
    aggregates = None
    survivors = None
    if args.input.endswith('.json'):
        data, aggregates, survivors = load_json_results(args.input)
    else:
        data = load_results(args.input)

    print(f"Found {len(data)} modes: {list(data.keys())}")
    for mode, episodes in data.items():
        n_survivors = len(survivors.get(mode, [])) if survivors else 0
        print(f"  {mode}: {len(episodes)} deaths, {n_survivors} survivors")
    print()

    # Determine what to generate
    annotated = args.annotated or args.all
    distributions = args.distributions or args.all

    # Generate figures
    paths = generate_all_figures(
        data, output_dir, annotated, distributions,
        aggregates=aggregates, survivors=survivors
    )

    print(f"\nGenerated {len(paths)} figures in {output_dir}/")


if __name__ == '__main__':
    main()
