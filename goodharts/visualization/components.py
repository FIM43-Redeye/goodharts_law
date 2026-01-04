"""
Shared visualization components.

Provides consistent theming and colors across visualizations.
Plotly/Dash-specific functions are used by parallel_stats dashboard.
"""
import plotly.graph_objects as go


# Dark theme constants (matching eval_dashboard.py)
THEME = {
    'background': '#1a1a2e',
    'paper': '#16213e',
    'text': '#e0e0e0',
    'grid': 'rgba(128, 128, 128, 0.15)',
}

# Mode color scheme - consistent across all visualizations
MODE_COLORS = {
    'ground_truth': '#16c79a',           # Teal green
    'ground_truth_handhold': '#9ee493',  # Light green
    'proxy': '#ff6b6b',                  # Coral red
    'ground_truth_blinded': '#ffa500',   # Orange
}
DEFAULT_COLOR = '#00aaff'  # Cyan fallback


def get_mode_color(mode: str) -> str:
    """Get consistent color for a training mode."""
    return MODE_COLORS.get(mode, DEFAULT_COLOR)


def get_dark_theme_index_string(title: str = "Goodhart's Law Visualization") -> str:
    """
    Return custom HTML template for dark theme Dash app.

    This prevents the white flash on load and ensures consistent
    dark background throughout the app lifecycle.
    """
    bg = THEME['background']
    return f'''
<!DOCTYPE html>
<html style="background: {bg} !important;">
    <head>
        {{%metas%}}
        <title>{title}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            html, body, #react-entry-point, ._dash-loading {{
                background-color: {bg} !important;
                background: {bg} !important;
                margin: 0 !important;
                padding: 0 !important;
                min-height: 100vh !important;
            }}
            body > div, #react-entry-point > div {{
                background-color: {bg} !important;
            }}
        </style>
    </head>
    <body style="background: {bg} !important;">
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """
    Apply dark theme to a Plotly figure.

    Modifies the figure in-place and returns it for chaining.
    """
    fig.update_layout(
        plot_bgcolor=THEME['paper'],
        paper_bgcolor=THEME['background'],
        font=dict(color=THEME['text']),
    )
    fig.update_xaxes(gridcolor=THEME['grid'])
    fig.update_yaxes(gridcolor=THEME['grid'])
    return fig


