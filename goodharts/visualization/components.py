"""
Shared visualization components.

Provides consistent theming and colors across ALL visualizations.
This is the SINGLE SOURCE OF TRUTH for colors and theme.
All other visualization modules should import from here.

Supports automatic system theme detection (light/dark mode) on:
- macOS 10.14+
- Windows 10 1607+
- Linux with GTK-based desktops (GNOME, etc.)
"""
import plotly.graph_objects as go
from typing import Literal

# Try to import darkdetect for system theme detection
try:
    import darkdetect
    _DARKDETECT_AVAILABLE = True
except ImportError:
    _DARKDETECT_AVAILABLE = False


# Theme definitions
DARK_THEME = {
    'name': 'dark',
    'background': '#1a1a2e',
    'paper': '#16213e',
    'text': '#e0e0e0',
    'grid': 'rgba(128, 128, 128, 0.15)',
    'text_secondary': '#a0a0a0',
}

LIGHT_THEME = {
    'name': 'light',
    'background': '#f5f5f5',
    'paper': '#ffffff',
    'text': '#1a1a1a',
    'grid': 'rgba(128, 128, 128, 0.2)',
    'text_secondary': '#666666',
}

# Mode color schemes - designed for good contrast on both themes
# Design: green (good) -> yellow (caution) -> orange (warning) -> red (danger)
MODE_COLORS_DARK = {
    'ground_truth': '#22c79a',           # Green - thriving, aligned
    'ground_truth_handhold': '#4ecdc4',  # Teal - experimental/guided
    'ground_truth_blinded': '#ffc107',   # Yellow/amber - blinded but true reward
    'proxy_mortal': '#ff9f43',           # Orange - proxy but can die
    'proxy': '#ff6b6b',                  # Red - completely unmoored Goodhart case
}

# Slightly darker/more saturated colors for light backgrounds
MODE_COLORS_LIGHT = {
    'ground_truth': '#1a9e7a',           # Darker green
    'ground_truth_handhold': '#3ab4a8',  # Darker teal
    'ground_truth_blinded': '#d4a000',   # Darker amber
    'proxy_mortal': '#e07b20',           # Darker orange
    'proxy': '#dc3545',                  # Darker red
}

DEFAULT_COLOR_DARK = '#00aaff'   # Cyan fallback for dark theme
DEFAULT_COLOR_LIGHT = '#0077cc'  # Darker cyan for light theme


def detect_system_theme() -> Literal['dark', 'light']:
    """
    Detect the system's current theme preference.

    Uses darkdetect library for cross-platform detection:
    - macOS: Reads system appearance settings
    - Windows: Reads registry theme preference
    - Linux: Reads GTK theme settings (GNOME, etc.)

    Returns:
        'dark' or 'light'. Defaults to 'dark' if detection fails or
        is unavailable (dark themes are more common for data visualization).
    """
    if not _DARKDETECT_AVAILABLE:
        return 'dark'

    try:
        theme = darkdetect.theme()
        if theme is None:
            # Detection not supported on this platform
            return 'dark'
        return 'dark' if theme.lower() == 'dark' else 'light'
    except Exception:
        # Any error in detection - fall back to dark
        return 'dark'


def is_dark_mode() -> bool:
    """Check if system is in dark mode."""
    return detect_system_theme() == 'dark'


def get_theme(force: Literal['dark', 'light', 'auto'] = 'auto') -> dict:
    """
    Get the appropriate theme dict.

    Args:
        force: 'dark', 'light', or 'auto' (detect from system)

    Returns:
        Theme dict with background, paper, text, grid colors
    """
    if force == 'dark':
        return DARK_THEME
    elif force == 'light':
        return LIGHT_THEME
    else:
        return DARK_THEME if is_dark_mode() else LIGHT_THEME


def get_mode_colors(force: Literal['dark', 'light', 'auto'] = 'auto') -> dict:
    """
    Get mode colors appropriate for the current/specified theme.

    Args:
        force: 'dark', 'light', or 'auto' (detect from system)

    Returns:
        Dict mapping mode names to hex color strings
    """
    if force == 'dark':
        return MODE_COLORS_DARK
    elif force == 'light':
        return MODE_COLORS_LIGHT
    else:
        return MODE_COLORS_DARK if is_dark_mode() else MODE_COLORS_LIGHT


def get_default_color(force: Literal['dark', 'light', 'auto'] = 'auto') -> str:
    """Get default/fallback color for unknown modes."""
    if force == 'dark':
        return DEFAULT_COLOR_DARK
    elif force == 'light':
        return DEFAULT_COLOR_LIGHT
    else:
        return DEFAULT_COLOR_DARK if is_dark_mode() else DEFAULT_COLOR_LIGHT


# Convenience exports that use auto-detection
# These are the primary interface for other modules
THEME = get_theme('auto')
MODE_COLORS = get_mode_colors('auto')
DEFAULT_COLOR = get_default_color('auto')


def get_mode_color(mode: str, force: Literal['dark', 'light', 'auto'] = 'auto') -> str:
    """
    Get consistent color for a training mode.

    Args:
        mode: Mode name (ground_truth, proxy, etc.)
        force: Theme override ('dark', 'light', or 'auto')

    Returns:
        Hex color string
    """
    colors = get_mode_colors(force)
    default = get_default_color(force)
    return colors.get(mode, default)


def get_theme_index_string(
    title: str = "Goodhart's Law Visualization",
    force: Literal['dark', 'light', 'auto'] = 'auto'
) -> str:
    """
    Return custom HTML template for themed Dash app.

    This prevents the white/black flash on load and ensures consistent
    background throughout the app lifecycle.

    Args:
        title: Page title
        force: Theme override ('dark', 'light', or 'auto')
    """
    theme = get_theme(force)
    bg = theme['background']
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


# Legacy alias for backwards compatibility
def get_dark_theme_index_string(title: str = "Goodhart's Law Visualization") -> str:
    """Legacy function - use get_theme_index_string() instead."""
    return get_theme_index_string(title, force='dark')


def apply_theme(
    fig: go.Figure,
    force: Literal['dark', 'light', 'auto'] = 'auto'
) -> go.Figure:
    """
    Apply theme to a Plotly figure.

    Modifies the figure in-place and returns it for chaining.

    Args:
        fig: Plotly figure to theme
        force: Theme override ('dark', 'light', or 'auto')

    Returns:
        The same figure (for chaining)
    """
    theme = get_theme(force)
    fig.update_layout(
        plot_bgcolor=theme['paper'],
        paper_bgcolor=theme['background'],
        font=dict(color=theme['text']),
    )
    fig.update_xaxes(gridcolor=theme['grid'])
    fig.update_yaxes(gridcolor=theme['grid'])
    return fig


# Legacy alias for backwards compatibility
def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Legacy function - use apply_theme() instead."""
    return apply_theme(fig, force='dark')
