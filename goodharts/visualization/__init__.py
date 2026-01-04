"""
Visualization for Goodhart's Law simulation.

Two modes:
- BrainView: Single-agent neural network introspection (matplotlib)
- ParallelStats: Multi-environment aggregate statistics (Plotly/Dash)
"""

from goodharts.visualization.brain_view import (
    MatplotlibBrainView, create_brain_view,
    # Legacy aliases
    BrainViewApp, create_brain_view_app,
)
from goodharts.visualization.parallel_stats import ParallelStatsApp, create_parallel_stats_app

__all__ = [
    # Brain view (matplotlib)
    'MatplotlibBrainView', 'create_brain_view',
    'BrainViewApp', 'create_brain_view_app',  # Legacy aliases
    # Parallel stats (Dash)
    'ParallelStatsApp', 'create_parallel_stats_app',
]
