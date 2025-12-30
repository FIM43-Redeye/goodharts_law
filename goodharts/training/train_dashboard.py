"""
Training Dashboard - Plotly-based visualization for PPO training.

Provides live visualization of training metrics with support for both:
- Jupyter/Colab: Interactive FigureWidget with in-place updates
- Standalone: Dash-based web server with live updates in browser

Panels:
- Reward History (smoothed)
- Losses (Policy, Value)
- Entropy
- Explained Variance
- Behavior (Food/Poison consumption)
- Action Distribution
"""
import numpy as np
import queue
import threading
import time
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict
import re

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from goodharts.behaviors.action_space import get_action_labels, DISCRETE_8


# =============================================================================
# CONSTANTS
# =============================================================================

DASHBOARD_MAX_POINTS = 1024       # Maximum data points to display in graphs
EMA_SMOOTHING_ALPHA = 0.9         # Exponential moving average smoothing factor
UPDATE_INTERVAL_MS = 100          # Milliseconds between dashboard updates
QUEUE_POLL_TIMEOUT = 0.05         # Seconds to wait when polling update queue
QUEUE_MAX_SIZE = 1000             # Maximum pending updates in queue

# Color scheme (dark theme)
COLORS = {
    'background': '#1a1a2e',
    'paper': '#16213e',
    'text': '#e0e0e0',
    'grid': '#2a2a4a',
    'reward': '#00ff88',
    'policy_loss': '#ff6b6b',
    'value_loss': '#4ecdc4',
    'entropy': '#ffd93d',
    'explained_var': '#c44dff',
    'food': '#16c79a',
    'poison': '#ff6b6b',
    'food_ratio': '#00d9ff',
    'action_bar': '#00d9ff',
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RunState:
    """State for a single training run.

    Metrics are stored both raw and smoothed. Smoothed values are computed
    incrementally as new data arrives (O(1) per update) rather than
    recomputing the full EMA each frame (O(n) per frame).
    """
    mode: str
    n_actions: int = 8

    # Current snapshot
    episode: int = 0
    total_steps: int = 0
    current_action_probs: np.ndarray = field(default_factory=lambda: np.zeros(8))

    # Raw metrics history (aligned by Update)
    rewards: list = field(default_factory=list)
    policy_losses: list = field(default_factory=list)
    value_losses: list = field(default_factory=list)
    entropies: list = field(default_factory=list)
    explained_variances: list = field(default_factory=list)
    food_history: list = field(default_factory=list)
    poison_history: list = field(default_factory=list)
    food_ratio_history: list = field(default_factory=list)

    # Pre-smoothed metrics (computed incrementally on append)
    rewards_smoothed: list = field(default_factory=list)
    policy_losses_smoothed: list = field(default_factory=list)
    value_losses_smoothed: list = field(default_factory=list)
    explained_variances_smoothed: list = field(default_factory=list)
    food_smoothed: list = field(default_factory=list)
    poison_smoothed: list = field(default_factory=list)
    food_ratio_smoothed: list = field(default_factory=list)

    # Status
    is_running: bool = True
    is_finished: bool = False

    # Timestamps
    last_update: float = 0.0

    def append_metrics(self, p_loss: float, v_loss: float, ent: float,
                       ev: float, reward: float, food: float, poison: float,
                       alpha: float = EMA_SMOOTHING_ALPHA):
        """Append new metrics with incremental EMA smoothing."""
        # Raw values
        self.policy_losses.append(p_loss)
        self.value_losses.append(v_loss)
        self.entropies.append(ent)
        self.explained_variances.append(ev)
        self.rewards.append(reward)
        self.food_history.append(food)
        self.poison_history.append(poison)

        # Compute food_ratio
        total = food + poison
        food_ratio = food / total if total > 0 else 0.5
        self.food_ratio_history.append(food_ratio)

        # Incremental EMA
        def ema_append(smoothed_list: list, raw_value: float) -> None:
            if smoothed_list:
                prev = smoothed_list[-1]
                smoothed_list.append(alpha * prev + (1 - alpha) * raw_value)
            else:
                smoothed_list.append(raw_value)

        ema_append(self.rewards_smoothed, reward)
        ema_append(self.policy_losses_smoothed, p_loss)
        ema_append(self.value_losses_smoothed, v_loss)
        ema_append(self.explained_variances_smoothed, ev)
        ema_append(self.food_smoothed, food)
        ema_append(self.poison_smoothed, poison)
        ema_append(self.food_ratio_smoothed, food_ratio)

        self.last_update = time.time()

        # Trim if over max
        if len(self.rewards) > DASHBOARD_MAX_POINTS * 1.5:
            trim_from = len(self.rewards) - DASHBOARD_MAX_POINTS
            self.rewards = self.rewards[trim_from:]
            self.policy_losses = self.policy_losses[trim_from:]
            self.value_losses = self.value_losses[trim_from:]
            self.entropies = self.entropies[trim_from:]
            self.explained_variances = self.explained_variances[trim_from:]
            self.food_history = self.food_history[trim_from:]
            self.poison_history = self.poison_history[trim_from:]
            self.food_ratio_history = self.food_ratio_history[trim_from:]
            self.rewards_smoothed = self.rewards_smoothed[trim_from:]
            self.policy_losses_smoothed = self.policy_losses_smoothed[trim_from:]
            self.value_losses_smoothed = self.value_losses_smoothed[trim_from:]
            self.explained_variances_smoothed = self.explained_variances_smoothed[trim_from:]
            self.food_smoothed = self.food_smoothed[trim_from:]
            self.poison_smoothed = self.poison_smoothed[trim_from:]
            self.food_ratio_smoothed = self.food_ratio_smoothed[trim_from:]


# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

def is_notebook() -> bool:
    """Detect if running in a Jupyter/Colab notebook environment."""
    try:
        from IPython import get_ipython
        ipy = get_ipython()
        if ipy is None:
            return False
        shell = ipy.__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook/lab
        elif shell == 'Google.Colab':
            return True  # Google Colab
        elif 'google.colab' in str(ipy):
            return True  # Another way Colab might appear
        return False
    except (ImportError, NameError, AttributeError):
        return False


def is_colab() -> bool:
    """Detect if running in Google Colab specifically."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


# =============================================================================
# PLOTLY DASHBOARD - Core implementation
# =============================================================================

class TrainingDashboard:
    """
    Plotly-based training dashboard.

    Automatically detects environment and uses appropriate display mode:
    - Jupyter/Colab: FigureWidget with in-place updates
    - Standalone: Dash web server with live updates
    """

    def __init__(self, modes: list[str], n_actions: int = 8, training_stop_event=None):
        """
        Initialize the dashboard.

        Args:
            modes: Training modes to display (e.g., ['ground_truth', 'proxy'])
            n_actions: Number of discrete actions for action distribution
            training_stop_event: Event to signal training should stop
        """
        self.modes = modes
        self.n_actions = n_actions
        self.action_labels = get_action_labels(DISCRETE_8)
        self._training_stop_event = training_stop_event

        # Data storage
        self.runs: dict[str, RunState] = {
            mode: RunState(mode=mode, n_actions=n_actions)
            for mode in modes
        }

        # Thread-safe update queue
        self.update_queue: queue.Queue = queue.Queue()

        # UI state
        self.fig: Optional[go.FigureWidget] = None
        self.dirty = False
        self.running = True
        self._update_thread: Optional[threading.Thread] = None
        self._stop_requested = False

        # Trace indices for each mode
        self._trace_indices: dict[str, dict[str, int]] = {}

    def _create_figure(self) -> go.Figure:
        """Create the Plotly figure with subplots."""
        # 2x3 grid layout
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Reward', 'Losses', 'Entropy',
                'Explained Variance', 'Behavior', 'Actions'
            ],
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'bar'}],
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.15,
        )

        trace_idx = 0

        for mode in self.modes:
            self._trace_indices[mode] = {}
            suffix = f' ({mode})' if len(self.modes) > 1 else ''

            # Row 1: Reward, Losses, Entropy
            # Reward
            fig.add_trace(
                go.Scatter(x=[], y=[], mode='lines', name=f'Reward{suffix}',
                          line=dict(color=COLORS['reward'], width=2)),
                row=1, col=1
            )
            self._trace_indices[mode]['reward'] = trace_idx
            trace_idx += 1

            # Policy loss
            fig.add_trace(
                go.Scatter(x=[], y=[], mode='lines', name=f'Policy{suffix}',
                          line=dict(color=COLORS['policy_loss'], width=2)),
                row=1, col=2
            )
            self._trace_indices[mode]['policy_loss'] = trace_idx
            trace_idx += 1

            # Value loss
            fig.add_trace(
                go.Scatter(x=[], y=[], mode='lines', name=f'Value{suffix}',
                          line=dict(color=COLORS['value_loss'], width=2)),
                row=1, col=2
            )
            self._trace_indices[mode]['value_loss'] = trace_idx
            trace_idx += 1

            # Entropy
            fig.add_trace(
                go.Scatter(x=[], y=[], mode='lines', name=f'Entropy{suffix}',
                          line=dict(color=COLORS['entropy'], width=2)),
                row=1, col=3
            )
            self._trace_indices[mode]['entropy'] = trace_idx
            trace_idx += 1

            # Row 2: EV, Behavior, Actions
            # Explained variance
            fig.add_trace(
                go.Scatter(x=[], y=[], mode='lines', name=f'Exp.Var{suffix}',
                          line=dict(color=COLORS['explained_var'], width=2)),
                row=2, col=1
            )
            self._trace_indices[mode]['explained_var'] = trace_idx
            trace_idx += 1

            # Food consumption
            fig.add_trace(
                go.Scatter(x=[], y=[], mode='lines', name=f'Food{suffix}',
                          line=dict(color=COLORS['food'], width=2)),
                row=2, col=2
            )
            self._trace_indices[mode]['food'] = trace_idx
            trace_idx += 1

            # Poison consumption
            fig.add_trace(
                go.Scatter(x=[], y=[], mode='lines', name=f'Poison{suffix}',
                          line=dict(color=COLORS['poison'], width=2)),
                row=2, col=2
            )
            self._trace_indices[mode]['poison'] = trace_idx
            trace_idx += 1

            # Action distribution (bar chart)
            fig.add_trace(
                go.Bar(x=self.action_labels, y=[0] * self.n_actions,
                      name=f'Actions{suffix}', marker_color=COLORS['action_bar']),
                row=2, col=3
            )
            self._trace_indices[mode]['actions'] = trace_idx
            trace_idx += 1

        # Style the figure
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['paper'],
            plot_bgcolor=COLORS['background'],
            font=dict(color=COLORS['text'], size=11),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                font=dict(size=9),
            ),
            margin=dict(l=50, r=30, t=80, b=50),
            height=550,
        )

        # Style axes
        for i in range(1, 3):
            for j in range(1, 4):
                fig.update_xaxes(
                    gridcolor=COLORS['grid'],
                    zerolinecolor=COLORS['grid'],
                    title_text='Update' if i == 2 else None,
                    row=i, col=j
                )
                fig.update_yaxes(
                    gridcolor=COLORS['grid'],
                    zerolinecolor=COLORS['grid'],
                    row=i, col=j
                )

        # Special formatting for action distribution
        fig.update_xaxes(tickangle=45, row=2, col=3)
        fig.update_yaxes(range=[0, 0.3], row=2, col=3)

        return fig

    def update(self, mode: str, update_type: str, data: Any):
        """
        Queue an update from a training thread.

        Args:
            mode: Training mode (e.g., 'ground_truth')
            update_type: 'update' or 'finished'
            data: Payload dict with metrics
        """
        try:
            self.update_queue.put_nowait((mode, update_type, data))
        except queue.Full:
            pass  # Drop if queue full

    def _process_updates(self):
        """Process all pending updates from the queue."""
        updates_processed = 0

        while True:
            try:
                mode, update_type, data = self.update_queue.get_nowait()
            except queue.Empty:
                break

            if mode not in self.runs:
                self.runs[mode] = RunState(mode=mode, n_actions=self.n_actions)

            run = self.runs[mode]

            if update_type == 'update':
                run.append_metrics(
                    p_loss=data.get('policy_loss', 0),
                    v_loss=data.get('value_loss', 0),
                    ent=data.get('entropy', 0),
                    ev=data.get('explained_var', 0),
                    reward=data.get('reward', 0),
                    food=data.get('food', 0),
                    poison=data.get('poison', 0),
                )
                run.total_steps = data.get('total_steps', run.total_steps)

                if 'action_probs' in data:
                    run.current_action_probs = np.array(data['action_probs'])

                updates_processed += 1

            elif update_type == 'finished':
                run.is_running = False
                run.is_finished = True

        if updates_processed > 0:
            self.dirty = True

    def _refresh_traces(self):
        """Update all traces with current data."""
        if self.fig is None:
            return

        with self.fig.batch_update():
            for mode, run in self.runs.items():
                if mode not in self._trace_indices:
                    continue

                indices = self._trace_indices[mode]
                n = len(run.rewards_smoothed)

                if n < 2:
                    continue

                start = max(0, n - DASHBOARD_MAX_POINTS)
                x = list(range(start, n))

                # Update traces
                self.fig.data[indices['reward']].x = x
                self.fig.data[indices['reward']].y = run.rewards_smoothed[start:]

                self.fig.data[indices['policy_loss']].x = x
                self.fig.data[indices['policy_loss']].y = run.policy_losses_smoothed[start:]

                self.fig.data[indices['value_loss']].x = x
                self.fig.data[indices['value_loss']].y = run.value_losses_smoothed[start:]

                self.fig.data[indices['entropy']].x = x
                self.fig.data[indices['entropy']].y = run.entropies[start:]

                if run.explained_variances_smoothed:
                    ev_n = len(run.explained_variances_smoothed)
                    ev_start = max(0, ev_n - DASHBOARD_MAX_POINTS)
                    ev_x = list(range(ev_start, ev_n))
                    self.fig.data[indices['explained_var']].x = ev_x
                    self.fig.data[indices['explained_var']].y = run.explained_variances_smoothed[ev_start:]

                if run.food_smoothed:
                    beh_n = len(run.food_smoothed)
                    beh_start = max(0, beh_n - DASHBOARD_MAX_POINTS)
                    beh_x = list(range(beh_start, beh_n))
                    self.fig.data[indices['food']].x = beh_x
                    self.fig.data[indices['food']].y = run.food_smoothed[beh_start:]
                    self.fig.data[indices['poison']].x = beh_x
                    self.fig.data[indices['poison']].y = run.poison_smoothed[beh_start:]

                if run.current_action_probs is not None:
                    self.fig.data[indices['actions']].y = run.current_action_probs.tolist()

    def _update_loop(self):
        """Background thread: poll queue and update figure."""
        while self.running and not self._stop_requested:
            try:
                self._process_updates()
                if self.dirty:
                    self._refresh_traces()
                    self.dirty = False
            except Exception as e:
                print(f"[Dashboard] Update error: {e}")

            time.sleep(UPDATE_INTERVAL_MS / 1000)

    def run(self):
        """Start the dashboard (blocking for standalone, non-blocking for notebook)."""
        fig = self._create_figure()

        if is_notebook():
            # Notebook mode: use FigureWidget
            self.fig = go.FigureWidget(fig)
            from IPython.display import display
            display(self.fig)

            # Start background update thread
            self._update_thread = threading.Thread(
                target=self._update_loop,
                daemon=True,
                name="DashboardUpdater"
            )
            self._update_thread.start()
            print("[Dashboard] Running in notebook mode - updates will appear above")
        else:
            # Standalone mode: use Dash
            self._run_dash_server(fig)

    def _run_dash_server(self, fig: go.Figure):
        """Run a Dash server for standalone visualization."""
        try:
            from dash import Dash, dcc, html
            from dash.dependencies import Input, Output
        except ImportError:
            print("[Dashboard] Dash not installed. Install with: pip install dash")
            print("[Dashboard] Falling back to static display...")
            fig.show()
            return

        app = Dash(__name__)

        # Dark theme for entire page
        app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Training Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body { background-color: ''' + COLORS['background'] + '''; margin: 0; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

        app.layout = html.Div([
            html.H2("Training Dashboard", style={'textAlign': 'center', 'color': COLORS['text']}),
            html.Div([
                html.Button('Stop Training', id='stop-btn', n_clicks=0,
                           style={'backgroundColor': COLORS['policy_loss'], 'color': 'white',
                                 'padding': '10px 20px', 'border': 'none', 'cursor': 'pointer'}),
                html.Span(id='stop-status', style={'marginLeft': '10px', 'color': COLORS['text']}),
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            dcc.Graph(id='live-graph', figure=fig),
            dcc.Interval(id='interval', interval=UPDATE_INTERVAL_MS, n_intervals=0),
        ], style={
            'backgroundColor': COLORS['background'],
            'padding': '20px',
            'minHeight': '100vh',
            'margin': '0',
        })

        @app.callback(
            Output('live-graph', 'figure'),
            Input('interval', 'n_intervals')
        )
        def update_graph(n):
            self._process_updates()
            fig_updated = self._create_figure()

            # Populate with current data
            for mode, run in self.runs.items():
                if mode not in self._trace_indices:
                    continue

                indices = self._trace_indices[mode]
                n_pts = len(run.rewards_smoothed)

                if n_pts < 2:
                    continue

                start = max(0, n_pts - DASHBOARD_MAX_POINTS)
                x = list(range(start, n_pts))

                fig_updated.data[indices['reward']].x = x
                fig_updated.data[indices['reward']].y = run.rewards_smoothed[start:]
                fig_updated.data[indices['policy_loss']].x = x
                fig_updated.data[indices['policy_loss']].y = run.policy_losses_smoothed[start:]
                fig_updated.data[indices['value_loss']].x = x
                fig_updated.data[indices['value_loss']].y = run.value_losses_smoothed[start:]
                fig_updated.data[indices['entropy']].x = x
                fig_updated.data[indices['entropy']].y = run.entropies[start:]

                if run.explained_variances_smoothed:
                    ev_n = len(run.explained_variances_smoothed)
                    ev_start = max(0, ev_n - DASHBOARD_MAX_POINTS)
                    fig_updated.data[indices['explained_var']].x = list(range(ev_start, ev_n))
                    fig_updated.data[indices['explained_var']].y = run.explained_variances_smoothed[ev_start:]

                if run.food_smoothed:
                    beh_n = len(run.food_smoothed)
                    beh_start = max(0, beh_n - DASHBOARD_MAX_POINTS)
                    fig_updated.data[indices['food']].x = list(range(beh_start, beh_n))
                    fig_updated.data[indices['food']].y = run.food_smoothed[beh_start:]
                    fig_updated.data[indices['poison']].x = list(range(beh_start, beh_n))
                    fig_updated.data[indices['poison']].y = run.poison_smoothed[beh_start:]

                if run.current_action_probs is not None:
                    fig_updated.data[indices['actions']].y = run.current_action_probs.tolist()

            return fig_updated

        @app.callback(
            Output('stop-status', 'children'),
            Input('stop-btn', 'n_clicks')
        )
        def stop_training(n_clicks):
            if n_clicks > 0:
                self._stop_requested = True
                if self._training_stop_event is not None:
                    self._training_stop_event.set()
                return "Stop signal sent..."
            return ""

        print("[Dashboard] Starting Dash server at http://127.0.0.1:8050")
        print("[Dashboard] Press Ctrl+C to stop")

        # Run server (blocking)
        app.run(debug=False, use_reloader=False)

    def stop(self):
        """Stop the dashboard."""
        self.running = False
        self._stop_requested = True
        if self._update_thread is not None:
            self._update_thread.join(timeout=1.0)

    def should_stop(self) -> bool:
        """Check if stop was requested."""
        return self._stop_requested


# =============================================================================
# DASHBOARD PROCESS - Compatible interface for training integration
# =============================================================================

class DashboardProcess:
    """
    Dashboard wrapper that provides process-like interface.

    For Plotly, we don't need true process isolation since the browser
    handles rendering separately. This class provides the same interface
    as the old matplotlib DashboardProcess for compatibility.
    """

    def __init__(self, modes: list[str], n_actions: int = 8):
        self.modes = modes
        self.n_actions = n_actions
        self._dashboard: Optional[TrainingDashboard] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._training_stop_event = threading.Event()

    def start(self):
        """Start the dashboard in a background thread."""
        self._dashboard = TrainingDashboard(
            self.modes,
            self.n_actions,
            training_stop_event=self._training_stop_event
        )

        def run_dashboard():
            try:
                self._dashboard.run()
            except Exception as e:
                print(f"[Dashboard] Error: {e}")
            finally:
                self._stop_event.set()

        self._thread = threading.Thread(target=run_dashboard, daemon=True, name="Dashboard")
        self._thread.start()

        # Give it a moment to start
        time.sleep(0.5)
        print(f"[Dashboard] Started")

    def send_update(self, mode: str, update_type: str, data: dict):
        """Send an update to the dashboard."""
        if self._dashboard is not None:
            self._dashboard.update(mode, update_type, data)

    def update(self, mode: str, update_type: str, data: dict):
        """Alias for send_update() - compatible with TrainingDashboard interface."""
        self.send_update(mode, update_type, data)

    def is_alive(self) -> bool:
        """Check if dashboard is still running."""
        return self._thread is not None and self._thread.is_alive()

    def should_stop(self) -> bool:
        """Check if user requested training stop via dashboard button."""
        return self._training_stop_event.is_set()

    def stop(self, timeout: float = 2.0):
        """Stop the dashboard."""
        if self._dashboard is not None:
            self._dashboard.stop()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)


def create_dashboard(modes: list[str], n_actions: int = 8) -> TrainingDashboard:
    """Factory function to create a TrainingDashboard."""
    return TrainingDashboard(modes, n_actions)


def create_dashboard_process(modes: list[str], n_actions: int = 8) -> DashboardProcess:
    """Factory function to create a process-like dashboard wrapper."""
    return DashboardProcess(modes, n_actions)


# =============================================================================
# LOG LOADING - View historical training runs
# =============================================================================

def load_run_from_logs(log_prefix: str) -> RunState:
    """
    Load a training run from CSV log files.

    Args:
        log_prefix: Path prefix for log files (without _updates.csv suffix)
                   Example: 'logs/ground_truth_20251214_053108'

    Returns:
        RunState populated with historical data
    """
    import csv

    updates_path = f"{log_prefix}_updates.csv"

    # Extract mode from prefix
    basename = os.path.basename(log_prefix)
    match = re.match(r'(.+?)_\d{8}_\d{6}', basename)
    mode = match.group(1) if match else basename

    run = RunState(mode=mode)
    run.is_running = False
    run.is_finished = True

    if os.path.exists(updates_path):
        with open(updates_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                run.policy_losses.append(float(row['policy_loss']))
                run.value_losses.append(float(row['value_loss']))
                run.entropies.append(float(row['entropy']))
                run.total_steps = int(row['total_steps'])

                if 'reward_mean' in row and row['reward_mean']:
                    run.rewards.append(float(row['reward_mean']))

                if 'explained_variance' in row and row['explained_variance']:
                    run.explained_variances.append(float(row['explained_variance']))

                if 'food_mean' in row and row['food_mean']:
                    food = float(row['food_mean'])
                    poison = float(row['poison_mean']) if row.get('poison_mean') else 0.0
                    run.food_history.append(food)
                    run.poison_history.append(poison)
                    if 'food_ratio' in row and row['food_ratio']:
                        run.food_ratio_history.append(float(row['food_ratio']))
                    else:
                        total = food + poison
                        run.food_ratio_history.append(food / total if total > 0 else 0.5)

        # Populate smoothed lists
        run.rewards_smoothed = list(run.rewards)
        run.policy_losses_smoothed = list(run.policy_losses)
        run.value_losses_smoothed = list(run.value_losses)
        run.explained_variances_smoothed = list(run.explained_variances)
        run.food_smoothed = list(run.food_history)
        run.poison_smoothed = list(run.poison_history)
        run.food_ratio_smoothed = list(run.food_ratio_history)

        print(f"   Loaded {len(run.policy_losses)} updates from {updates_path}")
    else:
        print(f"   Warning: Updates file not found: {updates_path}")

    return run


def find_latest_logs(log_dir: str = 'generated/logs', mode: str = None) -> list[str]:
    """
    Find the latest log prefixes in a directory.

    Args:
        log_dir: Directory containing log files
        mode: Optional mode filter (e.g., 'ground_truth')

    Returns:
        List of log prefixes (paths without _updates.csv suffix)
    """
    if not os.path.isdir(log_dir):
        print(f"Log directory not found: {log_dir}")
        return []

    pattern = re.compile(r'(.+?)_(\d{8}_\d{6})_updates\.csv')
    mode_timestamps = defaultdict(list)

    for filename in os.listdir(log_dir):
        match = pattern.match(filename)
        if match:
            m, ts = match.groups()
            if mode is None or m == mode:
                mode_timestamps[m].append(ts)

    prefixes = []
    for m, timestamps in mode_timestamps.items():
        latest_ts = sorted(timestamps)[-1]
        prefixes.append(os.path.join(log_dir, f"{m}_{latest_ts}"))

    return sorted(prefixes)


def view_logs(log_prefixes: list[str] = None, log_dir: str = 'generated/logs', mode: str = None):
    """
    Display a visualization of historical training runs.

    Args:
        log_prefixes: List of log file prefixes to visualize (optional)
        log_dir: Directory to search for logs if prefixes not provided
        mode: Mode filter when searching for logs
    """
    if log_prefixes is None:
        log_prefixes = find_latest_logs(log_dir, mode)

    if not log_prefixes:
        print("No log files to display.")
        return

    print(f"\nLoading {len(log_prefixes)} training run(s)...")
    runs = [load_run_from_logs(prefix) for prefix in log_prefixes]

    # Filter empty runs
    runs = [r for r in runs if r.rewards or r.policy_losses]
    if not runs:
        print("No valid data found in log files.")
        return

    # Create dashboard and populate with loaded data
    modes = [r.mode for r in runs]
    dashboard = TrainingDashboard(modes)

    for run in runs:
        dashboard.runs[run.mode] = run

    # Create and display figure
    fig = dashboard._create_figure()
    dashboard.fig = go.FigureWidget(fig) if is_notebook() else fig
    dashboard._refresh_traces()

    if is_notebook():
        from IPython.display import display
        display(dashboard.fig)
    else:
        dashboard.fig.show()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='View training logs')
    parser.add_argument('--log-dir', default='generated/logs', help='Log directory')
    parser.add_argument('--mode', default=None, help='Filter by mode')
    parser.add_argument('logs', nargs='*', help='Specific log prefixes to view')

    args = parser.parse_args()

    if args.logs:
        view_logs(args.logs)
    else:
        view_logs(log_dir=args.log_dir, mode=args.mode)
