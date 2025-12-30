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
import json
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

    def _create_figure_for_mode(self, mode: str) -> go.Figure:
        """Create a Plotly figure for a single mode."""
        # Dual y-axes for Losses (row1,col2) and Behavior (row2,col2)
        # Note: secondary_y requires type='xy' (the default), not 'scatter'
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Reward', 'Losses', 'Entropy',
                'Explained Variance', 'Behavior', 'Actions'
            ],
            specs=[
                [{}, {'secondary_y': True}, {}],
                [{}, {'secondary_y': True}, {'type': 'bar'}],
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.15,
        )

        # Reward (trace 0)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines', name='Reward',
                      line=dict(color=COLORS['reward'], width=2)),
            row=1, col=1
        )

        # Policy loss - left y-axis (trace 1)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines', name='Policy Loss',
                      line=dict(color=COLORS['policy_loss'], width=2)),
            row=1, col=2, secondary_y=False
        )

        # Value loss - right y-axis (trace 2)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines', name='Value Loss',
                      line=dict(color=COLORS['value_loss'], width=2)),
            row=1, col=2, secondary_y=True
        )

        # Entropy (trace 3)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines', name='Entropy',
                      line=dict(color=COLORS['entropy'], width=2)),
            row=1, col=3
        )

        # Explained variance (trace 4)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines', name='Explained Var',
                      line=dict(color=COLORS['explained_var'], width=2)),
            row=2, col=1
        )

        # Food consumption - left y-axis (trace 5)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines', name='Food',
                      line=dict(color=COLORS['food'], width=2)),
            row=2, col=2, secondary_y=False
        )

        # Poison consumption - left y-axis (trace 6)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines', name='Poison',
                      line=dict(color=COLORS['poison'], width=2)),
            row=2, col=2, secondary_y=False
        )

        # Food ratio - right y-axis, fixed 0-1 (trace 7)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines', name='Food Ratio',
                      line=dict(color=COLORS['food_ratio'], width=2, dash='dash')),
            row=2, col=2, secondary_y=True
        )

        # Action distribution - hidden from legend (trace 8)
        fig.add_trace(
            go.Bar(x=self.action_labels, y=[0] * self.n_actions,
                  name='Actions', marker_color=COLORS['action_bar'],
                  showlegend=False),
            row=2, col=3
        )

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
            margin=dict(l=50, r=50, t=60, b=50),  # Extra right margin for secondary axes
            height=500,
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

        # Secondary y-axis styling
        fig.update_yaxes(
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid'],
            row=1, col=2, secondary_y=True
        )
        fig.update_yaxes(
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid'],
            range=[0, 1],  # Food ratio fixed 0-1
            row=2, col=2, secondary_y=True
        )

        # Fixed axis ranges
        fig.update_yaxes(rangemode='tozero', row=1, col=3)  # Entropy: min=0
        fig.update_xaxes(tickangle=45, row=2, col=3)
        fig.update_yaxes(range=[0, 0.3], row=2, col=3)  # Action probs

        return fig

    def _populate_figure_for_mode(self, fig: go.Figure, mode: str) -> go.Figure:
        """Populate a figure with data for the given mode."""
        if mode not in self.runs:
            return fig

        run = self.runs[mode]
        n_pts = len(run.rewards_smoothed)

        if n_pts < 2:
            return fig

        start = max(0, n_pts - DASHBOARD_MAX_POINTS)
        x = list(range(start, n_pts))

        # Trace indices (fixed order from _create_figure_for_mode)
        fig.data[0].x = x  # Reward
        fig.data[0].y = run.rewards_smoothed[start:]
        fig.data[1].x = x  # Policy loss
        fig.data[1].y = run.policy_losses_smoothed[start:]
        fig.data[2].x = x  # Value loss
        fig.data[2].y = run.value_losses_smoothed[start:]
        fig.data[3].x = x  # Entropy
        fig.data[3].y = run.entropies[start:]

        if run.explained_variances_smoothed:
            ev_n = len(run.explained_variances_smoothed)
            ev_start = max(0, ev_n - DASHBOARD_MAX_POINTS)
            ev_data = run.explained_variances_smoothed[ev_start:]
            fig.data[4].x = list(range(ev_start, ev_n))
            fig.data[4].y = ev_data

            # Sticky [0, 1] range that expands only if data exceeds bounds
            ev_min = min(ev_data) if ev_data else 0
            ev_max = max(ev_data) if ev_data else 1
            range_min = min(0, ev_min - 0.05)  # Expand below 0 if needed
            range_max = max(1, ev_max + 0.05)  # Expand above 1 if needed
            fig.update_yaxes(range=[range_min, range_max], row=2, col=1)

        if run.food_smoothed:
            beh_n = len(run.food_smoothed)
            beh_start = max(0, beh_n - DASHBOARD_MAX_POINTS)
            beh_x = list(range(beh_start, beh_n))
            fig.data[5].x = beh_x  # Food
            fig.data[5].y = run.food_smoothed[beh_start:]
            fig.data[6].x = beh_x  # Poison
            fig.data[6].y = run.poison_smoothed[beh_start:]
            if run.food_ratio_smoothed:
                fig.data[7].x = beh_x  # Food ratio
                fig.data[7].y = run.food_ratio_smoothed[beh_start:]

        if run.current_action_probs is not None:
            fig.data[8].y = run.current_action_probs.tolist()

        return fig

    def _create_figure(self) -> go.Figure:
        """Create figure for single-mode display (legacy compatibility)."""
        if len(self.modes) == 1:
            return self._create_figure_for_mode(self.modes[0])
        # For multi-mode, create empty figure (tabs handle individual modes)
        return self._create_figure_for_mode(self.modes[0])

    def update(self, mode: str, update_type: str, data: Any):
        """
        Queue an update from a training thread and push to clients.

        Args:
            mode: Training mode (e.g., 'ground_truth')
            update_type: 'update' or 'finished'
            data: Payload dict with metrics
        """
        try:
            self.update_queue.put_nowait((mode, update_type, data))
            # Push immediately to WebSocket clients (no polling needed)
            self._push_to_clients()
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
        """Run a Dash server with WebSocket push updates."""
        try:
            from dash import Dash, dcc, html, ALL
            from dash.dependencies import Input, Output
            from dash_extensions import WebSocket
            from flask_sock import Sock
        except ImportError as e:
            print(f"[Dashboard] Missing dependency: {e}")
            print("[Dashboard] Install with: pip install dash dash-extensions flask-sock")
            fig.show()
            return

        app = Dash(__name__, suppress_callback_exceptions=True)
        sock = Sock(app.server)

        # WebSocket clients (thread-safe set) - stored on self for push access
        self._ws_clients: set = set()
        self._ws_lock = threading.Lock()

        # Dark theme
        bg = COLORS['background']
        tab_style = {
            'backgroundColor': COLORS['paper'],
            'color': COLORS['text'],
            'padding': '10px 20px',
            'border': 'none',
            'borderBottom': f'2px solid {COLORS["grid"]}',
        }
        tab_selected_style = {
            **tab_style,
            'backgroundColor': COLORS['background'],
            'borderBottom': f'2px solid {COLORS["reward"]}',
        }

        app.index_string = f'''
<!DOCTYPE html>
<html style="background: {bg} !important;">
    <head>
        {{%metas%}}
        <title>Training Dashboard</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            html, body, #react-entry-point, ._dash-loading {{
                background-color: {bg} !important;
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

        # Build layout based on number of modes
        if len(self.modes) == 1:
            # Single mode: simple layout without tabs
            content = dcc.Graph(id='graph-single', figure=fig)
        else:
            # Multiple modes: tabbed interface
            tabs = []
            for mode in self.modes:
                mode_fig = self._create_figure_for_mode(mode)
                tabs.append(dcc.Tab(
                    label=mode.replace('_', ' ').title(),
                    value=mode,
                    children=[dcc.Graph(id=f'graph-{mode}', figure=mode_fig)],
                    style=tab_style,
                    selected_style=tab_selected_style,
                ))
            content = dcc.Tabs(
                id='mode-tabs',
                value=self.modes[0],
                children=tabs,
                style={'marginBottom': '10px'},
            )

        app.layout = html.Div([
            html.H2("Training Dashboard", style={'textAlign': 'center', 'color': COLORS['text']}),
            html.Div([
                html.Button('Stop Training', id='stop-btn', n_clicks=0,
                           style={'backgroundColor': COLORS['policy_loss'], 'color': 'white',
                                 'padding': '10px 20px', 'border': 'none', 'cursor': 'pointer'}),
                html.Span(id='stop-status', style={'marginLeft': '10px', 'color': COLORS['text']}),
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            content,
            WebSocket(id='ws', url='ws://127.0.0.1:8050/ws'),
        ], style={
            'backgroundColor': COLORS['background'],
            'padding': '20px',
            'minHeight': '100vh',
            'margin': '0',
        })

        if len(self.modes) == 1:
            # Single mode callback
            @app.callback(
                Output('graph-single', 'figure'),
                Input('ws', 'message'),
                prevent_initial_call=True
            )
            def update_single_mode(msg):
                self._process_updates()
                fig_out = self._create_figure_for_mode(self.modes[0])
                return self._populate_figure_for_mode(fig_out, self.modes[0])
        else:
            # Multi-mode: update all graphs on each message
            outputs = [Output(f'graph-{mode}', 'figure') for mode in self.modes]

            @app.callback(
                outputs,
                Input('ws', 'message'),
                prevent_initial_call=True
            )
            def update_all_modes(msg):
                self._process_updates()
                figures = []
                for mode in self.modes:
                    fig_out = self._create_figure_for_mode(mode)
                    fig_out = self._populate_figure_for_mode(fig_out, mode)
                    figures.append(fig_out)
                return figures

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

        # WebSocket endpoint
        @sock.route('/ws')
        def ws_endpoint(ws):
            with self._ws_lock:
                self._ws_clients.add(ws)
            try:
                while True:
                    ws.receive(timeout=60)
            except Exception:
                pass
            finally:
                with self._ws_lock:
                    self._ws_clients.discard(ws)

        print("[Dashboard] Starting Dash server at http://127.0.0.1:8050")
        print(f"[Dashboard] Tracking modes: {', '.join(self.modes)}")
        print("[Dashboard] Using WebSocket push (updates on data arrival)")
        print("[Dashboard] Press Ctrl+C to stop")

        app.run(debug=False, use_reloader=False)

    def _push_to_clients(self):
        """Push update notification to all connected WebSocket clients.

        Called directly from update() when new data arrives - no polling needed.
        """
        if not hasattr(self, '_ws_clients'):
            return  # Not in Dash mode or not started yet

        with self._ws_lock:
            dead = []
            for client in self._ws_clients:
                try:
                    client.send(json.dumps({'type': 'update'}))
                except Exception:
                    dead.append(client)
            for d in dead:
                self._ws_clients.discard(d)

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
# LOG LOADING - View historical training runs (TensorBoard format)
# =============================================================================

def load_run_from_logs(log_dir: str) -> RunState:
    """
    Load a training run from TensorBoard event files.

    Args:
        log_dir: Directory containing TensorBoard event files
                 Example: 'generated/logs/ground_truth_20251214_053108'

    Returns:
        RunState populated with historical data
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("Warning: TensorBoard not installed. Cannot load logs.")
        return RunState(mode="unknown")

    # Extract mode from directory name
    basename = os.path.basename(log_dir)
    match = re.match(r'(.+?)_\d{8}_\d{6}', basename)
    mode = match.group(1) if match else basename

    run = RunState(mode=mode)
    run.is_running = False
    run.is_finished = True

    if not os.path.isdir(log_dir):
        print(f"   Warning: Log directory not found: {log_dir}")
        return run

    # Load TensorBoard events
    ea = EventAccumulator(log_dir)
    ea.Reload()

    available_scalars = ea.Tags().get('scalars', [])

    def get_scalar_values(tag: str) -> list[float]:
        if tag not in available_scalars:
            return []
        return [s.value for s in ea.Scalars(tag)]

    # Core metrics
    run.policy_losses = get_scalar_values('loss/policy')
    run.value_losses = get_scalar_values('loss/value')
    run.entropies = get_scalar_values('metrics/entropy')
    run.explained_variances = get_scalar_values('metrics/explained_variance')
    run.rewards = get_scalar_values('reward/episode')
    run.food_ratio_history = get_scalar_values('metrics/food_ratio')

    # Get total steps from any available scalar
    if run.policy_losses and 'loss/policy' in available_scalars:
        steps = [s.step for s in ea.Scalars('loss/policy')]
        run.total_steps = steps[-1] if steps else 0

    # Populate smoothed lists (no smoothing for loaded data)
    run.rewards_smoothed = list(run.rewards)
    run.policy_losses_smoothed = list(run.policy_losses)
    run.value_losses_smoothed = list(run.value_losses)
    run.explained_variances_smoothed = list(run.explained_variances)
    run.food_ratio_smoothed = list(run.food_ratio_history)

    n_points = len(run.policy_losses)
    print(f"   Loaded {n_points} updates from {log_dir}")

    return run


def find_latest_logs(log_dir: str = 'generated/logs', mode: str = None) -> list[str]:
    """
    Find the latest TensorBoard log directories.

    Args:
        log_dir: Parent directory containing run directories
        mode: Optional mode filter (e.g., 'ground_truth')

    Returns:
        List of log directory paths
    """
    if not os.path.isdir(log_dir):
        print(f"Log directory not found: {log_dir}")
        return []

    # Match directories like "ground_truth_20251214_053108"
    pattern = re.compile(r'(.+?)_(\d{8}_\d{6})$')
    mode_timestamps = defaultdict(list)

    for entry in os.listdir(log_dir):
        entry_path = os.path.join(log_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        match = pattern.match(entry)
        if match:
            m, ts = match.groups()
            if mode is None or m == mode:
                mode_timestamps[m].append((ts, entry_path))

    # Get the latest run for each mode
    results = []
    for m, runs in mode_timestamps.items():
        latest = sorted(runs, key=lambda x: x[0])[-1]
        results.append(latest[1])

    return sorted(results)


def view_logs(log_dirs: list[str] = None, log_dir: str = 'generated/logs', mode: str = None):
    """
    Display a visualization of historical training runs.

    Args:
        log_dirs: List of TensorBoard log directories to visualize (optional)
        log_dir: Parent directory to search for logs if log_dirs not provided
        mode: Mode filter when searching for logs
    """
    if log_dirs is None:
        log_dirs = find_latest_logs(log_dir, mode)

    if not log_dirs:
        print("No log directories to display.")
        return

    print(f"\nLoading {len(log_dirs)} training run(s)...")
    runs = [load_run_from_logs(d) for d in log_dirs]

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
        # Write to temp HTML with dark background wrapper
        import tempfile
        import webbrowser
        bg = COLORS['background']
        html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <style>
        html, body {{ background-color: {bg}; margin: 0; padding: 10px; }}
    </style>
</head>
<body>
{dashboard.fig.to_html(full_html=False, include_plotlyjs='cdn')}
</body>
</html>
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            temp_path = f.name
        webbrowser.open(f'file://{temp_path}')


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='View training logs (TensorBoard format)')
    parser.add_argument('--log-dir', default='generated/logs', help='Parent log directory')
    parser.add_argument('--mode', default=None, help='Filter by mode')
    parser.add_argument('logs', nargs='*', help='Specific log directories to view')

    args = parser.parse_args()

    if args.logs:
        view_logs(log_dirs=args.logs)
    else:
        view_logs(log_dir=args.log_dir, mode=args.mode)
