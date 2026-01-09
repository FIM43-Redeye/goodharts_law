"""
Testing Dashboard for trained Goodhart agents.

Clean visualization with timestep-synchronized X-axes across all modes.
Shows efficiency divergence between ground-truth and proxy agents in real time.

Layout:
- Left (75%): 3 stacked line plots (Efficiency, Deaths, Survival) with modes overlaid
- Right (25%): Live stats panel with per-mode numbers

Uses process isolation for zero overhead on the testing thread.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import queue
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue
import threading

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


# Dashboard constants
UPDATE_INTERVAL_MS = 100          # Fast updates for "evolving" feel
QUEUE_POLL_TIMEOUT = 0.02         # Queue poll timeout
QUEUE_MAX_SIZE = 2000             # Max queue size
ROLLING_WINDOW = 20               # Window for rolling average survival


# Import canonical colors and theme from single source of truth
from goodharts.visualization.components import (
    THEME, DEFAULT_COLOR, get_mode_color
)


@dataclass
class ModeStats:
    """
    Live statistics for a single mode, tracked over timesteps.

    All data is indexed by timesteps (shared X-axis), not death count.
    """
    mode: str
    color: str = field(default_factory=lambda: DEFAULT_COLOR)

    # Timestep checkpoints (X-axis values)
    timesteps: list = field(default_factory=list)

    # Line plot data (Y values at each timestep checkpoint)
    efficiency_history: list = field(default_factory=list)      # Running efficiency
    cumulative_deaths: list = field(default_factory=list)       # Total deaths so far
    survival_rolling: list = field(default_factory=list)        # Rolling avg survival

    # Raw counters for computing metrics
    total_timesteps: int = 0
    total_deaths: int = 0
    total_food: int = 0
    total_poison: int = 0

    # Recent survival times for rolling average
    recent_survivals: list = field(default_factory=list)

    # Status
    is_complete: bool = False

    def __post_init__(self):
        self.color = get_mode_color(self.mode)

    def record_checkpoint(self, timesteps: int, food: int, poison: int, deaths: int,
                          survival_times: list[int]):
        """
        Record a checkpoint with current totals.

        Called periodically from the evaluator to update dashboard state.
        """
        self.total_timesteps = timesteps
        self.total_food = food
        self.total_poison = poison
        self.total_deaths = deaths

        # Update recent survivals for rolling average
        if survival_times:
            self.recent_survivals.extend(survival_times)
            # Keep only recent window
            if len(self.recent_survivals) > ROLLING_WINDOW * 2:
                self.recent_survivals = self.recent_survivals[-ROLLING_WINDOW:]

        # Append to history
        self.timesteps.append(timesteps)

        # Efficiency: food / (food + poison)
        total_consumed = food + poison
        efficiency = food / total_consumed if total_consumed > 0 else 1.0
        self.efficiency_history.append(efficiency)

        # Cumulative deaths
        self.cumulative_deaths.append(deaths)

        # Rolling average survival
        if self.recent_survivals:
            window = self.recent_survivals[-ROLLING_WINDOW:]
            self.survival_rolling.append(np.mean(window))
        else:
            # No deaths yet - show 0 (or could show max possible)
            self.survival_rolling.append(0)

    @property
    def current_efficiency(self) -> float:
        """Current overall efficiency."""
        total = self.total_food + self.total_poison
        return self.total_food / total if total > 0 else 1.0

    @property
    def current_survival_avg(self) -> float:
        """Current rolling average survival time."""
        if self.recent_survivals:
            return np.mean(self.recent_survivals[-ROLLING_WINDOW:])
        return 0.0

    @property
    def deaths_per_1k(self) -> float:
        """Deaths per 1000 timesteps."""
        if self.total_timesteps > 0:
            return self.total_deaths / (self.total_timesteps / 1000.0)
        return 0.0


class TestingDashboard:
    """
    Real-time testing dashboard with synchronized timestep axes.

    Layout:
    - Left column: 3 stacked plots (Efficiency, Deaths, Survival)
    - Right column: Live stats panel with numbers

    All modes are overlaid on the same plots for direct comparison.
    """

    def __init__(self, modes: list[str], total_timesteps: int, stop_event=None):
        """
        Args:
            modes: List of modes being tested
            total_timesteps: Target timesteps (for X-axis scaling)
            stop_event: Optional event to signal testing should stop
        """
        self.modes = modes
        self.total_timesteps = total_timesteps
        self._stop_event = stop_event

        # Per-mode stats
        self.stats: dict[str, ModeStats] = {
            mode: ModeStats(mode=mode) for mode in modes
        }

        # Update queue (from evaluator process)
        self.update_queue: queue.Queue = queue.Queue()

        # UI state
        self.running = True
        self.is_complete = False
        self.dirty = True

        # Trace indices for updating
        self._trace_indices: dict[str, dict[str, int]] = {}

    def send_checkpoint(self, mode: str, timesteps: int, food: int, poison: int,
                        deaths: int, survival_times: list[int]):
        """Queue a checkpoint update from evaluator."""
        try:
            self.update_queue.put_nowait((
                'checkpoint', mode,
                (timesteps, food, poison, deaths, survival_times)
            ))
            self.dirty = True
        except queue.Full:
            pass

    def send_complete(self, mode: str):
        """Signal that a mode has finished testing."""
        try:
            self.update_queue.put_nowait(('complete', mode, None))
            self.dirty = True
        except queue.Full:
            pass

    def _process_updates(self):
        """Process pending updates from queue."""
        processed = 0
        max_updates = 500  # Batch limit per frame

        while not self.update_queue.empty() and processed < max_updates:
            try:
                msg_type, mode, data = self.update_queue.get_nowait()
                stats = self.stats.get(mode)
                if stats is None:
                    continue

                if msg_type == 'checkpoint':
                    timesteps, food, poison, deaths, survivals = data
                    stats.record_checkpoint(timesteps, food, poison, deaths, survivals)
                elif msg_type == 'complete':
                    stats.is_complete = True

                processed += 1
            except queue.Empty:
                break

        # Check if all modes complete
        if all(s.is_complete for s in self.stats.values()):
            self.is_complete = True

    def _create_figure(self) -> go.Figure:
        """Create Plotly figure with 3 subplots + annotations for stats."""
        # Create subplots: 3 rows for plots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Efficiency (food / total consumed)',
                           'Cumulative Deaths',
                           f'Survival Time (rolling avg, window={ROLLING_WINDOW})'],
            vertical_spacing=0.12,
            row_heights=[0.33, 0.33, 0.34],
        )

        trace_idx = 0

        # Add traces for each mode
        for mode in self.modes:
            color = get_mode_color(mode)
            self._trace_indices[mode] = {}

            # Efficiency trace
            fig.add_trace(
                go.Scatter(x=[], y=[], mode='lines', name=mode,
                          line=dict(color=color, width=2),
                          showlegend=True),
                row=1, col=1
            )
            self._trace_indices[mode]['efficiency'] = trace_idx
            trace_idx += 1

            # Deaths trace
            fig.add_trace(
                go.Scatter(x=[], y=[], mode='lines', name=mode,
                          line=dict(color=color, width=2),
                          showlegend=False),
                row=2, col=1
            )
            self._trace_indices[mode]['deaths'] = trace_idx
            trace_idx += 1

            # Survival trace
            fig.add_trace(
                go.Scatter(x=[], y=[], mode='lines', name=mode,
                          line=dict(color=color, width=2),
                          showlegend=False),
                row=3, col=1
            )
            self._trace_indices[mode]['survival'] = trace_idx
            trace_idx += 1

        # Layout
        fig.update_layout(
            title=dict(text='Testing Dashboard', x=0.5,
                      font=dict(size=18, color=THEME['text'])),
            plot_bgcolor=THEME['paper'],
            paper_bgcolor=THEME['background'],
            font=dict(color=THEME['text']),
            height=800,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.3)'),
            margin=dict(r=250),  # Space for stats panel
        )

        # Configure axes
        fig.update_xaxes(range=[0, self.total_timesteps], gridcolor=THEME['grid'])
        fig.update_yaxes(gridcolor=THEME['grid'])

        # Efficiency Y-axis
        fig.update_yaxes(range=[0, 1.05], row=1, col=1)
        fig.add_hline(y=0.5, line_dash='dash', line_color='gray',
                      opacity=0.3, row=1, col=1)
        fig.add_hline(y=1.0, line_dash='dash', line_color='#00ff88',
                      opacity=0.2, row=1, col=1)

        # X-axis label on bottom plot only
        fig.update_xaxes(title_text='Timesteps', row=3, col=1)

        return fig

    def _create_stats_annotations(self) -> list[dict]:
        """Create annotations for the stats panel."""
        annotations = []

        # Title
        annotations.append(dict(
            text='<b>Live Stats</b>',
            x=1.02, y=0.98,
            xref='paper', yref='paper',
            showarrow=False,
            font=dict(size=14, color=THEME['text']),
            xanchor='left',
        ))

        # Per-mode stats
        y_pos = 0.90
        spacing = 0.18

        for mode in self.modes:
            stats = self.stats[mode]
            color = get_mode_color(mode)
            status = "DONE" if stats.is_complete else "running"

            # Mode name
            annotations.append(dict(
                text=f'<b>{mode}</b>',
                x=1.02, y=y_pos,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=12, color=color),
                xanchor='left',
            ))

            # Stats text
            stats_text = (
                f"Efficiency: {stats.current_efficiency:>6.1%}<br>"
                f"Deaths:     {stats.total_deaths:>6,}<br>"
                f"Survival:   {stats.current_survival_avg:>6.0f} steps<br>"
                f"Deaths/1k:  {stats.deaths_per_1k:>6.1f}<br>"
                f"[{status}]"
            )
            annotations.append(dict(
                text=stats_text,
                x=1.02, y=y_pos - 0.03,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=10, color='#cccccc', family='monospace'),
                xanchor='left',
                align='left',
            ))

            y_pos -= spacing

        return annotations

    def run(self):
        """Start the dashboard using Dash."""
        app = Dash(__name__)

        # Dark theme - override ALL Dash containers
        bg = THEME['background']
        app.index_string = f'''
<!DOCTYPE html>
<html style="background: {bg} !important;">
    <head>
        {{%metas%}}
        <title>Testing Dashboard</title>
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

        fig = self._create_figure()

        app.layout = html.Div([
            dcc.Graph(id='live-graph', figure=fig),
            dcc.Interval(id='interval', interval=UPDATE_INTERVAL_MS, n_intervals=0),
        ], style={
            'backgroundColor': THEME['background'],
            'padding': '20px',
            'minHeight': '100vh',
        })

        @app.callback(
            Output('live-graph', 'figure'),
            Input('interval', 'n_intervals')
        )
        def update_graph(n):
            self._process_updates()

            fig = self._create_figure()

            # Update data for each mode
            for mode in self.modes:
                stats = self.stats[mode]
                indices = self._trace_indices[mode]

                if not stats.timesteps:
                    continue

                x = stats.timesteps

                # Update traces
                fig.data[indices['efficiency']].x = x
                fig.data[indices['efficiency']].y = stats.efficiency_history

                fig.data[indices['deaths']].x = x
                fig.data[indices['deaths']].y = stats.cumulative_deaths

                fig.data[indices['survival']].x = x
                fig.data[indices['survival']].y = stats.survival_rolling

            # Auto-scale deaths Y-axis
            max_deaths = max((s.total_deaths for s in self.stats.values()), default=0)
            if max_deaths > 0:
                fig.update_yaxes(range=[0, max_deaths * 1.1], row=2, col=1)

            # Auto-scale survival Y-axis
            max_survival = max(
                (max(s.survival_rolling) if s.survival_rolling else 0
                 for s in self.stats.values()),
                default=0
            )
            if max_survival > 0:
                fig.update_yaxes(range=[0, max_survival * 1.2], row=3, col=1)

            # Add stats annotations
            fig.update_layout(annotations=self._create_stats_annotations())

            # Update title if complete
            if self.is_complete:
                fig.update_layout(title=dict(
                    text='Testing Complete (close window when done)',
                    font=dict(color='#88ff88')
                ))

            return fig

        # Run server
        app.run(debug=False, use_reloader=False, port=8051)

    def close(self):
        """Close the dashboard."""
        self.running = False


def _dashboard_worker(modes: list[str], total_timesteps: int,
                      update_queue: MPQueue, stop_event, testing_stop_event):
    """Worker function for dashboard process."""
    dashboard = TestingDashboard(modes, total_timesteps, stop_event=testing_stop_event)

    def poll_queue():
        """Poll multiprocessing queue and forward to dashboard."""
        while not stop_event.is_set():
            try:
                item = update_queue.get(timeout=QUEUE_POLL_TIMEOUT)
                if item is None:
                    # Poison pill - testing done
                    dashboard.running = False
                    break
                msg_type, mode, data = item
                dashboard.update_queue.put_nowait((msg_type, mode, data))
                dashboard.dirty = True
            except queue.Empty:
                pass

    poll_thread = threading.Thread(target=poll_queue, daemon=True)
    poll_thread.start()

    try:
        dashboard.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()


class TestingDashboardProcess:
    """
    Process-isolated testing dashboard.

    Runs Plotly/Dash in a separate process for zero overhead on testing.
    Survives after testing completes so user can inspect results.
    """

    def __init__(self, modes: list[str], total_timesteps: int = 100000):
        self.modes = modes
        self.total_timesteps = total_timesteps

        ctx = mp.get_context('spawn')
        self._queue: MPQueue = ctx.Queue(maxsize=QUEUE_MAX_SIZE)
        self._stop_event = ctx.Event()
        self._testing_stop_event = ctx.Event()
        self._process: Optional[Process] = None

    def start(self):
        """Start dashboard process."""
        ctx = mp.get_context('spawn')
        self._process = ctx.Process(
            target=_dashboard_worker,
            args=(self.modes, self.total_timesteps,
                  self._queue, self._stop_event, self._testing_stop_event),
            daemon=False  # Don't daemon - we want it to survive
        )
        self._process.start()
        print(f"[Dashboard] Started (PID {self._process.pid})")

    def send_checkpoint(self, mode: str, timesteps: int, food: int, poison: int,
                        deaths: int, survival_times: list[int]):
        """Send checkpoint to dashboard."""
        try:
            self._queue.put_nowait((
                'checkpoint', mode,
                (timesteps, food, poison, deaths, survival_times)
            ))
        except queue.Full:
            pass

    def send_complete(self, mode: str):
        """Signal mode completion."""
        try:
            self._queue.put_nowait(('complete', mode, None))
        except queue.Full:
            pass

    # Backwards compatibility aliases (no-ops, kept to prevent AttributeError)
    def send_episode(self, mode: str, episode):
        """Legacy compatibility - no-op, checkpoint-based updates replaced this."""
        pass

    def send_progress(self, mode: str, steps: int, deaths: int):
        """Legacy compatibility - no-op, checkpoint-based updates replaced this."""
        pass

    def is_alive(self) -> bool:
        """Check if dashboard is still running."""
        return self._process is not None and self._process.is_alive()

    def should_stop(self) -> bool:
        """Check if user requested stop via dashboard."""
        return self._testing_stop_event.is_set()

    def stop(self, timeout: float = 2.0):
        """Stop dashboard gracefully."""
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._process is not None:
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)

    def wait(self):
        """Wait for dashboard to close (user closes window)."""
        if self._process is not None:
            self._process.join()


# Legacy alias for backwards compatibility
EvalDashboard = TestingDashboard
EvalDashboardProcess = TestingDashboardProcess


def create_testing_dashboard(modes: list[str],
                             total_timesteps: int = 100000) -> TestingDashboardProcess:
    """Factory for process-isolated testing dashboard."""
    return TestingDashboardProcess(modes, total_timesteps)
