"""
Parallel Stats: Multi-environment statistics dashboard.

Runs many environments in parallel and tracks aggregate statistics
for side-by-side comparison of different training modes.

Demonstrates Goodhart's Law through statistical divergence:
- Ground truth agents maintain high food ratio
- Proxy agents eat poison despite food being more interesting (interestingness doesn't encode harm)
"""
import logging
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from goodharts.visualization.components import (
    THEME, get_mode_color, get_dark_theme_index_string, apply_dark_theme
)


# Dashboard constants
UPDATE_INTERVAL_MS = 200
ROLLING_WINDOW = 50


@dataclass
class ModeStats:
    """
    Per-mode running statistics tracked over time.

    All data is indexed by timesteps for synchronized X-axis.
    """
    mode: str
    color: str = field(default_factory=lambda: '#00aaff')

    # Time series data (Y values at each checkpoint)
    timesteps: list = field(default_factory=list)
    survival_rolling: list = field(default_factory=list)
    food_ratio: list = field(default_factory=list)
    cumulative_deaths: list = field(default_factory=list)

    # Raw counters
    total_timesteps: int = 0
    total_food: int = 0
    total_poison: int = 0
    total_deaths: int = 0

    # Banked survival times (confirmed at death)
    banked_survivals: list = field(default_factory=list)

    # Status
    is_complete: bool = False

    def __post_init__(self):
        self.color = get_mode_color(self.mode)

    def record_checkpoint(
        self,
        timesteps: int,
        food: int,
        poison: int,
        deaths: int,
        death_times: list[int],
        current_ages: list[int] | None = None,
    ):
        """
        Record a checkpoint with current totals.

        Called periodically from the runner to update dashboard state.

        Args:
            timesteps: Total timesteps so far
            food: Total food eaten
            poison: Total poison eaten
            deaths: Total death count
            death_times: Lifetimes of agents that died this checkpoint (banked)
            current_ages: Current ages of all living agents (for rolling metric)
        """
        self.total_timesteps = timesteps
        self.total_food = food
        self.total_poison = poison
        self.total_deaths = deaths

        # Bank death times (these are confirmed final lifetimes)
        if death_times:
            self.banked_survivals.extend(death_times)
            if len(self.banked_survivals) > ROLLING_WINDOW * 2:
                self.banked_survivals = self.banked_survivals[-ROLLING_WINDOW:]

        # Append to time series
        self.timesteps.append(timesteps)

        # Food ratio: food / (food + poison)
        total_consumed = food + poison
        ratio = food / total_consumed if total_consumed > 0 else 1.0
        self.food_ratio.append(ratio)

        # Cumulative deaths
        self.cumulative_deaths.append(deaths)

        # Rolling survival metric: combine banked deaths with current ages
        # This ensures we have data even when agents rarely die
        if current_ages is not None and len(current_ages) > 0:
            # Use current ages as the primary metric (what's happening now)
            # plus recent banked deaths for context
            recent_banked = self.banked_survivals[-ROLLING_WINDOW:] if self.banked_survivals else []
            combined = list(current_ages) + recent_banked
            self.survival_rolling.append(np.mean(combined))
        elif self.banked_survivals:
            # Fall back to banked deaths only
            window = self.banked_survivals[-ROLLING_WINDOW:]
            self.survival_rolling.append(np.mean(window))
        else:
            self.survival_rolling.append(0)

    @property
    def current_efficiency(self) -> float:
        """Current overall food ratio."""
        total = self.total_food + self.total_poison
        return self.total_food / total if total > 0 else 1.0

    @property
    def current_survival_avg(self) -> float:
        """Current rolling average survival time."""
        # Use the most recent rolling value (includes current ages)
        if self.survival_rolling:
            return self.survival_rolling[-1]
        return 0.0


class ParallelStatsDashboard:
    """
    Dash dashboard for parallel environment statistics.

    Layout:
    - Left (75%): 3 stacked line plots with modes overlaid
    - Right (25%): Live stats panel
    """

    def __init__(self, modes: list[str], total_timesteps: int):
        self.modes = modes
        self.total_timesteps = total_timesteps
        self.stats: dict[str, ModeStats] = {
            mode: ModeStats(mode=mode) for mode in modes
        }
        self.update_queue: queue.Queue = queue.Queue()

    def send_checkpoint(
        self,
        mode: str,
        timesteps: int,
        food: int,
        poison: int,
        deaths: int,
        death_times: list[int],
        current_ages: list[int] | None = None,
    ):
        """Queue a checkpoint update."""
        try:
            self.update_queue.put_nowait((
                'checkpoint', mode,
                (timesteps, food, poison, deaths, death_times, current_ages)
            ))
        except queue.Full:
            pass

    def send_complete(self, mode: str):
        """Signal that a mode has finished."""
        try:
            self.update_queue.put_nowait(('complete', mode, None))
        except queue.Full:
            pass

    def _process_updates(self):
        """Process pending updates from queue."""
        max_updates = 500
        processed = 0

        while not self.update_queue.empty() and processed < max_updates:
            try:
                msg_type, mode, data = self.update_queue.get_nowait()
                stats = self.stats.get(mode)
                if stats is None:
                    continue

                if msg_type == 'checkpoint':
                    timesteps, food, poison, deaths, death_times, current_ages = data
                    stats.record_checkpoint(timesteps, food, poison, deaths, death_times, current_ages)
                elif msg_type == 'complete':
                    stats.is_complete = True

                processed += 1
            except queue.Empty:
                break

    def _create_figure(self) -> go.Figure:
        """Create 3-row subplot figure."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                f'Rolling Survival Time (window={ROLLING_WINDOW})',
                'Food Ratio (food / total consumed)',
                'Cumulative Deaths',
            ],
            vertical_spacing=0.12,
            row_heights=[0.33, 0.33, 0.34],
        )

        # Initialize empty traces for each mode
        for mode in self.modes:
            color = get_mode_color(mode)

            # Survival trace
            fig.add_trace(
                go.Scatter(x=[], y=[], mode='lines', name=mode,
                          line=dict(color=color, width=2),
                          showlegend=True),
                row=1, col=1
            )

            # Food ratio trace
            fig.add_trace(
                go.Scatter(x=[], y=[], mode='lines', name=mode,
                          line=dict(color=color, width=2),
                          showlegend=False),
                row=2, col=1
            )

            # Deaths trace
            fig.add_trace(
                go.Scatter(x=[], y=[], mode='lines', name=mode,
                          line=dict(color=color, width=2),
                          showlegend=False),
                row=3, col=1
            )

        # Layout
        fig.update_layout(
            title=dict(text='Parallel Stats Dashboard', x=0.5,
                      font=dict(size=18, color=THEME['text'])),
            height=800,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.3)'),
            margin=dict(r=280),  # Space for stats panel
        )

        # Configure axes
        fig.update_xaxes(range=[0, self.total_timesteps], gridcolor=THEME['grid'])
        fig.update_yaxes(gridcolor=THEME['grid'])

        # Food ratio Y-axis (0-1)
        fig.update_yaxes(range=[0, 1.05], row=2, col=1)
        fig.add_hline(y=0.5, line_dash='dash', line_color='gray',
                      opacity=0.3, row=2, col=1)

        # X-axis label on bottom plot only
        fig.update_xaxes(title_text='Timesteps', row=3, col=1)

        return apply_dark_theme(fig)

    def _create_stats_annotations(self) -> list[dict]:
        """Create annotations for the live stats panel."""
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
        y_pos = 0.88
        spacing = 0.20

        for mode in self.modes:
            stats = self.stats[mode]
            color = stats.color
            status = "DONE" if stats.is_complete else "running"

            # Mode name header
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
                f"Food Ratio: {stats.current_efficiency:>6.1%}<br>"
                f"Deaths:     {stats.total_deaths:>6,}<br>"
                f"Survival:   {stats.current_survival_avg:>6.0f} steps<br>"
                f"[{status}]"
            )
            annotations.append(dict(
                text=stats_text,
                x=1.02, y=y_pos - 0.06,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=11, color=THEME['text'], family='monospace'),
                xanchor='left',
                align='left',
            ))

            y_pos -= spacing

        return annotations

    def _build_figure(self) -> go.Figure:
        """Build figure with current data."""
        fig = self._create_figure()

        # Update trace data
        trace_idx = 0
        for mode in self.modes:
            stats = self.stats[mode]

            if stats.timesteps:
                x = stats.timesteps

                # Survival
                fig.data[trace_idx].x = x
                fig.data[trace_idx].y = stats.survival_rolling

                # Food ratio
                fig.data[trace_idx + 1].x = x
                fig.data[trace_idx + 1].y = stats.food_ratio

                # Deaths
                fig.data[trace_idx + 2].x = x
                fig.data[trace_idx + 2].y = stats.cumulative_deaths

            trace_idx += 3

        # Auto-scale Y axes based on data
        max_survival = max(
            (max(s.survival_rolling) if s.survival_rolling else 0
             for s in self.stats.values()),
            default=0
        )
        if max_survival > 0:
            fig.update_yaxes(range=[0, max_survival * 1.2], row=1, col=1)

        max_deaths = max(s.total_deaths for s in self.stats.values())
        if max_deaths > 0:
            fig.update_yaxes(range=[0, max_deaths * 1.1], row=3, col=1)

        # Add stats annotations
        fig.update_layout(annotations=self._create_stats_annotations())

        return fig

    def run(self, port: int = 8051):
        """Start Dash server (blocking)."""
        app = Dash(__name__)
        app.index_string = get_dark_theme_index_string('Parallel Stats')

        fig = self._create_figure()

        app.layout = html.Div([
            dcc.Graph(id='stats-graph', figure=fig,
                      style={'height': '95vh'},
                      config={'displayModeBar': False}),
            dcc.Interval(id='interval', interval=UPDATE_INTERVAL_MS, n_intervals=0),
        ], style={'backgroundColor': THEME['background'], 'padding': '5px'})

        @app.callback(
            Output('stats-graph', 'figure'),
            Input('interval', 'n_intervals')
        )
        def update(n):
            self._process_updates()
            return self._build_figure()

        # Suppress Werkzeug request logging (POST spam)
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

        print(f"[ParallelStats] Dashboard running at http://localhost:{port}")
        app.run(debug=False, use_reloader=False, port=port)


class ParallelStatsApp:
    """
    Process-isolated parallel stats application.

    Runs dashboard in separate process for zero overhead on simulation.
    """

    def __init__(
        self,
        modes: list[str],
        total_timesteps: int = 100000,
        port: int = 8051,
    ):
        self.modes = modes
        self.total_timesteps = total_timesteps
        self.port = port

        ctx = mp.get_context('spawn')
        self._queue: MPQueue = ctx.Queue(maxsize=2000)
        self._stop_event = ctx.Event()
        self._process: Optional[Process] = None

    def start(self):
        """Start dashboard process."""
        ctx = mp.get_context('spawn')
        self._process = ctx.Process(
            target=_parallel_stats_worker,
            args=(self.modes, self.total_timesteps, self._queue,
                  self._stop_event, self.port),
            daemon=False
        )
        self._process.start()
        print(f"[ParallelStats] Started (PID {self._process.pid})")

    def send_checkpoint(
        self,
        mode: str,
        timesteps: int,
        food: int,
        poison: int,
        deaths: int,
        death_times: list[int],
        current_ages: list[int] | None = None,
    ):
        """Send checkpoint update to dashboard."""
        try:
            self._queue.put_nowait((
                'checkpoint', mode,
                (timesteps, food, poison, deaths, death_times, current_ages)
            ))
        except queue.Full:
            pass  # Dashboard will catch up on next update

    def send_complete(self, mode: str):
        """Signal mode completion."""
        try:
            self._queue.put_nowait(('complete', mode, None))
        except queue.Full:
            pass  # Dashboard will detect completion via other means

    def stop(self, timeout: float = 2.0):
        """Stop dashboard gracefully."""
        self._stop_event.set()
        if self._process and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()

    def wait(self):
        """Wait for dashboard to close."""
        if self._process:
            self._process.join()

    def is_running(self) -> bool:
        """Check if dashboard process is still running."""
        return self._process is not None and self._process.is_alive()


def _parallel_stats_worker(
    modes: list[str],
    total_timesteps: int,
    update_queue: MPQueue,
    stop_event,
    port: int,
):
    """Worker function for parallel stats process."""
    dashboard = ParallelStatsDashboard(modes, total_timesteps)

    def poll_queue():
        while not stop_event.is_set():
            try:
                item = update_queue.get(timeout=0.02)
                msg_type, mode, data = item
                if msg_type == 'checkpoint':
                    timesteps, food, poison, deaths, death_times, current_ages = data
                    dashboard.stats[mode].record_checkpoint(
                        timesteps, food, poison, deaths, death_times, current_ages
                    )
                elif msg_type == 'complete':
                    dashboard.stats[mode].is_complete = True
            except queue.Empty:
                pass
            except Exception:
                pass

    poll_thread = threading.Thread(target=poll_queue, daemon=True)
    poll_thread.start()

    try:
        dashboard.run(port=port)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()


def create_parallel_stats_app(
    modes: list[str],
    total_timesteps: int = 100000,
    port: int = 8051,
) -> ParallelStatsApp:
    """
    Factory function for parallel stats application.

    Args:
        modes: List of training modes to track
        total_timesteps: Expected total timesteps (for X-axis scaling)
        port: Dashboard server port

    Returns:
        ParallelStatsApp instance (call .start() to launch)
    """
    return ParallelStatsApp(modes, total_timesteps, port)
