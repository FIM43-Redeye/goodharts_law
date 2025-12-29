"""
Evaluation Dashboard for trained Goodhart agents.

Clean visualization with timestep-synchronized X-axes across all modes.
Shows efficiency divergence between ground-truth and proxy agents in real time.

Layout:
- Left (75%): 3 stacked line plots (Efficiency, Deaths, Survival) with modes overlaid
- Right (25%): Live stats panel with per-mode numbers

Uses process isolation for zero overhead on the evaluation thread.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional
import queue
import time
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue
import threading

from goodharts.evaluation.evaluator import DeathEvent


# Dashboard constants
ANIMATION_INTERVAL_MS = 100       # Fast updates for "evolving" feel
QUEUE_POLL_TIMEOUT = 0.02         # Queue poll timeout
QUEUE_MAX_SIZE = 2000             # Max queue size
ROLLING_WINDOW = 20               # Window for rolling average survival


# Color scheme for modes (consistent across plots and stats)
MODE_COLORS = {
    'ground_truth': '#00ff88',      # Bright green
    'ground_truth_handhold': '#88ff00',  # Lime
    'proxy': '#ff4444',             # Red
    'proxy_jammed': '#ffaa00',      # Orange
}
DEFAULT_COLOR = '#00aaff'           # Cyan fallback


def get_mode_color(mode: str) -> str:
    """Get consistent color for a mode."""
    return MODE_COLORS.get(mode, DEFAULT_COLOR)


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


class EvalDashboard:
    """
    Real-time evaluation dashboard with synchronized timestep axes.

    Layout:
    - Left column: 3 stacked plots (Efficiency, Deaths, Survival)
    - Right column: Live stats panel with numbers

    All modes are overlaid on the same plots for direct comparison.
    """

    def __init__(self, modes: list[str], total_timesteps: int, stop_event=None):
        """
        Args:
            modes: List of modes being evaluated
            total_timesteps: Target timesteps (for X-axis scaling)
            stop_event: Optional event to signal evaluation should stop
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
        self.fig = None
        self.axes: dict[str, Any] = {}
        self.lines: dict[str, dict[str, Any]] = {}  # mode -> plot_name -> line
        self.stat_texts: dict[str, Any] = {}        # mode -> text artist
        self.running = True
        self.is_complete = False
        self.dirty = True  # Start dirty to draw initial state

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
        """Signal that a mode has finished evaluation."""
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

    def _create_figure(self):
        """Create matplotlib figure with 3-plot + stats layout."""
        plt.style.use('dark_background')

        # Figure size
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor('#1a1a2e')

        # GridSpec: 3 rows, 4 columns (3 for plots, 1 for stats)
        gs = gridspec.GridSpec(3, 4, figure=self.fig,
                               width_ratios=[3, 3, 3, 2],
                               hspace=0.3, wspace=0.3,
                               left=0.06, right=0.98, top=0.92, bottom=0.08)

        # Title
        self.fig.suptitle('Evaluation Dashboard', fontsize=16,
                         fontweight='bold', color='white')

        # Create the 3 main plots (spanning columns 0-2)
        self._create_efficiency_plot(gs[0, :3])
        self._create_deaths_plot(gs[1, :3])
        self._create_survival_plot(gs[2, :3])

        # Create stats panel (column 3, all rows)
        self._create_stats_panel(gs[:, 3])

        # Initialize lines for each mode
        self._init_mode_lines()

        # Legend (in efficiency plot)
        if len(self.modes) > 1:
            self.axes['efficiency'].legend(loc='lower left', fontsize=9,
                                           framealpha=0.7, facecolor='#2a2a4e')

    def _create_efficiency_plot(self, gs_spec):
        """Create efficiency over time plot."""
        ax = self.fig.add_subplot(gs_spec)
        ax.set_facecolor('#16213e')
        ax.set_title('Efficiency (food / total consumed)', fontsize=11,
                    color='white', fontweight='bold')
        ax.set_ylabel('Efficiency', fontsize=10, color='gray')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, self.total_timesteps)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=1.0, color='#00ff88', linestyle='--', alpha=0.2, linewidth=1)
        ax.grid(True, alpha=0.15, color='gray')
        ax.tick_params(colors='gray', labelsize=9)
        self.axes['efficiency'] = ax

    def _create_deaths_plot(self, gs_spec):
        """Create cumulative deaths plot."""
        ax = self.fig.add_subplot(gs_spec)
        ax.set_facecolor('#16213e')
        ax.set_title('Cumulative Deaths', fontsize=11,
                    color='white', fontweight='bold')
        ax.set_ylabel('Deaths', fontsize=10, color='gray')
        ax.set_xlim(0, self.total_timesteps)
        ax.grid(True, alpha=0.15, color='gray')
        ax.tick_params(colors='gray', labelsize=9)
        self.axes['deaths'] = ax

    def _create_survival_plot(self, gs_spec):
        """Create rolling average survival time plot."""
        ax = self.fig.add_subplot(gs_spec)
        ax.set_facecolor('#16213e')
        ax.set_title(f'Survival Time (rolling avg, window={ROLLING_WINDOW})',
                    fontsize=11, color='white', fontweight='bold')
        ax.set_ylabel('Steps', fontsize=10, color='gray')
        ax.set_xlabel('Timesteps', fontsize=10, color='gray')
        ax.set_xlim(0, self.total_timesteps)
        ax.grid(True, alpha=0.15, color='gray')
        ax.tick_params(colors='gray', labelsize=9)
        self.axes['survival'] = ax

    def _create_stats_panel(self, gs_spec):
        """Create live stats panel."""
        ax = self.fig.add_subplot(gs_spec)
        ax.set_facecolor('#0f0f23')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Title for stats panel
        ax.text(0.5, 0.97, 'Live Stats', fontsize=13, fontweight='bold',
               color='white', ha='center', va='top',
               transform=ax.transAxes)

        # Create text placeholders for each mode
        n_modes = len(self.modes)
        spacing = 0.85 / max(n_modes, 1)

        for i, mode in enumerate(self.modes):
            y_pos = 0.88 - (i * spacing)
            color = get_mode_color(mode)

            # Mode name header
            ax.text(0.1, y_pos, mode, fontsize=11, fontweight='bold',
                   color=color, ha='left', va='top',
                   transform=ax.transAxes)

            # Stats text (will be updated)
            text = ax.text(0.1, y_pos - 0.05, '', fontsize=10,
                          color='#cccccc', ha='left', va='top',
                          transform=ax.transAxes, family='monospace',
                          linespacing=1.4)
            self.stat_texts[mode] = text

        self.axes['stats'] = ax

    def _init_mode_lines(self):
        """Initialize line artists for each mode on each plot."""
        for mode in self.modes:
            color = get_mode_color(mode)
            self.lines[mode] = {}

            # Efficiency line
            line, = self.axes['efficiency'].plot(
                [], [], color=color, linewidth=2, alpha=0.9, label=mode
            )
            self.lines[mode]['efficiency'] = line

            # Deaths line
            line, = self.axes['deaths'].plot(
                [], [], color=color, linewidth=2, alpha=0.9
            )
            self.lines[mode]['deaths'] = line

            # Survival line
            line, = self.axes['survival'].plot(
                [], [], color=color, linewidth=2, alpha=0.9
            )
            self.lines[mode]['survival'] = line

    def _update_frame(self, frame):
        """Animation update callback."""
        if not self.running:
            return []

        # Process any pending updates
        self._process_updates()

        # Skip redraw if nothing changed
        if not self.dirty:
            return []
        self.dirty = False

        all_artists = []
        max_deaths = 0
        max_survival = 0

        for mode in self.modes:
            stats = self.stats[mode]
            lines = self.lines[mode]

            if not stats.timesteps:
                continue

            x = np.array(stats.timesteps)

            # Update efficiency line
            y_eff = np.array(stats.efficiency_history)
            lines['efficiency'].set_data(x, y_eff)
            all_artists.append(lines['efficiency'])

            # Update deaths line
            y_deaths = np.array(stats.cumulative_deaths)
            lines['deaths'].set_data(x, y_deaths)
            all_artists.append(lines['deaths'])
            max_deaths = max(max_deaths, y_deaths[-1] if len(y_deaths) else 0)

            # Update survival line
            y_surv = np.array(stats.survival_rolling)
            lines['survival'].set_data(x, y_surv)
            all_artists.append(lines['survival'])
            if len(y_surv) > 0:
                max_survival = max(max_survival, np.max(y_surv))

            # Update stats text
            text = self.stat_texts[mode]
            status = "DONE" if stats.is_complete else "running"
            text.set_text(
                f"Efficiency: {stats.current_efficiency:>6.1%}\n"
                f"Deaths:     {stats.total_deaths:>6,}\n"
                f"Survival:   {stats.current_survival_avg:>6.0f} steps\n"
                f"Deaths/1k:  {stats.deaths_per_1k:>6.1f}\n"
                f"[{status}]"
            )
            all_artists.append(text)

        # Auto-scale Y axes
        if max_deaths > 0:
            self.axes['deaths'].set_ylim(0, max_deaths * 1.1)
        if max_survival > 0:
            self.axes['survival'].set_ylim(0, max_survival * 1.2)

        # Update title if complete
        if self.is_complete:
            self.fig.suptitle('Evaluation Complete (close window when done)',
                            fontsize=16, fontweight='bold', color='#88ff88')

        return all_artists

    def run(self):
        """Start the dashboard (blocking)."""
        self._create_figure()

        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            interval=ANIMATION_INTERVAL_MS,
            blit=False,  # Need full redraw for text updates
            cache_frame_data=False
        )

        plt.show()

    def close(self):
        """Close the dashboard."""
        self.running = False
        if self.fig:
            plt.close(self.fig)


def _dashboard_worker(modes: list[str], total_timesteps: int,
                      update_queue: MPQueue, stop_event, eval_stop_event):
    """Worker function for dashboard process."""
    dashboard = EvalDashboard(modes, total_timesteps, stop_event=eval_stop_event)

    def poll_queue():
        """Poll multiprocessing queue and forward to dashboard."""
        while not stop_event.is_set():
            try:
                item = update_queue.get(timeout=QUEUE_POLL_TIMEOUT)
                if item is None:
                    # Poison pill - evaluation done
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


class EvalDashboardProcess:
    """
    Process-isolated evaluation dashboard.

    Runs matplotlib in a separate process for zero overhead on evaluation.
    Survives after evaluation completes so user can inspect results.
    """

    def __init__(self, modes: list[str], total_timesteps: int = 100000):
        self.modes = modes
        self.total_timesteps = total_timesteps

        ctx = mp.get_context('spawn')
        self._queue: MPQueue = ctx.Queue(maxsize=QUEUE_MAX_SIZE)
        self._stop_event = ctx.Event()
        self._eval_stop_event = ctx.Event()
        self._process: Optional[Process] = None

    def start(self):
        """Start dashboard process."""
        ctx = mp.get_context('spawn')
        self._process = ctx.Process(
            target=_dashboard_worker,
            args=(self.modes, self.total_timesteps,
                  self._queue, self._stop_event, self._eval_stop_event),
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

    # Backwards compatibility aliases
    def send_episode(self, mode: str, episode):
        """Legacy compatibility - convert episode to checkpoint."""
        from dataclasses import asdict
        ep = asdict(episode) if hasattr(episode, '__dataclass_fields__') else episode
        # Can't fully reconstruct, but this keeps old code from crashing
        pass

    def send_progress(self, mode: str, steps: int, deaths: int):
        """Legacy compatibility."""
        pass

    def is_alive(self) -> bool:
        """Check if dashboard is still running."""
        return self._process is not None and self._process.is_alive()

    def should_stop(self) -> bool:
        """Check if user requested stop via dashboard."""
        return self._eval_stop_event.is_set()

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


def create_testing_dashboard(modes: list[str],
                             total_timesteps: int = 100000) -> EvalDashboardProcess:
    """Factory for process-isolated evaluation dashboard."""
    return EvalDashboardProcess(modes, total_timesteps)
