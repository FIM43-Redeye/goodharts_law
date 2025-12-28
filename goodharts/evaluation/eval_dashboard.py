"""
Testing Dashboard for trained Goodhart agents.

Uses continuous survival paradigm - tracks death events, not episodes.

Real-time visualization of testing metrics:
- Survival Time (steps lived before each death)
- Food per Death
- Poison per Death
- Deaths per 1k Steps (population death rate)
- Efficiency (food / total consumed)

Uses the same process-isolation pattern as TrainingDashboard for zero overhead.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
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
DASHBOARD_MAX_POINTS = 500        # Maximum data points to display
EMA_SMOOTHING_ALPHA = 0.9         # Smoothing factor
ANIMATION_INTERVAL_MS = 200       # Update interval
QUEUE_POLL_TIMEOUT = 0.05         # Queue poll timeout
QUEUE_MAX_SIZE = 1000             # Max queue size


@dataclass
class TestRunState:
    """State for a single testing run.

    Tracks death events with incremental EMA smoothing.
    In continuous survival paradigm, every recorded event IS a death.
    """
    mode: str

    # Death counts
    n_deaths: int = 0
    total_steps: int = 0

    # Raw metrics (per death)
    survival_times: list = field(default_factory=list)
    food_counts: list = field(default_factory=list)
    poison_counts: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    efficiencies: list = field(default_factory=list)

    # Smoothed metrics (incremental EMA)
    survival_smoothed: list = field(default_factory=list)
    food_smoothed: list = field(default_factory=list)
    poison_smoothed: list = field(default_factory=list)
    rewards_smoothed: list = field(default_factory=list)
    efficiency_smoothed: list = field(default_factory=list)

    # Deaths per 1k steps tracking (running estimate)
    deaths_per_1k: list = field(default_factory=list)

    # Status
    is_running: bool = True
    last_update: float = 0.0

    def add_death(self, death: dict, alpha: float = EMA_SMOOTHING_ALPHA):
        """Add death event metrics with incremental smoothing."""
        self.n_deaths += 1

        # Extract from dict (comes from asdict(DeathEvent))
        # Handle both old 'episode_length' and new 'survival_time' keys
        survival = death.get('survival_time', death.get('episode_length', 0))
        food = death['food_eaten']
        poison = death['poison_eaten']
        reward = death['total_reward']

        # Compute efficiency
        total = food + poison
        efficiency = food / total if total > 0 else 1.0

        # Raw values
        self.survival_times.append(survival)
        self.food_counts.append(food)
        self.poison_counts.append(poison)
        self.rewards.append(reward)
        self.efficiencies.append(efficiency)

        # Deaths per 1k steps (rough estimate based on running totals)
        # This is approximate since we don't track exact step count per death
        if self.total_steps > 0:
            rate = self.n_deaths / (self.total_steps / 1000.0)
            self.deaths_per_1k.append(rate)
        else:
            self.deaths_per_1k.append(0.0)

        # Incremental EMA smoothing
        def ema_append(smoothed: list, raw: float) -> None:
            if smoothed:
                prev = smoothed[-1]
                smoothed.append(alpha * prev + (1 - alpha) * raw)
            else:
                smoothed.append(raw)

        ema_append(self.survival_smoothed, survival)
        ema_append(self.food_smoothed, food)
        ema_append(self.poison_smoothed, poison)
        ema_append(self.rewards_smoothed, reward)
        ema_append(self.efficiency_smoothed, efficiency)

        # Trim to prevent unbounded growth
        max_pts = DASHBOARD_MAX_POINTS
        if len(self.survival_times) > max_pts * 2:
            trim = len(self.survival_times) - max_pts
            self.survival_times = self.survival_times[trim:]
            self.food_counts = self.food_counts[trim:]
            self.poison_counts = self.poison_counts[trim:]
            self.rewards = self.rewards[trim:]
            self.efficiencies = self.efficiencies[trim:]
            self.deaths_per_1k = self.deaths_per_1k[trim:]
            self.survival_smoothed = self.survival_smoothed[trim:]
            self.food_smoothed = self.food_smoothed[trim:]
            self.poison_smoothed = self.poison_smoothed[trim:]
            self.rewards_smoothed = self.rewards_smoothed[trim:]
            self.efficiency_smoothed = self.efficiency_smoothed[trim:]

        self.last_update = time.time()

    # Backwards compatibility alias
    def add_episode(self, ep: dict, alpha: float = EMA_SMOOTHING_ALPHA):
        """Backwards compatibility alias for add_death."""
        self.add_death(ep, alpha)


class TestingDashboard:
    """
    Real-time visualization dashboard for testing runs.

    5 panels per mode (continuous survival paradigm):
    1. Survival Time (steps lived before death)
    2. Food per Death
    3. Poison per Death
    4. Deaths/1k Steps (population death rate)
    5. Efficiency (line, key Goodhart metric)
    """
    
    def __init__(self, modes: list[str], stop_event=None):
        """
        Args:
            modes: List of modes being tested
            stop_event: Optional event to signal testing should stop
        """
        self.modes = modes
        self.n_runs = len(modes)
        self._stop_event = stop_event
        
        # State per run
        self.runs: dict[str, TestRunState] = {
            mode: TestRunState(mode=mode) for mode in modes
        }
        
        # Update queue
        self.update_queue: queue.Queue = queue.Queue()
        
        # UI state
        self.paused = False
        self.dirty = False
        self.fig = None
        self.axes: dict[str, dict[str, Any]] = {}
        self.artists: dict[str, dict[str, Any]] = {}
        self.running = True
    
    def send_death(self, mode: str, death: DeathEvent):
        """Queue a death event update."""
        from dataclasses import asdict
        try:
            self.update_queue.put_nowait(('death', mode, asdict(death)))
            self.dirty = True
        except queue.Full:
            pass

    # Backwards compatibility alias
    def send_episode(self, mode: str, episode):
        """Backwards compatibility alias for send_death."""
        self.send_death(mode, episode)
    
    def send_progress(self, mode: str, steps: int, deaths: int):
        """Queue a progress update."""
        try:
            self.update_queue.put_nowait(('progress', mode, (steps, deaths)))
            self.dirty = True
        except queue.Full:
            pass

    def _process_updates(self):
        """Process pending updates."""
        max_updates = DASHBOARD_MAX_POINTS
        processed = 0

        while not self.update_queue.empty() and processed < max_updates:
            try:
                update_type, mode, data = self.update_queue.get_nowait()
                run = self.runs.get(mode)
                if run is None:
                    continue

                if update_type == 'death' or update_type == 'episode':
                    run.add_death(data)
                elif update_type == 'progress':
                    steps, deaths = data
                    run.total_steps = steps

                processed += 1
            except queue.Empty:
                break
    
    def _create_figure(self):
        """Create matplotlib figure with 5-panel layout per mode."""
        plt.style.use('dark_background')

        rows = self.n_runs
        cols = 5  # Survival, Food, Poison, Deaths/1k, Efficiency

        fig_height = 3 * rows + 1
        self.fig = plt.figure(figsize=(12, fig_height))
        self.fig.suptitle("Survival Analysis Dashboard", fontsize=14, fontweight='bold', color='white')

        gs = gridspec.GridSpec(rows, cols, figure=self.fig,
                               hspace=0.4, wspace=0.35,
                               left=0.05, right=0.97, top=0.9, bottom=0.1)

        for row_idx, mode in enumerate(self.modes):
            self._create_run_row(gs, row_idx, mode)

        # Add pause/stop buttons
        self._add_controls()
    
    def _create_run_row(self, gs, row_idx: int, mode: str):
        """Create 5-panel row for a mode."""
        self.axes[mode] = {}
        self.artists[mode] = {}

        # Column titles (continuous survival paradigm)
        titles = ['Survival', 'Food/Death', 'Poison/Death', 'Deaths/1k', 'Efficiency']
        colors = ['cyan', 'lime', 'red', 'magenta', 'cyan']

        for col, (title, color) in enumerate(zip(titles, colors)):
            ax = self.fig.add_subplot(gs[row_idx, col])

            # Mode label only on first column
            if col == 0:
                ax.set_title(f"{mode}: {title}", fontsize=10, fontweight='bold', color='white')
            else:
                ax.set_title(title, fontsize=10, color='white')

            ax.tick_params(colors='gray', labelsize=8)
            ax.grid(True, alpha=0.2, color='gray')

            self.axes[mode][title] = ax

            if title == 'Deaths/1k':
                # Line plot for death rate
                line, = ax.plot([], [], color=color, linewidth=1.5, alpha=0.9)
                ax.set_ylim(0, 10)  # Initial scale, will auto-adjust
                self.artists[mode][title] = line
            elif title == 'Efficiency':
                # Line with reference at 1.0
                ax.axhline(y=1.0, color='lime', linestyle='--', alpha=0.3, linewidth=1)
                line, = ax.plot([], [], color=color, linewidth=1.5, alpha=0.9)
                ax.set_ylim(0, 1.1)
                self.artists[mode][title] = line
            else:
                # Standard line plot
                line, = ax.plot([], [], color=color, linewidth=1.5, alpha=0.9)
                self.artists[mode][title] = line
    
    def _add_controls(self):
        """Add pause and stop buttons."""
        ax_pause = self.fig.add_axes([0.45, 0.02, 0.05, 0.03])
        ax_stop = self.fig.add_axes([0.51, 0.02, 0.05, 0.03])
        
        self.btn_pause = Button(ax_pause, 'Pause', color='dimgray', hovercolor='gray')
        self.btn_stop = Button(ax_stop, 'Stop', color='darkred', hovercolor='red')
        
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_stop.on_clicked(self._on_stop)
    
    def _on_pause(self, event):
        """Toggle pause."""
        self.paused = not self.paused
        self.btn_pause.label.set_text('Resume' if self.paused else 'Pause')
    
    def _on_stop(self, event):
        """Request stop."""
        if self._stop_event:
            self._stop_event.set()
        print("[Dashboard] Stop requested")
    
    def _update_frame(self, frame):
        """Animation update callback."""
        if not self.running:
            return []

        if self.paused:
            return []

        # Only process if dirty
        if not self.dirty:
            return []

        self._process_updates()
        self.dirty = False

        all_artists = []

        for mode in self.modes:
            run = self.runs[mode]
            artists = self.artists[mode]
            axes = self.axes[mode]

            n = len(run.survival_times)
            if n == 0:
                continue

            x = np.arange(n)

            # Update each panel
            for title, smoothed_attr in [
                ('Survival', 'survival_smoothed'),
                ('Food/Death', 'food_smoothed'),
                ('Poison/Death', 'poison_smoothed'),
            ]:
                line = artists[title]
                data = getattr(run, smoothed_attr)
                if data:
                    line.set_data(x[:len(data)], data)
                    ax = axes[title]
                    ax.set_xlim(0, max(n, 10))
                    if data:
                        ymin, ymax = min(data), max(data)
                        margin = (ymax - ymin) * 0.1 or 1
                        ax.set_ylim(max(0, ymin - margin), ymax + margin)
                all_artists.append(line)

            # Deaths per 1k steps
            if run.deaths_per_1k:
                line = artists['Deaths/1k']
                line.set_data(x[:len(run.deaths_per_1k)], run.deaths_per_1k)
                ax = axes['Deaths/1k']
                ax.set_xlim(0, max(n, 10))
                ymax = max(run.deaths_per_1k) * 1.2 or 10
                ax.set_ylim(0, ymax)
                all_artists.append(line)

            # Efficiency
            if run.efficiency_smoothed:
                line = artists['Efficiency']
                line.set_data(x[:len(run.efficiency_smoothed)], run.efficiency_smoothed)
                axes['Efficiency'].set_xlim(0, max(n, 10))
                all_artists.append(line)

        return all_artists
    
    def run(self):
        """Start the dashboard (blocking)."""
        self._create_figure()
        
        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            interval=ANIMATION_INTERVAL_MS,
            blit=True,
            cache_frame_data=False
        )
        
        plt.show()
    
    def close(self):
        """Close the dashboard."""
        self.running = False
        plt.close(self.fig)


def _dashboard_worker(modes: list[str], update_queue: MPQueue, stop_event, testing_stop_event):
    """Worker function for dashboard process."""
    dashboard = TestingDashboard(modes, stop_event=testing_stop_event)
    
    def poll_queue():
        while not stop_event.is_set():
            try:
                item = update_queue.get(timeout=QUEUE_POLL_TIMEOUT)
                if item is None:
                    dashboard.running = False
                    break
                update_type, mode, data = item
                if update_type == 'episode':
                    dashboard.update_queue.put_nowait(('episode', mode, data))
                    dashboard.dirty = True
                elif update_type == 'progress':
                    dashboard.update_queue.put_nowait(('progress', mode, data))
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
    
    Runs in separate process for zero overhead on testing thread.
    """
    
    def __init__(self, modes: list[str]):
        self.modes = modes
        
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
            args=(self.modes, self._queue, self._stop_event, self._testing_stop_event),
            daemon=True
        )
        self._process.start()
        print(f"[Dashboard] Started testing dashboard (PID {self._process.pid})")
    
    def send_episode(self, mode: str, episode):
        """Send episode to dashboard."""
        from dataclasses import asdict
        try:
            ep_dict = asdict(episode) if hasattr(episode, '__dataclass_fields__') else episode
            self._queue.put_nowait(('episode', mode, ep_dict))
        except queue.Full:
            pass
    
    def send_progress(self, mode: str, steps: int, episodes: int):
        """Send progress update."""
        try:
            self._queue.put_nowait(('progress', mode, (steps, episodes)))
        except queue.Full:
            pass
    
    def is_alive(self) -> bool:
        """Check if dashboard is still running."""
        return self._process is not None and self._process.is_alive()
    
    def should_stop(self) -> bool:
        """Check if user requested stop."""
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
        """Wait for dashboard to close."""
        if self._process is not None:
            self._process.join()


def create_testing_dashboard(modes: list[str]) -> TestingDashboardProcess:
    """Factory for process-isolated testing dashboard."""
    return TestingDashboardProcess(modes)
