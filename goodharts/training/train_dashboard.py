"""
Unified Training Dashboard (Graph Only).

Single matplotlib window showing:
- Reward History
- Losses (Policy, Value)
- Entropy
- Explained Variance
- Behavior (Food/Poison)
- Action Distribution

Optimized for fast vectorized training (no image rendering).
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
import numpy as np
from dataclasses import dataclass, field
from typing import Any
import queue
import time

from goodharts.behaviors.action_space import get_action_labels, DISCRETE_8, create_action_space


# Dashboard constants
DASHBOARD_MAX_POINTS = 1024       # Maximum data points to display in graphs
EMA_SMOOTHING_ALPHA = 0.9         # Exponential moving average smoothing factor
ANIMATION_INTERVAL_MS = 50        # Milliseconds between dashboard frame updates (low = responsive)
QUEUE_POLL_TIMEOUT = 0.01         # Seconds to wait when polling update queue
QUEUE_MAX_SIZE = 1000             # Maximum pending updates in queue


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
    food_ratio_history: list = field(default_factory=list)  # food / (food + poison)

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
        """Append new metrics with incremental EMA smoothing.

        This is O(1) per call instead of O(n) per frame.
        """
        # Raw values
        self.policy_losses.append(p_loss)
        self.value_losses.append(v_loss)
        self.entropies.append(ent)
        self.explained_variances.append(ev)
        self.rewards.append(reward)
        self.food_history.append(food)
        self.poison_history.append(poison)

        # Compute food_ratio (curriculum-invariant discrimination metric)
        total = food + poison
        food_ratio = food / total if total > 0 else 0.5
        self.food_ratio_history.append(food_ratio)

        # Incremental EMA: smoothed = alpha * prev_smoothed + (1-alpha) * new_value
        def ema_append(smoothed_list: list, raw_value: float) -> None:
            if smoothed_list:
                prev = smoothed_list[-1]
                smoothed_list.append(alpha * prev + (1 - alpha) * raw_value)
            else:
                smoothed_list.append(raw_value)

        ema_append(self.policy_losses_smoothed, p_loss)
        ema_append(self.value_losses_smoothed, v_loss)
        ema_append(self.explained_variances_smoothed, ev)
        ema_append(self.rewards_smoothed, reward)
        ema_append(self.food_smoothed, food)
        ema_append(self.poison_smoothed, poison)
        ema_append(self.food_ratio_smoothed, food_ratio)

        # Trim to max points to prevent unbounded growth
        max_pts = DASHBOARD_MAX_POINTS
        if len(self.rewards) > max_pts * 2:
            # Trim all lists to last max_pts
            trim_from = len(self.rewards) - max_pts
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


class TrainingDashboard:
    """
    Unified visualization dashboard for training runs.
    """

    def __init__(self, modes: list[str], n_actions: int = 8, training_stop_event=None):
        """
        Args:
            modes: List of training modes to display
            n_actions: Number of actions for distribution plot
            training_stop_event: Optional multiprocessing.Event to signal training stop.
                                 If provided, the Stop button sets this event.
                                 If None, Stop button has no effect (standalone mode).
        """
        self.modes = modes
        self.n_actions = n_actions
        self.n_runs = len(modes)
        self._training_stop_event = training_stop_event

        # State per run
        self.runs: dict[str, RunState] = {
            mode: RunState(mode=mode, n_actions=n_actions)
            for mode in modes
        }

        # Thread-safe update queue
        self.update_queue: queue.Queue = queue.Queue()

        # UI State
        self.paused = False
        self.dirty = False  # Set when new data arrives, cleared after redraw
        self.fig = None
        self.axes: dict[str, dict[str, Any]] = {}
        self.artists: dict[str, dict[str, Any]] = {}
        self.running = True
    
    def update(self, mode: str, update_type: str, data: Any):
        """
        Queue an update from a training thread.
        
        Args:
            mode: Training mode
            update_type: 'update' (unified), 'finished'
            data: Payload
        """
        try:
            self.update_queue.put_nowait((mode, update_type, data))
            self.dirty = True  # Signal that redraw is needed
        except queue.Full:
            pass
    
    def _process_updates(self):
        """Process all pending updates."""
        updates_processed = 0
        max_updates = DASHBOARD_MAX_POINTS  # Process more updates per frame to catch up
        
        while not self.update_queue.empty() and updates_processed < max_updates:
            try:
                mode, update_type, data = self.update_queue.get_nowait()
                run = self.runs.get(mode)
                if run is None:
                    continue
                
                run.last_update = time.time()
                
                if update_type == 'update':
                    # Unified Update Payload:
                    # {
                    #   'ppo': (p_loss, v_loss, ent, probs, ev),
                    #   'episodes': { 'reward': ..., 'food': ..., 'poison': ... } OR None
                    #   'steps': int
                    # }

                    ppo = data.get('ppo')
                    episodes = data.get('episodes')
                    steps = data.get('steps', 0)

                    if ppo:
                        p_loss, v_loss, ent, probs, ev = ppo
                        run.current_action_probs = np.array(probs)

                        # Episode stats (forward fill if missing)
                        if episodes:
                            reward = episodes['reward']
                            food = episodes['food']
                            poison = episodes['poison']
                        else:
                            reward = run.rewards[-1] if run.rewards else 0.0
                            food = run.food_history[-1] if run.food_history else 0.0
                            poison = run.poison_history[-1] if run.poison_history else 0.0

                        # Append with incremental smoothing (O(1) per update)
                        run.append_metrics(p_loss, v_loss, ent, ev, reward, food, poison)

                    run.total_steps = steps
                
                elif update_type == 'finished':
                    run.is_finished = True
                    run.is_running = False
                
                updates_processed += 1
                
            except queue.Empty:
                break

    def _create_figure(self):
        """Create the matplotlib figure with concise layout."""
        # Clean styling
        plt.style.use('dark_background')
        
        # Layout: 1 row per run
        rows = self.n_runs
        cols = 7  # Reward, Pol Loss, Val Loss, Entropy, Expl Var, Behavior, Action Dist
        
        # Adjust height based on rows
        fig_height = 3 * rows + 1
        self.fig = plt.figure(figsize=(16, fig_height))
        self.fig.suptitle("Training Dashboard (Unified)", fontsize=14, fontweight='bold', color='white')
        
        # Grid
        gs = gridspec.GridSpec(rows, cols, figure=self.fig, 
                               hspace=0.4, wspace=0.3, 
                               left=0.03, right=0.97, top=0.9, bottom=0.1)
        
        for i, mode in enumerate(self.modes):
            self._create_run_row(gs, i, mode)
        
        self._create_controls()
    
    def _create_run_row(self, gs, row_idx: int, mode: str):
        """Create a row of graphs for one run."""
        self.axes[mode] = {}
        self.artists[mode] = {}
        
        # 1. Reward
        ax = self.fig.add_subplot(gs[row_idx, 0])
        ax.set_title(f"{mode}: Rewards", fontsize=10, fontweight='bold', color='yellow')
        ax.grid(True, alpha=0.15)
        # ax.set_xlabel("Update") # Save space
        
        l_rew, = ax.plot([], [], 'c-', linewidth=1.5, alpha=0.9)
        self.axes[mode]['reward'] = ax
        self.artists[mode]['reward'] = l_rew
        
        # 2. Policy Loss
        ax = self.fig.add_subplot(gs[row_idx, 1])
        ax.set_title("Pol Loss", fontsize=10, color='silver')
        ax.grid(True, alpha=0.15)
        
        l_pol, = ax.plot([], [], 'r-', linewidth=1, label='Pol', alpha=0.8)
        self.axes[mode]['pol_loss'] = ax
        self.artists[mode]['policy'] = l_pol
        
        # 3. Value Loss
        ax = self.fig.add_subplot(gs[row_idx, 2])
        ax.set_title("Val Loss", fontsize=10, color='silver')
        ax.grid(True, alpha=0.15)
        
        l_val, = ax.plot([], [], 'g-', linewidth=1, label='Val', alpha=0.8)
        self.axes[mode]['val_loss'] = ax
        self.artists[mode]['value'] = l_val
        
        # 4. Entropy
        ax = self.fig.add_subplot(gs[row_idx, 3])
        ax.set_title("Entropy", fontsize=10, color='silver')
        ax.grid(True, alpha=0.15)
        
        max_ent = np.log(self.n_actions)
        ax.axhline(y=max_ent, color='gray', linestyle='--', alpha=0.3)
        ax.set_ylim(0, max_ent * 1.1)
        
        l_ent, = ax.plot([], [], 'm-', linewidth=1.5, alpha=0.9)
        self.axes[mode]['entropy'] = ax
        self.artists[mode]['entropy'] = l_ent
        
        # 5. Explained Variance
        ax = self.fig.add_subplot(gs[row_idx, 4])
        ax.set_title("Expl. Var.", fontsize=10, color='silver')
        ax.grid(True, alpha=0.15)
        ax.axhline(y=1.0, color='lime', linestyle='--', alpha=0.3)
        ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylim(-0.5, 1.1)
        
        l_ev, = ax.plot([], [], 'y-', linewidth=1.5, alpha=0.9)
        self.axes[mode]['ev'] = ax
        self.artists[mode]['ev'] = l_ev
        
        # 6. Behavior (food/poison counts + food_ratio on secondary axis)
        ax = self.fig.add_subplot(gs[row_idx, 5])
        ax.set_title("Behavior", fontsize=10, color='silver')
        ax.grid(True, alpha=0.15)

        l_food, = ax.plot([], [], 'g-', linewidth=1.5, alpha=0.9, label='Food')
        l_poison, = ax.plot([], [], 'r-', linewidth=1.5, alpha=0.9, label='Pois')

        # Secondary y-axis for food_ratio (0-1 scale)
        ax2 = ax.twinx()
        ax2.set_ylim(0, 1)
        ax2.axhline(y=0.5, color='cyan', linestyle=':', alpha=0.4)  # Random baseline
        l_ratio, = ax2.plot([], [], 'c--', linewidth=1.5, alpha=0.8, label='Ratio')
        ax2.tick_params(axis='y', labelcolor='cyan', labelsize=7)
        ax2.set_ylabel('F/(F+P)', color='cyan', fontsize=7)

        # Combined legend
        lines = [l_food, l_poison, l_ratio]
        labels = ['Food', 'Pois', 'Ratio']
        ax.legend(lines, labels, fontsize=7, loc='upper right', framealpha=0.3)

        self.axes[mode]['beh'] = ax
        self.axes[mode]['beh2'] = ax2  # Secondary axis for ratio
        self.artists[mode]['food'] = l_food
        self.artists[mode]['poison'] = l_poison
        self.artists[mode]['food_ratio'] = l_ratio
        
        # 7. Action Distribution (Bar Chart)
        ax = self.fig.add_subplot(gs[row_idx, 6])
        ax.set_title("Actions", fontsize=10, color='silver')
        ax.grid(False)
        ax.set_ylim(0, 0.6) # reasonable max prob
        ax.axhline(y=1.0/self.n_actions, color='gray', linestyle='--', alpha=0.3)
        
        # Generate proper action labels from action space
        if self.n_actions == 8:
            action_labels = get_action_labels(DISCRETE_8)
        else:
            # For non-standard action counts, create temporary action space
            action_space = create_action_space('discrete_grid', max_move_distance=1)
            if action_space.n_outputs == self.n_actions:
                action_labels = get_action_labels(action_space)
            else:
                action_labels = [str(i) for i in range(self.n_actions)]

        x = np.arange(self.n_actions)
        bars = ax.bar(x, np.zeros(self.n_actions), color='#00d9ff', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(action_labels)
        
        self.axes[mode]['act'] = ax
        self.artists[mode]['act_bars'] = bars

    def _create_controls(self):
        """Create control buttons."""
        # Pause button
        ax_pause = self.fig.add_axes([0.42, 0.02, 0.08, 0.04])
        self.btn_pause = Button(ax_pause, 'Pause', color='#333333', hovercolor='#555555')
        self.btn_pause.label.set_color('white')
        
        def toggle_pause(event):
            self.paused = not self.paused
            self.btn_pause.label.set_text('Resume' if self.paused else 'Pause')
            
        self.btn_pause.on_clicked(toggle_pause)
        
        # Stop button - sets event that trainers check via should_stop()
        ax_stop = self.fig.add_axes([0.51, 0.02, 0.08, 0.04])
        self.btn_stop = Button(ax_stop, 'Stop', color='#442222', hovercolor='#663333')
        self.btn_stop.label.set_color('#ff6666')

        def request_stop(event):
            if self._training_stop_event is not None:
                self._training_stop_event.set()
                self.btn_stop.label.set_text('Stopping...')
                self.btn_stop.color = '#222222'
                self.btn_stop.hovercolor = '#222222'
                print("[Dashboard] Stop signal sent - training will stop at next update cycle")
            else:
                print("[Dashboard] No training stop event configured (standalone mode)")

        self.btn_stop.on_clicked(request_stop)
        
    def _update_frame(self, frame):
        """Update graphs (only if new data has arrived)."""
        if self.paused:
            return []
        
        # Skip all work if no new data (key optimization for training speed)
        if not self.dirty:
            return []
        
        self._process_updates()
        self.dirty = False  # Clear flag after processing
        
        artists_to_draw = []
        
        for mode, run in self.runs.items():
            artists = self.artists[mode]
            axes = self.axes[mode]
            
            # Common X-axis (Updates)
            n_updates = len(run.rewards_smoothed)
            if n_updates < 2:
                continue

            # X-axis data (pre-smoothed lists are already trimmed by append_metrics)
            x_data = list(range(n_updates))

            # Helper for Y-axis auto-scaling
            def autoscale_y(ax, data):
                if data:
                    mn, mx = min(data), max(data)
                    rng = mx - mn if mx != mn else 1.0
                    ax.set_ylim(mn - rng * 0.1, mx + rng * 0.1)

            # 1. Rewards (pre-smoothed)
            y_rew = run.rewards_smoothed
            artists['reward'].set_data(x_data, y_rew)
            axes['reward'].set_xlim(0, n_updates + 5)
            autoscale_y(axes['reward'], y_rew)

            # 2. Policy Loss (pre-smoothed)
            y_pol = run.policy_losses_smoothed
            artists['policy'].set_data(x_data, y_pol)
            axes['pol_loss'].set_xlim(0, n_updates + 5)
            autoscale_y(axes['pol_loss'], y_pol)

            # 3. Value Loss (pre-smoothed)
            y_val = run.value_losses_smoothed
            artists['value'].set_data(x_data, y_val)
            axes['val_loss'].set_xlim(0, n_updates + 5)
            autoscale_y(axes['val_loss'], y_val)

            # 4. Entropy (raw, no smoothing)
            artists['entropy'].set_data(x_data, run.entropies)
            axes['entropy'].set_xlim(0, n_updates + 5)

            # 5. Explained Variance (pre-smoothed)
            y_ev = run.explained_variances_smoothed
            artists['ev'].set_data(x_data, y_ev)
            axes['ev'].set_xlim(0, n_updates + 5)

            # 6. Behavior (pre-smoothed + food_ratio on secondary axis)
            y_food = run.food_smoothed
            y_pois = run.poison_smoothed
            y_ratio = run.food_ratio_smoothed
            artists['food'].set_data(x_data, y_food)
            artists['poison'].set_data(x_data, y_pois)
            artists['food_ratio'].set_data(x_data, y_ratio)
            axes['beh'].set_xlim(0, n_updates + 5)
            all_beh = y_food + y_pois
            if all_beh:
                mn, mx = min(all_beh), max(all_beh)
                rng = mx - mn if mx != mn else 1.0
                axes['beh'].set_ylim(max(0, mn - rng * 0.1), mx + rng * 0.1)

            # 7. Action Distribution
            probs = run.current_action_probs
            for bar, prob in zip(artists['act_bars'], probs):
                bar.set_height(prob)
                # Color code
                uniform = 1.0 / self.n_actions
                if prob > uniform * 1.5:
                    bar.set_color('#ff6b6b') # Red = High
                elif prob < uniform * 0.5:
                    bar.set_color('#4ecdc4') # Teal = Low
                else:
                    bar.set_color('#00d9ff') # Blue = Normal
            
            # Collect artists?
            # Matplotlib FuncAnimation with blit=False redraws everything, returning list helps if blit=True
            # We used blit=False so update is implicit
            
        return []

    def run(self):
        """Start the dashboard (blocking)."""
        self._create_figure()
        
        # Handle window close
        def on_close(event):
            self.running = False
        self.fig.canvas.mpl_connect('close_event', on_close)
        
        self.ani = FuncAnimation(
            self.fig,
            self._update_frame,
            interval=ANIMATION_INTERVAL_MS,
            blit=False,
            cache_frame_data=False
        )
        
        plt.show(block=True)

    def stop(self):
        self.running = False
        try:
            plt.close(self.fig)
        except Exception:
            pass 

def create_dashboard(modes: list[str], n_actions: int = 8) -> TrainingDashboard:
    return TrainingDashboard(modes, n_actions)


# =============================================================================
# MULTIPROCESSING DASHBOARD - Runs in separate process for zero overhead
# =============================================================================

import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue


def _dashboard_process_worker(modes: list[str], n_actions: int, update_queue: MPQueue,
                               stop_event, training_stop_event=None):
    """
    Worker function that runs in separate process.

    Completely isolated from training process - has its own GIL and GPU context.

    Args:
        modes: Training modes to display
        n_actions: Number of actions for action distribution plot
        update_queue: IPC queue for receiving updates from training
        stop_event: Event to signal dashboard shutdown
        training_stop_event: Event to signal training should stop (set by Stop button)
    """
    # Create dashboard in this process
    dashboard = TrainingDashboard(modes, n_actions, training_stop_event=training_stop_event)

    # Override the queue to use multiprocessing queue
    # We'll poll the mp queue and put items into the dashboard's internal queue

    def poll_queue():
        """Poll mp queue and transfer to dashboard queue."""
        while not stop_event.is_set():
            try:
                item = update_queue.get(timeout=QUEUE_POLL_TIMEOUT)
                if item is None:  # Poison pill
                    dashboard.running = False
                    break
                mode, update_type, data = item
                dashboard.update(mode, update_type, data)
            except queue.Empty:
                pass  # Queue empty or timeout

    # Start polling thread within dashboard process
    import threading
    poll_thread = threading.Thread(target=poll_queue, daemon=True)
    poll_thread.start()

    # Run dashboard (blocks until window closed)
    try:
        dashboard.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()


class DashboardProcess:
    """
    Dashboard that runs in a completely separate process.

    This eliminates GIL contention and GPU/display sync overhead by keeping
    matplotlib in its own isolated process. Training communicates via IPC queue.

    Usage:
        dashboard = DashboardProcess(modes=['ground_truth', 'proxy'])
        dashboard.start()

        # In training loop:
        dashboard.send_update('ground_truth', 'update', {'ppo': (...), ...})
        if dashboard.should_stop():
            break

        # When done:
        dashboard.stop()
    """

    def __init__(self, modes: list[str], n_actions: int = 8):
        self.modes = modes
        self.n_actions = n_actions

        # Use 'spawn' context to ensure clean process (no forked state)
        # This is critical for CUDA/ROCm to work correctly in child process
        ctx = mp.get_context('spawn')

        self._queue: MPQueue = ctx.Queue(maxsize=QUEUE_MAX_SIZE)
        self._stop_event = ctx.Event()  # Dashboard process shutdown
        self._training_stop_event = ctx.Event()  # User requested training stop
        self._process: Process = None
    
    def start(self):
        """Start the dashboard process."""
        ctx = mp.get_context('spawn')
        self._process = ctx.Process(
            target=_dashboard_process_worker,
            args=(self.modes, self.n_actions, self._queue,
                  self._stop_event, self._training_stop_event),
            daemon=True
        )
        self._process.start()
        print(f"[Dashboard] Started in separate process (PID {self._process.pid})")
    
    def send_update(self, mode: str, update_type: str, data: dict):
        """
        Send an update to the dashboard process.
        
        Non-blocking. If queue is full, update is dropped (dashboard is behind).
        
        Args:
            mode: Training mode (e.g., 'ground_truth')
            update_type: 'update' or 'finished'
            data: Payload dict (must be picklable - no torch tensors!)
        """
        try:
            self._queue.put_nowait((mode, update_type, data))
        except queue.Full:
            pass  # Queue full, drop update
    
    # Alias for compatibility with existing code
    def update(self, mode: str, update_type: str, data: dict):
        """Alias for send_update() - compatible with TrainingDashboard interface."""
        self.send_update(mode, update_type, data)
    
    def is_alive(self) -> bool:
        """Check if dashboard process is still running."""
        return self._process is not None and self._process.is_alive()

    def should_stop(self) -> bool:
        """Check if user requested training stop via dashboard button.

        This replaces the file-based stop signal with proper IPC.
        Trainers should call this periodically in their update loop.
        """
        return self._training_stop_event.is_set()

    def stop(self, timeout: float = 2.0):
        """Stop the dashboard process gracefully."""
        self._stop_event.set()
        self._queue.put(None)  # Poison pill
        
        if self._process is not None:
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                print("[Dashboard] Force terminating...")
                self._process.terminate()
                self._process.join(timeout=1.0)
    
    def wait(self):
        """Wait for dashboard to close (user closed window)."""
        if self._process is not None:
            self._process.join()


def create_dashboard_process(modes: list[str], n_actions: int = 8) -> DashboardProcess:
    """Factory function to create a process-isolated dashboard."""
    return DashboardProcess(modes, n_actions)


# =============================================================================
# LOG INGESTION - View historical training runs from CSV logs
# =============================================================================

def load_run_from_logs(log_prefix: str) -> RunState:
    """
    Load a training run from CSV log files.
    
    Args:
        log_prefix: Path prefix for log files (without _episodes.csv / _updates.csv suffix)
                   Example: 'logs/ground_truth_20251214_053108'
    
    Returns:
        RunState populated with historical data
    """
    import csv
    import os
    
    episodes_path = f"{log_prefix}_episodes.csv"
    updates_path = f"{log_prefix}_updates.csv"
    
    # Extract mode from prefix (e.g., 'ground_truth' from 'logs/ground_truth_20251214...')
    basename = os.path.basename(log_prefix)
    # Mode is everything before the timestamp (YYYYMMDD_HHMMSS pattern)
    import re
    match = re.match(r'(.+?)_\d{8}_\d{6}', basename)
    mode = match.group(1) if match else basename
    
    run = RunState(mode=mode)
    run.is_running = False
    run.is_finished = True
    
    # Load episodes (rewards)
    if os.path.exists(episodes_path):
        with open(episodes_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                run.rewards.append(float(row['reward']))
                run.episode = int(row['episode'])
        print(f"   Loaded {len(run.rewards)} episodes from {episodes_path}")
    else:
        print(f"   Warning: Episodes file not found: {episodes_path}")
    
    # Load updates (losses, entropy, explained_variance, food/poison metrics)
    if os.path.exists(updates_path):
        with open(updates_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                run.policy_losses.append(float(row['policy_loss']))
                run.value_losses.append(float(row['value_loss']))
                run.entropies.append(float(row['entropy']))
                run.total_steps = int(row['total_steps'])
                # EV may not exist in old logs
                if 'explained_variance' in row and row['explained_variance']:
                    run.explained_variances.append(float(row['explained_variance']))
                # Food/poison metrics (may not exist in old logs)
                if 'food_mean' in row and row['food_mean']:
                    food = float(row['food_mean'])
                    poison = float(row['poison_mean']) if row.get('poison_mean') else 0.0
                    run.food_history.append(food)
                    run.poison_history.append(poison)
                    # food_ratio may be in CSV or computed
                    if 'food_ratio' in row and row['food_ratio']:
                        run.food_ratio_history.append(float(row['food_ratio']))
                    else:
                        total = food + poison
                        run.food_ratio_history.append(food / total if total > 0 else 0.5)
        ev_status = f" ({len(run.explained_variances)} EV)" if run.explained_variances else ""
        beh_status = f" ({len(run.food_history)} beh)" if run.food_history else ""
        print(f"   Loaded {len(run.policy_losses)} updates from {updates_path}{ev_status}{beh_status}")

        # Populate smoothed lists for visualization (use raw values from historical data)
        run.policy_losses_smoothed = list(run.policy_losses)
        run.value_losses_smoothed = list(run.value_losses)
        run.explained_variances_smoothed = list(run.explained_variances)
        run.food_smoothed = list(run.food_history)
        run.poison_smoothed = list(run.poison_history)
        run.food_ratio_smoothed = list(run.food_ratio_history)
    else:
        print(f"   Warning: Updates file not found: {updates_path}")

    return run


def find_latest_logs(log_dir: str = 'generated/logs', mode: str = None) -> list[str]:
    """
    Find the latest log prefixes in a directory.
    
    Args:
        log_dir: Directory containing log files
        mode: Optional mode filter (e.g., 'ground_truth'). If None, finds all modes.
    
    Returns:
        List of log prefixes (one per mode, most recent)
    """
    import os
    import re
    from collections import defaultdict
    
    if not os.path.isdir(log_dir):
        print(f"Log directory not found: {log_dir}")
        return []
    
    # Find all episode files and extract mode + timestamp
    pattern = re.compile(r'(.+?)_(\d{8}_\d{6})_episodes\.csv')
    mode_timestamps = defaultdict(list)
    
    for filename in os.listdir(log_dir):
        match = pattern.match(filename)
        if match:
            m, ts = match.groups()
            if mode is None or m == mode:
                mode_timestamps[m].append(ts)
    
    # Get latest timestamp per mode
    prefixes = []
    for m, timestamps in mode_timestamps.items():
        latest_ts = sorted(timestamps)[-1]
        prefixes.append(os.path.join(log_dir, f"{m}_{latest_ts}"))
    
    return sorted(prefixes)


def view_logs(log_prefixes: list[str], smoothing: float = 0.9):
    """
    Display a static visualization of historical training runs.
    
    Args:
        log_prefixes: List of log file prefixes to visualize
        smoothing: EMA smoothing factor (0.0 = no smoothing, 1.0 = maximum)
    """
    if not log_prefixes:
        print("No log files to display.")
        return
    
    print(f"\nLoading {len(log_prefixes)} training run(s)...")
    runs = [load_run_from_logs(prefix) for prefix in log_prefixes]
    
    # Filter out empty runs
    runs = [r for r in runs if r.rewards or r.policy_losses]
    if not runs:
        print("No valid data found in log files.")
        return
    
    # Create static figure
    plt.style.use('dark_background')
    
    n_runs = len(runs)
    fig, axes = plt.subplots(n_runs, 4, figsize=(16, 3 * n_runs + 1), squeeze=False)
    fig.suptitle("Training Log Viewer", fontsize=14, fontweight='bold', color='white')
    
    def smooth(data, alpha):
        if not data or alpha <= 0:
            return data
        result = []
        last = data[0]
        for val in data:
            last = last * alpha + val * (1 - alpha)
            result.append(last)
        return result
    
    for i, run in enumerate(runs):
        # Rewards
        ax = axes[i, 0]
        ax.set_title(f"{run.mode}: Rewards ({len(run.rewards)} episodes)", 
                     fontsize=10, fontweight='bold', color='yellow')
        if run.rewards:
            y = smooth(run.rewards, smoothing)
            ax.plot(y, 'c-', linewidth=1, alpha=0.9)
            ax.set_xlim(0, len(y))
        ax.grid(True, alpha=0.15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Losses
        ax = axes[i, 1]
        ax.set_title(f"Losses ({len(run.policy_losses)} updates)", fontsize=10, color='silver')
        if run.policy_losses:
            x = list(range(len(run.policy_losses)))
            ax.plot(x, smooth(run.policy_losses, smoothing), 'r-', linewidth=1, label='Policy', alpha=0.8)
            ax.plot(x, smooth(run.value_losses, smoothing), 'g-', linewidth=1, label='Value', alpha=0.8)
            ax.legend(fontsize=7, loc='upper right', framealpha=0.3)
            ax.set_xlim(0, len(x))
        ax.grid(True, alpha=0.15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Entropy
        ax = axes[i, 2]
        ax.set_title("Entropy", fontsize=10, color='silver')
        if run.entropies:
            ax.plot(run.entropies, 'm-', linewidth=1, alpha=0.9)
            ax.axhline(y=np.log(8), color='gray', linestyle='--', alpha=0.3, label='Max (8 actions)')
            ax.set_xlim(0, len(run.entropies))
            ax.set_ylim(0, np.log(8) * 1.1)
        ax.grid(True, alpha=0.15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Explained Variance (may be empty for old logs)
        ax = axes[i, 3]
        ax.set_title("Expl. Var.", fontsize=10, color='silver')
        ax.axhline(y=1.0, color='lime', linestyle='--', alpha=0.3)
        ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylim(-0.5, 1.1)
        if run.explained_variances:
            ax.plot(smooth(run.explained_variances, smoothing), 'y-', linewidth=1, alpha=0.9)
            ax.set_xlim(0, len(run.explained_variances))
        ax.grid(True, alpha=0.15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Training Dashboard - View training logs')
    parser.add_argument('logs', nargs='*', help='Log file prefixes to view (without _episodes.csv suffix)')
    parser.add_argument('--dir', '-d', default='generated/logs', help='Log directory to search (default: generated/logs)')
    parser.add_argument('--mode', '-m', help='Filter by mode (e.g., ground_truth)')
    parser.add_argument('--smooth', '-s', type=float, default=0.9, help='Smoothing factor 0.0-1.0 (default: 0.9)')
    parser.add_argument('--latest', '-l', action='store_true', help='View latest run(s) automatically')
    
    args = parser.parse_args()
    
    if args.logs:
        # User specified explicit log prefixes
        view_logs(args.logs, smoothing=args.smooth)
    elif args.latest or not args.logs:
        # Auto-discover latest logs
        prefixes = find_latest_logs(args.dir, mode=args.mode)
        if prefixes:
            print(f"Found {len(prefixes)} run(s):")
            for p in prefixes:
                print(f"  - {p}")
            view_logs(prefixes, smoothing=args.smooth)
        else:
            print(f"No log files found in {args.dir}")
            if args.mode:
                print(f"  (filtered by mode: {args.mode})")
