"""
Unified Training Dashboard (Graph Only).

Single matplotlib window showing:
- Reward History
- Losses (Policy, Value)
- Entropy

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


@dataclass
class RunState:
    """State for a single training run."""
    mode: str
    n_actions: int = 8
    
    # Episode data
    episode: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0
    food_eaten: int = 0
    food_density: int = 0
    
    # Metrics history (unlimited for full run visualization)
    rewards: list = field(default_factory=list)
    policy_losses: list = field(default_factory=list)
    value_losses: list = field(default_factory=list)
    entropies: list = field(default_factory=list)
    explained_variances: list = field(default_factory=list)  # Value function quality
    
    # Status
    is_running: bool = True
    is_finished: bool = False
    total_steps: int = 0
    best_efficiency: float = float('-inf')
    
    # Timestamps
    last_update: float = 0.0


class TrainingDashboard:
    """
    Unified visualization dashboard for training runs.
    """
    
    def __init__(self, modes: list[str], n_actions: int = 8):
        self.modes = modes
        self.n_actions = n_actions
        self.n_runs = len(modes)
        
        # State per run
        self.runs: dict[str, RunState] = {
            mode: RunState(mode=mode, n_actions=n_actions)
            for mode in modes
        }
        
        # Thread-safe update queue
        self.update_queue: queue.Queue = queue.Queue()
        
        # Speed control
        self.speed_mode = 'normal'
        self.paused = False
        
        # Matplotlib state
        self.fig = None
        self.axes: dict[str, dict[str, Any]] = {}  # mode -> {axis_name: axis}
        self.line_artists: dict[str, dict[str, Any]] = {}  # mode -> {name: Line2D}
        
        self.running = True
    
    def update(self, mode: str, update_type: str, data: Any):
        """Queue an update from a training thread."""
        try:
            self.update_queue.put_nowait((mode, update_type, data))
        except queue.Full:
            pass
    
    def _process_updates(self):
        """Process all pending updates."""
        updates_processed = 0
        max_updates = 200
        
        while not self.update_queue.empty() and updates_processed < max_updates:
            try:
                mode, update_type, data = self.update_queue.get_nowait()
                run = self.runs.get(mode)
                if run is None:
                    continue
                
                run.last_update = time.time()
                
                if update_type == 'step':
                    # data: (agent_view, action_probs, total_steps) -> IGNORE view/probs
                    run.total_steps = data[2]
                
                elif update_type == 'episode':
                    # data: (episode, reward, length, food, density)
                    run.episode = data[0]
                    run.episode_reward = data[1]
                    run.episode_length = data[2]
                    run.rewards.append(data[1])
                
                elif update_type == 'ppo':
                    # data: (policy_loss, value_loss, entropy, action_probs) or
                    #       (policy_loss, value_loss, entropy, action_probs, explained_var)
                    run.policy_losses.append(data[0])
                    run.value_losses.append(data[1])
                    run.entropies.append(data[2])
                    # Handle EV if present (new logs have it, old logs don't)
                    if len(data) > 4:
                        run.explained_variances.append(data[4])
                
                elif update_type == 'finished':
                    run.is_finished = True
                    run.is_running = False
                
                updates_processed += 1
                
            except queue.Empty:
                break
    
    def _smooth(self, data: list[float], alpha: float = 0.9) -> list[float]:
        """Apply exponential moving average smoothing."""
        if not data:
            return []
        smoothed = []
        last = data[0]
        for val in data:
            # Simple EMA
            last = last * alpha + val * (1 - alpha)
            smoothed.append(last)
        return smoothed

    def _create_figure(self):
        """Create the matplotlib figure with concise layout."""
        # Clean styling
        plt.style.use('dark_background')
        
        # Layout: 1 row per run
        rows = self.n_runs
        cols = 4  # Reward, Loss, Entropy, Explained Variance
        
        # Adjust height based on rows
        fig_height = 3 * rows + 1
        self.fig = plt.figure(figsize=(14, fig_height))
        self.fig.suptitle("🧠 Training Dashboard", fontsize=14, fontweight='bold', color='white')
        
        # Grid
        gs = gridspec.GridSpec(rows, cols, figure=self.fig, 
                               hspace=0.4, wspace=0.25, 
                               left=0.05, right=0.95, top=0.9, bottom=0.1)
        
        for i, mode in enumerate(self.modes):
            self._create_run_row(gs, i, mode)
        
        self._create_controls()
    
    def _create_run_row(self, gs, row_idx: int, mode: str):
        """Create a row of graphs for one run."""
        self.axes[mode] = {}
        self.line_artists[mode] = {}
        
        # 1. Reward
        ax_rew = self.fig.add_subplot(gs[row_idx, 0])
        ax_rew.set_title(f"{mode}: Rewards", fontsize=10, fontweight='bold', color='yellow')
        ax_rew.grid(True, alpha=0.15)
        # Force integer x-axis ticks
        ax_rew.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        line_rew, = ax_rew.plot([], [], 'c-', linewidth=1.5, alpha=0.9)
        self.axes[mode]['reward'] = ax_rew
        self.line_artists[mode]['reward'] = line_rew
        
        # 2. Losses
        ax_loss = self.fig.add_subplot(gs[row_idx, 1])
        ax_loss.set_title("Losses", fontsize=10, color='silver')
        ax_loss.grid(True, alpha=0.15)
        ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        l_pol, = ax_loss.plot([], [], 'r-', linewidth=1, label='Policy', alpha=0.8)
        l_val, = ax_loss.plot([], [], 'g-', linewidth=1, label='Value', alpha=0.8)
        ax_loss.legend(fontsize=7, loc='upper right', framealpha=0.3)
        
        self.axes[mode]['loss'] = ax_loss
        self.line_artists[mode]['policy'] = l_pol
        self.line_artists[mode]['value'] = l_val
        
        # 3. Entropy
        ax_ent = self.fig.add_subplot(gs[row_idx, 2])
        ax_ent.set_title("Entropy", fontsize=10, color='silver')
        ax_ent.grid(True, alpha=0.15)
        ax_ent.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Target entropy reference
        max_ent = np.log(self.n_actions)
        ax_ent.axhline(y=max_ent, color='gray', linestyle='--', alpha=0.3)
        ax_ent.set_ylim(0, max_ent * 1.1)
        
        l_ent, = ax_ent.plot([], [], 'm-', linewidth=1.5, alpha=0.9)
        
        self.axes[mode]['entropy'] = ax_ent
        self.line_artists[mode]['entropy'] = l_ent
        
        # 4. Explained Variance (Value function quality)
        ax_ev = self.fig.add_subplot(gs[row_idx, 3])
        ax_ev.set_title("Expl. Var.", fontsize=10, color='silver')
        ax_ev.grid(True, alpha=0.15)
        ax_ev.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Reference line at 1.0 (perfect value prediction)
        ax_ev.axhline(y=1.0, color='lime', linestyle='--', alpha=0.3, label='Perfect')
        ax_ev.axhline(y=0.0, color='gray', linestyle=':', alpha=0.3)
        ax_ev.set_ylim(-0.5, 1.1)  # EV can be negative if predictions are worse than mean
        
        l_ev, = ax_ev.plot([], [], 'y-', linewidth=1.5, alpha=0.9)
        
        self.axes[mode]['ev'] = ax_ev
        self.line_artists[mode]['ev'] = l_ev

    def _create_controls(self):
        """Create simple control buttons."""
        # Place buttons at very bottom
        ax_pause = self.fig.add_axes([0.35, 0.02, 0.1, 0.04])
        self.btn_pause = Button(ax_pause, '⏸ Pause', color='#333333', hovercolor='#555555')
        self.btn_pause.label.set_color('white')
        
        ax_finish = self.fig.add_axes([0.55, 0.02, 0.1, 0.04])
        self.btn_finish = Button(ax_finish, '🏁 Finish', color='#333333', hovercolor='#555555')
        self.btn_finish.label.set_color('white')
        
        def toggle_pause(event):
            self.paused = not self.paused
            self.btn_pause.label.set_text('▶ Resume' if self.paused else '⏸ Pause')
            
        def finish_training(event):
            self.finish_requested = True
            self.btn_finish.label.set_text('Stopping...')
            # Queue finish for all runs? Or just signal globally.
            # We'll set a flag that the trainer checks.
            # Note: Trainer runs in separate process. Dashboard update queue is one-way.
            # Actually, typically trainer pushes to dashboard. Dashboard can't easily push back
            # without a shared flag or file.
            # User is running single process with threads? "python -m ... --mode all"
            # It likely uses multiprocessing.
            # If multiprocessing, we can't easily stop them from here unless they listen for a file or shared value.
            # Simplest hack: Create a 'training.stop' file.
            import os
            with open('.training_stop_signal', 'w') as f:
                f.write('stop')
            print("\n🛑 Stop signal sent (file created). Waiting for trainers...")
            
        self.btn_pause.on_clicked(toggle_pause)
        self.btn_finish.on_clicked(finish_training)
        
    def _update_frame(self, frame):
        """Update graphs."""
        if self.paused:
            return []
            
        self._process_updates()
        
        for mode, run in self.runs.items():
            artists = self.line_artists[mode]
            axes = self.axes[mode]
            
            # --- Update Data ---
            
            # REWARDS
            if len(run.rewards) > 1:
                # X-axis is simply local index for now, could be episodes
                raw_y = list(run.rewards)
                smooth_y = self._smooth(raw_y, alpha=0.9)
                
                x_data = list(range(len(raw_y)))
                artists['reward'].set_data(x_data, smooth_y)
                
                axes['reward'].set_xlim(0, len(raw_y) + 5)
                # Dynamic Y-limits with padding (based on smoothed data for stability)
                min_y, max_y = min(smooth_y), max(smooth_y)
                rng = max_y - min_y if max_y != min_y else 1.0
                axes['reward'].set_ylim(min_y - rng*0.1, max_y + rng*0.1)

            # LOSSES
            if len(run.policy_losses) > 1:
                x_data = list(range(len(run.policy_losses)))
                
                p_loss = self._smooth(list(run.policy_losses), alpha=0.9)
                v_loss = self._smooth(list(run.value_losses), alpha=0.9)
                
                artists['policy'].set_data(x_data, p_loss)
                artists['value'].set_data(x_data, v_loss)
                
                axes['loss'].set_xlim(0, len(x_data) + 5)
                
                # Combine limits
                all_vals = p_loss + v_loss
                min_y, max_y = min(all_vals), max(all_vals)
                rng = max_y - min_y if max_y != min_y else 1.0
                axes['loss'].set_ylim(min_y - rng*0.1, max_y + rng*0.1)

            # ENTROPY
            if len(run.entropies) > 1:
                y_data = list(run.entropies) # Entropy usually doesn't need much smoothing, but consistency is good
                # smooth_y = self._smooth(y_data, alpha=0.9) 
                # Keep entropy raw as it's less noisy and good for debugging collapse
                
                x_data = list(range(len(y_data)))
                artists['entropy'].set_data(x_data, y_data)
                
                axes['entropy'].set_xlim(0, len(x_data) + 5)
                # Entropy Y is fixed 0..log(N) usually, but let's autoscale bottom slightly
                axes['entropy'].set_ylim(0, np.log(self.n_actions)*1.1)
            
            # EXPLAINED VARIANCE (blank if no data from old logs)
            if len(run.explained_variances) > 1:
                y_data = self._smooth(list(run.explained_variances), alpha=0.9)
                x_data = list(range(len(y_data)))
                artists['ev'].set_data(x_data, y_data)
                axes['ev'].set_xlim(0, len(x_data) + 5)
                # Keep fixed Y limits for consistency
                
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
            interval=100, # 10fps is plenty for graphs
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
    
    # Load updates (losses, entropy, explained_variance)
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
        ev_status = f" ({len(run.explained_variances)} EV)" if run.explained_variances else ""
        print(f"   Loaded {len(run.policy_losses)} updates from {updates_path}{ev_status}")
    else:
        print(f"   Warning: Updates file not found: {updates_path}")
    
    return run


def find_latest_logs(log_dir: str = 'logs', mode: str = None) -> list[str]:
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
    parser.add_argument('--dir', '-d', default='logs', help='Log directory to search (default: logs)')
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
