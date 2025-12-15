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


@dataclass
class RunState:
    """State for a single training run."""
    mode: str
    n_actions: int = 8
    
    # Current snapshot
    episode: int = 0
    total_steps: int = 0
    current_action_probs: np.ndarray = field(default_factory=lambda: np.zeros(8))
    
    # Metrics history (aligned by Update)
    rewards: list = field(default_factory=list)
    policy_losses: list = field(default_factory=list)
    value_losses: list = field(default_factory=list)
    entropies: list = field(default_factory=list)
    explained_variances: list = field(default_factory=list)
    food_history: list = field(default_factory=list)
    poison_history: list = field(default_factory=list)
    
    # Status
    is_running: bool = True
    is_finished: bool = False
    
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
        
        # UI State
        self.paused = False
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
        except queue.Full:
            pass
    
    def _process_updates(self):
        """Process all pending updates."""
        updates_processed = 0
        max_updates = 500  # Process more updates per frame to catch up
        
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
                        # 1. PPO Stats (Always present)
                        p_loss, v_loss, ent, probs, ev = ppo
                        run.policy_losses.append(p_loss)
                        run.value_losses.append(v_loss)
                        run.entropies.append(ent)
                        run.explained_variances.append(ev)
                        run.current_action_probs = np.array(probs)
                        
                        # 2. Episode Stats (Forward fill if missing)
                        # If no episodes finished this update, repeat the last known value
                        # to maintain graph continuity with the "Update" axis.
                        if episodes:
                            run.rewards.append(episodes['reward'])
                            run.food_history.append(episodes['food'])
                            run.poison_history.append(episodes['poison'])
                        else:
                            # Forward fill
                            last_reward = run.rewards[-1] if run.rewards else 0.0
                            last_food = run.food_history[-1] if run.food_history else 0.0
                            last_poison = run.poison_history[-1] if run.poison_history else 0.0
                            
                            run.rewards.append(last_reward)
                            run.food_history.append(last_food)
                            run.poison_history.append(last_poison)
                            
                    run.total_steps = steps
                
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
        cols = 6  # Reward, Loss, Entropy, Expl Var, Behavior, Action Dist
        
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
        
        # 2. Losses
        ax = self.fig.add_subplot(gs[row_idx, 1])
        ax.set_title("Losses", fontsize=10, color='silver')
        ax.grid(True, alpha=0.15)
        
        l_pol, = ax.plot([], [], 'r-', linewidth=1, label='Pol', alpha=0.8)
        l_val, = ax.plot([], [], 'g-', linewidth=1, label='Val', alpha=0.8)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.3)
        
        self.axes[mode]['loss'] = ax
        self.artists[mode]['policy'] = l_pol
        self.artists[mode]['value'] = l_val
        
        # 3. Entropy
        ax = self.fig.add_subplot(gs[row_idx, 2])
        ax.set_title("Entropy", fontsize=10, color='silver')
        ax.grid(True, alpha=0.15)
        
        max_ent = np.log(self.n_actions)
        ax.axhline(y=max_ent, color='gray', linestyle='--', alpha=0.3)
        ax.set_ylim(0, max_ent * 1.1)
        
        l_ent, = ax.plot([], [], 'm-', linewidth=1.5, alpha=0.9)
        self.axes[mode]['entropy'] = ax
        self.artists[mode]['entropy'] = l_ent
        
        # 4. Explained Variance
        ax = self.fig.add_subplot(gs[row_idx, 3])
        ax.set_title("Expl. Var.", fontsize=10, color='silver')
        ax.grid(True, alpha=0.15)
        ax.axhline(y=1.0, color='lime', linestyle='--', alpha=0.3)
        ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylim(-0.5, 1.1)
        
        l_ev, = ax.plot([], [], 'y-', linewidth=1.5, alpha=0.9)
        self.axes[mode]['ev'] = ax
        self.artists[mode]['ev'] = l_ev
        
        # 5. Behavior
        ax = self.fig.add_subplot(gs[row_idx, 4])
        ax.set_title("Behavior", fontsize=10, color='silver')
        ax.grid(True, alpha=0.15)
        
        l_food, = ax.plot([], [], 'g-', linewidth=1.5, alpha=0.9, label='Food')
        l_poison, = ax.plot([], [], 'r-', linewidth=1.5, alpha=0.9, label='Pois')
        ax.legend(fontsize=7, loc='upper right', framealpha=0.3)
        
        self.axes[mode]['beh'] = ax
        self.artists[mode]['food'] = l_food
        self.artists[mode]['poison'] = l_poison
        
        # 6. Action Distribution (Bar Chart)
        ax = self.fig.add_subplot(gs[row_idx, 5])
        ax.set_title("Actions", fontsize=10, color='silver')
        ax.grid(False)
        ax.set_ylim(0, 0.6) # reasonable max prob
        ax.axhline(y=1.0/self.n_actions, color='gray', linestyle='--', alpha=0.3)
        
        action_labels = ['↑', '↓', '←', '→', '↖', '↗', '↙', '↘'][:self.n_actions]
        x = np.arange(self.n_actions)
        bars = ax.bar(x, np.zeros(self.n_actions), color='#00d9ff', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(action_labels)
        
        self.axes[mode]['act'] = ax
        self.artists[mode]['act_bars'] = bars

    def _create_controls(self):
        """Create simple control buttons."""
        ax_pause = self.fig.add_axes([0.45, 0.02, 0.1, 0.04])
        self.btn_pause = Button(ax_pause, 'Pause', color='#333333', hovercolor='#555555')
        self.btn_pause.label.set_color('white')
        
        def toggle_pause(event):
            self.paused = not self.paused
            self.btn_pause.label.set_text('Resume' if self.paused else 'Pause')
            
        self.btn_pause.on_clicked(toggle_pause)
        
    def _update_frame(self, frame):
        """Update graphs."""
        if self.paused:
            return []
            
        self._process_updates()
        
        artists_to_draw = []
        
        for mode, run in self.runs.items():
            artists = self.artists[mode]
            axes = self.axes[mode]
            
            # Common X-axis (Updates)
            n_updates = len(run.rewards)
            if n_updates < 2:
                continue
                
            x_data = list(range(n_updates))
            
            # 1. Rewards
            y_rew = self._smooth(run.rewards, 0.9)
            artists['reward'].set_data(x_data, y_rew)
            axes['reward'].set_xlim(0, n_updates + 5)
            # Auto-scale Y with padding
            if y_rew:
                mn, mx = min(y_rew), max(y_rew)
                rng = mx - mn if mx != mn else 1.0
                axes['reward'].set_ylim(mn - rng*0.1, mx + rng*0.1)
                
            # 2. Losses
            y_pol = self._smooth(run.policy_losses, 0.9)
            y_val = self._smooth(run.value_losses, 0.9)
            artists['policy'].set_data(x_data, y_pol)
            artists['value'].set_data(x_data, y_val)
            axes['loss'].set_xlim(0, n_updates + 5)
            # Combine for scale
            all_loss = y_pol + y_val
            if all_loss:
                mn, mx = min(all_loss), max(all_loss)
                rng = mx - mn if mx != mn else 1.0
                axes['loss'].set_ylim(mn - rng*0.1, mx + rng*0.1)
                
            # 3. Entropy
            artists['entropy'].set_data(x_data, run.entropies)
            axes['entropy'].set_xlim(0, n_updates + 5)
            
            # 4. Explained Variance
            y_ev = self._smooth(run.explained_variances, 0.9)
            artists['ev'].set_data(x_data, y_ev)
            axes['ev'].set_xlim(0, n_updates + 5)
            
            # 5. Behavior
            y_food = self._smooth(run.food_history, 0.9)
            y_pois = self._smooth(run.poison_history, 0.9)
            artists['food'].set_data(x_data, y_food)
            artists['poison'].set_data(x_data, y_pois)
            axes['beh'].set_xlim(0, n_updates + 5)
            # Scale
            all_beh = y_food + y_pois
            if all_beh:
                mn, mx = min(all_beh), max(all_beh)
                rng = mx - mn if mx != mn else 1.0
                axes['beh'].set_ylim(max(0, mn - rng*0.1), mx + rng*0.1)
                
            # 6. Action Distribution
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
            interval=200, 
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
