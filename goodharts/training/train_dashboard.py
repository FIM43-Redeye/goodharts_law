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
from collections import deque
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
    
    # Metrics history (keep more history for graphs)
    rewards: deque = field(default_factory=lambda: deque(maxlen=1000))
    policy_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    value_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    entropies: deque = field(default_factory=lambda: deque(maxlen=1000))
    
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
                    # data: (policy_loss, value_loss, entropy, action_probs)
                    run.policy_losses.append(data[0])
                    run.value_losses.append(data[1])
                    run.entropies.append(data[2])
                
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
        cols = 3 # Reward, Loss, Entropy
        
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
