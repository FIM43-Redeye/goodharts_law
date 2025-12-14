"""
Unified Training Dashboard.

Single matplotlib window showing:
- Agent's view (world around agent)
- CNN input channels (what the network sees)
- Action probabilities
- Training graphs (reward, loss, entropy)

Supports multiple concurrent training runs in a tiled layout.
Main thread runs matplotlib, training runs in background threads.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Any
import threading
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
    
    # Current step data
    agent_view: np.ndarray | None = None  # Shape: (channels, H, W)
    action_probs: np.ndarray = field(default_factory=lambda: np.ones(8) / 8)
    
    # Metrics history
    rewards: deque = field(default_factory=lambda: deque(maxlen=200))
    policy_losses: deque = field(default_factory=lambda: deque(maxlen=200))
    value_losses: deque = field(default_factory=lambda: deque(maxlen=200))
    entropies: deque = field(default_factory=lambda: deque(maxlen=200))
    
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
    
    Runs matplotlib on main thread with training in background.
    Supports multiple concurrent runs in a tiled layout.
    """
    
    def __init__(self, modes: list[str], n_actions: int = 8):
        """
        Initialize dashboard.
        
        Args:
            modes: List of training mode names (determines grid layout)
            n_actions: Number of possible actions
        """
        self.modes = modes
        self.n_actions = n_actions
        self.n_runs = len(modes)
        
        # State per run
        self.runs: dict[str, RunState] = {
            mode: RunState(mode=mode, n_actions=n_actions)
            for mode in modes
        }
        
        # Thread-safe update queue: (mode, update_type, data)
        self.update_queue: queue.Queue = queue.Queue()
        
        # Speed control
        self.speed_mode = 'normal'  # 'fast', 'normal', 'pause'
        self.paused = False
        
        # Matplotlib state
        self.fig = None
        self.axes: dict[str, dict[str, Any]] = {}  # mode -> {axis_name: axis}
        self.run_panels: dict[str, dict[str, Any]] = {}  # mode -> {name: artist}
        
        # Control
        self.running = True
    
    def update(self, mode: str, update_type: str, data: Any):
        """
        Queue an update from a training thread (thread-safe).
        
        Args:
            mode: Training mode name
            update_type: 'step', 'episode', 'ppo', 'finished'
            data: Update data (varies by type)
        """
        try:
            self.update_queue.put_nowait((mode, update_type, data))
        except queue.Full:
            pass  # Drop updates if queue is full
    
    def _process_updates(self):
        """Process all pending updates from training threads."""
        updates_processed = 0
        max_updates = 100  # Limit to avoid blocking
        
        while not self.update_queue.empty() and updates_processed < max_updates:
            try:
                mode, update_type, data = self.update_queue.get_nowait()
                run = self.runs.get(mode)
                if run is None:
                    continue
                
                run.last_update = time.time()
                
                if update_type == 'step':
                    # data: (agent_view, action_probs, total_steps)
                    run.agent_view = data[0]
                    run.action_probs = data[1]
                    run.total_steps = data[2]
                
                elif update_type == 'episode':
                    # data: (episode, reward, length, food, density)
                    run.episode = data[0]
                    run.episode_reward = data[1]
                    run.episode_length = data[2]
                    run.food_eaten = data[3]
                    run.food_density = data[4]
                    run.rewards.append(data[1])
                
                elif update_type == 'ppo':
                    # data: (policy_loss, value_loss, entropy, action_probs)
                    run.policy_losses.append(data[0])
                    run.value_losses.append(data[1])
                    run.entropies.append(data[2])
                    run.action_probs = data[3]
                
                elif update_type == 'finished':
                    run.is_finished = True
                    run.is_running = False
                
                updates_processed += 1
                
            except queue.Empty:
                break
    
    def _create_figure(self):
        """Create the matplotlib figure with grid layout."""
        # Determine grid size based on number of runs
        if self.n_runs == 1:
            rows, cols = 1, 1
        elif self.n_runs == 2:
            rows, cols = 1, 2
        elif self.n_runs <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
        
        # Create figure
        fig_width = 6 * cols
        fig_height = 8 * rows
        self.fig = plt.figure(figsize=(fig_width, min(fig_height, 12)))
        self.fig.suptitle("🧠 Training Dashboard", fontsize=14, fontweight='bold')
        
        # Create outer grid for runs
        outer_gs = gridspec.GridSpec(rows, cols, figure=self.fig, hspace=0.3, wspace=0.2)
        
        # Create each run's panel
        for i, mode in enumerate(self.modes):
            row, col = i // cols, i % cols
            self._create_run_panel(outer_gs[row, col], mode)
        
        # Add speed control buttons at bottom
        self._create_controls()
    
    def _create_run_panel(self, outer_spec, mode: str):
        """Create visualization panel for a single run."""
        # Inner grid: 3 rows
        # Row 0: Agent View | Action Probs
        # Row 1: Reward graph
        # Row 2: Loss/Entropy graphs
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            3, 2, subplot_spec=outer_spec, height_ratios=[1.5, 1, 1],
            hspace=0.3, wspace=0.2
        )
        
        self.axes[mode] = {}
        self.run_panels[mode] = {}
        
        # Agent view (left top)
        ax = self.fig.add_subplot(inner_gs[0, 0])
        ax.set_title(f"{mode}", fontsize=10, fontweight='bold')
        ax.axis('off')
        self.axes[mode]['view'] = ax
        
        # Placeholder image for agent view
        dummy = np.zeros((11, 11, 3))
        img = ax.imshow(dummy, interpolation='nearest')
        self.run_panels[mode]['view_img'] = img
        
        # Action probabilities (right top)
        ax = self.fig.add_subplot(inner_gs[0, 1])
        ax.set_title("Actions", fontsize=9)
        ax.set_ylim(0, 0.5)
        labels = ['↑', '↓', '←', '→', '↖', '↗', '↙', '↘'][:self.n_actions]
        x = np.arange(self.n_actions)
        bars = ax.bar(x, np.ones(self.n_actions) / self.n_actions, color='#00d9ff')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(y=1/self.n_actions, color='red', linestyle='--', alpha=0.5)
        self.axes[mode]['actions'] = ax
        self.run_panels[mode]['action_bars'] = bars
        
        # Reward graph (middle row, spans both columns)
        ax = self.fig.add_subplot(inner_gs[1, :])
        ax.set_title("Episode Reward", fontsize=9)
        ax.grid(True, alpha=0.3)
        line, = ax.plot([], [], 'b-', linewidth=1)
        self.axes[mode]['reward'] = ax
        self.run_panels[mode]['reward_line'] = line
        
        # Loss graph (bottom left)
        ax = self.fig.add_subplot(inner_gs[2, 0])
        ax.set_title("Losses", fontsize=9)
        ax.grid(True, alpha=0.3)
        pol_line, = ax.plot([], [], 'r-', linewidth=1, label='policy')
        val_line, = ax.plot([], [], 'm-', linewidth=1, label='value')
        ax.legend(fontsize=7, loc='upper right')
        self.axes[mode]['loss'] = ax
        self.run_panels[mode]['policy_line'] = pol_line
        self.run_panels[mode]['value_line'] = val_line
        
        # Entropy graph (bottom right)
        ax = self.fig.add_subplot(inner_gs[2, 1])
        ax.set_title("Entropy", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=np.log(self.n_actions), color='red', linestyle='--', alpha=0.5)
        line, = ax.plot([], [], 'c-', linewidth=1)
        self.axes[mode]['entropy'] = ax
        self.run_panels[mode]['entropy_line'] = line
    
    def _create_controls(self):
        """Create speed control buttons."""
        # Button axes at the bottom
        ax_fast = self.fig.add_axes([0.2, 0.01, 0.15, 0.03])
        ax_normal = self.fig.add_axes([0.4, 0.01, 0.15, 0.03])
        ax_pause = self.fig.add_axes([0.6, 0.01, 0.15, 0.03])
        
        self.btn_fast = Button(ax_fast, '⏩ Fast')
        self.btn_normal = Button(ax_normal, '▶ Normal')
        self.btn_pause = Button(ax_pause, '⏸ Pause')
        
        self.btn_fast.on_clicked(lambda e: self._set_speed('fast'))
        self.btn_normal.on_clicked(lambda e: self._set_speed('normal'))
        self.btn_pause.on_clicked(lambda e: self._set_speed('pause'))
    
    def _set_speed(self, mode: str):
        """Set speed mode."""
        self.speed_mode = mode
        self.paused = (mode == 'pause')
        print(f"Dashboard: Speed set to {mode}")
    
    def _view_to_rgb(self, view: np.ndarray) -> np.ndarray:
        """Convert multi-channel view to RGB image for display."""
        if view is None:
            return np.zeros((11, 11, 3))
        
        # view shape: (channels, H, W)
        h, w = view.shape[1], view.shape[2]
        rgb = np.zeros((h, w, 3))
        
        # Color mapping by channel (assuming order: EMPTY, WALL, FOOD, POISON, ...)
        colors = [
            (0.1, 0.1, 0.2),  # EMPTY - dark blue
            (0.5, 0.5, 0.5),  # WALL - gray
            (0.2, 0.8, 0.5),  # FOOD - green
            (0.9, 0.3, 0.3),  # POISON - red
            (0.0, 0.8, 0.8),  # PREY - cyan
            (1.0, 0.2, 0.2),  # PREDATOR - bright red
        ]
        
        for c in range(min(view.shape[0], len(colors))):
            mask = view[c] > 0
            rgb[mask] = colors[c]
        
        # Mark center (agent position)
        center_y, center_x = h // 2, w // 2
        rgb[center_y, center_x] = (1.0, 1.0, 0.0)  # Yellow for agent
        
        return rgb
    
    def _update_frame(self, frame):
        """Update all visualizations (called by FuncAnimation)."""
        self._process_updates()
        
        # Skip visualization updates in fast mode
        skip_viz = (self.speed_mode == 'fast')
        
        for mode, run in self.runs.items():
            panels = self.run_panels[mode]
            axes = self.axes[mode]
            
            # Status in title
            status = "✓ Done" if run.is_finished else f"Ep {run.episode}"
            axes['view'].set_title(f"{mode} - {status}", fontsize=10, fontweight='bold')
            
            if not skip_viz:
                # Agent view
                if run.agent_view is not None:
                    rgb = self._view_to_rgb(run.agent_view)
                    panels['view_img'].set_array(rgb)
                
                # Action probabilities
                for bar, prob in zip(panels['action_bars'], run.action_probs):
                    bar.set_height(prob)
                    # Color by deviation from uniform
                    uniform = 1 / self.n_actions
                    if prob > uniform * 1.5:
                        bar.set_color('#ff6b6b')
                    elif prob < uniform * 0.5:
                        bar.set_color('#4ecdc4')
                    else:
                        bar.set_color('#00d9ff')
            
            # Graphs always update (even in fast mode)
            if run.rewards:
                x = list(range(len(run.rewards)))
                panels['reward_line'].set_data(x, list(run.rewards))
                axes['reward'].relim()
                axes['reward'].autoscale_view()
            
            if run.policy_losses:
                x = list(range(len(run.policy_losses)))
                panels['policy_line'].set_data(x, list(run.policy_losses))
                panels['value_line'].set_data(x, list(run.value_losses))
                axes['loss'].relim()
                axes['loss'].autoscale_view()
            
            if run.entropies:
                x = list(range(len(run.entropies)))
                panels['entropy_line'].set_data(x, list(run.entropies))
                axes['entropy'].relim()
                axes['entropy'].autoscale_view()
        
        return []
    
    def run(self):
        """Start the dashboard (blocking - call from main thread)."""
        self._create_figure()
        
        # Handle window close
        def on_close(event):
            self.running = False
        
        self.fig.canvas.mpl_connect('close_event', on_close)
        
        # Animation - match main.py speed (50ms)
        # Fast mode uses 200ms and skips viz updates for performance
        interval = 50 if self.speed_mode != 'fast' else 200
        self.ani = FuncAnimation(
            self.fig,
            self._update_frame,
            interval=interval,
            blit=False,
            cache_frame_data=False
        )
        
        plt.show(block=True)
    
    def stop(self):
        """Signal dashboard to stop."""
        self.running = False
        try:
            plt.close(self.fig)
        except Exception:
            pass
    
    def is_paused(self) -> bool:
        """Check if training should pause."""
        return self.paused


def create_dashboard(modes: list[str], n_actions: int = 8) -> TrainingDashboard:
    """
    Factory function to create a training dashboard.
    
    Args:
        modes: List of training mode names
        n_actions: Number of actions
    
    Returns:
        TrainingDashboard instance (not yet running)
    """
    return TrainingDashboard(modes, n_actions)
