"""
Live training visualization dashboard.

Opens a matplotlib window that displays real-time training statistics:
- Loss curves (policy loss, value loss, entropy)
- Action probability distribution
- Reward and episode length trends
- Curriculum progress

Designed to help diagnose training issues like uniform action probabilities.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Callable
import threading
import queue


@dataclass
class TrainingStats:
    """Container for training statistics, updated by training loop."""
    
    # Rolling windows for smoothed plots
    window_size: int = 100
    
    # Core metrics (deques for efficient rolling window)
    episode_rewards: deque = field(default_factory=lambda: deque(maxlen=500))
    episode_lengths: deque = field(default_factory=lambda: deque(maxlen=500))
    policy_losses: deque = field(default_factory=lambda: deque(maxlen=500))
    value_losses: deque = field(default_factory=lambda: deque(maxlen=500))
    entropies: deque = field(default_factory=lambda: deque(maxlen=500))
    
    # Action probability distribution (updated each PPO update)
    action_probs: np.ndarray = field(default_factory=lambda: np.ones(8) / 8)
    action_probs_std: float = 0.0  # Standard deviation - key diagnostic!
    
    # Curriculum info
    current_food: int = 0
    curriculum_progress: float = 0.0
    
    # Training state
    episode: int = 0
    total_steps: int = 0
    best_efficiency: float = float('-inf')
    
    # Mode name for display
    mode: str = ""
    
    def add_episode(self, reward: float, length: int, food_count: int, progress: float):
        """Record episode completion."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.current_food = food_count
        self.curriculum_progress = progress
        self.episode += 1
    
    def add_ppo_update(self, policy_loss: float, value_loss: float, 
                       entropy: float, action_probs: np.ndarray):
        """Record PPO update metrics."""
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        self.action_probs = action_probs
        self.action_probs_std = action_probs.std()
    
    def get_rolling_mean(self, data: deque, window: int = None) -> float:
        """Get rolling mean of recent data."""
        if not data:
            return 0.0
        window = window or self.window_size
        recent = list(data)[-window:]
        return np.mean(recent) if recent else 0.0


class TrainingVisualizer:
    """
    Live visualization window for training progress.
    
    Runs in a separate thread with a message queue for thread-safe updates.
    """
    
    def __init__(self, mode: str, n_actions: int = 8):
        """
        Initialize the visualizer.
        
        Args:
            mode: Training mode name (e.g., 'ground_truth', 'proxy')
            n_actions: Number of possible actions
        """
        self.mode = mode
        self.n_actions = n_actions
        self.stats = TrainingStats(mode=mode)
        
        # Thread-safe queue for updates from training loop
        self.update_queue: queue.Queue = queue.Queue()
        
        # Plot state
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.bars = {}
        self.texts = {}
        
        # Control
        self.running = False
        self._thread = None
    
    def start(self):
        """Start the visualization in a background thread."""
        if self.running:
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the visualization."""
        self.running = False
        if self.fig:
            plt.close(self.fig)
    
    def update(self, **kwargs):
        """
        Queue an update from the training loop (thread-safe).
        
        Accepts any TrainingStats field as a keyword argument:
            visualizer.update(episode_reward=100.5, episode_length=250)
        
        Or special update types:
            visualizer.update(episode=(reward, length, food, progress))
            visualizer.update(ppo=(policy_loss, value_loss, entropy, action_probs))
        """
        self.update_queue.put(kwargs)
    
    def _process_updates(self):
        """Process all pending updates from the queue."""
        while not self.update_queue.empty():
            try:
                update = self.update_queue.get_nowait()
                
                # Handle special composite updates
                if 'episode' in update:
                    reward, length, food, progress = update['episode']
                    self.stats.add_episode(reward, length, food, progress)
                
                if 'ppo' in update:
                    policy_loss, value_loss, entropy, action_probs = update['ppo']
                    self.stats.add_ppo_update(policy_loss, value_loss, entropy, action_probs)
                
                # Handle direct field updates
                for key, value in update.items():
                    if key not in ('episode', 'ppo') and hasattr(self.stats, key):
                        setattr(self.stats, key, value)
                        
            except queue.Empty:
                break
    
    def _create_figure(self):
        """Create the matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle(f"🧠 Training: {self.mode}", fontsize=14, fontweight='bold')
        
        # 3x3 grid layout
        gs = gridspec.GridSpec(3, 3, figure=self.fig, hspace=0.4, wspace=0.3)
        
        # Row 1: Rewards | Episode Length | Action Probs
        self.axes['reward'] = self.fig.add_subplot(gs[0, 0])
        self.axes['length'] = self.fig.add_subplot(gs[0, 1])
        self.axes['actions'] = self.fig.add_subplot(gs[0, 2])
        
        # Row 2: Policy Loss | Value Loss | Entropy
        self.axes['policy_loss'] = self.fig.add_subplot(gs[1, 0])
        self.axes['value_loss'] = self.fig.add_subplot(gs[1, 1])
        self.axes['entropy'] = self.fig.add_subplot(gs[1, 2])
        
        # Row 3: Curriculum | Action Std (diagnostic) | Stats
        self.axes['curriculum'] = self.fig.add_subplot(gs[2, 0])
        self.axes['action_std'] = self.fig.add_subplot(gs[2, 1])
        self.axes['stats'] = self.fig.add_subplot(gs[2, 2])
        
        self._setup_plots()
    
    def _setup_plots(self):
        """Initialize all plot elements."""
        # Reward plot
        ax = self.axes['reward']
        ax.set_title("Episode Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        self.lines['reward'], = ax.plot([], [], 'b-', linewidth=1, alpha=0.5, label='Raw')
        self.lines['reward_smooth'], = ax.plot([], [], 'b-', linewidth=2, label='Smoothed')
        ax.legend(loc='upper left', fontsize=8)
        
        # Episode length plot
        ax = self.axes['length']
        ax.set_title("Episode Length")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.grid(True, alpha=0.3)
        self.lines['length'], = ax.plot([], [], 'g-', linewidth=1, alpha=0.5)
        self.lines['length_smooth'], = ax.plot([], [], 'g-', linewidth=2)
        
        # Action probabilities (bar chart)
        ax = self.axes['actions']
        ax.set_title("Action Probabilities")
        ax.set_xlabel("Action")
        ax.set_ylabel("Probability")
        action_labels = ['↑', '↓', '←', '→', '↖', '↗', '↙', '↘'][:self.n_actions]
        x = np.arange(self.n_actions)
        self.bars['actions'] = ax.bar(x, np.ones(self.n_actions) / self.n_actions, 
                                      color='#00d9ff', edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(action_labels)
        ax.set_ylim(0, 0.5)
        ax.axhline(y=1/self.n_actions, color='red', linestyle='--', alpha=0.5, label='Uniform')
        ax.legend(fontsize=8)
        
        # Policy loss
        ax = self.axes['policy_loss']
        ax.set_title("Policy Loss")
        ax.set_xlabel("Update")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        self.lines['policy_loss'], = ax.plot([], [], 'r-', linewidth=1)
        
        # Value loss
        ax = self.axes['value_loss']
        ax.set_title("Value Loss")
        ax.set_xlabel("Update")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        self.lines['value_loss'], = ax.plot([], [], 'm-', linewidth=1)
        
        # Entropy (key diagnostic!)
        ax = self.axes['entropy']
        ax.set_title("Entropy 🔍")
        ax.set_xlabel("Update")
        ax.set_ylabel("Entropy")
        ax.grid(True, alpha=0.3)
        self.lines['entropy'], = ax.plot([], [], 'c-', linewidth=2)
        # Add reference line for uniform distribution entropy
        max_entropy = np.log(self.n_actions)
        ax.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.5, label=f'Max (uniform) = {max_entropy:.2f}')
        ax.legend(fontsize=8)
        
        # Curriculum progress
        ax = self.axes['curriculum']
        ax.set_title("Curriculum Progress")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Food Density")
        ax.grid(True, alpha=0.3)
        self.lines['curriculum'], = ax.plot([], [], 'orange', linewidth=2)
        
        # Action probability std (KEY DIAGNOSTIC)
        ax = self.axes['action_std']
        ax.set_title("Action Prob Std 🔍 (higher = more opinionated)")
        ax.set_xlabel("Update")
        ax.set_ylabel("Std Dev")
        ax.grid(True, alpha=0.3)
        self.lines['action_std'], = ax.plot([], [], 'purple', linewidth=2)
        ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Uniform = 0')
        ax.legend(fontsize=8)
        
        # Stats text box
        ax = self.axes['stats']
        ax.axis('off')
        self.texts['stats'] = ax.text(0.1, 0.9, "", transform=ax.transAxes,
                                      fontsize=11, verticalalignment='top',
                                      fontfamily='monospace')
    
    def _update_plots(self, frame):
        """Update all plots with current data."""
        self._process_updates()
        stats = self.stats
        
        # Reward plot
        if stats.episode_rewards:
            episodes = list(range(len(stats.episode_rewards)))
            rewards = list(stats.episode_rewards)
            self.lines['reward'].set_data(episodes, rewards)
            
            # Smoothed line
            if len(rewards) > 10:
                smooth = np.convolve(rewards, np.ones(10)/10, mode='valid')
                self.lines['reward_smooth'].set_data(range(9, len(rewards)), smooth)
            
            ax = self.axes['reward']
            ax.relim()
            ax.autoscale_view()
        
        # Episode length
        if stats.episode_lengths:
            episodes = list(range(len(stats.episode_lengths)))
            lengths = list(stats.episode_lengths)
            self.lines['length'].set_data(episodes, lengths)
            
            if len(lengths) > 10:
                smooth = np.convolve(lengths, np.ones(10)/10, mode='valid')
                self.lines['length_smooth'].set_data(range(9, len(lengths)), smooth)
            
            ax = self.axes['length']
            ax.relim()
            ax.autoscale_view()
        
        # Action probabilities
        for bar, prob in zip(self.bars['actions'], stats.action_probs):
            bar.set_height(prob)
            # Color bars by deviation from uniform
            uniform = 1 / self.n_actions
            if prob > uniform * 1.5:
                bar.set_color('#ff6b6b')  # Red = high
            elif prob < uniform * 0.5:
                bar.set_color('#4ecdc4')  # Teal = low
            else:
                bar.set_color('#00d9ff')  # Blue = normal
        
        # Losses
        if stats.policy_losses:
            updates = list(range(len(stats.policy_losses)))
            self.lines['policy_loss'].set_data(updates, list(stats.policy_losses))
            self.axes['policy_loss'].relim()
            self.axes['policy_loss'].autoscale_view()
        
        if stats.value_losses:
            updates = list(range(len(stats.value_losses)))
            self.lines['value_loss'].set_data(updates, list(stats.value_losses))
            self.axes['value_loss'].relim()
            self.axes['value_loss'].autoscale_view()
        
        # Entropy
        if stats.entropies:
            updates = list(range(len(stats.entropies)))
            self.lines['entropy'].set_data(updates, list(stats.entropies))
            self.axes['entropy'].relim()
            self.axes['entropy'].autoscale_view()
        
        # Curriculum - track food density over episodes
        if stats.episode_rewards:  # Use episodes as x-axis
            # Create a line showing food density trend
            episodes = list(range(len(stats.episode_rewards)))
            # We only have current food, estimate history
            if stats.curriculum_progress > 0:
                # Reconstruct approximate food curve
                initial_food = 2500  # From config
                final_food = 50
                foods = [initial_food - (i / max(1, len(episodes)-1)) * (initial_food - final_food) * stats.curriculum_progress
                         for i in range(len(episodes))]
                foods[-1] = stats.current_food  # Correct current value
                self.lines['curriculum'].set_data(episodes, foods)
                self.axes['curriculum'].relim()
                self.axes['curriculum'].autoscale_view()
        
        # Action std history (build from entropies as proxy, or track separately)
        # For now, show current value as horizontal line
        ax = self.axes['action_std']
        if stats.entropies:
            # Approximate: std from entropy (higher entropy = lower std)
            # This is a rough approximation
            stds = [max(0, 1 - e / np.log(self.n_actions)) * 0.3 for e in stats.entropies]
            updates = list(range(len(stds)))
            self.lines['action_std'].set_data(updates, stds)
            ax.relim()
            ax.autoscale_view()
        
        # Stats text
        avg_reward = stats.get_rolling_mean(stats.episode_rewards, 10)
        avg_length = stats.get_rolling_mean(stats.episode_lengths, 10)
        
        stats_text = f"""Episode: {stats.episode}
Total Steps: {stats.total_steps:,}

Recent (10 ep):
  Reward: {avg_reward:+.1f}
  Length: {avg_length:.0f}

Curriculum:
  Food: {stats.current_food}
  Progress: {stats.curriculum_progress*100:.0f}%

Action Std: {stats.action_probs_std:.4f}
Best Eff: {stats.best_efficiency:.2f}"""
        
        self.texts['stats'].set_text(stats_text)
        
        return list(self.lines.values()) + list(self.bars.values()) + list(self.texts.values())
    
    def _run_loop(self):
        """Main visualization loop (runs in background thread)."""
        plt.ion()  # Interactive mode
        self._create_figure()
        
        # Animation with manual updates
        self.ani = FuncAnimation(
            self.fig, 
            self._update_plots,
            interval=200,  # Update every 200ms
            blit=False,
            cache_frame_data=False
        )
        
        plt.show(block=True)


def create_training_visualizer(mode: str, n_actions: int = 8) -> TrainingVisualizer:
    """
    Factory function to create and start a training visualizer.
    
    Args:
        mode: Training mode name
        n_actions: Number of possible actions
    
    Returns:
        Started TrainingVisualizer instance
    
    Example:
        viz = create_training_visualizer('ground_truth')
        
        # In training loop:
        viz.update(episode=(reward, length, food, progress))
        viz.update(ppo=(policy_loss, value_loss, entropy, action_probs))
        
        # When done:
        viz.stop()
    """
    viz = TrainingVisualizer(mode, n_actions)
    viz.start()
    return viz
