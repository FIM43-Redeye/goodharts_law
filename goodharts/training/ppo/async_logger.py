"""
Async logging for PPO training.

Offloads I/O (console, file, dashboard) to a background thread.
GPU sync (.item() calls) happens in main thread to avoid CUDA contention.
"""
import queue
import threading
from dataclasses import dataclass
from typing import Optional



@dataclass
class LogPayload:
    """Data queued for async logging - all values already synced to CPU."""
    # Metrics (already synced to CPU floats in main thread)
    policy_loss: float
    value_loss: float
    entropy: float
    explained_var: float
    action_probs: list  # Already computed as list[float]

    # Metadata
    update_count: int
    total_steps: int
    best_reward: float
    mode: str

    # Episode stats (already on CPU)
    episode_stats: Optional[dict]  # {'reward': float, 'food': int, 'poison': int}

    # Profiler summary (string)
    profiler_summary: str

    # Timing info - three sps metrics for different perspectives
    sps_instant: float   # This update only
    sps_rolling: float   # Last 4 updates
    sps_global: float    # Total steps / total time

    # Validation metrics (optional, only present on validation runs)
    validation_metrics: Optional[dict] = None

    # Episode aggregates (computed on GPU, only 5 floats transferred)
    # Replaces per-episode arrays that caused serialization overhead
    episodes_count: int = 0
    reward_sum: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0
    food_sum: int = 0
    poison_sum: int = 0


class AsyncLogger:
    """
    Thread-safe async logging for PPO training.
    
    GPU sync happens in main thread; only I/O is offloaded to background.
    This avoids CUDA contention that caused severe performance regression.
    """
    
    def __init__(
        self,
        tb_writer,       # TensorBoard SummaryWriter (unified logging target)
        dashboard,       # Training dashboard or None
        mode: str,
    ):
        self.tb_writer = tb_writer
        self.dashboard = dashboard
        self.mode = mode
        
        self._queue: queue.Queue[Optional[LogPayload]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._console_counter = 0
        self._console_interval = 10  # Print to console every N updates
    
    def start(self):
        """Start the background logging thread."""
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="AsyncLogger")
        self._thread.start()
    
    def log_update(self, payload: LogPayload):
        """Queue metrics for async logging. Non-blocking."""
        self._queue.put(payload)
    
    def shutdown(self, timeout: float = 5.0):
        """Stop the background thread and flush remaining logs."""
        self._running = False
        self._queue.put(None)  # Sentinel to wake up thread
        if self._thread is not None:
            self._thread.join(timeout=timeout)
    
    def _worker(self):
        """Background thread: processes log queue (I/O only, no GPU access)."""
        while self._running or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if payload is None:
                break
            
            self._process_payload(payload)
    
    def _process_payload(self, p: LogPayload):
        """Do I/O work for one update (no GPU access here)."""
        # Console output - throttled to reduce spam
        self._console_counter += 1
        if self._console_counter >= self._console_interval:
            self._console_counter = 0
            print(f"   [{p.mode}] Step {p.total_steps:,}: {p.sps_instant:,.0f} sps [{p.sps_rolling:,.0f}/{p.sps_global:,.0f}] | Best R={p.best_reward:.0f} | Ent={p.entropy:.3f} | ValL={p.value_loss:.4f} | ExpV={p.explained_var:.4f}")
            if p.profiler_summary and p.profiler_summary != "No data":
                print(f"   [Profile] {p.profiler_summary}")
        
        # Compute derived metrics for logging
        n = p.episodes_count if p.episodes_count > 0 else 1
        food_mean = p.food_sum / n
        poison_mean = p.poison_sum / n
        reward_mean = p.reward_sum / n
        total_consumed = food_mean + poison_mean
        food_ratio = food_mean / total_consumed if total_consumed > 0 else 0.5
        reward_per_consumed = reward_mean / total_consumed if total_consumed > 0 else 0.0

        # TensorBoard logging (unified logging target)
        if self.tb_writer:
            self.tb_writer.add_scalar('loss/policy', p.policy_loss, p.total_steps)
            self.tb_writer.add_scalar('loss/value', p.value_loss, p.total_steps)
            self.tb_writer.add_scalar('metrics/entropy', p.entropy, p.total_steps)
            self.tb_writer.add_scalar('metrics/explained_variance', p.explained_var, p.total_steps)
            # Curriculum-invariant metrics
            self.tb_writer.add_scalar('metrics/food_ratio', food_ratio, p.total_steps)
            self.tb_writer.add_scalar('metrics/reward_per_consumed', reward_per_consumed, p.total_steps)
            if p.episode_stats:
                self.tb_writer.add_scalar('reward/episode', p.episode_stats['reward'], p.total_steps)
            # Validation metrics
            if p.validation_metrics:
                self.tb_writer.add_scalar('validation/reward', p.validation_metrics['reward'], p.total_steps)
                self.tb_writer.add_scalar('validation/food', p.validation_metrics['food'], p.total_steps)
                self.tb_writer.add_scalar('validation/poison', p.validation_metrics['poison'], p.total_steps)
            self.tb_writer.flush()
        
        # Dashboard update
        if self.dashboard:
            # Use new aggregate fields (food_sum, poison_sum, reward_sum)
            # These are populated by BackgroundBookkeeper; episode_stats is legacy
            n = p.episodes_count if p.episodes_count > 0 else 1

            # Flat dictionary format expected by dashboard._process_updates()
            payload = {
                'policy_loss': p.policy_loss,
                'value_loss': p.value_loss,
                'entropy': p.entropy,
                'explained_var': p.explained_var,
                'action_probs': p.action_probs,
                'total_steps': p.total_steps,
                # Episode stats from aggregates (new format)
                'reward': p.reward_sum / n,
                'food': p.food_sum / n,
                'poison': p.poison_sum / n,
            }
            self.dashboard.update(p.mode, 'update', payload)
