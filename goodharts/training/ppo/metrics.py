"""
Metrics pipeline for PPO training.

Provides asynchronous GPU->CPU metrics transfer with double-buffered pinned memory.
This eliminates GPU idle time during bookkeeping by processing metrics in a
background thread while the main thread continues GPU work.
"""
import time
import threading
import queue
from dataclasses import dataclass
from typing import Optional

import torch

from .async_logger import LogPayload


# =============================================================================
# METRICS SCHEMA - Single source of truth for GPU->CPU metrics transfer
# =============================================================================
# Order matters! Pack on GPU in this order, unpack on CPU in same order.
# To add a metric: add here, add to pack list, it unpacks automatically.

METRICS_SCHEMA = [
    # Episode aggregates (computed on GPU from finished episodes)
    ('n_episodes', int),
    ('reward_sum', float),
    ('reward_min', float),
    ('reward_max', float),
    ('food_sum', int),
    ('poison_sum', int),
    # PPO training metrics
    ('policy_loss', float),
    ('value_loss', float),
    ('entropy', float),
    ('explained_var', float),
    # Action probs follow (variable length, not in schema)
]

N_SCALAR_METRICS = len(METRICS_SCHEMA)


def unpack_metrics(cpu_array) -> dict:
    """Unpack transferred metrics according to schema."""
    result = {}
    for i, (name, dtype) in enumerate(METRICS_SCHEMA):
        result[name] = dtype(cpu_array[i])
    # Action probs are the tail
    result['action_probs'] = cpu_array[N_SCALAR_METRICS:].tolist()
    return result


@dataclass
class PendingMetrics:
    """
    Context for async metrics transfer.

    GPU starts transfer, continues to next update. CPU processes this later.
    By the time we need it (next update end), transfer finished long ago.

    Episode data (n_episodes, reward_sum, etc.) is in the transferred tensor,
    extracted in _process_pending_metrics after transfer completes.
    """
    n_metrics: int  # Length of metrics in pinned buffer
    update_count: int
    total_steps: int
    sps_instant: float
    sps_rolling: float
    sps_global: float
    profiler_summary: str
    validation_metrics: Optional[dict]


@dataclass
class BookkeepingWork:
    """
    Work item for background bookkeeping thread.

    Contains everything needed to process metrics without touching GPU.
    All timing data is captured at submission time on main thread.
    """
    buffer_idx: int           # Which pinned buffer to read from (0 or 1)
    n_metrics: int            # How many elements to read
    update_count: int
    total_steps: int
    best_reward: float        # Current best (for updating)
    mode: str
    profiler_events: list     # CUDA event pairs for async profiler summary
    profiler_enabled: bool    # Whether profiling is enabled
    validation_metrics: Optional[dict]
    # Timing data for SPS calculation (captured on main thread)
    submit_time: float        # time.perf_counter() at submission
    update_start_time: float  # When this update started
    prev_update_steps: int    # Steps at end of previous update
    training_start_time: float  # When training started (for global SPS)


class BackgroundBookkeeper:
    """
    Processes metrics in background thread while main thread continues GPU work.

    Uses double-buffered pinned memory to eliminate data races:
    - Main thread writes to buffer A, signals "A ready"
    - Background thread reads from A while main writes to B
    - No locks needed - each buffer has exactly one owner at a time

    This eliminates the ~10-20ms GPU idle time that occurred when the main
    thread did bookkeeping between PPO updates.
    """

    def __init__(
        self,
        buffer_size: int,
        async_logger,  # AsyncLogger instance
        device: torch.device,
    ):
        self.async_logger = async_logger
        self.device = device

        # Double-buffered pinned memory
        self.pinned_buffers = [
            torch.zeros(buffer_size, dtype=torch.float32, pin_memory=True),
            torch.zeros(buffer_size, dtype=torch.float32, pin_memory=True),
        ]
        self.current_buffer = 0  # Which buffer main thread is writing to

        # Signaling: buffer_available[i] means background can read from i
        self.buffer_available = [threading.Event(), threading.Event()]
        self.buffer_available[0].set()  # Buffer 0 starts available
        self.buffer_available[1].set()  # Buffer 1 starts available

        # Work queue and thread
        self.work_queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Shared state (updated by background thread)
        self._best_reward_lock = threading.Lock()
        self._best_reward = float('-inf')
        self._episode_count_lock = threading.Lock()
        self._episode_count = 0

    @property
    def best_reward(self) -> float:
        with self._best_reward_lock:
            return self._best_reward

    @property
    def episode_count(self) -> int:
        with self._episode_count_lock:
            return self._episode_count

    def start(self):
        """Start background bookkeeping thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="BackgroundBookkeeper"
        )
        self._thread.start()

    def stop(self):
        """Stop background thread and wait for completion."""
        if self._thread is None:
            return
        self._stop_event.set()
        self.work_queue.put(None)  # Poison pill
        self._thread.join(timeout=5.0)
        self._thread = None

    def submit(
        self,
        gpu_tensor: torch.Tensor,
        update_count: int,
        total_steps: int,
        mode: str,
        profiler_events: list,
        profiler_enabled: bool,
        validation_metrics: Optional[dict],
        update_start_time: float,
        prev_update_steps: int,
        training_start_time: float,
    ):
        """
        Submit metrics for background processing.

        Copies GPU tensor to pinned buffer asynchronously, then queues work.
        Returns immediately so main thread can start next GPU work.
        """
        # Wait for current buffer to be available (background done reading)
        buf_idx = self.current_buffer
        self.buffer_available[buf_idx].wait()
        self.buffer_available[buf_idx].clear()  # Mark as in-use by main

        # Async copy to pinned memory (non-blocking on GPU)
        n_metrics = gpu_tensor.shape[0]
        self.pinned_buffers[buf_idx][:n_metrics].copy_(gpu_tensor, non_blocking=True)

        # Queue work item
        work = BookkeepingWork(
            buffer_idx=buf_idx,
            n_metrics=n_metrics,
            update_count=update_count,
            total_steps=total_steps,
            best_reward=self.best_reward,
            mode=mode,
            profiler_events=profiler_events,
            profiler_enabled=profiler_enabled,
            validation_metrics=validation_metrics,
            submit_time=time.perf_counter(),
            update_start_time=update_start_time,
            prev_update_steps=prev_update_steps,
            training_start_time=training_start_time,
        )
        self.work_queue.put(work)

        # Flip to other buffer for next call
        self.current_buffer = 1 - self.current_buffer

    def _worker(self):
        """Background thread: processes metrics from queue."""
        while not self._stop_event.is_set():
            try:
                work = self.work_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if work is None:  # Poison pill
                break

            try:
                self._process_work(work)
            finally:
                # Always release buffer for main thread, even on error
                self.buffer_available[work.buffer_idx].set()

    def _process_work(self, work: BookkeepingWork):
        """Process a single work item."""
        # Read from pinned buffer (CPU memory, no GPU sync needed)
        cpu_data = self.pinned_buffers[work.buffer_idx][:work.n_metrics].numpy()
        m = unpack_metrics(cpu_data)

        # Update shared state
        n_eps = m['n_episodes']
        if n_eps > 0:
            avg_reward = m['reward_sum'] / n_eps
            with self._best_reward_lock:
                if avg_reward > self._best_reward:
                    self._best_reward = avg_reward

            with self._episode_count_lock:
                self._episode_count += n_eps

        # Calculate SPS (on CPU, no GPU sync)
        now = work.submit_time
        update_duration = now - work.update_start_time
        steps_this_update = work.total_steps - work.prev_update_steps
        sps_instant = steps_this_update / max(update_duration, 1e-6)

        elapsed = now - work.training_start_time
        sps_global = work.total_steps / max(elapsed, 1e-6)

        # Generate profiler summary (from pre-captured CUDA events)
        profiler_summary = ""
        if work.profiler_enabled and work.profiler_events:
            try:
                summaries = []
                for name, start_event, end_event in work.profiler_events:
                    # This synchronizes the events (already complete by now)
                    duration_ms = start_event.elapsed_time(end_event)
                    summaries.append(f"{name}:{duration_ms:.1f}ms")
                profiler_summary = " | ".join(summaries)
            except RuntimeError:
                # CUDA event timing can fail if events weren't recorded properly
                # This is non-critical - just skip profiler summary
                profiler_summary = "(profiler timing unavailable)"

        # Log via AsyncLogger
        log_payload = LogPayload(
            update_count=work.update_count,
            total_steps=work.total_steps,
            sps_instant=sps_instant,
            sps_rolling=sps_instant,  # Rolling average could be added
            sps_global=sps_global,
            policy_loss=m['policy_loss'],
            value_loss=m['value_loss'],
            entropy=m['entropy'],
            explained_var=m['explained_var'],
            best_reward=self.best_reward,
            profiler_summary=profiler_summary,
            validation_metrics=work.validation_metrics,
            action_probs=m['action_probs'],
            mode=work.mode,
            # Episode stats as dict (legacy format for episode_stats field)
            episode_stats=None,  # Not using legacy dict format
            # Episode aggregates (new format - individual fields)
            episodes_count=m['n_episodes'],
            reward_sum=m['reward_sum'],
            reward_min=m['reward_min'],
            reward_max=m['reward_max'],
            food_sum=int(m['food_sum']),
            poison_sum=int(m['poison_sum']),
        )
        self.async_logger.log_update(log_payload)
