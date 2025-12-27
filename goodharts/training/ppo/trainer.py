"""
PPO Trainer - Main training orchestrator.

Provides a clean, subclassable interface for PPO training.
Can be extended for multi-agent scenarios.
"""
import os
import time
import threading
import queue
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.distributions import Categorical
from torch.profiler import record_function
from dataclasses import dataclass
from typing import Optional

from goodharts.configs.default_config import get_config
from goodharts.config import get_training_config
from goodharts.utils.device import get_device, apply_system_optimizations, is_tpu, sync_device
from goodharts.utils.seed import set_seed
from goodharts.behaviors.brains import create_brain, save_brain, _clean_state_dict
from goodharts.behaviors.action_space import create_action_space, ActionSpace
from goodharts.environments.torch_env import create_torch_vec_env
from goodharts.training.process_logger import ProcessLogger
from goodharts.modes import RewardComputer

from .models import Profiler, ValueHead, PopArtValueHead
from .algorithms import compute_gae, ppo_update
from .async_logger import AsyncLogger, LogPayload


class GPUMonitor:
    """
    Background thread that logs GPU utilization at fixed intervals.

    Uses sysfs (AMD, ~0.02ms) or nvidia-smi (NVIDIA) to sample GPU use %.
    Starts after warmup, stops when training ends. Output is CSV with
    millisecond timestamps for correlation with training events.
    """

    def __init__(self, interval_ms: int = 50, output_path: str = "gpu_utilization.csv"):
        self.interval_ms = interval_ms
        self.output_path = output_path
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_time_ms: int = 0
        self._sysfs_path: Optional[str] = None  # Fast path for AMD
        self._nvidia_cmd: Optional[list] = None  # Fallback for NVIDIA

    def _detect_amd_card(self) -> Optional[str]:
        """
        Find the active AMD GPU's sysfs path by testing utilization.

        Returns path like '/sys/class/drm/card1/device/gpu_busy_percent'
        or None if not found.
        """
        import glob

        candidates = glob.glob('/sys/class/drm/card*/device/gpu_busy_percent')
        if not candidates:
            return None

        # If only one card, use it
        if len(candidates) == 1:
            return candidates[0]

        # Multiple cards: find the one PyTorch is using by running a quick workload
        try:
            import torch
            if not torch.cuda.is_available():
                return None

            # Baseline read
            baseline = {}
            for path in candidates:
                with open(path) as f:
                    baseline[path] = int(f.read().strip())

            # Quick GPU work
            x = torch.randn(1024, 1024, device='cuda')
            for _ in range(20):
                x = x @ x
            torch.cuda.synchronize()

            # Find which card spiked
            for path in candidates:
                with open(path) as f:
                    now = int(f.read().strip())
                if now > baseline[path] + 20:  # Significant increase
                    return path

            # Fallback: return first card with vendor 0x1002 (AMD)
            for path in candidates:
                vendor_path = os.path.join(os.path.dirname(path), 'vendor')
                if os.path.exists(vendor_path):
                    with open(vendor_path) as f:
                        if '0x1002' in f.read():
                            return path
        except Exception:
            pass

        return candidates[0]  # Last resort

    def start(self):
        """Start the monitoring thread."""
        import subprocess

        # Try AMD sysfs first (fast: ~0.02ms per read)
        self._sysfs_path = self._detect_amd_card()
        if self._sysfs_path:
            card = self._sysfs_path.split('/')[4]  # Extract 'card1' from path
            print(f"   [GPUMonitor] Using sysfs ({card}) - {self.interval_ms}ms intervals")
        else:
            # Fallback to nvidia-smi (slower: ~5ms per read)
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    self._nvidia_cmd = ["nvidia-smi", "--query-gpu=utilization.gpu",
                                        "--format=csv,noheader,nounits"]
                    print(f"   [GPUMonitor] Using nvidia-smi - {self.interval_ms}ms intervals")
                else:
                    print(f"   [GPUMonitor] No supported GPU monitoring found")
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print(f"   [GPUMonitor] No supported GPU monitoring found")
                return

        # Write CSV header
        with open(self.output_path, 'w') as f:
            f.write("timestamp_ms,gpu_use_pct\n")

        self._start_time_ms = int(time.time() * 1000)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="GPUMonitor")
        self._thread.start()

    def stop(self):
        """Stop the monitoring thread and wait for it to finish."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        print(f"   [GPUMonitor] Stopped")

    def _worker(self):
        """Background thread: samples GPU utilization."""
        import subprocess

        interval_s = self.interval_ms / 1000.0

        with open(self.output_path, 'a') as f:
            while not self._stop_event.is_set():
                try:
                    if self._sysfs_path:
                        # Fast path: direct sysfs read (~0.02ms)
                        with open(self._sysfs_path) as gpu_file:
                            use_pct = int(gpu_file.read().strip())
                    else:
                        # NVIDIA fallback
                        result = subprocess.run(
                            self._nvidia_cmd, capture_output=True, text=True, timeout=1
                        )
                        use_pct = int(result.stdout.strip().split('\n')[0])

                    elapsed_ms = int(time.time() * 1000) - self._start_time_ms
                    f.write(f"{elapsed_ms},{use_pct}\n")
                    f.flush()
                except Exception:
                    pass  # Skip failed samples

                self._stop_event.wait(interval_s)


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
        self.buffers = [
            torch.empty(buffer_size, dtype=torch.float32, pin_memory=True),
            torch.empty(buffer_size, dtype=torch.float32, pin_memory=True),
        ]
        self.events = [
            torch.cuda.Event(),
            torch.cuda.Event(),
        ]
        self.current_buffer = 0  # Which buffer main thread writes to

        # Communication with background thread
        self._work_queue: queue.Queue[Optional[BookkeepingWork]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Buffer ownership: set = available to write, cleared = being read
        # Main thread waits on buffer before writing, background sets after reading
        # This prevents the race condition where main overwrites buffer being read
        self.buffer_available = [threading.Event(), threading.Event()]
        self.buffer_available[0].set()  # Buffer 0 initially free
        self.buffer_available[1].set()  # Buffer 1 initially free

        # Shared state (updated by background, read by main for display)
        # These are updated atomically (single assignment) so no lock needed
        self.best_reward = float('-inf')
        self.episode_count = 0

        # SPS tracking (background thread maintains rolling window)
        self._sps_window: list[tuple[int, float]] = []  # (steps, time) pairs

    @property
    def pinned_buffer(self) -> torch.Tensor:
        """Current buffer for main thread to write to."""
        return self.buffers[self.current_buffer]

    @property
    def current_event(self) -> torch.cuda.Event:
        """Current event for main thread to record."""
        return self.events[self.current_buffer]

    def start(self):
        """Start background processing thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="BackgroundBookkeeper"
        )
        self._thread.start()

    def stop(self):
        """Stop background thread, process remaining work."""
        if self._thread is None:
            return
        self._work_queue.put(None)  # Sentinel
        # Release any potentially blocked submit() to prevent deadlock
        self.buffer_available[0].set()
        self.buffer_available[1].set()
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def submit(
        self,
        gpu_tensor: torch.Tensor,
        update_count: int,
        total_steps: int,
        best_reward: float,
        mode: str,
        profiler_events: list,
        profiler_enabled: bool,
        validation_metrics: Optional[dict],
        update_start_time: float,
        prev_update_steps: int,
        training_start_time: float,
    ):
        """
        Copy metrics to pinned buffer and signal background thread.

        Returns IMMEDIATELY - does not block. GPU can continue working.
        Profiler events are passed as a list to avoid sync on main thread.
        """
        buf_idx = self.current_buffer
        n = len(gpu_tensor)

        # Wait if background thread is still reading this buffer
        # (Should be instant - background finishes in ~5ms, next submit ~800ms later)
        # Only blocks in the rare case where background falls behind
        self.buffer_available[buf_idx].wait()
        self.buffer_available[buf_idx].clear()  # Mark as in-use

        # Copy to pinned buffer (non-blocking DMA)
        self.buffers[buf_idx][:n].copy_(gpu_tensor, non_blocking=True)

        # Record event so background thread knows when copy is done
        self.events[buf_idx].record()

        # Capture timing NOW (on main thread, accurate)
        submit_time = time.perf_counter()

        # Queue work for background thread
        work = BookkeepingWork(
            buffer_idx=buf_idx,
            n_metrics=n,
            update_count=update_count,
            total_steps=total_steps,
            best_reward=best_reward,
            mode=mode,
            profiler_events=profiler_events,
            profiler_enabled=profiler_enabled,
            validation_metrics=validation_metrics,
            submit_time=submit_time,
            update_start_time=update_start_time,
            prev_update_steps=prev_update_steps,
            training_start_time=training_start_time,
        )
        self._work_queue.put(work)

        # Swap to other buffer for next submission
        self.current_buffer = 1 - self.current_buffer

    def _worker(self):
        """Background thread: processes metrics, computes SPS, creates LogPayload."""
        while True:
            try:
                work = self._work_queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if work is None:  # Sentinel - stop
                break

            self._process_work(work)

    def _process_work(self, work: BookkeepingWork):
        """Process one update's metrics (runs in background thread)."""
        try:
            # Wait for GPU->CPU copy to complete (should be instant)
            self.events[work.buffer_idx].synchronize()

            # Read from pinned buffer (safe now that copy is done)
            cpu_array = self.buffers[work.buffer_idx][:work.n_metrics].numpy()
            m = unpack_metrics(cpu_array)

            # Compute profiler summary from events (sync already done above)
            # Since DMA event was recorded AFTER profiler ticks, all profiler
            # events are guaranteed complete by now - no additional sync needed
            if work.profiler_enabled and work.profiler_events:
                times = {}
                for name, start_evt, end_evt in work.profiler_events:
                    dt = start_evt.elapsed_time(end_evt) / 1000.0  # ms -> seconds
                    times[name] = times.get(name, 0.0) + dt
                total = sum(times.values())
                if total > 0:
                    parts = []
                    for k, v in sorted(times.items(), key=lambda x: x[1], reverse=True):
                        pct = v / total * 100
                        parts.append(f"{k}: {v:.2f}s ({pct:.0f}%)")
                    profiler_summary = " | ".join(parts)
                else:
                    profiler_summary = "No data"
            else:
                profiler_summary = "Profiling disabled"

            # Update running totals
            if m['n_episodes'] > 0:
                if m['reward_max'] > self.best_reward:
                    self.best_reward = m['reward_max']
                self.episode_count += m['n_episodes']

            # Compute SPS metrics
            update_steps = work.total_steps - work.prev_update_steps
            update_time = work.submit_time - work.update_start_time

            sps_instant = update_steps / update_time if update_time > 0 else 0

            # Rolling window (last 4 updates)
            self._sps_window.append((update_steps, update_time))
            if len(self._sps_window) > 4:
                self._sps_window.pop(0)

            window_steps = sum(s for s, t in self._sps_window)
            window_time = sum(t for s, t in self._sps_window)
            sps_rolling = window_steps / window_time if window_time > 0 else 0

            # Global SPS
            total_elapsed = work.submit_time - work.training_start_time
            sps_global = work.total_steps / total_elapsed if total_elapsed > 0 else 0

            # Prepare episode stats
            ep_stats = None
            if m['n_episodes'] > 0:
                ep_stats = {
                    'reward': m['reward_sum'] / m['n_episodes'],
                    'food': m['food_sum'] / m['n_episodes'],
                    'poison': m['poison_sum'] / m['n_episodes'],
                }

            # Create LogPayload and queue to async logger
            log_payload = LogPayload(
                policy_loss=m['policy_loss'],
                value_loss=m['value_loss'],
                entropy=m['entropy'],
                explained_var=m['explained_var'],
                action_probs=m['action_probs'],
                update_count=work.update_count,
                total_steps=work.total_steps,
                best_reward=work.best_reward,  # Use value at submission time
                mode=work.mode,
                episode_stats=ep_stats,
                profiler_summary=profiler_summary,
                sps_instant=sps_instant,
                sps_rolling=sps_rolling,
                sps_global=sps_global,
                validation_metrics=work.validation_metrics,
                episodes_count=m['n_episodes'],
                reward_sum=m['reward_sum'],
                reward_min=m['reward_min'] if m['n_episodes'] > 0 else 0.0,
                reward_max=m['reward_max'] if m['n_episodes'] > 0 else 0.0,
                food_sum=m['food_sum'],
                poison_sum=m['poison_sum'],
            )
            self.async_logger.log_update(log_payload)
        finally:
            # Always release buffer for main thread, even on error
            self.buffer_available[work.buffer_idx].set()


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


# Note: TORCHINDUCTOR_CACHE_DIR is set in train_ppo.py before torch is imported
# This ensures the cache persists across runs

# Lock to serialize torch.compile() calls across threads.
# Dynamo has global state that is not thread-safe during compilation.
# This only affects startup; compiled models run in parallel fine.
_COMPILE_LOCK = threading.Lock()

# Global warmup state - shared across sequential/parallel training runs
# Once warmup is done for one mode, subsequent modes skip it
_WARMUP_LOCK = threading.Lock()
_WARMUP_DONE = False

# Global abort flag - checked by all trainers to enable coordinated shutdown
_ABORT_LOCK = threading.Lock()
_ABORT_REQUESTED = False


def request_abort():
    """Signal all trainers to abort gracefully."""
    global _ABORT_REQUESTED
    with _ABORT_LOCK:
        _ABORT_REQUESTED = True


def clear_abort():
    """Clear the abort flag (call before starting new training)."""
    global _ABORT_REQUESTED
    with _ABORT_LOCK:
        _ABORT_REQUESTED = False


def is_abort_requested() -> bool:
    """Check if abort has been requested."""
    with _ABORT_LOCK:
        return _ABORT_REQUESTED


def reset_training_state():
    """
    Reset all global training state.

    Call this after an aborted run to ensure clean state for the next run.
    This is important because globals persist across runs in the same process.
    """
    global _WARMUP_DONE, _ABORT_REQUESTED
    with _WARMUP_LOCK:
        _WARMUP_DONE = False
    with _ABORT_LOCK:
        _ABORT_REQUESTED = False


@dataclass
class PPOConfig:
    """
    Configuration for PPO training.

    Use PPOConfig.from_config() to load defaults from config.toml,
    with CLI arguments as optional overrides.
    """
    mode: str = 'ground_truth'
    brain_type: str = 'base_cnn'
    value_head_type: str = 'popart'  # 'simple' or 'popart'
    action_space_type: str = 'discrete_grid'
    max_move_distance: int = 1
    n_envs: int = 192  # Larger batches = less GPU burstiness
    total_timesteps: int = 100_000
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    k_epochs: int = 4
    steps_per_env: int = 128
    n_minibatches: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    output_path: str = 'models/ppo_agent.pth'
    log_to_file: bool = True
    log_dir: str = 'generated/logs'
    use_amp: bool = False
    compile_models: bool = True
    tensorboard: bool = False
    skip_warmup: bool = False
    use_torch_env: bool = True
    hyper_verbose: bool = False
    clean_cache: bool = False
    profile_enabled: bool = True  # Disable with --no-profile for production
    benchmark_mode: bool = False  # Skip saving, just measure throughput
    gpu_log_interval_ms: int = 0  # GPU utilization logging interval (0 = disabled)

    # Reproducibility
    seed: Optional[int] = None  # None = random seed (logged for reproducibility)
    deterministic: bool = False  # Full determinism (slower)

    # Validation episodes (periodic eval without exploration)
    validation_interval: int = 0     # Every N updates (0 = disabled)
    validation_episodes: int = 16     # Episodes per validation
    validation_mode: str = "training" # "training" or "fixed"
    validation_food: int = 100        # Fixed mode: food count
    validation_poison: int = 50       # Fixed mode: poison count
    
    @classmethod
    def from_config(cls, mode: str = 'ground_truth', **overrides) -> 'PPOConfig':
        """
        Create PPOConfig from config.toml with optional CLI overrides.

        Config file provides defaults; explicit kwargs override them.
        This allows CLI args to be truly optional.

        Args:
            mode: Training mode (ground_truth, proxy, etc.)
            **overrides: Any PPOConfig fields to override

        Returns:
            PPOConfig with values from config file + overrides
        """
        train_cfg = get_training_config()

        # Build config from file defaults
        config_values = {
            'mode': mode,
            'brain_type': train_cfg.get('brain_type', 'base_cnn'),
            'value_head_type': train_cfg.get('value_head_type', 'popart'),
            'action_space_type': train_cfg.get('action_space_type', 'discrete_grid'),
            'max_move_distance': train_cfg.get('max_move_distance', 1),
            'n_envs': train_cfg.get('n_envs', 64),
            'lr': train_cfg.get('learning_rate', 3e-4),
            'gamma': train_cfg.get('gamma', 0.99),
            'gae_lambda': train_cfg.get('gae_lambda', 0.95),
            'eps_clip': train_cfg.get('eps_clip', 0.2),
            'k_epochs': train_cfg.get('k_epochs', 4),
            'steps_per_env': train_cfg.get('steps_per_env', 128),
            'n_minibatches': train_cfg.get('n_minibatches', 4),
            'entropy_coef': train_cfg.get('entropy_coef', 0.01),
            'value_coef': train_cfg.get('value_coef', 0.5),
            'use_amp': train_cfg.get('use_amp', False),
            'compile_models': train_cfg.get('compile_models', True),
            # Validation
            'validation_interval': train_cfg.get('validation_interval', 0),
            'validation_episodes': train_cfg.get('validation_episodes', 16),
            'validation_mode': train_cfg.get('validation_mode', 'training'),
            'validation_food': train_cfg.get('validation_food', 100),
            'validation_poison': train_cfg.get('validation_poison', 50),
        }

        # Apply overrides (CLI args take precedence)
        for key, value in overrides.items():
            if value is not None:  # Only override if explicitly set
                config_values[key] = value

        return cls(**config_values)


def _vprint(msg: str, verbose: bool):
    """Verbose print helper."""
    if verbose:
        print(f"[VERBOSE] {msg}", flush=True)


class PPOTrainer:
    """
    PPO training orchestrator.
    
    Handles the full training loop including:
    - Environment creation
    - Experience collection
    - GAE computation
    - PPO updates
    - Logging and checkpointing
    
    Can be subclassed for multi-agent or custom scenarios.
    """
    
    def __init__(
        self,
        config: PPOConfig,
        device: Optional[torch.device] = None,
        dashboard = None,
    ):
        """
        Initialize trainer.
        
        Args:
            config: PPO configuration
            device: Torch device (auto-detected if None)
            dashboard: Optional training dashboard for live visualization
        """
        self.config = config
        self.device = device or get_device()
        self.dashboard = dashboard
        
        # Will be initialized on train()
        self.vec_env = None
        self.policy = None
        self.value_head = None
        self.optimizer = None
        self.logger = None
        self.reward_computer = None
        self.profiler = None
        self.async_logger = None

        # External profiler callback (called after each update, returns False to stop)
        self._profiler_callback = None

        # GPU monitor (started after warmup if enabled)
        self.gpu_monitor = None

        # Training state
        self.total_steps = 0
        self.update_count = 0
        self.best_reward = float('-inf')
        self.episode_count = 0
        self._aborted = False  # Track if training was aborted

    def _warmup_forward_backward(self, batch_size: int, include_backward: bool = True, label: str = "Warmup") -> float:
        """
        Run forward (and optionally backward) pass to warm up JIT/cuDNN.

        This triggers algorithm selection for cuDNN benchmark mode and compiles
        torch.compile graphs. Running this once at startup avoids the compilation
        penalty during actual training.

        Args:
            batch_size: Batch size for dummy tensors (should match training batch)
            include_backward: Whether to run backward pass (needed for gradient kernels)
            label: Label for progress messages

        Returns:
            Time taken in seconds
        """
        cfg = self.config
        start_time = time.time()

        # Create dummy tensors matching training shapes
        dummy_obs = torch.zeros(
            (batch_size, self.vec_env.n_channels, self.vec_env.view_size, self.vec_env.view_size),
            device=self.device, requires_grad=False
        )

        with autocast(device_type=self.device_type, enabled=cfg.use_amp):
            logits = self.policy(dummy_obs)
            features = self.policy.get_features(dummy_obs)
            values = self.value_head(features).squeeze(-1)

            if include_backward:
                dummy_actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                dummy_returns = torch.zeros(batch_size, device=self.device)

                dist = Categorical(logits=logits, validate_args=False)
                log_probs = dist.log_prob(dummy_actions)
                dummy_loss = -log_probs.mean() + F.mse_loss(values, dummy_returns)
                dummy_loss.backward()

        # Synchronize to ensure kernels complete
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Clean up
        if include_backward:
            self.policy.zero_grad()
            self.value_head.zero_grad()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return time.time() - start_time

    def _run_warmup_update(self, states: torch.Tensor, episode_rewards: torch.Tensor) -> torch.Tensor:
        """
        Run one full training update as warmup (discarded).

        This triggers all lazy initialization in a realistic context:
        - Real environment steps (not synthetic data)
        - Full forward/backward through compiled models
        - All MIOpen/cuDNN algorithm selection

        The results are discarded; this just warms up the runtime.

        Args:
            states: Current environment states
            episode_rewards: Episode reward accumulator

        Returns:
            New states after the warmup steps
        """
        cfg = self.config

        # Use pre-allocated buffers (same as main training loop)
        states_buf = self._states_buf
        actions_buf = self._actions_buf
        log_probs_buf = self._log_probs_buf
        rewards_buf = self._rewards_buf
        dones_buf = self._dones_buf
        values_buf = self._values_buf

        for step in range(cfg.steps_per_env):
            with torch.no_grad():
                # Use compiled functions if available (triggers compilation on first call)
                if self._compiled_inference is not None:
                    logits, features, values = self._compiled_inference(states)
                    actions, log_probs = self._compiled_sample(logits)
                else:
                    states_t = states.float()
                    with autocast(device_type=self.device_type, enabled=cfg.use_amp):
                        logits, features = self.policy.forward_with_features(states_t)
                        values = self.value_head(features).squeeze(-1)
                        dist = Categorical(logits=logits, validate_args=False)
                        actions = dist.sample()
                        log_probs = dist.log_prob(actions)

            current_states = states.clone()
            next_states, rewards, dones = self.vec_env.step(actions)
            shaped_rewards = self.reward_computer.compute(rewards, current_states, next_states, dones)

            # Store in pre-allocated tensor buffers
            states_buf[step] = current_states
            actions_buf[step] = actions
            log_probs_buf[step] = log_probs.detach()
            rewards_buf[step] = shaped_rewards
            dones_buf[step] = dones
            values_buf[step] = values

            episode_rewards += rewards
            episode_rewards *= (~dones)
            states = next_states

        # Bootstrap value
        with torch.no_grad():
            states_t = states.float()
            _, features = self.policy.forward_with_features(states_t)
            next_value = self.value_head(features).squeeze(-1)

        # Compute GAE (pass tensors directly, no stacking needed)
        advantages, returns = compute_gae(
            rewards_buf, values_buf, dones_buf,
            next_value, cfg.gamma, cfg.gae_lambda, device=self.device
        )

        # Update PopArt statistics for value normalization
        if hasattr(self.value_head, 'update_stats'):
            self.value_head.update_stats(returns.flatten())

        # PPO update (this triggers backward pass lazy init)
        # Reshape pre-allocated buffers (no torch.cat needed)
        batch_size = cfg.steps_per_env * cfg.n_envs
        all_states = states_buf.reshape(batch_size, -1, self.vec_env.view_size, self.vec_env.view_size).float()
        all_actions = actions_buf.flatten()
        all_log_probs = log_probs_buf.flatten()
        all_values = values_buf.flatten()
        all_returns = returns.flatten()
        all_advantages = advantages.flatten()

        # Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        ppo_update(
            self.policy, self.value_head, self.optimizer,
            all_states, all_actions, all_log_probs,
            all_returns, all_advantages, all_values,
            self.device,
            eps_clip=cfg.eps_clip,
            k_epochs=cfg.k_epochs,
            entropy_coef=cfg.entropy_coef,
            value_coef=cfg.value_coef,
            n_minibatches=cfg.n_minibatches,
            scaler=self.scaler,
        )

        # Sync to ensure all kernels complete
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        return states

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns:
            Summary dict with training results
        """
        self._setup()

        try:
            self._training_loop()
        except KeyboardInterrupt:
            self._aborted = True
            print(f"\n[{self.config.mode}] Training interrupted by user")
        except Exception as e:
            self._aborted = True
            print(f"\n[{self.config.mode}] Training failed: {e}")
            self.cleanup()
            raise

        # Check if abort was requested globally (e.g., by signal handler)
        if is_abort_requested():
            self._aborted = True

        return self._finalize()

    def cleanup(self):
        """
        Release all resources.

        Call this after training completes or on abort to ensure
        GPU memory, threads, and file handles are properly released.
        """
        # Stop GPU monitor first
        if self.gpu_monitor is not None:
            try:
                self.gpu_monitor.stop()
            except Exception:
                pass
            self.gpu_monitor = None

        # Stop bookkeeper before async_logger (bookkeeper submits to async_logger)
        if hasattr(self, 'bookkeeper') and self.bookkeeper is not None:
            try:
                self.bookkeeper.stop()
            except Exception:
                pass
            self.bookkeeper = None

        # Shutdown async logger (flushes any pending logs)
        if self.async_logger:
            try:
                self.async_logger.shutdown(timeout=2.0)
            except Exception:
                pass  # Best effort
            self.async_logger = None

        # Close TensorBoard writer
        if self.tb_writer:
            try:
                self.tb_writer.close()
            except Exception:
                pass
            self.tb_writer = None

        # Release environment (frees GPU tensors)
        if self.vec_env:
            try:
                # VecEnv doesn't have an explicit close(), but we can
                # release references to allow garbage collection
                del self.vec_env
            except Exception:
                pass
            self.vec_env = None

        # Release model tensors
        # Note: Can't use truthiness check on torch.compile wrapped modules
        # (they don't support __len__), so check against None explicitly
        if self.policy is not None:
            del self.policy
            self.policy = None
        if self.value_head is not None:
            del self.value_head
            self.value_head = None
        if self.optimizer is not None:
            del self.optimizer
            self.optimizer = None

        # Clear GPU cache
        if self.device and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    
    def _setup(self):
        """Initialize environment, networks, and logging."""
        cfg = self.config

        # Reproducibility: set all random seeds
        self.seed = set_seed(
            seed=cfg.seed,
            deterministic=cfg.deterministic,
            verbose=False
        )

        v = cfg.hyper_verbose
        _vprint("_setup() starting", v)

        print(f"\n[PPO] Starting training: {cfg.mode}")
        print(f"   Device: {self.device}, Envs: {cfg.n_envs}, Seed: {self.seed}")
        if is_tpu(self.device):
            print("   Note: TPU uses XLA - first step compiles the graph (may take a few minutes)")
        
        # Apply hardware optimizations
        _vprint("Applying system optimizations...", v)
        apply_system_optimizations(self.device, verbose=True)
        
        # Load configs
        _vprint("Loading configs...", v)
        sim_config = get_config()
        train_cfg = get_training_config()
        
        # Observation spec
        _vprint("Creating observation spec...", v)
        self.spec = sim_config['get_observation_spec'](cfg.mode)

        # Action space (pluggable - stored for serialization)
        self.action_space = create_action_space(
            cfg.action_space_type,
            max_move_distance=cfg.max_move_distance
        )
        n_actions = self.action_space.n_outputs
        
        # Environment - ALWAYS use GPU-native TorchVecEnv
        _vprint("Creating TorchVecEnv...", v)
        self.vec_env = create_torch_vec_env(n_envs=cfg.n_envs, obs_spec=self.spec, device=self.device)
        env_type = "TorchVecEnv (TPU-native)" if is_tpu(self.device) else "TorchVecEnv (GPU-native)"
        print(f"   Env: {env_type}")
        _vprint("Environment created", v)
        print(f"   View: {self.vec_env.view_size}x{self.vec_env.view_size}, Channels: {self.vec_env.n_channels}")
        
        # Networks
        _vprint("Creating networks...", v)
        self.policy = create_brain(cfg.brain_type, self.spec, output_size=n_actions).to(self.device)

        # Value head - configurable between simple and PopArt
        if cfg.value_head_type == 'popart':
            self.value_head = PopArtValueHead(input_size=self.policy.hidden_size).to(self.device)
        elif cfg.value_head_type == 'simple':
            self.value_head = ValueHead(input_size=self.policy.hidden_size).to(self.device)
        else:
            raise ValueError(f"Unknown value_head_type: {cfg.value_head_type}. Use 'simple' or 'popart'.")
        # Store architecture info before potential torch.compile (for serialization)
        self._policy_arch_info = self.policy.get_architecture_info()
        _vprint("Networks created and moved to device", v)

        # AMP
        self.device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.scaler = GradScaler(enabled=cfg.use_amp) if cfg.use_amp else None

        # Compile models for extra speed if torch.compile is available (PyTorch 2.0+)
        # Use lock to serialize compilation - Dynamo's global state is not thread-safe
        # Skip on TPU - XLA uses its own JIT compilation
        # Note: Actual warmup happens in _training_loop via _run_warmup_update()
        #
        # Key insight: Compiling just the models helps, but compiling the ENTIRE
        # inference step (forward + sampling) eliminates inter-update gaps by fusing
        # many small kernels into fewer large ones. This reduces kernel launch overhead
        # from ~1ms to ~0.006ms per update.
        self._compiled_inference = None
        self._compiled_sample = None

        with _COMPILE_LOCK:
            if cfg.compile_models and hasattr(torch, 'compile') and not is_tpu(self.device):
                try:
                    # Use max-autotune-no-cudagraphs for best kernel optimization
                    # without CUDA graph tensor reuse conflicts
                    compile_mode = 'max-autotune-no-cudagraphs'

                    # Compile the full inference step, not just individual models
                    # This fuses .float(), autocast, forward, and squeeze into one graph
                    policy = self.policy
                    value_head = self.value_head
                    device_type = self.device_type
                    use_amp = cfg.use_amp

                    @torch.compile(mode=compile_mode)
                    def compiled_inference(states):
                        """Fused inference: float conversion + policy + value."""
                        states_t = states.float()
                        with autocast(device_type=device_type, enabled=use_amp):
                            logits, features = policy.forward_with_features(states_t)
                            values = value_head(features).squeeze(-1)
                        return logits, features, values

                    @torch.compile(mode=compile_mode)
                    def compiled_sample(logits):
                        """Fused sampling: distribution + sample + log_prob."""
                        dist = Categorical(logits=logits, validate_args=False)
                        actions = dist.sample()
                        log_probs = dist.log_prob(actions)
                        return actions, log_probs

                    self._compiled_inference = compiled_inference
                    self._compiled_sample = compiled_sample

                    print(f"   [JIT] torch.compile enabled (max-autotune-no-cudagraphs)")
                except RuntimeError as e:
                    if "FX" in str(e) or "dynamo" in str(e).lower():
                        print(f"   [JIT] Warning: torch.compile failed ({e}). Using eager mode.")
                    else:
                        raise e
        
        # Optimizer - use fused=True on CUDA to eliminate .item() sync overhead
        # Fused runs entirely on GPU, avoiding 384 CPU round-trips per update
        use_fused = self.device.type == 'cuda'
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=cfg.lr,
            fused=use_fused
        )
        
        
        # Reward computer - unified class handles both numpy and torch
        self.reward_computer = RewardComputer.create(
            cfg.mode, self.spec, cfg.gamma, 
            shaping_coef=train_cfg['shaping_food_attract'], 
            device=self.device
        )
        
        # Logger (skip in benchmark mode)
        # Uses process-isolated logging to completely escape the GIL
        if cfg.log_to_file and not cfg.benchmark_mode:
            self.logger = ProcessLogger(mode=cfg.mode, output_dir=cfg.log_dir)
            self.logger.start()  # Start writer process
            self.logger.set_hyperparams(
                mode=cfg.mode,
                brain_type=cfg.brain_type,
                n_envs=cfg.n_envs,
                total_timesteps=cfg.total_timesteps,
                lr=cfg.lr,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                eps_clip=cfg.eps_clip,
                k_epochs=cfg.k_epochs,
                steps_per_env=cfg.steps_per_env,
                n_minibatches=cfg.n_minibatches,
                entropy_coef=cfg.entropy_coef,
                vectorized=True,
            )
        
        # TensorBoard (optional - works on Colab without display server)
        self.tb_writer = None
        if cfg.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(cfg.log_dir, 'tensorboard', cfg.mode)
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                print(f"   TensorBoard: {tb_dir}")
            except ImportError:
                print("   TensorBoard: Not available (install tensorboard package)")
        
        # Profiler (disable with --no-profile to remove GPU sync overhead)
        self.profiler = Profiler(self.device, enabled=cfg.profile_enabled)
        
        # Async logger - handles all I/O in background thread to avoid GPU stalls
        self.async_logger = AsyncLogger(
            trainer_logger=self.logger,
            tb_writer=self.tb_writer,
            dashboard=self.dashboard,
            mode=cfg.mode
        )
        self.async_logger.start()

        # Background bookkeeper - processes metrics in parallel with GPU work
        # Uses double-buffered pinned memory for zero-copy submission
        buffer_size = N_SCALAR_METRICS + 16  # Schema metrics + action probs
        self.bookkeeper = BackgroundBookkeeper(
            buffer_size=buffer_size,
            async_logger=self.async_logger,
            device=self.device,
        )
        self.bookkeeper.start()
        
        # Curriculum settings - inform VecEnv of ranges (each env randomizes on reset)
        self.min_food = train_cfg.get('min_food', 50)
        self.max_food = train_cfg.get('max_food', 200)
        self.min_poison = train_cfg.get('min_poison', 20)
        self.max_poison = train_cfg.get('max_poison', 100)
        self.vec_env.set_curriculum_ranges(
            self.min_food, self.max_food, 
            self.min_poison, self.max_poison
        )
        # Force reset to apply new ranges
        self.vec_env.reset()
        
        # Checkpoint settings
        self.checkpoint_interval = train_cfg.get('checkpoint_interval', 0)
        self.checkpoint_dir = os.path.dirname(cfg.output_path) or 'models'

        # Pre-allocate rollout buffers (eliminates torch.cat burstiness)
        # These are GPU tensors that get overwritten each update cycle
        obs_shape = (cfg.steps_per_env, cfg.n_envs,
                     self.vec_env.n_channels, self.vec_env.view_size, self.vec_env.view_size)
        scalar_shape = (cfg.steps_per_env, cfg.n_envs)

        self._states_buf = torch.zeros(obs_shape, device=self.device, dtype=torch.uint8)
        self._actions_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.long)
        self._log_probs_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.float32)
        self._rewards_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.float32)
        self._dones_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.bool)
        self._values_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.float32)

        # Episode tracking buffers (for async logging)
        self._finished_dones_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.bool)
        self._finished_rewards_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.float32)

        # Async metrics pipeline: GPU transfers to pinned memory while continuing work
        # We process PREVIOUS update's metrics while GPU runs NEXT update
        # 18 = 6 episode aggregates + 4 PPO metrics + 8 action probs (max)
        self._metrics_pinned = torch.zeros(32, dtype=torch.float32, pin_memory=True)
        self._metrics_event = None  # CUDA event for pending transfer
        self._pending_log_data = None  # Metadata for pending metrics

        print(f"   Brain: {cfg.brain_type} (hidden={self.policy.hidden_size})")
        print(f"   Value head: {cfg.value_head_type}")
        print(f"   AMP: {'Enabled' if cfg.use_amp else 'Disabled'}")
    
    def _training_loop(self):
        """Main training loop."""
        cfg = self.config
        v = cfg.hyper_verbose  # Verbose debug mode

        # Use pre-allocated GPU tensor buffers (from _setup)
        # These eliminate torch.cat overhead that caused GPU burstiness
        states_buf = self._states_buf
        actions_buf = self._actions_buf
        log_probs_buf = self._log_probs_buf
        rewards_buf = self._rewards_buf
        dones_buf = self._dones_buf
        values_buf = self._values_buf
        finished_dones_buf = self._finished_dones_buf
        finished_rewards_buf = self._finished_rewards_buf

        # Current step within the buffer (reset each update)
        step_in_buffer = 0
        
        # Initial state
        _vprint("Resetting environment for initial state...", v)
        states = self.vec_env.reset()
        _vprint("Environment reset done", v)

        # Initialize reward computer
        _vprint("Initializing reward computer...", v)
        self.reward_computer.initialize(states)  # stays on GPU
        episode_rewards = torch.zeros(cfg.n_envs, device=self.device)
        _vprint("Training loop ready to start", v)

        # === WARMUP UPDATE (lazy init) ===
        # Run one full update cycle to trigger all lazy initialization:
        # - cuDNN/MIOpen algorithm selection
        # - Autograd graph construction
        # - Any remaining JIT compilation
        # This is discarded; real training starts fresh after.
        # For sequential/parallel training, warmup only runs once.
        global _WARMUP_DONE
        should_warmup = not cfg.skip_warmup

        with _WARMUP_LOCK:
            if _WARMUP_DONE:
                should_warmup = False
                print(f"   [Warmup] Skipped (already done)", flush=True)

        if should_warmup:
            print(f"   [Warmup] Running warmup update (lazy init)...", flush=True)
            warmup_start = time.perf_counter()

            # Save model state before warmup (will be restored after)
            policy_state = {k: v.clone() for k, v in self.policy.state_dict().items()}
            value_head_state = {k: v.clone() for k, v in self.value_head.state_dict().items()}
            optimizer_state = self.optimizer.state_dict()

            # Run warmup (modifies model, but we'll restore)
            states = self._run_warmup_update(states, episode_rewards)

            # Restore model state (discard warmup training)
            self.policy.load_state_dict(policy_state)
            self.value_head.load_state_dict(value_head_state)
            self.optimizer.load_state_dict(optimizer_state)

            warmup_elapsed = time.perf_counter() - warmup_start
            print(f"   [Warmup] Complete ({warmup_elapsed:.1f}s) - weights restored", flush=True)

            # Mark warmup as done globally
            with _WARMUP_LOCK:
                _WARMUP_DONE = True

            # Reset environment state for clean start
            states = self.vec_env.reset()
            self.reward_computer.initialize(states)
            episode_rewards.zero_()

        # Start GPU monitor after warmup (if enabled)
        if cfg.gpu_log_interval_ms > 0:
            self.gpu_monitor = GPUMonitor(
                interval_ms=cfg.gpu_log_interval_ms,
                output_path="gpu_utilization.csv"
            )
            self.gpu_monitor.start()

        step_in_update = 0
        self.start_time = time.perf_counter()

        # Rolling window for sps calculation (all updates valid post-warmup)
        sps_window = []  # (steps, time) pairs for last 4 updates
        last_update_time = self.start_time
        last_update_steps = 0

        # Inference prefetch: True if inference was issued after PPO update
        # GPU runs inference while CPU does bookkeeping (overlap)
        inference_pending = False

        while self.total_steps < cfg.total_timesteps:
            self.profiler.start()
            
            # Check for abort request (signal handler or dashboard stop button)
            if is_abort_requested():
                print(f"\n[{cfg.mode}] Abort signal received")
                break

            # Check stop signal via dashboard (if available)
            if self.dashboard and hasattr(self.dashboard, 'should_stop') and self.dashboard.should_stop():
                print(f"\n[{cfg.mode}] Dashboard stop requested")
                break
            
            if v:
                _vprint(f"Step {self.total_steps}: collecting experience...", v)
            
            # Collect experience
            with torch.no_grad():
                if inference_pending:
                    # Inference was issued after previous PPO update (overlapped with bookkeeping)
                    # Results are already in logits, features, values, actions, log_probs
                    inference_pending = False
                else:
                    # Normal inference path
                    if v:
                        _vprint(f"Step {self.total_steps}: running inference...", v)

                    # Use compiled functions if available (fuses many small kernels)
                    with record_function("INFERENCE"):
                        if self._compiled_inference is not None:
                            logits, features, values = self._compiled_inference(states)
                            actions, log_probs = self._compiled_sample(logits)
                        else:
                            # Fallback to eager mode
                            states_t = states.float()
                            with autocast(device_type=self.device_type, enabled=cfg.use_amp):
                                logits, features = self.policy.forward_with_features(states_t)
                                values = self.value_head(features).squeeze(-1)
                                dist = Categorical(logits=logits, validate_args=False)
                                actions = dist.sample()
                                log_probs = dist.log_prob(actions)

                    # Store last logits for action_probs logging (computed once per update)
                    last_logits = logits

                    self.profiler.tick("Inference")

            # Environment step
            # GPU-native path: everything stays on GPU
            # CRITICAL: Snapshot state BEFORE step because env may mutate it in-place!
            with record_function("ENV_STEP"):
                current_states = states.clone()
                next_states, rewards, dones = self.vec_env.step(actions)

            # Compute shaped rewards (torch-native, stays on GPU)
            with record_function("REWARD_SHAPE"):
                shaped_rewards = self.reward_computer.compute(
                    rewards, current_states, next_states, dones
                )
            
            # Store experience in pre-allocated tensor buffers (no torch.cat needed)
            states_buf[step_in_buffer] = current_states
            actions_buf[step_in_buffer] = actions
            log_probs_buf[step_in_buffer] = log_probs.detach()
            rewards_buf[step_in_buffer] = shaped_rewards
            dones_buf[step_in_buffer] = dones
            values_buf[step_in_buffer] = values

            # Track episode stats (defer logging to avoid Sync)
            finished_dones_buf[step_in_buffer] = dones
            finished_rewards_buf[step_in_buffer] = episode_rewards

            step_in_buffer += 1
            self.total_steps += cfg.n_envs
            episode_rewards += rewards  # Tensor addition

            self.profiler.tick("Env Step")
            
            # Reset rewards for done agents (Sync-free)
            # episode_rewards = episode_rewards * (1 - dones)
            episode_rewards *= (~dones)
            
            states = next_states
            
            # PPO Update (when buffer is full)
            if step_in_buffer >= cfg.steps_per_env:
                self.profiler.tick("Collection")

                # Get bootstrap value (keep on GPU)
                with record_function("GAE_COMPUTE"):
                    with torch.no_grad():
                        states_t = states.float()
                        _, features = self.policy.forward_with_features(states_t)
                        next_value = self.value_head(features).squeeze(-1)

                    # Compute GAE (pass pre-allocated tensors directly, no stacking)
                    advantages, returns = compute_gae(
                        rewards_buf, values_buf, dones_buf,
                        next_value, cfg.gamma, cfg.gae_lambda, device=self.device
                    )

                    # Update PopArt statistics for value normalization
                    if hasattr(self.value_head, 'update_stats'):
                        self.value_head.update_stats(returns.flatten())

                self.profiler.tick("GAE Calc")

                # EPISODE LOGGING - Fixed-size GPU aggregates, NO sync until after PPO
                with record_function("BUFFER_FLATTEN"):
                    # Pre-allocated buffers are already (steps, envs) tensors
                    all_step_dones = finished_dones_buf  # Already (steps, envs)
                    all_step_rewards = finished_rewards_buf  # Already (steps, envs)

                    # Masked aggregates - all ops stay on GPU
                    done_mask = all_step_dones.float()
                    INF = torch.tensor(float('inf'), device=self.device)

                    # Food/poison: sum for envs that finished at least once this update
                    any_done_per_env = all_step_dones.any(dim=0)  # (envs,)

                    # Episode aggregates - order must match metrics_keys in sync section
                    episode_agg_values = [
                        done_mask.sum(),
                        (all_step_rewards * done_mask).sum(),
                        torch.where(all_step_dones, all_step_rewards, INF).min(),
                        torch.where(all_step_dones, all_step_rewards, -INF).max(),
                        (self.vec_env.last_episode_food * any_done_per_env).sum().float(),
                        (self.vec_env.last_episode_poison * any_done_per_env).sum().float(),
                    ]

                    # Stack into tensor (size determined by list length, not hardcoded)
                    episode_agg_tensor = torch.stack(episode_agg_values)

                    # Reshape pre-allocated buffers (no torch.cat needed!)
                    # states_buf is (steps, envs, C, H, W) -> flatten to (steps*envs, C, H, W)
                    batch_size = cfg.steps_per_env * cfg.n_envs
                    all_states = states_buf.reshape(batch_size, -1, self.vec_env.view_size, self.vec_env.view_size).float()
                    all_actions = actions_buf.flatten()
                    all_log_probs = log_probs_buf.flatten()
                    all_old_values = values_buf.flatten()

                    # Returns and advantages from compute_gae
                    all_returns = returns.flatten()
                    all_advantages = advantages.flatten()

                    if v:
                        _vprint(f"   [DEBUG] Rewards: mean={rewards_buf.mean():.4f}, std={rewards_buf.std():.4f}", v)
                        _vprint(f"   [DEBUG] Advantages: mean={all_advantages.mean():.4f}, std={all_advantages.std():.4f}", v)
                        _vprint(f"   [DEBUG] Returns: mean={all_returns.mean():.4f}, std={all_returns.std():.4f}", v)

                    # Normalize advantages
                    all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

                # PPO update - tensors go in directly, no CPU sync needed
                with record_function("PPO_UPDATE"):
                    policy_loss, value_loss, entropy, explained_var = ppo_update(
                        self.policy, self.value_head, self.optimizer,
                        all_states, all_actions, all_log_probs,
                        all_returns, all_advantages, all_old_values,
                        self.device,
                        eps_clip=cfg.eps_clip,
                        k_epochs=cfg.k_epochs,
                        entropy_coef=cfg.entropy_coef,
                        value_coef=cfg.value_coef,
                        n_minibatches=cfg.n_minibatches,
                        scaler=self.scaler,
                    verbose=True,
                )
                # Clear progress line
                print(" " * 40, end='\r')

                # Sync device only for TPU (forces XLA execution)
                # CUDA profiling is now async - sync happens in background thread
                if is_tpu(self.device):
                    sync_device(self.device)
                self.profiler.tick("PPO Update")

                self.update_count += 1

                # External profiler callback (for --profile-trace)
                if self._profiler_callback is not None:
                    if not self._profiler_callback(self.update_count):
                        # Callback returned False - stop training early
                        break

                # Reset buffer index (buffers are pre-allocated, just overwrite)
                step_in_buffer = 0

                # Run validation if needed (before metrics submission)
                val_metrics = None
                if cfg.validation_interval > 0 and self.update_count % cfg.validation_interval == 0:
                    val_metrics = self._run_validation_episodes()
                    if cfg.validation_mode == "training":
                        states = self.vec_env.reset()
                        self.reward_computer.initialize(states)
                        episode_rewards.zero_()

                # ============================================================
                # INFERENCE PREFETCH: Issue next rollout's inference NOW
                # GPU runs inference while CPU does bookkeeping (overlap)
                # ============================================================
                # Save old logits for metrics packing (action probs from previous update)
                prev_logits = last_logits

                with torch.no_grad():
                    with record_function("INFERENCE"):
                        if self._compiled_inference is not None:
                            logits, features, values = self._compiled_inference(states)
                            actions, log_probs = self._compiled_sample(logits)
                        else:
                            states_t = states.float()
                            with autocast(device_type=self.device_type, enabled=cfg.use_amp):
                                logits, features = self.policy.forward_with_features(states_t)
                                values = self.value_head(features).squeeze(-1)
                                dist = Categorical(logits=logits, validate_args=False)
                                actions = dist.sample()
                                log_probs = dist.log_prob(actions)
                last_logits = logits
                inference_pending = True

                # ============================================================
                # CPU BOOKKEEPING: GPU is running inference in parallel
                # ============================================================

                # Pack metrics tensor (GPU work - fast)
                # Use prev_logits (from last step of previous update) for action probs
                action_probs_gpu = F.softmax(prev_logits, dim=1)[0]
                metrics_values = torch.cat([
                    episode_agg_tensor,           # n_episodes, reward_sum/min/max, food/poison
                    policy_loss.unsqueeze(0),
                    value_loss.unsqueeze(0),
                    entropy.unsqueeze(0),
                    explained_var.unsqueeze(0),
                    action_probs_gpu,             # Variable length tail
                ])

                # ZERO-BLOCK SUBMISSION: Send metrics to background thread
                # Main thread continues immediately, GPU keeps running
                # Background thread handles: sync, unpack, SPS calc, LogPayload, async_logger
                #
                # Pass profiler events instead of summary to avoid sync on main thread!
                # The background thread computes summary after its DMA sync completes.
                profiler_events = list(self.profiler._pending_events) if cfg.profile_enabled else []
                self.bookkeeper.submit(
                    gpu_tensor=metrics_values,
                    update_count=self.update_count,
                    total_steps=self.total_steps,
                    best_reward=self.bookkeeper.best_reward,
                    mode=cfg.mode,
                    profiler_events=profiler_events,
                    profiler_enabled=cfg.profile_enabled,
                    validation_metrics=val_metrics,
                    update_start_time=last_update_time,
                    prev_update_steps=last_update_steps,
                    training_start_time=self.start_time,
                )

                # Reset profiler for next update (clears pending events)
                self.profiler.reset()

                # Update timing reference for NEXT submission
                last_update_time = time.perf_counter()
                last_update_steps = self.total_steps
                
                # Checkpoint
                if self.checkpoint_interval > 0 and self.update_count % self.checkpoint_interval == 0:
                    self._save_checkpoint()
    
    def _save_checkpoint(self):
        """Save training checkpoint with full architecture metadata."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{self.config.mode}_u{self.update_count}_s{self.total_steps}.pth"
        )
        torch.save({
            # Brain metadata (for architecture-agnostic loading)
            'brain_type': self.config.brain_type,
            'architecture': self._policy_arch_info,
            'action_space': self.action_space.get_config(),
            'mode': self.config.mode,
            'seed': self.seed,
            # Training state
            'update': self.update_count,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            # Model weights (cleaned of torch.compile prefixes)
            'state_dict': _clean_state_dict(self.policy.state_dict()),
            'value_head_state_dict': _clean_state_dict(self.value_head.state_dict()),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"   Checkpoint saved: {checkpoint_path}")

    def _process_pending_metrics(self):
        """
        Process metrics from PREVIOUS update (if any).

        Called at the start of each update cycle. By now, the async transfer
        that started at the end of the previous update has long since completed
        (transfer takes ~8ms, update takes ~800ms).
        """
        if self._metrics_event is None:
            return  # No pending metrics (first update)

        # Sync transfer (should be instant - finished 800ms ago)
        self._metrics_event.synchronize()

        # Unpack using schema (single source of truth)
        pending = self._pending_log_data
        cpu_array = self._metrics_pinned[:pending.n_metrics].numpy()
        m = unpack_metrics(cpu_array)

        # Update running totals (one update behind, nobody notices)
        if m['n_episodes'] > 0:
            if m['reward_max'] > self.best_reward:
                self.best_reward = m['reward_max']
            self.episode_count += m['n_episodes']

        # Prepare episode stats
        ep_stats = None
        if m['n_episodes'] > 0:
            ep_stats = {
                'reward': m['reward_sum'] / m['n_episodes'],
                'food': m['food_sum'] / m['n_episodes'],
                'poison': m['poison_sum'] / m['n_episodes'],
            }

        # Queue to async logger
        log_payload = LogPayload(
            policy_loss=m['policy_loss'],
            value_loss=m['value_loss'],
            entropy=m['entropy'],
            explained_var=m['explained_var'],
            action_probs=m['action_probs'],
            update_count=pending.update_count,
            total_steps=pending.total_steps,
            best_reward=self.best_reward,
            mode=self.config.mode,
            episode_stats=ep_stats,
            profiler_summary=pending.profiler_summary,
            sps_instant=pending.sps_instant,
            sps_rolling=pending.sps_rolling,
            sps_global=pending.sps_global,
            validation_metrics=pending.validation_metrics,
            episodes_count=m['n_episodes'],
            reward_sum=m['reward_sum'],
            reward_min=m['reward_min'] if m['n_episodes'] > 0 else 0.0,
            reward_max=m['reward_max'] if m['n_episodes'] > 0 else 0.0,
            food_sum=m['food_sum'],
            poison_sum=m['poison_sum'],
        )
        self.async_logger.log_update(log_payload)

        # Clear pending state
        self._metrics_event = None
        self._pending_log_data = None

    def _start_metrics_transfer(
        self,
        metrics_values: torch.Tensor,
        update_count: int,
        total_steps: int,
        sps_instant: float,
        sps_rolling: float,
        sps_global: float,
        profiler_summary: str,
        validation_metrics: Optional[dict],
    ):
        """
        Start async transfer of metrics to CPU.

        GPU continues immediately to next update. Transfer completes in background.
        Processing happens at start of NEXT update (or at training end).

        Metrics must be packed according to METRICS_SCHEMA order.
        """
        n_metrics = len(metrics_values)

        # Copy to pinned buffer (non-blocking DMA transfer)
        self._metrics_pinned[:n_metrics].copy_(metrics_values, non_blocking=True)

        # Record event so we know when transfer completes
        self._metrics_event = torch.cuda.Event()
        self._metrics_event.record()

        # Store context for later processing
        self._pending_log_data = PendingMetrics(
            n_metrics=n_metrics,
            update_count=update_count,
            total_steps=total_steps,
            sps_instant=sps_instant,
            sps_rolling=sps_rolling,
            sps_global=sps_global,
            profiler_summary=profiler_summary,
            validation_metrics=validation_metrics,
        )

    def _run_validation_episodes(self) -> dict:
        """
        Run deterministic validation episodes.
        
        Returns metrics dict with mean reward, food, and poison.
        Uses argmax for action selection (no exploration).
        """
        cfg = self.config
        
        # Select environment based on mode
        if cfg.validation_mode == "fixed":
            # Create temporary env with fixed density
            val_env = create_torch_vec_env(
                n_envs=cfg.n_envs, 
                obs_spec=self.spec, 
                device=self.device
            )
            val_env.set_curriculum_ranges(
                cfg.validation_food, cfg.validation_food,
                cfg.validation_poison, cfg.validation_poison
            )
            val_env.reset()
        else:
            # Use training env directly (faster, same randomization)
            val_env = self.vec_env
        
        # Run episodes
        total_reward = 0.0
        total_food = 0
        total_poison = 0
        episodes_done = 0
        
        states = val_env.reset()
        episode_rewards = torch.zeros(cfg.n_envs, device=self.device)
        
        # Run until we have enough completed episodes
        max_steps = cfg.n_envs * 600  # Safety limit
        steps = 0
        
        with torch.no_grad():
            while episodes_done < cfg.validation_episodes and steps < max_steps:
                states_t = states.float()
                
                # Deterministic action selection (argmax instead of sample)
                logits = self.policy(states_t)
                actions = logits.argmax(dim=-1)
                
                next_states, rewards, dones = val_env.step(actions)
                episode_rewards += rewards
                
                # Check for completed episodes
                done_mask = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_mask.numel() > 0:
                    for idx in done_mask:
                        if episodes_done >= cfg.validation_episodes:
                            break
                        total_reward += episode_rewards[idx].item()
                        total_food += val_env.last_episode_food[idx].item()
                        total_poison += val_env.last_episode_poison[idx].item()
                        episodes_done += 1
                        episode_rewards[idx] = 0.0
                
                states = next_states
                steps += cfg.n_envs
        
        # Compute means
        n = max(episodes_done, 1)
        metrics = {
            'reward': total_reward / n,
            'food': total_food / n,
            'poison': total_poison / n,
            'episodes': episodes_done,
            'mode': cfg.validation_mode,
        }
        
        # Print validation results
        print(f"   [Validation] reward={metrics['reward']:.2f}, "
              f"food={metrics['food']:.1f}, poison={metrics['poison']:.1f} "
              f"({episodes_done} episodes, {cfg.validation_mode} env)")
        
        return metrics
    
    
    def _finalize(self) -> dict:
        """Save final model and return summary."""
        cfg = self.config

        # Let bookkeeper finish processing any remaining metrics
        # (It processes in background, so give it a moment to catch up)
        if hasattr(self, 'bookkeeper') and self.bookkeeper is not None:
            time.sleep(0.1)  # Brief pause for background thread
            best_reward = self.bookkeeper.best_reward
        else:
            best_reward = self.best_reward

        # Calculate throughput (handle case where start_time wasn't set)
        elapsed = 0.0
        throughput = 0.0
        if hasattr(self, 'start_time'):
            elapsed = time.perf_counter() - self.start_time
            throughput = self.total_steps / elapsed if elapsed > 0 else 0

        # Handle aborted training
        if self._aborted:
            print(f"\n[{cfg.mode}] Training aborted after {self.total_steps:,} steps")
            # Still try to flush logs
            if self.async_logger:
                self.async_logger.shutdown(timeout=2.0)
            if self.dashboard:
                self.dashboard.update(cfg.mode, 'aborted', None)
            # Full cleanup
            self.cleanup()
            return {
                'mode': cfg.mode,
                'total_steps': self.total_steps,
                'elapsed_seconds': elapsed,
                'throughput': throughput,
                'best_reward': best_reward,
                'model_path': None,
                'aborted': True,
            }

        # Normal completion path
        # Shutdown async logger first - flushes all queued logs
        if self.async_logger:
            self.async_logger.shutdown()

        if cfg.benchmark_mode:
            # Benchmark mode: just report throughput, don't save
            print(f"\n{'='*60}")
            print(f"BENCHMARK RESULTS ({cfg.mode})")
            print(f"{'='*60}")
            print(f"  Total steps:    {self.total_steps:,}")
            print(f"  Elapsed time:   {elapsed:.2f}s")
            print(f"  Throughput:     {throughput:,.0f} steps/sec")
            print(f"  Best reward:    {best_reward:.1f}")
            print(f"{'='*60}")
        else:
            # Save with full metadata for architecture-agnostic loading
            checkpoint = {
                'brain_type': cfg.brain_type,
                'architecture': self._policy_arch_info,
                'action_space': self.action_space.get_config(),
                'state_dict': _clean_state_dict(self.policy.state_dict()),
                'mode': cfg.mode,
                'training_steps': self.total_steps,
                'seed': self.seed,
            }
            torch.save(checkpoint, cfg.output_path)
            print(f"\n[PPO] Training complete: {cfg.output_path} (seed={self.seed})")

            if self.logger:
                self.logger.finalize(best_efficiency=best_reward, final_model_path=cfg.output_path)

        # Close TensorBoard writer to flush all data
        if self.tb_writer:
            self.tb_writer.close()

        if self.dashboard:
            self.dashboard.update(cfg.mode, 'finished', None)

        # Full cleanup after normal completion too
        self.cleanup()

        return {
            'mode': cfg.mode,
            'total_steps': self.total_steps,
            'elapsed_seconds': elapsed,
            'throughput': throughput,
            'best_reward': best_reward,
            'model_path': None if cfg.benchmark_mode else cfg.output_path,
            'aborted': False,
        }
