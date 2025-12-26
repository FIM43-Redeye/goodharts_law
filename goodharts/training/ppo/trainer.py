"""
PPO Trainer - Main training orchestrator.

Provides a clean, subclassable interface for PPO training.
Can be extended for multi-agent scenarios.
"""
import os
import time
import threading
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import Optional

from goodharts.configs.default_config import get_config
from goodharts.config import get_training_config
from goodharts.utils.device import get_device, apply_system_optimizations, is_tpu, sync_device
from goodharts.utils.seed import set_seed
from goodharts.behaviors.brains import create_brain, save_brain, _clean_state_dict
from goodharts.behaviors.action_space import create_action_space, ActionSpace
from goodharts.environments.torch_env import create_torch_vec_env
from goodharts.training.train_log import TrainingLogger
from goodharts.modes import RewardComputer

from .models import Profiler, ValueHead
from .algorithms import compute_gae, ppo_update
from .async_logger import AsyncLogger, LogPayload


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
    action_space_type: str = 'discrete_grid'
    max_move_distance: int = 1
    n_envs: int = 64
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

    # Reproducibility
    seed: Optional[int] = None  # None = random seed (logged for reproducibility)
    deterministic: bool = False  # Full determinism (slower)

    # Validation episodes (periodic eval without exploration)
    validation_interval: int = 8     # Every N updates (0 = disabled)
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
            'validation_interval': train_cfg.get('validation_interval', 8),
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

        # Collect one update worth of experience
        states_buffer = []
        actions_buffer = []
        log_probs_buffer = []
        rewards_buffer = []
        dones_buffer = []
        values_buffer = []

        for _ in range(cfg.steps_per_env):
            with torch.no_grad():
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

            states_buffer.append(current_states)
            actions_buffer.append(actions)
            log_probs_buffer.append(log_probs.detach())
            rewards_buffer.append(shaped_rewards)
            dones_buffer.append(dones.clone())
            values_buffer.append(values)

            episode_rewards += rewards
            episode_rewards *= (~dones)
            states = next_states

        # Bootstrap value
        with torch.no_grad():
            states_t = states.float()
            _, features = self.policy.forward_with_features(states_t)
            next_value = self.value_head(features).squeeze(-1)

        # Compute GAE
        advantages, returns = compute_gae(
            rewards_buffer, values_buffer, dones_buffer,
            next_value, cfg.gamma, cfg.gae_lambda, device=self.device
        )

        # PPO update (this triggers backward pass lazy init)
        # Use cat to flatten (steps, envs) -> (steps * envs)
        all_states = torch.cat(states_buffer, dim=0)
        all_actions = torch.cat(actions_buffer, dim=0)
        all_log_probs = torch.cat(log_probs_buffer, dim=0)
        all_values = torch.cat(values_buffer, dim=0)
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
        # Shutdown async logger first (flushes any pending logs)
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
        if self.policy:
            del self.policy
            self.policy = None
        if self.value_head:
            del self.value_head
            self.value_head = None
        if self.optimizer:
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
        self.value_head = ValueHead(input_size=self.policy.hidden_size).to(self.device)
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
        with _COMPILE_LOCK:
            if cfg.compile_models and hasattr(torch, 'compile') and not is_tpu(self.device):
                try:
                    self.policy = torch.compile(self.policy)
                    self.value_head = torch.compile(self.value_head)
                    print(f"   [JIT] torch.compile enabled (warmup deferred)")
                except RuntimeError as e:
                    if "FX" in str(e) or "dynamo" in str(e).lower():
                        print(f"   [JIT] Warning: torch.compile failed ({e}). Using eager mode.")
                    else:
                        raise e
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=cfg.lr
        )
        
        
        # Reward computer - unified class handles both numpy and torch
        self.reward_computer = RewardComputer.create(
            cfg.mode, self.spec, cfg.gamma, 
            shaping_coef=train_cfg['shaping_food_attract'], 
            device=self.device
        )
        
        # Logger (skip in benchmark mode)
        if cfg.log_to_file and not cfg.benchmark_mode:
            self.logger = TrainingLogger(mode=cfg.mode, output_dir=cfg.log_dir)
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
        
        print(f"   Brain: {cfg.brain_type} (hidden={self.policy.hidden_size})")
        print(f"   AMP: {'Enabled' if cfg.use_amp else 'Disabled'}")
    
    def _training_loop(self):
        """Main training loop."""
        cfg = self.config
        v = cfg.hyper_verbose  # Verbose debug mode
        
        # Experience buffers
        states_buffer = []
        actions_buffer = []
        log_probs_buffer = []
        rewards_buffer = []
        dones_buffer = []
        values_buffer = []
        
        # Async logging buffers (new)
        finished_dones_buffer = []
        finished_rewards_buffer = []
        
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

        # Dashboard aggregation (per-update)
        # We track how many episodes finished during this collection phase
        update_episodes_count = 0
        update_reward_sum = 0.0
        update_food_sum = 0
        update_poison_sum = 0

        step_in_update = 0
        self.start_time = time.perf_counter()

        # Rolling window for sps calculation (all updates valid post-warmup)
        sps_window = []  # (steps, time) pairs for last 4 updates
        last_update_time = self.start_time
        last_update_steps = 0

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
                # States are already tensors on device
                states_t = states.float()
                
                if v:
                    _vprint(f"Step {self.total_steps}: running inference...", v)
                with autocast(device_type=self.device_type, enabled=cfg.use_amp):
                    # Use combined forward to avoid computing CNN features twice
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
            current_states = states.clone()
            
            next_states, rewards, dones = self.vec_env.step(actions)
            
            # Compute shaped rewards (torch-native, stays on GPU)
            shaped_rewards = self.reward_computer.compute(
                rewards, current_states, next_states, dones
            )
            
            # Store experience as tensors
            states_buffer.append(current_states)
            actions_buffer.append(actions) # Actions are new tensors from sampling
            log_probs_buffer.append(log_probs.detach())
            rewards_buffer.append(shaped_rewards) # Rewards are usually new tensors
            dones_buffer.append(dones.clone()) # Dones might be reused, but usually new. Clone to be safe?
            values_buffer.append(values)
            
            self.total_steps += cfg.n_envs
            episode_rewards += rewards  # Tensor addition
            
            self.profiler.tick("Env Step")
            
            # Track episode stats (defer logging to avoid Sync)
            # Snapshot current state for logging
            # We store full tensors to avoid boolean indexing syncs
            finished_dones_buffer.append(dones)
            finished_rewards_buffer.append(episode_rewards.clone())
            
            # Reset rewards for done agents (Sync-free)
            # episode_rewards = episode_rewards * (1 - dones)
            episode_rewards *= (~dones)
            
            states = next_states
            
            # PPO Update
            if len(states_buffer) >= cfg.steps_per_env:
                self.profiler.tick("Collection")
                
                # Get bootstrap value (keep on GPU)
                with torch.no_grad():
                    states_t = states.float()
                    _, features = self.policy.forward_with_features(states_t)
                    next_value = self.value_head(features).squeeze(-1)
                
                # Compute GAE
                advantages, returns = compute_gae(
                    rewards_buffer, values_buffer, dones_buffer,
                    next_value, cfg.gamma, cfg.gae_lambda, device=self.device
                )
                self.profiler.tick("GAE Calc")
                
                # PROCESS LOGGING (Batch Sync)
                if True: # Always runs now
                    # Stack buffers (steps, envs)
                    all_step_dones = torch.stack(finished_dones_buffer)
                    all_step_rewards = torch.stack(finished_rewards_buffer)
                    
                    # Find ANY completed episodes
                    # This is the ONLY sync per update
                    done_coords = all_step_dones.nonzero(as_tuple=False) # (N_events, 2) -> [step, env]
                    
                    if done_coords.shape[0] > 0:
                        # Extract data
                        steps_idx = done_coords[:, 0]
                        envs_idx = done_coords[:, 1]
                        
                        # Get rewards for the identified events
                        event_rewards = all_step_rewards[steps_idx, envs_idx]
                        
                        # Get food/poison stats (using vectorized lookup)
                        # Note: last_episode_food might have been overwritten if env finished twice.
                        # We accept this inaccuracy for performance.
                        event_food = self.vec_env.last_episode_food[envs_idx]
                        event_poison = self.vec_env.last_episode_poison[envs_idx]
                        
                        # Move to CPU for loop
                        cpu_rewards = event_rewards.cpu().numpy()
                        cpu_food = event_food.cpu().numpy()
                        cpu_poison = event_poison.cpu().numpy()
                        cpu_env_ids = envs_idx.cpu().numpy()
                        
                        # Log
                        for j, env_id in enumerate(cpu_env_ids):
                            self.episode_count += 1
                            r = float(cpu_rewards[j])
                            f = int(cpu_food[j])
                            p = int(cpu_poison[j])
                            
                            update_episodes_count += 1
                            update_reward_sum += r
                            update_food_sum += f
                            update_poison_sum += p
                            
                            if r > self.best_reward:
                                self.best_reward = r
                            
                            if self.logger:
                                grid_id = self.vec_env.grid_indices[env_id]
                                if isinstance(grid_id, torch.Tensor):
                                    grid_id = grid_id.item()
                                food_density = self.vec_env.grid_food_counts[grid_id]
                                if isinstance(food_density, torch.Tensor):
                                    food_density = food_density.item()
                                avg_food = (self.min_food + self.max_food) / 2
                                self.logger.log_episode(
                                    episode=self.episode_count,
                                    reward=r,
                                    length=500,
                                    food_eaten=f,
                                    poison_eaten=p,
                                    food_density=food_density,
                                    curriculum_progress=(avg_food - self.min_food) / (self.max_food - self.min_food + 1e-8),
                                    action_prob_std=0.0,
                                )
                
                # Flatten and prepare buffers for PPO (all tensors stay on GPU)
                # Note: Buffers always contain tensors now (converted during collection)
                all_states = torch.cat(states_buffer, dim=0)
                all_actions = torch.cat(actions_buffer, dim=0)
                all_log_probs = torch.cat(log_probs_buffer, dim=0)
                all_old_values = torch.cat(values_buffer, dim=0)
                
                # Returns and advantages from compute_gae (always tensors now)
                all_returns = returns.flatten()
                all_advantages = advantages.flatten()
                
                if v:
                    _vprint(f"   [DEBUG] Rewards: mean={torch.cat(rewards_buffer).mean():.4f}, std={torch.cat(rewards_buffer).std():.4f}", v)
                    _vprint(f"   [DEBUG] Advantages: mean={all_advantages.mean():.4f}, std={all_advantages.std():.4f}", v)
                    _vprint(f"   [DEBUG] Returns: mean={all_returns.mean():.4f}, std={all_returns.std():.4f}", v)
                
                # Normalize advantages
                all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
                
                # PPO update - tensors go in directly, no CPU sync needed
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
                
                # Sync device - critical for TPU to force XLA execution
                sync_device(self.device)
                self.profiler.tick("PPO Update")
                
                self.update_count += 1
                
                # Clear buffers
                states_buffer = []
                actions_buffer = []
                log_probs_buffer = []
                rewards_buffer = []
                dones_buffer = []
                values_buffer = []

                # Clear async logging buffers
                finished_dones_buffer = []
                finished_rewards_buffer = []
                
                # Progress - calculate three sps metrics
                now = time.perf_counter()
                update_steps = self.total_steps - last_update_steps
                update_time = now - last_update_time

                # Instantaneous sps (this update only)
                sps_instant = update_steps / update_time if update_time > 0 else 0

                # Add to rolling window
                sps_window.append((update_steps, update_time))
                if len(sps_window) > 4:
                    sps_window.pop(0)

                # Rolling sps (last 4 updates)
                window_steps = sum(s for s, t in sps_window)
                window_time = sum(t for s, t in sps_window)
                sps_rolling = window_steps / window_time if window_time > 0 else 0

                # Global sps (total steps / total elapsed)
                elapsed = now - self.start_time
                sps_global = self.total_steps / elapsed if elapsed > 0 else 0
                
                # Prepare episode stats
                ep_stats = None
                if update_episodes_count > 0:
                    ep_stats = {
                        'reward': update_reward_sum / update_episodes_count,
                        'food': update_food_sum / update_episodes_count,
                        'poison': update_poison_sum / update_episodes_count
                    }
                
                # SYNC POINT: Convert GPU tensors to CPU floats (single sync)
                # This keeps all CUDA ops on main thread; background thread only does I/O
                policy_loss_f = policy_loss.item()
                value_loss_f = value_loss.item()
                entropy_f = entropy.item()
                explained_var_f = explained_var.item()
                action_probs = F.softmax(last_logits, dim=1)[0].cpu().numpy().tolist()
                
                # Run validation episodes periodically
                val_metrics = None
                if cfg.validation_interval > 0 and self.update_count % cfg.validation_interval == 0:
                    val_metrics = self._run_validation_episodes()
                
                # ASYNC LOGGING - queue CPU data only, no GPU access in background
                log_payload = LogPayload(
                    policy_loss=policy_loss_f,
                    value_loss=value_loss_f,
                    entropy=entropy_f,
                    explained_var=explained_var_f,
                    action_probs=action_probs,
                    update_count=self.update_count,
                    total_steps=self.total_steps,
                    best_reward=self.best_reward,
                    mode=cfg.mode,
                    episode_stats=ep_stats,
                    profiler_summary=self.profiler.summary(),
                    sps_instant=sps_instant,
                    sps_rolling=sps_rolling,
                    sps_global=sps_global,
                    validation_metrics=val_metrics,
                )
                self.async_logger.log_update(log_payload)
                self.profiler.reset()
                
                # Reset episode aggregators
                update_episodes_count = 0
                update_reward_sum = 0.0
                update_food_sum = 0
                update_poison_sum = 0
                
                last_update_time = now
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
                'best_reward': self.best_reward,
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
            print(f"  Best reward:    {self.best_reward:.1f}")
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
                self.logger.finalize(best_efficiency=self.best_reward, final_model_path=cfg.output_path)

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
            'best_reward': self.best_reward,
            'model_path': None if cfg.benchmark_mode else cfg.output_path,
            'aborted': False,
        }
