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
from goodharts.behaviors.brains import create_brain
from goodharts.behaviors.action_space import num_actions
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


@dataclass
class PPOConfig:
    """
    Configuration for PPO training.
    
    Use PPOConfig.from_config() to load defaults from config.toml,
    with CLI arguments as optional overrides.
    """
    mode: str = 'ground_truth'
    brain_type: str = 'base_cnn'
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
    log_dir: str = 'logs'
    use_amp: bool = False
    compile_models: bool = True
    tensorboard: bool = False
    skip_warmup: bool = False
    use_torch_env: bool = True
    hyper_verbose: bool = False
    clean_cache: bool = False
    
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
            print("\nTraining interrupted by user")
        
        return self._finalize()
    
    def _setup(self):
        """Initialize environment, networks, and logging."""
        cfg = self.config
        
        # Seeding
        seed = int(time.time())
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        v = cfg.hyper_verbose
        _vprint("_setup() starting", v)
        
        print(f"\n[PPO] Starting training: {cfg.mode}")
        print(f"   Device: {self.device}, Envs: {cfg.n_envs}")
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
        n_actions = num_actions(1)
        
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
        _vprint("Networks created and moved to device", v)

        # AMP
        self.device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.scaler = GradScaler(enabled=cfg.use_amp) if cfg.use_amp else None

        # Compile models for extra speed if torch.compile is available (PyTorch 2.0+)
        # Use lock to serialize compilation - Dynamo's global state is not thread-safe
        # Skip on TPU - XLA uses its own JIT compilation
        with _COMPILE_LOCK:
            if cfg.compile_models and hasattr(torch, 'compile') and not is_tpu(self.device):
                
                # Keep references to originals in case compilation fails
                orig_policy = self.policy
                orig_value_head = self.value_head
                try:
                    # Note: NOT using dynamic=True because we warmup with exact batch size
                    # Fixed-shape kernels should cache properly across runs
                    self.policy = torch.compile(self.policy)
                    self.value_head = torch.compile(self.value_head)
                    
                    # Skip explicit warmup if global warmup was already done (parallel training)
                    if not cfg.skip_warmup:
                        # Explicit JIT Warmup - MUST use actual training batch size
                        # Otherwise torch.compile recompiles on first real batch (85s+ forward, 270s+ backward!)
                        warmup_batch = (cfg.n_envs * cfg.steps_per_env) // cfg.n_minibatches
                        
                        print(f"   [JIT] Warming up torch.compile (batch={warmup_batch})...", flush=True)
                        warmup_start = time.time()
                        
                        dummy_obs = torch.zeros(
                            (warmup_batch, self.vec_env.n_channels, self.vec_env.view_size, self.vec_env.view_size), 
                            device=self.device, requires_grad=False
                        )
                        dummy_actions = torch.zeros(warmup_batch, dtype=torch.long, device=self.device)
                        dummy_returns = torch.zeros(warmup_batch, device=self.device)
                        
                        # Forward warmup (with autocast if using AMP)
                        with autocast(device_type=self.device_type, enabled=cfg.use_amp):
                            logits = self.policy(dummy_obs)
                            features = self.policy.get_features(dummy_obs)
                            values = self.value_head(features).squeeze(-1)
                            
                            # Compute dummy loss (triggers backward graph compilation)
                            dist = Categorical(logits=logits, validate_args=False)
                            log_probs = dist.log_prob(dummy_actions)
                            dummy_loss = -log_probs.mean() + F.mse_loss(values, dummy_returns)
                        
                        # Backward warmup (compiles gradient kernels)
                        dummy_loss.backward()
                        
                        # Clear gradients and CUDA cache
                        self.policy.zero_grad()
                        self.value_head.zero_grad()
                        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
                        
                        warmup_time = time.time() - warmup_start
                        print(f"   [JIT] Compilation complete ({warmup_time:.1f}s)", flush=True)
                    
                except RuntimeError as e:
                    # Fallback if compilation fails (common in complex multi-threaded setups)
                    if "FX" in str(e) or "dynamo" in str(e).lower():
                        print(f"   [JIT] Warning: torch.compile failed ({e}). Reverting to eager mode.", flush=True)
                        self.policy = orig_policy
                        self.value_head = orig_value_head
                    else:
                        # Re-raise unexpected errors
                        raise e
        
        # Eager warmup (even without torch.compile) for cuDNN algorithm selection
        # This warms up cuDNN benchmark mode and autograd graph construction
        if not cfg.skip_warmup:
            warmup_batch = (cfg.n_envs * cfg.steps_per_env) // cfg.n_minibatches
            print(f"   [Warmup] Running eager warmup (cuDNN + autograd)...", flush=True)
            warmup_start = time.time()
            
            dummy_obs = torch.zeros(
                (warmup_batch, self.vec_env.n_channels, self.vec_env.view_size, self.vec_env.view_size), 
                device=self.device, requires_grad=False
            )
            dummy_actions = torch.zeros(warmup_batch, dtype=torch.long, device=self.device)
            dummy_returns = torch.zeros(warmup_batch, device=self.device)
            
            # Forward + backward warmup
            with autocast(device_type=self.device_type, enabled=cfg.use_amp):
                logits = self.policy(dummy_obs)
                features = self.policy.get_features(dummy_obs)
                values = self.value_head(features).squeeze(-1)
                
                dist = Categorical(logits=logits, validate_args=False)
                log_probs = dist.log_prob(dummy_actions)
                dummy_loss = -log_probs.mean() + F.mse_loss(values, dummy_returns)
            
            dummy_loss.backward()
            self.policy.zero_grad()
            self.value_head.zero_grad()
            torch.cuda.empty_cache() if self.device.type == 'cuda' else None
            
            print(f"   [Warmup] Complete ({time.time() - warmup_start:.1f}s)", flush=True)
        
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
        
        # Logger
        if cfg.log_to_file:
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
        
        # Profiler
        self.profiler = Profiler(self.device)
        
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
        
        # Dashboard aggregation (per-update)
        # We track how many episodes finished during this collection phase
        update_episodes_count = 0
        update_reward_sum = 0.0
        update_food_sum = 0
        update_poison_sum = 0
        
        step_in_update = 0
        start_time = time.perf_counter()
        
        # Rolling window for sps calculation (exclude compilation warmup)
        sps_window = []  # (steps, time) pairs for last 4 updates
        last_update_time = start_time
        last_update_steps = 0
        
        while self.total_steps < cfg.total_timesteps:
            self.profiler.start()
            
            # Check stop signal
            if os.path.exists('.training_stop_signal'):
                print(f"\n[PPO] Stop signal received!")
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
                    logits = self.policy(states_t)
                    features = self.policy.get_features(states_t)
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
                    features = self.policy.get_features(states_t)
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
                
                # Progress - use rolling window for sps (excludes compilation warmup)
                now = time.perf_counter()
                update_steps = self.total_steps - last_update_steps
                update_time = now - last_update_time
                sps_window.append((update_steps, update_time))
                if len(sps_window) > 4:
                    sps_window.pop(0)
                
                # Calculate sps from window
                window_steps = sum(s for s, t in sps_window)
                window_time = sum(t for s, t in sps_window)
                sps = window_steps / window_time if window_time > 0 else 0
                
                # Prepare episode stats
                ep_stats = None
                if update_episodes_count > 0:
                    ep_stats = {
                        'reward': update_reward_sum / update_episodes_count,
                        'food': update_food_sum / update_episodes_count,
                        'poison': update_poison_sum / update_episodes_count
                    }
                
                # ASYNC LOGGING - queue payload, no GPU sync or I/O here
                # All .item() calls, file writes, console prints happen in background thread
                log_payload = LogPayload(
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    entropy=entropy,
                    explained_var=explained_var,
                    action_probs_tensor=last_logits.detach(),  # Keep on GPU
                    update_count=self.update_count,
                    total_steps=self.total_steps,
                    best_reward=self.best_reward,
                    mode=cfg.mode,
                    episode_stats=ep_stats,
                    profiler_summary=self.profiler.summary(),
                    sps=sps,
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
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{self.config.mode}_u{self.update_count}_s{self.total_steps}.pth"
        )
        torch.save({
            'update': self.update_count,
            'total_steps': self.total_steps,
            'policy_state_dict': self.policy.state_dict(),
            'value_head_state_dict': self.value_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_reward': self.best_reward,
        }, checkpoint_path)
        print(f"   Checkpoint saved: {checkpoint_path}")
    
    def _finalize(self) -> dict:
        """Save final model and return summary."""
        cfg = self.config
        
        # Shutdown async logger first - flushes all queued logs
        if self.async_logger:
            self.async_logger.shutdown()
        
        torch.save(self.policy.state_dict(), cfg.output_path)
        print(f"\n[PPO] Training complete: {cfg.output_path}")
        
        if self.logger:
            self.logger.finalize(best_efficiency=self.best_reward, final_model_path=cfg.output_path)
        
        # Close TensorBoard writer to flush all data
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.dashboard:
            self.dashboard.update(cfg.mode, 'finished', None)
        
        return {
            'mode': cfg.mode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'model_path': cfg.output_path,
        }
