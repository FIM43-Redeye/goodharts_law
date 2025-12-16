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


# Enable persistent kernel cache for torch.compile
# This avoids ~300s recompilation on every run
# Cache is GPU-specific (device name + compute capability in cache key)
_CACHE_DIR = os.path.expanduser("~/.cache/torch_inductor")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _CACHE_DIR)


# Lock to serialize torch.compile() calls across threads.
# Dynamo has global state that is not thread-safe during compilation.
# This only affects startup; compiled models run in parallel fine.
_COMPILE_LOCK = threading.Lock()
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.distributions import Categorical
from dataclasses import dataclass, field
from typing import Optional

from goodharts.configs.default_config import get_config
from goodharts.config import get_training_config
from goodharts.utils.device import get_device, apply_system_optimizations
from goodharts.behaviors.brains import create_brain
from goodharts.behaviors.action_space import num_actions
from goodharts.environments.vec_env import create_vec_env
from goodharts.training.train_log import TrainingLogger

from .models import Profiler, ValueHead
from .algorithms import compute_gae, ppo_update
from goodharts.modes import RewardComputer


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    mode: str = 'ground_truth'
    brain_type: str = 'base_cnn'
    n_envs: int = 64
    total_timesteps: int = 100_000
    lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    k_epochs: int = 4
    steps_per_env: int = 128
    n_minibatches: int = 4
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    output_path: str = 'models/ppo_agent.pth'
    log_to_file: bool = True
    log_dir: str = 'logs'
    use_amp: bool = False
    compile_models: bool = True  # Set to False when training multiple models in parallel


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
        
        print(f"\n[PPO] Starting training: {cfg.mode}")
        print(f"   Device: {self.device}, Envs: {cfg.n_envs}")
        
        # Apply hardware optimizations
        apply_system_optimizations(self.device, verbose=True)
        
        # Load configs
        sim_config = get_config()
        train_cfg = get_training_config()
        
        # Observation spec
        self.spec = sim_config['get_observation_spec'](cfg.mode)
        n_actions = num_actions(1)
        
        # Environment
        self.vec_env = create_vec_env(n_envs=cfg.n_envs, obs_spec=self.spec)
        print(f"   View: {self.vec_env.view_size}x{self.vec_env.view_size}, Channels: {self.vec_env.n_channels}")
        
        # Networks
        self.policy = create_brain(cfg.brain_type, self.spec, output_size=n_actions).to(self.device)
        self.value_head = ValueHead(input_size=self.policy.hidden_size).to(self.device)

        # AMP
        self.device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.scaler = GradScaler(enabled=cfg.use_amp) if cfg.use_amp else None

        # Compile models for extra speed if torch.compile is available (PyTorch 2.0+)
        # Use lock to serialize compilation - Dynamo's global state is not thread-safe
        with _COMPILE_LOCK:
            if cfg.compile_models and hasattr(torch, 'compile'):
                # Keep references to originals in case compilation fails
                orig_policy = self.policy
                orig_value_head = self.value_head
                
                try:
                    self.policy = torch.compile(self.policy, dynamic=True)
                    self.value_head = torch.compile(self.value_head, dynamic=True)
                    
                    # Explicit JIT Warmup - MUST use actual training batch size
                    # Otherwise torch.compile recompiles on first real batch (85s+ forward, 270s+ backward!)
                    warmup_batch = (cfg.n_envs * cfg.steps_per_env) // cfg.n_minibatches
                    
                    print(f"   [JIT] Warming up torch.compile (batch={warmup_batch})...")
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
                        values = self.value_head(features).squeeze()
                        
                        # Compute dummy loss (triggers backward graph compilation)
                        from torch.distributions import Categorical
                        dist = Categorical(logits=logits)
                        log_probs = dist.log_prob(dummy_actions)
                        dummy_loss = -log_probs.mean() + F.mse_loss(values, dummy_returns)
                    
                    # Backward warmup (compiles gradient kernels)
                    dummy_loss.backward()
                    
                    # Clear gradients and CUDA cache
                    self.policy.zero_grad()
                    self.value_head.zero_grad()
                    torch.cuda.empty_cache() if self.device.type == 'cuda' else None
                    
                    print(f"   [JIT] Compilation complete ({time.time() - warmup_start:.1f}s)")
                    
                except RuntimeError as e:
                    # Fallback if compilation fails (common in complex multi-threaded setups)
                    if "FX" in str(e) or "dynamo" in str(e).lower():
                        print(f"   [JIT] Warning: torch.compile failed ({e}). Reverting to eager mode.")
                        self.policy = orig_policy
                        self.value_head = orig_value_head
                    else:
                        # Re-raise unexpected errors
                        raise e
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=cfg.lr
        )
        
        
        # Reward computer
        self.reward_computer = RewardComputer.create(cfg.mode, self.spec, cfg.gamma)
        
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
        
        # Profiler
        self.profiler = Profiler(self.device)
        
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
        
        # Experience buffers
        states_buffer = []
        actions_buffer = []
        log_probs_buffer = []
        rewards_buffer = []
        dones_buffer = []
        values_buffer = []
        
        # Initial state
        states = self.vec_env.reset()
        self.reward_computer.initialize(states)
        episode_rewards = np.zeros(cfg.n_envs)
        
        # Dashboard aggregation (per-update)
        # We track how many episodes finished during this collection phase
        update_episodes_count = 0
        update_reward_sum = 0.0
        update_food_sum = 0
        update_poison_sum = 0
        
        start_time = time.perf_counter()
        
        while self.total_steps < cfg.total_timesteps:
            self.profiler.start()
            
            # Check stop signal
            if os.path.exists('.training_stop_signal'):
                print(f"\n[PPO] Stop signal received!")
                break
            
            # Collect experience
            with torch.no_grad():
                states_t = torch.from_numpy(states).float().to(self.device)
                with autocast(device_type=self.device_type, enabled=cfg.use_amp):
                    logits = self.policy(states_t)
                    features = self.policy.get_features(states_t)
                    values = self.value_head(features).squeeze().cpu().numpy()
                    
                    dist = Categorical(logits=logits)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                    action_probs = F.softmax(logits, dim=1).cpu().numpy()
            
            self.profiler.tick("Inference")
            actions_np = actions.cpu().numpy()
            
            # Environment step
            next_states, rewards, dones = self.vec_env.step(actions_np)
            self.profiler.tick("Env Step")
            
            # Compute shaped rewards
            shaped_rewards = self.reward_computer.compute(
                rewards, states, next_states, dones
            )
            
            # Store experience
            states_buffer.append(states)
            actions_buffer.append(actions_np)
            log_probs_buffer.append(log_probs.detach())
            rewards_buffer.append(shaped_rewards)
            dones_buffer.append(dones)
            values_buffer.append(values)
            
            self.total_steps += cfg.n_envs
            episode_rewards += rewards
            
            # Track episodes
            if dones.any():
                done_envs = np.where(dones)[0]
                for i in done_envs:
                    self.episode_count += 1
                    
                    # Accumulate for update stats
                    r = episode_rewards[i]
                    f = self.vec_env.last_episode_food[i]
                    p = self.vec_env.last_episode_poison[i]
                    
                    update_episodes_count += 1
                    update_reward_sum += r
                    update_food_sum += f
                    update_poison_sum += p
                    
                    if r > self.best_reward:
                        self.best_reward = r
                        
                    # Logger still logs every episode to CSV for detailed analysis
                    if self.logger:
                        grid_id = self.vec_env.grid_indices[i]
                        food_density = self.vec_env.grid_food_counts[grid_id]
                        avg_food = (self.min_food + self.max_food) / 2
                        self.logger.log_episode(
                            episode=self.episode_count,
                            reward=r,
                            length=500,
                            food_eaten=f,
                            poison_eaten=p,
                            food_density=food_density,
                            curriculum_progress=(avg_food - self.min_food) / (self.max_food - self.min_food + 1e-8),
                            action_prob_std=np.std(action_probs[i]),
                        )
                    episode_rewards[i] = 0.0
            
            states = next_states
            
            # PPO Update
            if len(states_buffer) >= cfg.steps_per_env:
                self.profiler.tick("Collection")
                
                # Get bootstrap value
                with torch.no_grad():
                    states_t = torch.from_numpy(states).float().to(self.device)
                    features = self.policy.get_features(states_t)
                    next_value = self.value_head(features).squeeze().cpu().numpy()
                
                # Compute GAE
                advantages, returns = compute_gae(
                    rewards_buffer, values_buffer, dones_buffer,
                    next_value, cfg.gamma, cfg.gae_lambda
                )
                self.profiler.tick("GAE Calc")
                
                # Flatten buffers
                all_states = np.concatenate(states_buffer, axis=0)
                all_actions = np.concatenate(actions_buffer, axis=0)
                all_log_probs = torch.cat(log_probs_buffer, dim=0).cpu().numpy()
                all_returns = returns.flatten()
                all_advantages = advantages.flatten()
                all_old_values = np.stack(values_buffer).flatten()
                
                # Normalize advantages
                all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
                
                # PPO update
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
                self.profiler.tick("PPO Update")
                
                self.update_count += 1
                
                # Clear buffers
                states_buffer = []
                actions_buffer = []
                log_probs_buffer = []
                rewards_buffer = []
                dones_buffer = []
                values_buffer = []
                
                # Logging
                if self.logger:
                    self.logger.log_update(
                        update_num=self.update_count,
                        total_steps=self.total_steps,
                        policy_loss=policy_loss,
                        value_loss=value_loss,
                        entropy=entropy,
                        explained_variance=explained_var,
                        action_probs=action_probs[0].tolist()
                    )
                
                # Dashboard update (Unified!)
                if self.dashboard:
                    # Prepare episode stats if any finished
                    ep_stats = None
                    if update_episodes_count > 0:
                        ep_stats = {
                            'reward': update_reward_sum / update_episodes_count,
                            'food': update_food_sum / update_episodes_count,
                            'poison': update_poison_sum / update_episodes_count
                        }
                    
                    # Unified payload
                    payload = {
                        'ppo': (policy_loss, value_loss, entropy, action_probs[0].tolist(), explained_var),
                        'episodes': ep_stats,
                        'steps': self.total_steps
                    }
                    self.dashboard.update(cfg.mode, 'update', payload)
                    
                    # Reset aggregators
                    update_episodes_count = 0
                    update_reward_sum = 0.0
                    update_food_sum = 0
                    update_poison_sum = 0
                
                # Progress
                elapsed = time.perf_counter() - start_time
                sps = self.total_steps / elapsed
                print(f"   [{cfg.mode}] Step {self.total_steps:,}: {sps:,.0f} sps | Best R={self.best_reward:.0f} | Ent={entropy:.3f}")
                print(f"   [Profile] {self.profiler.summary()}")
                self.profiler.reset()
                
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
        
        torch.save(self.policy.state_dict(), cfg.output_path)
        print(f"\n[PPO] Training complete: {cfg.output_path}")
        
        if self.logger:
            self.logger.finalize(best_efficiency=self.best_reward, final_model_path=cfg.output_path)
        
        if self.dashboard:
            self.dashboard.update(cfg.mode, 'finished', None)
        
        return {
            'mode': cfg.mode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'model_path': cfg.output_path,
        }
