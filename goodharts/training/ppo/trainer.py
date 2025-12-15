"""
PPO Trainer - Main training orchestrator.

Provides a clean, subclassable interface for PPO training.
Can be extended for multi-agent scenarios.
"""
import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.distributions import Categorical
from dataclasses import dataclass, field
from typing import Optional

from goodharts.configs.default_config import get_config
from goodharts.config import get_training_config
from goodharts.utils.device import get_device
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
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    output_path: str = 'models/ppo_agent.pth'
    log_to_file: bool = True
    log_dir: str = 'logs'
    use_amp: bool = False


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
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=cfg.lr
        )
        
        # AMP
        self.device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.scaler = GradScaler(enabled=cfg.use_amp) if cfg.use_amp else None
        
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
                entropy_coef=cfg.entropy_coef,
                vectorized=True,
            )
        
        # Profiler
        self.profiler = Profiler(self.device)
        
        # Curriculum settings
        self.min_food = train_cfg.get('min_food', 50)
        self.max_food = train_cfg.get('max_food', 200)
        self.min_poison = train_cfg.get('min_poison', 20)
        self.max_poison = train_cfg.get('max_poison', 100)
        self.vec_env.initial_food = (self.min_food + self.max_food) // 2
        
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
        
        # Dashboard aggregation (per-update, not per-episode)
        update_episodes = 0
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
                    # Track for this update's aggregation
                    update_episodes += 1
                    update_reward_sum += episode_rewards[i]
                    update_food_sum += self.vec_env.last_episode_food[i]
                    update_poison_sum += self.vec_env.last_episode_poison[i]
                    
                    if episode_rewards[i] > self.best_reward:
                        self.best_reward = episode_rewards[i]
                    if self.logger:
                        self.logger.log_episode(
                            episode=self.episode_count,
                            reward=episode_rewards[i],
                            length=500,
                            food_eaten=self.vec_env.last_episode_food[i],
                            poison_eaten=self.vec_env.last_episode_poison[i],
                            food_density=self.vec_env.initial_food,
                            curriculum_progress=(self.vec_env.initial_food - self.min_food) / (self.max_food - self.min_food + 1e-8),
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
                    scaler=self.scaler,
                )
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
                
                # Curriculum: randomize food/poison
                self.vec_env.initial_food = np.random.randint(self.min_food, self.max_food + 1)
                self.vec_env.poison_count = np.random.randint(self.min_poison, self.max_poison + 1)
                
                # Dashboard update (aggregated per-update, not per-episode)
                if self.dashboard:
                    self.dashboard.update(cfg.mode, 'ppo', (policy_loss, value_loss, entropy, action_probs[0], explained_var))
                    # Send aggregated episode stats for this update
                    if update_episodes > 0:
                        avg_reward = update_reward_sum / update_episodes
                        avg_food = update_food_sum / update_episodes
                        avg_poison = update_poison_sum / update_episodes
                        self.dashboard.update(cfg.mode, 'episode', (
                            self.episode_count,
                            avg_reward,
                            500,
                            0,  # legacy
                            self.vec_env.initial_food,
                            avg_food,
                            avg_poison,
                        ))
                    # Reset accumulators for next update
                    update_episodes = 0
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
