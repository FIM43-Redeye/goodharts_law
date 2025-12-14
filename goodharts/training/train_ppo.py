"""
PPO (Proximal Policy Optimization) training with GAE-Lambda.

Uses BaseCNN for the policy network (compatible with LearnedBehavior).
Adds a separate value head for the critic.

Features:
- Fully Vectorized (VecEnv)
- Generalized Advantage Estimation (GAE)
- Proper Value Bootstrapping
- Live Dashboard
- Structured Logging
- Curriculum Learning
- Detailed Profiling
"""
import os
import sys

# Ensure config is loaded and env vars set BEFORE torch is imported/initialized
# This allows HSA_OVERRIDE_GFX_VERSION to take effect for ROCm
try:
    from goodharts.config import get_config
    get_config()
except ImportError:
    pass # Config might not be importable yet during some setup phases

import threading
import time
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class Profiler:
    """Simple accumulated timer for profiling loop components."""
    def __init__(self, device=None):
        self.times = {}
        self.counts = {}
        self.last_t = 0
        if device is None:
            device = get_device()
        # Only synchronize if we are actually using a CUDA device
        self.sync_cuda = (device.type == 'cuda')

    def start(self):
        if self.sync_cuda: torch.cuda.synchronize()
        self.last_t = time.perf_counter()

    def tick(self, name):
        if self.sync_cuda: torch.cuda.synchronize()
        now = time.perf_counter()
        dt = now - self.last_t
        self.times[name] = self.times.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1
        self.last_t = now
    
    def reset(self):
        self.times = {}
        self.counts = {}

    def summary(self):
        total = sum(self.times.values())
        if total == 0: return "No data"
        parts = []
        # Sort by duration
        for k, v in sorted(self.times.items(), key=lambda x: x[1], reverse=True):
            pct = v / total * 100
            parts.append(f"{k}: {v:.2f}s ({pct:.0f}%)")
        return " | ".join(parts)

# Global synchronization for stop signal
_TRAINING_LOCK = threading.Lock()
_TRAINING_COUNTER = 0

from goodharts.configs.default_config import TRAINING_DEFAULTS
from goodharts.config import get_training_config
from goodharts.utils.device import get_device
from goodharts.behaviors.brains import create_brain, get_brain_names
from goodharts.behaviors.action_space import build_action_space, num_actions


class ValueHead(nn.Module):
    """Simple value head that attaches to BaseCNN features."""
    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
    
    def forward(self, features):
        return self.fc(features)



def _get_potentials_np(states):
    """
    Calculate potential-based shaping using INVERSE distance to nearest visible food.
    Potential = 0.5 / Distance. (Max potential 0.5 at dist=1).
    If no food visible, Potential = 0.
    
    Why this curve?
    - Eating food (dist 1 -> no food) loses 0.5 potential.
    - Gaining food reward (+1.0) outweighs this loss (Net +0.5).
    - If we used linear distance (e.g. -dist), the drop from -1 to -15 would be -14,
      overwhelming the +1 reward and teaching agents to starve!
    """
    n, c, h, w = states.shape
    # Cache dist map
    if not hasattr(_get_potentials_np, 'dist_map') or _get_potentials_np.dist_map.shape != (h, w):
        y, x = np.ogrid[:h, :w]
        center = (h//2, w//2)
        # Distance map (euclidean)
        dist = np.sqrt((y - center[0])**2 + (x - center[1])**2)
        # Avoid division by zero at center (though food can't be at center 0 for agent)
        # Actually food can be at dist 1.
        dist[center] = 1e-6 # Just in case
        _get_potentials_np.dist_map = dist
    
    dist_map = _get_potentials_np.dist_map
    
    # Channel 2 is Food
    food = states[:, 2, :, :] > 0.5
    
    # Vectorized min distance
    potentials = np.zeros(n, dtype=np.float32)
    
    # Find envs with food
    has_food = food.any(axis=(1, 2))
    if has_food.any():
        # Mask distance map where food is present
        masked_dist = np.where(food[has_food], dist_map, np.inf)
        min_dists = masked_dist.reshape(masked_dist.shape[0], -1).min(axis=1)
        
        # Inverse potential: 0.5 / dist
        # dist is at least 1.0 (grid cells)
        potentials[has_food] = 0.5 / (min_dists + 1e-6)
        
    return potentials


def train_ppo(*args, **kwargs):
    """
    Wrapper for PPO training that handles stop signal synchronization.
    Ensures the last finishing thread removes the stop signal file.
    """
    global _TRAINING_COUNTER
    
    with _TRAINING_LOCK:
        _TRAINING_COUNTER += 1
        
    try:
        _train_ppo_core(*args, **kwargs)
    finally:
        with _TRAINING_LOCK:
            _TRAINING_COUNTER -= 1
            remaining = _TRAINING_COUNTER
            
        if remaining == 0 and os.path.exists('.training_stop_signal'):
            print(f"\\n🧹 Last thread finished. Removing stop signal.")
            try:
                os.remove('.training_stop_signal')
            except OSError:
                pass

def _train_ppo_core(
    mode: str = 'ground_truth',
    brain_type: str = 'base_cnn',
    n_envs: int = 64,
    total_timesteps: int = 100_000,
    lr: float = 5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    eps_clip: float = 0.2,
    k_epochs: int = 4,
    steps_per_env: int = 128,
    entropy_coef: float = 0.0,
    value_coef: float = 0.5,
    output_path: str = 'models/ppo_agent.pth',
    device: torch.device = None,
    dashboard = None,  # Dashboard for live visualization
    log_to_file: bool = True,
    log_dir: str = 'logs',
):
    """
    Vectorized PPO training with GAE (Core Implementation).
    """
    from goodharts.environments.vec_env import create_vec_env
    from goodharts.training.train_log import TrainingLogger
    
    if device is None:
        device = get_device()
    
    print(f"\n🚀 Vectorized PPO Training (w/ GAE): {mode} on {device}")
    
    # Ensure seeding is effectively random unless specified otherwise
    import time
    seed = int(time.time())
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"   Random Seed: {seed}")
    print(f"   {n_envs} parallel environments")
    
    config = get_config()
    train_cfg = get_training_config()
    
    # Get observation spec
    spec = config['get_observation_spec'](mode)
    n_actions = num_actions(1)
    
    # Create vectorized environment
    vec_env = create_vec_env(n_envs=n_envs, obs_spec=spec)
    print(f"   View: {vec_env.view_size}x{vec_env.view_size}, Channels: {vec_env.n_channels}")
    print(f"   Loop Mode: {vec_env.loop} (Config: {config.get('WORLD_LOOP')})")
    
    # Compute auto-scale so max reward fits in clamp range [-5, 5]
    CellType = config['CellType']
    max_raw_reward = max(CellType.FOOD.energy_reward, CellType.POISON.energy_penalty)
    auto_reward_scale = 5.0 / max_raw_reward
    print(f"   Auto reward scale: {auto_reward_scale:.4f} (max_signal={max_raw_reward})")
    
    # Create policy and value head
    policy = create_brain(brain_type, spec, output_size=n_actions).to(device)
    value_head = ValueHead(input_size=policy.hidden_size).to(device)
    
    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_head.parameters()),
        lr=lr
    )
    
    # LR Scheduler (optional, disabled by default)
    lr_scheduler = None
    if train_cfg.get('lr_decay', False):
        total_updates = total_timesteps // (n_envs * steps_per_env)
        lr_decay_factor = train_cfg.get('lr_decay_factor', 0.1)
        # Linear decay from lr to lr * factor
        lr_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=lr_decay_factor, 
            total_iters=total_updates
        )
        print(f"   LR Decay: {lr} → {lr * lr_decay_factor} over {total_updates} updates")
    
    # Checkpoint settings
    checkpoint_interval = train_cfg.get('checkpoint_interval', 0)
    checkpoint_dir = os.path.dirname(output_path) or 'models'
    if checkpoint_interval > 0:
        print(f"   Checkpoints: Every {checkpoint_interval} updates → {checkpoint_dir}/")
    
    print(f"   Brain: {brain_type} (hidden_size={policy.hidden_size})")
    print(f"   Entropy: {entropy_coef}, GAE Lambda: {gae_lambda}")
    
    # Initialize logging
    logger = None
    if log_to_file:
        logger = TrainingLogger(mode=mode, output_dir=log_dir)
        logger.set_hyperparams(
            mode=mode,
            brain_type=brain_type,
            n_envs=n_envs,
            total_timesteps=total_timesteps,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda, # Record lambda
            eps_clip=eps_clip,
            k_epochs=k_epochs,
            steps_per_env=steps_per_env,
            entropy_coef=entropy_coef,
            vectorized=True,
        )
    
    # Experience buffers
    states_buffer = []
    actions_buffer = []
    log_probs_buffer = []
    rewards_buffer = []
    dones_buffer = []
    values_buffer = [] # Store values for GAE
    
    # Tracking
    total_steps = 0
    update_count = 0
    best_reward = float('-inf')
    episode_rewards = np.zeros(n_envs)
    episode_count = 0
    
    # Initial observation
    states = vec_env.reset()
    
    import time
    start_time = time.perf_counter()
    
    # Curriculum settings - configurable ranges for robustness
    min_food = train_cfg.get('min_food', 50)
    max_food = train_cfg.get('max_food', 200)
    min_poison = train_cfg.get('min_poison', 20)
    max_poison = train_cfg.get('max_poison', 100)
    
    vec_env.initial_food = (min_food + max_food) // 2  # Start at midpoint
    
    # Initialize potentials for shaping
    prev_potentials = _get_potentials_np(states)

    profiler = Profiler(device=device)


    while total_steps < total_timesteps:
        profiler.start()
        # Check for stop signal (from dashboard)

        if os.path.exists('.training_stop_signal'):
            print(f"\n🛑 Stop signal received! Finalizing training...")
            break

        # Get actions and values from policy
        with torch.no_grad():
            states_t = torch.from_numpy(states).float().to(device)
            logits = policy(states_t)
            features = policy.get_features(states_t)
            values = value_head(features).squeeze().cpu().numpy() # Get value estimate
            
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            action_probs = F.softmax(logits, dim=1).cpu().numpy()
        
        profiler.tick("Inference")
        actions_np = actions.cpu().numpy()

        
        # Step all environments
        next_states, rewards, dones = vec_env.step(actions_np)
        profiler.tick("Env Step")

        
        # --- REWARD SHAPING & SCALING ---
        # 1. Scale raw rewards based on Reward Type
        if spec.reward_type == 'interestingness':
            # PROXY ILL-ADJUSTED MODE
            # Optimizes "Interestingness", not Energy.
            # Food (+) -> +1.0 Interesting
            # Poison (-) -> +0.9 Interesting (The Trap!)
            scaled_rewards = np.zeros_like(rewards)
            scaled_rewards[rewards > 0] = 1.0
            scaled_rewards[rewards < 0] = 0.9  # Poison looks almost as good!
            
        elif spec.reward_type == 'shaped':
            # HANDHOLD MODE
            # Simple +1/-1 signals (clear, balanced)
            scaled_rewards = np.zeros_like(rewards)
            scaled_rewards[rewards > 0] = 1.0   # Food
            scaled_rewards[rewards < 0] = -1.0  # Poison
            
        else:  # 'energy_delta'
            # TRUE ENERGY MODE
            # Use actual energy values, auto-scaled to fit clamp range
            scaled_rewards = rewards * auto_reward_scale
        
        # 2. Potential-Based Shaping: gamma * phi(s') - phi(s)
        next_potentials = _get_potentials_np(next_states)
        # If terminal, next_potential is 0.0 (no food visible state)
        target_potentials = np.where(dones, 0.0, next_potentials)
        
        shaping = (target_potentials * gamma) - prev_potentials
        
        # Add shaping (Total Reward)
        total_rewards = scaled_rewards + shaping
        
        # Clip only the total to avoid crazy outliers, but allow range
        # Food(1.0) + Shaping(~1.0) = 2.0. Clipping to [-5, 5] is safe.
        total_rewards = np.clip(total_rewards, -5.0, 5.0)
        
        # Update potentials
        prev_potentials = next_potentials # next_states is already reset state
        
        # Store experience
        states_buffer.append(states)
        actions_buffer.append(actions_np)
        log_probs_buffer.append(log_probs.cpu().numpy())
        rewards_buffer.append(total_rewards) # Use shaped rewards for training
        dones_buffer.append(dones)
        values_buffer.append(values)
        
        total_steps += n_envs
        episode_rewards += rewards
        
        # Handle done episodes (stats tracking only)
        # Note: GAE handles value bootstrapping for non-terminal steps in buffer
        if dones.any():
            done_envs = np.where(dones)[0]
            for i in done_envs:
                episode_count += 1
                if episode_rewards[i] > best_reward:
                    best_reward = episode_rewards[i]
                if logger:
                    logger.log_episode(
                        episode=episode_count,
                        reward=episode_rewards[i],
                        length=500, # Approx
                        food_eaten=vec_env.current_episode_food[i],
                        food_density=vec_env.initial_food,
                        curriculum_progress=0.0, # Removed
                        action_prob_std=action_probs.std(axis=1).mean(),
                    )
            episode_rewards[dones] = 0
            vec_env.reset(done_envs)
        
        states = next_states
        
        profiler.tick("Storage")

        # Update dashboard
        if dashboard:
            dashboard.update(mode, 'step', (states[0], action_probs[0], total_steps))
            profiler.tick("Dashboard Step")
        
        if len(states_buffer) % 10 == 0:

             print(".", end="", flush=True)

        # UPDATE STEP
        if len(states_buffer) >= steps_per_env:
            print("")
            
            # BOOTSTRAPPING & GAE
            with torch.no_grad():
                next_states_t = torch.from_numpy(states).float().to(device)
                next_features = policy.get_features(next_states_t)
                next_value = value_head(next_features).squeeze().cpu().numpy()
            
            # Initialize GAE result container
            # (steps, envs)
            returns = np.zeros_like(rewards_buffer)
            advantages = np.zeros_like(rewards_buffer)
            
            lastgae = 0
            
            # Iterate backwards
            for t in reversed(range(len(rewards_buffer))):
                if t == len(rewards_buffer) - 1:
                    nextnonterminal = 1.0 - dones # Current done status? No.
                    # Correctness check: dones_buffer[t] indicates if step t resulted in completion
                    # If step t was done, next value should be 0.
                    # But wait, next_value is V(s_{t+1}).
                    # If done, s_{t+1} is the FIRST step of new episode (reset).
                    # PPO standard: value of terminal state is 0 (or reward handled).
                    # Standard GAE implementation:
                    # delta = r_t + gamma * V(s_{t+1}) * (1-d_t) - V(s_t)
                    nextvalues = next_value
                else:
                    nextvalues = values_buffer[t+1]
                
                # Check done flag. If done[t] is true, it means s_t -> s_{t+1} was terminal.
                # So value of future is 0.
                mask = 1.0 - dones_buffer[t]
                
                delta = rewards_buffer[t] + gamma * nextvalues * mask - values_buffer[t]
                lastgae = delta + gamma * gae_lambda * mask * lastgae
                advantages[t] = lastgae
            
            # Returns = Advantages + Values
            returns = advantages + np.stack(values_buffer)
            profiler.tick("GAE Calc")

            
            # Flatten for update - keep as NumPy arrays (no .tolist()!)
            all_states = np.concatenate(states_buffer, axis=0)  # (steps*envs, ...)
            all_actions = np.concatenate(actions_buffer, axis=0)
            all_log_probs = np.concatenate(log_probs_buffer, axis=0)
            all_returns = returns.flatten()
            all_advantages = advantages.flatten()
            all_old_values = np.stack(values_buffer).flatten()
            
            # Normalize advantages
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
            
            # PPO UPDATE - pass NumPy arrays directly for efficiency
            losses = _ppo_update(
                policy, value_head, optimizer,
                all_states, all_actions, all_log_probs,
                gamma, eps_clip, k_epochs, device,
                entropy_coef=entropy_coef, value_coef=value_coef,
                returns=all_returns,
                advantages=all_advantages,
                old_values=all_old_values
            )
            profiler.tick("PPO Update")
            
            update_count += 1

            policy_loss, value_loss, entropy = losses
            
            # Clear buffers
            states_buffer = []
            actions_buffer = []
            log_probs_buffer = []
            rewards_buffer = []
            dones_buffer = []
            values_buffer = []
            
            if logger:
                logger.log_update(
                    update_num=update_count,
                    total_steps=total_steps,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    entropy=entropy,
                    action_probs=action_probs[0].tolist()
                )
            
            
            # Randomize every update to prevent overfitting
            new_food = np.random.randint(min_food, max_food + 1)
            new_poison = np.random.randint(min_poison, max_poison + 1)
            vec_env.initial_food = new_food
            vec_env.poison_count = new_poison
            # Note: This only affects NEW episodes (resets), not currently running ones.
            # Which is perfect.
            
            if dashboard:
                current_reward = best_reward
                if logger and hasattr(logger, 'episode_rewards') and len(logger.episode_rewards) > 0:
                    current_reward = np.mean(logger.episode_rewards[-10:])
                elif current_reward == float('-inf'):
                    current_reward = 0.0  # Default to 0 if no episodes finished yet

                dashboard.update(mode, 'ppo', (policy_loss, value_loss, entropy, action_probs[0]))
                dashboard.update(mode, 'episode', (episode_count, current_reward, 500, 0, vec_env.initial_food))
            
            if dashboard:
                profiler.tick("Dashboard Log")

            elapsed = time.perf_counter() - start_time
            sps = total_steps / elapsed
            prob_std = action_probs.std(axis=1).mean()
            print(f"   [{mode}] Step {total_steps:,}: {sps:,.0f} sps | "
                  f"Best R={best_reward:.0f} | ProbStd={prob_std:.3f} | Ent={entropy:.3f}")
            print(f"   [Profile] {profiler.summary()}")
            profiler.reset()
            
            # Step LR scheduler if enabled
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            # Save checkpoint if interval is set
            if checkpoint_interval > 0 and update_count % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, 
                    f"checkpoint_{mode}_u{update_count}_s{total_steps}.pth"
                )
                torch.save({
                    'update': update_count,
                    'total_steps': total_steps,
                    'policy_state_dict': policy.state_dict(),
                    'value_head_state_dict': value_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_reward': best_reward,
                }, checkpoint_path)
                print(f"   💾 Checkpoint saved: {checkpoint_path}")

    
    torch.save(policy.state_dict(), output_path)
    elapsed = time.perf_counter() - start_time
    print(f"\n✅ [{mode}] Done! {total_steps:,} steps in {elapsed:.1f}s ({total_steps/elapsed:,.0f} steps/sec)")
    print(f"   Saved: {output_path}")
    
    if logger:
        logger.finalize(best_efficiency=best_reward, final_model_path=output_path)
    if dashboard:
        dashboard.update(mode, 'finished', None)


def _ppo_update(
    policy, value_head, optimizer,
    states: np.ndarray,
    actions: np.ndarray, 
    log_probs: np.ndarray,
    gamma: float,
    eps_clip: float,
    k_epochs: int,
    device: torch.device,
    entropy_coef: float = 0.001,
    value_coef: float = 0.5, 
    returns: np.ndarray = None,
    advantages: np.ndarray = None,
    old_values: np.ndarray = None
):
    """
    Perform PPO update on collected experience.
    
    Args:
        states: Observations, shape (batch, channels, h, w)
        actions: Action indices, shape (batch,)
        log_probs: Log probabilities of actions taken, shape (batch,)
        returns: Computed returns (rewards-to-go), shape (batch,)
        advantages: GAE advantages, shape (batch,)
        old_values: Value estimates from collection, shape (batch,)
    """
    # Convert NumPy arrays to tensors efficiently (from_numpy is zero-copy when possible)
    old_states = torch.from_numpy(states).float().to(device)
    old_actions = torch.from_numpy(actions).long().to(device)
    old_log_probs = torch.from_numpy(log_probs).float().to(device)
    
    returns_t = torch.from_numpy(returns).float().to(device)
    advantages_t = torch.from_numpy(advantages).float().to(device)
    
    old_values_t = None
    if old_values is not None:
        old_values_t = torch.from_numpy(old_values).float().to(device)

    final_policy_loss = 0.0
    final_value_loss = 0.0
    final_entropy = 0.0
    
    for _ in range(k_epochs):
        logits = policy(old_states)
        features = policy.get_features(old_states)
        values = value_head(features).squeeze()
        
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(old_actions)
        entropy = dist.entropy()
        
        # Ratio and clipped surrogate
        ratios = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratios * advantages_t
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages_t
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value Loss with Clipping
        if old_values_t is not None:
            v_clip = old_values_t + torch.clamp(values - old_values_t, -eps_clip, eps_clip)
            v_loss1 = F.mse_loss(values, returns_t)
            v_loss2 = F.mse_loss(v_clip, returns_t)
            value_loss = torch.max(v_loss1, v_loss2)
        else:
            value_loss = F.mse_loss(values, returns_t)
            
        entropy_bonus = entropy.mean()
        
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(value_head.parameters(), 0.5)
        optimizer.step()

        final_policy_loss = policy_loss.item()
        final_value_loss = value_loss.item()
        final_entropy = entropy_bonus.item()
    
    return final_policy_loss, final_value_loss, final_entropy


if __name__ == '__main__':
    import argparse
    import threading
    from goodharts.configs.observation_spec import get_all_mode_names
    from goodharts.configs.default_config import get_config
    
    # Cleanup stale stop signal
    if os.path.exists('.training_stop_signal'):
        try:
            os.remove('.training_stop_signal')
            print("🧹 Removed stale stop signal.")
        except OSError:
            pass

    config = get_config()
    all_modes = get_all_mode_names(config)
    brain_names = get_brain_names()
    
    parser = argparse.ArgumentParser(description='PPO training for Goodhart agents')
    parser.add_argument('--mode', default='ground_truth', choices=all_modes + ['all'],
                        help='Training mode (or "all" for parallel training)')
    parser.add_argument('--brain', default='base_cnn', choices=brain_names,
                        help='Neural network architecture')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total environment steps (default: 100,000)')
    parser.add_argument('--updates', type=int, default=None,
                        help='Number of PPO updates (alternative to --timesteps)')
    parser.add_argument('--entropy', type=float, default=None,
                        help='Entropy coefficient (default: from config)')
    parser.add_argument('--dashboard', '-d', action='store_true',
                        help='Show live training dashboard')
    parser.add_argument('--sequential', '-s', action='store_true',
                        help='Train modes sequentially (saves VRAM for high n_envs)')
    parser.add_argument('--n-envs', type=int, default=64,
                        help='Number of parallel environments')
    args = parser.parse_args()
    
    # Get training config
    train_cfg = get_training_config()
    steps_per_env = train_cfg.get('steps_per_env', 128)
    
    # Determine timesteps from either --timesteps or --updates
    if args.updates is not None:
        total_timesteps = args.updates * args.n_envs * steps_per_env
        print(f"   📊 {args.updates} updates × {args.n_envs} envs × {steps_per_env} steps = {total_timesteps:,} timesteps")
    elif args.timesteps is not None:
        total_timesteps = args.timesteps
    else:
        total_timesteps = 100_000  # Default
    
    modes_to_train = all_modes if args.mode == 'all' else [args.mode]
    
    # Get entropy_coef from config if not specified on CLI
    entropy_coef = args.entropy if args.entropy is not None else train_cfg.get('entropy_coef', 0.02)
    print(f"   Entropy coefficient: {entropy_coef}")
    
    if args.dashboard:
        from goodharts.training.train_dashboard import create_dashboard
        from goodharts.behaviors.action_space import num_actions
        
        print(f"\n🎛️  Training with dashboard")
        dashboard = create_dashboard(modes_to_train, n_actions=num_actions(1))
        
        if args.sequential:
            # Sequential dashboard mode: train one at a time in background thread
            print(f"🔄 Sequential mode: training {len(modes_to_train)} modes one at a time")
            
            def sequential_training():
                for mode in modes_to_train:
                    output_path = f'models/ppo_{mode}.pth'
                    train_ppo(
                        mode=mode,
                        brain_type=args.brain,
                        n_envs=args.n_envs,
                        total_timesteps=total_timesteps,
                        entropy_coef=entropy_coef,
                        output_path=output_path,
                        dashboard=dashboard,
                        log_to_file=True,
                    )
            
            t = threading.Thread(target=sequential_training, daemon=True)
            t.start()
            
            try:
                dashboard.run()
            except KeyboardInterrupt:
                print("\n⏹️  Training interrupted")
            
            t.join(timeout=1.0)
        else:
            # Parallel dashboard mode
            threads = []
            for mode in modes_to_train:
                output_path = f'models/ppo_{mode}.pth'
                t = threading.Thread(
                    target=train_ppo,
                    kwargs={
                        'mode': mode,
                        'brain_type': args.brain,
                        'n_envs': args.n_envs,
                        'total_timesteps': total_timesteps,
                        'entropy_coef': entropy_coef,
                        'output_path': output_path,
                        'dashboard': dashboard,
                        'log_to_file': True,
                    },
                    daemon=True
                )
                threads.append(t)
                t.start()
            
            try:
                dashboard.run()
            except KeyboardInterrupt:
                print("\n⏹️  Training interrupted")
            
            for t in threads:
                t.join(timeout=1.0)
    
    else:
        # Sequential or Parallel without dashboard
        if len(modes_to_train) > 1 and not args.sequential:
            print(f"\n🚀 Parallel training: {len(modes_to_train)} modes")
            threads = []
            for mode in modes_to_train:
                output_path = f'models/ppo_{mode}.pth'
                t = threading.Thread(
                    target=train_ppo,
                    kwargs={
                        'mode': mode,
                        'brain_type': args.brain,
                        'n_envs': args.n_envs,
                        'total_timesteps': total_timesteps,
                        'entropy_coef': entropy_coef,
                        'output_path': output_path,
                        'log_to_file': True,
                    }
                )
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
        else:
            # Sequential training (single mode or --sequential flag)
            if len(modes_to_train) > 1:
                print(f"\n🔄 Sequential training: {len(modes_to_train)} modes")
            for mode in modes_to_train:
                output_path = f'models/ppo_{mode}.pth'
                train_ppo(
                    mode=mode,
                    brain_type=args.brain,
                    n_envs=args.n_envs,
                    total_timesteps=total_timesteps,
                    entropy_coef=entropy_coef,
                    output_path=output_path,
                    log_to_file=len(modes_to_train) > 1,  # Log to file if training multiple
                )
