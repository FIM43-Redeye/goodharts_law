"""
PPO (Proximal Policy Optimization) training with Curriculum Learning.

Uses TinyCNN for the policy network (compatible with LearnedBehavior).
Adds a separate value head for the critic.

Features:
- Live graphical visualization of training progress
- Structured file logging (CSV + JSON) for AI review
- Configurable entropy coefficient (reduced from 0.01 to 0.001 by default)
- Curriculum learning with gradual difficulty increase
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

from goodharts.configs.default_config import get_config
from goodharts.config import get_training_config
from goodharts.utils.device import get_device
from goodharts.environments.world import World
from goodharts.agents.organism import Organism
from goodharts.behaviors.brains import create_brain, get_brain_names
from goodharts.behaviors.action_space import build_action_space, num_actions


class ValueHead(nn.Module):
    """Simple value head that attaches to TinyCNN features."""
    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
    
    def forward(self, features):
        return self.fc(features)


def train_ppo(
    mode: str = 'ground_truth',
    brain_type: str = 'tiny_cnn',
    max_episodes: int = 500,
    lr: float = None,  # Read from config if None
    gamma: float = 0.99,
    eps_clip: float = 0.2,
    k_epochs: int = 4,
    update_timestep: int = None,  # Read from config if None
    entropy_coef: float = None,   # Read from config if None
    reward_scale: float = None,   # Read from config if None
    value_coef: float = 0.5,
    output_path: str = 'models/ppo_agent.pth',
    device: torch.device = None,
    visualize: bool = False,  # Enable live training visualization (old style)
    visualizer = None,  # External visualizer (for multi-mode training)
    dashboard = None,  # Unified training dashboard
    log_to_file: bool = True,  # Enable structured file logging
    log_dir: str = 'logs',  # Directory for log files
):
    """
    Train an agent using Proximal Policy Optimization.
    
    Args:
        mode: Training mode ('ground_truth', 'proxy', 'proxy_ill_adjusted', etc.)
        brain_type: Neural network architecture from brain registry
        max_episodes: Maximum training episodes
        lr: Learning rate for Adam optimizer
        gamma: Discount factor for returns
        eps_clip: PPO clipping parameter
        k_epochs: Number of PPO epochs per update
        update_timestep: Steps between PPO updates (from config if None)
        entropy_coef: Entropy bonus coefficient (from config if None, default 0)
        reward_scale: Multiply rewards by this factor (from config if None)
        value_coef: Value loss coefficient
        output_path: Where to save the best model
        device: Torch device (auto-detected if None)
        visualize: If True, open a live visualization window
        visualizer: External TrainingVisualizer (overrides visualize flag)
        log_to_file: If True, write structured logs to CSV/JSON files
        log_dir: Directory for log files
    """
    if device is None:
        device = get_device()
    
    print(f"\n🚀 PPO Training: {mode} on {device}", flush=True)
    
    # Setup visualization
    viz = visualizer
    if viz is None and visualize:
        from goodharts.training.train_viz import create_training_visualizer
        viz = create_training_visualizer(mode, n_actions=num_actions(1))
    
    # Setup file logging
    logger = None
    if log_to_file:
        from goodharts.training.train_log import TrainingLogger
        logger = TrainingLogger(mode, output_dir=log_dir)
    
    config = get_config()
    action_space = build_action_space(1)
    n_actions = num_actions(1)
    
    # Get observation spec from centralized config - no more hardcoding!
    spec = config['get_observation_spec'](mode)
    print(f"   Observation: {spec}", flush=True)
    
    # Curriculum: start easy, end hard (from config)
    train_cfg = get_training_config()
    initial_food = train_cfg.get('initial_food', 2500)
    final_food = train_cfg.get('final_food', 500)
    curriculum_fraction = train_cfg.get('curriculum_fraction', 0.7)
    curriculum_steps = max(1, int(max_episodes * curriculum_fraction))
    
    # Get hyperparameters from config if not specified
    if lr is None:
        lr = train_cfg.get('learning_rate', 3e-4)
    if update_timestep is None:
        update_timestep = train_cfg.get('update_timestep', 500)
    if entropy_coef is None:
        entropy_coef = train_cfg.get('entropy_coef', 0.0)
    if reward_scale is None:
        reward_scale = train_cfg.get('reward_scale', 10.0)
    
    # Reward shaping config
    enable_shaping = train_cfg.get('enable_shaping', True)
    shaping_food_attract = train_cfg.get('shaping_food_attract', 0.5)
    shaping_poison_repel = train_cfg.get('shaping_poison_repel', 0.3)
    
    # Build shaping targets (pluggable for predator/prey)
    shaping_targets = None
    if enable_shaping:
        from goodharts.training.reward_shaping import ShapingTarget, compute_shaping_reward
        from goodharts.configs.default_config import CellType
        shaping_targets = (
            ShapingTarget(CellType.FOOD, weight=shaping_food_attract, distance_decay=True),
            ShapingTarget(CellType.POISON, weight=-shaping_poison_repel, distance_decay=True),
        )
        print(f"   Shaping: food={shaping_food_attract:+.2f}, poison={-shaping_poison_repel:+.2f}", flush=True)
    
    # Use brain registry - architecture derived from observation spec
    policy = create_brain(brain_type, spec, output_size=n_actions).to(device)
    print(f"   Brain: {brain_type} (hidden_size={policy.hidden_size})", flush=True)
    print(f"   Entropy coef: {entropy_coef}, Reward scale: {reward_scale}x", flush=True)
    print(f"   Update every: {update_timestep} steps", flush=True)
    
    # Separate value head (critic) - connects to brain's feature layer
    value_head = ValueHead(input_size=policy.hidden_size).to(device)
    
    # Optimizer for both
    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_head.parameters()), 
        lr=lr
    )
    
    # Experience buffer
    states, actions, log_probs, rewards, dones = [], [], [], [], []
    
    # Tracking
    time_step = 0
    running_reward = 0
    running_food = 0
    best_efficiency = float('-inf')
    
    # For visualization: track losses
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0
    last_action_probs = np.ones(n_actions) / n_actions
    
    # Derived efficiency decay rate
    # Goal: An agent with constant "skill" should have equal decayed-efficiency at any curriculum stage
    # Math: decay = (1 / sqrt(food_ratio))^(1/curriculum_steps)
    # This ensures early easy-mode performance doesn't dominate late hard-mode performance
    food_ratio = initial_food / final_food
    efficiency_decay = (1 / np.sqrt(food_ratio)) ** (1 / curriculum_steps)
    print(f"   Efficiency decay: {efficiency_decay:.6f}/ep (derived from curriculum)", flush=True)
    
    # Record hyperparameters to log
    if logger:
        logger.set_hyperparams(
            mode=mode,
            brain_type=brain_type,
            max_episodes=max_episodes,
            lr=lr,
            gamma=gamma,
            eps_clip=eps_clip,
            k_epochs=k_epochs,
            update_timestep=update_timestep,
            entropy_coef=entropy_coef,
            reward_scale=reward_scale,
            value_coef=value_coef,
            initial_food=initial_food,
            final_food=final_food,
            curriculum_fraction=curriculum_fraction,
        )
    
    # Track update count for logging
    update_count = 0
    
    for ep in range(1, max_episodes + 1):
        # Curriculum: gradually reduce food density
        progress = min(1.0, ep / curriculum_steps)
        current_food = int(initial_food - progress * (initial_food - final_food))
        
        # Setup world
        world = World(config['GRID_WIDTH'], config['GRID_HEIGHT'], config)
        world.place_food(current_food)
        world.place_poison(train_cfg.get('poison_count', 30))
        
        # Create agent with appropriate behavior requirements
        behavior_req = 'ground_truth' if mode == 'ground_truth' else 'proxy_metric'
        class DummyBehavior:
            requirements = [behavior_req]
            def decide_action(self, a, b): return (0, 0)
        
        agent = Organism(
            np.random.randint(0, config['GRID_WIDTH']),
            np.random.randint(0, config['GRID_HEIGHT']),
            config['ENERGY_START'],
            config['AGENT_VIEW_RANGE'],
            world, DummyBehavior(), config
        )
        
        state = agent.get_local_view(mode=mode)
        ep_reward = 0
        ep_food = 0
        ep_length = 0
        
        for t in range(train_cfg.get('steps_per_episode', 500)):
            time_step += 1
            ep_length += 1
            
            # Get action from policy
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                logits = policy(state_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action).item()
                
                # Track action probabilities for visualization
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                last_action_probs = probs
            
            action_idx = action.item()
            dx, dy = action_space[action_idx]
            
            # Execute action
            energy_before = agent.energy
            agent.move(dx, dy)
            
            # Freeze energy if spec says so (for proxy_ill_adjusted training)
            if spec.freeze_energy_in_training:
                agent.energy = energy_before
            
            # Calculate reward based on mode
            if spec.reward_type == 'interestingness':
                # Reward for touching interesting cells
                interestingness = world.proxy_grid[agent.y, agent.x]
                reward = interestingness
                
                # Clear the cell (consume it)
                if interestingness > 0:
                    world.grid[agent.y, agent.x] = 0
                    world.proxy_grid[agent.y, agent.x] = 0.0
                    ep_food += 1  # Track touches
            else:
                # Standard energy delta reward
                consumed = agent.eat()
                reward = agent.energy - energy_before
                if consumed and consumed[0] == 'FOOD':
                    ep_food += 1
            
            done = not agent.alive
            if done:
                reward -= 10.0
            
            # Get new state for shaping calculation
            new_state = agent.get_local_view(mode=mode)
            
            # Apply reward shaping (only considers visible targets)
            if shaping_targets is not None:
                view_center = (spec.view_size // 2, spec.view_size // 2)
                shaping_reward = compute_shaping_reward(
                    view_before=state,
                    view_after=new_state,
                    agent_pos_before=view_center,
                    agent_pos_after=view_center,
                    targets=shaping_targets
                )
                reward += shaping_reward
            
            # Apply reward scaling
            reward = reward * reward_scale
            
            # Store experience
            states.append(state)
            actions.append(action_idx)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            
            ep_reward += reward
            state = new_state  # Use already-computed new state
            
            # Update dashboard with step data (agent view, action probs)
            if dashboard:
                dashboard.update(mode, 'step', (state, last_action_probs, time_step))
            
            # PPO Update when buffer is full
            if len(states) >= update_timestep:
                losses = _ppo_update(
                    policy, value_head, optimizer, states, actions, log_probs, 
                    rewards, dones, gamma, eps_clip, k_epochs, device,
                    entropy_coef=entropy_coef, value_coef=value_coef
                )
                last_policy_loss, last_value_loss, last_entropy = losses
                states, actions, log_probs, rewards, dones = [], [], [], [], []
                update_count += 1
                print(f"   [{mode}] Step {time_step}: PPO Update (ent={last_entropy:.3f})", flush=True)
                
                # Log PPO update
                if logger:
                    logger.log_update(
                        update_num=update_count,
                        total_steps=time_step,
                        policy_loss=last_policy_loss,
                        value_loss=last_value_loss,
                        entropy=last_entropy,
                        action_probs=last_action_probs.tolist()
                    )
                
                # Update visualization with PPO metrics
                if viz:
                    viz.update(
                        ppo=(last_policy_loss, last_value_loss, last_entropy, last_action_probs),
                        total_steps=time_step
                    )
                
                # Update dashboard with PPO metrics
                if dashboard:
                    dashboard.update(mode, 'ppo', 
                                    (last_policy_loss, last_value_loss, last_entropy, last_action_probs))
            
            if done:
                break
        
        running_reward += ep_reward
        running_food += ep_food
        
        # Update visualization with episode data
        if viz:
            viz.update(episode=(ep_reward, ep_length, current_food, progress))
        
        # Update dashboard with episode data
        if dashboard:
            dashboard.update(mode, 'episode', (ep, ep_reward, ep_length, ep_food, current_food))
        
        # Log episode
        if logger:
            logger.log_episode(
                episode=ep,
                reward=ep_reward,
                length=ep_length,
                food_eaten=ep_food,
                food_density=current_food,
                curriculum_progress=progress,
                action_prob_std=last_action_probs.std()
            )
        
        # Log every 10 episodes
        if ep % 10 == 0:
            avg_rew = running_reward / 10
            avg_food = running_food / 10
            eff = avg_rew / np.sqrt(current_food) if current_food > 0 else 0
            
            # Decay best_efficiency so early easy-mode wins fade over time
            # This makes late hard-mode performance more valuable
            best_efficiency *= efficiency_decay ** 10  # Apply 10 episodes of decay
            
            # Diagnostic: action probability std
            prob_std = last_action_probs.std()
            prob_indicator = "⚠️" if prob_std < 0.05 else "✓"
            
            print(f"[{mode}] Ep {ep:4d}: R={avg_rew:+.0f} | Eff={eff:+.1f} (best={best_efficiency:.1f}) | Food={avg_food:.1f} | "
                  f"Dens={current_food} ({progress*100:.0f}%) | ProbStd={prob_std:.3f} {prob_indicator}", flush=True)
            
            if viz:
                viz.update(best_efficiency=best_efficiency)
            
            if eff > best_efficiency and ep > 20:
                best_efficiency = eff
                # Save only the policy - compatible with LearnedBehavior!
                torch.save(policy.state_dict(), output_path)
                print(f"  ⭐ Saved (eff={eff:.2f})", flush=True)
            
            running_reward = 0
            running_food = 0
    
    # Save final
    torch.save(policy.state_dict(), output_path.replace('.pth', '_final.pth'))
    print(f"\n✅ Done! Best: {output_path}", flush=True)
    
    # Finalize logging
    if logger:
        summary = logger.finalize(best_efficiency=best_efficiency, final_model_path=output_path)
    
    # Close visualizer if we created it
    if viz and visualizer is None:
        viz.stop()


def train_ppo_vec(
    mode: str = 'ground_truth',
    brain_type: str = 'tiny_cnn',
    n_envs: int = 64,
    total_timesteps: int = 100_000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    eps_clip: float = 0.2,
    k_epochs: int = 4,
    update_timestep: int = 2048,
    entropy_coef: float = 0.0,
    value_coef: float = 0.5,
    output_path: str = 'models/ppo_agent.pth',
    device: torch.device = None,
    dashboard = None,  # Dashboard for live visualization
    log_to_file: bool = True,
    log_dir: str = 'logs',
):
    """
    Vectorized PPO training - 10-50x faster than regular train_ppo.
    
    Uses VecEnv to run N environments in parallel with batched NumPy operations.
    """
    from goodharts.environments.vec_env import create_vec_env
    from goodharts.configs.default_config import get_config
    from goodharts.config import get_training_config
    from goodharts.training.train_log import TrainingLogger
    
    if device is None:
        device = get_device()
    
    print(f"\n🚀 Vectorized PPO Training: {mode} on {device}")
    print(f"   {n_envs} parallel environments")
    
    config = get_config()
    train_cfg = get_training_config()
    
    # Get observation spec
    spec = config['get_observation_spec'](mode)
    n_actions = num_actions(1)
    
    # Create vectorized environment (pulls all settings from config)
    vec_env = create_vec_env(n_envs=n_envs, obs_spec=spec)
    print(f"   View: {vec_env.view_size}x{vec_env.view_size}, Channels: {vec_env.n_channels}")
    
    # Create policy and value head
    policy = create_brain(brain_type, spec, output_size=n_actions).to(device)
    value_head = ValueHead(input_size=policy.hidden_size).to(device)
    
    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_head.parameters()),
        lr=lr
    )
    
    print(f"   Brain: {brain_type} (hidden_size={policy.hidden_size})")
    print(f"   Entropy coef: {entropy_coef}")
    
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
            eps_clip=eps_clip,
            k_epochs=k_epochs,
            update_timestep=update_timestep,
            entropy_coef=entropy_coef,
            vectorized=True,
        )
    
    # Experience buffers
    states_buffer = []
    actions_buffer = []
    log_probs_buffer = []
    rewards_buffer = []
    dones_buffer = []
    
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
    
    while total_steps < total_timesteps:
        # Get actions from policy
        with torch.no_grad():
            states_t = torch.from_numpy(states).float().to(device)
            logits = policy(states_t)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            action_probs = F.softmax(logits, dim=1).cpu().numpy()
        
        actions_np = actions.cpu().numpy()
        
        # Step all environments
        next_states, rewards, dones = vec_env.step(actions_np)
        
        # Store experience
        states_buffer.append(states)
        actions_buffer.append(actions_np)
        log_probs_buffer.append(log_probs.cpu().numpy())
        rewards_buffer.append(rewards)
        dones_buffer.append(dones)
        
        total_steps += n_envs
        episode_rewards += rewards
        
        # Reset done environments and track episodes
        if dones.any():
            done_envs = np.where(dones)[0]
            for i in done_envs:
                episode_count += 1
                if episode_rewards[i] > best_reward:
                    best_reward = episode_rewards[i]
                # Log episode
                if logger:
                    logger.log_episode(
                        episode=episode_count,
                        reward=episode_rewards[i],
                        length=500,  # Approximate
                        food_eaten=0,
                        food_density=vec_env.initial_food,
                        curriculum_progress=total_steps / total_timesteps,
                        action_prob_std=action_probs.std(axis=1).mean(),
                    )
            episode_rewards[dones] = 0
            vec_env.reset(done_envs)
        
        states = next_states
        
        # Update dashboard with current state (sample first env)
        if dashboard:
            dashboard.update(mode, 'step', (states[0], action_probs[0], total_steps))
        
        # PPO Update
        if len(states_buffer) * n_envs >= update_timestep:
            all_states = np.concatenate(states_buffer, axis=0)
            all_actions = np.concatenate(actions_buffer, axis=0)
            all_log_probs = np.concatenate(log_probs_buffer, axis=0)
            all_rewards = np.concatenate(rewards_buffer, axis=0)
            all_dones = np.concatenate(dones_buffer, axis=0)
            
            losses = _ppo_update(
                policy, value_head, optimizer,
                all_states.tolist(), all_actions.tolist(), all_log_probs.tolist(),
                all_rewards.tolist(), all_dones.tolist(),
                gamma, eps_clip, k_epochs, device,
                entropy_coef=entropy_coef, value_coef=value_coef
            )
            
            update_count += 1
            policy_loss, value_loss, entropy = losses
            
            # Clear buffers
            states_buffer = []
            actions_buffer = []
            log_probs_buffer = []
            rewards_buffer = []
            dones_buffer = []
            
            # Log update
            if logger:
                logger.log_update(
                    update_num=update_count,
                    total_steps=total_steps,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    entropy=entropy,
                    action_probs=action_probs[0].tolist()
                )
            
            # Dashboard update
            if dashboard:
                dashboard.update(mode, 'ppo', (policy_loss, value_loss, entropy, action_probs[0]))
                dashboard.update(mode, 'episode', (episode_count, best_reward, 500, 0, vec_env.initial_food))
            
            # Progress
            elapsed = time.perf_counter() - start_time
            sps = total_steps / elapsed
            prob_std = action_probs.std(axis=1).mean()
            print(f"   [{mode}] Step {total_steps:,}: {sps:,.0f} sps | "
                  f"Best R={best_reward:.0f} | ProbStd={prob_std:.3f} | Ent={entropy:.3f}")
    
    # Save final model
    torch.save(policy.state_dict(), output_path)
    elapsed = time.perf_counter() - start_time
    print(f"\n✅ [{mode}] Done! {total_steps:,} steps in {elapsed:.1f}s ({total_steps/elapsed:,.0f} steps/sec)")
    print(f"   Saved: {output_path}")
    
    # Finalize logging
    if logger:
        logger.finalize(best_efficiency=best_reward, final_model_path=output_path)
    
    # Signal dashboard we're done
    if dashboard:
        dashboard.update(mode, 'finished', None)


def _ppo_update(policy, value_head, optimizer, states, actions, log_probs, rewards, dones, 
                gamma, eps_clip, k_epochs, device, entropy_coef=0.001, value_coef=0.5):
    """
    Perform PPO update on collected experience.
    
    Returns:
        Tuple of (policy_loss, value_loss, entropy) for logging
    """
    # Convert to tensors
    old_states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    old_actions = torch.tensor(actions, dtype=torch.long).to(device)
    old_log_probs = torch.tensor(log_probs, dtype=torch.float32).to(device)
    
    # Compute returns (rewards-to-go)
    returns = []
    discounted = 0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            discounted = 0
        discounted = r + gamma * discounted
        returns.insert(0, discounted)
    
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-7)
    
    # Track metrics for last epoch
    final_policy_loss = 0.0
    final_value_loss = 0.0
    final_entropy = 0.0
    
    # PPO epochs
    for _ in range(k_epochs):
        # Get policy logits and features
        logits = policy(old_states)
        
        # For value, we need to extract features from policy
        # Use the Brain interface's get_features() method
        features = policy.get_features(old_states)
        values = value_head(features).squeeze()
        
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(old_actions)
        entropy = dist.entropy()
        
        # Ratio and clipped surrogate
        ratios = torch.exp(new_log_probs - old_log_probs)
        advantages = returns - values.detach()
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_bonus = entropy.mean()
        
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(value_head.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        # Track final epoch metrics
        final_policy_loss = policy_loss.item()
        final_value_loss = value_loss.item()
        final_entropy = entropy_bonus.item()
    
    return final_policy_loss, final_value_loss, final_entropy


def _train_mode_wrapper(args_tuple):
    """Wrapper for multiprocessing - must be at module level for pickle."""
    mode, brain_type, episodes, visualize, dashboard = args_tuple
    output_path = f'models/ppo_{mode}.pth'
    train_ppo(mode=mode, brain_type=brain_type, max_episodes=episodes, 
              output_path=output_path, visualize=visualize, dashboard=dashboard)


def _train_thread_worker(mode: str, brain_type: str, episodes: int, 
                         dashboard: 'TrainingDashboard', log_to_file: bool = True):
    """Worker function for background training with dashboard integration."""
    output_path = f'models/ppo_{mode}.pth'
    
    try:
        train_ppo(
            mode=mode,
            brain_type=brain_type,
            max_episodes=episodes,
            output_path=output_path,
            visualize=False,
            dashboard=dashboard,
            log_to_file=log_to_file,
        )
    except Exception as e:
        print(f"[{mode}] Training error: {e}")
    finally:
        if dashboard:
            dashboard.update(mode, 'finished', None)


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
    import threading
    from goodharts.configs.observation_spec import get_all_mode_names
    from goodharts.configs.default_config import get_config

    config = get_config()
    all_modes = get_all_mode_names(config)
    brain_names = get_brain_names()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='ground_truth', choices=all_modes + ['all'])
    parser.add_argument('--brain', default='tiny_cnn', choices=brain_names,
                        help='Brain architecture to use')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--timesteps', type=int, default=100_000,
                        help='Total timesteps for vectorized training (default: 100k)')
    parser.add_argument('--entropy', type=float, default=None,
                        help='Entropy coefficient (default: from config)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Show live training visualization (old-style, per-process)')
    parser.add_argument('--dashboard', '-d', action='store_true',
                        help='Show unified training dashboard (all runs in one window)')
    parser.add_argument('--vec', action='store_true',
                        help='Use vectorized training (10-50x faster)')
    parser.add_argument('--n-envs', type=int, default=64,
                        help='Number of parallel environments for --vec mode (default: 64)')
    args = parser.parse_args()
    
    # Determine modes to train
    modes_to_train = all_modes if args.mode == 'all' else [args.mode]
    
    # Vectorized training with dashboard
    if args.vec and args.dashboard:
        from goodharts.training.train_dashboard import create_dashboard
        from goodharts.behaviors.action_space import num_actions
        
        print(f"\n🎛️  Vectorized training with dashboard")
        print(f"   Modes: {', '.join(modes_to_train)}")
        print(f"   {args.n_envs} parallel environments per mode\n")
        
        dashboard = create_dashboard(modes_to_train, n_actions=num_actions(1))
        
        # Start training threads
        threads = []
        for mode in modes_to_train:
            output_path = f'models/ppo_{mode}.pth'
            t = threading.Thread(
                target=train_ppo_vec,
                kwargs={
                    'mode': mode,
                    'brain_type': args.brain,
                    'n_envs': args.n_envs,
                    'total_timesteps': args.timesteps,
                    'entropy_coef': args.entropy if args.entropy else 0.0,
                    'output_path': output_path,
                    'dashboard': dashboard,
                    'log_to_file': True,
                },
                daemon=True
            )
            threads.append(t)
            t.start()
        
        # Run dashboard on main thread
        try:
            dashboard.run()
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted")
        
        for t in threads:
            t.join(timeout=1.0)
        
        print("\n✅ Dashboard closed")
    
    # Vectorized training without dashboard (sequential or parallel)
    elif args.vec:
        if len(modes_to_train) > 1:
            # Parallel vectorized training with threads
            print(f"\n🚀 Parallel vectorized training: {len(modes_to_train)} modes")
            threads = []
            for mode in modes_to_train:
                output_path = f'models/ppo_{mode}.pth'
                t = threading.Thread(
                    target=train_ppo_vec,
                    kwargs={
                        'mode': mode,
                        'brain_type': args.brain,
                        'n_envs': args.n_envs,
                        'total_timesteps': args.timesteps,
                        'entropy_coef': args.entropy if args.entropy else 0.0,
                        'output_path': output_path,
                        'log_to_file': True,
                    }
                )
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            print("\n✅ All modes trained!")
        else:
            # Single mode vectorized
            output_path = f'models/ppo_{modes_to_train[0]}.pth'
            train_ppo_vec(
                mode=modes_to_train[0],
                brain_type=args.brain,
                n_envs=args.n_envs,
                total_timesteps=args.timesteps,
                entropy_coef=args.entropy if args.entropy else 0.0,
                output_path=output_path,
            )
    
    # Standard dashboard mode (episode-based)
    elif args.dashboard:
        from goodharts.training.train_dashboard import create_dashboard
        from goodharts.behaviors.action_space import num_actions
        
        print(f"\n🎛️  Dashboard mode: {len(modes_to_train)} runs")
        print(f"   Modes: {', '.join(modes_to_train)}\n")
        
        dashboard = create_dashboard(modes_to_train, n_actions=num_actions(1))
        
        threads = []
        for mode in modes_to_train:
            t = threading.Thread(
                target=_train_thread_worker,
                args=(mode, args.brain, args.episodes, dashboard, True),
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
        
        print("\n✅ Dashboard closed")
        
    # All modes in parallel (multiprocessing)
    elif args.mode == 'all':
        print(f"\n🔄 Training ALL {len(all_modes)} modes in parallel (brain={args.brain})...\n")
        
        ctx = mp.get_context('spawn')
        processes = []
        for mode in all_modes:
            p = ctx.Process(target=_train_mode_wrapper, 
                           args=((mode, args.brain, args.episodes, args.visualize, None),))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        print("\n✅ All models trained!")
        for mode in all_modes:
            print(f"   models/ppo_{mode}.pth")
    
    # Single mode training
    else:
        train_ppo(
            mode=args.mode, 
            brain_type=args.brain, 
            max_episodes=args.episodes,
            entropy_coef=args.entropy,
            visualize=args.visualize
        )
