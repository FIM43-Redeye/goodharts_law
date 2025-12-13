"""
PPO (Proximal Policy Optimization) training with Curriculum Learning.

Uses TinyCNN for the policy network (compatible with LearnedBehavior).
Adds a separate value head for the critic.

Features:
- Live graphical visualization of training progress
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
    lr: float = 3e-4,
    gamma: float = 0.99,
    eps_clip: float = 0.2,
    k_epochs: int = 4,
    update_timestep: int = 2000,
    entropy_coef: float = 0.001,  # Reduced from 0.01 - less uniform pressure
    value_coef: float = 0.5,
    output_path: str = 'models/ppo_agent.pth',
    device: torch.device = None,
    visualize: bool = False,  # Enable live training visualization
    visualizer = None,  # External visualizer (for multi-mode training)
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
        update_timestep: Steps between PPO updates
        entropy_coef: Entropy bonus coefficient (lower = less uniform pressure)
        value_coef: Value loss coefficient
        output_path: Where to save the best model
        device: Torch device (auto-detected if None)
        visualize: If True, open a live visualization window
        visualizer: External TrainingVisualizer (overrides visualize flag)
    """
    if device is None:
        device = get_device()
    
    print(f"\n🚀 PPO Training: {mode} on {device}", flush=True)
    
    # Setup visualization
    viz = visualizer
    if viz is None and visualize:
        from goodharts.training.train_viz import create_training_visualizer
        viz = create_training_visualizer(mode, n_actions=num_actions(1))
    
    config = get_config()
    action_space = build_action_space(1)
    n_actions = num_actions(1)
    
    # Get observation spec from centralized config - no more hardcoding!
    spec = config['get_observation_spec'](mode)
    print(f"   Observation: {spec}", flush=True)
    
    # Curriculum: start easy, end hard (from config)
    train_cfg = get_training_config()
    initial_food = train_cfg.get('initial_food', 2500)
    final_food = train_cfg.get('final_food', 50)
    curriculum_fraction = train_cfg.get('curriculum_fraction', 0.9)
    curriculum_steps = max(1, int(max_episodes * curriculum_fraction))
    
    # Use brain registry - architecture derived from observation spec
    policy = create_brain(brain_type, spec, output_size=n_actions).to(device)
    print(f"   Brain: {brain_type} (hidden_size={policy.hidden_size})", flush=True)
    print(f"   Entropy coef: {entropy_coef} (lower = more opinionated)", flush=True)
    
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
            
            # Store experience
            states.append(state)
            actions.append(action_idx)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            
            ep_reward += reward
            state = agent.get_local_view(mode=mode)
            
            # PPO Update when buffer is full
            if len(states) >= update_timestep:
                losses = _ppo_update(
                    policy, value_head, optimizer, states, actions, log_probs, 
                    rewards, dones, gamma, eps_clip, k_epochs, device,
                    entropy_coef=entropy_coef, value_coef=value_coef
                )
                last_policy_loss, last_value_loss, last_entropy = losses
                states, actions, log_probs, rewards, dones = [], [], [], [], []
                print(f"   [{mode}] Step {time_step}: PPO Update (ent={last_entropy:.3f})", flush=True)
                
                # Update visualization with PPO metrics
                if viz:
                    viz.update(
                        ppo=(last_policy_loss, last_value_loss, last_entropy, last_action_probs),
                        total_steps=time_step
                    )
            
            if done:
                break
        
        running_reward += ep_reward
        running_food += ep_food
        
        # Update visualization with episode data
        if viz:
            viz.update(episode=(ep_reward, ep_length, current_food, progress))
        
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
    
    # Close visualizer if we created it
    if viz and visualizer is None:
        viz.stop()


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
    mode, brain_type, episodes, visualize = args_tuple
    output_path = f'models/ppo_{mode}.pth'
    train_ppo(mode=mode, brain_type=brain_type, max_episodes=episodes, 
              output_path=output_path, visualize=visualize)


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
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
    parser.add_argument('--entropy', type=float, default=0.001,
                        help='Entropy coefficient (lower = more opinionated, default: 0.001)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Show live training visualization window')
    args = parser.parse_args()
    
    if args.mode == 'all':
        # Run all modes in parallel
        print(f"\n🔄 Training ALL {len(all_modes)} modes in parallel (brain={args.brain})...\n")
        
        ctx = mp.get_context('spawn')
        processes = []
        for mode in all_modes:
            p = ctx.Process(target=_train_mode_wrapper, 
                           args=((mode, args.brain, args.episodes, args.visualize),))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        print("\n✅ All models trained!")
        for mode in all_modes:
            print(f"   models/ppo_{mode}.pth")
    else:
        train_ppo(
            mode=args.mode, 
            brain_type=args.brain, 
            max_episodes=args.episodes,
            entropy_coef=args.entropy,
            visualize=args.visualize
        )
