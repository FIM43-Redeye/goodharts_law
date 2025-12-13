"""
PPO (Proximal Policy Optimization) training with Curriculum Learning.

Uses TinyCNN for the policy network (compatible with LearnedBehavior).
Adds a separate value head for the critic.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

from goodharts.configs.default_config import get_config
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
    output_path: str = 'models/ppo_agent.pth',
    device: torch.device = None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n🚀 PPO Training: {mode} on {device}", flush=True)
    
    config = get_config()
    action_space = build_action_space(1)
    n_actions = num_actions(1)
    
    # Get observation spec from centralized config - no more hardcoding!
    spec = config['get_observation_spec'](mode)
    print(f"   Observation: {spec}", flush=True)
    
    # Curriculum: start easy, end hard
    initial_food = 2500
    final_food = 50
    curriculum_steps = max(1, int(max_episodes * 0.9))
    
    # Use brain registry - architecture derived from observation spec
    policy = create_brain(brain_type, spec, output_size=n_actions).to(device)
    print(f"   Brain: {brain_type} (hidden_size={policy.hidden_size})", flush=True)
    
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
        world.place_poison(30)
        
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
        
        for t in range(500):
            time_step += 1
            
            # Get action from policy
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                logits = policy(state_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action).item()
            
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
                _ppo_update(policy, value_head, optimizer, states, actions, log_probs, 
                           rewards, dones, gamma, eps_clip, k_epochs, device)
                states, actions, log_probs, rewards, dones = [], [], [], [], []
                print(f"   [{mode}] Step {time_step}: PPO Update", flush=True)
            
            if done:
                break
        
        running_reward += ep_reward
        running_food += ep_food
        
        # Log every 10 episodes
        if ep % 10 == 0:
            avg_rew = running_reward / 10
            avg_food = running_food / 10
            eff = avg_rew / np.sqrt(current_food) if current_food > 0 else 0
            
            # Decay best_efficiency so early easy-mode wins fade over time
            # This makes late hard-mode performance more valuable
            best_efficiency *= efficiency_decay ** 10  # Apply 10 episodes of decay
            
            print(f"[{mode}] Ep {ep:4d}: R={avg_rew:+.0f} | Eff={eff:+.1f} (best={best_efficiency:.1f}) | Food={avg_food:.1f} | "
                  f"Dens={current_food} ({progress*100:.0f}%)", flush=True)
            
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


def _ppo_update(policy, value_head, optimizer, states, actions, log_probs, rewards, dones, 
                gamma, eps_clip, k_epochs, device):
    """Perform PPO update on collected experience."""
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
        
        loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(values, returns) - 0.01 * entropy
        
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()


def _train_mode_wrapper(args_tuple):
    """Wrapper for multiprocessing - must be at module level for pickle."""
    mode, brain_type, episodes = args_tuple
    output_path = f'models/ppo_{mode}.pth'
    train_ppo(mode=mode, brain_type=brain_type, max_episodes=episodes, output_path=output_path)


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
    from goodharts.configs.observation_spec import get_all_mode_names
    
    all_modes = get_all_mode_names()
    brain_names = get_brain_names()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='ground_truth', choices=all_modes + ['all'])
    parser.add_argument('--brain', default='tiny_cnn', choices=brain_names,
                        help='Brain architecture to use')
    parser.add_argument('--episodes', type=int, default=500)
    args = parser.parse_args()
    
    if args.mode == 'all':
        # Run all modes in parallel
        print(f"\n🔄 Training ALL {len(all_modes)} modes in parallel (brain={args.brain})...\n")
        
        ctx = mp.get_context('spawn')
        processes = []
        for mode in all_modes:
            p = ctx.Process(target=_train_mode_wrapper, args=((mode, args.brain, args.episodes),))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        print("\n✅ All models trained!")
        for mode in all_modes:
            print(f"   models/ppo_{mode}.pth")
    else:
        train_ppo(mode=args.mode, brain_type=args.brain, max_episodes=args.episodes)




