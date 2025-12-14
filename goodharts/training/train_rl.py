"""
True Reinforcement Learning training for agent behaviors.

Uses policy gradient (REINFORCE) to train agents from scratch.
No expert demonstrations - agents learn purely from:
  ENERGY GOOD, DYING BAD

This replaces behavior cloning after our "Goodhart moment".
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import NamedTuple
from collections import deque

from goodharts.configs.default_config import get_config
from goodharts.environments.world import World
from goodharts.agents.organism import Organism
from goodharts.behaviors.learned import LearnedBehavior
from goodharts.behaviors.brains.base_cnn import BaseCNN
from goodharts.behaviors.action_space import build_action_space, num_actions
from goodharts.utils.logging_config import get_logger
from goodharts.utils.device import get_device

logger = get_logger("rl_training")


class Episode(NamedTuple):
    """A complete episode of agent experience."""
    states: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    total_reward: float
    steps_survived: int
    death_reason: str | None


def collect_episode(
    model: BaseCNN,
    config: dict,
    mode: str = 'ground_truth',
    temperature: float = 1.0,
    device: torch.device = None,
) -> Episode:
    """
    Run one agent until death or max steps, collecting experience.
    
    Args:
        model: The CNN policy to use
        config: Simulation config
        mode: 'ground_truth' or 'proxy'
        temperature: Softmax temperature for action sampling
        device: Torch device
        
    Returns:
        Episode with all states, actions, rewards
    """
    if device is None:
        device = get_device(verbose=False)
    
    model.eval()
    
    # Create a simple world with just one agent
    world = World(config['GRID_WIDTH'], config['GRID_HEIGHT'], config)
    world.place_food(config['GRID_FOOD_INIT'])
    world.place_poison(config['GRID_POISON_INIT'])
    
    # Create agent with a dummy behavior that we'll override
    class DummyBehavior:
        requirements = ['ground_truth'] if mode == 'ground_truth' else ['proxy_metric']
        def decide_action(self, agent, view):
            return (0, 0)
    
    x = np.random.randint(0, config['GRID_WIDTH'])
    y = np.random.randint(0, config['GRID_HEIGHT'])
    agent = Organism(x, y, config['ENERGY_START'], config['AGENT_VIEW_RANGE'], 
                     world, DummyBehavior(), config)
    
    states = []
    actions = []
    rewards = []
    
    max_steps = config.get('MAX_EPISODE_STEPS', 1000)
    action_space = build_action_space(1)
    
    for step in range(max_steps):
        if not agent.alive:
            break
        
        # Get observation
        state = agent.get_local_view(mode=mode)
        states.append(state)
        
        # Get action from policy
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            logits = model(state_tensor)
            probs = F.softmax(logits / temperature, dim=1)
            action_idx = torch.multinomial(probs, 1).item()
        
        actions.append(action_idx)
        
        # Take action
        energy_before = agent.energy
        dx, dy = action_space[action_idx]
        agent.move(dx, dy)
        agent.eat()
        
        # Compute reward: energy delta
        # This is the TRUE reward signal - not copying anyone
        reward = agent.energy - energy_before
        
        # Bonus/penalty for survival milestones
        if not agent.alive:
            reward -= 10.0  # Death penalty
        
        rewards.append(reward)
        
        # Respawn food/poison if configured
        if config.get('RESPAWN_RESOURCES', True):
            # Check if we ate something
            pass  # Already handled in agent.eat()
    
    return Episode(
        states=states,
        actions=actions,
        rewards=rewards,
        total_reward=sum(rewards),
        steps_survived=len(states),
        death_reason=agent.death_reason if not agent.alive else None
    )


def compute_returns(rewards: list[float], gamma: float = 0.99) -> list[float]:
    """
    Compute discounted returns for each timestep.
    
    G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train_reinforce(
    config: dict | None = None,
    mode: str = 'ground_truth',
    num_episodes: int = 1000,
    lr: float = 1e-3,
    gamma: float = 0.99,
    temperature: float = 1.0,
    entropy_coef: float = 0.01,
    output_path: str = 'models/rl_agent.pth',
    log_interval: int = 50,
    device: torch.device | None = None,
) -> BaseCNN:
    """
    Train an agent using REINFORCE (policy gradient).
    
    No behavior cloning - learns purely from energy rewards.
    
    Args:
        config: Simulation config
        mode: 'ground_truth' or 'proxy'
        num_episodes: Number of episodes to train
        lr: Learning rate
        gamma: Discount factor for returns
        temperature: Softmax temperature for exploration
        entropy_coef: Coefficient for entropy bonus (encourages exploration)
        output_path: Where to save the trained model
        log_interval: Print stats every N episodes
        device: Torch device
        
    Returns:
        Trained BaseCNN model
    """
    if config is None:
        config = get_config()
    
    if device is None:
        device = get_device()
    
    print(f"\n🚀 Starting REINFORCE training on {device}", flush=True)
    print(f"   Mode: {mode}, Episodes: {num_episodes}, LR: {lr}", flush=True)
    print(f"   Saving to: {output_path}\n", flush=True)
    
    # Higher food density for faster learning
    training_config = config.copy()
    training_config['GRID_FOOD_INIT'] = 150
    training_config['GRID_POISON_INIT'] = 30
    training_config['MAX_EPISODE_STEPS'] = 500
    
    # Get observation spec for dynamic channel count
    obs_spec = training_config['get_observation_spec'](mode)
    
    # Initialize model
    model = BaseCNN(
        input_shape=obs_spec.input_shape,
        input_channels=obs_spec.num_channels,
        output_size=num_actions(1)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Tracking
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_avg_reward = float('-inf')
    
    for episode in range(num_episodes):
        # Collect one episode
        ep = collect_episode(model, training_config, mode, temperature, device)
        
        episode_rewards.append(ep.total_reward)
        episode_lengths.append(ep.steps_survived)
        
        if len(ep.states) == 0:
            continue  # Empty episode, skip
        
        # Compute returns
        returns = compute_returns(ep.rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        
        # Normalize returns (helps with training stability)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy gradient loss
        model.train()
        
        states = torch.tensor(np.array(ep.states), dtype=torch.float32, device=device)
        actions = torch.tensor(ep.actions, dtype=torch.long, device=device)
        
        logits = model(states)
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Policy gradient loss: -E[log π(a|s) * G]
        policy_loss = -(action_log_probs * returns).mean()
        
        # Entropy bonus (encourages exploration)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        
        # Total loss
        loss = policy_loss - entropy_coef * entropy
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        # Logging - use print for immediate feedback
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            
            print(f"Ep {episode+1:4d}/{num_episodes}: "
                  f"Reward={avg_reward:+7.2f}, "
                  f"Steps={avg_length:5.0f}, "
                  f"Loss={loss.item():+.4f}", flush=True)
            
            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(model.state_dict(), output_path)
                print(f"  ⭐ New best! Saved.", flush=True)
    
    # Final save
    torch.save(model.state_dict(), output_path)
    print(f"\n✅ Training complete. Model saved to {output_path}", flush=True)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train agent with REINFORCE")
    parser.add_argument('--mode', choices=['ground_truth', 'proxy', 'both'], 
                        default='both', help='Which agent type to train')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    args = parser.parse_args()
    
    config = get_config()
    
    if args.mode in ['ground_truth', 'both']:
        logger.info("=" * 60)
        logger.info("Training GROUND TRUTH agent with RL")
        logger.info("=" * 60)
        train_reinforce(
            config=config,
            mode='ground_truth',
            num_episodes=args.episodes,
            lr=args.lr,
            gamma=args.gamma,
            output_path='models/rl_ground_truth.pth'
        )
    
    if args.mode in ['proxy', 'both']:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training PROXY agent with RL")
        logger.info("=" * 60)
        train_reinforce(
            config=config,
            mode='proxy',
            num_episodes=args.episodes,
            lr=args.lr,
            gamma=args.gamma,
            output_path='models/rl_proxy.pth'
        )
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("RL Training complete!")
    logger.info("  - models/rl_ground_truth.pth")
    logger.info("  - models/rl_proxy.pth")
    logger.info("=" * 60)
