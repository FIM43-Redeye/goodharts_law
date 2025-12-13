"""
Training loop for learned behaviors.

Supports:
- Reward-weighted behavior cloning (supervised)
- Imitation learning from expert agents
- Stubs for full RL (policy gradient, Q-learning)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from pathlib import Path

from goodharts.behaviors.brains.tiny_cnn import TinyCNN
from goodharts.behaviors import LearnedBehavior
from goodharts.training.dataset import ReplayBuffer, SimulationDataset
from goodharts.training.collect import collect_from_expert, collect_experiences
from goodharts.configs.default_config import get_config
from goodharts.utils.logging_config import setup_logging, get_logger

logger = get_logger("training")


def train_behavior_cloning(
    config: dict,
    mode: str = 'ground_truth',
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    collection_steps: int = 1000,
    num_collection_agents: int = 10,
    output_path: str = 'models/model.pth',
    use_reward_weighting: bool = True,
    device: torch.device | None = None,
) -> TinyCNN:
    """
    Train a behavior via behavior cloning from expert demonstrations.
    
    Args:
        config: Simulation configuration
        mode: 'ground_truth' or 'proxy' - determines what signal agents use
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        collection_steps: Steps to run for data collection
        num_collection_agents: Number of expert agents during collection
        output_path: Where to save the trained model
        use_reward_weighting: If True, weight samples by reward
        device: Torch device (auto-detected if None)
        
    Returns:
        Trained TinyCNN model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Mode: {mode}, Epochs: {epochs}, Batch size: {batch_size}")
    
    # Determine expert class based on mode
    from ..behaviors import OmniscientSeeker, ProxySeeker
    from .collect import generate_poison_avoidance_samples
    expert_class = OmniscientSeeker if mode == 'ground_truth' else ProxySeeker
    
    # Collect expert demonstrations
    logger.info(f"Collecting {collection_steps} steps of expert ({expert_class.__name__}) demonstrations...")
    buffer = collect_from_expert(
        config=config,
        expert_class=expert_class,
        num_steps=collection_steps,
        num_agents=num_collection_agents,
        seed=42
    )
    logger.info(f"Collected {len(buffer)} experiences")
    
    # For ground-truth mode, add synthetic poison-avoidance samples
    # (Expert avoids poison so well that we have no negative examples!)
    if mode == 'ground_truth':
        poison_buffer = generate_poison_avoidance_samples(num_samples=500, reward_weight=15.0)
        logger.info(f"Added {len(poison_buffer)} poison-avoidance samples")
        for exp in poison_buffer.buffer:
            buffer.buffer.append(exp)
    
    if len(buffer) == 0:
        raise ValueError("No experiences collected! Check simulation setup.")
    
    # Convert to dataset
    dataset = SimulationDataset()
    dataset.from_replay_buffer(buffer)
    
    # Compute sample weights combining reward and visibility
    # This gives high weight to samples with visible targets + positive rewards
    sample_weights = None
    if use_reward_weighting:
        sample_weights = dataset.compute_combined_weights(visibility_mult=10.0)
        logger.info(f"Combined weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model - handle both 2D (H, W) and 3D (C, H, W) states
    sample_state = buffer.buffer[0].state
    if sample_state.ndim == 3:
        # New format: (channels, height, width)
        input_channels, height, width = sample_state.shape
        input_shape = (height, width)
    else:
        # Old format: (height, width)
        input_channels = 1
        input_shape = sample_state.shape
    
    # Determine output size based on action space (8-directional for max_move_distance=1)
    num_actions = 8
    
    model = TinyCNN(
        input_shape=input_shape,
        input_channels=input_channels,
        output_size=num_actions
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss for weighting
    
    # Training loop
    logger.info("Starting training...")
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, (states, actions, rewards) in enumerate(loader):
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            
            optimizer.zero_grad()
            
            logits = model(states)
            loss_per_sample = criterion(logits, actions)
            
            if use_reward_weighting:
                # Weight by reward: higher reward = more important to get right
                # Shift rewards to be positive
                weights = torch.clamp(rewards - rewards.min() + 0.1, min=0.1)
                weights = weights / weights.sum() * len(weights)
                loss = (loss_per_sample * weights).mean()
            else:
                loss = loss_per_sample.mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"Model saved to {output_path}")
    
    return model


def train_ground_truth_and_proxy(config: dict, **kwargs):
    """
    Train both a ground-truth model and a proxy model for comparison.
    
    This is the main entry point for demonstrating Goodhart's Law.
    """
    # Use higher food density for training (better learning signal)
    training_config = config.copy()
    training_config['GRID_FOOD_INIT'] = 200   # More food = more visible during training
    training_config['GRID_POISON_INIT'] = 40  # More poison too for balance
    
    logger.info("=" * 60)
    logger.info("Training GROUND TRUTH model...")
    logger.info("=" * 60)
    
    train_behavior_cloning(
        config=training_config,
        mode='ground_truth',
        output_path='models/ground_truth.pth',
        **kwargs
    )
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training PROXY model...")
    logger.info("=" * 60)
    
    train_behavior_cloning(
        config=training_config,
        mode='proxy',
        output_path='models/proxy_trained.pth',
        **kwargs
    )
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training complete! Models saved to models/")
    logger.info("  - models/ground_truth.pth")
    logger.info("  - models/proxy_trained.pth")
    logger.info("=" * 60)


# =============================================================================
# RL Training Stubs (for future implementation)
# =============================================================================

def train_policy_gradient(
    config: dict,
    mode: str = 'ground_truth',
    episodes: int = 100,
    output_path: str = 'models/pg_model.pth',
) -> TinyCNN:
    """
    Train using REINFORCE policy gradient.
    
    TODO: Implement for full RL support.
    
    Pseudocode:
    1. Collect trajectories with current policy
    2. Compute returns (cumulative discounted rewards)
    3. loss = -sum(log_prob(action) * return)
    4. Backprop and update
    """
    raise NotImplementedError("Policy gradient training not yet implemented")


def train_dqn(
    config: dict,
    mode: str = 'ground_truth',
    episodes: int = 100,
    output_path: str = 'models/dqn_model.pth',
) -> TinyCNN:
    """
    Train using DQN (Deep Q-Network).
    
    TODO: Implement for full RL support.
    
    Requires:
    - Target network (updated periodically)
    - Experience replay
    - Epsilon-greedy exploration decay
    """
    raise NotImplementedError("DQN training not yet implemented")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train learned behaviors")
    parser.add_argument('--mode', choices=['ground_truth', 'proxy', 'both'], 
                        default='both', help='Which mode to train')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--collection-steps', type=int, default=1000, 
                        help='Simulation steps for data collection')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output path (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    setup_logging()
    config = get_config()
    
    if args.mode == 'both':
        train_ground_truth_and_proxy(
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            collection_steps=args.collection_steps,
        )
    else:
        output = args.output or f'models/{args.mode}.pth'
        train_behavior_cloning(
            config=config,
            mode=args.mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            collection_steps=args.collection_steps,
            output_path=output,
        )


if __name__ == "__main__":
    main()
