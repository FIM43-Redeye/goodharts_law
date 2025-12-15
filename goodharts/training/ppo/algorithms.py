"""
Core PPO algorithm implementations.

Pure functions for GAE calculation and PPO clipped objective updates.
No state - easy to test and reuse.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.distributions import Categorical


def compute_gae(
    rewards: list[np.ndarray],
    values: list[np.ndarray],
    dones: list[np.ndarray],
    next_value: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of reward arrays, shape [(n_envs,), ...]
        values: List of value arrays, shape [(n_envs,), ...]
        dones: List of done flags, shape [(n_envs,), ...]
        next_value: Bootstrap value for final state, shape (n_envs,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        advantages: Shape (steps, n_envs)
        returns: Shape (steps, n_envs)
    """
    steps = len(rewards)
    n_envs = rewards[0].shape[0]
    
    advantages = np.zeros((steps, n_envs), dtype=np.float32)
    lastgae = np.zeros(n_envs, dtype=np.float32)
    
    for t in reversed(range(steps)):
        if t == steps - 1:
            nextvalues = next_value
        else:
            nextvalues = values[t + 1]
        
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * nextvalues * mask - values[t]
        lastgae = delta + gamma * gae_lambda * mask * lastgae
        advantages[t] = lastgae
    
    # Returns = Advantages + Values
    returns = advantages + np.stack(values)
    
    return advantages, returns


def ppo_update(
    policy: torch.nn.Module,
    value_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    states: np.ndarray,
    actions: np.ndarray,
    log_probs: np.ndarray,
    returns: np.ndarray,
    advantages: np.ndarray,
    old_values: np.ndarray,
    device: torch.device,
    eps_clip: float = 0.2,
    k_epochs: int = 4,
    entropy_coef: float = 0.001,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    scaler: GradScaler = None,
) -> tuple[float, float, float, float]:
    """
    Perform PPO clipped objective update.
    
    Args:
        policy: Policy network (outputs logits)
        value_head: Value network head
        optimizer: Shared optimizer for both networks
        states: Observations, shape (batch, channels, h, w)
        actions: Action indices, shape (batch,)
        log_probs: Old log probabilities, shape (batch,)
        returns: Computed returns, shape (batch,)
        advantages: GAE advantages (should be normalized), shape (batch,)
        old_values: Value estimates from collection, shape (batch,)
        device: Torch device
        eps_clip: PPO clipping parameter
        k_epochs: Number of update epochs
        entropy_coef: Entropy bonus coefficient
        value_coef: Value loss coefficient
        max_grad_norm: Gradient clipping threshold
        scaler: GradScaler for AMP (None if disabled)
        
    Returns:
        (policy_loss, value_loss, entropy, explained_variance)
    """
    use_amp = scaler is not None
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    # Convert NumPy arrays to tensors
    old_states = torch.from_numpy(states).float().to(device)
    old_actions = torch.from_numpy(actions).long().to(device)
    old_log_probs = torch.from_numpy(log_probs).float().to(device)
    returns_t = torch.from_numpy(returns).float().to(device)
    advantages_t = torch.from_numpy(advantages).float().to(device)
    old_values_t = torch.from_numpy(old_values).float().to(device)

    final_policy_loss = 0.0
    final_value_loss = 0.0
    final_entropy = 0.0
    
    for _ in range(k_epochs):
        with autocast(device_type=device_type, enabled=use_amp):
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
            
            # Value loss with clipping
            v_clip = old_values_t + torch.clamp(values - old_values_t, -eps_clip, eps_clip)
            v_loss1 = F.mse_loss(values, returns_t)
            v_loss2 = F.mse_loss(v_clip, returns_t)
            value_loss = torch.max(v_loss1, v_loss2)
                
            entropy_bonus = entropy.mean()
            
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
        
        # Backward pass (outside autocast)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(value_head.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(value_head.parameters(), max_grad_norm)
            optimizer.step()

        final_policy_loss = policy_loss.item()
        final_value_loss = value_loss.item()
        final_entropy = entropy_bonus.item()
    
    # Compute explained variance
    with torch.no_grad():
        var_returns = returns_t.var()
        if var_returns > 1e-8:
            explained_var = 1 - (returns_t - old_values_t).var() / var_returns
            explained_var = explained_var.item()
        else:
            explained_var = 0.0
    
    return final_policy_loss, final_value_loss, final_entropy, explained_var
