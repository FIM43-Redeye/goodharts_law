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
    n_minibatches: int = 4,
    verbose: bool = False,
) -> tuple[float, float, float, float]:
    """
    Perform PPO clipped objective update with minibatches.
    """
    use_amp = scaler is not None
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    # Convert NumPy arrays to tensors ONCE
    # Note: If VRAM is tight, we should keep these on CPU and move minibatches to GPU.
    # But for now, user has a decent GPU, so keeping full batch on GPU is faster if it fits.
    # If 192 envs * 128 steps * ~5KB state is too big, move `.to(device)` inside loop.
    old_states_d = torch.from_numpy(states).float().to(device)
    old_actions_d = torch.from_numpy(actions).long().to(device)
    old_log_probs_d = torch.from_numpy(log_probs).float().to(device)
    returns_d = torch.from_numpy(returns).float().to(device)
    advantages_d = torch.from_numpy(advantages).float().to(device)
    old_values_d = torch.from_numpy(old_values).float().to(device)

    batch_size = states.shape[0]
    minibatch_size = batch_size // n_minibatches
    
    # Track metrics
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0
    
    for epoch in range(k_epochs):
        # Shuffle indices for this epoch
        indices = torch.randperm(batch_size, device=device)
        
        for mb_idx in range(n_minibatches):
            start = mb_idx * minibatch_size
            end = start + minibatch_size
            mb_inds = indices[start:end]
            
            # Progress reporting (minimal - only show epoch completion)
            # if verbose and mb_idx == n_minibatches - 1:
            #     print(f"     [GPU] Epoch {epoch+1}/{k_epochs} complete", end='\r')
            
            # Slice minibatch
            mb_states = old_states_d[mb_inds]
            mb_actions = old_actions_d[mb_inds]
            mb_log_probs = old_log_probs_d[mb_inds]
            mb_returns = returns_d[mb_inds]
            mb_advantages = advantages_d[mb_inds]
            mb_old_values = old_values_d[mb_inds]
            
            with autocast(device_type=device_type, enabled=use_amp):
                logits = policy(mb_states)
                features = policy.get_features(mb_states)
                values = value_head(features).squeeze()
                
                dist = Categorical(logits=logits, validate_args=False)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()
                
                # Ratio and clipped surrogate
                ratios = torch.exp(new_log_probs - mb_log_probs)
                
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * mb_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping
                v_clip = mb_old_values + torch.clamp(values - mb_old_values, -eps_clip, eps_clip)
                v_loss1 = F.mse_loss(values, mb_returns)
                v_loss2 = F.mse_loss(v_clip, mb_returns)
                value_loss = torch.max(v_loss1, v_loss2)
                    
                entropy_bonus = entropy.mean()
                
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
            
            # Backward pass
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
    
            # Accumulate stats
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_bonus.item()
            n_updates += 1
    
    # Compute explained variance (on full batch for accuracy, or just return 0 to save time)
    # Let's do a quick estimate on the last minibatch to save time/memory
    with torch.no_grad():
        var_returns = mb_returns.var()
        if var_returns > 1e-8:
            explained_var = 1 - (mb_returns - mb_old_values).var() / var_returns
            explained_var = explained_var.item()
        else:
            explained_var = 0.0
    
    return (
        total_policy_loss / n_updates,
        total_value_loss / n_updates,
        total_entropy / n_updates,
        explained_var
    )
