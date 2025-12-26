"""
Core PPO algorithm implementations.

Pure functions for GAE calculation and PPO clipped objective updates.
No state - easy to test and reuse.
"""
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.distributions import Categorical


def compute_gae(
    rewards: torch.Tensor | list[torch.Tensor],
    values: torch.Tensor | list[torch.Tensor],
    dones: torch.Tensor | list[torch.Tensor],
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    device: torch.device = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Accepts either pre-allocated tensors (T, n_envs) or lists of tensors.
    Using tensors directly avoids torch.stack overhead.

    Args:
        rewards: Tensor (T, n_envs) or list of (n_envs,) tensors
        values: Tensor (T, n_envs) or list of (n_envs,) tensors
        dones: Tensor (T, n_envs) or list of (n_envs,) tensors
        next_value: Bootstrap value for final state, shape (n_envs,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        device: Torch device (inferred from rewards if not provided)

    Returns:
        advantages: Shape (T, n_envs)
        returns: Shape (T, n_envs)
    """
    # Accept either pre-stacked tensors or lists (for backwards compatibility)
    if isinstance(rewards, list):
        rewards_t = torch.stack(rewards)      # (T, n_envs)
        values_t = torch.stack(values)        # (T, n_envs)
        dones_t = torch.stack(dones).float()  # (T, n_envs)
    else:
        rewards_t = rewards
        values_t = values
        dones_t = dones.float() if dones.dtype == torch.bool else dones
    
    device = device or rewards_t.device
    T, n_envs = rewards_t.shape
    
    # Pre-compute masks: 1 where episode continues, 0 where it ends
    masks = 1.0 - dones_t  # (T, n_envs)
    
    # Build next_values tensor: values shifted by 1, with bootstrap at end
    # next_values[t] = values[t+1] for t < T-1, next_value for t = T-1
    next_values = torch.cat([values_t[1:], next_value.unsqueeze(0)], dim=0)  # (T, n_envs)
    
    # Compute all TD residuals (deltas) in one vectorized operation
    # delta[t] = r[t] + gamma * V(s[t+1]) * mask[t] - V(s[t])
    deltas = rewards_t + gamma * next_values * masks - values_t  # (T, n_envs)
    
    # GAE recurrence: A[t] = delta[t] + gamma * lambda * mask[t] * A[t+1]
    # Must iterate backwards since each step depends on the next
    # But now we're iterating over tensor slices, not Python objects
    advantages = torch.zeros_like(deltas)  # (T, n_envs)
    gae_coef = gamma * gae_lambda
    lastgae = torch.zeros(n_envs, device=device, dtype=torch.float32)
    
    for t in range(T - 1, -1, -1):
        lastgae = deltas[t] + gae_coef * masks[t] * lastgae
        advantages[t] = lastgae
    
    # Returns = Advantages + Values
    returns = advantages + values_t
    
    return advantages, returns


def ppo_update(
    policy: torch.nn.Module,
    value_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    old_values: torch.Tensor,
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
    
    All inputs should be torch tensors already on the target device.
    """
    use_amp = scaler is not None
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    # Ensure correct dtypes (tensors should already be on device)
    old_states_d = states.float()
    old_actions_d = actions.long()
    old_log_probs_d = log_probs.float()
    returns_d = returns.float()
    advantages_d = advantages.float()
    old_values_d = old_values.float()

    batch_size = states.shape[0]
    minibatch_size = batch_size // n_minibatches

    # Track metrics on GPU to avoid sync during loop
    total_policy_loss = torch.tensor(0.0, device=device)
    total_value_loss = torch.tensor(0.0, device=device)
    total_entropy = torch.tensor(0.0, device=device)
    n_updates = 0

    for epoch in range(k_epochs):
        # Shuffle indices for this epoch
        indices = torch.argsort(torch.rand(batch_size, device=device))
        
        for mb_idx in range(n_minibatches):
            start = mb_idx * minibatch_size
            end = start + minibatch_size
            mb_inds = indices[start:end]
            
            # Slice minibatch
            mb_states = old_states_d[mb_inds]
            mb_actions = old_actions_d[mb_inds]
            mb_log_probs = old_log_probs_d[mb_inds]
            mb_returns = returns_d[mb_inds]
            mb_advantages = advantages_d[mb_inds]
            mb_old_values = old_values_d[mb_inds]
            
            with autocast(device_type=device_type, enabled=use_amp):
                # CRITICAL: Compute features ONCE and reuse for both logits and values
                # Using interface method keeps this architecture-agnostic
                features = policy.get_features(mb_states)
                logits = policy.logits_from_features(features)

                dist = Categorical(logits=logits, validate_args=False)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                # Ratio and clipped surrogate
                ratios = torch.exp(new_log_probs - mb_log_probs)

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * mb_advantages

                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                # Uniform interface: works for both ValueHead and PopArtValueHead
                # PopArt normalizes values/targets; simple head passes through unchanged
                values = value_head.get_training_value(features).squeeze()
                target_returns, target_old_values = value_head.prepare_targets(
                    mb_returns, mb_old_values
                )

                v_clip = target_old_values + torch.clamp(
                    values - target_old_values, -eps_clip, eps_clip
                )
                v_loss1 = F.mse_loss(values, target_returns)
                v_loss2 = F.mse_loss(v_clip, target_returns)

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
    
            # Accumulate stats (stay on GPU, no sync)
            total_policy_loss = total_policy_loss + policy_loss.detach()
            total_value_loss = total_value_loss + value_loss.detach()
            total_entropy = total_entropy + entropy_bonus.detach()
            n_updates += 1
    
    # Compute explained variance on last minibatch
    with torch.no_grad():
        var_returns = mb_returns.var()
        if var_returns > 1e-8:
            explained_var = 1 - (mb_returns - mb_old_values).var() / var_returns
        else:
            explained_var = torch.tensor(0.0, device=device)
    
    # Return TENSORS - .item() sync happens in AsyncLogger background thread
    # This eliminates GPU stalls in the training loop
    return (
        total_policy_loss / n_updates,
        total_value_loss / n_updates,
        total_entropy / n_updates,
        explained_var
    )

