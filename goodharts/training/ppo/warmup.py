"""
JIT and cuDNN warmup for PPO training.

Provides functions to trigger lazy initialization of torch.compile graphs,
cuDNN algorithm selection, and MIOpen kernel caching. Running these once
at startup avoids compilation penalties during actual training.

Usage:
    from .warmup import warmup_forward_backward, run_warmup_update

    # In setup:
    warmup_forward_backward(policy, value_head, optimizer, ...)

    # Before training loop:
    states = run_warmup_update(policy, value_head, vec_env, ...)
"""
import time
from typing import Optional, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.distributions import Categorical

from .algorithms import compute_gae, ppo_update


@dataclass
class WarmupBuffers:
    """Pre-allocated buffers needed for warmup update."""
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    terminated: torch.Tensor
    values: torch.Tensor
    aux: Optional[torch.Tensor]


@dataclass
class CompiledFunctions:
    """Container for compiled functions used in warmup."""
    rollout_step: Optional[Callable]
    ppo_update: Optional[Callable]
    gae: Optional[Callable]


def warmup_forward_backward(
    policy: nn.Module,
    value_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    device_type: str,
    batch_size: int,
    n_channels: int,
    view_size: int,
    use_amp: bool = False,
    include_backward: bool = True,
    label: str = "Warmup",
) -> float:
    """
    Run forward (and optionally backward) pass to warm up JIT/cuDNN.

    This triggers algorithm selection for cuDNN benchmark mode and compiles
    torch.compile graphs. Running this once at startup avoids the compilation
    penalty during actual training.

    Args:
        policy: Policy network
        value_head: Value head network
        optimizer: Optimizer (for gradient clearing)
        device: Torch device
        device_type: Device type string ('cuda', 'cpu', etc.)
        batch_size: Batch size for dummy tensors (should match training batch)
        n_channels: Number of observation channels
        view_size: Observation spatial size
        use_amp: Whether to use automatic mixed precision
        include_backward: Whether to run backward pass (needed for gradient kernels)
        label: Label for progress messages

    Returns:
        Time taken in seconds
    """
    start_time = time.time()

    # Create dummy tensors matching training shapes
    dummy_obs = torch.zeros(
        (batch_size, n_channels, view_size, view_size),
        device=device, requires_grad=False
    )

    with autocast(device_type=device_type, enabled=use_amp):
        logits = policy(dummy_obs)
        features = policy.get_features(dummy_obs)
        values = value_head(features).squeeze(-1)

        if include_backward:
            dummy_actions = torch.zeros(batch_size, dtype=torch.long, device=device)
            dummy_returns = torch.zeros(batch_size, device=device)

            dist = Categorical(logits=logits, validate_args=False)
            log_probs = dist.log_prob(dummy_actions)
            dummy_loss = -log_probs.mean() + F.mse_loss(values, dummy_returns)
            dummy_loss.backward()

    # Synchronize to ensure kernels complete
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Clean up
    if include_backward:
        policy.zero_grad()
        value_head.zero_grad()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return time.time() - start_time


def run_warmup_update(
    policy: nn.Module,
    value_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    vec_env,  # TorchVecEnv - avoid circular import
    reward_computer,  # RewardComputer - avoid circular import
    states: torch.Tensor,
    episode_rewards: torch.Tensor,
    buffers: WarmupBuffers,
    compiled_fns: CompiledFunctions,
    device: torch.device,
    device_type: str,
    steps_per_env: int,
    n_envs: int,
    gamma: float,
    gae_lambda: float,
    eps_clip: float,
    k_epochs: int,
    entropy_coef: float,
    value_coef: float,
    n_minibatches: int,
    use_amp: bool = False,
    entropy_floor: float = 0.0,
    entropy_floor_penalty: float = 0.0,
) -> torch.Tensor:
    """
    Run one full training update as warmup (results discarded).

    This triggers all lazy initialization in a realistic context:
    - Real environment steps (not synthetic data)
    - Full forward/backward through compiled models
    - All MIOpen/cuDNN algorithm selection

    Args:
        policy: Policy network
        value_head: Value head network
        optimizer: Optimizer
        scaler: AMP gradient scaler (or None)
        vec_env: Vectorized environment
        reward_computer: Reward shaping computer
        states: Current environment states
        episode_rewards: Episode reward accumulator
        buffers: Pre-allocated buffer tensors
        compiled_fns: Compiled function container (rollout_step, ppo_update, gae)
        device: Torch device
        device_type: Device type string
        steps_per_env: Steps per environment per update
        n_envs: Number of parallel environments
        gamma: Discount factor
        gae_lambda: GAE lambda
        eps_clip: PPO clip epsilon
        k_epochs: PPO epochs per update
        entropy_coef: Entropy coefficient
        value_coef: Value loss coefficient
        n_minibatches: Number of minibatches
        use_amp: Whether to use AMP
        entropy_floor: Minimum entropy threshold
        entropy_floor_penalty: Penalty for entropy below floor

    Returns:
        New states after the warmup steps
    """
    # Unpack buffers
    states_buf = buffers.states
    actions_buf = buffers.actions
    log_probs_buf = buffers.log_probs
    rewards_buf = buffers.rewards
    dones_buf = buffers.dones
    terminated_buf = buffers.terminated
    values_buf = buffers.values
    aux_buf = buffers.aux

    # Initial inference to get first actions
    potentials = reward_computer.get_initial_potentials(states)
    critic_aux = vec_env.get_critic_aux() if aux_buf is not None else None

    with torch.no_grad():
        with autocast(device_type=device_type, enabled=use_amp):
            logits, features = policy.forward_with_features(states.float())
            dist = Categorical(logits=logits, validate_args=False)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            values = value_head(features, critic_aux).squeeze(-1)

    for step in range(steps_per_env):
        with torch.no_grad():
            # Use compiled rollout step if available (triggers CUDA graph capture)
            if compiled_fns.rollout_step is not None:
                torch.compiler.cudagraph_mark_step_begin()
                (
                    next_states, next_actions, next_log_probs, next_values,
                    next_potentials, logits,
                    _current_states, _shaped_rewards, _dones, _terminated, _critic_aux,
                    _finished_episode_rewards
                ) = compiled_fns.rollout_step(
                    actions, log_probs, values, states, potentials
                )
                # Warmup doesn't need to write buffers, just exercise the compiled path

                # Clone outputs to prevent CUDA graph buffer reuse issues
                states = next_states.clone()
                actions = next_actions.clone()
                log_probs = next_log_probs.clone()
                values = next_values.clone()
                potentials = next_potentials.clone()
            else:
                # Fallback: eager mode
                critic_aux = vec_env.get_critic_aux() if aux_buf is not None else None
                with autocast(device_type=device_type, enabled=use_amp):
                    logits, features = policy.forward_with_features(states.float())
                    dist = Categorical(logits=logits, validate_args=False)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                    values = value_head(features, critic_aux).squeeze(-1)

                current_states = states.clone()
                next_states, eating_info, terminated, truncated = vec_env.step(actions)
                dones = terminated | truncated
                shaped_rewards = reward_computer.compute(eating_info, current_states, next_states, terminated)

                # Store in pre-allocated tensor buffers
                states_buf[step] = current_states
                actions_buf[step] = actions
                log_probs_buf[step] = log_probs.detach()
                rewards_buf[step] = shaped_rewards
                dones_buf[step] = dones
                terminated_buf[step] = terminated
                values_buf[step] = values
                if aux_buf is not None:
                    aux_buf[step] = critic_aux

                episode_rewards += shaped_rewards
                episode_rewards *= (~dones)
                states = next_states

    # Bootstrap value
    with torch.no_grad():
        states_t = states.float()
        _, features = policy.forward_with_features(states_t)
        bootstrap_aux = vec_env.get_critic_aux() if aux_buf is not None else None
        next_value = value_head(features, bootstrap_aux).squeeze(-1)

    # Compute GAE
    gae_fn = compiled_fns.gae if compiled_fns.gae is not None else compute_gae
    advantages, returns = gae_fn(
        rewards_buf, values_buf, terminated_buf,
        next_value, gamma, gae_lambda, device=device
    )

    # Update PopArt statistics for value normalization
    if hasattr(value_head, 'update_stats'):
        value_head.update_stats(returns.flatten(), optimizer)

    # PPO update
    batch_size = steps_per_env * n_envs
    view_size = vec_env.view_size
    all_states = states_buf.reshape(batch_size, -1, view_size, view_size).float()
    all_actions = actions_buf.flatten()
    all_log_probs = log_probs_buf.flatten()
    all_values = values_buf.flatten()
    all_returns = returns.flatten()
    all_advantages = advantages.flatten()

    # Normalize advantages
    all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

    # Use compiled version if available
    all_aux = aux_buf.reshape(batch_size, -1) if aux_buf is not None else None
    ppo_fn = compiled_fns.ppo_update if compiled_fns.ppo_update is not None else ppo_update
    ppo_fn(
        policy, value_head, optimizer,
        all_states, all_actions, all_log_probs,
        all_returns, all_advantages, all_values,
        device,
        eps_clip=eps_clip,
        k_epochs=k_epochs,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        n_minibatches=n_minibatches,
        scaler=scaler,
        aux_inputs=all_aux,
        entropy_floor=entropy_floor,
        entropy_floor_penalty=entropy_floor_penalty,
    )

    # Sync to ensure all kernels complete
    if device.type == 'cuda':
        torch.cuda.synchronize()

    return states
