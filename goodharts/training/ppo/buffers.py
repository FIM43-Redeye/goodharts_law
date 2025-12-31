"""
Pre-allocated GPU buffers for PPO training.

Provides typed buffer containers and allocation functions for rollout collection.
All buffers are GPU tensors that get overwritten each update cycle.

Usage:
    from .buffers import RolloutBuffers, allocate_rollout_buffers

    buffers = allocate_rollout_buffers(
        n_envs=256,
        steps_per_env=128,
        n_channels=2,
        view_size=7,
        device=device,
        num_aux_inputs=3,
    )
"""
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RolloutBuffers:
    """
    Pre-allocated GPU tensors for rollout collection.

    These buffers are reused each update cycle to avoid allocation overhead.
    Shape convention: (steps_per_env, n_envs, ...) for rollout data.

    Attributes:
        states: Observation tensors [steps, envs, channels, H, W]
        actions: Selected actions [steps, envs]
        log_probs: Log probabilities of actions [steps, envs]
        rewards: Shaped rewards [steps, envs]
        dones: Episode termination flags [steps, envs]
        terminated: True termination (not truncation) [steps, envs]
        values: Value estimates [steps, envs]
        aux: Auxiliary critic inputs [steps, envs, aux_dim] (optional)
        finished_dones: Episode completion tracking for async logging [steps, envs]
        finished_rewards: Final rewards for completed episodes [steps, envs]
    """
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    terminated: torch.Tensor
    values: torch.Tensor
    aux: Optional[torch.Tensor]
    finished_dones: torch.Tensor
    finished_rewards: torch.Tensor


@dataclass
class MetricsConstants:
    """
    Pre-allocated constants for GPU metrics aggregation.

    Avoids tensor creation overhead during BUFFER_FLATTEN operations.
    """
    inf: torch.Tensor
    neg_inf: torch.Tensor


def allocate_rollout_buffers(
    n_envs: int,
    steps_per_env: int,
    n_channels: int,
    view_size: int,
    device: torch.device,
    num_aux_inputs: int = 0,
) -> RolloutBuffers:
    """
    Allocate pre-sized GPU tensors for rollout collection.

    Must be called before torch.compile block since closures capture
    these tensor references at compile time.

    Args:
        n_envs: Number of parallel environments
        steps_per_env: Steps collected per environment per update
        n_channels: Number of observation channels
        view_size: Observation spatial size (height/width)
        device: Torch device for allocation
        num_aux_inputs: Number of auxiliary critic inputs (0 if no privileged critic)

    Returns:
        RolloutBuffers containing all pre-allocated tensors
    """
    obs_shape = (steps_per_env, n_envs, n_channels, view_size, view_size)
    scalar_shape = (steps_per_env, n_envs)

    # Core rollout buffers
    states = torch.zeros(obs_shape, device=device, dtype=torch.uint8)
    actions = torch.zeros(scalar_shape, device=device, dtype=torch.long)
    log_probs = torch.zeros(scalar_shape, device=device, dtype=torch.float32)
    rewards = torch.zeros(scalar_shape, device=device, dtype=torch.float32)
    dones = torch.zeros(scalar_shape, device=device, dtype=torch.bool)
    terminated = torch.zeros(scalar_shape, device=device, dtype=torch.bool)
    values = torch.zeros(scalar_shape, device=device, dtype=torch.float32)

    # Privileged critic auxiliary inputs
    if num_aux_inputs > 0:
        aux_shape = (steps_per_env, n_envs, num_aux_inputs)
        aux = torch.zeros(aux_shape, device=device, dtype=torch.float32)
    else:
        aux = None

    # Episode tracking for async logging
    finished_dones = torch.zeros(scalar_shape, device=device, dtype=torch.bool)
    finished_rewards = torch.zeros(scalar_shape, device=device, dtype=torch.float32)

    return RolloutBuffers(
        states=states,
        actions=actions,
        log_probs=log_probs,
        rewards=rewards,
        dones=dones,
        terminated=terminated,
        values=values,
        aux=aux,
        finished_dones=finished_dones,
        finished_rewards=finished_rewards,
    )


def allocate_metrics_constants(device: torch.device) -> MetricsConstants:
    """
    Allocate pre-computed constants for GPU metrics aggregation.

    These avoid tensor creation overhead during BUFFER_FLATTEN operations.

    Args:
        device: Torch device for allocation

    Returns:
        MetricsConstants with inf and neg_inf tensors
    """
    return MetricsConstants(
        inf=torch.tensor(float('inf'), device=device),
        neg_inf=torch.tensor(float('-inf'), device=device),
    )
