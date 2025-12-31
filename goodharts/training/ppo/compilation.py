"""
torch.compile factory for PPO training.

Creates compiled functions for rollout steps, PPO updates, and GAE computation.
The key insight is that torch.compile traces the call graph at compile time,
so functions can be defined anywhere as long as dependencies are captured.

Usage:
    from .compilation import CompiledFunctions, create_compiled_functions

    compiled = create_compiled_functions(
        policy=policy,
        value_head=value_head,
        vec_env=vec_env,
        reward_computer=reward_computer,
        compile_mode="reduce-overhead",
        ...
    )
"""
from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.distributions import Categorical

from .algorithms import compute_gae, ppo_update


@dataclass
class CompiledFunctions:
    """
    Container for torch.compile'd training functions.

    All functions are optional - if None, eager mode fallback is used.
    """
    rollout_step: Optional[Callable]
    ppo_update: Optional[Callable]
    gae: Optional[Callable]


def create_compiled_functions(
    policy: nn.Module,
    value_head: nn.Module,
    vec_env,  # TorchVecEnv - avoid circular import
    reward_computer,  # RewardComputer - avoid circular import
    device_type: str,
    use_amp: bool,
    privileged_critic: bool,
    compile_mode: str = "reduce-overhead",
    verbose: bool = True,
) -> CompiledFunctions:
    """
    Create torch.compile'd training functions.

    Defines closures that capture the passed-in objects, then compiles them.
    torch.compile traces the full call graph at compile time, so the function
    definition location doesn't matter.

    Args:
        policy: Policy network (will be captured in closure)
        value_head: Value head network (will be captured in closure)
        vec_env: Vectorized environment (will be captured in closure)
        reward_computer: Reward shaping computer (will be captured in closure)
        device_type: Device type string for autocast
        use_amp: Whether to use automatic mixed precision
        privileged_critic: Whether critic receives auxiliary info
        compile_mode: torch.compile mode (default: reduce-overhead)
        verbose: Whether to print compilation status

    Returns:
        CompiledFunctions with compiled rollout_step, ppo_update, and gae
    """
    # ============================================================
    # FUSED ROLLOUT STEP: ENV + REWARD + INFERENCE + TRACK
    # ============================================================
    # All operations compiled into ONE graph for maximum fusion.
    # This eliminates CPU dispatch overhead between operations.
    @torch.compile(mode=compile_mode)
    def compiled_rollout_step(actions, log_probs, values, states, potentials):
        """
        Fully fused rollout step with buffer storage and episode tracking.

        Args:
            actions: Current actions to execute (also stored to buffer)
            log_probs: Current log probs (stored to buffer)
            values: Current values (stored to buffer)
            states: Current observations (before step)
            potentials: Current potential values for reward shaping

        Returns:
            Tuple for next iteration:
            - next_states, next_actions, next_log_probs, next_values
            - next_potentials, logits (for action prob logging)
            - current_states, shaped_rewards, dones, terminated, critic_aux
            - finished_episode_rewards (pre-reset rewards for logging)
        """
        # Snapshot current states before env mutates them
        current_states = states.clone()

        # ENV_STEP
        next_states, eating_info, terminated, truncated = vec_env.step(actions)
        dones = terminated | truncated

        # Get critic aux (density + privileged view) for value function
        critic_aux = vec_env.get_critic_aux() if privileged_critic else None

        # REWARD_SHAPE (stateless)
        shaped_rewards, next_potentials = reward_computer.compute_stateless(
            eating_info, current_states, next_states, terminated, potentials
        )

        # INFERENCE (for next step)
        with autocast(device_type=device_type, enabled=use_amp):
            logits, features = policy.forward_with_features(next_states)
            dist = Categorical(logits=logits, validate_args=False)
            next_actions = dist.sample()
            next_log_probs = dist.log_prob(next_actions)
            next_values = value_head(features, critic_aux).squeeze(-1)

        # EPISODE_TRACK - accumulate rewards, reset on done
        # Access through vec_env (not captured separately) to maintain
        # nn.Module buffer semantics for CUDA graph compatibility
        vec_env.episode_rewards.add_(shaped_rewards)
        # Capture episode rewards BEFORE reset (for logging finished episodes)
        finished_episode_rewards = vec_env.episode_rewards.clone()
        vec_env.episode_rewards.mul_(~dones)  # Reset for done agents

        # Return all values needed for buffer writes (done outside with Python int index)
        # NOTE: Buffer writes with tensor indices cause implicit .item() calls,
        # breaking the graph into multiple regions. We return values instead.
        return (
            next_states, next_actions, next_log_probs, next_values,
            next_potentials, logits,
            # Additional returns for buffer storage (written outside compiled function)
            current_states, shaped_rewards, dones, terminated, critic_aux,
            finished_episode_rewards  # Pre-reset rewards for logging
        )

    # NOTE: We intentionally do NOT compile a separate inference function.
    # Having two compiled functions (compiled_rollout_step + compiled_inference)
    # that share the same model weights causes segfaults on ROCm when weights
    # are modified between calls (PPO update). The prefetch uses eager mode,
    # which is fine since it's only 1 call per update vs 128 for the rollout.

    # Compile PPO update function
    compiled_ppo = torch.compile(ppo_update, mode=compile_mode)

    # Compile GAE computation (27x speedup from loop optimization)
    # Use reduce-overhead for better CUDA graph capture of the loop
    compiled_gae = torch.compile(compute_gae, mode=compile_mode)

    if verbose:
        print(f"   [JIT] torch.compile enabled ({compile_mode}) - fused rollout step + GAE")

    return CompiledFunctions(
        rollout_step=compiled_rollout_step,
        ppo_update=compiled_ppo,
        gae=compiled_gae,
    )
