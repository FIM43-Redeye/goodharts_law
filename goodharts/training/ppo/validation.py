"""
Validation episode runner for PPO training.

Provides deterministic evaluation of policy performance during training
using argmax action selection (no exploration noise).

Usage:
    from .validation import run_validation_episodes

    # In trainer:
    metrics = run_validation_episodes(
        policy=self.policy,
        vec_env=self.vec_env,
        reward_computer=self.reward_computer,
        config=self.config,
        spec=self.spec,
        device=self.device,
    )
"""

import torch
import torch.nn as nn

from ...modes import ObservationSpec


def run_validation_episodes(
    policy: nn.Module,
    vec_env,  # TorchVecEnv - avoid circular import
    reward_computer,  # RewardComputer - avoid circular import
    n_envs: int,
    validation_episodes: int,
    validation_mode: str,
    validation_food: int,
    validation_poison: int,
    spec: ObservationSpec,
    device: torch.device,
    create_env_fn=None,  # Optional factory for fixed-density validation env
    verbose: bool = True,
) -> dict:
    """
    Run deterministic validation episodes.

    Uses argmax for action selection (no exploration) to evaluate
    the learned policy's actual performance.

    Args:
        policy: Policy network to evaluate
        vec_env: Vectorized training environment (used if validation_mode != "fixed")
        reward_computer: Reward shaping computer
        n_envs: Number of parallel environments
        validation_episodes: Number of episodes to run
        validation_mode: "fixed" for controlled density, "training" for same as training
        validation_food: Fixed food count for validation_mode="fixed"
        validation_poison: Fixed poison count for validation_mode="fixed"
        spec: Observation specification
        device: Torch device
        create_env_fn: Optional callable to create fixed-density env
        verbose: Whether to print validation results

    Returns:
        Dict with mean reward, food, poison, episode count, and mode
    """
    # Select environment based on mode
    if validation_mode == "fixed" and create_env_fn is not None:
        # Create temporary env with fixed density
        val_env = create_env_fn(
            n_envs=n_envs,
            obs_spec=spec,
            device=device
        )
        val_env.set_curriculum_ranges(
            validation_food, validation_food,
            validation_poison, validation_poison
        )
        val_env.reset()
    else:
        # Use training env directly (faster, same randomization)
        val_env = vec_env

    # Run episodes
    total_reward = 0.0
    total_food = 0
    total_poison = 0
    episodes_done = 0

    states = val_env.reset()
    episode_rewards = torch.zeros(n_envs, device=device)

    # Run until we have enough completed episodes
    max_steps = n_envs * 600  # Safety limit
    steps = 0

    with torch.no_grad():
        while episodes_done < validation_episodes and steps < max_steps:
            states_t = states.float()

            # Deterministic action selection (argmax instead of sample)
            logits = policy(states_t)
            actions = logits.argmax(dim=-1)

            next_states, eating_info, terminated, truncated = val_env.step(actions)
            dones = terminated | truncated
            shaped_rewards = reward_computer.compute(eating_info, states, next_states, terminated)
            episode_rewards += shaped_rewards

            # Check for completed episodes
            done_mask = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_mask.numel() > 0:
                for idx in done_mask:
                    if episodes_done >= validation_episodes:
                        break
                    total_reward += episode_rewards[idx].item()
                    total_food += val_env.last_episode_food[idx].item()
                    total_poison += val_env.last_episode_poison[idx].item()
                    episodes_done += 1
                    episode_rewards[idx] = 0.0

            states = next_states
            steps += n_envs

    # Compute means
    n = max(episodes_done, 1)
    metrics = {
        'reward': total_reward / n,
        'food': total_food / n,
        'poison': total_poison / n,
        'episodes': episodes_done,
        'mode': validation_mode,
    }

    # Print validation results
    if verbose:
        print(f"   [Validation] reward={metrics['reward']:.2f}, "
              f"food={metrics['food']:.1f}, poison={metrics['poison']:.1f} "
              f"({episodes_done} episodes, {validation_mode} env)")

    return metrics
