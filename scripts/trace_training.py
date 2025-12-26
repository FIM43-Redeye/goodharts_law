#!/usr/bin/env python3
"""
Native torch.profiler trace of the training loop.

Generates a Chrome trace (JSON) that can be opened in:
- chrome://tracing
- https://ui.perfetto.dev

This shows exactly what kernels are running, when, and where gaps exist.

Usage:
    python scripts/trace_training.py              # Default: 8 updates
    python scripts/trace_training.py --updates 4  # Custom update count
"""
import argparse
import os
import torch
from torch.profiler import profile, ProfilerActivity, schedule

# Ensure trace directory exists
os.makedirs('profile_trace', exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Profile training with torch.profiler')
    parser.add_argument('--updates', type=int, default=8, help='Number of updates to profile')
    parser.add_argument('--envs', type=int, default=192, help='Number of environments')
    parser.add_argument('--steps', type=int, default=128, help='Steps per environment per update')
    args = parser.parse_args()

    from goodharts.training.ppo.trainer import PPOTrainer, PPOConfig

    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name()}")

    # Calculate exact timesteps: n_envs * steps_per_env * n_updates
    n_updates = args.updates
    n_envs = args.envs
    steps_per_env = args.steps
    total_timesteps = n_envs * steps_per_env * n_updates

    print(f"\nProfile configuration:")
    print(f"  Updates: {n_updates}")
    print(f"  Envs: {n_envs}")
    print(f"  Steps/env: {steps_per_env}")
    print(f"  Total timesteps: {total_timesteps:,}")

    config = PPOConfig.from_config(
        mode='ground_truth',
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        steps_per_env=steps_per_env,
        benchmark_mode=True,
        profile_enabled=False,  # Disable our profiler, use torch.profiler instead
    )

    trainer = PPOTrainer(config)
    trainer._setup()

    # Warmup first (without profiling)
    print("Warming up (compilation)...")
    states = trainer.vec_env.reset()
    trainer.reward_computer.initialize(states)

    # Run warmup update
    from torch.amp import autocast
    from torch.distributions import Categorical

    cfg = trainer.config

    for _ in range(cfg.steps_per_env):
        with torch.no_grad():
            if trainer._compiled_inference is not None:
                logits, features, values = trainer._compiled_inference(states)
                actions, log_probs = trainer._compiled_sample(logits)
            else:
                states_t = states.float()
                with autocast(device_type=trainer.device_type, enabled=cfg.use_amp):
                    logits, features = trainer.policy.forward_with_features(states_t)
                    values = trainer.value_head(features).squeeze(-1)
                    dist = Categorical(logits=logits, validate_args=False)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)

        states, _, _ = trainer.vec_env.step(actions)

    torch.cuda.synchronize()
    print("Warmup complete.\n")

    # Now profile
    print("\n" + "=" * 75)
    print(f"PROFILING {n_updates} UPDATES WITH torch.profiler")
    print("=" * 75)

    states = trainer.vec_env.reset()
    trainer.reward_computer.initialize(states)

    # Profile schedule: active for all updates
    # Each update = steps_per_env steps
    total_profile_steps = steps_per_env * n_updates + 10  # +10 for safety margin

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=0, active=total_profile_steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('profile_trace'),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:

        update_count = 0

        while update_count < n_updates:
            # Collection phase
            states_buffer = []
            actions_buffer = []
            log_probs_buffer = []
            rewards_buffer = []
            dones_buffer = []
            values_buffer = []
            episode_rewards = torch.zeros(n_envs, device=device)

            with torch.no_grad():
                for step in range(steps_per_env):
                    prof.step()  # Mark each step for profiler

                    # Inference
                    if trainer._compiled_inference is not None:
                        logits, features, values = trainer._compiled_inference(states)
                        actions, log_probs = trainer._compiled_sample(logits)
                    else:
                        states_t = states.float()
                        with autocast(device_type=trainer.device_type, enabled=cfg.use_amp):
                            logits, features = trainer.policy.forward_with_features(states_t)
                            values = trainer.value_head(features).squeeze(-1)
                            dist = Categorical(logits=logits, validate_args=False)
                            actions = dist.sample()
                            log_probs = dist.log_prob(actions)

                    # Env step
                    current_states = states.clone()
                    next_states, rewards, dones = trainer.vec_env.step(actions)
                    shaped_rewards = trainer.reward_computer.compute(
                        rewards, current_states, next_states, dones
                    )

                    # Store
                    states_buffer.append(current_states)
                    actions_buffer.append(actions)
                    log_probs_buffer.append(log_probs.detach())
                    rewards_buffer.append(shaped_rewards)
                    dones_buffer.append(dones.clone())
                    values_buffer.append(values)

                    episode_rewards += rewards
                    episode_rewards *= (~dones)
                    states = next_states

            # GAE
            with torch.no_grad():
                if trainer._compiled_inference is not None:
                    _, features, next_value = trainer._compiled_inference(states)
                else:
                    states_t = states.float()
                    with autocast(device_type=trainer.device_type, enabled=cfg.use_amp):
                        _, features = trainer.policy.forward_with_features(states_t)
                        next_value = trainer.value_head(features).squeeze(-1)

            from goodharts.training.ppo.algorithms import compute_gae, ppo_update

            advantages, returns = compute_gae(
                rewards_buffer, values_buffer, dones_buffer,
                next_value, cfg.gamma, cfg.gae_lambda, device=device
            )

            # Flatten
            all_states = torch.cat(states_buffer, dim=0)
            all_actions = torch.cat(actions_buffer, dim=0)
            all_log_probs = torch.cat(log_probs_buffer, dim=0)
            all_old_values = torch.cat(values_buffer, dim=0)
            all_returns = returns.flatten()
            all_advantages = advantages.flatten()
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

            # PPO Update
            ppo_update(
                trainer.policy, trainer.value_head, trainer.optimizer,
                all_states, all_actions, all_log_probs,
                all_returns, all_advantages, all_old_values,
                device,
                eps_clip=cfg.eps_clip,
                k_epochs=cfg.k_epochs,
                entropy_coef=cfg.entropy_coef,
                value_coef=cfg.value_coef,
                n_minibatches=cfg.n_minibatches,
                scaler=trainer.scaler,
            )

            update_count += 1
            print(f"Update {update_count}/{n_updates} complete")

    print("\n" + "=" * 75)
    print("PROFILE COMPLETE")
    print("=" * 75)
    print("\nTrace saved to: profile_trace/")
    print("\nTo view:")
    print("  1. Find the .json file in profile_trace/")
    print("  2. Open chrome://tracing in Chrome/Chromium")
    print("  3. Click 'Load' and select the .json file")
    print("  OR")
    print("  1. Go to https://ui.perfetto.dev")
    print("  2. Drag and drop the .json file")
    print("\nLook for gaps between GPU kernels - those are the burstiness sources!")


if __name__ == "__main__":
    main()
