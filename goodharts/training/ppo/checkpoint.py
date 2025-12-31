"""
Checkpoint saving for PPO training.

Provides functions to save intermediate and final training checkpoints
with full architecture metadata for architecture-agnostic model loading.

Usage:
    from .checkpoint import save_training_checkpoint, save_final_model

    # During training:
    save_training_checkpoint(
        policy=policy,
        value_head=value_head,
        optimizer=optimizer,
        ...
    )

    # At end of training:
    save_final_model(policy=policy, output_path=path, ...)
"""
import os
from typing import Optional

import torch
import torch.nn as nn

from goodharts.behaviors.brains import _clean_state_dict


def save_training_checkpoint(
    policy: nn.Module,
    value_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    mode: str,
    update_count: int,
    total_steps: int,
    best_reward: float,
    brain_type: str,
    architecture_info: dict,
    action_space_config: dict,
    seed: int,
    verbose: bool = True,
) -> str:
    """
    Save intermediate training checkpoint.

    Includes optimizer state for training resumption.

    Args:
        policy: Policy network
        value_head: Value head network
        optimizer: Optimizer with training state
        checkpoint_dir: Directory for checkpoint files
        mode: Training mode name
        update_count: Current update number
        total_steps: Total environment steps so far
        best_reward: Best episode reward seen
        brain_type: Neural network architecture type
        architecture_info: Full architecture configuration
        action_space_config: Action space configuration
        seed: Training seed

    Returns:
        Path to saved checkpoint
    """
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"checkpoint_{mode}_u{update_count}_s{total_steps}.pth"
    )

    torch.save({
        # Brain metadata (for architecture-agnostic loading)
        'brain_type': brain_type,
        'architecture': architecture_info,
        'action_space': action_space_config,
        'mode': mode,
        'seed': seed,
        # Training state
        'update': update_count,
        'total_steps': total_steps,
        'best_reward': best_reward,
        # Model weights (cleaned of torch.compile prefixes)
        'state_dict': _clean_state_dict(policy.state_dict()),
        'value_head_state_dict': _clean_state_dict(value_head.state_dict()),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    if verbose:
        print(f"   Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def save_final_model(
    policy: nn.Module,
    output_path: str,
    mode: str,
    total_steps: int,
    brain_type: str,
    architecture_info: dict,
    action_space_config: dict,
    seed: int,
    verbose: bool = True,
) -> str:
    """
    Save final trained model.

    Does not include optimizer state (not needed for inference).

    Args:
        policy: Trained policy network
        output_path: Path to save model
        mode: Training mode name
        total_steps: Total environment steps completed
        brain_type: Neural network architecture type
        architecture_info: Full architecture configuration
        action_space_config: Action space configuration
        seed: Training seed

    Returns:
        Path to saved model
    """
    checkpoint = {
        'brain_type': brain_type,
        'architecture': architecture_info,
        'action_space': action_space_config,
        'state_dict': _clean_state_dict(policy.state_dict()),
        'mode': mode,
        'training_steps': total_steps,
        'seed': seed,
    }
    torch.save(checkpoint, output_path)

    if verbose:
        print(f"\n[PPO] Training complete: {output_path} (seed={seed})")

    return output_path
