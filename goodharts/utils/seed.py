"""
Seed management for reproducible experiments.

Provides utilities to set random seeds across all random sources
(Python, NumPy, PyTorch CPU/CUDA) for reproducible training and evaluation.

Usage:
    from goodharts.utils.seed import set_seed, get_random_seed

    # Use specific seed
    seed = set_seed(42)

    # Generate and use random seed (logged for reproducibility)
    seed = set_seed(None)  # Returns the seed that was used
    print(f"Using seed: {seed}")

    # Full determinism (slower, for debugging/evaluation)
    seed = set_seed(42, deterministic=True)

    # Training mode: set seeds but allow non-deterministic ops like multinomial
    seed = set_seed(42, deterministic=True, warn_only=True)

Determinism Notes:
    For CUDA: Requires CUBLAS_WORKSPACE_CONFIG=:4096:8 (set automatically).
    For ROCm: HIP respects use_deterministic_algorithms without special env vars.
    Operations like torch.multinomial have no deterministic CUDA implementation;
    use warn_only=True for training, or use argmax for deterministic evaluation.
"""
import os
import random
import time
import warnings
from typing import Optional

import numpy as np
import torch


def get_random_seed() -> int:
    """
    Generate a random seed from system entropy.

    Uses os.urandom for cryptographic randomness, falling back to
    time-based seed if unavailable.

    Returns:
        Random integer suitable for seeding
    """
    try:
        # Use 4 bytes of system entropy (gives us a 32-bit seed)
        return int.from_bytes(os.urandom(4), byteorder='little')
    except NotImplementedError:
        # Fallback for systems without os.urandom
        return int(time.time() * 1000) % (2**32)


def set_seed(
    seed: Optional[int] = None,
    deterministic: bool = False,
    warn_only: bool = False,
    verbose: bool = False,
) -> int:
    """
    Set random seeds for all random sources.

    Sets seeds for:
    - Python's random module
    - NumPy's random
    - PyTorch CPU
    - PyTorch CUDA/ROCm (if available)

    Args:
        seed: Seed value. If None, generates a random seed.
        deterministic: If True, enables PyTorch deterministic mode.
                      This may significantly impact performance but
                      guarantees reproducibility across runs.
        warn_only: If True with deterministic=True, warn instead of error
                   when non-deterministic ops are encountered. Use this for
                   training (which needs multinomial sampling) while still
                   setting all other determinism flags.
        verbose: If True, print the seed being used.

    Returns:
        The seed that was used (useful when seed=None)

    Note:
        Even with the same seed, results may differ between:
        - Different PyTorch versions
        - CPU vs GPU execution
        - Different GPU architectures

        For exact reproducibility, also control:
        - PyTorch version
        - CUDA/ROCm version
        - Hardware
    """
    if seed is None:
        seed = get_random_seed()

    if verbose:
        print(f"[Seed] Using seed: {seed}")

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA/ROCm (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Deterministic mode (optional, impacts performance)
    if deterministic:
        # cuBLAS workspace config required for some deterministic ops on CUDA
        # ROCm/HIP doesn't need this but it's harmless to set
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            if warn_only:
                # Training mode: warn about non-deterministic ops but don't fail
                # This allows multinomial sampling while keeping other ops deterministic
                torch.use_deterministic_algorithms(True, warn_only=True)
                if verbose:
                    print("[Seed] Deterministic mode (warn_only): will warn on non-deterministic ops")
            else:
                # Strict mode: fail on non-deterministic ops
                # Use this for evaluation with argmax (no multinomial needed)
                try:
                    torch.use_deterministic_algorithms(True)
                except RuntimeError as e:
                    # This shouldn't happen at seed time, but could happen later
                    # if the function is called after non-deterministic ops are queued
                    warnings.warn(
                        f"Could not enable full determinism: {e}. "
                        "Use warn_only=True for training, or ensure no non-deterministic "
                        "ops are pending."
                    )
        if verbose:
            print("[Seed] Deterministic mode enabled (may impact performance)")

    return seed


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize DataLoader worker with unique seed.

    Use this as the worker_init_fn for DataLoader to ensure
    each worker has a different random state.

    Example:
        loader = DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn)
    """
    # Derive worker seed from base seed + worker_id
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
