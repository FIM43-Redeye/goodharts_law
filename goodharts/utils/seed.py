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

    # Full determinism (slower, for debugging)
    seed = set_seed(42, deterministic=True)
"""
import os
import random
import time
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
    verbose: bool = False,
) -> int:
    """
    Set random seeds for all random sources.

    Sets seeds for:
    - Python's random module
    - NumPy's random
    - PyTorch CPU
    - PyTorch CUDA (if available)

    Args:
        seed: Seed value. If None, generates a random seed.
        deterministic: If True, enables PyTorch deterministic mode.
                      This may significantly impact performance but
                      guarantees reproducibility across runs.
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

    # PyTorch CUDA (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Deterministic mode (optional, impacts performance)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.8+ deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                pass
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
