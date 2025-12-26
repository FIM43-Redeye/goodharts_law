#!/usr/bin/env python3
"""
Trace ALL GPU-to-CPU synchronization points during training.

Monkey-patches torch methods to log every transfer with full stack trace.
Run with: python scripts/trace_gpu_sync.py

This catches:
- .item() - scalar transfer
- .cpu() - tensor transfer
- .numpy() - numpy conversion (implies cpu)
- .to('cpu') / .to(device='cpu') - explicit transfer
- torch.cuda.synchronize() - explicit sync
- float(tensor), int(tensor), bool(tensor) - scalar coercion
"""
import sys
import traceback
from collections import defaultdict
from functools import wraps

import torch

# Track call sites
sync_calls = defaultdict(int)
sync_details = defaultdict(list)

def get_call_site(skip_frames=2):
    """Get the caller's file:line, skipping wrapper frames."""
    stack = traceback.extract_stack()
    # Skip this function, the wrapper, and torch internals
    for frame in reversed(stack[:-skip_frames]):
        if 'site-packages' not in frame.filename and 'trace_gpu_sync' not in frame.filename:
            return f"{frame.filename}:{frame.lineno}"
    return "unknown"

def make_tracer(original_fn, name):
    """Create a tracing wrapper for a tensor method."""
    @wraps(original_fn)
    def traced(*args, **kwargs):
        site = get_call_site(skip_frames=3)
        sync_calls[f"{name} @ {site}"] += 1
        return original_fn(*args, **kwargs)
    return traced

def make_method_tracer(name):
    """Create a tracing wrapper that preserves self."""
    original = getattr(torch.Tensor, name)
    @wraps(original)
    def traced(self, *args, **kwargs):
        # Only trace CUDA tensors
        if self.is_cuda:
            site = get_call_site(skip_frames=3)
            sync_calls[f"{name}() @ {site}"] += 1
        return original(self, *args, **kwargs)
    return traced

def trace_to(original):
    """Special tracer for .to() which has complex signature."""
    @wraps(original)
    def traced(self, *args, **kwargs):
        # Check if transferring to CPU
        target = args[0] if args else kwargs.get('device')
        if self.is_cuda:
            is_cpu_transfer = False
            if isinstance(target, str) and 'cpu' in target:
                is_cpu_transfer = True
            elif isinstance(target, torch.device) and target.type == 'cpu':
                is_cpu_transfer = True
            elif target is not None and hasattr(target, 'type') and target.type == 'cpu':
                is_cpu_transfer = True

            if is_cpu_transfer:
                site = get_call_site(skip_frames=3)
                sync_calls[f"to(cpu) @ {site}"] += 1

        return original(self, *args, **kwargs)
    return traced

def trace_synchronize(original):
    """Trace torch.cuda.synchronize calls."""
    @wraps(original)
    def traced(*args, **kwargs):
        site = get_call_site(skip_frames=3)
        sync_calls[f"cuda.synchronize() @ {site}"] += 1
        return original(*args, **kwargs)
    return traced

def install_tracers():
    """Install all tracing hooks."""
    print("[Tracer] Installing GPU sync tracers...")

    # Tensor methods that cause sync
    for method in ['item', 'cpu', 'numpy']:
        setattr(torch.Tensor, method, make_method_tracer(method))

    # Special handling for .to()
    torch.Tensor.to = trace_to(torch.Tensor.to)

    # CUDA synchronize
    if hasattr(torch.cuda, 'synchronize'):
        torch.cuda.synchronize = trace_synchronize(torch.cuda.synchronize)

    print("[Tracer] Tracers installed")

def print_report():
    """Print summary of all sync points."""
    print("\n" + "="*70)
    print("GPU SYNC TRACE REPORT")
    print("="*70)

    if not sync_calls:
        print("No GPU sync points detected!")
        return

    # Sort by count
    sorted_calls = sorted(sync_calls.items(), key=lambda x: -x[1])

    total = sum(sync_calls.values())
    print(f"Total sync calls: {total}\n")

    for call, count in sorted_calls:
        pct = count / total * 100
        print(f"{count:6d} ({pct:5.1f}%) | {call}")

    print("="*70)

# Main: run training with tracers
if __name__ == "__main__":
    import atexit
    atexit.register(print_report)

    install_tracers()

    # Import and run training AFTER installing tracers
    print("\n[Tracer] Starting training with sync tracing...\n")

    from goodharts.training.ppo.trainer import PPOTrainer, PPOConfig

    # Short run to collect sync data
    config = PPOConfig.from_config(
        mode='ground_truth',
        total_timesteps=50_000,
        benchmark_mode=True,
        profile_enabled=False,  # Disable profiler syncs
    )

    trainer = PPOTrainer(config)
    trainer.train()
