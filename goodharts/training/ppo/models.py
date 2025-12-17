"""
Neural network components for PPO training.

Contains the value head for the critic and training utilities.
"""
import time
import torch
import torch.nn as nn

from goodharts.utils.device import get_device


class ValueHead(nn.Module):
    """Simple value head that attaches to BaseCNN features."""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class Profiler:
    """
    Simple accumulated timer for profiling training loop components.
    
    Usage:
        profiler = Profiler(device)
        profiler.start()
        ... some work ...
        profiler.tick("step_1")
        ... more work ...
        profiler.tick("step_2")
        print(profiler.summary())
    """
    
    def __init__(self, device: torch.device = None, enabled: bool = True):
        self.times: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.last_t: float = 0
        self.enabled = enabled
        if device is None:
            device = get_device()
        # Only synchronize if we are actually using a CUDA device
        self.sync_cuda = (device.type == 'cuda')

    def start(self):
        """Start the timer."""
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.last_t = time.perf_counter()

    def tick(self, name: str):
        """Record elapsed time since last tick under the given name."""
        if not self.enabled:
            return  # No-op for production runs
        if self.sync_cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        dt = now - self.last_t
        self.times[name] = self.times.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1
        self.last_t = now
    
    def reset(self):
        """Clear all recorded times."""
        self.times = {}
        self.counts = {}

    def summary(self) -> str:
        """Return a formatted summary of profiled times."""
        total = sum(self.times.values())
        if total == 0:
            return "No data"
        parts = []
        # Sort by duration (descending)
        for k, v in sorted(self.times.items(), key=lambda x: x[1], reverse=True):
            pct = v / total * 100
            parts.append(f"{k}: {v:.2f}s ({pct:.0f}%)")
        return " | ".join(parts)
