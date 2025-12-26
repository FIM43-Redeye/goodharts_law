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

    # Class attribute for compile-time branching (no hasattr needed)
    is_popart = False

    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    def get_training_value(self, features: torch.Tensor) -> torch.Tensor:
        """Return value for training loss computation (same as forward for simple head)."""
        return self.fc(features)

    def prepare_targets(
        self, returns: torch.Tensor, old_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare targets for value loss (no-op for simple head)."""
        return returns, old_values


class PopArtValueHead(nn.Module):
    """
    Value head with PopArt normalization for non-stationary returns.

    PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets)
    maintains running statistics of return values and adjusts the value head
    weights when statistics change, preventing catastrophic forgetting when
    reward scales shift during training.

    Key insight: when we update statistics (mean/std), we also inversely adjust
    the linear layer weights so the actual output remains unchanged. This lets
    the network adapt to new reward scales without losing what it learned.
    """

    # Class attribute for compile-time branching (no hasattr needed)
    is_popart = True

    def __init__(self, input_size: int, beta_min: float = 0.01):
        """
        Initialize PopArt value head.

        Args:
            input_size: Size of feature vector from policy network
            beta_min: Minimum EMA decay rate (asymptotic value after many updates)
        """
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        self.beta_min = beta_min

        # Running statistics as buffers (saved with model, moved with .to())
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('std', torch.ones(1))

        # Count-based adaptive beta: starts at 1.0 and decays to beta_min
        # This naturally handles cold start without needing special init logic
        self._update_count = 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Return DENORMALIZED value predictions.

        The linear layer outputs normalized values, which we denormalize
        using the running statistics for use in GAE/advantage computation.
        """
        normalized = self.fc(features)
        return self.mean + self.std * normalized

    def get_normalized_value(self, features: torch.Tensor) -> torch.Tensor:
        """
        Return raw (normalized) value output for training.

        Used when computing value loss against normalized targets.
        """
        return self.fc(features)

    def normalize_targets(self, returns: torch.Tensor) -> torch.Tensor:
        """Normalize returns for value function training."""
        return (returns - self.mean) / (self.std + 1e-8)

    def get_training_value(self, features: torch.Tensor) -> torch.Tensor:
        """Return normalized value for training loss computation."""
        return self.fc(features)  # Same as get_normalized_value

    def prepare_targets(
        self, returns: torch.Tensor, old_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize targets for PopArt value loss."""
        return self.normalize_targets(returns), self.normalize_targets(old_values)

    def update_stats(self, returns: torch.Tensor):
        """
        Update running statistics and adjust weights to preserve outputs.

        Uses count-based adaptive beta: starts at 1.0 (full batch init) and
        decays toward beta_min as updates accumulate. This naturally handles
        cold start without needing special init logic.

        IMPORTANT: All operations stay on GPU - no .item() or CPU transfers.
        """
        with torch.no_grad():
            self._update_count += 1

            # Adaptive beta: 1/count decays from 1.0 toward 0, clamped at beta_min
            # First update: beta=1.0 (full init), second: 0.5, third: 0.33, etc.
            beta = max(1.0 / self._update_count, self.beta_min)

            # Compute batch statistics (stays on GPU)
            batch_mean = returns.mean()
            batch_std = returns.std().clamp(min=1e-4)

            # Store old values for weight adjustment (clone stays on GPU)
            old_std = self.std.clone()
            old_mean = self.mean.clone()

            # Update running statistics with adaptive EMA
            # mean = (1 - beta) * mean + beta * batch_mean
            self.mean.mul_(1 - beta).add_(batch_mean, alpha=beta)

            # For std, we EMA the variance then sqrt
            old_var = old_std * old_std
            batch_var = batch_std * batch_std
            new_var = old_var * (1 - beta) + batch_var * beta
            self.std.copy_(new_var.sqrt().clamp(min=1e-4))

            # Adjust weights to preserve outputs: output = mean + std * (w @ x + b)
            # For output preservation: old_mean + old_std * z = new_mean + new_std * z_new
            # Solution: w_new = w * (old_std / new_std)
            #           b_new = b * (old_std / new_std) + (old_mean - new_mean) / new_std
            scale = old_std / (self.std + 1e-8)
            self.fc.weight.mul_(scale)
            self.fc.bias.mul_(scale).add_((old_mean - self.mean) / (self.std + 1e-8))


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
