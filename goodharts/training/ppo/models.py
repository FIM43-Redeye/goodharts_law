"""
Neural network components for PPO training.

Contains the value head for the critic and training utilities.
"""
import time
import torch
import torch.nn as nn

from goodharts.utils.device import get_device


class ValueHead(nn.Module):
    """
    Simple value head that attaches to BaseCNN features.

    Supports optional auxiliary inputs (e.g., episode density info) for
    privileged critic training - the value function can see info that
    the policy cannot, improving value estimation without affecting
    what the policy learns.
    """

    # Class attribute for compile-time branching (no hasattr needed)
    is_popart = False

    def __init__(self, input_size: int, num_aux_inputs: int = 0):
        """
        Initialize value head.

        Args:
            input_size: Size of feature vector from policy network
            num_aux_inputs: Number of auxiliary scalar inputs (density info, etc.)
        """
        super().__init__()
        self.num_aux_inputs = num_aux_inputs

        if num_aux_inputs > 0:
            # Small MLP to combine features + aux info
            combined_size = input_size + num_aux_inputs
            self.aux_combine = nn.Sequential(
                nn.Linear(combined_size, input_size),
                nn.ReLU(),
            )
        else:
            self.aux_combine = None
        self.fc = nn.Linear(input_size, 1)

    def forward(self, features: torch.Tensor, aux: torch.Tensor = None) -> torch.Tensor:
        """
        Compute value prediction.

        Args:
            features: Policy features (batch, hidden_size)
            aux: Optional auxiliary inputs (batch, num_aux_inputs)

        Returns:
            Value predictions (batch, 1)
        """
        if self.aux_combine is not None and aux is not None:
            combined = torch.cat([features, aux], dim=-1)
            features = self.aux_combine(combined)
        return self.fc(features)

    def get_training_value(self, features: torch.Tensor, aux: torch.Tensor = None) -> torch.Tensor:
        """Return value for training loss computation (same as forward for simple head)."""
        return self.forward(features, aux)

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

    Supports optional auxiliary inputs (e.g., episode density info) for
    privileged critic training.

    Key insight: when we update statistics (mean/std), we also inversely adjust
    the linear layer weights so the actual output remains unchanged. This lets
    the network adapt to new reward scales without losing what it learned.
    """

    # Class attribute for compile-time branching (no hasattr needed)
    is_popart = True

    def __init__(self, input_size: int, num_aux_inputs: int = 0, beta_min: float = 0.01):
        """
        Initialize PopArt value head.

        Args:
            input_size: Size of feature vector from policy network
            num_aux_inputs: Number of auxiliary scalar inputs (density info, etc.)
            beta_min: Minimum EMA decay rate (asymptotic value after many updates)
        """
        super().__init__()
        self.num_aux_inputs = num_aux_inputs
        self.beta_min = beta_min

        if num_aux_inputs > 0:
            # Small MLP to combine features + aux info
            combined_size = input_size + num_aux_inputs
            self.aux_combine = nn.Sequential(
                nn.Linear(combined_size, input_size),
                nn.ReLU(),
            )
        else:
            self.aux_combine = None

        self.fc = nn.Linear(input_size, 1)

        # Running statistics as buffers (saved with model, moved with .to())
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('std', torch.ones(1))

        # Count-based adaptive beta: starts at 1.0 and decays to beta_min
        # This naturally handles cold start without needing special init logic
        # IMPORTANT: Must be a buffer (not Python int) for:
        # 1. Proper save/restore with state_dict (fixes warmup restoration)
        # 2. CUDA graph compatibility (no Python int increment)
        self.register_buffer('_update_count', torch.zeros(1, dtype=torch.long))

    def _combine_features(self, features: torch.Tensor, aux: torch.Tensor = None) -> torch.Tensor:
        """Combine features with auxiliary inputs if present."""
        if self.aux_combine is not None and aux is not None:
            combined = torch.cat([features, aux], dim=-1)
            return self.aux_combine(combined)
        return features

    def forward(self, features: torch.Tensor, aux: torch.Tensor = None) -> torch.Tensor:
        """
        Return DENORMALIZED value predictions.

        The linear layer outputs normalized values, which we denormalize
        using the running statistics for use in GAE/advantage computation.

        Args:
            features: Policy features (batch, hidden_size)
            aux: Optional auxiliary inputs (batch, num_aux_inputs)
        """
        features = self._combine_features(features, aux)
        normalized = self.fc(features)
        return self.mean + self.std * normalized

    def get_normalized_value(self, features: torch.Tensor, aux: torch.Tensor = None) -> torch.Tensor:
        """
        Return raw (normalized) value output for training.

        Used when computing value loss against normalized targets.
        """
        features = self._combine_features(features, aux)
        return self.fc(features)

    def normalize_targets(self, returns: torch.Tensor) -> torch.Tensor:
        """Normalize returns for value function training."""
        return (returns - self.mean) / (self.std + 1e-8)

    def get_training_value(self, features: torch.Tensor, aux: torch.Tensor = None) -> torch.Tensor:
        """Return normalized value for training loss computation."""
        features = self._combine_features(features, aux)
        return self.fc(features)

    def prepare_targets(
        self, returns: torch.Tensor, old_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize targets for PopArt value loss."""
        return self.normalize_targets(returns), self.normalize_targets(old_values)

    def update_stats(self, returns: torch.Tensor, optimizer: torch.optim.Optimizer = None):
        """
        Update running statistics and adjust weights to preserve outputs.

        Uses count-based adaptive beta: starts at 1.0 (full batch init) and
        decays toward beta_min as updates accumulate. This naturally handles
        cold start without needing special init logic.

        IMPORTANT: All operations stay on GPU - no .item() or CPU transfers.
        Uses tensor ops (not Python ints) for CUDA graph compatibility.

        Args:
            returns: Batch of returns to update statistics from
            optimizer: If provided, rescales Adam momentum buffers to match
                       the weight rescaling. This prevents instability from
                       misaligned momentum after PopArt weight adjustment.
        """
        with torch.no_grad():
            # Increment count (tensor op, not Python int)
            self._update_count.add_(1)

            # Adaptive beta: 1/count decays from 1.0 toward 0, clamped at beta_min
            # First update: beta=1.0 (full init), second: 0.5, third: 0.33, etc.
            # Use tensor ops to stay on GPU and be graph-compatible
            count_float = self._update_count.float()
            beta = torch.clamp(1.0 / count_float, min=self.beta_min)

            # Compute batch statistics (stays on GPU)
            batch_mean = returns.mean()
            batch_std = returns.std().clamp(min=1e-4)

            # Store old values for weight adjustment (clone stays on GPU)
            old_std = self.std.clone()
            old_mean = self.mean.clone()

            # Update running statistics with adaptive EMA
            # mean = (1 - beta) * mean + beta * batch_mean
            # Use explicit multiplication (not alpha=) since beta is a tensor
            self.mean.mul_(1 - beta).add_(batch_mean * beta)

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

            # Rescale optimizer momentum to match weight rescaling
            # Without this, Adam's momentum buffers are misaligned with the new
            # weight scale, causing unstable updates (the "large bounce" problem)
            if optimizer is not None:
                self._rescale_optimizer_state(optimizer, scale)

    def _rescale_optimizer_state(self, optimizer: torch.optim.Optimizer, scale: torch.Tensor):
        """
        Rescale Adam optimizer momentum buffers after weight rescaling.

        When PopArt rescales weights by `scale`, the optimizer's momentum
        buffers (exp_avg, exp_avg_sq) must also be rescaled to stay aligned.
        Without this, the next gradient step uses stale momentum calibrated
        to the old weight scale, causing large unstable updates.

        For Adam:
        - exp_avg (first moment): scale by `scale` (linear with weights)
        - exp_avg_sq (second moment): scale by `scale^2` (squared)
        """
        scale_val = scale.item()  # Need scalar for in-place ops
        scale_sq = scale_val * scale_val

        for param in [self.fc.weight, self.fc.bias]:
            if param not in optimizer.state:
                continue
            state = optimizer.state[param]
            if 'exp_avg' in state:
                state['exp_avg'].mul_(scale_val)
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'].mul_(scale_sq)


class Profiler:
    """
    GPU-friendly profiler using CUDA events for zero-sync timing.

    Unlike naive profilers that call torch.cuda.synchronize() on every tick,
    this records CUDA events (non-blocking) and only syncs when summary() is
    called. This eliminates sync overhead from the hot training loop.

    Usage:
        profiler = Profiler(device)
        profiler.start()
        ... some work ...
        profiler.tick("step_1")
        ... more work ...
        profiler.tick("step_2")
        print(profiler.summary())  # Only sync happens here

    For CPU-only runs, falls back to time.perf_counter().
    """

    def __init__(self, device: torch.device = None, enabled: bool = True):
        self.enabled = enabled
        if device is None:
            device = get_device()
        self.use_cuda = (device.type == 'cuda')

        # Accumulated times (filled on summary() from events)
        self.times: dict[str, float] = {}
        self.counts: dict[str, int] = {}

        # CUDA event pairs: list of (name, start_event, end_event)
        # We defer time computation until summary() to avoid sync
        self._pending_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._last_event: torch.cuda.Event | None = None

        # CPU fallback
        self._last_cpu_t: float = 0.0

    def start(self):
        """Mark the start of a timed region (non-blocking)."""
        if not self.enabled:
            return
        if self.use_cuda:
            self._last_event = torch.cuda.Event(enable_timing=True)
            self._last_event.record()
        else:
            self._last_cpu_t = time.perf_counter()

    def tick(self, name: str):
        """Record elapsed time since last tick/start under the given name (non-blocking)."""
        if not self.enabled:
            return

        if self.use_cuda:
            # Record end event for this section
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()

            # Store the event pair for later processing
            if self._last_event is not None:
                self._pending_events.append((name, self._last_event, end_event))

            # This end becomes next section's start
            self._last_event = end_event
        else:
            # CPU path: immediate timing
            now = time.perf_counter()
            dt = now - self._last_cpu_t
            self.times[name] = self.times.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1
            self._last_cpu_t = now

    def reset(self):
        """Clear all recorded times and pending events."""
        self.times = {}
        self.counts = {}
        self._pending_events = []
        self._last_event = None

    def summary(self) -> str:
        """
        Compute and return a formatted summary of profiled times.

        For CUDA: syncs once to resolve all pending event pairs, then computes
        elapsed times. This is the ONLY sync point in the profiler.
        """
        if not self.enabled:
            return "Profiling disabled"

        # Process pending CUDA events (one sync for all)
        if self.use_cuda and self._pending_events:
            # Single sync to ensure all events are recorded
            torch.cuda.synchronize()

            # Compute elapsed times from event pairs
            for name, start_evt, end_evt in self._pending_events:
                # elapsed_time_ms returns milliseconds
                dt = start_evt.elapsed_time(end_evt) / 1000.0  # Convert to seconds
                self.times[name] = self.times.get(name, 0.0) + dt
                self.counts[name] = self.counts.get(name, 0) + 1

            # Clear processed events
            self._pending_events = []

        total = sum(self.times.values())
        if total == 0:
            return "No data"

        parts = []
        # Sort by duration (descending)
        for k, v in sorted(self.times.items(), key=lambda x: x[1], reverse=True):
            pct = v / total * 100
            parts.append(f"{k}: {v:.2f}s ({pct:.0f}%)")
        return " | ".join(parts)
