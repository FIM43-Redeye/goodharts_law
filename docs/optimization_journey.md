# Optimization Journey: From 50 sps to 29,000 sps

This document chronicles the performance optimization journey of the Goodhart's Law training pipeline, from a naive Python loop to a fully GPU-native implementation achieving ~600x speedup.

## TL;DR Performance Timeline

| Stage | Throughput | Bottleneck |
|-------|-----------|------------|
| Naive Python loop | ~50 sps | Python interpreter |
| NumPy vectorization | ~500 sps | CPU-GPU data transfer |
| Basic PyTorch GPU | ~3,000 sps | Frequent small transfers |
| Batched GPU ops | ~8,000 sps | Logging sync overhead |
| Async logging | ~15,000 sps | cuDNN/MIOpen warmup |
| JIT warmup | ~20,000 sps | GradScaler sync |
| GPU-first metrics | ~27,000 sps | AMP disabled (regression) |
| Fixed regression | ~29,000 sps | Remaining burstiness TBD |

## Phase 1: The Naive Beginning

### Starting Point
The original implementation used a standard Python training loop:

```python
for step in range(total_steps):
    for env in environments:
        obs = env.get_observation()
        action = policy(obs)
        reward = env.step(action)
        # ... individual updates
```

**Performance**: ~50 steps/second
**Bottleneck**: Python interpreter overhead, no parallelism

## Phase 2: NumPy Vectorization

### Changes
- Replaced per-environment loops with vectorized NumPy operations
- Batch observations across all environments
- Single policy forward pass for all envs

```python
# Before: N forward passes
for env in envs:
    action = policy(env.obs)

# After: 1 forward pass
actions = policy(np.stack([env.obs for env in envs]))
```

**Performance**: ~500 steps/second (10x improvement)
**Bottleneck**: CPU-GPU data transfer for each batch

## Phase 3: Basic PyTorch GPU Migration

### Changes
- Moved environment state to GPU tensors
- Created `TorchVecEnv` with all state as `torch.Tensor`
- Eliminated NumPy intermediaries

**Key insight**: The environment itself should live on GPU, not just the model.

```python
class TorchVecEnv:
    def __init__(self, n_envs, device):
        self.grid = torch.zeros((n_envs, H, W), device=device)
        self.agent_pos = torch.zeros((n_envs, 2), device=device)
        # All state on GPU from the start
```

**Performance**: ~3,000 steps/second (6x improvement)
**Bottleneck**: Still transferring rewards/dones to CPU for logging

## Phase 4: Batched GPU Operations

### Changes
- Batch all environment steps into single GPU kernel launches
- Use `torch.where()` instead of Python conditionals
- Vectorized collision detection and reward computation

```python
# Before: Python loop with conditionals
for i in range(n_envs):
    if grid[i, pos[i]] == FOOD:
        reward[i] = 1.0

# After: Fully vectorized
cell_values = grid[batch_idx, pos[:, 0], pos[:, 1]]
reward = torch.where(cell_values == FOOD, 1.0, 0.0)
```

**Performance**: ~8,000 steps/second (2.7x improvement)
**Bottleneck**: Logging calls causing GPU sync every step

## Phase 5: Async Logging

### The Problem
Every log statement with `.item()` or `.cpu()` forces GPU synchronization:

```python
# This innocent line causes a full GPU sync!
logger.log("reward", reward.mean().item())
```

### Solution: AsyncLogger
Created a background thread that handles all I/O:

```python
class AsyncLogger:
    def __init__(self):
        self.queue = Queue()
        self.thread = Thread(target=self._writer_loop)

    def log_update(self, payload: LogPayload):
        # Main thread just queues data (no sync)
        self.queue.put(payload)

    def _writer_loop(self):
        # Background thread does .item() and file I/O
        while True:
            payload = self.queue.get()
            # Sync happens here, off the critical path
```

**Performance**: ~15,000 steps/second (1.9x improvement)
**Bottleneck**: First few updates extremely slow (JIT compilation)

## Phase 6: JIT Warmup Strategy

### The Problem
`torch.compile()` and cuDNN/MIOpen benchmark mode defer compilation until first use, causing:
- First forward pass: 2-5 seconds (graph compilation)
- First backward pass: 3-10 seconds (autograd graph)
- cuDNN algorithm selection: 60-300 seconds on AMD

### Solution: Explicit Warmup
Run one full training update before timing begins:

```python
def _run_warmup_update(self):
    # Save model state
    policy_state = self.policy.state_dict()

    # Run real update (triggers all lazy init)
    self._collect_experience()
    self._ppo_update()

    # Restore state (discard warmup gradients)
    self.policy.load_state_dict(policy_state)
```

### AMD-Specific: MIOpen Benchmark Disabled
MIOpen (AMD's cuDNN) doesn't cache algorithm selection across processes.
Auto-detect AMD and disable benchmark mode:

```python
if 'AMD' in torch.cuda.get_device_name():
    torch.backends.cudnn.benchmark = False  # 4s startup vs 300s
```

**Performance**: ~20,000 steps/second (1.3x improvement)
**Bottleneck**: GradScaler checking for inf/nan via .item()

## Phase 7: GPU-First Metrics

### The Problem
Even with async logging, we were computing metrics eagerly:

```python
# Each .item() is a GPU sync!
n_episodes = dones.sum().item()
mean_reward = rewards[dones].mean().item()
```

### Solution: Aggregate on GPU, Transfer Once
Compute all aggregates as a single tensor, transfer once per update:

```python
# All ops stay on GPU
episode_agg = torch.stack([
    dones.sum(),
    (rewards * dones).sum(),
    torch.where(dones, rewards, INF).min(),
    torch.where(dones, rewards, -INF).max(),
])

# Single sync point after PPO update completes
cpu_agg = episode_agg.cpu().numpy()
```

**Performance**: ~27,000 steps/second (1.35x improvement)
**Bottleneck**: Mysterious regression appeared...

## Phase 8: The Great Regression Hunt

### Symptom
PPO update time jumped from 0.6s to 2.1s (3.5x slower).
Throughput dropped from 28k to 10k sps.

### Root Causes Found

#### 1. AMP Disabled (2x slowdown)
Someone disabled mixed precision "to avoid GradScaler sync overhead."
But fp32 uses **2x memory bandwidth** vs fp16, which dominates on memory-bound ops.

```toml
# BAD: fp32 doubles memory traffic
use_amp = false

# GOOD: fp16 halves memory traffic
use_amp = true
```

#### 2. Branching in Hot Path (1.5x slowdown)
PopArt value head added a conditional in the inner loop:

```python
# BAD: torch.compile sees both branches
if use_popart:
    values = value_head.get_normalized_value(features)
    targets = value_head.normalize_targets(returns)
else:
    values = value_head(features)
    targets = returns
```

This caused torch.compile to either create separate graphs or fall back to eager mode.

**Solution**: Uniform interface - both ValueHead types implement the same methods:

```python
class ValueHead:
    def get_training_value(self, features):
        return self.fc(features)  # Pass-through

    def prepare_targets(self, returns, old_values):
        return returns, old_values  # No-op

class PopArtValueHead:
    def get_training_value(self, features):
        return self.fc(features)  # Normalized

    def prepare_targets(self, returns, old_values):
        return self.normalize(returns), self.normalize(old_values)
```

Now the hot path has no branches:

```python
# GOOD: Single code path, torch.compile optimizes fully
values = value_head.get_training_value(features)
target_returns, target_old = value_head.prepare_targets(returns, old_values)
```

#### 3. Non-Fused Optimizer (Minor, ~300 sps)
Standard Adam calls `.item()` twice per parameter to check for inf/nan.
With 12 parameters and 4 epochs, that's 384 CPU round-trips per update.

```python
# BAD: 384 syncs per update
optimizer = Adam(params, lr=lr)

# GOOD: Runs entirely on GPU
optimizer = Adam(params, lr=lr, fused=True)
```

**Performance**: ~29,000 steps/second (2.9x improvement from regression)

## Current Architecture

```
Training Loop (single GPU, no CPU sync except logging)
    |
    +-- TorchVecEnv (all state as GPU tensors)
    |       |
    |       +-- Grid: (n_envs, H, W) int8
    |       +-- Agent positions: (n_envs, 2) int32
    |       +-- Observations: (n_envs, C, V, V) float16
    |
    +-- Policy (torch.compile'd CNN)
    |       |
    |       +-- Forward: observations -> logits, features
    |       +-- Shared features for actor + critic
    |
    +-- ValueHead (torch.compile'd, uniform interface)
    |       |
    |       +-- get_training_value() -> normalized or raw
    |       +-- prepare_targets() -> normalized or pass-through
    |
    +-- PPO Update (single kernel fusion via torch.compile)
    |       |
    |       +-- No branching in hot path
    |       +-- Fused Adam optimizer (no .item() calls)
    |       +-- AMP enabled (fp16 forward, fp32 master weights)
    |
    +-- AsyncLogger (background thread)
            |
            +-- Main thread: queue CPU-ready data
            +-- Background: .item() sync + file I/O
```

## Lessons Learned

### 1. Profile Before Optimizing
Every optimization should be driven by profiling data, not intuition.
Use `torch.profiler` and nvtop/rocm-smi to find actual bottlenecks.

### 2. GPU Sync is the Enemy
Any `.item()`, `.cpu()`, `.numpy()`, or `print(tensor)` forces the CPU to wait
for all pending GPU work. Batch these operations and move them off the critical path.

### 3. Memory Bandwidth Often Dominates
Modern GPUs have more compute than memory bandwidth. Operations like convolutions
and large matrix multiplies are often memory-bound. AMP (fp16) effectively doubles
your memory bandwidth.

### 4. Branching Breaks Compilation
`torch.compile` works best with straight-line code. Conditionals can cause:
- Multiple compiled graphs (memory overhead)
- Graph breaks (fallback to eager mode)
- Missed fusion opportunities

Use polymorphism (uniform interfaces) instead of conditionals.

### 5. AMD/ROCm Has Different Characteristics
- MIOpen doesn't cache algorithm selection across processes
- Fused optimizers may have different performance profiles
- Always test on your target hardware

### 6. The Environment Should Be GPU-Native
Don't just put the model on GPU - put the entire environment there.
CPU-GPU transfers for observations/rewards are a hidden bottleneck.

## Remaining Work

GPU burstiness is still visible in nvtop. Potential causes:
- Python GIL contention with async logger thread
- Kernel launch overhead for small operations
- Memory allocation patterns
- PCIe bandwidth limitations

Investigation ongoing.

## References

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [torch.compile Deep Dive](https://pytorch.org/docs/stable/torch.compiler.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)
