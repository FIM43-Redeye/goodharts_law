# Training Pipeline

This document covers the training infrastructure for learned agent behaviors.

---

## Training Pipeline Deep Dive

This section documents the **complete internal flow** of a training run, from startup to shutdown. Understanding this is essential for performance optimization and debugging GPU utilization issues.

### Overview: What Happens in a Training Run

```
TRAINING RUN LIFECYCLE
======================

1. STARTUP (one-time)
   - Parse CLI args, load config
   - Device selection (CUDA/CPU/TPU)
   - Create TorchVecEnv (GPU-native environment)
   - Create Policy CNN + ValueHead
   - torch.compile() with max-autotune-no-cudagraphs
   - Create fused Adam optimizer (GPU-native)
   - Start AsyncLogger (background I/O thread)
   - Start ProcessLogger (separate process for file I/O)

2. WARMUP (one-time, ~15s)
   - Run one full update with real data
   - Triggers torch.compile kernel autotuning
   - Triggers MIOpen/cuDNN algorithm selection
   - Discard results, restore initial weights
   - torch.cuda.synchronize() to ensure completion

3. TRAINING LOOP (repeated N times)

   UPDATE CYCLE (steps_per_env * n_envs timesteps per update)
   ----------------------------------------------------------

   A. COLLECTION PHASE (steps_per_env iterations)
      For each step:
        - compiled_inference(states) -> logits, features, values
        - compiled_sample(logits) -> actions, log_probs
        - states.clone() (snapshot before step)
        - vec_env.step(actions) -> next_states, rewards, dones
        - reward_computer.compute() (shaping)
        - Store in pre-allocated buffers

   B. GAE COMPUTATION
      - Bootstrap value: compiled_inference(final_states)
      - compute_gae() [Python loop over timesteps - sequential]
      - PopArt statistics update

   C. BUFFER PREPARATION
      - torch.cat() all experience buffers
      - Flatten returns and advantages
      - Normalize advantages

   D. PPO UPDATE (k_epochs * n_minibatches iterations)
      For each epoch:
        For each minibatch:
          - Forward pass (policy + value)
          - Compute PPO clipped loss
          - Compute value loss (with clipping)
          - Backward pass
          - Gradient clipping
          - Optimizer step (fused Adam on GPU)

   E. LOGGING (async, no GPU stall)
      - Queue metrics to AsyncLogger thread
      - AsyncLogger syncs GPU and extracts .item() values
      - ProcessLogger writes to disk in separate process

4. SHUTDOWN
   - Save final model checkpoint
   - Write summary JSON
   - Stop AsyncLogger thread
   - Stop ProcessLogger process
```

### Detailed Phase Breakdown

#### Phase A: Collection (GPU-bound, ~28% of update time)

The collection phase runs `steps_per_env` iterations (default: 64), each processing `n_envs` environments in parallel (default: 192).

```python
# Simplified collection step (actual code in trainer.py:773-793)
with torch.no_grad():
    # COMPILED: Fuses float() + forward + squeeze into one kernel graph
    logits, features, values = compiled_inference(states)

    # COMPILED: Fuses Categorical + sample + log_prob
    actions, log_probs = compiled_sample(logits)

# Environment step (all GPU tensors, no CPU transfer)
current_states = states.clone()
next_states, rewards, dones = vec_env.step(actions)
shaped_rewards = reward_computer.compute(rewards, current_states, next_states, dones)

# Store in pre-allocated buffer slots (no append, no allocation)
states_buffer[step_idx] = current_states
# ... etc
```

**GPU Operations:**
- `compiled_inference`: 3 conv layers + 2 FC layers + value head
- `compiled_sample`: Categorical distribution + multinomial sampling
- `states.clone()`: Memory copy
- `vec_env.step()`: Grid updates, collision detection, reward computation
- `reward_computer.compute()`: Distance calculations for shaping

**Potential Stalls:** None if properly compiled. The compiled functions fuse many small kernels.

#### Phase B: GAE Computation (Mixed, ~0.2% of time but sequential)

```python
# Bootstrap value for final state
with torch.no_grad():
    logits, features, values = compiled_inference(states)
    next_value = values

# GAE has a sequential dependency - each step depends on the next
# This is a Python loop but operates on GPU tensors
advantages, returns = compute_gae(
    rewards_buffer, values_buffer, dones_buffer,
    next_value, gamma, gae_lambda, device
)
```

**The GAE Loop (algorithms.py:69-71):**
```python
for t in range(T - 1, -1, -1):  # T = steps_per_env (64)
    lastgae = deltas[t] + gae_coef * masks[t] * lastgae
    advantages[t] = lastgae
```

**Potential Stalls:** 64 Python loop iterations with small tensor ops. Each iteration launches tiny kernels that don't saturate GPU. This is a known bottleneck but represents <1% of time.

#### Phase C: Buffer Preparation (~0.1% of time)

```python
# Concatenate lists of tensors into single tensors
all_states = torch.cat(states_buffer, dim=0)      # (64, 192, ...) -> (12288, ...)
all_actions = torch.cat(actions_buffer, dim=0)
all_log_probs = torch.cat(log_probs_buffer, dim=0)
all_old_values = torch.cat(values_buffer, dim=0)

# Flatten and normalize
all_returns = returns.flatten()
all_advantages = advantages.flatten()
all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
```

**Potential Stalls:** `torch.cat` allocates new memory. Could cause memory pressure spikes.

#### Phase D: PPO Update (GPU-bound, ~72% of update time)

This is the heaviest phase - 4 epochs * 4 minibatches = 16 forward+backward passes.

```python
for epoch in range(k_epochs):  # 4 epochs
    indices = torch.argsort(torch.rand(batch_size, device=device))  # Shuffle

    for mb_idx in range(n_minibatches):  # 4 minibatches
        mb_inds = indices[start:end]

        # Forward pass
        with autocast(enabled=use_amp):
            features = policy.get_features(mb_states)
            logits = policy.logits_from_features(features)
            values = value_head.get_training_value(features)

            # Compute losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.max(v_loss1, v_loss2)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(...)
        scaler.step(optimizer)
        scaler.update()
```

**GPU Operations:**
- 16x forward passes through CNN + FC layers
- 16x backward passes (autograd)
- 16x optimizer steps (fused Adam - single kernel per step)

**Potential Stalls:**
- `GradScaler.update()` can sync if scale changes
- Each epoch shuffles indices (small)

#### Phase E: Async Logging (Background, no stall)

```python
# Main training thread (NO sync, NO .item())
payload = LogPayload(
    policy_loss=policy_loss,      # GPU tensor
    value_loss=value_loss,        # GPU tensor
    entropy=entropy,              # GPU tensor
    # ... etc
)
async_logger.log(payload)  # Non-blocking queue.put()

# AsyncLogger thread (runs in background)
def _worker():
    while True:
        payload = queue.get()
        torch.cuda.synchronize()  # Sync happens HERE, in background

        # Now extract .item() values (safe, GPU already synced)
        policy_loss_val = payload.policy_loss.item()
        # ... write to files, update dashboard
```

**Critical Design:** The GPU sync and `.item()` calls happen in a background thread, not the training loop. This prevents GPU stalls from blocking training.

### GPU Sync Points (Places That Block Training)

| Location | Operation | Why It Syncs | Avoidable? |
|----------|-----------|--------------|------------|
| Warmup end | `torch.cuda.synchronize()` | Ensure warmup complete | No (one-time) |
| Profiler.tick() | `torch.cuda.synchronize()` | Accurate timing | Yes (disable with --no-profile) |
| End of training | `torch.cuda.synchronize()` | Final save | No (one-time) |

**Eliminated Sync Points:**
- `.item()` calls - moved to AsyncLogger background thread
- `nonzero()` in respawn - replaced with masked operations
- Episode logging - uses masked GPU aggregates

### Memory Layout

```
GPU Memory:
  Policy CNN (~4.1M params, fp16)
  Value Head (~0.5M params, fp16)
  Experience Buffers (pre-allocated):
    states_buffer: (steps_per_env, n_envs, channels, H, W)
    actions_buffer: (steps_per_env, n_envs)
    log_probs_buffer: (steps_per_env, n_envs)
    rewards_buffer: (steps_per_env, n_envs)
    dones_buffer: (steps_per_env, n_envs)
    values_buffer: (steps_per_env, n_envs)
  TorchVecEnv state:
    grids: (n_grids, H, W) int8
    agent positions: (n_envs, 2) int32
    agent energy: (n_envs,) float32
  Optimizer state (Adam momentum, fp32)
```

### Thread/Process Architecture

```
MAIN PROCESS
============
  Main Thread              AsyncLogger Thread
  (Training Loop)    --->  (Background I/O)

  - Collection             - torch.cuda.sync()
  - GAE                    - .item() extraction
  - PPO Update             - Dashboard updates
  - Queue metrics          - TensorBoard writes
                                   |
                                   v
                           ProcessLogger (IPC Queue)


LOGGER PROCESS (separate, GIL-free)
===================================
  ProcessLogger Worker
  - Receives log entries via multiprocessing.Queue
  - Writes CSV files
  - Writes JSON summaries
  - Completely isolated from training
```

### Compilation Strategy

```python
# Models are NOT compiled individually anymore.
# Instead, we compile FUNCTIONS that fuse multiple operations:

@torch.compile(mode='max-autotune-no-cudagraphs')
def compiled_inference(states):
    """Fuses: .float() + policy.forward_with_features() + value_head() + squeeze()"""
    states_t = states.float()
    with autocast(device_type=device_type, enabled=use_amp):
        logits, features = policy.forward_with_features(states_t)
        values = value_head(features).squeeze(-1)
    return logits, features, values

@torch.compile(mode='max-autotune-no-cudagraphs')
def compiled_sample(logits):
    """Fuses: Categorical() + sample() + log_prob()"""
    dist = Categorical(logits=logits, validate_args=False)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    return actions, log_probs
```

**Why `max-autotune-no-cudagraphs`:**
- `max-autotune`: Tries Triton kernels vs vendor libraries, picks fastest
- `no-cudagraphs`: Avoids CUDA graph tensor ownership issues with our buffer reuse

**Effect:** Reduces inter-update gaps from ~1ms to ~0.006ms by fusing 100+ small kernel launches into a few large fused kernels.

### Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Throughput | ~30,000 steps/sec | AMD RX 7700S, 192 envs |
| Update time | ~800-1000ms | 64 steps * 192 envs = 12,288 timesteps |
| Collection | ~28% | Dominated by env.step() and inference |
| PPO Update | ~72% | 16 forward+backward passes |
| GAE + Flatten | <1% | Small but sequential |
| Warmup | ~15s | One-time compilation cost |

### Known Issues and Mitigations

| Issue | Cause | Mitigation |
|-------|-------|------------|
| First update slow | torch.compile autotuning | Warmup update before training |
| MIOpen startup lag | Algorithm selection | Disabled benchmark mode (AMD) |
| Burstiness in nvtop | Under investigation | See below |
| GAE sequential loop | Recurrence relation | Could use parallel scan (complex) |

### The Burstiness Mystery

Despite eliminating all known sync points, nvtop still shows GPU utilization dips between updates. Measurements show:

- Inter-update gap: ~0.006ms (with compiled functions)
- GPU idle time: ~0% (wall time matches GPU time)
- Yet nvtop shows dips to ~75%

**Hypotheses:**
1. Memory bandwidth vs compute utilization (different metrics)
2. Phase transitions cause compute unit underutilization
3. nvtop sampling aliasing with update boundaries
4. ROCm-specific reporting artifacts

---

## Quick Start

```bash
# Train ground truth agent with PPO
python -m goodharts.training.train_ppo --mode ground_truth --timesteps 100000

# Train all modes in parallel with dashboard
python -m goodharts.training.train_ppo --mode all --dashboard --timesteps 100000

# Verify model fitness
python -m goodharts.training.verification --steps 500 --verbose

# Run visual demo with trained agents
python main.py --learned
```

---

## Training Modes

| Mode | Observation | Reward | Use Case |
|------|-------------|--------|----------|
| `ground_truth` | One-hot cell types | Energy delta | Baseline: full information |
| `proxy` | Interestingness | Interestingness | **Main Goodhart failure mode** |
| `proxy_jammed` | Interestingness | Energy delta | Information asymmetry (bonus) |

---

## Observation Encoding

Agents see a multi-channel view of their surroundings (default: 6 channels, 11×11):

### Ground Truth Mode
One-hot encoding—agent can distinguish cell types:
```
Channel 0: is_empty
Channel 1: is_wall  
Channel 2: is_food
Channel 3: is_poison
Channel 4: is_prey
Channel 5: is_predator
```

### Proxy Mode
Interestingness signal—food and poison look nearly identical:
```
Channel 0: is_empty
Channel 1: is_wall
Channels 2-5: interestingness (0.0-1.0)
```

**The trap:** Food has interestingness 1.0, poison has 0.9. They're almost indistinguishable!

---

## Neural Network Architecture (BaseCNN)

```
Input: (C, 11, 11) where C = num_channels (6 for current CellType count)
  ↓
Conv2D(C→32, 3×3, padding=1) + ReLU
  ↓
Conv2D(32→64, 3×3, padding=1) + ReLU
  ↓
Conv2D(64→64, 3×3, padding=1) + ReLU
  ↓
Flatten: 64 × 11 × 11 = 7,744 features
  ↓
Linear(7744→512) + ReLU  ← Value head branches here for PPO
  ↓
Linear(512→8) logits → 8 directional actions
```

Model parameters: ~4.1M (mostly in fc1)

---

## Action Space

Centralized in `behaviors/action_space.py`:

```
Index  (dx, dy)   Direction
0      (-1, -1)   ↖ Up-Left
1      (-1,  0)   ← Left
2      (-1,  1)   ↙ Down-Left
3      ( 0, -1)   ↑ Up
4      ( 0,  1)   ↓ Down
5      ( 1, -1)   ↗ Up-Right
6      ( 1,  0)   → Right
7      ( 1,  1)   ↘ Down-Right
```

No "stay in place" action—agents must always move.

---

## PPO Training Details

### Algorithm
- Proximal Policy Optimization with clipped surrogate objective
- Generalized Advantage Estimation (GAE-Lambda)
- Separate value head (attached to BaseCNN features)
- Value function clipping

### Modular Architecture

The PPO implementation is split into focused modules in `ppo/`:

```
ppo/
├── trainer.py      # PPOTrainer: orchestrates the training loop
├── algorithms.py   # Pure functions: compute_gae(), ppo_update()
└── models.py       # ValueHead, Profiler utilities
```

- **trainer.py**: `PPOTrainer` class handles environment creation, experience collection, logging, and checkpointing. Subclassable for multi-agent extensions.

- **algorithms.py**: Stateless functions for GAE computation and the PPO clipped update. Easy to test and reuse.

- **models.py**: `ValueHead` (MLP for value function) and `Profiler` (timing breakdown).

### Hyperparameters (from config.toml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.0003 | Adam optimizer LR |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE smoothing |
| `eps_clip` | 0.2 | PPO clipping epsilon |
| `k_epochs` | 4 | Updates per batch |
| `steps_per_env` | 128 | Horizon length |
| `entropy_coef` | 0.02 | Exploration bonus |
| `value_coef` | 0.5 | Value loss weight |

### Curriculum Learning

Food and poison counts are randomized each update to prevent overfitting:

```toml
[training]
min_food = 50
max_food = 200
min_poison = 20
max_poison = 100
```

### Reward Shaping

Potential-based shaping guides agents toward food:
- Potential = 0.5 / distance_to_nearest_food
- Shaping = γ × Φ(s') - Φ(s)

This doesn't change optimal policy but accelerates learning.

---

## Vectorized Training

Training uses `VecEnv` for high throughput:

```python
# 64 parallel environments (default)
python -m goodharts.training.train_ppo --n-envs 64

# Higher parallelism
python -m goodharts.training.train_ppo --n-envs 128
```

### Performance
- **GPU (AMD RX 7700S)**: ~4,000-6,000 steps/sec
- **CPU**: ~1,500-2,500 steps/sec
- 100k timesteps ≈ 15-25 seconds

### Profiling Output

Training shows profiling breakdown:
```
[Profile] PPO Update: 1.40s (80%) | Inference: 0.25s (14%) | Env Step: 0.07s (4%) ...
```

- **PPO Update**: Backprop through k_epochs (expected to dominate)
- **Inference**: Forward passes for action selection
- **Env Step**: Vectorized environment stepping
- **GAE Calc**: Advantage computation (very fast)

---

## Training Dashboard

Use `--dashboard` for a live multi-mode visualization:

```bash
python -m goodharts.training.train_ppo --mode all --dashboard
```

Shows per-mode:
- Episode rewards and lengths
- Policy/value losses
- Entropy (exploration indicator)
- Action probability distribution
- Current observation

Stop training cleanly by pressing "Stop Training" button.

---

## Logging

Training produces structured logs in `logs/`:

- `{mode}_{timestamp}_episodes.csv` — Per-episode statistics
- `{mode}_{timestamp}_updates.csv` — Per-update losses and metrics
- `{mode}_{timestamp}_summary.json` — Final summary with hyperparams

---

## Model Verification

```bash
# Run survival tests
python -m goodharts.training.verification

# With more steps and verbose output
python -m goodharts.training.verification --steps 500 --verbose
```

Tests:
1. **Directional accuracy**: Does model prefer food over poison?
2. **Survival test**: Can model survive in environment?

---

## GPU Configuration

### Check GPU Status
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name()}")
```

### AMD GPUs (ROCm)
```bash
# Install ROCm-compatible PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# May need GFX version override in config.toml:
[runtime]
hsa_override_gfx_version = "11.0.0"
```

### Device Selection
```bash
# Environment variable
GOODHARTS_DEVICE=cuda:1 python -m goodharts.training.train_ppo

# Or in config.toml
[runtime]
device = "cuda"
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `train_ppo.py` | CLI entry point for PPO training |
| `ppo/trainer.py` | PPOTrainer class - main training loop |
| `ppo/algorithms.py` | GAE computation & PPO update functions |
| `ppo/models.py` | ValueHead, Profiler utilities |
| `train_dashboard.py` | Multi-mode live visualization |
| `train_log.py` | Structured CSV/JSON logging |
| `collect.py` | Expert demonstration collection |
| `dataset.py` | Dataset utilities |
| `verification/` | Model fitness tests |
| `visualize_saliency.py` | Gradient-based interpretability |

---

## Deprecated: Behavior Cloning

`train.py` implements behavior cloning (imitate expert trajectories). This was our original approach but failed due to:

1. **No negative examples**: Expert never encounters poison
2. **Distributional shift**: CNN sees states expert never visited
3. **Proxy optimization**: Copying ≠ surviving

See `docs/goodhart_self_proven.md` for the full story.
