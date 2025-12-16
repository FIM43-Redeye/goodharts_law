# Training Pipeline

This document covers the training infrastructure for learned agent behaviors.

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
| `reward_shaping.py` | Potential-based reward shaping |
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
