# Training Pipeline

This document covers how to train learned agent behaviors using PPO.

For internal architecture details (GPU sync points, compilation strategy, memory layout), see CLAUDE.md.

---

## Quick Start

```bash
# Train ground truth agent with PPO
python -m goodharts.training.train_ppo --mode ground_truth --updates 128

# Train all modes in parallel with dashboard
python -m goodharts.training.train_ppo --mode all --dashboard --updates 128

# Run visual demo with trained agents
python main.py brain-view --mode ground_truth
```

---

## Training Modes

| Mode | Observation | Reward | Use Case |
|------|-------------|--------|----------|
| `ground_truth` | One-hot cell types | Energy delta | Baseline: full information |
| `ground_truth_blinded` | Interestingness | Energy delta | Control: correct rewards, blinded observation |
| `proxy_mortal` | Interestingness | Interestingness | Partial grounding: proxy reward but can die |
| `proxy` | Interestingness | Interestingness | **Main Goodhart failure mode** (immortal during training) |
| `ground_truth_handhold` | One-hot cell types | Shaped rewards | Experimental (not in default batch) |

---

## PPO Hyperparameters

Configured via `config.toml` or `config.default.toml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.0003 | Adam optimizer LR |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE smoothing |
| `eps_clip` | 0.2 | PPO clipping epsilon (decays to 0.1) |
| `k_epochs` | 4 | Updates per batch |
| `steps_per_env` | 128 | Horizon length |
| `n_envs` | 192 | Parallel environments |
| `entropy_initial` | 0.1 | Starting entropy coefficient |
| `entropy_final` | 0.001 | Final entropy coefficient |
| `value_coef` | 0.5 | Value loss weight |

### Entropy Scheduling

Entropy coefficient decays from `entropy_initial` to `entropy_final` over training,
encouraging exploration early and exploitation late. An entropy floor prevents
premature policy collapse during the learning phase.

### Environment Randomization

Food and poison counts randomize each episode to prevent overfitting:

```toml
[training]
min_food = 64
max_food = 256
min_poison = 64
max_poison = 256
```

---

## CLI Options

| Option | Description |
|--------|-------------|
| `-m, --mode MODE` | Training mode: `ground_truth`, `proxy`, `all`, or comma-separated |
| `-u, --updates N` | Number of PPO updates |
| `-t, --timesteps N` | Total environment steps (alternative to updates) |
| `-e, --n-envs N` | Parallel environments (higher = faster, more VRAM) |
| `-d, --dashboard` | Live training visualization |
| `--seed N` | Random seed for reproducibility (default: 42) |
| `-b, --benchmark` | Measure throughput without saving models |
| `--no-amp` | Disable mixed precision |
| `--compile-mode` | torch.compile mode: `reduce-overhead`, `max-autotune` |

---

## Training Dashboard

Use `--dashboard` for live multi-mode visualization:

```bash
python -m goodharts.training.train_ppo --mode all --dashboard
```

Shows per-mode:
- Episode rewards and lengths
- Policy/value losses
- Entropy (exploration indicator)
- Action probability distribution

Stop training cleanly with the "Stop Training" button.

---

## Logging

Training produces structured logs in `logs/`:

- `{mode}_{timestamp}_episodes.csv` - Per-episode statistics
- `{mode}_{timestamp}_updates.csv` - Per-update losses and metrics
- `{mode}_{timestamp}_summary.json` - Final summary with hyperparams

TensorBoard logs go to `runs/` (view with `tensorboard --logdir runs/`).

---

## GPU Configuration

### Device Selection

```bash
# Environment variable
GOODHARTS_DEVICE=cuda:1 python -m goodharts.training.train_ppo

# Or in config.toml
[runtime]
device = "cuda"
```

### AMD GPUs (ROCm)

MIOpen (AMD's cuDNN equivalent) doesn't cache algorithm selection across processes, causing slow startup with `cudnn.benchmark=True`. The trainer auto-detects AMD and disables benchmark mode for faster startup with minimal throughput impact.

---

## Reproducibility

Training uses seeded random number generators:

```bash
python -m goodharts.training.train_ppo --mode ground_truth --seed 42
```

**Why training is inherently non-deterministic:**

PPO training uses `Categorical.sample()` for action selection, which internally calls `torch.multinomial`. This operation has **no deterministic CUDA implementation** in PyTorch. This is a fundamental GPU limitation, not a bug.

**What the seed provides:**
- Identical initial weights
- Identical environment layouts
- Identical data ordering
- Very similar (but not bit-identical) training trajectories

**What this means in practice:**
- Training with the same seed produces *very similar* but not bit-for-bit identical results
- This is standard in RL research; papers report statistics over multiple seeds
- Run multiple seeds and report mean/variance for robust results

**Evaluation IS fully deterministic:** When using `--deterministic` during evaluation, actions are selected via `argmax` (no sampling), making results bit-for-bit reproducible. This is where thesis validation occurs.

---

## Output

Training produces:
- **Model weights**: `models/ppo_{mode}.pth`
- **TensorBoard logs**: `runs/{mode}_{timestamp}/`
- **Structured logs**: `logs/{mode}_{timestamp}_*.csv/json`
