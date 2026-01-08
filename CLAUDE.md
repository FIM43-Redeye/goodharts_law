# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Goodhart's Law Simulation is an AI safety demonstration that uses reinforcement learning to show how proxy metric optimization fails. Agents navigate a 2D grid world distinguishing food from poison. Ground-truth agents see real cell types and thrive; proxy agents see only "interestingness" values and inevitably get poisoned.

**Stack:** Python 3.9+, PyTorch (GPU-native), NumPy, matplotlib, Plotly/Dash

## Common Commands

```bash
# Install
pip install -e .

# Train
python main.py train --mode ground_truth --updates 128
python main.py train --mode all --dashboard

# Evaluate
python main.py evaluate --mode all --runs 5 --full-report
python main.py evaluate --mode all --timesteps 50000       # Single run

# Visualization dashboards
python main.py brain-view -m ground_truth                  # Neural network visualization (matplotlib)
python main.py brain-view -m proxy --speed 100
python main.py parallel-stats                              # Multi-mode statistics (Dash)
python main.py parallel-stats --modes ground_truth,proxy --envs 256

# Regenerate report from existing results
python main.py report                                      # Auto-detect latest results
python main.py report --input generated/reports/*/results.json

# Visualize results directly
python -m goodharts.analysis.visualize --input generated/eval_results.json --all

# Power analysis
python -c "from goodharts.analysis import print_power_table; print_power_table()"

# Test
pytest tests/ -v                                  # All tests
pytest tests/test_simulation.py -v                # Single file
pytest tests/ -k "test_ppo" -v                    # Pattern match
```

## Architecture

```
User → main.py or train_ppo → Load TOML Config → TorchVecEnv (GPU) → Behavior Registry → Step Loop
```

**Core Components:**

- **TorchVecEnv** (`environments/torch_env.py`): GPU-native vectorized environment. All state in PyTorch tensors, no CPU-GPU transfer during training. Supports "independent" (N worlds) and "shared_grid" (one world for visualization) modes.

- **Behavior Registry** (`behaviors/registry.py`): Auto-discovery via introspection. Inherit from `BehaviorStrategy` and it's automatically available. Two types: hardcoded (OmniscientSeeker, ProxySeeker) and learned (CNN+PPO).

- **Mode System** (`modes.py`): `ObservationSpec` + `RewardComputer` define training modes:
  - `ground_truth`: One-hot cell types, energy-based reward (baseline)
  - `ground_truth_blinded`: Proxy observations but real energy reward (control condition)
  - `proxy_mortal`: Proxy observations, proxy reward, but can die (partial grounding)
  - `proxy`: Proxy observations, proxy reward, immortal during training (main Goodhart case)
  - `ground_truth_handhold`: Ground truth with shaped rewards (experimental, manual-only)

- **Neural Network** (`behaviors/brains/base_cnn.py`): 3-layer CNN + 2 FC layers. Dynamic input channels based on observation format. Outputs 8-directional action logits.

- **PPO Training** (`training/ppo/`): Modular PPO implementation:
  - `trainer.py`: Main training orchestrator (PPOTrainer)
  - `ppo_config.py`: Configuration dataclass with TOML loading
  - `algorithms.py`: GAE computation and PPO update logic
  - `models.py`: ValueHead, PopArtValueHead for value function
  - `metrics.py`: Async GPU-to-CPU metrics transfer with double-buffered pinned memory
  - `async_logger.py`: Background thread for logging without GPU sync
  - `monitoring.py`: GPU utilization sampling (sysfs/nvidia-smi)
  - `globals.py`: Thread-safe warmup/abort state for parallel training

## Configuration

Precedence: CLI args > env vars > `config.toml` (user, gitignored) > `config.default.toml`

Key env var: `GOODHARTS_DEVICE` (force device: `cuda:1`, `cpu`, `tpu`)

## Code Conventions

- Use `CellType.FOOD`, `CellType.POISON` not raw integers
- Device selection always via `get_device()` from `utils/device.py`
- Factory functions preferred: `create_vec_env()`, `create_learned_behavior()`, `ObservationSpec.for_mode()`
- Type hints on all public functions

## Adding New Components

**New cell type:** Add to `CellType` class in `configs/default_config.py`. Observation channels auto-expand.

**New behavior:** Inherit from `BehaviorStrategy` in `behaviors/base.py`. It auto-registers.

**New training mode:** Define `ObservationSpec` channel names and `RewardComputer` in `modes.py`.

## Key Files

- `main.py`: Unified CLI dispatcher (train, evaluate, brain-view, parallel-stats, report)
- `goodharts/cli/`: CLI modules (evaluate.py, brain_view.py, parallel_stats.py, report.py)
- `goodharts/simulation.py`: Main simulation orchestrator
- `goodharts/environments/torch_env.py`: GPU-native vectorized environment
- `goodharts/behaviors/registry.py`: Behavior auto-discovery
- `goodharts/modes.py`: ObservationSpec and RewardComputer
- `goodharts/training/ppo/trainer.py`: PPO training loop
- `goodharts/configs/default_config.py`: CellType registry and get_config()
- `goodharts/visualization/`: matplotlib brain view, Dash parallel stats dashboard
- `goodharts/evaluation/`: Evaluation infrastructure (evaluator.py, multi_run.py)
- `goodharts/analysis/`: Statistical analysis (stats_helpers.py, power.py, visualize.py, report.py)
- `scripts/`: Development tools only (profiling, benchmarking, tracing)
