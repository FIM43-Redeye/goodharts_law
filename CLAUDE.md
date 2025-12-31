# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Goodhart's Law Simulation is an AI safety demonstration that uses reinforcement learning to show how proxy metric optimization fails. Agents navigate a 2D grid world distinguishing food from poison. Ground-truth agents see real cell types and thrive; proxy agents see only "interestingness" values and inevitably get poisoned.

**Stack:** Python 3.9+, PyTorch (GPU-native), NumPy, Plotly/Dash

## Common Commands

```bash
# Install
pip install -e .

# Visualization dashboards
python main.py --brain-view -m ground_truth       # Single-agent neural net visualization
python main.py --parallel-stats                   # Multi-mode statistics comparison
python scripts/brain_view.py -m proxy --speed 100 # Direct script with options
python scripts/parallel_stats.py --modes ground_truth,proxy --envs 256

# Train
python -m goodharts.training.train_ppo --mode ground_truth --timesteps 100000
python -m goodharts.training.train_ppo --mode all --dashboard

# Evaluate
python scripts/evaluate.py --mode all --timesteps 50000    # Single evaluation run
python scripts/evaluate.py --mode all --runs 5 --base-seed 42  # Multi-run with aggregation
python scripts/evaluate.py --full-report --runs 5          # Full report with figures

# Visualize results
python -m goodharts.analysis.visualize --input generated/eval_results.json --all
python -m goodharts.analysis.report --input generated/eval_results.json

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
  - `ground_truth`: One-hot cell types, energy-based reward
  - `ground_truth_handhold`: Ground truth observations with scaled rewards (easier learning)
  - `ground_truth_blinded`: Proxy observations but real energy reward (control condition)
  - `proxy`: Interestingness values only, interestingness gain as reward

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

- `goodharts/simulation.py`: Main simulation orchestrator
- `goodharts/environments/torch_env.py`: GPU-native vectorized environment
- `goodharts/behaviors/registry.py`: Behavior auto-discovery
- `goodharts/modes.py`: ObservationSpec and RewardComputer
- `goodharts/training/ppo/trainer.py`: PPO training loop
- `goodharts/configs/default_config.py`: CellType registry and get_config()
- `goodharts/visualization/`: Plotly/Dash dashboards (brain_view.py, parallel_stats.py)
- `goodharts/evaluation/`: Evaluation infrastructure (evaluator.py, multi_run.py)
- `goodharts/analysis/`: Statistical analysis (stats_helpers.py, power.py, visualize.py, report.py)
- `scripts/evaluate.py`: CLI for evaluation with multi-run and report generation
