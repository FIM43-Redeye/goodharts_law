# Goodhart's Law Simulation

> "When a measure becomes a target, it ceases to be a good measure." — Charles Goodhart

An empirical demonstration of Goodhart's Law in reinforcement learning. Agents navigate a 2D grid world collecting food and avoiding poison. **Ground-truth agents** see real cell types and thrive. **Proxy agents** see only an "interestingness" signal—and eat 4× more poison.

This project provides a concrete, reproducible example of how optimizing for proxy metrics leads to alignment failures, even when the proxy seems reasonable.

---

## Results

<!-- TODO (Goodhart Documentation):
     Fill in this table with actual experimental results. Include:
     - Data from at least 3 random seeds per mode
     - Confidence intervals or standard deviations
     - Statistical significance tests between ground_truth and proxy

     After filling in, write a brief interpretation:
     - What does the efficiency gap tell us about Goodhart's Law?
     - Why does proxy perform as it does?
     - What's surprising or confirming about these results?
-->

Trained agents evaluated using continuous survival testing (agents run until death, then respawn):

| Mode | Observation | Reward | Efficiency | Survival | Deaths/1k | Food/1k | Poison/1k |
|------|-------------|--------|------------|----------|-----------|---------|-----------|
| **ground_truth** | Cell types | Energy | TBD | TBD | TBD | TBD | TBD |
| **ground_truth_handhold** | Cell types | Shaped | TBD | TBD | TBD | TBD | TBD |
| **proxy** | Interestingness | Interestingness | TBD | TBD | TBD | TBD | TBD |
| **ground_truth_blinded** | Interestingness | Energy | TBD | TBD | TBD | TBD | TBD |

**Key metrics:**
- **Efficiency** = food / (food + poison) — the core Goodhart failure metric
- **Survival** = average steps lived before death
- **Deaths/1k** = population death rate per 1000 steps

**Expected finding:** Proxy agents will show lower efficiency than ground-truth agents because they cannot distinguish food from poison—and worse, poison is MORE interesting (1.0) than food (0.5), so they actively prefer it. They optimize the proxy metric successfully but fail catastrophically at the true objective.

---

## Table of Contents

- [Results](#results)
- [Core Concept](#core-concept)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Architecture Deep Dive](#architecture-deep-dive)
- [AI Safety Connection](#ai-safety-connection)
- [Roadmap](#roadmap)
- [Disclosure](#disclosure)

---

## Core Concept

This project explores a fundamental AI safety concern: **what happens when agents optimize for a measurable proxy instead of the true objective?**

### The Goodhart Trap

| Agent Type | Can See | Optimizes For | Outcome |
|------------|---------|---------------|---------|
| **Ground Truth Agent** | Real cell types (food vs poison) | Eating food, avoiding poison | Thrives |
| **Proxy Agent** | Proxy signal only (interestingness) | Highest signal cells | Gets poisoned |

**Key insight:** Poison has high "interestingness" (0.9) compared to food (1.0), making it nearly as attractive to proxy-optimizing agents—but eating it is lethal.

### Training Modes

| Mode | Observation | Reward Signal | Purpose |
|------|-------------|---------------|---------|
| `ground_truth` | One-hot cell types | Energy delta | Baseline: full information |
| `ground_truth_handhold` | One-hot cell types | Shaped rewards | Easier learning curve |
| `proxy` | Interestingness values | Interestingness gain | **Main Goodhart failure mode** |
| `ground_truth_blinded` | Interestingness values | Energy delta | Control: blinded but true rewards |

---

## Project Structure

```
goodharts_law/
├── main.py                     # Visual demo entry point
├── config.toml                 # Your configuration (gitignored)
├── config.default.toml         # Default configuration template
├── pyproject.toml              # Package metadata & dependencies
│
├── goodharts/                  # Main package
│   ├── simulation.py           # Visual demo: agents in shared-grid VecEnv
│   ├── visualization.py        # Matplotlib-based live visualization
│   ├── config.py               # TOML config loader with caching
│   ├── modes.py                # Mode registry (ObservationSpec, ModeSpec, RewardComputer)
│   │
│   ├── behaviors/              # Agent decision-making
│   │   ├── base.py             # BehaviorStrategy ABC + ROLE_COLORS
│   │   ├── registry.py         # Auto-discovery behavior registry
│   │   ├── action_space.py     # Centralized action definitions (8-directional)
│   │   ├── learned.py          # LearnedBehavior + presets factory
│   │   ├── hardcoded/          # Hand-coded baseline behaviors
│   │   │   ├── omniscient.py   # OmniscientSeeker: sees true cell types
│   │   │   └── proxy_seeker.py # ProxySeeker: only sees proxy signal
│   │   └── brains/
│   │       └── base_cnn.py     # BaseCNN neural network architecture
│   │
│   ├── environments/
│   │   ├── base.py             # Environment abstract base class
│   │   └── torch_env.py        # GPU-native vectorized environment
│   │
│   ├── configs/
│   │   └── default_config.py   # CellType enum, get_config(), defaults
│   │
│   ├── training/
│   │   ├── train_ppo.py        # CLI wrapper for PPO training
│   │   ├── ppo/                # Modular PPO implementation
│   │   │   ├── trainer.py      # PPOTrainer class (main training loop)
│   │   │   ├── algorithms.py   # GAE computation & PPO update
│   │   │   └── models.py       # ValueHead, Profiler utilities
│   │   ├── reward_shaping.py   # Potential-based reward shaping
│   │   ├── train_dashboard.py  # Live multi-mode training dashboard
│   │   ├── train_log.py        # Structured CSV/JSON logging
│   │   ├── verification/       # Model fitness testing
│   │   │   ├── directional.py  # Direction accuracy tests
│   │   │   └── survival.py     # Survival simulation tests
│   │   └── visualize_saliency.py  # Neural network interpretability
│   │
│   └── utils/
│       ├── device.py           # Centralized PyTorch device selection
│       ├── brain_viz.py        # Neural network visualization
│       └── logging_config.py   # Logging setup
│
├── tests/                      # pytest test suite
├── docs/                       # Additional documentation
├── models/                     # Saved model weights
├── logs/                       # Training logs (CSV/JSON)
│
├── Dockerfile.rocm             # AMD GPU (ROCm) container
├── Dockerfile.cuda             # NVIDIA GPU (CUDA) container
├── Dockerfile.cpu              # CPU-only container
└── compose.yaml                # Docker Compose with profiles
```

---

## Quick Start

### Prerequisites
- Python 3.9+
- PyTorch 2.0+ (with CUDA/ROCm for GPU acceleration)
- NumPy, Matplotlib, tqdm, pytest

### Installation

```bash
# Clone and install
git clone https://github.com/yourusername/goodharts_law.git
cd goodharts_law
pip install -r requirements.txt

# Or install as editable package
pip install -e .
```

### Running the Visual Demo

```bash
# Default: OmniscientSeeker vs ProxySeeker
python main.py

# With all agent types (hardcoded + learned)
python main.py --learned

# Brain view: visualize neural network internals
python main.py --brain-view --agent ground_truth
```

The visualization shows:
1. **Live Grid** — Agents navigating food (teal) and poison (coral)
2. **Energy Plot** — Average energy over time per behavior type
3. **Activity Heatmap** — Where agents spend time (with type filter)
4. **Death Statistics** — Stacked bar chart by cause and agent type

---

## Configuration

Configuration is managed via TOML files with a fallback chain:
- `config.toml` (your customizations, gitignored)
- `config.default.toml` (shipped defaults)

### Key Configuration Sections

```toml
[world]
width = 100
height = 100
loop = true         # Edges wrap (toroidal) vs walls

[resources]
food = 500
poison = 50
respawn = true      # Consumed items respawn

[agent]
view_range = 5      # Agent vision radius (view = 11x11)
energy_start = 50.0
energy_move_cost = 0.1

[training]
# Curriculum: density ranges for robust training
min_food = 50
max_food = 200
min_poison = 20
max_poison = 100

# PPO hyperparameters
learning_rate = 0.001
gamma = 0.99
steps_per_env = 128     # Horizon length for GAE
k_epochs = 4            # PPO update epochs
entropy_coef = 0.02     # Exploration bonus

[[agents]]
type = "OmniscientSeeker"
count = 5

[[agents]]
type = "ground_truth"   # Learned agent preset
count = 3
```

### CLI Overrides

```bash
# Override config values
python main.py --food 100 --poison 20 --agents 10

# Set device
GOODHARTS_DEVICE=cuda:1 python main.py
```

---

## Training

### PPO Training (Recommended)

The primary training method uses Proximal Policy Optimization with Generalized Advantage Estimation:

```bash
# Train a ground truth agent
python -m goodharts.training.train_ppo --mode ground_truth --timesteps 100000

# Train all modes in parallel
python -m goodharts.training.train_ppo --mode all --timesteps 100000

# With live dashboard showing all modes
python -m goodharts.training.train_ppo --mode all --dashboard --timesteps 100000

# More parallel environments for higher throughput
python -m goodharts.training.train_ppo --n-envs 128 --timesteps 200000
```

### Key CLI Options

| Option | Description |
|--------|-------------|
| `--mode MODE` | Training mode: `ground_truth`, `proxy`, `ground_truth_blinded`, `all`, or comma-separated |
| `--timesteps N` | Total environment steps (default: from config) |
| `--updates N` | PPO updates instead of timesteps (more intuitive for long runs) |
| `-d, --dashboard` | Live training visualization |
| `-e, --n-envs N` | Parallel environments (higher = faster, more VRAM) |
| `--benchmark` | Measure throughput without saving models |
| `--deterministic` | Full reproducibility (slower) |
| `--seed N` | Random seed for reproducibility |
| `--no-amp` | Disable mixed precision (for debugging) |
| `--no-compile` | Disable torch.compile (faster startup, slower training) |

### Training Performance

With the GPU-native vectorized environment and optimized PPO:
- GPU training is 10-30x faster than CPU
- Exact throughput varies by configuration (n_envs, compile settings) and hardware
- Use `--benchmark` flag to measure throughput on your system

### Training Output

Training produces:
- **Model weights**: `models/ppo_{mode}.pth`
- **Episode logs**: `logs/{mode}_{timestamp}_episodes.csv`
- **Update logs**: `logs/{mode}_{timestamp}_updates.csv`  
- **Summary**: `logs/{mode}_{timestamp}_summary.json`

### Model Verification

```bash
# Run verification suite
python -m goodharts.training.verification --steps 500 --verbose
```

---

## Evaluation

The evaluation system uses a **continuous survival paradigm**: agents run until they die (starvation), then auto-respawn. We track death events and survival times, not artificial "episodes".

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Efficiency** | food / (food + poison) — the Goodhart failure metric |
| **Survival** | Steps lived before each death |
| **Deaths/1k** | Population death rate per 1000 steps |
| **Food/1k** | Food consumption rate per 1000 steps |
| **Poison/1k** | Poison consumption rate per 1000 steps |

### Usage

```bash
# Evaluate a single mode
python scripts/evaluate.py --mode ground_truth --timesteps 100000

# Evaluate all modes with comparison
python scripts/evaluate.py --mode all --timesteps 100000

# With live dashboard
python scripts/evaluate.py --mode all --dashboard

# Deterministic evaluation (reproducible)
python scripts/evaluate.py --mode all --deterministic --seed 42
```

Results are saved to `generated/eval_results.json` with cross-mode comparison.

---

## Architecture Deep Dive

### CellType System

All cell types are defined in `configs/default_config.py` with intrinsic properties:

| Cell | Value | Color | Interestingness | Energy Effect |
|------|-------|-------|-----------------|---------------|
| EMPTY | 0 | Dark blue | 0.0 | — |
| WALL | 1 | Gray | 0.0 | — |
| FOOD | 2 | Teal | **1.0** | + (configurable) |
| POISON | 3 | Coral | **0.9** | - (configurable) |
| PREY | 4 | Cyan | 0.3 | — |
| PREDATOR | 5 | Red | 1.0 | + (configurable) |

Adding new cell types is simple—add to `CellType` class, and observation channels auto-expand.

### Observation Encoding

Observations are multi-channel tensors of shape `(num_channels, view_size, view_size)`:

**Ground Truth Mode** — One-hot encoding per cell type:
```
Channel 0: is_empty (0/1)
Channel 1: is_wall (0/1)
Channel 2: is_food (0/1)
Channel 3: is_poison (0/1)
Channel 4: is_prey (0/1)
Channel 5: is_predator (0/1)
```

**Proxy Mode** — Interestingness signal replaces food/poison distinction:
```
Channel 0: is_empty (0/1)
Channel 1: is_wall (0/1)
Channels 2-5: interestingness value (0.0-1.0)
```

### Neural Network (BaseCNN)

```
Input: (6, 11, 11) — 6 channels, 11×11 view
  ↓
Conv2D(6→32, 3×3, padding=1) + ReLU
  ↓
Conv2D(32→64, 3×3, padding=1) + ReLU
  ↓
Conv2D(64→64, 3×3, padding=1) + ReLU
  ↓
Flatten: 64 × 11 × 11 = 7,744 features
  ↓
Linear(7744→512) + ReLU  [PPO value head branches here]
  ↓
Linear(512→8) — 8 directional actions
```

### Action Space

8-directional movement (no "stay"):

| Index | Action | Direction |
|-------|--------|-----------|
| 0 | (-1, -1) | ↖ Up-Left |
| 1 | (-1, 0) | ← Left |
| 2 | (-1, 1) | ↙ Down-Left |
| 3 | (0, -1) | ↑ Up |
| 4 | (0, 1) | ↓ Down |
| 5 | (1, -1) | ↗ Up-Right |
| 6 | (1, 0) | → Right |
| 7 | (1, 1) | ↘ Down-Right |

### Vectorized Environment (TorchVecEnv)

The training environment is GPU-native—all state lives in PyTorch tensors with no CPU-GPU transfer during training:

- **Batched state**: `grids: (n_envs, H, W)`, `agent_x/y: (n_envs,)` — all on GPU
- **Vectorized step**: All environments advance simultaneously via tensor operations
- **Zero-copy observations**: Views into grid tensors, no data movement
- **Shared grid mode**: Multiple agents in single world (for visualization)

### Behavior Registry

Behaviors are auto-discovered via class introspection:

```python
from goodharts.behaviors import get_behavior, list_behavior_names, create_learned_behavior

# List all available behaviors
print(list_behavior_names())
# ['LearnedBehavior', 'OmniscientSeeker', 'ProxySeeker', ...]

# Get behavior by name
BehaviorClass = get_behavior('OmniscientSeeker')

# Create learned behavior from preset
behavior = create_learned_behavior('ground_truth', model_path='models/my_model.pth')
```

---

## AI Safety Connection

This simulation is a **toy model** for understanding real AI alignment failures:

### 1. Specification Gaming
The proxy-seeker optimizes exactly what we told it to (highest signal), but this doesn't align with what we actually want (survival). This mirrors real-world cases where AI systems find unintended solutions that technically satisfy the objective.

### 2. Information Asymmetry
The agent lacks access to ground truth—a common scenario when we can't fully specify what we want. In the real world, we often can't give AI systems complete information about human values.

### 3. Distributional Shift
The `proxy` mode shows what happens when even the *reward signal* is misaligned. The agent learns to seek "interesting" things regardless of their actual value—optimizing a proxy of a proxy.

### 4. Meta-Lesson: We Fell Into Our Own Trap
During development, we initially used behavior cloning (training CNNs to mimic expert heuristics). This failed because:
- The expert (OmniscientSeeker) never encountered poison (it avoided it perfectly)
- The CNN had zero training examples of "what to do near poison"
- We were optimizing for **imitation**, not **survival**

This accidental demonstration of Goodhart's Law on ourselves is documented in `docs/goodhart_self_proven.md`.

---

## Roadmap

### ✅ Phase 1: Measurement & Visualization
- [x] Death cause tracking (starvation vs poison)
- [x] Per-behavior energy charts
- [x] Activity heatmaps with type filtering
- [x] Behavior color system

### ✅ Phase 2: Learned Behaviors
- [x] BaseCNN architecture with dynamic channels
- [x] PPO training with GAE and curriculum
- [x] GPU-native vectorized training
- [x] Multi-mode dashboard
- [x] Structured logging (CSV/JSON)
- [x] Model verification suite

### ✅ Phase 2.5: Code Quality
- [x] TOML configuration system
- [x] Auto-discovery behavior registry
- [x] Centralized device selection
- [x] Comprehensive test suite

### 🔮 Phase 3: Emergent Deception
- [ ] Multi-agent signaling dynamics
- [ ] Resource competition under scarcity
- [ ] Agents that "game" the proxy metric
- [ ] Adversarial inspector/gamer co-evolution

---

## Docker Support

### AMD GPUs (ROCm)
```bash
docker compose --profile rocm up -d --build
docker compose exec -it dev-rocm bash
python main.py
```

### NVIDIA GPUs (CUDA)
```bash
docker compose --profile cuda up -d --build
docker compose exec -it dev-cuda bash
python main.py
```

### CPU Only
```bash
docker compose --profile cpu up -d --build
docker compose exec -it dev-cpu bash
python main.py
```

---

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Style
The project uses type hints throughout. Key conventions:
- `CellType` for cell type references (not raw integers)
- `ObservationSpec` for model input configuration
- Factory functions over direct instantiation (`create_learned_behavior`, `create_vec_env`)

---

## Disclosure

This project was developed initially using Google's experimental Antigravity agentic IDE, with the gracious assistance of Claude 4.5 Opus and Gemini 3 Pro. Due to emergent lag in Antigravity, I later switched to Claude Code, entirely with 4.5 Opus. I acted as architect and systems lead, manually writing some code where relevant, but I credit Opus for the vast majority of the implementation and documentation. This project would not have been possible without Google's free Gemini Pro subscription for students, the experimental high usage limits in Antigravity, and the extreme efficiency and potential of Claude Code.

I am grateful to Google and Anthropic for making this project possible.

---

## License

See the Unlicense.
