# Goodhart's Law Demonstrator

> "When a measure becomes a target, it ceases to be a good measure." - Charles Goodhart, paraphrased

This project is an empirical demonstration of Goodhart's Law in reinforcement learning. Agents navigate a 2D grid world with the goal of increasing their energy. **Ground truth** agents see real cell types and thrive. **Blinded** agents see only interestingness and fail to thrive. **Proxy** agents rewarded on interestingness rather than energy rapidly self-destruct.

Optimizing for proxy metrics can lead to alignment failures even when the proxy seems reasonable. The failure mode is robust across experimental conditions.

---

## Results

Trained agents (2048 updates, 192 envs, 1 minibatch, seed 42) evaluated using continuous survival testing (16384 steps, 8192 envs, 3 runs, base seed 42):

| Mode                  | Energy/1k | Efficiency | Survival | Deaths/1k | Food/1k | Poison/1k |
|-----------------------|-----------|------------|----------|-----------|---------|-----------|
| **ground_truth**      | **+144.6**|   99.9%    | 16364.0  |   0.00    |  154.7  |    0.1    |
| ground_truth_blinded  |   -12.8   |    29.9%   |    95.1  |  10.46    |   55.8  |   29.3    |
| proxy_mortal          |   -41.5   |    24.7%   |    41.5  |  24.03    |   59.0  |   45.3    |
| **proxy**             | **-58.7** | **23.8%**  |    29.0  |  34.43    |   79.2  |   64.0    |

**Key metrics:**
- **Energy/1k** = net energy change per 1000 steps (positive = thriving, negative = dying)
- **Efficiency** = mean per-agent efficiency (food / total consumed per agent)
- **Survival** = average steps lived before death
- **Deaths/1k** = agent deaths per 1000 total steps

### Interpretation

The proxy agent is completely unfit for the environment.

1. **Energy dynamics tell the story**: Ground truth agents gain +144.6 energy per 1000 steps - they're thriving. Proxy agents lose -58.7 energy per 1000 steps - they're hemorrhaging energy and dying rapidly. The 203 energy gap per 1000 steps is the quantified cost of Goodhart's Law.

2. **564x survival collapse**: Ground truth agents survive an average of 16,364 steps. Proxy agents survive just 29 steps. This 564x ratio captures the catastrophic failure mode.

3. **Two distinct failure modes**:
   - **Proxy agents** fail by *over-poisoning*: they consume everything aggressively (79.1 food + 63.9 poison per 1k steps) because interestingness rewards consumption without encoding harm. Per-agent efficiency is only 23.8%.
   - **Blinded agents** fail by *under-consumption*: they consume cautiously (55.7 food + 29.3 poison per 1k steps) because they can't distinguish food from poison, so they avoid both. Per-agent efficiency is 29.9%.

4. **Proxy metric design doesn't save you**: Food is MORE interesting than poison (1.0 vs 0.5), yet proxy agents still consume poison at catastrophic rates. Making the proxy "reasonable" doesn't prevent failure - the metric simply doesn't encode harm.

The 76% efficiency gap (99.9% vs 23.8%) tells the story: ground truth agents almost never eat poison, while proxy agents eat poison more often than food.

### Statistical notes

- Results were aggregated across three independent evaluation runs for each mode, with different random seeds derived from the same base seed.
- Cohen's d is extremely large for the efficiency comparison between ground_truth and proxy; the distributions are almost completely non-overlapping.
- The 564x survival collapse ratio is stable across runs.
- See [docs/evaluation_protocol.md](docs/evaluation_protocol.md) for the full methodology.

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
- [Future Directions](#future-directions)
- [Disclosure](#disclosure)

---

## Core Concept

This project is a demonstrator for the results of optimizing for a proxy objective when it is not directly tied to the true objective.

### The Goodhart Trap

| Agent Type | Can See | Optimizes For | Outcome         |
|------------|---------|---------------|-----------------|
| **Ground Truth Agent** | Real cell types (food vs poison) | Eating food, avoiding poison | Thrives         |
| **Proxy Agent** | Proxy signal only (interestingness) | Highest signal cells | Fails to thrive |

**Key insight:** Whether poison's interestingness is higher or lower than food, proxy agents devour both as fast as they can. The actual difference in measure is minimally relevant.

### Training Modes

| Mode | Observation | Reward Signal | Can Die | Purpose |
|------|-------------|---------------|---------|---------|
| `ground_truth` | One-hot cell types | Energy delta | Yes | Baseline: full information |
| `ground_truth_blinded` | Interestingness values | Energy delta | Yes | Control: correct rewards, blinded observation |
| `proxy_mortal` | Interestingness values | Interestingness gain | Yes | Partial grounding: proxy reward but real consequences |
| `proxy` | Interestingness values | Interestingness gain | No* | **Main Goodhart case**: completely unmoored from reality |
| `ground_truth_handhold` | One-hot cell types | Shaped rewards | Yes | Experimental (not in default evaluation) |

*Proxy agents are immortal during training (frozen energy) to isolate the proxy optimization effect.

---

## Project Structure

```
goodharts_law/
├── main.py                     # Visual demo entry point
├── config.default.toml         # Configuration template
├── goodharts/                  # Main package
│   ├── simulation.py           # Visual demo orchestrator
│   ├── modes.py                # Training modes (ObservationSpec, RewardComputer)
│   ├── config.py               # TOML config loader
│   ├── behaviors/              # Agent decision-making
│   │   ├── registry.py         # Auto-discovery system
│   │   ├── learned.py          # Neural network agents
│   │   ├── hardcoded/          # Baseline heuristics (OmniscientSeeker, ProxySeeker)
│   │   └── brains/base_cnn.py  # CNN architecture
│   ├── environments/
│   │   └── torch_env.py        # GPU-native vectorized environment
│   ├── training/
│   │   ├── train_ppo.py        # Training CLI
│   │   └── ppo/                # PPO implementation (trainer, algorithms, models)
│   ├── evaluation/             # Evaluation infrastructure
│   │   └── evaluator.py        # Core evaluation logic
│   └── analysis/               # Statistical analysis and visualization
│   └── cli/                    # CLI modules
│       ├── evaluate.py         # Evaluation CLI
│       ├── brain_view.py       # Neural network visualization
│       └── parallel_stats.py   # Multi-mode comparison
├── scripts/                    # Development tools (profiling, benchmarking)
├── tests/                      # pytest suite
├── models/                     # Saved weights
└── generated/                  # Evaluation outputs
```

---

## Quick Start

### Prerequisites
- Python 3.9+
- PyTorch 2.0+ (with CUDA/ROCm for GPU acceleration)
- Dependencies: NumPy, matplotlib, Plotly, Dash, tqdm, pytest

### Installation

```bash
git clone https://github.com/yourusername/goodharts_law.git
cd goodharts_law
pip install -e .
```

### Running the Visual Demo

```bash
# Brain view: visualize neural network internals (matplotlib)
python main.py brain-view -m ground_truth

# Parallel stats: compare multiple modes (Dash dashboard)
python main.py parallel-stats --modes ground_truth,proxy --envs 256
```

The visualization shows an agent navigating the grid, consuming food (teal) and poison (coral). Energy plots track survival over time per behavior type.

---

## Configuration

Configuration is managed via TOML files with a fallback chain:
- `config.toml` (your customizations, gitignored)
- `config.default.toml` (shipped defaults)

### Key Configuration Sections

```toml
[world]
width = 128
height = 128

[resources]
food = 128
poison = 128
respawn = true

[agent]
view_range = 5           # Observation radius (creates 11x11 view)
energy_start = 1.0
energy_move_cost = 0.01

[training]
learning_rate = 0.0003
gamma = 0.99
n_envs = 192
steps_per_env = 128

[cell_types.food]
energy_delta = 1.0
interestingness = 0.5

[cell_types.poison]
energy_delta = -2.0
interestingness = 1.0    # MORE interesting than food - the Goodhart trap
```

### CLI Overrides

```bash
# Set device via environment variable
GOODHARTS_DEVICE=cuda:1 python main.py brain-view

# Config customization: copy config.default.toml to config.toml and edit
cp config.default.toml config.toml
# Edit config.toml to change grid size, resource counts, etc.
```

---

## Training

### PPO

The primary training method uses Proximal Policy Optimization with Generalized Advantage Estimation:

```bash
# Train a ground truth agent
python -m goodharts.training.train_ppo --mode ground_truth --updates 128

# Train all modes in parallel
python -m goodharts.training.train_ppo --mode all --updates 128

# With live dashboard showing all modes
python -m goodharts.training.train_ppo --mode all --dashboard --updates 128

# More parallel environments for higher throughput
python -m goodharts.training.train_ppo --n-envs 128 --updates 128
```

### Key CLI Options

| Option | Description |
|--------|-------------|
| `-m, --mode MODE` | Training mode: `ground_truth`, `proxy`, `ground_truth_blinded`, `all`, or comma-separated |
| `-t, --timesteps N` | Total environment steps |
| `-u, --updates N` | PPO updates (alternative to timesteps) |
| `-e, --n-envs N` | Parallel environments (higher = faster, more VRAM) |
| `-d, --dashboard` | Live training visualization |
| `--seed N` | Random seed for reproducibility |
| `--deterministic` | Full reproducibility (slower) |
| `-b, --benchmark` | Measure throughput without saving models |
| `--no-amp` | Disable mixed precision (for debugging) |
| `--compile-mode` | torch.compile mode: `reduce-overhead`, `max-autotune` |

### Training Performance

The training pipeline evolved from ~50 steps/second (naive Python) to ~25,000 steps/second through GPU-native environment design, async logging, and torch.compile optimization. Key insight: the environment itself must live on GPU, not just the model.

Use `--benchmark` to measure throughput on your system. Results vary by hardware and settings.

### Training Output

Training produces:
- **Model weights**: `models/ppo_{mode}.pth`
- **TensorBoard logs**: `runs/{mode}_{timestamp}/` (view with `tensorboard --logdir runs/`)

### Model Verification

Verify trained models meet fitness thresholds:

```bash
python -m goodharts.training.verification --steps 1000 --verbose
```

This runs directional and survival checks to ensure models learned meaningful behaviors.

---

## Evaluation

In evaluation, agents run until death, then respawn on the step after death. We track death events and survival times directly.

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
python main.py evaluate --mode ground_truth --timesteps 100000

# Evaluate all modes with comparison
python main.py evaluate --mode all --timesteps 100000

# Multi-run with statistical aggregation
python main.py evaluate --mode all --runs 5 --base-seed 42

# Full report with figures and markdown
python main.py evaluate --full-report --runs 5 --timesteps 50000
```

Results are saved to `generated/eval_results.json` with cross-mode comparison.

---

## Architecture Deep Dive

### CellType System

Cell types are defined in `config.default.toml` with intrinsic properties:

| Cell | Value | Color | Interestingness | Energy Delta |
|------|-------|-------|-----------------|--------------|
| EMPTY | 0 | Dark blue | 0.0 | 0.0 |
| FOOD | 1 | Teal | **1.0** | +1.0 |
| POISON | 2 | Coral | **0.5** | -2.0 |

The interestingness values demonstrate an incomplete proxy metric: food is MORE interesting than poison, yet agents still consume poison because interestingness doesn't encode harm. The proxy isn't adversarial - it's just incomplete.

### Observation Encoding

Observations are 2-channel tensors of shape `(2, view_size, view_size)`:

**Ground Truth Mode** — Separate channels for food and poison:
```
Channel 0 (food):   Food=[1], Poison=[0], Empty=[0]
Channel 1 (poison): Food=[0], Poison=[1], Empty=[0]
```

**Proxy Mode** — Both channels show interestingness (indistinguishable):
```
Channel 0: Food=[0.5], Poison=[1.0], Empty=[0]
Channel 1: Food=[0.5], Poison=[1.0], Empty=[0]
```

The proxy agent cannot distinguish food from poison - both appear as "interesting" cells with different magnitudes but no type information.

### Neural Network (BaseCNN)

```
Input: (2, 11, 11) — 2 channels, 11×11 view
  ↓
Conv2D(2→32, 3×3, padding=1) + ReLU
  ↓
Conv2D(32→64, 3×3, padding=1) + ReLU
  ↓
Conv2D(64→64, 3×3, padding=1) + ReLU
  ↓
Flatten: 64 × 11 × 11 = 7,744 features
  ↓
Linear(7744→512) + ReLU  [Value head branches here]
  ↓
Linear(512→8) — 8 directional actions
```

The value head (for PPO) branches after the fc1 layer and computes V(s) from the shared features.

### Action Space

8-directional movement with no stay-in-place option:

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

The action space is pluggable via `action_space_type` config. Default is `discrete_grid` with `max_move_distance=1`. Other options: `continuous`, `factored`.

### Vectorized Environment (TorchVecEnv)

The training environment is GPU-native. All state lives in PyTorch tensors with as little CPU-GPU transfer during training as possible.

- **Batched state**: `grids: (n_envs, H, W)`, `agent_x/y: (n_envs,)` — all on GPU
- **Vectorized step**: All environments advance simultaneously via tensor operations
- **Zero-copy observations**: Views into grid tensors, no data movement

Performance optimizations:
- `torch.compile` on policy forward pass (configurable mode)
- Mixed precision (AMP) for faster GPU compute
- Pinned memory for async GPU-to-CPU metric transfer
- Environment step is also compilable for additional speedup

### Behavior Registry

Behaviors are auto-discovered via class introspection:

```python
from goodharts.behaviors import get_behavior, list_behavior_names, create_learned_behavior

# List all available behaviors
print(list_behavior_names())
# ['LearnedBehavior', 'OmniscientSeeker', 'ProxySeeker']

# Get behavior class by name
BehaviorClass = get_behavior('OmniscientSeeker')

# Create learned behavior from preset
behavior = create_learned_behavior('ground_truth', model_path='models/ppo_ground_truth.pth')
```

---

## AI Safety Connection

This simulation is a **toy model** for understanding real AI alignment failures, as well as general optimization failures in the real world.

### 1. Precision in Specification
The proxy agent optimizes for exactly what it is rewarded on - consume as much interestingness as possible. Training freezes energy to prevent the agent from learning poison is dangerous, which only *works* because there's a backup reward signal. When that penalty is removed and the agent optimizes purely for the proxy metric, it fails completely. This mirrors the real world; so long as an AI agent is given a sufficiently robust objective, it will accomplish the goal, but the robustness of that objective is vital.

**Real-world parallel:** The different algorithms on TikTok and its Chinese variant Douyin illustrate this well. Chinese law requires Douyin to emphasize educational and positive content for minors, while TikTok optimizes purely for engagement and profit - as any company would absent such constraints. Same technology, different regulatory objectives, different outcomes. The algorithm does exactly what it's optimized for.

### 2. Information Asymmetry
Describing an objective to an agent in exact terms is already nearly impossible; giving the agent *perfect information* is *genuinely* impossible.

**Real-world parallel:** Educational systems, both human and AI-operated, optimize for test scores rather than genuine understanding. Test scores become targets, then cease to measure what they were designed to measure. Humans do not have the necessary energy to decode true objective information. AI agents do not have any *reason* to seek out a deeper 'true' objective. They only optimize for what they are rewarded on.

### 3. Distributional Shift
The proxy mode demonstrates what happens when the reward system itself is misaligned. The agent learns only to seek interesting things. Critically, proxy agents are *immortal during training* - they cannot experience the consequences of their choices. This is deliberate: real-world AI systems rarely have clean "death penalties" that provide corrective feedback. The `proxy_mortal` mode tests what happens when consequences ARE available, and still shows significant failure (56.6% efficiency vs 100%).

**Real-world parallel:** Click-through rate optimization across the entire internet. Charitably, CTR can be said to proxy whether the user finds a link valuable. Less charitably, it proxies likely engagement with advertisements and other material. When only positive content of varying quality is available, CTR selects for the best of it, enlightening and entertaining the audience. In our modern world, laden with misinformation and propaganda, CTR selects blindly for engagement and warps our societies in the process. The proxy selects for what is worst for the public.

### 4. Meta-Lesson: We Fell Into Our Own Trap
During development, we initially used behavior cloning, training CNNs to mimic expert heuristics. This failed miserably because:
- The expert (OmniscientSeeker) never encountered poison (it avoided it perfectly)
- The CNN had zero training examples of what to do about poison
- We were optimizing for **imitation**, not **survival**

This accidental demonstration of Goodhart's Law is documented in `docs/goodhart_self_proven.md`.

---

## Limitations

This project demonstrates Goodhart's Law in a simplified and heavily controlled setting. We acknowledge the following limitations.

### Intentional Simplifications

- **Simplified proxy design**: The proxy metric (interestingness) is incomplete rather than adversarial - food is actually MORE interesting than poison, but the metric doesn't encode harm. This is more realistic than an anti-correlated proxy, but still simpler than real-world proxy failures where the relationship between proxy and objective is often unknown or shifts over time.

- **Single environment**: The 2D grid world is a toy domain. Multi-agent simulation would enable richer failure modes but require architectural rework.

- **Single architecture**: All learned agents use the same CNN architecture. The failure mode is architecture-agnostic, stemming from information and reward rather than model capacity (and thus likely to worsen with more powerful models), but this has not been explicitly demonstrated.

### What This Doesn't Demonstrate

- **Emergent deception**: Agents don't learn to hide their proxy-seeking behavior. This is a future aspiration that requires reworking the Torch vectorized environment to allow multiple agents to exist on the same grid.

- **Reward hacking**: Agents optimize the given reward faithfully. The failure is in *what* they're rewarded for, not *how* they optimize it. While this is useful in many cases, it excludes the more concerning case of true reward hacking (likely not possible in this simplified environment).

- **Distributional shift during deployment**: Training and evaluation use the same environment distribution. Real alignment failures often emerge under distribution shift.

### What This Does Demonstrate

Even with limitations, this project demonstrates the core mechanism of Goodhart's Law.
- Optimizing a measurable proxy (interestingness) produces catastrophic outcomes even when the proxy isn't adversarial - the metric simply fails to encode harm
- The failure is quantifiable: dramatic efficiency gaps and multi-order-of-magnitude death rate increases between ground truth and proxy agents
- Control conditions (ground_truth_blinded) isolate the effect of information vs reward - a model with bad information but good rewards will either learn to perform or fail gracefully
- This failure is global across different random seeds and training runs

The claim is narrowed by these limitations, but the core thesis is not. **Optimization makes a bad proxy worse**.

---

## Future Directions

The demonstration is complete as a single-agent Goodhart failure mode. Multi-agent dynamics (signaling, competition, adversarial co-evolution) would extend this to emergent deception, but require architectural changes to TorchVecEnv.

---

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Style
This project uses type hints throughout. Key conventions:
- `CellType` for cell type references (not raw integers)
- `ObservationSpec` for model input configuration
- Factory functions over direct instantiation (`create_learned_behavior`, `create_vec_env`)

---

## Disclosure

This project was developed with AI assistance, initially using Google's Antigravity IDE (Gemini 3 Pro), then Claude Code (Claude 4.5 Opus).

**My contributions:** All architectural decisions were mine - the GPU-native vectorization strategy, eliminating CPU-GPU sync points, CUDA graph integration, and condensing the rollout loop into a single compiled graph. The experimental design (ground truth vs proxy dichotomy, the blinded control condition, immortality-during-training as a deliberate confound) was mine. For debugging, Claude would assess symptoms while I identified root causes - pair programming where I provided the "why" and Claude provided the "how."

**Claude's contributions:** The vast majority of implementation code and documentation. Claude transformed my architectural sketches into working systems and served as an invaluable advisor throughout.

I am grateful to Google and Anthropic for making this project possible.
