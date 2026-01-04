# Goodhart's Law Demonstrator

> "When a measure becomes a target, it ceases to be a good measure." - Charles Goodhart, paraphrased

This project is an empirical demonstration of Goodhart's law in reinforcement learning. Agents navigate a 2D grid world with the goal of increasing their energy. **Ground truth** agents see real cell types and thrive. **Blinded** agents see only interestingness and fail to thrive. **Proxy** agents rewarded on interestingness rather than energy rapidly self-destruct.

Optimizing for proxy metrics nearly always leads to alignment failures. Whether the proxy is reasonable or unreasonable, alignment still fails.

---

## Results

Trained agents (2048 updates, 192 envs, 1 minibatch, seed 42) evaluated using continuous survival testing (16384 steps, 8192 envs, 3 runs, base seed 42):

| Mode                  | Observation     | Reward          | Efficiency | Survival | Deaths/1k | Food/1k | Poison/1k |
|-----------------------|-----------------|-----------------|------------|----------|-----------|---------|-----------|
| **ground_truth**      | Cell types      | Energy          | 100.0%     | 16371.3  | 0.00      | 156.4   | 0.0       |
| ground_truth_handhold | Cell types      | Shaped          | 99.9%      | 16360.7  | 0.00      | 149.7   | 0.1       |
| **proxy**             | Interestingness | Interestingness | **45.3%**  | 55.1     | 18.08     | 60.8    | 77.6      |
| ground_truth_blinded  | Interestingness | Energy          | 95.8%      | 116.2    | 8.55      | 0.4     | 0.0       |

**Key metrics:**
- **Efficiency** = total food / total consumed (food + poison), the core Goodhart metric
- **Survival** = average steps lived before death in each run
- **Deaths/1k** = agent deaths per 1000 total steps

### Interpretation

The proxy agent is completely unfit for the environment.

1. **Worse than random**: Proxy agents eat significantly LESS food than poison (60.8/1k versus 77.6/1k). Agents actively prefer poison because it has higher interestingness than food. The effect is also very minor; even when reversed so food is twice as interesting, the difference in consumption is minimal.

2. **383,000x death rate**: Ground truth agents die so infrequently that it is not measured in the table above (see the generated report). Proxy agents die approximately every 55 steps.

3. **Blinded control:** The ground_truth_blinded agent receives real energy rewards but can only see interestingness. This agent achieves very high efficiency but consumes very little of anything, leading to a massive amount of death by starvation.

The efficiency gap of 54.7% appears to be moderate, but the death count speaks for itself. Optimizing a measurable proxy goes beyond being minimally effective and easily enters the realm of catastrophic failure.

### Statistical notes

- Results were aggregated across three independent validation runs for each mode, with different random seeds derived from the same base seed.
- Cohen's d is extremely large for the efficiency comparison between ground_truth and proxy; the distributions are almost completely incomparable.
- The 383,000x death rate does not meaningfully change between runs.
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
- [Roadmap](#roadmap)
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

With the GPU-native vectorized environment and optimized PPO:
- GPU training is 10-30x faster than CPU
- Exact throughput varies by configuration (n_envs, compile settings) and hardware
- Use `--benchmark` flag to measure throughput on your system

Note: Throughput depends heavily on hardware and settings. On an AMD Radeon RX 7700S with ROCm 7.10, typical throughput is 15-25k steps/second with torch.compile enabled.

### Training Output

Training produces:
- **Model weights**: `models/ppo_{mode}.pth`
- **TensorBoard logs**: `runs/{mode}_{timestamp}/` (view with `tensorboard --logdir runs/`)

### Model Verification

Note: The verification module (`goodharts.training.verification`) currently has an import error and needs repair.

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
| FOOD | 1 | Teal | **0.5** | +1.0 |
| POISON | 2 | Coral | **1.0** | -2.0 |

The interestingness values are deliberately anti-correlated: poison is MORE interesting than food, creating the Goodhart trap.

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

**Real-world parallel:** The different algorithms on TikTok and its internal Chinese variant, Douyin, are an excellent example. Douyin emphasizes positive and educational content, while TikTok optimizes solely for profit. This proxy metric demands only that as many ads be served as possible, and the best way to do that is to make people angry, or scared, or anything else that keeps them engaged. Douyin's algorithm benefits the Chinese people, while TikTok's drives global tension up. Same company, same technology, different objectives, different outcomes.

### 2. Information Asymmetry
Describing an objective to an agent in exact terms is already nearly impossible; giving the agent *perfect information* is *genuinely* impossible.

**Real-world parallel:** Educational systems, both human and AI-operated, optimize for test scores rather than genuine understanding. Teaching to the test is a plague throughout the United States, not least because those test scores are used as judges of capability by *other* systems as well. Humans do not have the necessary energy to decode true objective information. AI agents do not have any *reason* to seek out a deeper 'true' objective. They only optimize for what they are rewarded on.

### 3. Distributional Shift
The proxy mode demonstrates what happens when the reward system itself is misaligned. The agent learns only to seek interesting things. Because the agent cannot experience death in the training loop, it never learns anything about its true goals.

**Real-world parallel:** Click-through rate optimization across the entire internet. Charitably, CTR can be said to proxy whether the user finds a link valuable. Less charitably, it proxies likely engagement with advertisements and other material. When only positive content of varying quality is available, CTR selects for the best of it, enlightening and entertaining the audience. In our modern world, laden with misinformation and propaganda, CTR selects blindly for engagement and warps our societies in the process. The proxy selects for what is worst for the public.

### 4. Meta-Lesson: We Fell Into Our Own Trap
During development, we initially used behavior cloning, training CNNs to mimic expert heuristics. This failed miserably because:
- The expert (OmniscientSeeker) never encountered poison (it avoided it perfectly)
- The CNN had zero training examples of what to do about poison
- We were optimizing for **imitation**, not **survival**

This accidental demonstration of Goodhart's law is documented in `docs/goodhart_self_proven.md`.

---

## Limitations

This project demonstrates Goodhart's law in a simplified and heavily controlled setting. We acknowledge the following limitations.

### Intentional Simplifications

- **Extreme proxy design**: Interestingness is *deliberately* anti-correlated with the true objective. Poison being twice as interesting as food can be related to some proxy failures (junk food, drugs, CTR optimization) but is significantly less subtle than real-world proxy failures. The demonstration is unmistakable but likely very overstated regarding how detectable and dramatic misalignment is in the real world.

- **Single environment**: The 2D grid world is a toy domain. Multi-agent simulation would enable richer failure modes but require architectural rework.

- **Single architecture**: All learned agents use the same CNN architecture. The failure mode is architecture-agnostic, stemming from information and reward rather than model capacity (and thus likely to worsen with more powerful models), but this has not been explicitly demonstrated.

### What This Doesn't Demonstrate

- **Emergent deception**: Agents don't learn to hide their proxy-seeking behavior. This is a future aspiration that requires reworking the Torch vectorized environment to allow multiple agents to exist on the same grid.

- **Reward hacking**: Agents optimize the given reward faithfully. The failure is in *what* they're rewarded for, not *how* they optimize it. While this is useful in many cases, it excludes the more concerning case of true reward hacking (likely not possible in this simplified environment).

- **Distributional shift during deployment**: Training and evaluation use the same environment distribution. Real alignment failures often emerge under distribution shift.

### What This Does Demonstrate

Even with limitations, this project empirically validates the core mechanism of Goodhart's law.
- Optimizing a measurable proxy (interestingness) produces behavior that anti-correlates with the true objective (survival) unless the proxy is extremely strongly correlated to the true objective
- The failure is quantifiable to an absurd degree: Worse-than-random proxy efficiency, 6-order-of-magnitude death rate increase
- Control conditions (ground_truth_blinded) isolate the effect of information vs reward - a model with bad information but good rewards will either learn to perform or fail gracefully
- This failure is global across different random seeds and training runs

The claim is narrowed by these limitations, but the core thesis is not. **Optimization makes a bad proxy worse**.

---

## Roadmap

### Phase 1: Measurement & Visualization
- [x] Death cause tracking (starvation vs poison)
- [x] Per-behavior energy charts
- [x] Activity heatmaps with type filtering
- [x] Behavior color system

### Phase 2: Learned Behaviors
- [x] BaseCNN architecture with dynamic channels
- [x] PPO training with GAE and curriculum
- [x] GPU-native vectorized training
- [x] Multi-mode dashboard
- [x] Structured logging (CSV/JSON)
- [x] Model verification suite

### Phase 2.5: Code Quality
- [x] TOML configuration system
- [x] Auto-discovery behavior registry
- [x] Centralized device selection
- [x] Comprehensive test suite

### Phase 3: Emergent Deception
- [ ] Multi-agent signaling dynamics
- [ ] Resource competition under scarcity
- [ ] Agents that "game" the proxy metric
- [ ] Adversarial inspector/gamer co-evolution

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

This project was developed initially using Google's experimental Antigravity agentic IDE, with the gracious assistance of Claude 4.5 Opus and Gemini 3 Pro. Due to emergent lag in Antigravity, I later switched to Claude Code, primarily with 4.5 Opus. I acted as architect and systems lead, manually writing some code where relevant, but I credit Opus for the vast majority of the implementation and documentation along with serving as a valuable advisor. This project would not have been possible without Google's free Gemini Pro subscription for students, the high usage limits in Antigravity, and the extreme efficiency and potential of Claude Code.

I am grateful to Google and Anthropic for making this project possible.
