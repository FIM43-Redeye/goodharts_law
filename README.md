# Goodhart's Law Simulation

> "When a measure becomes a target, it ceases to be a good measure." — Goodhart's Law

A multi-agent simulation demonstrating AI alignment failures through **metric optimization vs. true objectives**. Agents navigate a 2D grid world where they can see either the *ground truth* (what's actually food vs. poison) or only a *proxy metric* (an "interestingness" signal that both food and poison emit). Proxy-optimizing agents inevitably eat poison—a visceral demonstration of Goodhart's Law in action.

## Core Concept

This project explores a fundamental AI safety concern: **what happens when agents optimize for a measurable proxy instead of the true objective?**

| Agent Type | Can See | Optimizes For | Outcome |
|------------|---------|---------------|---------|
| **OmniscientSeeker** | Ground truth (food vs poison) | Eating food, avoiding poison | Thrives |
| **ProxySeeker** | Proxy signal only (interestingness) | Highest signal cells | Gets poisoned |

Key insight: Poison has high "interestingness" (0.9) compared to food (1.0), making it nearly as attractive to proxy-optimizing agents—but eating it is lethal.

---

## Project Structure

```
goodharts_law/
├── main.py                     # Entry point: runs animated simulation
├── config.default.toml         # Default configuration (copy to config.toml to customize)
├── pyproject.toml              # Package configuration & dependencies
├── requirements.txt            # Python dependencies
│
├── goodharts/                  # Main package
│   ├── simulation.py           # Core simulation loop (wraps VecEnv)
│   ├── config.py               # TOML config loader with caching
│   ├── visualization.py        # Matplotlib visualization functions
│   │
│   ├── agents/
│   │   └── (Agent logic moved to VecEnv)
│   │
│   ├── behaviors/
│   │   ├── base.py             # BehaviorStrategy base class + ROLE_COLORS
│   │   ├── registry.py         # Auto-discovery behavior registry
│   │   ├── learned.py          # LearnedBehavior + create_learned_behavior()
│   │   ├── action_space.py     # Centralized action definitions
│   │   ├── hardcoded/          # Baseline behaviors (hand-coded)
│   │   │   ├── omniscient.py   # OmniscientSeeker: sees true cell types
│   │   │   └── proxy_seeker.py # ProxySeeker: only sees proxy signal
│   │   └── brains/
│   │       └── base_cnn.py     # BaseCNN model for learned behaviors
│   │
│   ├── environments/
│   │   ├── base.py             # Environment abstract base class
│   │   └── vec_env.py          # VecEnv: High-performance vectorized environment
│   │
│   ├── training/
│   │   ├── train.py            # Behavior cloning training loop
│   │   ├── train_ppo.py        # PPO algorithm implementation (Vectorized)
│   │   ├── collect.py          # Expert demonstration collection
│   │   ├── dataset.py          # Dataset utilities for training
│   │   ├── verification/       # Model fitness testing suite
│   │   │   ├── directional.py  # Direction accuracy tests
│   │   │   └── survival.py     # Survival simulation tests
│   │   └── visualize_saliency.py  # Neural network interpretability
│   │
│   ├── configs/
│   │   ├── default_config.py   # CellType definitions, get_config()
│   │   └── observation_spec.py # Observation encoding specifications
│   │
│   ├── models/                 # Trained model weights (.pth files)
│   │
│   └── utils/
│       ├── device.py           # Centralized PyTorch device selection
│       ├── logging_config.py   # Logging setup
│       ├── brain_viz.py        # Neural network visualization
│       └── numba_utils.py      # Numba JIT acceleration utilities
│
├── tests/                      # pytest test suite
│
├── Dockerfile.rocm             # AMD GPU (ROCm) environment
├── Dockerfile.cuda             # NVIDIA GPU (CUDA) environment
├── Dockerfile.cpu              # CPU-only environment
├── compose.yaml                # Docker Compose with profiles
└── docker_directions.txt       # Docker workflow instructions
```

---

## Quick Start

### Prerequisites
- Python 3.9+
- Dependencies: `numpy`, `matplotlib`, `torch`, `torchvision`, `tqdm`, `pytest`

### Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run the simulation
python main.py
```

A matplotlib window will open showing:
1. **Live Simulation Grid** — agents navigating food (green) and poison (red)
2. **Energy Plot** — average energy over time per species
3. **Activity Heatmap** — where agents spend time
4. **Death Statistics** — starvation vs. poisoning counts

### Configuration

Configuration is managed via TOML files:

```bash
# Uses config.toml if present, otherwise config.default.toml
python main.py

# Or specify a config file
python main.py --config my_config.toml

# CLI flags override config
python main.py --food 100 --poison 20 --agents 10
```

Key config sections:
- `[world]` — Grid dimensions, loop mode
- `[resources]` — Food/poison counts
- `[agent]` — View range, energy settings
- `[runtime]` — Device configuration (CPU/GPU)
- `[training]` — Curriculum learning, PPO hyperparameters
- `[[agents]]` — Which behaviors to spawn

### Device Configuration
> [!NOTE]
> **Architecture Note**: This project originally used a "TinyCNN" (2 layers, 64 hidden units). It failed to learn spatial navigation effectively, wasting significant development time. The default architecture is now `BaseCNN` (3 layers, 512 hidden units), which provides the necessary capacity for the 11x11x6 observation space. The incredible power of Replace In Files has obsoleted the TinyCNN.

PyTorch device selection is centralized. Control it via:

```bash
# Environment variable (highest priority)
GOODHARTS_DEVICE=cpu python main.py
GOODHARTS_DEVICE=cuda:1 python -m goodharts.training.train_ppo

# Or in config.toml:
[runtime]
device = "cuda"  # Options: "cpu", "cuda", "cuda:0", "cuda:1", etc.
```

If not specified, the system auto-detects (CUDA > CPU).

### Running with Docker (GPU Support)

For AMD GPUs (ROCm):
```bash
docker compose --profile rocm up -d --build
docker compose exec -it dev-rocm bash
# Inside container:
python main.py
```

For NVIDIA GPUs (CUDA):
```bash
docker compose --profile cuda up -d --build
docker compose exec -it dev-cuda bash
python main.py
```

For CPU only:
```bash
docker compose --profile cpu up -d --build
docker compose exec -it dev-cpu bash
python main.py
```

See `docker_directions.txt` for detailed instructions and troubleshooting.

---

## Key Components

### CellType System (`configs/default_config.py`)

Cells have intrinsic properties that drive the "Goodhart trap":

| Cell | Value | Color | Interestingness | Energy Reward | Energy Penalty |
|------|-------|-------|-----------------|---------------|----------------|
| EMPTY | 0 | Dark blue | 0.0 | 0 | 0 |
| WALL | 1 | Gray | 0.0 | 0 | 0 |
| FOOD | 2 | Teal green | **1.0** | +15 | 0 |
| POISON | 3 | Coral red | **0.9** | 0 | -50 |
| PREY | 4 | Cyan | 0.3 | 0 | 0 |
| PREDATOR | 5 | Red | 1.0 | +25 | 0 |

Each CellType has a `color` property for visualization and a `channel_index` property for observation encoding.

### Behavior Registry

Behaviors are auto-discovered and registered. No manual registration needed:

```python
from goodharts.behaviors import get_behavior, list_behavior_names

# List all available behaviors
print(list_behavior_names())
# ['LearnedBehavior', 'LearnedGroundTruth', 'LearnedProxy', 
#  'LearnedProxyIllAdjusted', 'OmniscientSeeker', 'ProxySeeker']

# Get a behavior class by name
BehaviorClass = get_behavior('OmniscientSeeker')
```

### Behavior Colors

Each behavior has a `color` property for visualization:

```python
class BehaviorStrategy(ABC):
    _color: tuple[int, int, int] | None = None  # Override per-subclass
    
    @property
    def color(self) -> tuple[int, int, int]:
        if self._color is not None:
            return self._color
        return ROLE_COLORS.get(self.role, (128, 128, 128))
```

Behaviors default to role-based colors (prey=cyan, predator=red) but can override with their own.

### Learned Behaviors

Create learned behaviors using presets:

```python
from goodharts.behaviors import create_learned_behavior

# Factory function (preferred)
behavior = create_learned_behavior('ground_truth', model_path='models/my_model.pth')
behavior = create_learned_behavior('proxy_ill_adjusted')

# Available presets: 'ground_truth', 'proxy', 'proxy_ill_adjusted'
```

### Statistics & Visualization

The simulation tracks:
- **Death reasons** — distinguishes starvation (energy ran out) from poisoning
- **Energy history** — per-agent energy over time
- **Heatmaps** — where agents spend time vs. where food/poison is located
- **Suspicion score** — how often an agent chooses high-proxy cells that turn out to be poison

---

## Training

### Train Models

```bash
# Standard training (episode-based)
python -m goodharts.training.train_ppo --mode ground_truth --episodes 500

# Train all modes in parallel
python -m goodharts.training.train_ppo --mode all --episodes 500

# With live visualization
python -m goodharts.training.train_ppo --mode ground_truth --visualize

# With unified dashboard (all modes in one window)
python -m goodharts.training.train_ppo --mode all --dashboard
```

### Vectorized Training (Fast!)

Training uses vectorized environments (64 parallel by default) for high throughput:

```bash
# Fast vectorized training (~6,000+ steps/sec)
python -m goodharts.training.train_ppo --mode ground_truth --timesteps 100000

# With more parallel environments
python -m goodharts.training.train_ppo --n-envs 128 --timesteps 200000

# Train all modes in parallel
python -m goodharts.training.train_ppo --mode all --timesteps 100000
```

### Training Visualization

The `--visualize` flag opens a live dashboard showing:
- **Episode rewards and lengths** (with smoothing)
- **Policy/value losses and entropy**
- **Action probability distribution** (key diagnostic for uniform action issue)
- **Curriculum progress** (food density over time)

This helps diagnose training problems like:
- **Uniform action probabilities** (entropy too high, reward signal too weak)
- **Value loss not decreasing** (learning rate issues)
- **Entropy staying at maximum** (not learning to discriminate)

Training configuration is in `config.default.toml` under `[training]`:
```toml
[training]
initial_food = 200      # Curriculum: start sparse
final_food = 50         # Curriculum: end very sparse
curriculum_fraction = 0.7
min_food = 50           # Randomization range for robustness
max_food = 200
min_poison = 20
max_poison = 100
steps_per_episode = 500
reward_scale = 0.1      # Normalized reward scale
entropy_coef = 0.02     # Allow exploration
```

### Verify Models

```bash
# Run model verification suite
python -m goodharts.training.verification

# With more steps and verbose output
python -m goodharts.training.verification --steps 500 --verbose
```

### Run with Trained Models

```bash
# Visual demo with learned agents
python main.py --learned

# Brain view mode (visualize neural network internals)
python main.py --brain-view --agent LearnedGroundTruth --model models/ground_truth.pth
```

---

## Roadmap

### ✅ Phase 1: Better Measurement & Visualization
- [x] Track death causes (starvation vs poison)
- [x] Per-agent energy charts
- [x] Activity heatmaps
- [x] "Suspicion" metric

### ✅ Phase 2: Learned Behaviors (CNN)
- [x] **BaseCNN architecture** — dynamic channel input, configurable action output
- [x] **Training pipeline** — behavior cloning + PPO with curriculum
- [x] **One-hot observation encoding** — ground-truth vs proxy modes
- [x] **Centralized action space** — `behaviors/action_space.py`
- [x] **Temperature-based sampling** — natural exploration when uncertain
- [x] **Resource respawning** — simulation runs indefinitely
- [x] **Model verification suite** — `training/verification/`

### ✅ Phase 2.5: Code Quality Refactoring
- [x] **Vectorized Environment** — Replaced Python loops with NumPy vectorization (6000+ steps/sec)
- [x] **Behavior auto-discovery registry** — no manual registration
- [x] **CellType enhancements** — color, channel_index properties
- [x] **D1 rendering** — grid + agent overlay with behavior.color
- [x] **TOML configuration** — all hyperparameters configurable
- [x] **Visualization extraction** — main.py reduced from 529 to ~200 lines

### 🔮 Phase 3: Emergent Deception
- [ ] Multi-agent signaling dynamics
- [ ] Resource competition under scarcity
- [ ] Agents that "game" the proxy metric
- [ ] Adversarial inspector/gamer co-evolution

---

## AI Safety Connection

This simulation is a **toy model** for understanding real AI alignment failures:

1. **Specification Gaming**: The proxy-seeker optimizes exactly what we told it to (highest signal), but this doesn't align with what we actually want (survival).

2. **Information Asymmetry**: The agent lacks access to ground truth—a common scenario when we can't fully specify what we want.

3. **Emergent Failure Modes**: Future phases aim to show agents *discovering* deceptive strategies, not just failing on fixed rules.

---

## Disclosure

This project was developed almost entirely using Google's experimental Antigravity agentic IDE, with the gracious assistance of Claude 4.5 Opus and Gemini 3 Pro. I acted as an architect and systems lead, manually writing some code where relevant, but I credit Opus for the vast majority of the actual implementation and documentation writing. This project would not have been possible without Google's free Gemini Pro subscription for students and the extremely high usage limits currently available in Antigravity.

I am grateful to Google and Anthropic for making this project possible.

---

## License

See the Unlicense.
