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
├── main.py                 # Entry point: runs animated simulation
├── simulation.py           # Core simulation loop & statistics tracking
│
├── agents/
│   └── organism.py         # Agent class: movement, eating, energy, death
│
├── behaviors/
│   ├── base.py             # BehaviorStrategy abstract base class
│   ├── omniscient.py       # OmniscientSeeker: sees true cell types
│   ├── proxy_seeker.py     # ProxySeeker: only sees proxy signal
│   ├── learned.py          # LearnedBehavior: neural net controller (WIP)
│   └── brains/
│       └── tiny_cnn.py     # TinyCNN model skeleton for learned behaviors
│
├── environments/
│   ├── base.py             # Environment abstract base class
│   └── world.py            # World: grid with ground_truth + proxy_metric
│
├── training/
│   ├── train.py            # Training loop skeleton
│   └── dataset.py          # Dataset utilities for RL/supervised learning
│
├── configs/
│   └── default_config.py   # Hyperparameters, CellType definitions
│
├── Dockerfile.rocm         # AMD GPU (ROCm) environment
├── Dockerfile.cuda         # NVIDIA GPU (CUDA) environment
├── Dockerfile.cpu          # CPU-only environment
├── compose.yaml            # Docker Compose with profiles: rocm, cuda, cpu
├── docker_directions.txt   # Docker workflow instructions
│
├── TODO.txt                # Roadmap & future enhancements
└── requirements.txt        # Python dependencies
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Dependencies: `numpy`, `matplotlib`, `torch`, `torchvision`

### Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run the simulation
python main.py
```

A matplotlib window will open showing:
1. **Live Simulation Grid** — agents (cyan/magenta) navigating food (green) and poison (red)
2. **Energy Plot** — average energy over time per species
3. **Activity Heatmap** — where agents spend time
4. **Death Statistics** — starvation vs. poisoning counts

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

| Cell | Value | Interestingness | Energy Reward | Energy Penalty |
|------|-------|-----------------|---------------|----------------|
| EMPTY | 0 | 0.0 | 0 | 0 |
| WALL | 1 | 0.0 | 0 | 0 |
| FOOD | 2 | **1.0** | +15 | 0 |
| POISON | 3 | **0.9** | 0 | -50 |

The **interestingness** field populates the `proxy_grid`—both food and poison appear "interesting", but only food is actually beneficial.

### Behavior Strategy Pattern

Behaviors are modular and declare their **requirements**:

```python
class OmniscientSeeker(BehaviorStrategy):
    @property
    def requirements(self) -> list[str]:
        return ['ground_truth']  # Sees actual cell types

class ProxySeeker(BehaviorStrategy):
    @property
    def requirements(self) -> list[str]:
        return ['proxy_metric']  # Only sees interestingness signal
```

The `Organism` validates compatibility at construction time, ensuring behaviors only see what they're designed for.

### Statistics & Visualization

The simulation tracks:
- **Death reasons** — distinguishes starvation (energy ran out) from poisoning
- **Energy history** — per-agent energy over time
- **Heatmaps** — where agents spend time vs. where food/poison is located
- **Suspicion score** — how often an agent chooses high-proxy cells that turn out to be poison

---

## Hyperparameters (`configs/default_config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENERGY_START` | 50.0 | Initial energy for new agents |
| `ENERGY_MOVE_COST` | 0.1 | Base cost per unit distance moved |
| `MOVE_COST_EXPONENT` | 1.5 | Nonlinear scaling: farther moves cost more |
| `MAX_MOVE_DISTANCE` | 3 | Speed cap per step |
| `GRID_WIDTH/HEIGHT` | 100×100 | World dimensions |
| `GRID_FOOD_INIT` | 50 | Initial food items |
| `GRID_POISON_INIT` | 10 | Initial poison items |
| `AGENT_VIEW_RANGE` | 5 | Sight radius (Manhattan) |

---

## Roadmap (from `TODO.txt`)

### ✅ Phase 1: Better Measurement & Visualization
- [x] Track death causes (starvation vs poison)
- [x] Per-agent energy charts
- [x] Activity heatmaps
- [x] "Suspicion" metric

### ✅ Phase 2: Learned Behaviors (CNN)
- [x] **TinyCNN architecture** — 4-channel input (one-hot cells), 8-action output
- [x] **Training pipeline** — behavior cloning from expert demonstrations
- [x] **One-hot observation encoding** — ground-truth vs proxy modes
- [x] **Centralized action space** — `behaviors/action_space.py`
- [x] **Temperature-based sampling** — natural exploration when uncertain
- [x] **Visibility-weighted training** — 10× weight for samples with visible targets
- [x] **Resource respawning** — simulation runs indefinitely
- [x] **CLI verification tools** — `training/verify_models.py`

Train and run learned agents:
```bash
# Train both models (takes ~2-3 min with GPU)
python training/train.py --mode both --epochs 100

# Verify model fitness (headless)
python training/verify_models.py

# Run visual demo with trained CNNs
python main.py --learned
```

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

## License

See the Unlicense.
