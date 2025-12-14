# Phase 2 Training Pipeline

## How It Works

This document covers the CNN training pipeline for learned agent behaviors.

## Quick Start

```bash
# Train all models
python training/train.py --mode all --epochs 100 --collection-steps 2000

# Verify model fitness (headless)
python training/verify_models.py

# Run visual demo
python main.py --learned
```

## Architecture

### Observation System (One-Hot Encoding)
Agents see a **4-channel** view of their surroundings:

| Channel | Ground-Truth Mode | Proxy Mode |
|---------|-------------------|------------|
| 0 | is_empty (0/1) | interestingness |
| 1 | is_wall (0/1) | interestingness |
| 2 | is_food (0/1) | interestingness |
| 3 | is_poison (0/1) | interestingness |

In proxy mode, all channels contain the same interestingness values - the agent **cannot distinguish food from poison**.

### CNN Architecture (BaseCNN)
```
Input: (4, 11, 11) - 4 channels, 11×11 view
  ↓
Conv2D(4→16, 3×3, padding=1) + ReLU
  ↓
Conv2D(16→32, 3×3, padding=1) + ReLU
  ↓
Flatten: 32 × 11 × 11 = 3872 features
  ↓
Linear(3872→32) + ReLU
  ↓
Linear(32→8) - 8 directional actions
  ↓
Softmax (temperature-scaled)
```

### Action Space (Centralized in `behaviors/action_space.py`)
```
Index  Action    Direction
0      (-1, -1)  ↖ Up-Left
1      (-1,  0)  ← Left
2      (-1,  1)  ↙ Down-Left
3      ( 0, -1)  ↑ Up
4      ( 0,  1)  ↓ Down
5      ( 1, -1)  ↗ Up-Right
6      ( 1,  0)  → Right
7      ( 1,  1)  ↘ Down-Right
```

## Training Details

### Data Collection
- Expert demonstrations from `OmniscientSeeker` (ground-truth) or `ProxySeeker` (proxy)
- Training uses **200 food / 40 poison** for better visibility signal
- Samples weighted by:
  - Reward (positive = ate food)
  - Visibility (10× weight when food/poison visible)

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-3 (Adam) |
| Batch Size | 32 |
| Epochs | 100 |
| Collection Steps | 2000 |
| Visibility Weight Multiplier | 10.0 |

### Inference
- **Temperature=0.5** for production (sharper decisions)
- Uses `torch.multinomial` sampling from softmax (not argmax)
- Provides natural exploration when uncertain

## GPU Usage

Check if CUDA is being used:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")
```

For AMD GPUs (RX 7700S), you need ROCm:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
```

Current training uses small batch sizes and models - may actually be **CPU-bound** for this size. The model has only ~130K parameters.

## Files Modified

| File | Purpose |
|------|---------|
| `goodharts/behaviors/action_space.py` | Single source of truth for action indices |
| `goodharts/behaviors/learned.py` | LearnedBehavior with temperature sampling |
| `goodharts/behaviors/brains/base_cnn.py` | CNN architecture |
| `goodharts/agents/organism.py` | One-hot observation encoding |
| `goodharts/training/train.py` | Behavior cloning training loop |
| `goodharts/training/train_rl.py` | Reinforcement learning training |
| `goodharts/training/train_ppo.py` | PPO algorithm implementation |
| `goodharts/training/dataset.py` | SimulationDataset with combined weights |
| `goodharts/training/collect.py` | Expert demonstration collection |
| `goodharts/training/visualize_saliency.py` | Neural network interpretability |

