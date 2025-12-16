"""
Goodhart's Law Simulation Package.

Demonstrates AI alignment failures through proxy optimization in a 2D grid world.
Agents trained on proxy signals (interestingness) inevitably make mistakes that
ground-truth agents avoid - a visceral demonstration of "when a measure becomes
a target, it ceases to be a good measure."

Key Modules
-----------
simulation
    Visual demo orchestration with shared-grid VecEnv.
modes
    Training mode definitions (ObservationSpec, ModeSpec, RewardComputer).
behaviors
    Agent decision strategies (hardcoded heuristics and learned CNNs).
environments
    VecEnv for high-performance parallel simulation.
training
    PPO training pipeline with live dashboard and structured logging.
configs
    CellType registry and runtime configuration.
utils
    Device selection, logging, and visualization utilities.

Quick Start
-----------
Visual demo::

    python main.py --learned

Training::

    python -m goodharts.training.train_ppo --mode ground_truth --timesteps 100000

See README.md for full documentation.
"""
