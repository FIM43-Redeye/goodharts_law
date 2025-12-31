#!/bin/bash
# Comprehensive Goodhart's Law evaluation with statistical rigor.
#
# Uses evaluation settings from config.toml/config.default.toml:
#   - n_envs: 8192 (parallel environments)
#   - steps_per_env: 4096 (steps per environment)
#   - runs: 3 (evaluation runs per mode)
#   - base_seed: 42 (starting seed)
#
# Total steps per run: 8192 * 4096 = 33.5M steps
#
# Prerequisites:
#   - Trained models in models/ directory
#
# Usage:
#   ./scripts/run_full_evaluation.sh              # Use config defaults
#   ./scripts/run_full_evaluation.sh --full-report  # Include markdown report

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Check for trained models
if ! ls models/ppo_*.pth 1>/dev/null 2>&1; then
    echo "No trained models found in models/"
    echo "Run training first: python -m goodharts.training.train_ppo --mode all"
    exit 1
fi

echo "========================================"
echo "Goodhart's Law - Full Evaluation"
echo "========================================"
echo "Using config defaults (see [evaluation] in config.default.toml)"
echo ""

# Run evaluation with config defaults, pass through any CLI args
python scripts/evaluate.py --mode all -s "$@"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "========================================"
