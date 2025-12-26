#!/bin/bash
# Profile training with rocprof (external profiler, minimal overhead)
#
# This attaches to the process externally without modifying the training loop.
# Much lower overhead than torch.profiler.
#
# Usage:
#   ./scripts/rocprof_training.sh           # Default: 8 updates, benchmark mode
#   ./scripts/rocprof_training.sh 16        # Custom update count
#
# Output:
#   profile_trace/rocprof_results.json  - Chrome trace (open in chrome://tracing)
#   profile_trace/rocprof_results.csv   - Kernel timing data

set -e

UPDATES=${1:-8}
OUTPUT_DIR="./profile_trace"
OUTPUT_PREFIX="${OUTPUT_DIR}/rocprof_results"

# Calculate timesteps: 192 envs * 128 steps * N updates
TIMESTEPS=$((192 * 128 * UPDATES))

echo "========================================"
echo "ROCm Profiler - Training Trace"
echo "========================================"
echo "Updates: $UPDATES"
echo "Timesteps: $TIMESTEPS"
echo "Output: ${OUTPUT_PREFIX}.*"
echo ""

mkdir -p "$OUTPUT_DIR"

# Run with rocprof
# --hip-trace: Trace HIP API calls (kernel launches, memory ops)
# --roctx-trace: Trace ROCm annotations (if any)
# -o: Output prefix
rocprof --hip-trace \
    -o "$OUTPUT_PREFIX" \
    python -m goodharts.training.train_ppo \
        --mode ground_truth \
        --updates "$UPDATES" \
        --benchmark \
        --no-profile

echo ""
echo "========================================"
echo "Profile complete!"
echo "========================================"
echo ""
echo "Files created:"
ls -la ${OUTPUT_PREFIX}* 2>/dev/null || echo "  (no output files found)"
echo ""
echo "To view trace:"
echo "  1. Open chrome://tracing"
echo "  2. Load ${OUTPUT_PREFIX}.json"
