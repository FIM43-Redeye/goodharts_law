#!/bin/bash
# Profile training with rocprofv3 (ROCm's modern profiler)
#
# This traces kernel dispatches, HIP API calls, and memory operations
# with minimal overhead compared to torch.profiler.
#
# Usage:
#   ./scripts/rocprof_training.sh              # Default: 4 updates, full trace
#   ./scripts/rocprof_training.sh 8            # Custom update count
#   ./scripts/rocprof_training.sh 4 summary    # Quick summary (no trace files)
#   ./scripts/rocprof_training.sh 4 stats      # Stats with trace files
#
# Output (full/stats mode):
#   profile_trace/<pid>/               - Output directory
#     kernel_trace.csv                 - Kernel timing data
#     hip_api_trace.csv                - HIP API call timings
#     memory_copy_trace.csv            - Memory transfer timings
#     results.json                     - JSON trace (perfetto format)
#
# Viewing:
#   - JSON traces: https://ui.perfetto.dev (drag and drop)
#   - Or use: rocprofv3 --help for otf2 format (for Vampir, etc.)

set -e

UPDATES=${1:-4}
MODE=${2:-full}  # full, summary, or stats
OUTPUT_DIR="./profile_trace"

# Calculate timesteps: 192 envs * 128 steps * N updates
TIMESTEPS=$((192 * 128 * UPDATES))

echo "========================================"
echo "ROCm Profiler v3 - Training Trace"
echo "========================================"
echo "Updates: $UPDATES"
echo "Timesteps: $TIMESTEPS"
echo "Mode: $MODE"
echo ""

mkdir -p "$OUTPUT_DIR"

# Common training command
TRAIN_CMD="python -m goodharts.training.train_ppo \
    --mode ground_truth \
    --updates $UPDATES \
    --benchmark \
    --no-profile"

case "$MODE" in
    summary)
        # Quick summary - no trace files, just aggregate stats to stderr
        echo "Running with summary output (no trace files)..."
        echo ""
        rocprofv3 \
            --runtime-trace \
            --stats \
            --summary \
            --summary-units msec \
            -- $TRAIN_CMD
        ;;
    stats)
        # Stats mode - trace files + statistics
        echo "Running with stats (trace files + statistics)..."
        echo "Output: $OUTPUT_DIR/"
        echo ""
        rocprofv3 \
            --runtime-trace \
            --stats \
            --summary \
            --summary-units msec \
            -d "$OUTPUT_DIR" \
            -f csv -f json \
            -- $TRAIN_CMD
        ;;
    full|*)
        # Full trace - all details, no summary clutter
        echo "Running full trace..."
        echo "Output: $OUTPUT_DIR/"
        echo ""
        rocprofv3 \
            --kernel-trace \
            --hip-runtime-trace \
            --memory-copy-trace \
            -d "$OUTPUT_DIR" \
            -f csv -f json \
            -- $TRAIN_CMD
        ;;
esac

echo ""
echo "========================================"
echo "Profile complete!"
echo "========================================"
echo ""

# Show output files (if any were created)
if [ "$MODE" != "summary" ]; then
    echo "Files created:"
    find "$OUTPUT_DIR" -type f -newer /proc/$$/fd/0 2>/dev/null | head -20 || echo "  (check $OUTPUT_DIR/)"
    echo ""
    echo "To view JSON trace:"
    echo "  1. Open https://ui.perfetto.dev"
    echo "  2. Drag and drop the .json file"
fi
