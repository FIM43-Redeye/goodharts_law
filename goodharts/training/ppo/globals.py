"""
Global training state for PPO.

Provides thread-safe abort control and warmup state tracking across
sequential/parallel training runs.
"""
import threading

# Lock to serialize torch.compile() calls across threads.
# Dynamo has global state that is not thread-safe during compilation.
# This only affects startup; compiled models run in parallel fine.
_COMPILE_LOCK = threading.Lock()

# Global warmup state - shared across sequential/parallel training runs
# Once warmup is done for one mode, subsequent modes skip it
_WARMUP_LOCK = threading.Lock()
_WARMUP_DONE = False

# Global abort flag - checked by all trainers to enable coordinated shutdown
_ABORT_LOCK = threading.Lock()
_ABORT_REQUESTED = False


def request_abort():
    """Signal all trainers to abort gracefully."""
    global _ABORT_REQUESTED
    with _ABORT_LOCK:
        _ABORT_REQUESTED = True


def clear_abort():
    """Clear the abort flag (call before starting new training)."""
    global _ABORT_REQUESTED
    with _ABORT_LOCK:
        _ABORT_REQUESTED = False


def is_abort_requested() -> bool:
    """Check if abort has been requested."""
    with _ABORT_LOCK:
        return _ABORT_REQUESTED


def mark_warmup_done():
    """Mark warmup as completed (called after first trainer warms up)."""
    global _WARMUP_DONE
    with _WARMUP_LOCK:
        _WARMUP_DONE = True


def check_warmup_done() -> bool:
    """Check if warmup has been completed by any trainer."""
    with _WARMUP_LOCK:
        return _WARMUP_DONE


def reset_training_state():
    """
    Reset all global training state.

    Call this after an aborted run to ensure clean state for the next run.
    This is important because globals persist across runs in the same process.
    """
    global _WARMUP_DONE, _ABORT_REQUESTED
    with _WARMUP_LOCK:
        _WARMUP_DONE = False
    with _ABORT_LOCK:
        _ABORT_REQUESTED = False


def get_compile_lock() -> threading.Lock:
    """Get the compile lock for serializing torch.compile() calls."""
    return _COMPILE_LOCK


def get_warmup_lock() -> threading.Lock:
    """Get the warmup lock for thread-safe warmup state access."""
    return _WARMUP_LOCK
