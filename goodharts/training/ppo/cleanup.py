"""
Resource cleanup for PPO training.

Provides safe cleanup of GPU memory, background threads, and file handles.
All cleanup operations are exception-tolerant to ensure resources are released
even if some operations fail.

Usage:
    from .cleanup import cleanup_training_resources

    cleanup_training_resources(
        gpu_monitor=trainer.gpu_monitor,
        bookkeeper=trainer.bookkeeper,
        async_logger=trainer.async_logger,
        ...
    )
"""
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def cleanup_training_resources(
    gpu_monitor=None,  # GPUMonitor - avoid circular import
    bookkeeper=None,  # BackgroundBookkeeper - avoid circular import
    async_logger=None,  # AsyncLogger - avoid circular import
    tb_writer=None,  # SummaryWriter
    vec_env=None,  # TorchVecEnv
    policy: Optional[nn.Module] = None,
    value_head: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Release all training resources safely.

    All cleanup operations are wrapped in try/except to ensure resources
    are released even if some operations fail. Call this after training
    completes or on abort.

    Args:
        gpu_monitor: GPU monitoring thread (stopped first)
        bookkeeper: Background bookkeeper thread (stopped before async_logger)
        async_logger: Async logging thread (flushed and stopped)
        tb_writer: TensorBoard writer (closed to flush data)
        vec_env: Vectorized environment (reference released)
        policy: Policy network (reference released)
        value_head: Value head network (reference released)
        optimizer: Optimizer (reference released)
        device: Device for GPU cache clearing
    """
    # Stop GPU monitor first
    if gpu_monitor is not None:
        try:
            gpu_monitor.stop()
        except Exception as e:
            logger.debug(f"Cleanup: gpu_monitor.stop() failed: {e}")

    # Stop bookkeeper before async_logger (bookkeeper submits to async_logger)
    if bookkeeper is not None:
        try:
            bookkeeper.stop()
        except Exception as e:
            logger.debug(f"Cleanup: bookkeeper.stop() failed: {e}")

    # Shutdown async logger (flushes any pending logs)
    if async_logger is not None:
        try:
            async_logger.shutdown(timeout=2.0)
        except Exception as e:
            logger.debug(f"Cleanup: async_logger.shutdown() failed: {e}")

    # Close TensorBoard writer
    if tb_writer is not None:
        try:
            tb_writer.close()
        except Exception as e:
            logger.debug(f"Cleanup: tb_writer.close() failed: {e}")

    # Note: We don't delete passed-in references here since Python's
    # reference semantics mean the caller's variables would still hold
    # references. The caller is responsible for setting their references
    # to None after calling this function.

    # Clear GPU cache
    if device is not None and device.type == 'cuda':
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"Cleanup: torch.cuda.empty_cache() failed: {e}")
