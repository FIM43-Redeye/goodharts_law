"""
Async logging for PPO training.

Offloads GPU syncs and I/O to a background thread to avoid stalling training.
The training loop queues GPU tensors directly; the background thread handles
.item() calls, file writes, console prints, and dashboard updates.
"""
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Any

import torch


@dataclass
class LogPayload:
    """Data queued for async logging."""
    # GPU tensors (will be synced in background thread)
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    explained_var: torch.Tensor
    action_probs_tensor: torch.Tensor  # Last logits for action probs
    
    # Already-synced data (cheap to copy)
    update_count: int
    total_steps: int
    best_reward: float
    mode: str
    
    # Episode stats (already on CPU)
    episode_stats: Optional[dict]  # {'reward': float, 'food': int, 'poison': int}
    
    # Profiler summary (string, already computed)
    profiler_summary: str
    
    # Timing info
    sps: float


class AsyncLogger:
    """
    Thread-safe async logging for PPO training.
    
    Usage:
        logger = AsyncLogger(trainer_logger, tb_writer, dashboard, mode)
        logger.start()
        
        # In training loop:
        logger.log_update(payload)
        
        # At end:
        logger.shutdown()
    """
    
    def __init__(
        self,
        trainer_logger,  # TrainingLogger instance
        tb_writer,       # TensorBoard SummaryWriter or None
        dashboard,       # Training dashboard or None
        mode: str,
    ):
        self.trainer_logger = trainer_logger
        self.tb_writer = tb_writer
        self.dashboard = dashboard
        self.mode = mode
        
        self._queue: queue.Queue[Optional[LogPayload]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self):
        """Start the background logging thread."""
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="AsyncLogger")
        self._thread.start()
    
    def log_update(self, payload: LogPayload):
        """Queue metrics for async logging. Non-blocking."""
        self._queue.put(payload)
    
    def shutdown(self, timeout: float = 5.0):
        """Stop the background thread and flush remaining logs."""
        self._running = False
        self._queue.put(None)  # Sentinel to wake up thread
        if self._thread is not None:
            self._thread.join(timeout=timeout)
    
    def _worker(self):
        """Background thread: processes log queue."""
        while self._running or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if payload is None:
                break
            
            self._process_payload(payload)
    
    def _process_payload(self, p: LogPayload):
        """Do all the sync and I/O work for one update."""
        # GPU -> CPU sync happens here (off critical path)
        policy_loss = p.policy_loss.item()
        value_loss = p.value_loss.item()
        entropy = p.entropy.item()
        explained_var = p.explained_var.item()
        
        # Action probs: softmax and transfer to CPU
        import torch.nn.functional as F
        action_probs = F.softmax(p.action_probs_tensor, dim=1)[0].cpu().numpy().tolist()
        
        # Console output
        print(f"   [{p.mode}] Step {p.total_steps:,}: {p.sps:,.0f} sps | Best R={p.best_reward:.0f} | Ent={entropy:.3f} | ValL={value_loss:.4f} | ExpV={explained_var:.4f}")
        print(f"   [Profile] {p.profiler_summary}")
        
        # File logging
        if self.trainer_logger:
            self.trainer_logger.log_update(
                update_num=p.update_count,
                total_steps=p.total_steps,
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy,
                explained_variance=explained_var,
                action_probs=action_probs
            )
        
        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar('loss/policy', policy_loss, p.total_steps)
            self.tb_writer.add_scalar('loss/value', value_loss, p.total_steps)
            self.tb_writer.add_scalar('metrics/entropy', entropy, p.total_steps)
            self.tb_writer.add_scalar('metrics/explained_variance', explained_var, p.total_steps)
            if p.episode_stats:
                self.tb_writer.add_scalar('reward/episode', p.episode_stats['reward'], p.total_steps)
            self.tb_writer.flush()
        
        # Dashboard update
        if self.dashboard:
            ep_stats = p.episode_stats
            payload = {
                'ppo': (policy_loss, value_loss, entropy, action_probs, explained_var),
                'episodes': ep_stats,
                'steps': p.total_steps
            }
            self.dashboard.update(p.mode, 'update', payload)
