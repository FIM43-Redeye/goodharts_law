"""
Process-isolated CSV writer for training logs.

Completely escapes the GIL by running file I/O in a separate process.
The main training process just does queue.put() which releases the GIL
almost instantly (it's implemented in C).

Episode stats are aggregated on GPU (sum/min/max) and only 5 floats are
transferred per update, eliminating the serialization overhead that caused
jitter with individual episode logging.
"""
import csv
import json
import multiprocessing as mp
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .train_log import UpdateLog, ExtendedJSONEncoder


def _writer_process(queue: mp.Queue, updates_path: Path, update_fields: list[str]):
    """
    Worker process that handles all file I/O.

    Runs in a completely separate process with its own GIL.
    Can take as long as it wants - zero impact on training.
    """
    updates_file = open(updates_path, 'a', newline='')
    updates_writer = csv.DictWriter(updates_file, fieldnames=update_fields)

    try:
        while True:
            msg = queue.get()

            if msg is None:
                # Shutdown signal
                break

            msg_type = msg[0]
            if msg_type == 'update':
                # msg = ('update', row_dict)
                _, row = msg
                updates_writer.writerow(row)

    finally:
        updates_file.close()


class ProcessLogger:
    """
    Process-isolated training logger.

    All file I/O happens in a separate process, completely escaping the GIL.
    The main process just does queue.put() which is nearly instantaneous.

    Episode data is aggregated on GPU (sum/min/max computed in CUDA kernels)
    and only 5 floats are passed to the logger per update.
    """

    @staticmethod
    def archive_existing_logs(output_dir: str = "logs", mode: str | None = None):
        """Move existing log files to logs/previous/."""
        from .train_log import TrainingLogger
        TrainingLogger.archive_existing_logs(output_dir, mode)

    def __init__(self, mode: str, output_dir: str = "logs", log_episodes: bool = True):
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Archive existing logs
        self.archive_existing_logs(str(self.output_dir), mode=mode)

        # Generate run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{mode}_{timestamp}"

        # File paths (only updates.csv now - episode stats are aggregated in updates)
        self.updates_path = self.output_dir / f"{self.run_id}_updates.csv"
        self.summary_path = self.output_dir / f"{self.run_id}_summary.json"

        # State tracking (stays in main process for dashboard)
        self.episode_count = 0
        self.update_count = 0
        self.start_time = datetime.now()
        self.hyperparams: dict = {}
        self.episode_rewards: list[float] = []

        # Get field names from dataclass
        self._update_fields = [f.name for f in UpdateLog.__dataclass_fields__.values()]

        # Write CSV headers
        with open(self.updates_path, 'w', newline='') as f:
            csv.writer(f).writerow(self._update_fields)

        # Process communication
        self._queue: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None

        print(f"Logging: {self.output_dir}/{self.run_id}_updates.csv (GPU-aggregated stats)")

    def start(self):
        """Start the writer process. Call before logging."""
        self._queue = mp.Queue()
        self._process = mp.Process(
            target=_writer_process,
            args=(self._queue, self.updates_path, self._update_fields),
            daemon=True,
            name=f"LogWriter-{self.mode}",
        )
        self._process.start()

    def set_hyperparams(self, **kwargs):
        """Record hyperparameters for the summary."""
        self.hyperparams.update(kwargs)

    def log_update(
        self,
        update_num: int,
        total_steps: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        action_probs: list[float],
        explained_variance: float = 0.0,
        # Episode summary stats (aggregated on GPU - no serialization overhead)
        episodes_count: int = 0,
        reward_mean: float = 0.0,
        reward_min: float = 0.0,
        reward_max: float = 0.0,
        food_mean: float = 0.0,
        poison_mean: float = 0.0,
        # Derived curriculum-invariant metrics
        food_ratio: float = 0.5,
        reward_per_consumed: float = 0.0,
    ):
        """Queue update for async write. Nearly instant."""
        if self._queue is None:
            return

        self.update_count = update_num
        self.episode_count += episodes_count

        # Build row dict with episode summary stats
        log = UpdateLog(
            update_num=update_num,
            total_steps=total_steps,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            explained_variance=explained_variance,
            episodes_count=episodes_count,
            reward_mean=reward_mean,
            reward_min=reward_min,
            reward_max=reward_max,
            food_mean=food_mean,
            poison_mean=poison_mean,
            food_ratio=food_ratio,
            reward_per_consumed=reward_per_consumed,
            action_probs=action_probs,
        )
        row = asdict(log)
        row['action_probs'] = json.dumps(action_probs, cls=ExtendedJSONEncoder)

        # Queue for process - releases GIL almost instantly
        self._queue.put(('update', row))

    def finalize(self, best_efficiency: float, final_model_path: str) -> dict:
        """Shutdown writer process and write summary JSON."""
        # Signal process to shutdown
        if self._queue is not None:
            self._queue.put(None)
        if self._process is not None:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()

        # Write summary (in main process, happens once at end)
        duration = (datetime.now() - self.start_time).total_seconds()

        summary = {
            "run_id": self.run_id,
            "mode": self.mode,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": duration,
            "total_episodes": self.episode_count,
            "total_updates": self.update_count,
            "best_efficiency": best_efficiency,
            "final_model_path": final_model_path,
            "hyperparameters": self.hyperparams,
            "log_files": {
                "updates": str(self.updates_path),
            }
        }

        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=ExtendedJSONEncoder)

        print(f"Training summary: {self.summary_path}")
        return summary

    def close(self):
        """Alias for finalize() compatibility."""
        pass
