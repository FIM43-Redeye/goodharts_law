"""
Training logging utilities.

Outputs structured logs (CSV and JSON) for training runs that can be
reviewed by AI agents or analyzed programmatically.

DESIGN PHILOSOPHY: GPU-First Analysis
=====================================
All per-step/per-episode statistics should be computed ON THE GPU using
PyTorch's parallel reduction operations (sum, mean, min, max, std, etc.).
Only the final aggregated results are transferred to CPU for logging.

This eliminates:
- Serialization overhead for large arrays
- GIL contention during logging
- Memory copies that stall training

See goodharts/training/analysis/ for GPU analysis utilities.
"""
import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class UpdateLog:
    """
    One row in the updates CSV.

    Episode statistics are pre-aggregated on GPU (sum/min/max computed in
    CUDA kernels) and only 5 floats are transferred per update.
    """
    update_num: int
    total_steps: int
    policy_loss: float
    value_loss: float
    entropy: float
    explained_variance: float = 0.0
    # Episode summary stats (aggregated on GPU to avoid serialization overhead)
    episodes_count: int = 0
    reward_mean: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0
    food_mean: float = 0.0
    poison_mean: float = 0.0
    action_probs: list[float] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ExtendedJSONEncoder(json.JSONEncoder):
    """JSON Encoder that handles NumPy and PyTorch types."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass

        try:
            import torch
            if isinstance(obj, torch.Tensor):
                if obj.numel() == 1:
                    return obj.item()
                return obj.tolist()
        except ImportError:
            pass

        return super().default(obj)


class TrainingLogger:
    """
    Structured logging for training runs.

    NOTE: For new code, prefer ProcessLogger which runs file I/O in a
    separate process, completely escaping the GIL.

    This class is kept for backward compatibility and simpler use cases.
    """

    @staticmethod
    def archive_existing_logs(output_dir: str = "logs", mode: str | None = None):
        """Move existing log files to logs/previous/ to keep current run clean."""
        logs_dir = Path(output_dir)
        if not logs_dir.exists():
            return

        if mode:
            log_files = list(logs_dir.glob(f"{mode}_*_updates.csv")) + \
                        list(logs_dir.glob(f"{mode}_*_summary.json")) + \
                        list(logs_dir.glob(f"{mode}_*_analysis.json"))
        else:
            log_files = list(logs_dir.glob("*_updates.csv")) + \
                        list(logs_dir.glob("*_summary.json")) + \
                        list(logs_dir.glob("*_analysis.json"))

        if not log_files:
            return

        previous_dir = logs_dir / "previous"
        previous_dir.mkdir(exist_ok=True)

        moved = 0
        for f in log_files:
            dest = previous_dir / f.name
            f.rename(dest)
            moved += 1

        if moved > 0:
            mode_str = f" ({mode})" if mode else ""
            print(f"[Logs] Archived {moved} previous{mode_str} log files to {previous_dir}/")


def get_latest_log(mode: str, output_dir: str = "logs") -> Path | None:
    """Find the most recent log file for a given mode."""
    logs_dir = Path(output_dir)
    if not logs_dir.exists():
        return None

    pattern = f"{mode}_*_summary.json"
    files = sorted(logs_dir.glob(pattern), reverse=True)
    return files[0] if files else None


def analyze_training_log(summary_path: str | Path) -> dict:
    """
    Analyze a training log from per-update data.

    Uses the GPU-aggregated episode statistics stored in updates.csv
    rather than individual episode logs.

    Returns:
        Dict with analysis results and potential issues.
    """
    summary_path = Path(summary_path)
    with open(summary_path) as f:
        summary = json.load(f)

    # Load updates CSV (contains GPU-aggregated episode stats)
    updates_path = Path(summary["log_files"]["updates"])
    updates = []
    with open(updates_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            updates.append({
                "update_num": int(row["update_num"]),
                "total_steps": int(row["total_steps"]),
                "policy_loss": float(row["policy_loss"]),
                "value_loss": float(row["value_loss"]),
                "entropy": float(row["entropy"]),
                "explained_variance": float(row["explained_variance"]),
                "episodes_count": int(row["episodes_count"]),
                "reward_mean": float(row["reward_mean"]),
                "reward_min": float(row["reward_min"]),
                "reward_max": float(row["reward_max"]),
                "food_mean": float(row["food_mean"]),
                "poison_mean": float(row["poison_mean"]),
            })

    analysis = {
        "summary": summary,
        "diagnostics": {},
        "issues": [],
        "recommendations": [],
    }

    if not updates:
        return analysis

    # Analyze final updates (last 20% or last 10, whichever is larger)
    n_final = max(10, len(updates) // 5)
    final_updates = updates[-n_final:]

    # Filter to updates that have episode data
    updates_with_episodes = [u for u in final_updates if u["episodes_count"] > 0]

    # Entropy analysis
    avg_entropy = sum(u["entropy"] for u in final_updates) / len(final_updates)
    max_entropy = 2.079  # ln(8) for 8 actions
    analysis["diagnostics"]["avg_final_entropy"] = avg_entropy
    analysis["diagnostics"]["entropy_ratio"] = avg_entropy / max_entropy

    if avg_entropy / max_entropy > 0.9:
        analysis["issues"].append("Entropy near maximum (policy nearly uniform)")
        analysis["recommendations"].append("Reduce entropy_coef to let policy specialize")

    # Value loss
    avg_value_loss = sum(u["value_loss"] for u in final_updates) / len(final_updates)
    analysis["diagnostics"]["avg_final_value_loss"] = avg_value_loss

    # Explained variance
    avg_explained_var = sum(u["explained_variance"] for u in final_updates) / len(final_updates)
    analysis["diagnostics"]["avg_final_explained_variance"] = avg_explained_var

    if avg_explained_var < 0.1:
        analysis["issues"].append("Low explained variance (value function not learning)")
        analysis["recommendations"].append("Check reward scale or increase training time")

    # Episode reward analysis (from GPU-aggregated stats)
    if updates_with_episodes:
        # Weighted average by episode count
        total_episodes = sum(u["episodes_count"] for u in updates_with_episodes)
        weighted_reward = sum(u["reward_mean"] * u["episodes_count"] for u in updates_with_episodes)
        avg_reward = weighted_reward / total_episodes if total_episodes > 0 else 0

        best_reward = max(u["reward_max"] for u in updates_with_episodes)
        worst_reward = min(u["reward_min"] for u in updates_with_episodes)

        analysis["diagnostics"]["avg_final_reward"] = avg_reward
        analysis["diagnostics"]["best_reward_seen"] = best_reward
        analysis["diagnostics"]["worst_reward_seen"] = worst_reward

        # Food/poison analysis
        weighted_food = sum(u["food_mean"] * u["episodes_count"] for u in updates_with_episodes)
        weighted_poison = sum(u["poison_mean"] * u["episodes_count"] for u in updates_with_episodes)
        avg_food = weighted_food / total_episodes if total_episodes > 0 else 0
        avg_poison = weighted_poison / total_episodes if total_episodes > 0 else 0

        analysis["diagnostics"]["avg_final_food_eaten"] = avg_food
        analysis["diagnostics"]["avg_final_poison_eaten"] = avg_poison

        if avg_food < 1.0:
            analysis["issues"].append("Agent rarely eating food")
            analysis["recommendations"].append("Check reward shaping or curriculum")

        if avg_poison > avg_food:
            analysis["issues"].append("Agent eating more poison than food")
            analysis["recommendations"].append("Check observation space or reward signal")

    # Reward trend (compare first vs last updates with episodes)
    first_with_eps = [u for u in updates[:n_final] if u["episodes_count"] > 0]
    if first_with_eps and updates_with_episodes:
        first_reward = sum(u["reward_mean"] for u in first_with_eps) / len(first_with_eps)
        last_reward = sum(u["reward_mean"] for u in updates_with_episodes) / len(updates_with_episodes)
        analysis["diagnostics"]["reward_trend"] = last_reward - first_reward

        if last_reward < first_reward:
            analysis["issues"].append("Reward decreasing over training")
            analysis["recommendations"].append("Check for reward hacking or curriculum issues")

    return analysis
