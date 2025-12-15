"""
Training logging utilities.

Outputs structured logs (CSV and JSON) for training runs that can be
reviewed by AI agents or analyzed programmatically.

Each training run creates:
- {run_id}_episodes.csv: Per-episode metrics
- {run_id}_updates.csv: Per-PPO-update metrics  
- {run_id}_summary.json: Final summary with hyperparameters
"""
import csv
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EpisodeLog:
    """One row in the episodes CSV."""
    episode: int
    reward: float
    length: int
    food_eaten: int
    poison_eaten: int
    food_density: int = 0
    curriculum_progress: float = 0.0
    action_prob_std: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass  
class UpdateLog:
    """One row in the updates CSV."""
    update_num: int
    total_steps: int
    policy_loss: float
    value_loss: float
    entropy: float
    explained_variance: float = 0.0  # How well value function predicts returns
    action_probs: list[float] = field(default_factory=list)
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class TrainingLogger:
    """
    Structured logging for training runs.
    
    Creates CSV files for episodes and updates, plus a JSON summary.
    Designed for easy AI review of training progress.
    """
    
    def __init__(self, mode: str, output_dir: str = "logs", log_episodes: bool = True):
        """
        Initialize logger for a training run.
        
        Args:
            mode: Training mode name (e.g., 'ground_truth')
            output_dir: Directory to write logs to
            log_episodes: If True, log per-episode data (batched). If False, skip episode logging.
        """
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_episodes = log_episodes
        
        # Generate run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{mode}_{timestamp}"
        
        # File paths
        self.episodes_path = self.output_dir / f"{self.run_id}_episodes.csv"
        self.updates_path = self.output_dir / f"{self.run_id}_updates.csv"
        self.summary_path = self.output_dir / f"{self.run_id}_summary.json"
        
        # State
        self.episode_count = 0
        self.update_count = 0
        self.start_time = datetime.now()
        self.hyperparams: dict[str, Any] = {}
        self.episode_rewards: list[float] = []  # Track rewards in memory for dashboard
        
        # Batched episode logging buffer
        self._episode_buffer: list[dict] = []
        
        # Initialize CSV files with headers
        if self.log_episodes:
            self._init_csv(self.episodes_path, EpisodeLog)
        self._init_csv(self.updates_path, UpdateLog)
        
        log_mode = "batched" if self.log_episodes else "disabled"
        print(f"Logging: {self.output_dir}/{self.run_id}_*.csv (episodes: {log_mode})")
    
    def _init_csv(self, path: Path, dataclass_type):
        """Initialize a CSV file with headers from a dataclass."""
        fields = [f.name for f in dataclass_type.__dataclass_fields__.values()]
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
    
    def _append_csv(self, path: Path, row: dict):
        """Append a row to a CSV file."""
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    
    def set_hyperparams(self, **kwargs):
        """Record hyperparameters for the summary."""
        self.hyperparams.update(kwargs)
    
    def log_episode(self, episode: int, reward: float, length: int, 
                    food_eaten: int, poison_eaten: int,
                    food_density: int = 0, curriculum_progress: float = 0.0, 
                    action_prob_std: float = 0.0):
        """Log an episode completion (batched write)."""
        self.episode_count = episode
        self.episode_rewards.append(reward)
        
        if not self.log_episodes:
            return
        
        log = EpisodeLog(
            episode=episode,
            reward=reward,
            length=length,
            food_eaten=food_eaten,
            poison_eaten=poison_eaten,
            food_density=food_density,
            curriculum_progress=curriculum_progress,
            action_prob_std=action_prob_std,
        )
        
        self._episode_buffer.append(asdict(log))
        
        # Safety flush if buffer gets too large (prevents memory issues on very long runs)
        if len(self._episode_buffer) >= 1000:
            self._flush_episodes()
    
    def _flush_episodes(self):
        """Write buffered episodes to CSV."""
        if not self._episode_buffer or not self.log_episodes:
            return
        
        with open(self.episodes_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._episode_buffer[0].keys())
            writer.writerows(self._episode_buffer)
        
        self._episode_buffer.clear()
    
    def log_update(self, update_num: int, total_steps: int,
                   policy_loss: float, value_loss: float, 
                   entropy: float, action_probs: list[float],
                   explained_variance: float = 0.0):
        """Log a PPO update (also flushes buffered episodes)."""
        self.update_count = update_num
        
        # Flush episode buffer on each update
        self._flush_episodes()
        
        log = UpdateLog(
            update_num=update_num,
            total_steps=total_steps,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            explained_variance=explained_variance,
            action_probs=action_probs,
        )
        
        row = asdict(log)
        # Convert list to string for CSV
        row['action_probs'] = json.dumps(action_probs)
        self._append_csv(self.updates_path, row)
    
    def finalize(self, best_efficiency: float, final_model_path: str):
        """Write the final summary JSON."""
        # Flush any remaining episodes
        self._flush_episodes()
        
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
                "episodes": str(self.episodes_path),
                "updates": str(self.updates_path),
            }
        }
        
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary: {self.summary_path}")
        return summary


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
    Analyze a training log and return diagnostic information.
    
    This is designed for AI agents to review training runs.
    
    Returns:
        Dict with analysis results and potential issues.
    """
    import csv
    
    summary_path = Path(summary_path)
    with open(summary_path) as f:
        summary = json.load(f)
    
    # Load episodes CSV
    episodes_path = Path(summary["log_files"]["episodes"])
    episodes = []
    with open(episodes_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append({
                "episode": int(row["episode"]),
                "reward": float(row["reward"]),
                "length": int(row["length"]),
                "food_eaten": int(row["food_eaten"]),
                "action_prob_std": float(row["action_prob_std"]),
            })
    
    # Load updates CSV
    updates_path = Path(summary["log_files"]["updates"])
    updates = []
    with open(updates_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            updates.append({
                "update_num": int(row["update_num"]),
                "policy_loss": float(row["policy_loss"]),
                "value_loss": float(row["value_loss"]),
                "entropy": float(row["entropy"]),
            })
    
    # Analyze
    analysis = {
        "summary": summary,
        "diagnostics": {},
        "issues": [],
        "recommendations": [],
    }
    
    if episodes:
        # Action probability std analysis
        final_episodes = episodes[-50:] if len(episodes) >= 50 else episodes
        avg_prob_std = sum(e["action_prob_std"] for e in final_episodes) / len(final_episodes)
        
        analysis["diagnostics"]["avg_final_action_prob_std"] = avg_prob_std
        
        if avg_prob_std < 0.05:
            analysis["issues"].append("Action probabilities nearly uniform (std < 0.05)")
            analysis["recommendations"].append("Reduce entropy_coef further (try 0.0001)")
            analysis["recommendations"].append("Increase reward magnitude")
        
        # Reward trend
        if len(episodes) >= 20:
            first_20 = sum(e["reward"] for e in episodes[:20]) / 20
            last_20 = sum(e["reward"] for e in episodes[-20:]) / 20
            analysis["diagnostics"]["reward_trend"] = last_20 - first_20
            
            if last_20 < first_20:
                analysis["issues"].append("Reward decreasing over training")
                analysis["recommendations"].append("Check curriculum steepness")
        
        # Food eating
        avg_food = sum(e["food_eaten"] for e in final_episodes) / len(final_episodes)
        analysis["diagnostics"]["avg_final_food_eaten"] = avg_food
        
        if avg_food < 1.0:
            analysis["issues"].append("Agent rarely eating food in final episodes")
            analysis["recommendations"].append("Slow curriculum or increase initial_food")
    
    if updates:
        # Entropy analysis
        final_updates = updates[-20:] if len(updates) >= 20 else updates
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
    
    return analysis
