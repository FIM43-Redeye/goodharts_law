"""
Generic receiver for GPU-computed analysis results.

This module provides a simple interface for receiving pre-computed
analysis data from the GPU and writing it to files. The key principle:
all heavy computation happens on GPU, this just handles the I/O.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


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


@dataclass
class AnalysisResult:
    """
    Container for GPU-computed analysis results.

    All fields should contain CPU-transferred data (numpy arrays or
    Python primitives). The GPU computation should be complete before
    creating this object.
    """
    mode: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Standard metrics (all pre-computed on GPU)
    reward_stats: dict[str, float] = field(default_factory=dict)
    action_stats: dict[str, Any] = field(default_factory=dict)
    episode_stats: dict[str, float] = field(default_factory=dict)

    # Custom metrics (whatever the training code computes)
    custom: dict[str, Any] = field(default_factory=dict)

    # Diagnostics and recommendations (from analyze_training_log)
    diagnostics: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class AnalysisReceiver:
    """
    Receives GPU-computed analysis and writes to files.

    Usage:
        receiver = AnalysisReceiver(output_dir="logs")

        # At end of training, after GPU computation
        result = AnalysisResult(
            mode="ground_truth",
            reward_stats={"mean": 42.5, "std": 12.3},
            action_stats={"distribution": [0.3, 0.2, ...]},
        )
        receiver.write(result)

    The receiver handles:
    - JSON serialization of numpy/torch types
    - Consistent file naming
    - Directory creation
    """

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, result: AnalysisResult) -> Path:
        """
        Write analysis result to JSON file.

        Returns the path to the written file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.mode}_{timestamp}_analysis.json"
        filepath = self.output_dir / filename

        # Convert dataclass to dict, handling numpy/torch types
        data = {
            "mode": result.mode,
            "timestamp": result.timestamp,
            "reward_stats": result.reward_stats,
            "action_stats": result.action_stats,
            "episode_stats": result.episode_stats,
            "custom": result.custom,
            "diagnostics": result.diagnostics,
            "issues": result.issues,
            "recommendations": result.recommendations,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=ExtendedJSONEncoder)

        return filepath

    def receive(
        self,
        mode: str,
        reward_stats: dict[str, float] | None = None,
        action_stats: dict[str, Any] | None = None,
        episode_stats: dict[str, float] | None = None,
        custom: dict[str, Any] | None = None,
        diagnostics: dict[str, float] | None = None,
        issues: list[str] | None = None,
        recommendations: list[str] | None = None,
    ) -> Path:
        """
        Convenience method to receive and write in one call.

        All dict/list arguments are optional. Pass whatever metrics
        your training code computes on the GPU.
        """
        result = AnalysisResult(
            mode=mode,
            reward_stats=reward_stats or {},
            action_stats=action_stats or {},
            episode_stats=episode_stats or {},
            custom=custom or {},
            diagnostics=diagnostics or {},
            issues=issues or [],
            recommendations=recommendations or [],
        )
        return self.write(result)
