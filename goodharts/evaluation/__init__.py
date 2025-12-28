"""
Evaluation module for trained Goodhart agents.

Uses continuous survival paradigm: agents run until death, then auto-respawn.
We track death events and survival times, not artificial "episodes".

Provides:
- EvaluationConfig: Configuration dataclass for evaluation runs
- Evaluator/ModelTester: Core evaluation orchestrator
- DeathEvent: Per-death metrics dataclass (EpisodeMetrics is alias for compatibility)
- ModeAggregates: Aggregate statistics dataclass
- EvaluationDashboard: Real-time visualization (optional)
"""

from goodharts.evaluation.evaluator import (
    EvaluationConfig,
    Evaluator,
    ModelTester,
    DeathEvent,
    EpisodeMetrics,  # Backwards compatibility alias
    ModeAggregates,
)

__all__ = [
    'EvaluationConfig',
    'Evaluator',
    'ModelTester',
    'DeathEvent',
    'EpisodeMetrics',  # Backwards compatibility
    'ModeAggregates',
]
