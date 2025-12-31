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

Multi-run support:
- MultiRunEvaluator: Run evaluation N times with different seeds
- RunResult: Per-run results dataclass
- MultiRunAggregates: Cross-run statistics with confidence intervals
- MultiRunComparison: Statistical comparison between modes
"""

from goodharts.evaluation.evaluator import (
    EvaluationConfig,
    Evaluator,
    ModelTester,
    DeathEvent,
    EpisodeMetrics,  # Backwards compatibility alias
    ModeAggregates,
)

from goodharts.evaluation.multi_run import (
    RunResult,
    MultiRunAggregates,
    MultiRunComparison,
    MultiRunEvaluator,
    aggregate_runs,
    generate_seeds,
    compute_confidence_interval,
    compute_pooled_std,
)

from goodharts.evaluation.cuda_graph_evaluator import (
    GraphConfig,
    GraphMetrics,
    CUDAGraphEvaluator,
    MultiModelGraphEvaluator,
    compare_graph_vs_standard,
)

__all__ = [
    # Core evaluation
    'EvaluationConfig',
    'Evaluator',
    'ModelTester',
    'DeathEvent',
    'EpisodeMetrics',  # Backwards compatibility
    'ModeAggregates',
    # Multi-run support
    'RunResult',
    'MultiRunAggregates',
    'MultiRunComparison',
    'MultiRunEvaluator',
    'aggregate_runs',
    'generate_seeds',
    'compute_confidence_interval',
    'compute_pooled_std',
    # CUDA graph acceleration
    'GraphConfig',
    'GraphMetrics',
    'CUDAGraphEvaluator',
    'MultiModelGraphEvaluator',
    'compare_graph_vs_standard',
]
