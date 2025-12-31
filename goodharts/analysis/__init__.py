"""
Analysis and evaluation tools for Goodhart's Law experiments.

See README.md in this directory for the GPU-first analysis philosophy.

Provides:
- AnalysisReceiver: Base receiver for real-time analysis
- StatisticalComparison: Dataclass for publication-quality stats
- PowerAnalysisResult: Sample size guidance
- ReportGenerator: Unified markdown report generation
"""

from .receiver import AnalysisReceiver, AnalysisResult
from .stats_helpers import (
    StatisticalComparison,
    compute_comparison,
    compute_confidence_interval,
    compute_effect_sizes,
    compute_goodhart_failure_index,
    format_p_value,
    interpret_effect_size,
)
from .power import (
    PowerAnalysisResult,
    power_analysis,
    required_sample_size,
    achieved_power,
    minimum_detectable_effect,
    print_power_table,
)
from .report import ReportConfig, ReportGenerator

__all__ = [
    # Receiver
    "AnalysisReceiver",
    "AnalysisResult",
    # Statistics
    "StatisticalComparison",
    "compute_comparison",
    "compute_confidence_interval",
    "compute_effect_sizes",
    "compute_goodhart_failure_index",
    "format_p_value",
    "interpret_effect_size",
    # Power analysis
    "PowerAnalysisResult",
    "power_analysis",
    "required_sample_size",
    "achieved_power",
    "minimum_detectable_effect",
    "print_power_table",
    # Report generation
    "ReportConfig",
    "ReportGenerator",
]
