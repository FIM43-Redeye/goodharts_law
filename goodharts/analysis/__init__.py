"""
Analysis and evaluation tools for Goodhart's Law experiments.

See README.md in this directory for the GPU-first analysis philosophy.
"""

from .receiver import AnalysisReceiver, AnalysisResult

__all__ = ["AnalysisReceiver", "AnalysisResult"]
