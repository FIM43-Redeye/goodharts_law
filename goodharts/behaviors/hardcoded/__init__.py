"""
Hardcoded baseline behaviors for comparison.

These are hand-coded behaviors that serve as baselines for evaluating
learned behaviors. They don't use neural networks.
"""
from .omniscient import OmniscientSeeker
from .proxy_seeker import ProxySeeker

__all__ = ['OmniscientSeeker', 'ProxySeeker']
