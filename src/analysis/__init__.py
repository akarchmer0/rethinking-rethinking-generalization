"""Analysis tools for neural network properties."""

from .metrics import SmoothnessMetrics, GeneralizationMetrics
from .statistical_tests import (
    bootstrap_confidence_intervals,
    hypothesis_test_smoothness,
    correlation_analysis,
)
from . import visualization

__all__ = [
    'SmoothnessMetrics',
    'GeneralizationMetrics',
    'bootstrap_confidence_intervals',
    'hypothesis_test_smoothness',
    'correlation_analysis',
    'visualization',
]

