"""Read-only, pre-registration-driven Strategy Lab analysis."""

from .dataset import AnalysisError, ScopeLockedError, open_dataset
from .evaluate import AnalysisResult, evaluate_scope

__all__ = [
    "AnalysisError",
    "AnalysisResult",
    "ScopeLockedError",
    "evaluate_scope",
    "open_dataset",
]
