"""Read-only, pre-registration-driven Strategy Lab analysis."""

from .dataset import AnalysisError, ScopeLockedError, open_dataset
from .allocation import (
    AllocationResult,
    DatasetInput,
    OFFICIAL_TICKER_SCORER,
    SelectedISTickerView,
    TickerScorer,
    evaluate_allocation,
)
from .evaluate import AnalysisResult, evaluate_scope

__all__ = [
    "AnalysisError",
    "AnalysisResult",
    "AllocationResult",
    "DatasetInput",
    "OFFICIAL_TICKER_SCORER",
    "SelectedISTickerView",
    "ScopeLockedError",
    "TickerScorer",
    "evaluate_allocation",
    "evaluate_scope",
    "open_dataset",
]
