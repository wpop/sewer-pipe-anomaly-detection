from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdSelectionResult:
    """Store anomaly threshold selection metrics."""

    threshold: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
