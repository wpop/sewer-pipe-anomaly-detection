from dataclasses import dataclass


@dataclass(frozen=True)
class BalancedThresholdSelectionResult:
    """Store balanced anomaly threshold selection metrics."""

    threshold: float
    recall: float
    specificity: float
    balanced_accuracy: float
    roc_auc: float
