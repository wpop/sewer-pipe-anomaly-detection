from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdEvaluationResult:
    """Store metrics calculated for a fixed anomaly threshold."""

    threshold: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
