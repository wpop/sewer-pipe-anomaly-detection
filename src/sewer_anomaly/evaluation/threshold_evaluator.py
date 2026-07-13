import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sewer_anomaly.evaluation.threshold_evaluation_result import (
    ThresholdEvaluationResult,
)


class ThresholdEvaluator:
    """Evaluate a fixed anomaly threshold."""

    def evaluate(
        self,
        labels: NDArray[np.int64],
        scores: NDArray[np.float64],
        threshold: float,
    ) -> ThresholdEvaluationResult:
        """Return classification metrics for a fixed threshold."""

        if labels.shape != scores.shape:
            raise ValueError(
                "Labels and scores must have the same shape."
            )

        if labels.ndim != 1:
            raise ValueError(
                "Labels and scores must be one-dimensional."
            )

        if labels.size == 0:
            raise ValueError(
                "Labels and scores cannot be empty."
            )

        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "Scores must contain only finite values."
            )

        if not np.isfinite(threshold):
            raise ValueError(
                "Threshold must be finite."
            )

        unique_labels = set(labels.tolist())

        if unique_labels != {0, 1}:
            raise ValueError(
                "Labels must contain both classes 0 and 1."
            )

        predictions = (
            scores >= threshold
        ).astype(np.int64)

        matrix = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        )

        true_negative = int(matrix[0, 0])
        false_positive = int(matrix[0, 1])
        false_negative = int(matrix[1, 0])
        true_positive = int(matrix[1, 1])

        normal_count = true_negative + false_positive

        specificity = (
            true_negative / normal_count
            if normal_count > 0
            else 0.0
        )

        return ThresholdEvaluationResult(
            threshold=float(threshold),
            precision=float(
                precision_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),
            recall=float(
                recall_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),
            specificity=float(specificity),
            f1_score=float(
                f1_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),
            roc_auc=float(
                roc_auc_score(
                    labels,
                    scores,
                )
            ),
            pr_auc=float(
                average_precision_score(
                    labels,
                    scores,
                )
            ),
            true_negative=true_negative,
            false_positive=false_positive,
            false_negative=false_negative,
            true_positive=true_positive,
        )
