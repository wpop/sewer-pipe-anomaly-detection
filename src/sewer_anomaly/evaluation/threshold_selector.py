import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_curve, roc_auc_score

from sewer_anomaly.evaluation.threshold_result import (
    ThresholdSelectionResult,
)


class ThresholdSelector:
    """Select an anomaly threshold by maximizing the F1 score."""

    def select(
        self,
        labels: NDArray[np.int64],
        scores: NDArray[np.float64],
    ) -> ThresholdSelectionResult:
        """Return the threshold and metrics with the highest F1 score."""

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

        unique_labels = set(labels.tolist())

        if unique_labels != {0, 1}:
            raise ValueError(
                "Labels must contain both classes 0 and 1."
            )

        precision_values, recall_values, thresholds = (
            precision_recall_curve(
                labels,
                scores,
            )
        )

        precision_for_thresholds = precision_values[:-1]
        recall_for_thresholds = recall_values[:-1]

        denominator = (
            precision_for_thresholds + recall_for_thresholds
        )

        f1_values = np.divide(
            2.0
            * precision_for_thresholds
            * recall_for_thresholds,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0.0,
        )

        best_index = int(np.argmax(f1_values))

        roc_auc = float(
            roc_auc_score(
                labels,
                scores,
            )
        )

        return ThresholdSelectionResult(
            threshold=float(thresholds[best_index]),
            precision=float(
                precision_for_thresholds[best_index]
            ),
            recall=float(
                recall_for_thresholds[best_index]
            ),
            f1_score=float(f1_values[best_index]),
            roc_auc=roc_auc,
        )
