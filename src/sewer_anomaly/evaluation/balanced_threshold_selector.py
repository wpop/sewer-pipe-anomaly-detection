import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, roc_curve

from sewer_anomaly.evaluation.balanced_threshold_result import (
    BalancedThresholdSelectionResult,
)


class BalancedThresholdSelector:
    """Select a threshold by maximizing balanced accuracy."""

    def select(
        self,
        labels: NDArray[np.int64],
        scores: NDArray[np.float64],
    ) -> BalancedThresholdSelectionResult:
        """Return the threshold with the best balanced accuracy."""

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

        false_positive_rates, true_positive_rates, thresholds = roc_curve(
            labels,
            scores,
        )

        recalls = true_positive_rates
        specificities = 1.0 - false_positive_rates

        balanced_accuracies = (
            recalls + specificities
        ) / 2.0

        finite_threshold_mask = np.isfinite(thresholds)

        finite_indices = np.flatnonzero(
            finite_threshold_mask
        )

        finite_balanced_accuracies = balanced_accuracies[
            finite_threshold_mask
        ]

        best_finite_position = int(
            np.argmax(finite_balanced_accuracies)
        )

        best_index = int(
            finite_indices[best_finite_position]
        )

        roc_auc = float(
            roc_auc_score(
                labels,
                scores,
            )
        )

        return BalancedThresholdSelectionResult(
            threshold=float(thresholds[best_index]),
            recall=float(recalls[best_index]),
            specificity=float(specificities[best_index]),
            balanced_accuracy=float(
                balanced_accuracies[best_index]
            ),
            roc_auc=roc_auc,
        )
