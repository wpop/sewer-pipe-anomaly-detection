from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)


class PrecisionRecallCurvePlotter:
    """Plot a precision-recall curve."""

    def save(
        self,
        labels: NDArray[np.int64],
        scores: NDArray[np.float64],
        output_path: Path,
    ) -> None:
        """Save a precision-recall curve for reconstruction scores."""

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

        precision_values, recall_values, _ = precision_recall_curve(
            labels,
            scores,
        )

        pr_auc = float(
            average_precision_score(
                labels,
                scores,
            )
        )

        defective_fraction = float(
            np.count_nonzero(labels == 1) / labels.size
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure, axis = plt.subplots(
            figsize=(8, 6),
        )

        axis.plot(
            recall_values,
            precision_values,
            linewidth=2,
            label=f"Autoencoder PR-AUC = {pr_auc:.4f}",
        )

        axis.axhline(
            defective_fraction,
            linestyle="--",
            linewidth=1.5,
            label=(
                "Class prevalence baseline "
                f"= {defective_fraction:.4f}"
            ),
        )

        axis.set_title(
            "Precision–Recall Curve"
        )
        axis.set_xlabel(
            "Recall"
        )
        axis.set_ylabel(
            "Precision"
        )

        axis.set_xlim(
            0.0,
            1.0,
        )
        axis.set_ylim(
            0.0,
            1.0,
        )

        axis.grid(
            alpha=0.3,
        )
        axis.legend(
            loc="lower left",
        )

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=150,
        )

        plt.close(figure)
