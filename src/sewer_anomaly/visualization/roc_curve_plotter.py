from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, roc_curve


class RocCurvePlotter:
    """Plot a receiver operating characteristic curve."""

    def save(
        self,
        labels: NDArray[np.int64],
        scores: NDArray[np.float64],
        output_path: Path,
    ) -> None:
        """Save a ROC curve for reconstruction scores."""

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

        false_positive_rates, true_positive_rates, _ = roc_curve(
            labels,
            scores,
        )

        roc_auc = float(
            roc_auc_score(
                labels,
                scores,
            )
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure, axis = plt.subplots(
            figsize=(8, 6),
        )

        axis.plot(
            false_positive_rates,
            true_positive_rates,
            linewidth=2,
            label=f"Autoencoder ROC-AUC = {roc_auc:.4f}",
        )

        axis.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            linewidth=1.5,
            label="Random classifier",
        )

        axis.set_title(
            "Receiver Operating Characteristic"
        )
        axis.set_xlabel(
            "False positive rate"
        )
        axis.set_ylabel(
            "True positive rate"
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
            loc="lower right",
        )

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=150,
        )

        plt.close(figure)
