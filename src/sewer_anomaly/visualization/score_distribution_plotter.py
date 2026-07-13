from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class ScoreDistributionPlotter:
    """Plot reconstruction score distributions."""

    def save(
        self,
        labels: NDArray[np.int64],
        scores: NDArray[np.float64],
        output_path: Path,
        f1_threshold: float,
        balanced_threshold: float,
        bins: int = 80,
    ) -> None:
        """Save normal and defective reconstruction score distributions."""

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

        if bins <= 0:
            raise ValueError(
                "Bins must be greater than zero."
            )

        unique_labels = set(labels.tolist())

        if unique_labels != {0, 1}:
            raise ValueError(
                "Labels must contain both classes 0 and 1."
            )

        if not np.isfinite(f1_threshold):
            raise ValueError(
                "F1 threshold must be finite."
            )

        if not np.isfinite(balanced_threshold):
            raise ValueError(
                "Balanced threshold must be finite."
            )

        normal_scores = scores[labels == 0]
        defective_scores = scores[labels == 1]

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure, axis = plt.subplots(
            figsize=(10, 6),
        )

        axis.hist(
            normal_scores,
            bins=bins,
            density=True,
            alpha=0.6,
            label="Normal",
        )

        axis.hist(
            defective_scores,
            bins=bins,
            density=True,
            alpha=0.6,
            label="Defective",
        )

        axis.axvline(
            f1_threshold,
            linestyle="--",
            linewidth=2,
            label="Maximum F1 threshold",
        )

        axis.axvline(
            balanced_threshold,
            linestyle=":",
            linewidth=2,
            label="Maximum balanced-accuracy threshold",
        )

        axis.set_title(
            "Reconstruction Score Distributions"
        )
        axis.set_xlabel(
            "Reconstruction score"
        )
        axis.set_ylabel(
            "Density"
        )
        axis.legend()
        axis.grid(
            alpha=0.3,
        )

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=150,
        )

        plt.close(figure)
