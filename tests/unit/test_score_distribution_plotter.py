from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from sewer_anomaly.visualization.score_distribution_plotter import (
    ScoreDistributionPlotter,
)


def test_save_creates_png_file(
    tmp_path: Path,
) -> None:
    """Create a non-empty PNG file."""

    plotter = ScoreDistributionPlotter()

    labels = np.array(
        [0, 0, 1, 1],
        dtype=np.int64,
    )
    scores = np.array(
        [0.10, 0.20, 0.80, 0.90],
        dtype=np.float64,
    )

    output_path = tmp_path / "figures" / "distribution.png"

    plotter.save(
        labels=labels,
        scores=scores,
        output_path=output_path,
        f1_threshold=0.40,
        balanced_threshold=0.60,
        bins=10,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_save_rejects_different_shapes(
    tmp_path: Path,
) -> None:
    """Reject labels and scores with different shapes."""

    plotter = ScoreDistributionPlotter()

    labels = np.array(
        [0, 1],
        dtype=np.int64,
    )
    scores = np.array(
        [0.1, 0.2, 0.3],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="Labels and scores must have the same shape",
    ):
        plotter.save(
            labels=labels,
            scores=scores,
            output_path=tmp_path / "distribution.png",
            f1_threshold=0.2,
            balanced_threshold=0.3,
        )


def test_save_rejects_multidimensional_arrays(
    tmp_path: Path,
) -> None:
    """Reject labels and scores that are not one-dimensional."""

    plotter = ScoreDistributionPlotter()

    labels = np.array(
        [[0, 1]],
        dtype=np.int64,
    )
    scores = np.array(
        [[0.1, 0.9]],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="must be one-dimensional",
    ):
        plotter.save(
            labels=labels,
            scores=scores,
            output_path=tmp_path / "distribution.png",
            f1_threshold=0.2,
            balanced_threshold=0.3,
        )


def test_save_rejects_empty_arrays(
    tmp_path: Path,
) -> None:
    """Reject empty labels and scores."""

    plotter = ScoreDistributionPlotter()

    labels = np.array(
        [],
        dtype=np.int64,
    )
    scores = np.array(
        [],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        plotter.save(
            labels=labels,
            scores=scores,
            output_path=tmp_path / "distribution.png",
            f1_threshold=0.2,
            balanced_threshold=0.3,
        )


def test_save_rejects_non_finite_scores(
    tmp_path: Path,
) -> None:
    """Reject reconstruction scores containing NaN."""

    plotter = ScoreDistributionPlotter()

    labels = np.array(
        [0, 1],
        dtype=np.int64,
    )
    scores = np.array(
        [0.1, np.nan],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="Scores must contain only finite values",
    ):
        plotter.save(
            labels=labels,
            scores=scores,
            output_path=tmp_path / "distribution.png",
            f1_threshold=0.2,
            balanced_threshold=0.3,
        )


def test_save_rejects_invalid_bins(
    tmp_path: Path,
) -> None:
    """Reject a non-positive histogram bin count."""

    plotter = ScoreDistributionPlotter()

    labels = np.array(
        [0, 1],
        dtype=np.int64,
    )
    scores = np.array(
        [0.1, 0.9],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="Bins must be greater than zero",
    ):
        plotter.save(
            labels=labels,
            scores=scores,
            output_path=tmp_path / "distribution.png",
            f1_threshold=0.2,
            balanced_threshold=0.3,
            bins=0,
        )


@pytest.mark.parametrize(
    ("f1_threshold", "balanced_threshold", "expected_message"),
    [
        (
            np.nan,
            0.3,
            "F1 threshold must be finite",
        ),
        (
            0.2,
            np.inf,
            "Balanced threshold must be finite",
        ),
    ],
)
def test_save_rejects_non_finite_thresholds(
    tmp_path: Path,
    f1_threshold: float,
    balanced_threshold: float,
    expected_message: str,
) -> None:
    """Reject non-finite anomaly thresholds."""

    plotter = ScoreDistributionPlotter()

    labels: NDArray[np.int64] = np.array(
        [0, 1],
        dtype=np.int64,
    )
    scores: NDArray[np.float64] = np.array(
        [0.1, 0.9],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        plotter.save(
            labels=labels,
            scores=scores,
            output_path=tmp_path / "distribution.png",
            f1_threshold=f1_threshold,
            balanced_threshold=balanced_threshold,
        )
