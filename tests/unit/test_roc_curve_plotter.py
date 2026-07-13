from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from sewer_anomaly.visualization.roc_curve_plotter import (
    RocCurvePlotter,
)


def test_save_creates_png_file(
    tmp_path: Path,
) -> None:
    """Create a non-empty ROC curve PNG file."""

    plotter = RocCurvePlotter()

    labels = np.array(
        [0, 0, 1, 1],
        dtype=np.int64,
    )
    scores = np.array(
        [0.10, 0.20, 0.80, 0.90],
        dtype=np.float64,
    )

    output_path = tmp_path / "figures" / "roc_curve.png"

    plotter.save(
        labels=labels,
        scores=scores,
        output_path=output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_save_rejects_different_shapes(
    tmp_path: Path,
) -> None:
    """Reject labels and scores with different shapes."""

    plotter = RocCurvePlotter()

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
            output_path=tmp_path / "roc_curve.png",
        )


def test_save_rejects_multidimensional_arrays(
    tmp_path: Path,
) -> None:
    """Reject labels and scores that are not one-dimensional."""

    plotter = RocCurvePlotter()

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
            output_path=tmp_path / "roc_curve.png",
        )


def test_save_rejects_empty_arrays(
    tmp_path: Path,
) -> None:
    """Reject empty labels and scores."""

    plotter = RocCurvePlotter()

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
            output_path=tmp_path / "roc_curve.png",
        )


@pytest.mark.parametrize(
    "invalid_score",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_save_rejects_non_finite_scores(
    tmp_path: Path,
    invalid_score: float,
) -> None:
    """Reject non-finite reconstruction scores."""

    plotter = RocCurvePlotter()

    labels: NDArray[np.int64] = np.array(
        [0, 1],
        dtype=np.int64,
    )
    scores: NDArray[np.float64] = np.array(
        [0.1, invalid_score],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="Scores must contain only finite values",
    ):
        plotter.save(
            labels=labels,
            scores=scores,
            output_path=tmp_path / "roc_curve.png",
        )


@pytest.mark.parametrize(
    "labels",
    [
        np.array([0, 0, 0], dtype=np.int64),
        np.array([1, 1, 1], dtype=np.int64),
    ],
)
def test_save_requires_both_classes(
    tmp_path: Path,
    labels: NDArray[np.int64],
) -> None:
    """Require both normal and defective labels."""

    plotter = RocCurvePlotter()

    scores = np.array(
        [0.1, 0.2, 0.3],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="Labels must contain both classes 0 and 1",
    ):
        plotter.save(
            labels=labels,
            scores=scores,
            output_path=tmp_path / "roc_curve.png",
        )
