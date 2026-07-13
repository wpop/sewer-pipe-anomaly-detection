import numpy as np
import pytest
from numpy.typing import NDArray

from sewer_anomaly.evaluation.threshold_selector import ThresholdSelector


def test_select_finds_perfect_threshold() -> None:
    """Select a threshold that perfectly separates both classes."""

    selector = ThresholdSelector()

    labels = np.array(
        [0, 0, 1, 1],
        dtype=np.int64,
    )
    scores = np.array(
        [0.10, 0.20, 0.80, 0.90],
        dtype=np.float64,
    )

    result = selector.select(
        labels=labels,
        scores=scores,
    )

    assert result.threshold == pytest.approx(0.80)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1_score == pytest.approx(1.0)
    assert result.roc_auc == pytest.approx(1.0)


def test_select_returns_metrics_between_zero_and_one() -> None:
    """Return valid metrics for overlapping score distributions."""

    selector = ThresholdSelector()

    labels = np.array(
        [0, 0, 0, 1, 1, 1],
        dtype=np.int64,
    )
    scores = np.array(
        [0.10, 0.40, 0.60, 0.30, 0.70, 0.90],
        dtype=np.float64,
    )

    result = selector.select(
        labels=labels,
        scores=scores,
    )

    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1_score <= 1.0
    assert 0.0 <= result.roc_auc <= 1.0
    assert result.threshold in scores


def test_select_rejects_different_shapes() -> None:
    """Reject labels and scores with different shapes."""

    selector = ThresholdSelector()

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
        selector.select(
            labels=labels,
            scores=scores,
        )


def test_select_rejects_multidimensional_arrays() -> None:
    """Reject labels and scores that are not one-dimensional."""

    selector = ThresholdSelector()

    labels = np.array(
        [
            [0, 1],
            [0, 1],
        ],
        dtype=np.int64,
    )
    scores = np.array(
        [
            [0.1, 0.8],
            [0.2, 0.9],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="must be one-dimensional",
    ):
        selector.select(
            labels=labels,
            scores=scores,
        )


def test_select_rejects_empty_arrays() -> None:
    """Reject empty labels and scores."""

    selector = ThresholdSelector()

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
        selector.select(
            labels=labels,
            scores=scores,
        )


@pytest.mark.parametrize(
    "labels",
    [
        np.array([0, 0, 0], dtype=np.int64),
        np.array([1, 1, 1], dtype=np.int64),
    ],
)
def test_select_requires_both_classes(
    labels: NDArray[np.int64],
) -> None:
    """Require both normal and defective labels."""

    selector = ThresholdSelector()

    scores = np.array(
        [0.1, 0.2, 0.3],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="Labels must contain both classes 0 and 1",
    ):
        selector.select(
            labels=labels,
            scores=scores,
        )
