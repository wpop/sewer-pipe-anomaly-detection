import numpy as np
import pytest
from numpy.typing import NDArray

from sewer_anomaly.evaluation.balanced_threshold_selector import (
    BalancedThresholdSelector,
)


def test_select_finds_perfect_balanced_threshold() -> None:
    """Select a threshold that perfectly separates both classes."""

    selector = BalancedThresholdSelector()

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
    assert result.recall == pytest.approx(1.0)
    assert result.specificity == pytest.approx(1.0)
    assert result.balanced_accuracy == pytest.approx(1.0)
    assert result.roc_auc == pytest.approx(1.0)


def test_select_maximizes_balanced_accuracy() -> None:
    """Select the threshold with the highest balanced accuracy."""

    selector = BalancedThresholdSelector()

    labels = np.array(
        [0, 0, 0, 1, 1, 1],
        dtype=np.int64,
    )
    scores = np.array(
        [0.10, 0.20, 0.80, 0.40, 0.70, 0.90],
        dtype=np.float64,
    )

    result = selector.select(
        labels=labels,
        scores=scores,
    )

    assert result.threshold == pytest.approx(0.40)
    assert result.recall == pytest.approx(1.0)
    assert result.specificity == pytest.approx(2.0 / 3.0)
    assert result.balanced_accuracy == pytest.approx(5.0 / 6.0)


def test_select_rejects_different_shapes() -> None:
    """Reject labels and scores with different shapes."""

    selector = BalancedThresholdSelector()

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

    selector = BalancedThresholdSelector()

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
        selector.select(
            labels=labels,
            scores=scores,
        )


def test_select_rejects_empty_arrays() -> None:
    """Reject empty labels and scores."""

    selector = BalancedThresholdSelector()

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


def test_select_rejects_non_finite_scores() -> None:
    """Reject reconstruction scores containing NaN."""

    selector = BalancedThresholdSelector()

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

    selector = BalancedThresholdSelector()

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
