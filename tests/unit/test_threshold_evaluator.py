import numpy as np
import pytest
from numpy.typing import NDArray

from sewer_anomaly.evaluation.threshold_evaluator import (
    ThresholdEvaluator,
)


def test_evaluate_returns_perfect_metrics() -> None:
    """Return perfect metrics for perfectly separated scores."""

    evaluator = ThresholdEvaluator()

    labels = np.array(
        [0, 0, 1, 1],
        dtype=np.int64,
    )
    scores = np.array(
        [0.10, 0.20, 0.80, 0.90],
        dtype=np.float64,
    )

    result = evaluator.evaluate(
        labels=labels,
        scores=scores,
        threshold=0.50,
    )

    assert result.threshold == pytest.approx(0.50)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.specificity == pytest.approx(1.0)
    assert result.f1_score == pytest.approx(1.0)
    assert result.roc_auc == pytest.approx(1.0)
    assert result.pr_auc == pytest.approx(1.0)

    assert result.true_negative == 2
    assert result.false_positive == 0
    assert result.false_negative == 0
    assert result.true_positive == 2


def test_evaluate_returns_expected_confusion_matrix() -> None:
    """Return expected metrics for partially overlapping scores."""

    evaluator = ThresholdEvaluator()

    labels = np.array(
        [0, 0, 1, 1],
        dtype=np.int64,
    )
    scores = np.array(
        [0.20, 0.70, 0.40, 0.90],
        dtype=np.float64,
    )

    result = evaluator.evaluate(
        labels=labels,
        scores=scores,
        threshold=0.50,
    )

    assert result.true_negative == 1
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.true_positive == 1

    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.specificity == pytest.approx(0.5)
    assert result.f1_score == pytest.approx(0.5)


def test_evaluate_rejects_different_shapes() -> None:
    """Reject labels and scores with different shapes."""

    evaluator = ThresholdEvaluator()

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
        evaluator.evaluate(
            labels=labels,
            scores=scores,
            threshold=0.5,
        )


def test_evaluate_rejects_multidimensional_arrays() -> None:
    """Reject labels and scores that are not one-dimensional."""

    evaluator = ThresholdEvaluator()

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
        evaluator.evaluate(
            labels=labels,
            scores=scores,
            threshold=0.5,
        )


def test_evaluate_rejects_empty_arrays() -> None:
    """Reject empty labels and scores."""

    evaluator = ThresholdEvaluator()

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
        evaluator.evaluate(
            labels=labels,
            scores=scores,
            threshold=0.5,
        )


def test_evaluate_rejects_non_finite_scores() -> None:
    """Reject reconstruction scores containing NaN."""

    evaluator = ThresholdEvaluator()

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
        evaluator.evaluate(
            labels=labels,
            scores=scores,
            threshold=0.5,
        )


@pytest.mark.parametrize(
    "threshold",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_evaluate_rejects_non_finite_threshold(
    threshold: float,
) -> None:
    """Reject NaN and infinite thresholds."""

    evaluator = ThresholdEvaluator()

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
        match="Threshold must be finite",
    ):
        evaluator.evaluate(
            labels=labels,
            scores=scores,
            threshold=threshold,
        )


@pytest.mark.parametrize(
    "labels",
    [
        np.array([0, 0, 0], dtype=np.int64),
        np.array([1, 1, 1], dtype=np.int64),
    ],
)
def test_evaluate_requires_both_classes(
    labels: NDArray[np.int64],
) -> None:
    """Require both normal and defective labels."""

    evaluator = ThresholdEvaluator()

    scores = np.array(
        [0.1, 0.2, 0.3],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="Labels must contain both classes 0 and 1",
    ):
        evaluator.evaluate(
            labels=labels,
            scores=scores,
            threshold=0.2,
        )
