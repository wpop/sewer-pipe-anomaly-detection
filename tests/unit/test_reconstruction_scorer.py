import pytest
import torch

from sewer_anomaly.inference.reconstruction_scorer import (
    ReconstructionScorer,
)


def test_score_returns_zero_for_identical_images() -> None:
    """Return zero reconstruction error for identical tensors."""

    scorer = ReconstructionScorer()

    images = torch.rand(
        2,
        3,
        8,
        8,
    )

    scores = scorer.score(
        images=images,
        reconstructions=images.clone(),
    )

    assert scores.shape == (2,)
    assert torch.equal(scores, torch.zeros(2))


def test_score_returns_expected_mean_squared_error() -> None:
    """Calculate the expected mean squared error for each image."""

    scorer = ReconstructionScorer()

    images = torch.zeros(
        2,
        1,
        2,
        2,
    )

    reconstructions = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                ]
            ],
            [
                [
                    [2.0, 2.0],
                    [2.0, 2.0],
                ]
            ],
        ]
    )

    scores = scorer.score(
        images=images,
        reconstructions=reconstructions,
    )

    # First image:
    # (0 - 1)^2 = 1 for every pixel, therefore MSE = 1.
    #
    # Second image:
    # (0 - 2)^2 = 4 for every pixel, therefore MSE = 4.
    expected_scores = torch.tensor([1.0, 4.0])

    assert torch.allclose(scores, expected_scores)


def test_score_calculates_each_batch_sample_independently() -> None:
    """Preserve one independent score for every batch sample."""

    scorer = ReconstructionScorer()

    images = torch.zeros(
        3,
        3,
        4,
        4,
    )

    reconstructions = torch.zeros_like(images)

    reconstructions[0] = 0.1
    reconstructions[1] = 0.5
    reconstructions[2] = 1.0

    scores = scorer.score(
        images=images,
        reconstructions=reconstructions,
    )

    assert scores.shape == (3,)
    assert scores[0] < scores[1]
    assert scores[1] < scores[2]


def test_score_rejects_different_tensor_shapes() -> None:
    """Reject image and reconstruction tensors with different shapes."""

    scorer = ReconstructionScorer()

    images = torch.zeros(
        2,
        3,
        8,
        8,
    )
    reconstructions = torch.zeros(
        2,
        3,
        16,
        16,
    )

    with pytest.raises(
        ValueError,
        match="must have the same shape",
    ):
        scorer.score(
            images=images,
            reconstructions=reconstructions,
        )


@pytest.mark.parametrize(
    "shape",
    [
        (3, 8, 8),
        (2, 3, 8),
        (2, 3, 8, 8, 1),
    ],
)
def test_score_rejects_non_bchw_tensors(
    shape: tuple[int, ...],
) -> None:
    """Reject tensors that are not four-dimensional BCHW batches."""

    scorer = ReconstructionScorer()

    images = torch.zeros(shape)
    reconstructions = torch.zeros(shape)

    with pytest.raises(
        ValueError,
        match="Expected image tensors with shape",
    ):
        scorer.score(
            images=images,
            reconstructions=reconstructions,
        )
