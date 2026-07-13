import pytest
import torch

from sewer_anomaly.inference.reconstruction_heatmap_generator import (
    ReconstructionHeatmapGenerator,
)


def test_generate_returns_expected_heatmap() -> None:
    """Return the channel-averaged squared reconstruction error."""

    generator = ReconstructionHeatmapGenerator()

    image = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            [
                [2.0, 4.0],
                [6.0, 8.0],
            ],
        ],
        dtype=torch.float32,
    )

    reconstruction = torch.tensor(
        [
            [
                [0.0, 1.0],
                [2.0, 3.0],
            ],
            [
                [0.0, 2.0],
                [4.0, 6.0],
            ],
        ],
        dtype=torch.float32,
    )

    heatmap = generator.generate(
        image=image,
        reconstruction=reconstruction,
    )

    expected = torch.full(
        (2, 2),
        2.5,
        dtype=torch.float32,
    )

    assert heatmap.shape == (2, 2)
    assert torch.allclose(
        heatmap,
        expected,
    )


def test_generate_returns_zero_heatmap_for_identical_images() -> None:
    """Return a zero heatmap for identical tensors."""

    generator = ReconstructionHeatmapGenerator()

    image = torch.rand(
        3,
        4,
        5,
    )

    heatmap = generator.generate(
        image=image,
        reconstruction=image.clone(),
    )

    assert heatmap.shape == (4, 5)
    assert torch.count_nonzero(heatmap).item() == 0


def test_generate_rejects_different_shapes() -> None:
    """Reject tensors with different shapes."""

    generator = ReconstructionHeatmapGenerator()

    image = torch.zeros(
        3,
        4,
        4,
    )
    reconstruction = torch.zeros(
        3,
        5,
        4,
    )

    with pytest.raises(
        ValueError,
        match="Image and reconstruction must have the same shape",
    ):
        generator.generate(
            image=image,
            reconstruction=reconstruction,
        )


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (1, 3, 4, 4),
    ],
)
def test_generate_rejects_invalid_dimensions(
    shape: tuple[int, ...],
) -> None:
    """Reject tensors without channels-height-width dimensions."""

    generator = ReconstructionHeatmapGenerator()

    image = torch.zeros(shape)
    reconstruction = torch.zeros(shape)

    with pytest.raises(
        ValueError,
        match="Expected image tensors with shape",
    ):
        generator.generate(
            image=image,
            reconstruction=reconstruction,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_generate_rejects_non_finite_image(
    invalid_value: float,
) -> None:
    """Reject non-finite values in the input image."""

    generator = ReconstructionHeatmapGenerator()

    image = torch.zeros(
        3,
        2,
        2,
    )
    image[0, 0, 0] = invalid_value

    reconstruction = torch.zeros_like(image)

    with pytest.raises(
        ValueError,
        match="Image must contain only finite values",
    ):
        generator.generate(
            image=image,
            reconstruction=reconstruction,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_generate_rejects_non_finite_reconstruction(
    invalid_value: float,
) -> None:
    """Reject non-finite values in the reconstruction."""

    generator = ReconstructionHeatmapGenerator()

    image = torch.zeros(
        3,
        2,
        2,
    )

    reconstruction = torch.zeros_like(image)
    reconstruction[0, 0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="Reconstruction must contain only finite values",
    ):
        generator.generate(
            image=image,
            reconstruction=reconstruction,
        )
