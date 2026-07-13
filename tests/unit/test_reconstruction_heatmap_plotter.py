from pathlib import Path

import numpy as np
import pytest
import torch

from sewer_anomaly.visualization.reconstruction_heatmap_plotter import (
    ReconstructionHeatmapPlotter,
)


def test_save_creates_png_file(
    tmp_path: Path,
) -> None:
    """Create a non-empty reconstruction heatmap PNG file."""

    plotter = ReconstructionHeatmapPlotter()

    image = torch.rand(
        3,
        8,
        8,
    )
    reconstruction = torch.rand(
        3,
        8,
        8,
    )
    heatmap = torch.rand(
        8,
        8,
    )

    output_path = (
        tmp_path
        / "figures"
        / "sample_heatmap.png"
    )

    plotter.save(
        image=image,
        reconstruction=reconstruction,
        heatmap=heatmap,
        output_path=output_path,
        filename="sample.png",
        reconstruction_score=0.00125,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_save_rejects_different_image_shapes(
    tmp_path: Path,
) -> None:
    """Reject image and reconstruction tensors with different shapes."""

    plotter = ReconstructionHeatmapPlotter()

    image = torch.zeros(
        3,
        8,
        8,
    )
    reconstruction = torch.zeros(
        3,
        7,
        8,
    )
    heatmap = torch.zeros(
        8,
        8,
    )

    with pytest.raises(
        ValueError,
        match="Image and reconstruction must have the same shape",
    ):
        plotter.save(
            image=image,
            reconstruction=reconstruction,
            heatmap=heatmap,
            output_path=tmp_path / "heatmap.png",
            filename="sample.png",
            reconstruction_score=0.1,
        )


@pytest.mark.parametrize(
    "shape",
    [
        (8, 8),
        (1, 3, 8, 8),
    ],
)
def test_save_rejects_invalid_image_dimensions(
    tmp_path: Path,
    shape: tuple[int, ...],
) -> None:
    """Reject tensors without channels-height-width dimensions."""

    plotter = ReconstructionHeatmapPlotter()

    image = torch.zeros(shape)
    reconstruction = torch.zeros(shape)
    heatmap = torch.zeros(
        8,
        8,
    )

    with pytest.raises(
        ValueError,
        match="Expected image tensors with shape",
    ):
        plotter.save(
            image=image,
            reconstruction=reconstruction,
            heatmap=heatmap,
            output_path=tmp_path / "heatmap.png",
            filename="sample.png",
            reconstruction_score=0.1,
        )


def test_save_requires_three_rgb_channels(
    tmp_path: Path,
) -> None:
    """Require three RGB image channels."""

    plotter = ReconstructionHeatmapPlotter()

    image = torch.zeros(
        1,
        8,
        8,
    )
    reconstruction = torch.zeros_like(image)
    heatmap = torch.zeros(
        8,
        8,
    )

    with pytest.raises(
        ValueError,
        match="three RGB channels",
    ):
        plotter.save(
            image=image,
            reconstruction=reconstruction,
            heatmap=heatmap,
            output_path=tmp_path / "heatmap.png",
            filename="sample.png",
            reconstruction_score=0.1,
        )


def test_save_rejects_non_two_dimensional_heatmap(
    tmp_path: Path,
) -> None:
    """Reject a heatmap that is not two-dimensional."""

    plotter = ReconstructionHeatmapPlotter()

    image = torch.zeros(
        3,
        8,
        8,
    )
    reconstruction = torch.zeros_like(image)

    heatmap = torch.zeros(
        1,
        8,
        8,
    )

    with pytest.raises(
        ValueError,
        match="Expected heatmap tensor with shape",
    ):
        plotter.save(
            image=image,
            reconstruction=reconstruction,
            heatmap=heatmap,
            output_path=tmp_path / "heatmap.png",
            filename="sample.png",
            reconstruction_score=0.1,
        )


def test_save_rejects_wrong_heatmap_spatial_shape(
    tmp_path: Path,
) -> None:
    """Reject a heatmap with a different spatial shape."""

    plotter = ReconstructionHeatmapPlotter()

    image = torch.zeros(
        3,
        8,
        8,
    )
    reconstruction = torch.zeros_like(image)

    heatmap = torch.zeros(
        7,
        8,
    )

    with pytest.raises(
        ValueError,
        match="Heatmap spatial shape must match image shape",
    ):
        plotter.save(
            image=image,
            reconstruction=reconstruction,
            heatmap=heatmap,
            output_path=tmp_path / "heatmap.png",
            filename="sample.png",
            reconstruction_score=0.1,
        )


@pytest.mark.parametrize(
    (
        "tensor_name",
        "expected_message",
    ),
    [
        (
            "image",
            "Image must contain only finite values",
        ),
        (
            "reconstruction",
            "Reconstruction must contain only finite values",
        ),
        (
            "heatmap",
            "Heatmap must contain only finite values",
        ),
    ],
)
def test_save_rejects_non_finite_tensors(
    tmp_path: Path,
    tensor_name: str,
    expected_message: str,
) -> None:
    """Reject non-finite tensor values."""

    plotter = ReconstructionHeatmapPlotter()

    image = torch.zeros(
        3,
        8,
        8,
    )
    reconstruction = torch.zeros_like(image)
    heatmap = torch.zeros(
        8,
        8,
    )

    if tensor_name == "image":
        image[0, 0, 0] = float("nan")
    elif tensor_name == "reconstruction":
        reconstruction[0, 0, 0] = float("nan")
    else:
        heatmap[0, 0] = float("nan")

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        plotter.save(
            image=image,
            reconstruction=reconstruction,
            heatmap=heatmap,
            output_path=tmp_path / "heatmap.png",
            filename="sample.png",
            reconstruction_score=0.1,
        )


@pytest.mark.parametrize(
    "score",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_save_rejects_non_finite_score(
    tmp_path: Path,
    score: float,
) -> None:
    """Reject a non-finite reconstruction score."""

    plotter = ReconstructionHeatmapPlotter()

    image = torch.zeros(
        3,
        8,
        8,
    )
    reconstruction = torch.zeros_like(image)
    heatmap = torch.zeros(
        8,
        8,
    )

    with pytest.raises(
        ValueError,
        match="Reconstruction score must be finite",
    ):
        plotter.save(
            image=image,
            reconstruction=reconstruction,
            heatmap=heatmap,
            output_path=tmp_path / "heatmap.png",
            filename="sample.png",
            reconstruction_score=score,
        )


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "   ",
    ],
)
def test_save_rejects_empty_filename(
    tmp_path: Path,
    filename: str,
) -> None:
    """Reject an empty filename."""

    plotter = ReconstructionHeatmapPlotter()

    image = torch.zeros(
        3,
        8,
        8,
    )
    reconstruction = torch.zeros_like(image)
    heatmap = torch.zeros(
        8,
        8,
    )

    with pytest.raises(
        ValueError,
        match="Filename cannot be empty",
    ):
        plotter.save(
            image=image,
            reconstruction=reconstruction,
            heatmap=heatmap,
            output_path=tmp_path / "heatmap.png",
            filename=filename,
            reconstruction_score=0.1,
        )
