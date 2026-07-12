from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from sewer_anomaly.data.dataset import SewerMLDataset
from sewer_anomaly.data.preprocessing import (
    ImagePreprocessingConfig,
    ImagePreprocessor,
)


def create_test_image(image_path: Path) -> None:
    """Create a small synthetic image for dataset tests."""

    # Use a constant image to keep test output deterministic.
    image = np.full((16, 24, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)


def test_dataset_returns_expected_sample(tmp_path: Path) -> None:
    """Load one manifest row as a preprocessed dataset sample."""

    image_directory = tmp_path / "images"
    image_directory.mkdir()

    image_path = image_directory / "sample.png"
    create_test_image(image_path)

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "Filename": ["sample.png"],
            "Defect": [1],
        }
    ).to_csv(manifest_path, index=False)

    preprocessor = ImagePreprocessor(
        ImagePreprocessingConfig(
            image_width=64,
            image_height=32,
        )
    )

    dataset = SewerMLDataset(
        manifest_path=manifest_path,
        image_directory=image_directory,
        preprocessor=preprocessor,
    )

    sample = dataset[0]

    # Verify image, label, and metadata returned by the dataset.
    assert len(dataset) == 1
    assert sample["image"].shape == (3, 32, 64)
    assert sample["image"].dtype == torch.float32
    assert sample["label"].item() == 1
    assert sample["filename"] == "sample.png"


def test_missing_manifest_raises_error(tmp_path: Path) -> None:
    """Reject a manifest path that does not exist."""

    image_directory = tmp_path / "images"
    image_directory.mkdir()

    preprocessor = ImagePreprocessor(ImagePreprocessingConfig())

    with pytest.raises(FileNotFoundError, match="Manifest file not found"):
        SewerMLDataset(
            manifest_path=tmp_path / "missing.csv",
            image_directory=image_directory,
            preprocessor=preprocessor,
        )


def test_missing_image_directory_raises_error(tmp_path: Path) -> None:
    """Reject an image directory that does not exist."""

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "Filename": ["sample.png"],
            "Defect": [0],
        }
    ).to_csv(manifest_path, index=False)

    preprocessor = ImagePreprocessor(ImagePreprocessingConfig())

    with pytest.raises(FileNotFoundError, match="Image directory not found"):
        SewerMLDataset(
            manifest_path=manifest_path,
            image_directory=tmp_path / "missing_images",
            preprocessor=preprocessor,
        )


def test_manifest_without_defect_column_raises_error(tmp_path: Path) -> None:
    """Reject a training manifest without binary defect labels."""

    image_directory = tmp_path / "images"
    image_directory.mkdir()

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "Filename": ["sample.png"],
        }
    ).to_csv(manifest_path, index=False)

    preprocessor = ImagePreprocessor(ImagePreprocessingConfig())

    with pytest.raises(ValueError, match="Defect column"):
        SewerMLDataset(
            manifest_path=manifest_path,
            image_directory=image_directory,
            preprocessor=preprocessor,
        )


def test_index_out_of_range_raises_error(tmp_path: Path) -> None:
    """Reject an index outside the manifest range."""

    image_directory = tmp_path / "images"
    image_directory.mkdir()

    image_path = image_directory / "sample.png"
    create_test_image(image_path)

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "Filename": ["sample.png"],
            "Defect": [0],
        }
    ).to_csv(manifest_path, index=False)

    preprocessor = ImagePreprocessor(ImagePreprocessingConfig())

    dataset = SewerMLDataset(
        manifest_path=manifest_path,
        image_directory=image_directory,
        preprocessor=preprocessor,
    )

    with pytest.raises(IndexError, match="out of range"):
        _ = dataset[1]
