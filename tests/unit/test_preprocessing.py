from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from sewer_anomaly.data.preprocessing import (
    ImagePreprocessingConfig,
    ImagePreprocessor,
)


def test_load_image_converts_bgr_to_rgb(tmp_path: Path) -> None:
    """Convert OpenCV BGR input to RGB output."""

    image_path = tmp_path / "test.png"

    # Create a blue pixel image in OpenCV BGR channel order.
    bgr_image = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr_image[:, :] = [255, 0, 0]
    cv2.imwrite(str(image_path), bgr_image)

    preprocessor = ImagePreprocessor(ImagePreprocessingConfig())
    rgb_image = preprocessor.load_image(image_path)

    # BGR blue becomes RGB blue.
    assert rgb_image[0, 0].tolist() == [0, 0, 255]


def test_preprocess_returns_normalized_chw_tensor() -> None:
    """Resize and convert an image to a normalized CHW tensor."""

    image = np.full((20, 30, 3), 255, dtype=np.uint8)

    config = ImagePreprocessingConfig(
        image_width=64,
        image_height=32,
    )
    preprocessor = ImagePreprocessor(config)

    tensor = preprocessor.preprocess(image)

    # PyTorch image tensors use channels-first layout.
    assert tensor.shape == (3, 32, 64)

    # Pixel values must be stored as float32 in the [0, 1] range.
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0

    # Contiguous storage is expected after the HWC-to-CHW permutation.
    assert tensor.is_contiguous()


def test_load_and_preprocess_image(tmp_path: Path) -> None:
    """Run the complete image preprocessing pipeline."""

    image_path = tmp_path / "test.png"
    image = np.full((10, 15, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    config = ImagePreprocessingConfig(
        image_width=40,
        image_height=24,
    )
    preprocessor = ImagePreprocessor(config)

    tensor = preprocessor.load_and_preprocess(image_path)

    assert tensor.shape == (3, 24, 40)


def test_missing_image_raises_error(tmp_path: Path) -> None:
    """Reject an image path that does not exist."""

    preprocessor = ImagePreprocessor(ImagePreprocessingConfig())

    with pytest.raises(FileNotFoundError, match="could not be loaded"):
        preprocessor.load_image(tmp_path / "missing.png")
