from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class ImagePreprocessingConfig:
    """Store image preprocessing parameters."""

    image_width: int = 256
    image_height: int = 256


class ImagePreprocessor:
    """Load and preprocess images for model input."""

    def __init__(self, config: ImagePreprocessingConfig) -> None:
        # Keep preprocessing parameters immutable and reusable.
        self._config = config

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load an image and convert it from BGR to RGB."""

        # OpenCV loads colour images in BGR channel order.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Image could not be loaded: {image_path}")

        # Convert to RGB for PyTorch and visualisation tools.
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Resize, normalise, and convert an image to a tensor."""

        # Resize every image to the fixed model input size.
        resized = cv2.resize(
            image,
            (self._config.image_width, self._config.image_height),
            interpolation=cv2.INTER_AREA,
        )

        # Scale pixel values from [0, 255] to [0, 1].
        normalized = resized.astype(np.float32) / 255.0

        # Convert HWC image layout to PyTorch CHW tensor layout.
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).contiguous()

        return tensor

    def load_and_preprocess(self, image_path: Path) -> torch.Tensor:
        """Load an image and apply the complete preprocessing pipeline."""

        image = self.load_image(image_path)
        return self.preprocess(image)
