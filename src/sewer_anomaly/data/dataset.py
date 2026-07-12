from pathlib import Path
from typing import TypedDict

import pandas as pd
import torch
from torch.utils.data import Dataset

from sewer_anomaly.data.preprocessing import ImagePreprocessor


class SewerMLSample(TypedDict):
    """Describe one sample returned by the dataset."""

    image: torch.Tensor
    label: torch.Tensor
    filename: str


class SewerMLDataset(Dataset[SewerMLSample]):
    """Load Sewer-ML images using a prepared manifest file."""

    def __init__(
        self,
        manifest_path: Path,
        image_directory: Path,
        preprocessor: ImagePreprocessor,
    ) -> None:
        # Validate all external paths before reading any data.
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        if not image_directory.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {image_directory}"
            )

        dataframe = pd.read_csv(manifest_path)

        # Training and validation manifests require filenames and binary labels.
        if "Filename" not in dataframe.columns:
            raise ValueError("Manifest must contain a Filename column.")

        if "Defect" not in dataframe.columns:
            raise ValueError("Manifest must contain a Defect column.")

        if dataframe.empty:
            raise ValueError("Manifest file is empty.")

        # Keep metadata and preprocessing dependencies inside the dataset.
        self._dataframe = dataframe
        self._image_directory = image_directory
        self._preprocessor = preprocessor

    def __len__(self) -> int:
        """Return the number of manifest rows."""

        return len(self._dataframe)

    def __getitem__(self, index: int) -> SewerMLSample:
        """Load and preprocess one dataset sample."""

        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")

        row = self._dataframe.iloc[index]

        filename = str(row["Filename"])
        image_path = self._image_directory / filename

        # Load the RGB image and convert it to a normalized CHW tensor.
        image = self._preprocessor.load_and_preprocess(image_path)

        # Store the binary defect label as a PyTorch integer tensor.
        label = torch.tensor(
            int(row["Defect"]),
            dtype=torch.long,
        )

        return {
            "image": image,
            "label": label,
            "filename": filename,
        }
