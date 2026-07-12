from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from sewer_anomaly.data.dataset import SewerMLDataset
from sewer_anomaly.data.preprocessing import (
    ImagePreprocessingConfig,
    ImagePreprocessor,
)


def create_available_manifest(
    normal_manifest_path: Path,
    defective_manifest_path: Path,
    image_directory: Path,
    output_path: Path,
    samples_per_class: int,
) -> Path:
    """Create a balanced manifest containing available validation images."""

    normal_dataframe = pd.read_csv(normal_manifest_path)
    defective_dataframe = pd.read_csv(defective_manifest_path)

    # valid00 contains only part of the official validation split.
    normal_available = normal_dataframe[
        normal_dataframe["Filename"].map(
            lambda filename: (image_directory / str(filename)).is_file()
        )
    ]
    defective_available = defective_dataframe[
        defective_dataframe["Filename"].map(
            lambda filename: (image_directory / str(filename)).is_file()
        )
    ]

    # Keep an equal number of normal and defective images.
    normal_subset = normal_available.head(samples_per_class)
    defective_subset = defective_available.head(samples_per_class)

    inspection_dataframe = pd.concat(
        [normal_subset, defective_subset],
        ignore_index=True,
    )

    if inspection_dataframe.empty:
        raise RuntimeError("No validation images were found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inspection_dataframe.to_csv(output_path, index=False)

    return output_path


def save_batch_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
) -> None:
    """Save a visual grid of images from one DataLoader batch."""

    # make_grid expects a BCHW tensor and returns a CHW image grid.
    grid = make_grid(images.cpu(), nrow=4, padding=4)

    # Matplotlib expects image channels in the final dimension.
    grid_image = grid.permute(1, 2, 0).numpy()

    figure, axis = plt.subplots(figsize=(12, 8))
    axis.imshow(grid_image)
    axis.set_title(f"Validation batch labels: {labels.tolist()}")
    axis.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Inspect a real Sewer-ML DataLoader batch."""

    image_directory = Path("data/raw/sewer_ml/valid00")
    processed_directory = Path("data/processed")
    interim_directory = Path("data/interim")
    output_directory = Path("outputs/figures")

    # Build a small balanced manifest from images available in valid00.
    manifest_path = create_available_manifest(
        normal_manifest_path=processed_directory / "val_normal.csv",
        defective_manifest_path=processed_directory / "val_defective.csv",
        image_directory=image_directory,
        output_path=interim_directory / "valid00_inspection.csv",
        samples_per_class=16,
    )

    preprocessor = ImagePreprocessor(
        ImagePreprocessingConfig(
            image_width=256,
            image_height=256,
        )
    )

    dataset = SewerMLDataset(
        manifest_path=manifest_path,
        image_directory=image_directory,
        preprocessor=preprocessor,
    )

    # DataLoader loads and combines multiple dataset samples into batches.
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(dataloader))

    images = cast(torch.Tensor, batch["image"])
    labels = cast(torch.Tensor, batch["label"])
    filenames = cast(list[str], batch["filename"])

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch image shape: {tuple(images.shape)}")
    print(f"Batch labels: {labels.tolist()}")
    print(f"Batch filenames: {filenames}")

    # Transfer the batch to CUDA when an NVIDIA GPU is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device, non_blocking=True)

    print(f"Device: {device}")
    print(f"Tensor device: {images.device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    save_batch_grid(
        images=images,
        labels=labels,
        output_path=output_directory / "validation_batch.png",
    )

    print(
        "Batch figure saved to: "
        f"{output_directory / 'validation_batch.png'}"
    )


if __name__ == "__main__":
    main()
