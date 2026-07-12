from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from sewer_anomaly.data.preprocessing import (
    ImagePreprocessingConfig,
    ImagePreprocessor,
)


def find_existing_image(
    manifest_path: Path,
    image_directory: Path,
) -> Path:
    """Find the first manifest image available in the selected directory."""

    dataframe = pd.read_csv(manifest_path)

    # valid00 contains only part of the official validation split.
    for filename in dataframe["Filename"]:
        image_path = image_directory / str(filename)

        if image_path.is_file():
            return image_path

    raise FileNotFoundError(
        f"No manifest images were found in: {image_directory}"
    )


def tensor_to_image(tensor: torch.Tensor) -> NDArray[np.float32]:
    """Convert a CHW tensor to an HWC NumPy image."""

    # Matplotlib expects channels in the final dimension.
    image: NDArray[np.float32] = (
        tensor.permute(1, 2, 0).cpu().numpy()
    )
    return image


def main() -> None:
    """Visualise preprocessing for normal and defective pipe images."""

    image_directory = Path("data/raw/sewer_ml/valid00")
    processed_directory = Path("data/processed")
    output_directory = Path("outputs/figures")

    # Create the output directory when it does not exist.
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select real normal and defective samples available in valid00.
    normal_path = find_existing_image(
        processed_directory / "val_normal.csv",
        image_directory,
    )
    defective_path = find_existing_image(
        processed_directory / "val_defective.csv",
        image_directory,
    )

    # Use the same fixed input size planned for the autoencoder.
    config = ImagePreprocessingConfig(
        image_width=256,
        image_height=256,
    )
    preprocessor = ImagePreprocessor(config)

    # Load original RGB images.
    normal_original = preprocessor.load_image(normal_path)
    defective_original = preprocessor.load_image(defective_path)

    # Apply resize, normalization, and CHW tensor conversion.
    normal_tensor = preprocessor.preprocess(normal_original)
    defective_tensor = preprocessor.preprocess(defective_original)

    print(f"Normal image: {normal_path}")
    print(f"Normal tensor shape: {tuple(normal_tensor.shape)}")
    print(
        "Normal tensor range: "
        f"{normal_tensor.min().item():.4f} to "
        f"{normal_tensor.max().item():.4f}"
    )

    print(f"\nDefective image: {defective_path}")
    print(f"Defective tensor shape: {tuple(defective_tensor.shape)}")
    print(
        "Defective tensor range: "
        f"{defective_tensor.min().item():.4f} to "
        f"{defective_tensor.max().item():.4f}"
    )

    # Compare original images with tensors prepared for the model.
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].imshow(normal_original)
    axes[0, 0].set_title(f"Normal original: {normal_path.name}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(tensor_to_image(normal_tensor))
    axes[0, 1].set_title("Normal preprocessed: 256 x 256")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(defective_original)
    axes[1, 0].set_title(f"Defective original: {defective_path.name}")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(tensor_to_image(defective_tensor))
    axes[1, 1].set_title("Defective preprocessed: 256 x 256")
    axes[1, 1].axis("off")

    figure.tight_layout()

    output_path = output_directory / "preprocessing_comparison.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")

    # Release figure memory because no interactive GUI backend is used.
    plt.close(figure)

    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()
