from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

from sewer_anomaly.data.dataset import SewerMLDataset
from sewer_anomaly.data.preprocessing import (
    ImagePreprocessingConfig,
    ImagePreprocessor,
)
from sewer_anomaly.models.autoencoder import ConvolutionalAutoencoder


def main() -> None:
    """Run the autoencoder on one real Sewer-ML batch."""

    manifest_path = Path("data/interim/valid00_inspection.csv")
    image_directory = Path("data/raw/sewer_ml/valid00")

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

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(dataloader))
    images = cast(torch.Tensor, batch["image"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvolutionalAutoencoder(latent_channels=128).to(device)
    model.eval()

    images = images.to(device, non_blocking=True)

    with torch.inference_mode():
        latent = model.encoder(images)
        reconstructed = model(images)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Input shape: {tuple(images.shape)}")
    print(f"Latent shape: {tuple(latent.shape)}")
    print(f"Output shape: {tuple(reconstructed.shape)}")

    print(
        "Output range: "
        f"{reconstructed.min().item():.6f} to "
        f"{reconstructed.max().item():.6f}"
    )

    assert images.shape == reconstructed.shape
    assert latent.shape == (images.shape[0], 128, 16, 16)
    assert reconstructed.min().item() >= 0.0
    assert reconstructed.max().item() <= 1.0

    print("Autoencoder runtime test passed.")


if __name__ == "__main__":
    main()
