from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

from sewer_anomaly.data.dataset import SewerMLDataset
from sewer_anomaly.data.preprocessing import (
    ImagePreprocessingConfig,
    ImagePreprocessor,
)
from sewer_anomaly.inference.checkpoint_loader import CheckpointLoader
from sewer_anomaly.models.autoencoder import ConvolutionalAutoencoder


def main() -> None:
    """Load a trained checkpoint and run inference on real Sewer-ML images."""

    checkpoint_path = Path(
        "outputs/checkpoints/entrypoint_smoke_autoencoder.pt"
    )
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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ConvolutionalAutoencoder(
        latent_channels=128,
    ).to(device)

    loader = CheckpointLoader(
        model=model,
        device=device,
    )

    epoch, validation_loss = loader.load(checkpoint_path)

    batch = next(iter(dataloader))
    images = cast(torch.Tensor, batch["image"]).to(
        device,
        non_blocking=True,
    )

    with torch.inference_mode():
        reconstruction = model(images)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint epoch: {epoch}")
    print(f"Checkpoint validation loss: {validation_loss:.6f}")
    print(f"Input shape: {tuple(images.shape)}")
    print(f"Output shape: {tuple(reconstruction.shape)}")
    print(
        "Output range: "
        f"{reconstruction.min().item():.6f} to "
        f"{reconstruction.max().item():.6f}"
    )

    assert reconstruction.shape == images.shape
    assert reconstruction.min().item() >= 0.0
    assert reconstruction.max().item() <= 1.0

    print("Checkpoint inference smoke test passed.")


if __name__ == "__main__":
    main()
