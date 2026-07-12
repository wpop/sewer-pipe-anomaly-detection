from pathlib import Path
from typing import cast

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from sewer_anomaly.data.dataset import SewerMLDataset
from sewer_anomaly.data.preprocessing import (
    ImagePreprocessingConfig,
    ImagePreprocessor,
)
from sewer_anomaly.models.autoencoder import ConvolutionalAutoencoder
from sewer_anomaly.training.autoencoder_trainer import AutoencoderTrainer


def main() -> None:
    """Run a short autoencoder training smoke test on real Sewer-ML images."""

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
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvolutionalAutoencoder(
        latent_channels=128,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )

    trainer = AutoencoderTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Dataset size: {len(dataset)}")

    validation_loss_before = trainer.validate_epoch(dataloader)
    training_loss = trainer.train_epoch(dataloader)
    validation_loss_after = trainer.validate_epoch(dataloader)

    print(f"Validation loss before training: {validation_loss_before:.6f}")
    print(f"Training loss: {training_loss:.6f}")
    print(f"Validation loss after training: {validation_loss_after:.6f}")

    batch = next(iter(dataloader))
    images = cast(torch.Tensor, batch["image"]).to(
        device,
        non_blocking=True,
    )

    model.eval()

    with torch.inference_mode():
        reconstruction = model(images)

    print(f"Input shape: {tuple(images.shape)}")
    print(f"Output shape: {tuple(reconstruction.shape)}")
    print(
        "Output range: "
        f"{reconstruction.min().item():.6f} to "
        f"{reconstruction.max().item():.6f}"
    )

    assert reconstruction.shape == images.shape
    assert training_loss > 0.0
    assert validation_loss_before > 0.0
    assert validation_loss_after > 0.0

    print("Autoencoder training smoke test passed.")


if __name__ == "__main__":
    main()
