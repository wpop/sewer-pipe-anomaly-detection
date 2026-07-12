from pathlib import Path

import pandas as pd
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
from sewer_anomaly.training.training_runner import TrainingRunner


def main() -> None:
    """Run a short end-to-end training runner smoke test."""

    manifest_path = Path("data/interim/valid00_inspection.csv")
    image_directory = Path("data/raw/sewer_ml/valid00")

    # Keep smoke-test artifacts separate from full training outputs.
    checkpoint_path = Path(
        "outputs/checkpoints/smoke_best_autoencoder.pt"
    )
    history_path = Path(
        "outputs/metrics/smoke_training_history.csv"
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

    # Reuse the small inspection dataset for this runtime smoke test.
    train_dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    validation_dataloader = DataLoader(
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

    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )

    trainer = AutoencoderTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
    )

    runner = TrainingRunner(
        trainer=trainer,
        model=model,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Dataset size: {len(dataset)}")

    history = runner.run(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        epochs=3,
    )

    # Confirm that both persistent training artifacts were created.
    if not checkpoint_path.is_file():
        raise RuntimeError("Training checkpoint was not created.")

    if not history_path.is_file():
        raise RuntimeError("Training history CSV was not created.")

    history_dataframe = pd.read_csv(history_path)

    print(f"History records: {len(history)}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"History saved to: {history_path}")
    print(history_dataframe.to_string(index=False))

    assert len(history) == 3
    assert len(history_dataframe) == 3

    print("Training runner smoke test passed.")


if __name__ == "__main__":
    main()
