from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import cast

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from sewer_anomaly.config.training_config import TrainingConfig
from sewer_anomaly.data.dataset import SewerMLDataset, SewerMLSample
from sewer_anomaly.data.preprocessing import (
    ImagePreprocessingConfig,
    ImagePreprocessor,
)
from sewer_anomaly.models.autoencoder import ConvolutionalAutoencoder
from sewer_anomaly.training.autoencoder_trainer import AutoencoderTrainer
from sewer_anomaly.training.training_runner import TrainingRunner


class ParsedArguments(Namespace):
    """Describe the parsed command-line arguments."""

    config: Path


def parse_arguments() -> Path:
    """Parse the training configuration path."""

    parser = ArgumentParser(
        description="Train the Sewer-ML convolutional autoencoder."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_autoencoder.yaml"),
        help="Path to the YAML training configuration.",
    )

    arguments = cast(ParsedArguments, parser.parse_args())
    return arguments.config


def create_dataloader(
    manifest_path: Path,
    image_directory: Path,
    preprocessor: ImagePreprocessor,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader[SewerMLSample]:
    """Create a configured Sewer-ML DataLoader."""

    dataset = SewerMLDataset(
        manifest_path=manifest_path,
        image_directory=image_directory,
        preprocessor=preprocessor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    """Train the autoencoder using a YAML configuration."""

    config_path = parse_arguments()
    config = TrainingConfig.from_yaml(config_path)

    preprocessor = ImagePreprocessor(
        ImagePreprocessingConfig(
            image_width=config.image_width,
            image_height=config.image_height,
        )
    )

    train_dataset = SewerMLDataset(
        manifest_path=config.train_manifest,
        image_directory=config.train_image_directory,
        preprocessor=preprocessor,
    )
    train_dataloader = DataLoader[SewerMLSample](
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    validation_dataset = SewerMLDataset(
        manifest_path=config.validation_manifest,
        image_directory=config.validation_image_directory,
        preprocessor=preprocessor,
    )
    validation_dataloader = DataLoader[SewerMLSample](
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ConvolutionalAutoencoder(
        latent_channels=config.latent_channels,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    trainer = AutoencoderTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
    )

    runner = TrainingRunner(
        trainer=trainer,
        model=model,
        checkpoint_path=config.checkpoint_path,
        history_path=config.history_path,
    )

    print(f"Configuration: {config_path}")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Training samples: {len(train_dataset)}")
    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    runner.run(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        epochs=config.epochs,
    )

    print(f"Best checkpoint: {config.checkpoint_path}")
    print(f"Training history: {config.history_path}")


if __name__ == "__main__":
    main()
