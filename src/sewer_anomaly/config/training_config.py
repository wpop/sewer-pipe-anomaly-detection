from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    """Store autoencoder training configuration values."""

    epochs: int
    batch_size: int
    learning_rate: float
    latent_channels: int
    num_workers: int

    image_width: int
    image_height: int

    train_manifest: Path
    validation_manifest: Path
    train_image_directory: Path
    validation_image_directory: Path
    checkpoint_path: Path
    history_path: Path

    @classmethod
    def from_yaml(cls, config_path: Path) -> "TrainingConfig":
        """Load and validate training configuration from YAML."""

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Training configuration not found: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as file:
            loaded_data = yaml.safe_load(file)

        # PyYAML returns Any because YAML content has no fixed schema.
        data = cast(dict[str, Any], loaded_data)

        training = cast(dict[str, Any], data["training"])
        image = cast(dict[str, Any], data["image"])
        paths = cast(dict[str, Any], data["paths"])

        config = cls(
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            latent_channels=int(training["latent_channels"]),
            num_workers=int(training["num_workers"]),
            image_width=int(image["width"]),
            image_height=int(image["height"]),
            train_manifest=Path(str(paths["train_manifest"])),
            validation_manifest=Path(
                str(paths["validation_manifest"])
            ),
            train_image_directory=Path(
                str(paths["train_image_directory"])
            ),
            validation_image_directory=Path(
                str(paths["validation_image_directory"])
            ),
            checkpoint_path=Path(str(paths["checkpoint"])),
            history_path=Path(str(paths["history"])),
        )

        config._validate()

        return config

    def _validate(self) -> None:
        """Validate numeric configuration values."""

        if self.epochs <= 0:
            raise ValueError("Epoch count must be greater than zero.")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        if self.learning_rate <= 0.0:
            raise ValueError("Learning rate must be greater than zero.")

        if self.latent_channels <= 0:
            raise ValueError("Latent channels must be greater than zero.")

        if self.num_workers < 0:
            raise ValueError("Worker count cannot be negative.")

        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be greater than zero.")
