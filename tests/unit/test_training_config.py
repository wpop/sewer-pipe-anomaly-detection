from pathlib import Path

import pytest

from sewer_anomaly.config.training_config import TrainingConfig


def write_config(
    config_path: Path,
    *,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    latent_channels: int = 128,
    num_workers: int = 4,
    image_width: int = 256,
    image_height: int = 256,
) -> None:
    """Write a complete temporary YAML configuration for tests."""

    # Keep every required section present so each test can change one value.
    config_path.write_text(
        f"""
training:
  epochs: {epochs}
  batch_size: {batch_size}
  learning_rate: {learning_rate}
  latent_channels: {latent_channels}
  num_workers: {num_workers}

image:
  width: {image_width}
  height: {image_height}

paths:
  train_manifest: data/processed/train_normal.csv
  validation_manifest: data/processed/val_normal.csv
  train_image_directory: data/raw/sewer_ml/train
  validation_image_directory: data/raw/sewer_ml/valid00
  checkpoint: outputs/checkpoints/best_autoencoder.pt
  history: outputs/metrics/training_history.csv
""".strip(),
        encoding="utf-8",
    )


def test_from_yaml_loads_training_configuration(tmp_path: Path) -> None:
    """Load all training, image, and path values from YAML."""

    config_path = tmp_path / "training.yaml"
    write_config(config_path)

    config = TrainingConfig.from_yaml(config_path)

    assert config.epochs == 20
    assert config.batch_size == 16
    assert config.learning_rate == pytest.approx(0.001)
    assert config.latent_channels == 128
    assert config.num_workers == 4

    assert config.image_width == 256
    assert config.image_height == 256

    assert config.train_manifest == Path(
        "data/processed/train_normal.csv"
    )
    assert config.validation_manifest == Path(
        "data/processed/val_normal.csv"
    )
    assert config.train_image_directory == Path(
        "data/raw/sewer_ml/train"
    )
    assert config.validation_image_directory == Path(
        "data/raw/sewer_ml/valid00"
    )
    assert config.checkpoint_path == Path(
        "outputs/checkpoints/best_autoencoder.pt"
    )
    assert config.history_path == Path(
        "outputs/metrics/training_history.csv"
    )


def test_from_yaml_rejects_missing_file(tmp_path: Path) -> None:
    """Reject a configuration path that does not exist."""

    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(
        FileNotFoundError,
        match="Training configuration not found",
    ):
        TrainingConfig.from_yaml(missing_path)


def test_from_yaml_rejects_invalid_epoch_count(tmp_path: Path) -> None:
    """Reject a zero epoch count."""

    config_path = tmp_path / "training.yaml"
    write_config(config_path, epochs=0)

    with pytest.raises(
        ValueError,
        match="Epoch count must be greater than zero",
    ):
        TrainingConfig.from_yaml(config_path)


def test_from_yaml_rejects_negative_worker_count(
    tmp_path: Path,
) -> None:
    """Reject a negative DataLoader worker count."""

    config_path = tmp_path / "training.yaml"
    write_config(config_path, num_workers=-1)

    with pytest.raises(
        ValueError,
        match="Worker count cannot be negative",
    ):
        TrainingConfig.from_yaml(config_path)


@pytest.mark.parametrize(
    ("image_width", "image_height"),
    [
        (0, 256),
        (256, 0),
        (-1, 256),
        (256, -1),
    ],
)
def test_from_yaml_rejects_invalid_image_dimensions(
    tmp_path: Path,
    image_width: int,
    image_height: int,
) -> None:
    """Reject zero or negative image dimensions."""

    config_path = tmp_path / "training.yaml"

    write_config(
        config_path,
        image_width=image_width,
        image_height=image_height,
    )

    with pytest.raises(
        ValueError,
        match="Image dimensions must be greater than zero",
    ):
        TrainingConfig.from_yaml(config_path)
