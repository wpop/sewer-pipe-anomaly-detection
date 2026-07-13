from pathlib import Path

from sewer_anomaly.data.normal_split import NormalSplitBuilder


def main() -> None:
    """Create normal-only training and validation manifests from valid00."""

    source_manifest = Path("data/processed/val_normal.csv")
    image_directory = Path("data/raw/sewer_ml/valid00")

    train_output_path = Path(
        "data/interim/valid00_normal_train.csv"
    )
    validation_output_path = Path(
        "data/interim/valid00_normal_val.csv"
    )

    builder = NormalSplitBuilder(
        source_manifest=source_manifest,
        image_directory=image_directory,
        validation_fraction=0.2,
        random_seed=42,
    )

    train_count, validation_count = builder.build(
        train_output_path=train_output_path,
        validation_output_path=validation_output_path,
    )

    print(f"Training samples: {train_count}")
    print(f"Validation samples: {validation_count}")
    print(f"Training manifest: {train_output_path}")
    print(f"Validation manifest: {validation_output_path}")


if __name__ == "__main__":
    main()
