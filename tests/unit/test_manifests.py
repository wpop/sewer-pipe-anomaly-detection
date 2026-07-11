from pathlib import Path

import pandas as pd

from sewer_anomaly.data.manifests import DatasetManifestBuilder


def test_build_dataset_manifests(tmp_path: Path) -> None:
    """Build and validate all dataset manifest files."""

    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"
    test_csv = tmp_path / "test.csv"
    output_directory = tmp_path / "manifests"

    # Include both normal and defective training samples.
    pd.DataFrame(
        {
            "Filename": ["train_normal.png", "train_defect.png"],
            "WaterLevel": [0, 10],
            "Defect": [0, 1],
        }
    ).to_csv(train_csv, index=False)

    # Include both normal and defective validation samples.
    pd.DataFrame(
        {
            "Filename": ["val_normal.png", "val_defect.png"],
            "WaterLevel": [0, 20],
            "Defect": [0, 1],
        }
    ).to_csv(validation_csv, index=False)

    # Official test metadata contains filenames only.
    pd.DataFrame(
        {
            "Filename": ["test_1.png", "test_2.png"],
        }
    ).to_csv(test_csv, index=False)

    builder = DatasetManifestBuilder(output_directory)

    output_paths = builder.build(
        train_csv=train_csv,
        validation_csv=validation_csv,
        test_csv=test_csv,
    )

    # Confirm that every expected manifest was created.
    for path in output_paths.values():
        assert path.is_file()

    train_normal = pd.read_csv(output_paths["train_normal"])
    validation_normal = pd.read_csv(output_paths["validation_normal"])
    validation_defective = pd.read_csv(output_paths["validation_defective"])
    test_dataframe = pd.read_csv(output_paths["test"])

    # Verify that training contains normal images only.
    assert train_normal["Filename"].tolist() == ["train_normal.png"]

    # Verify that validation data is separated by defect status.
    assert validation_normal["Filename"].tolist() == ["val_normal.png"]
    assert validation_defective["Filename"].tolist() == ["val_defect.png"]

    # Verify that test filenames are preserved.
    assert test_dataframe["Filename"].tolist() == [
        "test_1.png",
        "test_2.png",
    ]

def test_rebuild_overwrites_existing_manifests(tmp_path: Path) -> None:
    """Rebuild manifests without creating duplicate rows."""

    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"
    test_csv = tmp_path / "test.csv"
    output_directory = tmp_path / "manifests"

    # Create a minimal labeled training split.
    pd.DataFrame(
        {
            "Filename": ["train_normal.png"],
            "WaterLevel": [0],
            "Defect": [0],
        }
    ).to_csv(train_csv, index=False)

    # Create normal and defective validation samples.
    pd.DataFrame(
        {
            "Filename": ["val_normal.png", "val_defect.png"],
            "WaterLevel": [0, 10],
            "Defect": [0, 1],
        }
    ).to_csv(validation_csv, index=False)

    # Create unlabeled test metadata.
    pd.DataFrame(
        {
            "Filename": ["test.png"],
        }
    ).to_csv(test_csv, index=False)

    builder = DatasetManifestBuilder(output_directory)

    # Build the same manifests twice.
    first_paths = builder.build(
        train_csv=train_csv,
        validation_csv=validation_csv,
        test_csv=test_csv,
    )
    second_paths = builder.build(
        train_csv=train_csv,
        validation_csv=validation_csv,
        test_csv=test_csv,
    )

    # Rebuilding should preserve paths and overwrite files cleanly.
    assert first_paths == second_paths

    train_normal = pd.read_csv(second_paths["train_normal"])
    assert train_normal["Filename"].tolist() == ["train_normal.png"]
