from pathlib import Path

import pandas as pd
import pytest

from sewer_anomaly.data.metadata import SewerMLMetadata


def test_load_valid_metadata(tmp_path: Path) -> None:
    """Load valid labeled metadata."""
    csv_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "Filename": ["normal.png", "defect.png"],
            "WaterLevel": [0, 10],
            "Defect": [0, 1],
        }
    ).to_csv(csv_path, index=False)

    metadata = SewerMLMetadata(csv_path)
    dataframe = metadata.load()

    assert len(dataframe) == 2
    assert metadata.is_labeled()


def test_select_normal_and_defective_images(tmp_path: Path) -> None:
    """Separate normal and defective metadata rows."""
    csv_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "Filename": ["normal.png", "defect_1.png", "defect_2.png"],
            "WaterLevel": [0, 10, 20],
            "Defect": [0, 1, 1],
        }
    ).to_csv(csv_path, index=False)

    metadata = SewerMLMetadata(csv_path)
    metadata.load()

    # Verify that binary defect labels produce the expected subsets.
    assert metadata.normal_images()["Filename"].tolist() == ["normal.png"]
    assert metadata.defective_images()["Filename"].tolist() == [
        "defect_1.png",
        "defect_2.png",
    ]


def test_missing_file_raises_error(tmp_path: Path) -> None:
    """Reject a metadata path that does not exist."""
    metadata = SewerMLMetadata(tmp_path / "missing.csv")

    with pytest.raises(FileNotFoundError):
        metadata.load()


def test_missing_filename_column_raises_error(tmp_path: Path) -> None:
    """Reject metadata without the required filename column."""
    csv_path = tmp_path / "metadata.csv"
    pd.DataFrame({"Defect": [0, 1]}).to_csv(csv_path, index=False)

    metadata = SewerMLMetadata(csv_path)

    with pytest.raises(ValueError, match="Missing required columns"):
        metadata.load()


def test_unlabeled_metadata_is_detected(tmp_path: Path) -> None:
    """Detect metadata without training or validation labels."""
    csv_path = tmp_path / "test_metadata.csv"
    pd.DataFrame({"Filename": ["image.png"]}).to_csv(csv_path, index=False)

    metadata = SewerMLMetadata(csv_path)
    metadata.load()

    assert not metadata.is_labeled()

    with pytest.raises(ValueError, match="does not contain labels"):
        metadata.normal_images()
