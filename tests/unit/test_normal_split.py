from pathlib import Path

import pandas as pd
import pytest

from sewer_anomaly.data.normal_split import NormalSplitBuilder


def create_image_files(
    image_directory: Path,
    filenames: list[str],
) -> None:
    """Create placeholder image files for split tests."""

    image_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # The split builder only checks whether each file exists.
    for filename in filenames:
        image_path = image_directory / filename
        image_path.write_bytes(b"test-image")


def create_manifest(
    manifest_path: Path,
) -> None:
    """Create a manifest with normal, defective, and missing samples."""

    dataframe = pd.DataFrame(
        {
            "Filename": [
                "normal_1.png",
                "normal_2.png",
                "normal_3.png",
                "normal_4.png",
                "missing_normal.png",
                "defective.png",
            ],
            "Defect": [
                0,
                0,
                0,
                0,
                0,
                1,
            ],
        }
    )

    dataframe.to_csv(
        manifest_path,
        index=False,
    )


def test_build_creates_normal_only_split(
    tmp_path: Path,
) -> None:
    """Create training and validation manifests from available normal images."""

    manifest_path = tmp_path / "source.csv"
    image_directory = tmp_path / "images"

    create_manifest(manifest_path)

    create_image_files(
        image_directory,
        [
            "normal_1.png",
            "normal_2.png",
            "normal_3.png",
            "normal_4.png",
            "defective.png",
        ],
    )

    train_output_path = tmp_path / "output" / "train.csv"
    validation_output_path = tmp_path / "output" / "validation.csv"

    builder = NormalSplitBuilder(
        source_manifest=manifest_path,
        image_directory=image_directory,
        validation_fraction=0.25,
        random_seed=42,
    )

    train_count, validation_count = builder.build(
        train_output_path=train_output_path,
        validation_output_path=validation_output_path,
    )

    train_dataframe = pd.read_csv(train_output_path)
    validation_dataframe = pd.read_csv(validation_output_path)

    assert train_count == 3
    assert validation_count == 1

    assert len(train_dataframe) == 3
    assert len(validation_dataframe) == 1

    assert train_dataframe["Defect"].eq(0).all()
    assert validation_dataframe["Defect"].eq(0).all()

    combined_filenames = set(
        train_dataframe["Filename"].tolist()
        + validation_dataframe["Filename"].tolist()
    )

    assert "defective.png" not in combined_filenames
    assert "missing_normal.png" not in combined_filenames

    assert combined_filenames == {
        "normal_1.png",
        "normal_2.png",
        "normal_3.png",
        "normal_4.png",
    }


def test_build_is_reproducible_with_same_seed(
    tmp_path: Path,
) -> None:
    """Create identical splits when the random seed is unchanged."""

    manifest_path = tmp_path / "source.csv"
    image_directory = tmp_path / "images"

    create_manifest(manifest_path)

    create_image_files(
        image_directory,
        [
            "normal_1.png",
            "normal_2.png",
            "normal_3.png",
            "normal_4.png",
        ],
    )

    first_builder = NormalSplitBuilder(
        source_manifest=manifest_path,
        image_directory=image_directory,
        validation_fraction=0.5,
        random_seed=123,
    )

    second_builder = NormalSplitBuilder(
        source_manifest=manifest_path,
        image_directory=image_directory,
        validation_fraction=0.5,
        random_seed=123,
    )

    first_train_path = tmp_path / "first_train.csv"
    first_validation_path = tmp_path / "first_validation.csv"

    second_train_path = tmp_path / "second_train.csv"
    second_validation_path = tmp_path / "second_validation.csv"

    first_builder.build(
        train_output_path=first_train_path,
        validation_output_path=first_validation_path,
    )

    second_builder.build(
        train_output_path=second_train_path,
        validation_output_path=second_validation_path,
    )

    first_train = pd.read_csv(first_train_path)
    second_train = pd.read_csv(second_train_path)

    first_validation = pd.read_csv(first_validation_path)
    second_validation = pd.read_csv(second_validation_path)

    pd.testing.assert_frame_equal(
        first_train,
        second_train,
    )
    pd.testing.assert_frame_equal(
        first_validation,
        second_validation,
    )


@pytest.mark.parametrize(
    "validation_fraction",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_constructor_rejects_invalid_validation_fraction(
    tmp_path: Path,
    validation_fraction: float,
) -> None:
    """Reject validation fractions outside the open interval zero to one."""

    manifest_path = tmp_path / "source.csv"
    image_directory = tmp_path / "images"

    create_manifest(manifest_path)
    image_directory.mkdir()

    with pytest.raises(
        ValueError,
        match="Validation fraction must be between zero and one",
    ):
        NormalSplitBuilder(
            source_manifest=manifest_path,
            image_directory=image_directory,
            validation_fraction=validation_fraction,
        )


def test_constructor_rejects_missing_manifest(
    tmp_path: Path,
) -> None:
    """Reject a source manifest that does not exist."""

    image_directory = tmp_path / "images"
    image_directory.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match="Source manifest not found",
    ):
        NormalSplitBuilder(
            source_manifest=tmp_path / "missing.csv",
            image_directory=image_directory,
        )


def test_constructor_rejects_missing_image_directory(
    tmp_path: Path,
) -> None:
    """Reject an image directory that does not exist."""

    manifest_path = tmp_path / "source.csv"
    create_manifest(manifest_path)

    with pytest.raises(
        FileNotFoundError,
        match="Image directory not found",
    ):
        NormalSplitBuilder(
            source_manifest=manifest_path,
            image_directory=tmp_path / "missing_images",
        )


def test_build_requires_at_least_two_available_normal_images(
    tmp_path: Path,
) -> None:
    """Reject a split when fewer than two normal images are available."""

    manifest_path = tmp_path / "source.csv"
    image_directory = tmp_path / "images"

    create_manifest(manifest_path)

    create_image_files(
        image_directory,
        ["normal_1.png"],
    )

    builder = NormalSplitBuilder(
        source_manifest=manifest_path,
        image_directory=image_directory,
    )

    with pytest.raises(
        RuntimeError,
        match="At least two available normal images are required",
    ):
        builder.build(
            train_output_path=tmp_path / "train.csv",
            validation_output_path=tmp_path / "validation.csv",
        )
