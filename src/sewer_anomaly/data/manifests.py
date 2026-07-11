from pathlib import Path

from sewer_anomaly.data.metadata import SewerMLMetadata


class DatasetManifestBuilder:
    """Build CSV manifests for model training and evaluation."""

    def __init__(self, output_directory: Path) -> None:
        # Store the destination for generated manifest files.
        self._output_directory = output_directory

    def build(
        self,
        train_csv: Path,
        validation_csv: Path,
        test_csv: Path,
    ) -> dict[str, Path]:
        """Build the primary dataset manifests."""

        # Create the output directory when it does not exist.
        self._output_directory.mkdir(parents=True, exist_ok=True)

        # The autoencoder must be trained exclusively on normal images.
        train_metadata = SewerMLMetadata(train_csv)
        train_metadata.load()
        train_normal = train_metadata.normal_images()

        # Keep normal and defective validation samples separate.
        validation_metadata = SewerMLMetadata(validation_csv)
        validation_metadata.load()
        validation_normal = validation_metadata.normal_images()
        validation_defective = validation_metadata.defective_images()

        # Official test metadata contains filenames without defect labels.
        test_metadata = SewerMLMetadata(test_csv)
        test_dataframe = test_metadata.load()

        output_paths = {
            "train_normal": self._output_directory / "train_normal.csv",
            "validation_normal": self._output_directory / "val_normal.csv",
            "validation_defective": self._output_directory / "val_defective.csv",
            "test": self._output_directory / "test.csv",
        }

        # Do not write the Pandas index as an extra CSV column.
        train_normal.to_csv(output_paths["train_normal"], index=False)
        validation_normal.to_csv(output_paths["validation_normal"], index=False)
        validation_defective.to_csv(
            output_paths["validation_defective"],
            index=False,
        )
        test_dataframe.to_csv(output_paths["test"], index=False)

        return output_paths
