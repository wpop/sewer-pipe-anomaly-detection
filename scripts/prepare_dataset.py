from pathlib import Path

from sewer_anomaly.data.manifests import DatasetManifestBuilder


def main() -> None:
    """Create processed dataset manifests from official Sewer-ML metadata."""

    # Store official Sewer-ML metadata outside the source package.
    external_directory = Path("data/external")

    # Write generated manifests to the processed data directory.
    processed_directory = Path("data/processed")

    builder = DatasetManifestBuilder(processed_directory)

    output_paths = builder.build(
        train_csv=external_directory / "SewerML_Train.csv",
        validation_csv=external_directory / "SewerML_Val.csv",
        test_csv=external_directory / "SewerML_Test.csv",
    )

    print("Dataset manifests created:")

    for name, path in output_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
