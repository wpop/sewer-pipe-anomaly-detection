from pathlib import Path

from sewer_anomaly.data.metadata import SewerMLMetadata


def print_split_summary(name: str, csv_path: Path) -> None:
    """Print basic statistics for a Sewer-ML metadata split."""
    metadata = SewerMLMetadata(csv_path)
    dataframe = metadata.load()

    print(f"\n{name}")
    print(f"Total images: {len(dataframe):,}")

    if metadata.is_labeled():
        normal_count = len(metadata.normal_images())
        defective_count = len(metadata.defective_images())
        defective_ratio = defective_count / len(dataframe)

        print(f"Normal images: {normal_count:,}")
        print(f"Defective images: {defective_count:,}")
        print(f"Defective ratio: {defective_ratio:.2%}")
    else:
        print("Labels: unavailable")


def main() -> None:
    """Analyse Sewer-ML metadata splits."""
    data_directory = Path("data/external")

    print_split_summary(
        "Training split",
        data_directory / "SewerML_Train.csv",
    )
    print_split_summary(
        "Validation split",
        data_directory / "SewerML_Val.csv",
    )
    print_split_summary(
        "Test split",
        data_directory / "SewerML_Test.csv",
    )


if __name__ == "__main__":
    main()
