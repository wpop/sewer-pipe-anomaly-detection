from pathlib import Path

import pandas as pd


class NormalSplitBuilder:
    """Build deterministic training and validation manifests for normal images."""

    def __init__(
        self,
        source_manifest: Path,
        image_directory: Path,
        validation_fraction: float = 0.2,
        random_seed: int = 42,
    ) -> None:
        if not source_manifest.is_file():
            raise FileNotFoundError(
                f"Source manifest not found: {source_manifest}"
            )

        if not image_directory.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {image_directory}"
            )

        if not 0.0 < validation_fraction < 1.0:
            raise ValueError(
                "Validation fraction must be between zero and one."
            )

        self._source_manifest = source_manifest
        self._image_directory = image_directory
        self._validation_fraction = validation_fraction
        self._random_seed = random_seed

    def build(
        self,
        train_output_path: Path,
        validation_output_path: Path,
    ) -> tuple[int, int]:
        """Create normal-only training and validation manifests."""

        dataframe = pd.read_csv(self._source_manifest)

        if "Filename" not in dataframe.columns:
            raise ValueError(
                "Source manifest must contain a Filename column."
            )

        if "Defect" not in dataframe.columns:
            raise ValueError(
                "Source manifest must contain a Defect column."
            )

        # Keep only normal samples.
        normal_dataframe = dataframe.loc[
            dataframe["Defect"] == 0
        ].copy()

        # valid00 contains only part of the official validation split.
        available_mask = normal_dataframe["Filename"].map(
            lambda filename: (
                self._image_directory / str(filename)
            ).is_file()
        )
        available_dataframe = normal_dataframe.loc[
            available_mask
        ].copy()

        if len(available_dataframe) < 2:
            raise RuntimeError(
                "At least two available normal images are required."
            )

        # Shuffle deterministically before splitting the samples.
        shuffled_dataframe = available_dataframe.sample(
            frac=1.0,
            random_state=self._random_seed,
        ).reset_index(drop=True)

        validation_size = max(
            1,
            round(
                len(shuffled_dataframe)
                * self._validation_fraction
            ),
        )

        validation_dataframe = shuffled_dataframe.iloc[
            :validation_size
        ].copy()
        train_dataframe = shuffled_dataframe.iloc[
            validation_size:
        ].copy()

        if train_dataframe.empty:
            raise RuntimeError(
                "The normal training split cannot be empty."
            )

        train_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        validation_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_dataframe.to_csv(
            train_output_path,
            index=False,
        )
        validation_dataframe.to_csv(
            validation_output_path,
            index=False,
        )

        return len(train_dataframe), len(validation_dataframe)
