from pathlib import Path

import pandas as pd


class SewerMLMetadata:
    """Load and validate Sewer-ML metadata."""

    REQUIRED_COLUMNS = {"Filename"}
    LABEL_COLUMNS = {"WaterLevel", "Defect"}

    def __init__(self, csv_path: Path) -> None:
        # Store the metadata path for later loading.
        self._csv_path = csv_path

        # Keep the loaded DataFrame inside the object.
        self._dataframe: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Load metadata from CSV and validate its structure."""
        if not self._csv_path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {self._csv_path}")

        dataframe = pd.read_csv(self._csv_path)
        self._validate(dataframe)

        self._dataframe = dataframe
        return dataframe

    def is_labeled(self) -> bool:
        """Return whether the metadata contains labels."""
        dataframe = self._require_loaded()
        return self.LABEL_COLUMNS.issubset(dataframe.columns)

    def normal_images(self) -> pd.DataFrame:
        """Return metadata rows for normal pipe images."""
        dataframe = self._require_labeled()

        # Select rows without any labeled defect.
        normal: pd.DataFrame = dataframe.loc[dataframe["Defect"].eq(0)].copy()
        return normal

    def defective_images(self) -> pd.DataFrame:
        """Return metadata rows for defective pipe images."""
        dataframe = self._require_labeled()

        # Select rows containing at least one labeled defect.
        defective: pd.DataFrame = dataframe.loc[dataframe["Defect"].eq(1)].copy()
        return defective

    def _validate(self, dataframe: pd.DataFrame) -> None:
        """Validate required columns and non-empty metadata."""
        missing_columns = self.REQUIRED_COLUMNS - set(dataframe.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        if dataframe.empty:
            raise ValueError("Metadata file is empty.")

    def _require_loaded(self) -> pd.DataFrame:
        """Return loaded metadata or raise an error."""
        if self._dataframe is None:
            raise RuntimeError("Metadata must be loaded first.")

        return self._dataframe

    def _require_labeled(self) -> pd.DataFrame:
        """Return labeled metadata or raise an error."""
        dataframe = self._require_loaded()

        if not self.is_labeled():
            raise ValueError("Metadata does not contain labels.")

        return dataframe
