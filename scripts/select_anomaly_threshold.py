import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sewer_anomaly.evaluation.threshold_selector import ThresholdSelector


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Select an anomaly threshold from reconstruction scores."
        )
    )

    parser.add_argument(
        "--scores-csv",
        type=Path,
        default=Path(
            "outputs/predictions/"
            "normal_only_reconstruction_scores.csv"
        ),
        help="CSV file containing labels and reconstruction scores.",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/metrics/anomaly_threshold.json"
        ),
        help="Output JSON file for threshold selection results.",
    )

    return parser.parse_args()


def main() -> None:
    """Select and save an anomaly threshold."""

    arguments = parse_arguments()

    if not arguments.scores_csv.is_file():
        raise FileNotFoundError(
            f"Scores CSV not found: {arguments.scores_csv}"
        )

    dataframe = pd.read_csv(arguments.scores_csv)

    required_columns = {
        "label",
        "reconstruction_score",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        missing_columns_text = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            "Scores CSV is missing required columns: "
            f"{missing_columns_text}"
        )

    labels = np.asarray(
        dataframe["label"].to_numpy(),
        dtype=np.int64,
    )
    scores = np.asarray(
        dataframe["reconstruction_score"].to_numpy(),
        dtype=np.float64,
    )

    selector = ThresholdSelector()

    result = selector.select(
        labels=labels,
        scores=scores,
    )

    arguments.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = {
        "source_csv": str(arguments.scores_csv),
        "sample_count": int(labels.size),
        "normal_count": int(np.count_nonzero(labels == 0)),
        "defective_count": int(
            np.count_nonzero(labels == 1)
        ),
        "threshold": result.threshold,
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "roc_auc": result.roc_auc,
    }

    with arguments.output_json.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            output_data,
            output_file,
            indent=2,
        )
        output_file.write("\n")

    print("Anomaly threshold selection completed.")
    print(f"Samples:   {labels.size}")
    print(f"Threshold: {result.threshold:.8f}")
    print(f"Precision: {result.precision:.6f}")
    print(f"Recall:    {result.recall:.6f}")
    print(f"F1 score:  {result.f1_score:.6f}")
    print(f"ROC-AUC:   {result.roc_auc:.6f}")
    print(f"Saved to:  {arguments.output_json}")


if __name__ == "__main__":
    main()
