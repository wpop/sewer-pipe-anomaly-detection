import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sewer_anomaly.visualization.precision_recall_curve_plotter import (
    PrecisionRecallCurvePlotter,
)
from sewer_anomaly.visualization.roc_curve_plotter import RocCurvePlotter
from sewer_anomaly.visualization.score_distribution_plotter import (
    ScoreDistributionPlotter,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate anomaly detection visualizations from "
            "reconstruction scores and threshold metrics."
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
        "--metrics-json",
        type=Path,
        default=Path(
            "outputs/metrics/"
            "anomaly_threshold_evaluation.json"
        ),
        help="JSON file containing selected anomaly thresholds.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory where visualization PNG files are saved.",
    )

    return parser.parse_args()


def main() -> None:
    """Generate anomaly detection visualizations."""

    arguments = parse_arguments()

    if not arguments.scores_csv.is_file():
        raise FileNotFoundError(
            f"Scores CSV not found: {arguments.scores_csv}"
        )

    if not arguments.metrics_json.is_file():
        raise FileNotFoundError(
            f"Metrics JSON not found: {arguments.metrics_json}"
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

    with arguments.metrics_json.open(
        "r",
        encoding="utf-8",
    ) as metrics_file:
        metrics = json.load(metrics_file)

    f1_threshold = float(
        metrics["maximum_f1"]["calibration"]["threshold"]
    )

    balanced_threshold = float(
        metrics["maximum_balanced_accuracy"]
        ["calibration"]
        ["threshold"]
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    distribution_output = (
        arguments.output_directory
        / "reconstruction_score_distribution.png"
    )

    roc_output = (
        arguments.output_directory
        / "roc_curve.png"
    )

    precision_recall_output = (
        arguments.output_directory
        / "precision_recall_curve.png"
    )

    distribution_plotter = ScoreDistributionPlotter()

    distribution_plotter.save(
        labels=labels,
        scores=scores,
        output_path=distribution_output,
        f1_threshold=f1_threshold,
        balanced_threshold=balanced_threshold,
    )

    roc_plotter = RocCurvePlotter()

    roc_plotter.save(
        labels=labels,
        scores=scores,
        output_path=roc_output,
    )

    precision_recall_plotter = PrecisionRecallCurvePlotter()

    precision_recall_plotter.save(
        labels=labels,
        scores=scores,
        output_path=precision_recall_output,
    )

    print("Anomaly visualizations generated.")
    print(f"Samples:          {labels.size}")
    print(f"Distribution:     {distribution_output}")
    print(f"ROC curve:        {roc_output}")
    print(f"Precision-recall: {precision_recall_output}")


if __name__ == "__main__":
    main()
