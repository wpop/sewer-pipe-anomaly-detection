import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sewer_anomaly.evaluation.balanced_threshold_selector import (
    BalancedThresholdSelector,
)
from sewer_anomaly.evaluation.threshold_evaluator import ThresholdEvaluator
from sewer_anomaly.evaluation.threshold_selector import ThresholdSelector


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare F1 and balanced-accuracy anomaly thresholds "
            "on an independent evaluation split."
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
            "outputs/metrics/anomaly_threshold_evaluation.json"
        ),
        help="Output JSON file for threshold evaluation metrics.",
    )

    parser.add_argument(
        "--evaluation-size",
        type=float,
        default=0.5,
        help="Fraction of samples reserved for independent evaluation.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for the stratified split.",
    )

    return parser.parse_args()


def main() -> None:
    """Compare F1 and balanced-accuracy anomaly thresholds."""

    arguments = parse_arguments()

    if not arguments.scores_csv.is_file():
        raise FileNotFoundError(
            f"Scores CSV not found: {arguments.scores_csv}"
        )

    if not 0.0 < arguments.evaluation_size < 1.0:
        raise ValueError(
            "Evaluation size must be between 0 and 1."
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

    (
        calibration_scores,
        evaluation_scores,
        calibration_labels,
        evaluation_labels,
    ) = train_test_split(
        scores,
        labels,
        test_size=arguments.evaluation_size,
        random_state=arguments.random_seed,
        stratify=labels,
    )

    f1_selector = ThresholdSelector()

    f1_selection_result = f1_selector.select(
        labels=calibration_labels,
        scores=calibration_scores,
    )

    balanced_selector = BalancedThresholdSelector()

    balanced_selection_result = balanced_selector.select(
        labels=calibration_labels,
        scores=calibration_scores,
    )

    evaluator = ThresholdEvaluator()

    f1_evaluation_result = evaluator.evaluate(
        labels=evaluation_labels,
        scores=evaluation_scores,
        threshold=f1_selection_result.threshold,
    )

    balanced_evaluation_result = evaluator.evaluate(
        labels=evaluation_labels,
        scores=evaluation_scores,
        threshold=balanced_selection_result.threshold,
    )

    f1_balanced_accuracy = (
        f1_evaluation_result.recall
        + f1_evaluation_result.specificity
    ) / 2.0

    balanced_evaluation_accuracy = (
        balanced_evaluation_result.recall
        + balanced_evaluation_result.specificity
    ) / 2.0

    arguments.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = {
        "source_csv": str(arguments.scores_csv),
        "random_seed": arguments.random_seed,
        "evaluation_size": arguments.evaluation_size,
        "calibration_sample_count": int(
            calibration_labels.size
        ),
        "evaluation_sample_count": int(
            evaluation_labels.size
        ),
        "evaluation_normal_count": int(
            np.count_nonzero(evaluation_labels == 0)
        ),
        "evaluation_defective_count": int(
            np.count_nonzero(evaluation_labels == 1)
        ),
        "maximum_f1": {
            "calibration": {
                "threshold": f1_selection_result.threshold,
                "precision": f1_selection_result.precision,
                "recall": f1_selection_result.recall,
                "f1_score": f1_selection_result.f1_score,
                "roc_auc": f1_selection_result.roc_auc,
            },
            "evaluation": {
                "threshold": f1_evaluation_result.threshold,
                "precision": f1_evaluation_result.precision,
                "recall": f1_evaluation_result.recall,
                "specificity": f1_evaluation_result.specificity,
                "balanced_accuracy": f1_balanced_accuracy,
                "f1_score": f1_evaluation_result.f1_score,
                "roc_auc": f1_evaluation_result.roc_auc,
                "pr_auc": f1_evaluation_result.pr_auc,
                "true_negative": (
                    f1_evaluation_result.true_negative
                ),
                "false_positive": (
                    f1_evaluation_result.false_positive
                ),
                "false_negative": (
                    f1_evaluation_result.false_negative
                ),
                "true_positive": (
                    f1_evaluation_result.true_positive
                ),
            },
        },
        "maximum_balanced_accuracy": {
            "calibration": {
                "threshold": (
                    balanced_selection_result.threshold
                ),
                "recall": balanced_selection_result.recall,
                "specificity": (
                    balanced_selection_result.specificity
                ),
                "balanced_accuracy": (
                    balanced_selection_result.balanced_accuracy
                ),
                "roc_auc": balanced_selection_result.roc_auc,
            },
            "evaluation": {
                "threshold": (
                    balanced_evaluation_result.threshold
                ),
                "precision": (
                    balanced_evaluation_result.precision
                ),
                "recall": balanced_evaluation_result.recall,
                "specificity": (
                    balanced_evaluation_result.specificity
                ),
                "balanced_accuracy": (
                    balanced_evaluation_accuracy
                ),
                "f1_score": (
                    balanced_evaluation_result.f1_score
                ),
                "roc_auc": (
                    balanced_evaluation_result.roc_auc
                ),
                "pr_auc": balanced_evaluation_result.pr_auc,
                "true_negative": (
                    balanced_evaluation_result.true_negative
                ),
                "false_positive": (
                    balanced_evaluation_result.false_positive
                ),
                "false_negative": (
                    balanced_evaluation_result.false_negative
                ),
                "true_positive": (
                    balanced_evaluation_result.true_positive
                ),
            },
        },
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

    print("Anomaly threshold comparison completed.")
    print(
        f"Calibration samples: {calibration_labels.size}"
    )
    print(
        f"Evaluation samples:  {evaluation_labels.size}"
    )

    print("\nMaximum F1 threshold")
    print(
        f"Threshold:           "
        f"{f1_selection_result.threshold:.8f}"
    )
    print(
        f"Precision:           "
        f"{f1_evaluation_result.precision:.6f}"
    )
    print(
        f"Recall:              "
        f"{f1_evaluation_result.recall:.6f}"
    )
    print(
        f"Specificity:         "
        f"{f1_evaluation_result.specificity:.6f}"
    )
    print(
        f"Balanced accuracy:   "
        f"{f1_balanced_accuracy:.6f}"
    )
    print(
        f"F1 score:            "
        f"{f1_evaluation_result.f1_score:.6f}"
    )

    print("\nMaximum balanced-accuracy threshold")
    print(
        f"Threshold:           "
        f"{balanced_selection_result.threshold:.8f}"
    )
    print(
        f"Precision:           "
        f"{balanced_evaluation_result.precision:.6f}"
    )
    print(
        f"Recall:              "
        f"{balanced_evaluation_result.recall:.6f}"
    )
    print(
        f"Specificity:         "
        f"{balanced_evaluation_result.specificity:.6f}"
    )
    print(
        f"Balanced accuracy:   "
        f"{balanced_evaluation_accuracy:.6f}"
    )
    print(
        f"F1 score:            "
        f"{balanced_evaluation_result.f1_score:.6f}"
    )

    print(
        f"\nROC-AUC:             "
        f"{balanced_evaluation_result.roc_auc:.6f}"
    )
    print(
        f"PR-AUC:              "
        f"{balanced_evaluation_result.pr_auc:.6f}"
    )
    print(
        f"Saved to:            {arguments.output_json}"
    )


if __name__ == "__main__":
    main()
