# Sewer Pipe Anomaly Detection

Industrial sewer pipe defect detection using convolutional autoencoders and reconstruction-based anomaly detection.

## Project Overview

This project develops an end-to-end machine learning pipeline for detecting anomalies in sewer pipe inspection images using the Sewer-ML dataset.

The primary model is a convolutional autoencoder trained only on normal pipe images. During inference, images with high reconstruction error are classified as anomalous.

The project is designed as a professional portfolio project for industrial computer vision, machine learning, and data science roles.

## Objectives

* Prepare and validate the Sewer-ML dataset.
* Build a reproducible image preprocessing pipeline.
* Train a convolutional autoencoder using normal pipe images.
* Detect anomalous pipe images using reconstruction error.
* Generate anomaly heatmaps.
* Select an anomaly threshold using validation data.
* Evaluate the model using precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrices.
* Analyse false positives and false negatives.
* Compare multiple anomaly scoring methods.
* Provide reproducible configuration, testing, documentation, and deployment support.

## Planned Technologies

* Python
* PyTorch
* OpenCV
* NumPy
* Pandas
* SciPy
* scikit-learn
* Matplotlib
* PyYAML
* pytest
* Ruff
* mypy
* Docker
* GitHub Actions

## Project Structure

```text
sewer-pipe-anomaly-detection/
├── .github/
│   └── workflows/
├── configs/
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── docker/
├── docs/
│   └── images/
├── notebooks/
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   ├── metrics/
│   └── predictions/
├── scripts/
├── src/
│   └── sewer_anomaly/
│       ├── config/
│       ├── data/
│       ├── evaluation/
│       ├── inference/
│       ├── models/
│       ├── training/
│       └── visualization/
├── tests/
│   ├── integration/
│   └── unit/
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── requirements-dev.txt
```

## Development Setup

Create or activate a Python environment, then install the project in editable mode with development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Verify the installation:

```bash
python -c "import sewer_anomaly; print('Package imported successfully.')"
```

## Development Status

This project is currently under active development.

Current phase:

* Project structure
* Python package configuration
* Development environment setup

## Methodology

The initial baseline will use a convolutional autoencoder trained exclusively on normal sewer pipe images.

For each input image, the model will produce a reconstructed image. The difference between the original and reconstructed images will be used to calculate an anomaly score.

Images with anomaly scores above a selected threshold will be classified as defective.

The project will later compare multiple anomaly scoring strategies, including:

* Mean squared reconstruction error
* Mean absolute reconstruction error
* Structural similarity-based error
* Pixel-level residual maps
* Latent-space distance

## Evaluation

The model will be evaluated using:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion matrix
* False-positive analysis
* False-negative analysis
* Per-defect-category performance

## Dataset

The project is based on the Sewer-ML dataset.

The dataset is not included in this repository. Dataset access, licensing, and preparation instructions will be documented separately.

## Reproducibility

The project will provide:

* Versioned configuration files
* Deterministic data splitting
* Random seed control
* Training checkpoints
* Evaluation reports
* Automated tests
* Docker support
* Continuous integration

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

William Popkov
