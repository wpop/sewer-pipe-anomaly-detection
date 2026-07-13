from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torch.utils.data import DataLoader

from sewer_anomaly.data.dataset import (
    SewerMLDataset,
    SewerMLSample,
)
from sewer_anomaly.data.preprocessing import (
    ImagePreprocessingConfig,
    ImagePreprocessor,
)
from sewer_anomaly.inference.checkpoint_loader import CheckpointLoader
from sewer_anomaly.inference.reconstruction_scorer import (
    ReconstructionScorer,
)
from sewer_anomaly.models.autoencoder import ConvolutionalAutoencoder


def create_available_defective_manifest(
    source_manifest: Path,
    image_directory: Path,
    output_path: Path,
) -> Path:
    """Create a manifest containing available defective images."""

    dataframe = pd.read_csv(source_manifest)

    # Keep only defective images available in the local valid00 directory.
    available_mask = dataframe["Filename"].map(
        lambda filename: (
            image_directory / str(filename)
        ).is_file()
    )
    available_dataframe = dataframe.loc[available_mask].copy()

    if available_dataframe.empty:
        raise RuntimeError(
            "No defective validation images were found."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    available_dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path


def collect_scores(
    model: ConvolutionalAutoencoder,
    dataloader: DataLoader[SewerMLSample],
    scorer: ReconstructionScorer,
    device: torch.device,
) -> list[dict[str, object]]:
    """Calculate reconstruction scores for every dataset sample."""

    records: list[dict[str, object]] = []

    with torch.inference_mode():
        for batch in dataloader:
            images = cast(torch.Tensor, batch["image"]).to(
                device,
                non_blocking=True,
            )
            labels = cast(torch.Tensor, batch["label"])
            filenames = cast(list[str], batch["filename"])

            reconstructions = model(images)
            scores = scorer.score(
                images=images,
                reconstructions=reconstructions,
            ).cpu()

            for filename, label, score in zip(
                filenames,
                labels.tolist(),
                scores.tolist(),
                strict=True,
            ):
                records.append(
                    {
                        "filename": filename,
                        "label": int(label),
                        "reconstruction_score": float(score),
                    }
                )

    return records


def main() -> None:
    """Compare normal and defective reconstruction scores."""

    checkpoint_path = Path(
        "outputs/checkpoints/normal_smoke_autoencoder.pt"
    )
    image_directory = Path("data/raw/sewer_ml/valid00")

    normal_manifest_path = Path(
        "data/interim/valid00_normal_val.csv"
    )

    defective_manifest_path = create_available_defective_manifest(
        source_manifest=Path(
            "data/processed/val_defective.csv"
        ),
        image_directory=image_directory,
        output_path=Path(
            "data/interim/valid00_defective_eval.csv"
        ),
    )

    output_path = Path(
        "outputs/predictions/normal_only_reconstruction_scores.csv"
    )

    preprocessor = ImagePreprocessor(
        ImagePreprocessingConfig(
            image_width=256,
            image_height=256,
        )
    )

    normal_dataset = SewerMLDataset(
        manifest_path=normal_manifest_path,
        image_directory=image_directory,
        preprocessor=preprocessor,
    )

    defective_dataset = SewerMLDataset(
        manifest_path=defective_manifest_path,
        image_directory=image_directory,
        preprocessor=preprocessor,
    )

    normal_dataloader = DataLoader(
        normal_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    defective_dataloader = DataLoader(
        defective_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ConvolutionalAutoencoder(
        latent_channels=128,
    ).to(device)

    checkpoint_loader = CheckpointLoader(
        model=model,
        device=device,
    )
    scorer = ReconstructionScorer()

    epoch, validation_loss = checkpoint_loader.load(
        checkpoint_path
    )

    normal_records = collect_scores(
        model=model,
        dataloader=normal_dataloader,
        scorer=scorer,
        device=device,
    )

    defective_records = collect_scores(
        model=model,
        dataloader=defective_dataloader,
        scorer=scorer,
        device=device,
    )

    dataframe = pd.DataFrame(
        normal_records + defective_records
    )

    normal_scores = dataframe.loc[
        dataframe["label"] == 0,
        "reconstruction_score",
    ]
    defective_scores = dataframe.loc[
        dataframe["label"] == 1,
        "reconstruction_score",
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {epoch}")
    print(
        f"Checkpoint validation loss: "
        f"{validation_loss:.6f}"
    )
    print(f"Normal samples: {len(normal_scores)}")
    print(f"Defective samples: {len(defective_scores)}")
    print(
        f"Normal mean score: "
        f"{normal_scores.mean():.6f}"
    )
    print(
        f"Defective mean score: "
        f"{defective_scores.mean():.6f}"
    )
    print(f"Scores saved to: {output_path}")

    assert len(normal_scores) > 0
    assert len(defective_scores) > 0
    assert dataframe["reconstruction_score"].ge(0.0).all()

    print("Normal-only reconstruction evaluation passed.")


if __name__ == "__main__":
    main()
