from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torch.utils.data import DataLoader

from sewer_anomaly.data.dataset import SewerMLDataset
from sewer_anomaly.data.preprocessing import (
    ImagePreprocessingConfig,
    ImagePreprocessor,
)
from sewer_anomaly.inference.checkpoint_loader import CheckpointLoader
from sewer_anomaly.inference.reconstruction_scorer import (
    ReconstructionScorer,
)
from sewer_anomaly.models.autoencoder import ConvolutionalAutoencoder


def main() -> None:
    """Compare reconstruction scores for normal and defective images."""

    checkpoint_path = Path(
        "outputs/checkpoints/entrypoint_smoke_autoencoder.pt"
    )
    manifest_path = Path("data/interim/valid00_inspection.csv")
    image_directory = Path("data/raw/sewer_ml/valid00")

    preprocessor = ImagePreprocessor(
        ImagePreprocessingConfig(
            image_width=256,
            image_height=256,
        )
    )

    dataset = SewerMLDataset(
        manifest_path=manifest_path,
        image_directory=image_directory,
        preprocessor=preprocessor,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
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

    epoch, validation_loss = checkpoint_loader.load(checkpoint_path)

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

    dataframe = pd.DataFrame(records)

    normal_scores = dataframe.loc[
        dataframe["label"] == 0,
        "reconstruction_score",
    ]
    defective_scores = dataframe.loc[
        dataframe["label"] == 1,
        "reconstruction_score",
    ]

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint epoch: {epoch}")
    print(f"Checkpoint validation loss: {validation_loss:.6f}")
    print(f"Normal samples: {len(normal_scores)}")
    print(f"Defective samples: {len(defective_scores)}")
    print(f"Normal mean score: {normal_scores.mean():.6f}")
    print(f"Defective mean score: {defective_scores.mean():.6f}")

    print("\nIndividual scores:")
    print(dataframe.to_string(index=False))

    assert len(normal_scores) > 0
    assert len(defective_scores) > 0
    assert dataframe["reconstruction_score"].ge(0.0).all()

    print("\nReconstruction scoring smoke test passed.")


if __name__ == "__main__":
    main()
