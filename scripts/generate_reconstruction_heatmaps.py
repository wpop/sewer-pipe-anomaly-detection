from pathlib import Path

import torch

from sewer_anomaly.data.preprocessing import (
    ImagePreprocessingConfig,
    ImagePreprocessor,
)
from sewer_anomaly.inference.checkpoint_loader import CheckpointLoader
from sewer_anomaly.inference.reconstruction_heatmap_generator import (
    ReconstructionHeatmapGenerator,
)
from sewer_anomaly.inference.reconstruction_scorer import (
    ReconstructionScorer,
)
from sewer_anomaly.models.autoencoder import ConvolutionalAutoencoder
from sewer_anomaly.visualization.reconstruction_heatmap_plotter import (
    ReconstructionHeatmapPlotter,
)


def generate_sample_heatmap(
    filename: str,
    category: str,
    model: ConvolutionalAutoencoder,
    preprocessor: ImagePreprocessor,
    scorer: ReconstructionScorer,
    heatmap_generator: ReconstructionHeatmapGenerator,
    plotter: ReconstructionHeatmapPlotter,
    device: torch.device,
    image_directory: Path,
    output_directory: Path,
) -> float:
    """Run inference and save a heatmap for one image."""

    image_path = image_directory / filename

    image = preprocessor.load_and_preprocess(image_path)

    image_batch = image.unsqueeze(0).to(
        device,
        non_blocking=True,
    )

    with torch.inference_mode():
        reconstruction_batch = model(image_batch)

    device_image = image_batch[0]
    reconstruction = reconstruction_batch[0]

    score_tensor = scorer.score(
        images=image_batch,
        reconstructions=reconstruction_batch,
    )
    reconstruction_score = float(score_tensor[0].item())

    heatmap = heatmap_generator.generate(
        image=device_image,
        reconstruction=reconstruction,
    )

    output_path = (
        output_directory
        / f"{category}_{Path(filename).stem}_heatmap.png"
    )

    plotter.save(
        image=device_image,
        reconstruction=reconstruction,
        heatmap=heatmap,
        output_path=output_path,
        filename=filename,
        reconstruction_score=reconstruction_score,
    )

    print(
        f"{category.capitalize()} sample: "
        f"{filename}, score={reconstruction_score:.6f}"
    )
    print(f"Figure saved to: {output_path}")

    return reconstruction_score


def main() -> None:
    """Generate heatmaps for normal and defective Sewer-ML samples."""

    checkpoint_path = Path(
        "outputs/checkpoints/normal_smoke_autoencoder.pt"
    )
    image_directory = Path(
        "data/raw/sewer_ml/valid00"
    )
    output_directory = Path(
        "outputs/figures/reconstruction_heatmaps"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    preprocessor = ImagePreprocessor(
        ImagePreprocessingConfig(
            image_width=256,
            image_height=256,
        )
    )

    model = ConvolutionalAutoencoder(
        latent_channels=128,
    ).to(device)

    checkpoint_loader = CheckpointLoader(
        model=model,
        device=device,
    )

    scorer = ReconstructionScorer()
    heatmap_generator = ReconstructionHeatmapGenerator()
    plotter = ReconstructionHeatmapPlotter()

    epoch, validation_loss = checkpoint_loader.load(
        checkpoint_path
    )

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {epoch}")
    print(
        f"Checkpoint validation loss: "
        f"{validation_loss:.6f}"
    )

    normal_score = generate_sample_heatmap(
        filename="00262683.png",
        category="normal",
        model=model,
        preprocessor=preprocessor,
        scorer=scorer,
        heatmap_generator=heatmap_generator,
        plotter=plotter,
        device=device,
        image_directory=image_directory,
        output_directory=output_directory,
    )

    defective_score = generate_sample_heatmap(
        filename="00085421.png",
        category="defective",
        model=model,
        preprocessor=preprocessor,
        scorer=scorer,
        heatmap_generator=heatmap_generator,
        plotter=plotter,
        device=device,
        image_directory=image_directory,
        output_directory=output_directory,
    )

    if defective_score <= normal_score:
        raise RuntimeError(
            "Expected the selected defective sample to have "
            "a higher reconstruction score."
        )

    print("Reconstruction heatmap generation passed.")


if __name__ == "__main__":
    main()
