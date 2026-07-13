from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


class ReconstructionHeatmapPlotter:
    """Plot an image, its reconstruction, and its error heatmap."""

    def save(
        self,
        image: torch.Tensor,
        reconstruction: torch.Tensor,
        heatmap: torch.Tensor,
        output_path: Path,
        filename: str,
        reconstruction_score: float,
    ) -> None:
        """Save a composite reconstruction heatmap figure."""

        if image.shape != reconstruction.shape:
            raise ValueError(
                "Image and reconstruction must have the same shape."
            )

        if image.ndim != 3:
            raise ValueError(
                "Expected image tensors with shape "
                "(channels, height, width)."
            )

        if image.shape[0] != 3:
            raise ValueError(
                "Expected image tensors with three RGB channels."
            )

        if heatmap.ndim != 2:
            raise ValueError(
                "Expected heatmap tensor with shape "
                "(height, width)."
            )

        if heatmap.shape != image.shape[1:]:
            raise ValueError(
                "Heatmap spatial shape must match image shape."
            )

        if not torch.isfinite(image).all():
            raise ValueError(
                "Image must contain only finite values."
            )

        if not torch.isfinite(reconstruction).all():
            raise ValueError(
                "Reconstruction must contain only finite values."
            )

        if not torch.isfinite(heatmap).all():
            raise ValueError(
                "Heatmap must contain only finite values."
            )

        if not np.isfinite(reconstruction_score):
            raise ValueError(
                "Reconstruction score must be finite."
            )

        if not filename.strip():
            raise ValueError(
                "Filename cannot be empty."
            )

        image_array = (
            image.detach()
            .cpu()
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .numpy()
        )

        reconstruction_array = (
            reconstruction.detach()
            .cpu()
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .numpy()
        )

        heatmap_array = (
            heatmap.detach()
            .cpu()
            .numpy()
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5),
        )

        axes[0].imshow(image_array)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(reconstruction_array)
        axes[1].set_title("Reconstruction")
        axes[1].axis("off")

        heatmap_image = axes[2].imshow(
            heatmap_array,
            cmap="inferno",
        )
        axes[2].set_title("Reconstruction Error")
        axes[2].axis("off")

        figure.colorbar(
            heatmap_image,
            ax=axes[2],
            fraction=0.046,
            pad=0.04,
        )

        figure.suptitle(
            f"{filename} | Reconstruction score: "
            f"{reconstruction_score:.6f}"
        )

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)
