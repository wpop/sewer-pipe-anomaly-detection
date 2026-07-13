import torch


class ReconstructionHeatmapGenerator:
    """Generate spatial reconstruction-error heatmaps."""

    def generate(
        self,
        image: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        """Return a two-dimensional heatmap for one image."""

        if image.shape != reconstruction.shape:
            raise ValueError(
                "Image and reconstruction must have the same shape."
            )

        if image.ndim != 3:
            raise ValueError(
                "Expected image tensors with shape "
                "(channels, height, width)."
            )

        if image.shape[0] <= 0:
            raise ValueError(
                "Image must contain at least one channel."
            )

        if not torch.isfinite(image).all():
            raise ValueError(
                "Image must contain only finite values."
            )

        if not torch.isfinite(reconstruction).all():
            raise ValueError(
                "Reconstruction must contain only finite values."
            )

        squared_error = (
            image - reconstruction
        ).pow(2)

        heatmap = squared_error.mean(dim=0)

        return heatmap
