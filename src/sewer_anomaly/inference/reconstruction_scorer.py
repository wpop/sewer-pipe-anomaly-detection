import torch


class ReconstructionScorer:
    """Calculate image-level anomaly scores from reconstruction errors."""

    def score(
        self,
        images: torch.Tensor,
        reconstructions: torch.Tensor,
    ) -> torch.Tensor:
        """Return one mean squared reconstruction error per image."""

        if images.shape != reconstructions.shape:
            raise ValueError(
                "Images and reconstructions must have the same shape."
            )

        if images.ndim != 4:
            raise ValueError(
                "Expected image tensors with shape "
                "(batch, channels, height, width)."
            )

        # Calculate the squared reconstruction error for every pixel.
        squared_error = (images - reconstructions).pow(2)

        # Average over channels and spatial dimensions while preserving
        # one independent anomaly score for every batch sample.
        anomaly_scores = squared_error.mean(dim=(1, 2, 3))

        return anomaly_scores
