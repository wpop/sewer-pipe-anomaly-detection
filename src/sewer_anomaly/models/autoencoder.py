from typing import cast

import torch
from torch import nn

from sewer_anomaly.models.decoder import Decoder
from sewer_anomaly.models.encoder import Encoder


class ConvolutionalAutoencoder(nn.Module):
    """Reconstruct pipe images through an encoder-decoder architecture."""

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()

        self.encoder = Encoder(latent_channels=latent_channels)
        self.decoder = Decoder(latent_channels=latent_channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of normalized RGB images."""

        latent = self.encoder(images)
        reconstruction = self.decoder(latent)

        output = cast(torch.Tensor, reconstruction)
        return output
