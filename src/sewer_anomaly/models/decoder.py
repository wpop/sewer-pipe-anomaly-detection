from typing import cast

import torch
from torch import nn


class Decoder(nn.Module):
    """Reconstruct images from their latent representation."""

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()

        self.network = nn.Sequential(
            # latent_channels x 16 x 16 -> 128 x 32 x 32
            nn.ConvTranspose2d(
                in_channels=latent_channels,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128 x 32 x 32 -> 64 x 64 x 64
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 x 64 x 64 -> 32 x 128 x 128
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 32 x 128 x 128 -> 3 x 256 x 256
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            # Match the [0, 1] range produced by image preprocessing.
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent features into reconstructed RGB images."""

        decoded = cast(torch.Tensor, self.network(latent))
        return decoded
