from typing import cast

import torch
from torch import nn


class Encoder(nn.Module):
    """Compress input images into a compact latent representation."""

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()

        self.network = nn.Sequential(
            # 3 x 256 x 256 -> 32 x 128 x 128
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 32 x 128 x 128 -> 64 x 64 x 64
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 x 64 x 64 -> 128 x 32 x 32
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128 x 32 x 32 -> latent_channels x 16 x 16
            nn.Conv2d(
                in_channels=128,
                out_channels=latent_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of normalized RGB images."""

        encoded = cast(torch.Tensor, self.network(images))
        return encoded
