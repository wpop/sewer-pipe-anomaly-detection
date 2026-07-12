from collections.abc import Callable, Iterable, Mapping
from typing import cast

import torch
from torch import nn
from torch.optim import Optimizer

from sewer_anomaly.models.autoencoder import ConvolutionalAutoencoder


class AutoencoderTrainer:
    """Run training and validation epochs for an autoencoder."""

    def __init__(
        self,
        model: ConvolutionalAutoencoder,
        optimizer: Optimizer,
        device: torch.device,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._device = device

        # Compare reconstructed pixels directly with the input pixels.
        self._loss_function = nn.MSELoss()

    def train_epoch(
        self,
        dataloader: Iterable[Mapping[str, object]],
    ) -> float:
        """Train the model for one complete epoch."""

        # Enable training behaviour for layers such as BatchNorm.
        self._model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            images = cast(torch.Tensor, batch["image"])
            images = images.to(
                self._device,
                non_blocking=True,
            )

            # Clear gradients accumulated during the previous batch.
            self._optimizer.zero_grad(set_to_none=True)

            reconstruction = self._model(images)

            # An autoencoder uses its input image as the training target.
            loss = cast(
                torch.Tensor,
                self._loss_function(reconstruction, images),
            )

            # PyTorch stubs do not expose a fully typed backward method.
            backward = cast(Callable[[], None], loss.backward)
            backward()
            self._optimizer.step()

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise RuntimeError("Training DataLoader produced no samples.")

        return total_loss / total_samples

    def validate_epoch(
        self,
        dataloader: Iterable[Mapping[str, object]],
    ) -> float:
        """Evaluate reconstruction loss without updating model weights."""

        self._model.eval()

        total_loss = 0.0
        total_samples = 0

        # Disable gradient tracking to reduce memory use during validation.
        with torch.inference_mode():
            for batch in dataloader:
                images = cast(torch.Tensor, batch["image"])
                images = images.to(
                    self._device,
                    non_blocking=True,
                )

                reconstruction = self._model(images)

                loss = cast(
                    torch.Tensor,
                    self._loss_function(reconstruction, images),
                )

                batch_size = images.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        if total_samples == 0:
            raise RuntimeError("Validation DataLoader produced no samples.")

        return total_loss / total_samples
