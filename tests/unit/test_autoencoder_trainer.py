from collections.abc import Iterator, Mapping

import pytest
import torch
from torch.optim import Adam

from sewer_anomaly.models.autoencoder import ConvolutionalAutoencoder
from sewer_anomaly.training.autoencoder_trainer import AutoencoderTrainer


class SyntheticDataLoader:
    """Provide deterministic synthetic image batches for trainer tests."""

    def __init__(
        self,
        batch_count: int,
        batch_size: int,
    ) -> None:
        self._batch_count = batch_count
        self._batch_size = batch_size

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        """Yield normalized synthetic image batches."""

        for _ in range(self._batch_count):
            images = torch.rand(
                self._batch_size,
                3,
                256,
                256,
            )

            yield {
                "image": images,
                "label": torch.zeros(
                    self._batch_size,
                    dtype=torch.long,
                ),
                "filename": ["synthetic.png"] * self._batch_size,
            }


def create_trainer() -> AutoencoderTrainer:
    """Create a CPU trainer for unit tests."""

    device = torch.device("cpu")
    model = ConvolutionalAutoencoder(
        latent_channels=16,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )

    return AutoencoderTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
    )


def test_train_epoch_returns_positive_float() -> None:
    """Train one epoch and return an average reconstruction loss."""

    trainer = create_trainer()
    dataloader = SyntheticDataLoader(
        batch_count=2,
        batch_size=2,
    )

    loss = trainer.train_epoch(dataloader)

    assert isinstance(loss, float)
    assert loss > 0.0


def test_validate_epoch_returns_positive_float() -> None:
    """Validate one epoch without updating model parameters."""

    trainer = create_trainer()
    dataloader = SyntheticDataLoader(
        batch_count=2,
        batch_size=2,
    )

    loss = trainer.validate_epoch(dataloader)

    assert isinstance(loss, float)
    assert loss > 0.0


def test_train_epoch_updates_model_parameters() -> None:
    """Confirm that optimizer steps modify model parameters."""

    device = torch.device("cpu")
    model = ConvolutionalAutoencoder(
        latent_channels=16,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )

    trainer = AutoencoderTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
    )

    dataloader = SyntheticDataLoader(
        batch_count=1,
        batch_size=2,
    )

    parameter_before = next(model.parameters()).detach().clone()

    trainer.train_epoch(dataloader)

    parameter_after = next(model.parameters()).detach().clone()

    assert not torch.equal(parameter_before, parameter_after)


def test_train_epoch_rejects_empty_dataloader() -> None:
    """Reject a training epoch without samples."""

    trainer = create_trainer()
    dataloader = SyntheticDataLoader(
        batch_count=0,
        batch_size=2,
    )

    with pytest.raises(
        RuntimeError,
        match="Training DataLoader produced no samples",
    ):
        trainer.train_epoch(dataloader)


def test_validate_epoch_rejects_empty_dataloader() -> None:
    """Reject a validation epoch without samples."""

    trainer = create_trainer()
    dataloader = SyntheticDataLoader(
        batch_count=0,
        batch_size=2,
    )

    with pytest.raises(
        RuntimeError,
        match="Validation DataLoader produced no samples",
    ):
        trainer.validate_epoch(dataloader)
