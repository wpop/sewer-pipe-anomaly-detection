from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import torch
from torch import nn

from sewer_anomaly.training.autoencoder_trainer import AutoencoderTrainer
from sewer_anomaly.training.training_runner import TrainingRunner


class StubTrainer:
    """Return deterministic training and validation losses."""

    def __init__(
        self,
        training_losses: list[float],
        validation_losses: list[float],
    ) -> None:
        self._training_losses = training_losses.copy()
        self._validation_losses = validation_losses.copy()

    def train_epoch(
        self,
        dataloader: Iterable[Mapping[str, object]],
    ) -> float:
        """Return the next configured training loss."""

        # The runner behaviour is tested independently of real batches.
        del dataloader
        return self._training_losses.pop(0)

    def validate_epoch(
        self,
        dataloader: Iterable[Mapping[str, object]],
    ) -> float:
        """Return the next configured validation loss."""

        del dataloader
        return self._validation_losses.pop(0)


def create_runner(
    tmp_path: Path,
    trainer: StubTrainer,
) -> tuple[TrainingRunner, Path, Path]:
    """Create a runner with temporary output paths."""

    model = nn.Linear(
        in_features=2,
        out_features=2,
    )

    checkpoint_path = tmp_path / "checkpoints" / "best_model.pt"
    history_path = tmp_path / "metrics" / "training_history.csv"

    # StubTrainer provides the same epoch interface required by the runner.
    typed_trainer = cast(AutoencoderTrainer, trainer)

    runner = TrainingRunner(
        trainer=typed_trainer,
        model=model,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    return runner, checkpoint_path, history_path


def test_run_returns_history_for_all_epochs(tmp_path: Path) -> None:
    """Return training and validation losses for every epoch."""

    trainer = StubTrainer(
        training_losses=[0.5, 0.4, 0.3],
        validation_losses=[0.6, 0.45, 0.35],
    )
    runner, _, _ = create_runner(tmp_path, trainer)

    empty_dataloader: tuple[Mapping[str, object], ...] = ()

    history = runner.run(
        train_dataloader=empty_dataloader,
        validation_dataloader=empty_dataloader,
        epochs=3,
    )

    assert history == [
        {
            "epoch": 1,
            "training_loss": 0.5,
            "validation_loss": 0.6,
        },
        {
            "epoch": 2,
            "training_loss": 0.4,
            "validation_loss": 0.45,
        },
        {
            "epoch": 3,
            "training_loss": 0.3,
            "validation_loss": 0.35,
        },
    ]


def test_run_saves_training_history_csv(tmp_path: Path) -> None:
    """Write the complete training history to CSV."""

    trainer = StubTrainer(
        training_losses=[0.5, 0.4],
        validation_losses=[0.6, 0.3],
    )
    runner, _, history_path = create_runner(tmp_path, trainer)

    empty_dataloader: tuple[Mapping[str, object], ...] = ()

    runner.run(
        train_dataloader=empty_dataloader,
        validation_dataloader=empty_dataloader,
        epochs=2,
    )

    dataframe = pd.read_csv(history_path)

    assert history_path.is_file()
    assert dataframe["epoch"].tolist() == [1, 2]
    assert dataframe["training_loss"].tolist() == [0.5, 0.4]
    assert dataframe["validation_loss"].tolist() == [0.6, 0.3]


def test_run_saves_checkpoint_for_best_epoch(tmp_path: Path) -> None:
    """Save the checkpoint associated with the lowest validation loss."""

    trainer = StubTrainer(
        training_losses=[0.5, 0.4, 0.3],
        validation_losses=[0.6, 0.2, 0.4],
    )
    runner, checkpoint_path, _ = create_runner(tmp_path, trainer)

    empty_dataloader: tuple[Mapping[str, object], ...] = ()

    runner.run(
        train_dataloader=empty_dataloader,
        validation_dataloader=empty_dataloader,
        epochs=3,
    )

    checkpoint_object = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint = cast(dict[str, object], checkpoint_object)

    assert checkpoint_path.is_file()
    assert checkpoint["epoch"] == 2
    assert checkpoint["validation_loss"] == pytest.approx(0.2)
    assert "model_state_dict" in checkpoint


@pytest.mark.parametrize("epochs", [0, -1])
def test_run_rejects_invalid_epoch_count(
    tmp_path: Path,
    epochs: int,
) -> None:
    """Reject zero and negative epoch counts."""

    trainer = StubTrainer(
        training_losses=[],
        validation_losses=[],
    )
    runner, _, _ = create_runner(tmp_path, trainer)

    empty_dataloader: tuple[Mapping[str, object], ...] = ()

    with pytest.raises(
        ValueError,
        match="Epoch count must be greater than zero",
    ):
        runner.run(
            train_dataloader=empty_dataloader,
            validation_dataloader=empty_dataloader,
            epochs=epochs,
        )
