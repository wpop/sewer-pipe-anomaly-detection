from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import cast

import pandas as pd
import torch

from sewer_anomaly.training.autoencoder_trainer import AutoencoderTrainer


class TrainingRunner:
    """Run multiple training epochs and persist the best model."""

    def __init__(
        self,
        trainer: AutoencoderTrainer,
        model: torch.nn.Module,
        checkpoint_path: Path,
        history_path: Path,
    ) -> None:
        self._trainer = trainer
        self._model = model
        self._checkpoint_path = checkpoint_path
        self._history_path = history_path

    def run(
        self,
        train_dataloader: Iterable[Mapping[str, object]],
        validation_dataloader: Iterable[Mapping[str, object]],
        epochs: int,
    ) -> list[dict[str, float | int]]:
        """Run training and validation for the requested epochs."""

        if epochs <= 0:
            raise ValueError("Epoch count must be greater than zero.")

        # Prepare output directories before training starts.
        self._checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        best_validation_loss = float("inf")
        history: list[dict[str, float | int]] = []

        for epoch in range(1, epochs + 1):
            training_loss = self._trainer.train_epoch(train_dataloader)
            validation_loss = self._trainer.validate_epoch(
                validation_dataloader
            )

            epoch_result: dict[str, float | int] = {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }
            history.append(epoch_result)

            print(
                f"Epoch {epoch}/{epochs}: "
                f"training_loss={training_loss:.6f}, "
                f"validation_loss={validation_loss:.6f}"
            )

            # Persist only the model with the lowest validation loss.
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                self._save_checkpoint(
                    epoch=epoch,
                    validation_loss=validation_loss,
                )

        self._save_history(history)

        return history

    def _save_checkpoint(
        self,
        epoch: int,
        validation_loss: float,
    ) -> None:
        """Save model weights and validation metadata."""

        checkpoint = {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": self._model.state_dict(),
        }

        # PyTorch stubs may expose torch.save as an untyped callable.
        save_checkpoint = cast(
            Callable[[object, Path], None],
            torch.save,
        )
        save_checkpoint(checkpoint, self._checkpoint_path)

    def _save_history(
        self,
        history: list[dict[str, float | int]],
    ) -> None:
        """Write epoch losses to a CSV file."""

        dataframe = pd.DataFrame(history)
        dataframe.to_csv(self._history_path, index=False)
