from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import torch
from torch import nn


class CheckpointLoader:
    """Restore model parameters from a PyTorch checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        self._model = model
        self._device = device

    def load(self, checkpoint_path: Path) -> tuple[int, float]:
        """Load model state and return epoch metadata."""

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}"
            )

        # PyTorch stubs may expose torch.load with incomplete typing.
        load_checkpoint = cast(Callable[..., object], torch.load)

        checkpoint_object = load_checkpoint(
            checkpoint_path,
            map_location=self._device,
            weights_only=True,
        )

        if not isinstance(checkpoint_object, Mapping):
            raise ValueError("Checkpoint must contain a mapping.")

        checkpoint = cast(Mapping[str, object], checkpoint_object)

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain model_state_dict."
            )

        if "epoch" not in checkpoint:
            raise ValueError("Checkpoint does not contain epoch.")

        if "validation_loss" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain validation_loss."
            )

        state_dict_object = checkpoint["model_state_dict"]
        epoch_object = checkpoint["epoch"]
        validation_loss_object = checkpoint["validation_loss"]

        if not isinstance(state_dict_object, Mapping):
            raise ValueError(
                "Checkpoint model_state_dict must be a mapping."
            )

        if not isinstance(epoch_object, int):
            raise ValueError("Checkpoint epoch must be an integer.")

        if not isinstance(validation_loss_object, int | float):
            raise ValueError(
                "Checkpoint validation_loss must be numeric."
            )

        state_dict = cast(
            Mapping[str, torch.Tensor],
            state_dict_object,
        )

        # Restore parameters and switch the model to inference mode.
        self._model.load_state_dict(state_dict)
        self._model.eval()

        return epoch_object, float(validation_loss_object)
