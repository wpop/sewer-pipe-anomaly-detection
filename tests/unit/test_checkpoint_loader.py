from pathlib import Path

import pytest
import torch
from torch import nn

from sewer_anomaly.inference.checkpoint_loader import CheckpointLoader


def create_model() -> nn.Module:
    """Create a small deterministic model for checkpoint tests."""

    torch.manual_seed(42)

    return nn.Sequential(
        nn.Linear(4, 3),
        nn.ReLU(),
        nn.Linear(3, 2),
    )


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    *,
    epoch: int = 5,
    validation_loss: float = 0.125,
) -> None:
    """Save a valid checkpoint for loader tests."""

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )


def test_load_restores_model_parameters(tmp_path: Path) -> None:
    """Restore model parameters from a valid checkpoint."""

    source_model = create_model()
    checkpoint_path = tmp_path / "checkpoints" / "model.pt"

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=source_model,
    )

    target_model = create_model()

    # Deliberately change the target parameters before loading.
    with torch.no_grad():
        for parameter in target_model.parameters():
            parameter.zero_()

    loader = CheckpointLoader(
        model=target_model,
        device=torch.device("cpu"),
    )

    loader.load(checkpoint_path)

    source_parameters = list(source_model.parameters())
    target_parameters = list(target_model.parameters())

    assert len(source_parameters) == len(target_parameters)

    for source_parameter, target_parameter in zip(
        source_parameters,
        target_parameters,
        strict=True,
    ):
        assert torch.equal(
            source_parameter,
            target_parameter,
        )


def test_load_returns_checkpoint_metadata(tmp_path: Path) -> None:
    """Return the saved epoch and validation loss."""

    model = create_model()
    checkpoint_path = tmp_path / "model.pt"

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        epoch=7,
        validation_loss=0.03125,
    )

    loader = CheckpointLoader(
        model=model,
        device=torch.device("cpu"),
    )

    epoch, validation_loss = loader.load(checkpoint_path)

    assert epoch == 7
    assert validation_loss == pytest.approx(0.03125)


def test_load_switches_model_to_evaluation_mode(
    tmp_path: Path,
) -> None:
    """Switch the restored model to evaluation mode."""

    model = create_model()
    model.train()

    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
    )

    loader = CheckpointLoader(
        model=model,
        device=torch.device("cpu"),
    )

    loader.load(checkpoint_path)

    assert model.training is False


def test_load_rejects_missing_checkpoint(tmp_path: Path) -> None:
    """Reject a checkpoint path that does not exist."""

    model = create_model()

    loader = CheckpointLoader(
        model=model,
        device=torch.device("cpu"),
    )

    with pytest.raises(
        FileNotFoundError,
        match="Checkpoint file not found",
    ):
        loader.load(tmp_path / "missing.pt")


@pytest.mark.parametrize(
    "missing_key",
    [
        "model_state_dict",
        "epoch",
        "validation_loss",
    ],
)
def test_load_rejects_missing_required_field(
    tmp_path: Path,
    missing_key: str,
) -> None:
    """Reject checkpoints without required fields."""

    model = create_model()
    checkpoint_path = tmp_path / "invalid.pt"

    checkpoint: dict[str, object] = {
        "epoch": 1,
        "validation_loss": 0.1,
        "model_state_dict": model.state_dict(),
    }
    del checkpoint[missing_key]

    torch.save(checkpoint, checkpoint_path)

    loader = CheckpointLoader(
        model=model,
        device=torch.device("cpu"),
    )

    with pytest.raises(
        ValueError,
        match=missing_key,
    ):
        loader.load(checkpoint_path)


def test_load_rejects_invalid_epoch_type(tmp_path: Path) -> None:
    """Reject a checkpoint with a non-integer epoch."""

    model = create_model()
    checkpoint_path = tmp_path / "invalid_epoch.pt"

    torch.save(
        {
            "epoch": "five",
            "validation_loss": 0.1,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    loader = CheckpointLoader(
        model=model,
        device=torch.device("cpu"),
    )

    with pytest.raises(
        ValueError,
        match="Checkpoint epoch must be an integer",
    ):
        loader.load(checkpoint_path)


def test_load_rejects_invalid_validation_loss_type(
    tmp_path: Path,
) -> None:
    """Reject a checkpoint with non-numeric validation loss."""

    model = create_model()
    checkpoint_path = tmp_path / "invalid_loss.pt"

    torch.save(
        {
            "epoch": 1,
            "validation_loss": "low",
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    loader = CheckpointLoader(
        model=model,
        device=torch.device("cpu"),
    )

    with pytest.raises(
        ValueError,
        match="Checkpoint validation_loss must be numeric",
    ):
        loader.load(checkpoint_path)
