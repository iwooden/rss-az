"""Save/load training checkpoints and checkpoint management."""

from __future__ import annotations

from pathlib import Path

import torch

from train.config import TrainingConfig


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    trainer_state: dict[str, object],
    config: TrainingConfig,
    metrics: dict[str, float],
    buffer_stats: dict[str, int],
) -> None:
    """Save checkpoint via torch.save. Creates parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "trainer_state": trainer_state,
            "config_json": config.to_json(),
            "metrics": metrics,
            "buffer_stats": buffer_stats,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    """Load checkpoint from disk with device mapping.

    Uses weights_only=False because optimizer/scheduler state dicts contain
    non-tensor objects. Only load checkpoints you trust.
    """
    return torch.load(path, map_location=device, weights_only=False)  # type: ignore[no-any-return]


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the most recent checkpoint_epoch_NNNN.pt in the directory."""
    if not checkpoint_dir.is_dir():
        return None
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    return checkpoints[-1] if checkpoints else None


def cleanup_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    """Remove old checkpoints, keeping only the N most recent."""
    if not checkpoint_dir.is_dir():
        return
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    for cp in checkpoints[:-keep_last_n]:
        cp.unlink()
