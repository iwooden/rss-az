"""Save/load training checkpoints and checkpoint management."""

from __future__ import annotations

from pathlib import Path

import torch

from train.config import TrainingConfig


def _unwrap_state_dict(
    state_dict: dict[str, object],
) -> dict[str, object]:
    """Strip ``_orig_mod.`` prefix added by ``torch.compile``."""
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state_dict):
        return {k.removeprefix(prefix): v for k, v in state_dict.items()}
    return state_dict


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    trainer_state: dict[str, object],
    config: TrainingConfig,
    metrics: dict[str, float],
    buffer_stats: dict[str, int],
    rng_state: dict[str, object] | None = None,
) -> None:
    """Save checkpoint via torch.save. Creates parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {
        "epoch": epoch,
        "model_state_dict": _unwrap_state_dict(model.state_dict()),
        "trainer_state": trainer_state,
        "config_json": config.to_json(),
        "metrics": metrics,
        "buffer_stats": buffer_stats,
    }
    if rng_state is not None:
        data["rng_state"] = rng_state
    torch.save(data, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    """Load checkpoint from disk with device mapping.

    Automatically strips ``_orig_mod.`` prefix from model keys so checkpoints
    saved from ``torch.compile``d models load into plain modules.

    Uses weights_only=False because optimizer/scheduler state dicts contain
    non-tensor objects. Only load checkpoints you trust.
    """
    cp: dict[str, object] = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in cp:
        cp["model_state_dict"] = _unwrap_state_dict(cp["model_state_dict"])  # type: ignore[arg-type]
    return cp


def load_model_from_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, TrainingConfig, dict[str, object]]:
    """Load checkpoint data and instantiate the architecture it was saved with."""
    cp = load_checkpoint(path, device)
    config = TrainingConfig.from_json(cp["config_json"])  # type: ignore[arg-type]

    from nn import create_model

    model = create_model(config).to(device)
    model.load_state_dict(cp["model_state_dict"])  # type: ignore[arg-type]
    return model, config, cp


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
