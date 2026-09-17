"""Shared model-family contract helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelKind(str, Enum):
    """Supported neural network input/output contracts."""

    TRANSFORMER = "transformer"


SUPPORTED_MODEL_TYPES = frozenset(kind.value for kind in ModelKind)


@dataclass(frozen=True)
class ModelInputSpec:
    """Resolved model input/output contract for a training config.

    For transformer models, ``num_players`` and ``value_dim`` are the padded
    model capacity. In mixed player-count training this is the config's
    effective maximum player count, not necessarily the actual players in a
    given state.
    """

    model_type: str
    num_players: int
    policy_dim: int
    value_dim: int
    num_tokens: int
    token_dim: int
    # Versions 2 and 3 share relation planes and token order.
    layout_version: int = 2


def normalize_model_type(model_type: str) -> ModelKind:
    """Return the normalized model kind or raise a clear config error."""
    try:
        return ModelKind(model_type)
    except ValueError as exc:
        valid = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        raise ValueError(
            f"model_type must be one of {{{valid}}}, got {model_type!r}"
        ) from exc
