"""Shared model-family contract helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelKind(str, Enum):
    """Supported neural network model families."""

    TRANSFORMER = "transformer"
    RESNET = "resnet"


SUPPORTED_MODEL_TYPES = frozenset(kind.value for kind in ModelKind)


@dataclass(frozen=True)
class ModelInputSpec:
    """Resolved model input/output contract for a training config."""

    model_type: str
    num_players: int
    policy_dim: int
    value_dim: int
    input_dim: int | None
    num_tokens: int | None
    token_dim: int | None
    uses_relations: bool
    values_are_active_relative: bool


def normalize_model_type(model_type: str) -> ModelKind:
    """Return the normalized model kind or raise a clear config error."""
    try:
        return ModelKind(model_type)
    except ValueError as exc:
        valid = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        raise ValueError(
            f"model_type must be one of {{{valid}}}, got {model_type!r}"
        ) from exc
