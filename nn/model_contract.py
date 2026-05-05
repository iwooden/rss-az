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
    input_dim: int | None
    num_tokens: int | None
    token_dim: int | None
    uses_relations: bool
    values_are_active_relative: bool


def canonical_player_for_relative(
    active_player: int,
    relative_slot: int,
    num_players: int,
) -> int:
    """Map an active-relative player slot back to canonical player id."""
    _validate_player_count(num_players)
    if not 0 <= active_player < num_players:
        raise ValueError(
            f"active_player must be in [0, {num_players}), got {active_player}"
        )
    if not 0 <= relative_slot < num_players:
        raise ValueError(
            f"relative_slot must be in [0, {num_players}), got {relative_slot}"
        )
    return (active_player + relative_slot) % num_players


def relative_slot_for_canonical(
    active_player: int,
    player_id: int,
    num_players: int,
) -> int:
    """Map a canonical player id into the active-relative slot order."""
    _validate_player_count(num_players)
    if not 0 <= active_player < num_players:
        raise ValueError(
            f"active_player must be in [0, {num_players}), got {active_player}"
        )
    if not 0 <= player_id < num_players:
        raise ValueError(f"player_id must be in [0, {num_players}), got {player_id}")
    return (player_id - active_player) % num_players


def rotate_values_to_relative(values_canonical, active_player: int, num_players: int):
    """Return values reordered from canonical player order to relative order."""
    order = [
        canonical_player_for_relative(active_player, rel, num_players)
        for rel in range(num_players)
    ]
    return _take_last_dim(values_canonical, order)


def unrotate_values_to_canonical(values_relative, active_player: int, num_players: int):
    """Return values reordered from relative player order to canonical order."""
    order = [
        relative_slot_for_canonical(active_player, player_id, num_players)
        for player_id in range(num_players)
    ]
    return _take_last_dim(values_relative, order)


def normalize_model_type(model_type: str) -> ModelKind:
    """Return the normalized model kind or raise a clear config error."""
    try:
        return ModelKind(model_type)
    except ValueError as exc:
        valid = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        raise ValueError(
            f"model_type must be one of {{{valid}}}, got {model_type!r}"
        ) from exc


def _validate_player_count(num_players: int) -> None:
    if not 3 <= num_players <= 5:
        raise ValueError(f"num_players must be in [3, 5], got {num_players}")


def _take_last_dim(values, order: list[int]):
    try:
        return values[..., order]
    except TypeError:
        if not values:
            return []
        if isinstance(values[0], (list, tuple)):
            return [type(row)(row[i] for i in order) for row in values]
        return type(values)(values[i] for i in order)
