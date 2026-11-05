"""Shared model-family contract helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import overload

import numpy as np
import torch

from games import RELATION_RECORD_WIDTH

# Flat relation records: (batch, relation, query, key, value), int16.
FLAT_RELATION_RECORD_WIDTH = RELATION_RECORD_WIDTH + 1


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

    ``policy_dim``, ``num_tokens``, and ``token_dim`` are those of ``game``'s
    adapter (``games.get_game``). A spec without ``game`` is RSS's.
    """

    model_type: str
    num_players: int
    policy_dim: int
    value_dim: int
    num_tokens: int
    token_dim: int
    # The game's token layout id, the model module's INPUT_LAYOUT_VERSION.
    # RSS versions 2 and 3 share token order and relation transport. V2
    # consumes only the original ten binary planes; v3 also uses the
    # share-count planes.
    layout_version: int
    game: str = "rss"


def normalize_model_type(model_type: str) -> ModelKind:
    """Return the normalized model kind or raise a clear config error."""
    try:
        return ModelKind(model_type)
    except ValueError as exc:
        valid = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        raise ValueError(
            f"model_type must be one of {{{valid}}}, got {model_type!r}"
        ) from exc


@overload
def flatten_relation_records(records: np.ndarray) -> np.ndarray: ...
@overload
def flatten_relation_records(records: torch.Tensor) -> torch.Tensor: ...
def flatten_relation_records(records: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Turn a batch's padded sparse records into its flat ones.

    Takes ``(B, E, RELATION_RECORD_WIDTH)`` uint8 records on the CPU, an array
    or a tensor, and returns the same kind of ``(N, FLAT_RELATION_RECORD_WIDTH)``
    int16 rows ``(batch, relation, query, key, value)``, in batch order: every
    record with a value above 0, whatever its relation id, which the model
    filters. N depends on the data, so this runs outside the compiled forward.
    """
    rows = torch.as_tensor(records)
    batch, edge = rows[:, :, RELATION_RECORD_WIDTH - 1].nonzero(as_tuple=True)
    flat = torch.cat([batch[:, None], rows[batch, edge]], dim=1).to(torch.int16)
    return flat.numpy() if isinstance(records, np.ndarray) else flat
