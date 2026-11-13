"""Shared model-family contract helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, overload

import numpy as np
import torch

from games import RELATION_RECORD_WIDTH

if TYPE_CHECKING:
    from games import Game

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

    A packed layout (1889's layout 3, ``nn/transformer-1889-v2.py``) has no
    token rows: ``input_size`` is its vector's length a state, and
    ``num_tokens`` and ``token_dim`` are 0. A token layout's ``input_size``
    is 0. The vector's parts are the engine's to give (1889:
    ``g1889.token_data.get_input_parts``); consumers here need only its length.
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
    input_size: int = 0

    @property
    def packed(self) -> bool:
        """One vector a state, no relations: the forward is ``(inputs, legal_mask)``."""
        return self.input_size > 0

    @property
    def input_shape(self) -> tuple[int, ...]:
        """A state's model input: ``(input_size,)``, or ``(num_tokens, token_dim)``."""
        return (self.input_size,) if self.packed else (self.num_tokens, self.token_dim)


def input_spec_for(
    game: Game, num_players: int, layout_version: int,
    model_type: str = "transformer",
) -> ModelInputSpec:
    """``game``'s input contract for layout ``layout_version`` at capacity
    ``num_players``: a packed layout's vector length, or a token layout's
    rows and width."""
    input_size = 0
    if layout_version in game.packed_layouts:
        assert game.input_size is not None, f"game {game.name!r} packs layout {layout_version} without input_size"
        input_size = int(game.input_size(num_players, layout_version=layout_version))
    packed = input_size > 0
    return ModelInputSpec(
        model_type=model_type,
        num_players=num_players,
        policy_dim=game.policy.unified_dim,
        value_dim=num_players,
        num_tokens=0 if packed else game.num_tokens(num_players),
        token_dim=0 if packed else game.token_dim(layout_version),
        layout_version=layout_version,
        game=game.name,
        input_size=input_size,
    )


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
