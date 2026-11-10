"""1889's start-position file (1889/v1-design.md §2.1): what
``utils_18xx/g1889_start_positions.py`` builds and self-play starts games from
(``Game.load_start_positions``).

The ``.npz`` holds the rows at one player capacity and, per row, its player
count, source game (an index into ``source_games``), decision index, decisions
left, game phase (an index into ``phase_names``), decision phase, and round;
and the row size, capacity, ``state_layout_hash()``, and the engine flags the
positions were filtered under. ``load_start_positions`` refuses a file whose
hash or row size is not the running engine's, or whose capacity or flags are
not the run's: rebuild it with the builder.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import g1889.constants as K
import g1889.data.tables as D
from g1889.state import (
    get_corp_fields, get_global_fields, get_hex_fields, get_layout, get_player_fields, get_private_fields,
    get_train_fields,
)
from train.config import _ENGINE_OPTIONS

if TYPE_CHECKING:
    from train.config import TrainingConfig

_NUM_PLAYERS = get_layout(0).global_offset + get_global_fields().num_players


def engine_options(config: TrainingConfig) -> dict[str, Any]:
    """``config``'s engine flags: what decides which positions self-play can
    continue from."""
    return {name: getattr(config, name) for name in _ENGINE_OPTIONS["1889"]}


def state_layout_hash() -> str:
    """What a row's slots mean: each block's offset, each block's fields and their
    offsets, the codes the row stores, and the id orders of the tables it indexes.
    The width, which the player capacity sets, is checked apart."""
    layout = {k: v for k, v in get_layout(0)._asdict().items() if k not in ("total_size", "max_players")}
    blocks = {"global": get_global_fields(), "corp": get_corp_fields(), "private": get_private_fields(),
              "hex": get_hex_fields(), "train": get_train_fields(), "player": get_player_fields()}
    blocks = {name: block._asdict() for name, block in blocks.items()}
    codes = {e.__name__: {m.name: int(m) for m in e} for e in (K.Round, K.Step, K.EndReason, K.Owner, K.TrainSource)}
    tables = {name: list(getattr(D, name)) for name in ("HEX_IDS", "TILE_NAMES", "CORP_SYMS", "PRIVATE_SYMS",
                                                        "TRAIN_NAMES", "PHASE_NAMES", "PAR_PRICE", "MARKET_PRICE")}
    text = json.dumps([layout, blocks, codes, tables], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class StartPositions:
    rows: np.ndarray            # (N, row_size) int16, at capacity max_players
    num_players: np.ndarray     # (N,)
    source_game: np.ndarray     # (N,) index into source_games
    decision_index: np.ndarray  # (N,) decisions before the position in its game
    decisions_left: np.ndarray  # (N,) decisions from it to the end, itself included
    game_phase: np.ndarray      # (N,) index into phase_names
    decision_phase: np.ndarray  # (N,) K.DecisionPhase
    round_type: np.ndarray      # (N,) K.Round
    source_games: np.ndarray    # (G,) game ids, the file stems
    phase_names: np.ndarray     # D.PHASE_NAMES
    max_players: int
    layout_hash: str
    engine_options: str         # JSON: the flags the positions were filtered under


def save_start_positions(path: Path, positions: StartPositions) -> None:
    arrays = {f.name: getattr(positions, f.name) for f in fields(positions)}
    np.savez_compressed(path, row_size=positions.rows.shape[1], **arrays)


def load_start_positions(path: Path, max_players: int | None = None,
                         options: Mapping[str, Any] | None = None) -> StartPositions:
    """The positions in ``path``, checked against the running engine's layout and,
    if given, the run's player capacity and engine flags (``engine_options``)."""
    rebuild = f"rebuild it: python -m utils_18xx.g1889_start_positions --out {path}"
    names = [f.name for f in fields(StartPositions)]
    with np.load(path, allow_pickle=False) as data:
        if missing := sorted({*names, "row_size"} - set(data.files)):
            raise ValueError(f"{path}: not a start-position file of this format, missing {missing}; {rebuild}")
        loaded = {name: data[name] for name in names}
        row_size = int(data["row_size"])
    for name in ("max_players", "layout_hash", "engine_options"):
        loaded[name] = loaded[name].item()
    positions = StartPositions(**loaded)
    if positions.layout_hash != state_layout_hash():
        raise ValueError(f"{path}: state layout hash {positions.layout_hash} is not the engine's "
                         f"{state_layout_hash()}; {rebuild}")
    width = get_layout(positions.max_players).total_size
    if not row_size == positions.rows.shape[1] == width:
        raise ValueError(f"{path}: rows of {positions.rows.shape[1]} slots (row_size {row_size}) are not the "
                         f"engine's {width} at capacity {positions.max_players}; {rebuild}")
    if max_players is not None and positions.max_players != max_players:
        raise ValueError(f"{path}: rows at capacity {positions.max_players}, not the run's {max_players}; {rebuild}")
    if options is not None and json.loads(positions.engine_options) != dict(options):
        raise ValueError(f"{path}: positions filtered under engine flags {positions.engine_options}, not the "
                         f"run's {json.dumps(dict(options), sort_keys=True)}; {rebuild}")
    if not np.array_equal(positions.rows[:, _NUM_PLAYERS], positions.num_players):
        raise ValueError(f"{path}: the rows' player counts disagree with num_players")
    return positions
