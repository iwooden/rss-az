"""Game adapters: what search, self-play, evaluation, and training need from a game.

Each process builds its own ``Game`` from its config with ``get_game``, which
imports the game's module, and with it the engine, on first use. Importing
this package imports no engine.

Conventions every game follows:

- ``apply_action`` returns ``STATUS_OK``, ``STATUS_GAME_OVER``, or
  ``STATUS_INVALID``.
- Action ids are phase-local and fit ``uint16``. ``policy`` maps them to
  unified slots.
- A state exposes its raw ``int16`` row as ``_array`` and its player capacity
  as ``max_players``. ``state.rebind(row, num_players, max_players=...)``
  repoints it at another row without allocating, keeping its runtime flags.
- Scores and values are per player, in canonical order.
- Sparse relation records are ``uint8`` rows of ``RELATION_RECORD_WIDTH``:
  ``(relation, query token, key token, value)``. Rows past the count are zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np

    from nn.policy_layout import PolicyLayout
    from train.config import TrainingConfig


STATUS_OK = 0
STATUS_GAME_OVER = 1
STATUS_INVALID = 2

RELATION_RECORD_WIDTH = 4

_GAME_MODULES = {"rss": "games.rss", "1889": "games.g1889"}
GAME_NAMES = tuple(_GAME_MODULES)


@dataclass(frozen=True)
class Game:
    """One game's constants and engine entry points."""

    name: str  # registry key; stored in configs and checkpoints
    policy: PolicyLayout

    # Raw state: int16 rows padded to a player capacity
    state_size: Callable[[int], int]  # max_players -> row width
    storage_player_capacity: Callable[[int], int]  # row width -> max_players
    num_players_from_row: Callable[[np.ndarray], int]  # row -> actual player count
    # (num_players, max_players, seed) -> initial state with the config's
    # runtime flags
    new_game: Callable[[int, int, int], Any]
    # (root, row) -> state over row carrying root's runtime flags. Hot paths
    # build one per search and rebind it across pool rows.
    scratch_state: Callable[[Any, np.ndarray], Any]

    # Engine
    # (state, action_id, history=None) -> status. Validates, dispatches, and
    # auto-chains.
    apply_action: Callable[..., int]
    active_player: Callable[[Any], int]
    is_terminal: Callable[[Any], bool]
    decision_phase: Callable[[Any], int]
    # (state, uint16 buf of policy.max_action_size) -> count. The searched set,
    # a subset of what the driver accepts: evaluator priors, search expansion,
    # self-play, and replay masks and targets. Restrictions come from the
    # state's runtime flags.
    enumerate_policy_actions: Callable[[Any, np.ndarray], int]
    final_scores: Callable[[Any], list[int]]  # one per player; feeds terminal values

    # Model inputs: dense planes for trainer/NNEvaluator, sparse records for
    # eval IPC. Batches take raw rows, which may mix player counts within one
    # capacity.
    num_tokens: Callable[[int], int]  # max_players -> token rows
    token_dim: Callable[[int], int]  # token layout version -> token width
    num_relations: int
    max_relation_edges: int  # sparse record rows per state
    # (state | rows, float32 buf, *, max_players, layout_version)
    fill_tokens: Callable[..., None]
    fill_tokens_batch: Callable[..., None]
    # (state | rows, uint8 (num_relations, T, T) planes, *, max_players)
    fill_relations_dense: Callable[..., None]
    fill_relations_dense_batch: Callable[..., None]
    # (state | rows, uint8 (max_relation_edges, RELATION_RECORD_WIDTH)
    # records, *, max_players); the single-state form returns the count
    fill_relations_sparse: Callable[..., int]
    fill_relations_sparse_batch: Callable[..., None]


def get_game(config: TrainingConfig) -> Game:
    """Build the adapter for ``config.game`` with ``config``'s engine options."""
    module = _GAME_MODULES.get(config.game)
    if module is None:
        raise ValueError(
            f"Unknown game {config.game!r}; registered games: {', '.join(GAME_NAMES)}"
        )
    return import_module(module).make_game(config)
