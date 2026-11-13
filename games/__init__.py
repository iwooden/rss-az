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
- ``relation_format`` says how relations reach a token layout's model:
  ``"dense"`` games send planes from replay, the trainer, and ``NNEvaluator``
  and padded records from the eval server; ``"flat"`` games send flat records
  (``nn.model_contract.flatten_relation_records``) from all of them. It follows
  from the game's sizes, so it is a constant, not a config field.
- A packed layout (``packed_layouts``; 1889's layout 3) has no token rows and no
  relations: one float vector a state (``fill_inputs``), and the model's
  forward is ``(inputs, legal_mask)``. Its relation format is ``"none"``
  (``relation_format_of``). A game's token and packed layouts share its
  raw state, so evaluators of different layouts can play one game.
- Self-play diagnostics are a game hook (``SelfPlayStats``).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Protocol

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


class SelfPlayStats(Protocol):
    """One self-play game's diagnostics (``Game.new_selfplay_stats``).

    ``finish`` returns ``{name: (numerator, denominator)}``. train/main.py sums
    both over each reporting group's games (all games, and each player count)
    and logs ``self_play_<group>/<name>`` as numerator / max(denominator, 1).
    A denominator of 1 per game makes a mean over the games that report the
    name; event counts make a rate over the group's events.
    """

    def observe(
        self,
        state: Any,
        legal_actions: np.ndarray,
        chosen_idx: int,
        priors: np.ndarray,
        visit_counts: np.ndarray,
    ) -> None:
        """Record a played decision before it is applied. ``priors`` (the
        unnoised root priors) and ``visit_counts`` align with ``legal_actions``,
        the policy action set."""
        ...

    def finish(self, state: Any) -> dict[str, tuple[float, float]]:
        """This game's totals, from its terminal state."""
        ...


@dataclass(frozen=True)
class StartRows:
    """Positions self-play may start games from (``Game.load_start_positions``)."""

    rows: np.ndarray  # (N, state_size(max_players)) int16
    num_players: np.ndarray  # (N,) each row's actual player count
    # (N,) decisions before the row in its game, counted as play_game counts
    # moves: the move number a game started there begins at
    start_moves: np.ndarray


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

    # Token layouts' model inputs: tokens, and planes or records by
    # relation_format. Workers write records in either format. Batches take
    # raw rows, which may mix player counts within one capacity.
    num_tokens: Callable[[int], int]  # max_players -> token rows
    token_dim: Callable[[int], int]  # token layout version -> token width
    relation_format: str  # a token layout's: "dense" or "flat"
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

    # Self-play diagnostics: num_players -> one game's collector
    new_selfplay_stats: Callable[[int], SelfPlayStats]

    # Packed layouts' model inputs: one float32 vector a state, no token rows
    # and no relations. A game without them leaves these empty.
    packed_layouts: frozenset[int] = frozenset()
    # (max_players, *, layout_version) -> the vector's length
    input_size: Callable[..., int] | None = None
    # (state, float32 (input_size,) buf, max_players=0, *, layout_version)
    fill_inputs: Callable[..., None] | None = None
    # (rows, float32 (n, input_size) buf, max_players=0, *, layout_version)
    fill_inputs_batch: Callable[..., None] | None = None

    # --- 1889-only; None for other games ---
    # (rows in write order, max_players) -> positions of the rows that are their
    # game's first decision (train/capacity_probe.py)
    game_start_rows: Callable[[np.ndarray, int], np.ndarray] | None = None
    # path -> the file's positions for self-play to start games from, checked
    # against the engine's layout and the config's player capacity and engine
    # options (train/main.py, start_positions_path)
    load_start_positions: Callable[[str], StartRows] | None = None

    # --- RSS-only; None for other games ---
    # num_players -> strategy-trace builder, for
    # play_game(collect_strategy_trace=True) (train/strategy_data.py)
    new_strategy_trace: Callable[[int], Any] | None = None

    def relation_format_of(self, layout_version: int) -> str:
        """How relations reach a model of ``layout_version``: ``"none"`` for a
        packed layout, else the game's ``relation_format``."""
        return "none" if layout_version in self.packed_layouts else self.relation_format


def get_game(config: TrainingConfig) -> Game:
    """Build the adapter for ``config.game`` with ``config``'s engine options."""
    module = _GAME_MODULES.get(config.game)
    if module is None:
        raise ValueError(
            f"Unknown game {config.game!r}; registered games: {', '.join(GAME_NAMES)}"
        )
    return import_module(module).make_game(config)
