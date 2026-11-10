"""Shikoku 1889 adapter over the g1889 engine (1889/engine-design.md §11) and
its extraction (``g1889.token_data``, ``g1889.relations``; 1889/extraction-design.md).

The model takes the static relation planes from
``g1889.relations.static_relation_planes`` itself (extraction-design §6.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import g1889.constants as K
from g1889.actions import (
    enumerate_policy_actions_py,
    final_scores_py,
    get_decision_phase_py,
    is_terminal_py,
)
from g1889.data.tables import NUM_PRIVATES
from g1889.driver import DRIVER
from g1889.entities.turn import TURN
from g1889.relations import (
    RelationId,
    RelationSize,
    get_relation_coord_data,
    get_relation_coord_data_batch,
    get_relation_data,
    get_relation_data_batch,
)
from g1889.state import (
    GameState,
    get_global_fields,
    get_layout,
    get_player_fields,
    get_private_fields,
    get_storage_player_capacity,
)
from g1889.token_data import (
    get_num_tokens,
    get_token_data,
    get_token_data_batch,
    get_token_dim,
)
from games import Game, StartRows
from games.g1889_selfplay import G1889SelfPlayStats
from games.g1889_starts import engine_options, load_start_positions
from nn.policy_layout import PolicyLayout

if TYPE_CHECKING:
    from train.config import TrainingConfig


POLICY_LAYOUT = PolicyLayout(
    phase_sizes=tuple(K.PHASE_ACTION_SIZES),
    # DecisionPhase order; engine-design.md §5.2 names the metrics.
    phase_names=(
        "auction", "bid", "stock", "par", "sell_amount", "private_pre", "track", "tile", "token", "div",
        "discard", "train", "select_train", "train_price", "emergency_sell_amount",
        "private_post", "private_price", "er_bonus", "port",
    ),
    # Slot 0 passes before the phase's other choices, as RSS's pass slots do;
    # DIVIDEND's withhold is a choice, not a pass, like RSS's zero dividend.
    phases_with_pass_slot=K.PHASES_WITH_PASS_SLOT,
)

# Only total_size depends on the player count.
_NUM_PLAYERS_OFFSET = get_layout(0).global_offset + get_global_fields().num_players


def _state_size(max_players: int) -> int:
    return get_layout(max_players).total_size


def _num_players_from_row(row: np.ndarray) -> int:
    return int(row[_NUM_PLAYERS_OFFSET])


def _scratch_state(root: GameState, row: np.ndarray) -> GameState:
    scratch = GameState.from_buffer(
        row, _num_players_from_row(row),
        max_players=get_storage_player_capacity(row.shape[0]),
    )
    scratch.private_sale_same_president = root.private_sale_same_president
    scratch.train_sale_same_president = root.train_sale_same_president
    scratch.auction_pass_payout_cap = root.auction_pass_payout_cap
    return scratch


def _game_start_rows(states: np.ndarray, max_players: int) -> np.ndarray:
    """Positions of the rows that are their game's first decision.

    A game opens in the auction's first normal turn: no private sold, no bid, no
    pass, no discount, and no payout. Its first choice changes one of those or
    moves to BID, so no later row looks like this. A 3-player game has a private
    out of play.
    """
    layout = get_layout(max_players)
    fields = get_global_fields()
    private, player = get_private_fields(), get_player_fields()
    global_row = states[:, layout.global_offset:]
    privates = states[:, layout.privates_offset:layout.privates_offset + private.size * NUM_PRIVATES]
    owners = privates[:, private.owner::private.size]
    players = states[:, layout.players_offset:].reshape(len(states), max_players, player.size)
    return np.flatnonzero(
        (global_row[:, fields.round] == K.ROUND_AUCTION)
        & (global_row[:, fields.step] == K.STEP_AUCTION)
        & (global_row[:, fields.auction_discount] == 0)
        & (global_row[:, fields.auction_pass_payouts] == 0)
        & np.isin(owners, (K.OWNER_UNSOLD, K.OWNER_NOT_IN_PLAY)).all(axis=1)
        & (players[:, :, player.bids:player.bids + NUM_PRIVATES] == 0).all(axis=(1, 2))
        & (players[:, :, player.passed] == 0).all(axis=1)
    )


def make_game(config: TrainingConfig) -> Game:
    """Build the 1889 adapter for ``config``'s engine options."""
    private_sale_same_president = config.private_sale_same_president
    train_sale_same_president = config.train_sale_same_president
    auction_pass_payout_cap = config.auction_pass_payout_cap

    def new_game(num_players: int, max_players: int, seed: int) -> GameState:
        state = GameState(
            num_players, max_players=max_players,
            private_sale_same_president=private_sale_same_president,
            train_sale_same_president=train_sale_same_president,
            auction_pass_payout_cap=auction_pass_payout_cap,
        )
        state.initialize_game(num_players, seed=seed, max_players=max_players)
        return state

    def load_starts(path: str) -> StartRows:
        positions = load_start_positions(Path(path), config.effective_max_players, engine_options(config))
        return StartRows(positions.rows, positions.num_players, positions.decision_index)

    return Game(
        name="1889",
        policy=POLICY_LAYOUT,
        state_size=_state_size,
        storage_player_capacity=get_storage_player_capacity,
        num_players_from_row=_num_players_from_row,
        new_game=new_game,
        scratch_state=_scratch_state,
        apply_action=DRIVER.apply_action,
        active_player=TURN.active_player,
        is_terminal=is_terminal_py,
        decision_phase=get_decision_phase_py,
        enumerate_policy_actions=enumerate_policy_actions_py,
        final_scores=final_scores_py,
        num_tokens=get_num_tokens,
        token_dim=get_token_dim,
        # A sample's planes are 262 KB against 16 KB of records (model-design.md §6.1).
        relation_format="flat",
        num_relations=int(RelationId.NUM_RELATIONS),
        max_relation_edges=int(RelationSize.MAX_RELATION_EDGES),
        fill_tokens=get_token_data,
        fill_tokens_batch=get_token_data_batch,
        fill_relations_dense=get_relation_data,
        fill_relations_dense_batch=get_relation_data_batch,
        fill_relations_sparse=get_relation_coord_data,
        fill_relations_sparse_batch=get_relation_coord_data_batch,
        new_selfplay_stats=G1889SelfPlayStats,
        game_start_rows=_game_start_rows,
        load_start_positions=load_starts,
    )
