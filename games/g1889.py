"""Shikoku 1889 adapter over the g1889 engine (1889/engine-design.md §11).

The engine fields are complete. Extraction lands in pipeline step 7: until
then its callables raise NotImplementedError, and its sizes are ``NO_SIZE``,
which any buffer or layer sized from them rejects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import numpy as np

import g1889.constants as K
from g1889.actions import (
    enumerate_policy_actions_py,
    final_scores_py,
    get_decision_phase_py,
    is_terminal_py,
)
from g1889.driver import DRIVER
from g1889.entities.turn import TURN
from g1889.state import (
    GameState,
    get_global_fields,
    get_layout,
    get_storage_player_capacity,
)
from games import EmptySelfPlayStats, Game
from nn.policy_layout import PolicyLayout

if TYPE_CHECKING:
    from train.config import TrainingConfig


POLICY_LAYOUT = PolicyLayout(
    phase_sizes=tuple(K.PHASE_ACTION_SIZES),
    # DecisionPhase order; engine-design.md §5.2 names the metrics.
    phase_names=(
        "auction", "bid", "stock", "par", "sell_amount", "track", "tile", "token", "div",
        "discard", "train", "select_train", "train_price", "emergency_sell_amount",
        "private", "private_price", "er_bonus", "port",
    ),
    # Slot 0 passes before the phase's other choices, as RSS's pass slots do;
    # DIVIDEND's withhold is a choice, not a pass, like RSS's zero dividend.
    phases_with_pass_slot=K.PHASES_WITH_PASS_SLOT,
)

EXTRACTION_STEP = "1889 extraction lands in pipeline step 7"
NO_SIZE = -1  # num_relations and max_relation_edges until step 7

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


def _no_extraction(*args: object, **kwargs: object) -> NoReturn:
    raise NotImplementedError(EXTRACTION_STEP)


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
        num_tokens=_no_extraction,
        token_dim=_no_extraction,
        num_relations=NO_SIZE,
        max_relation_edges=NO_SIZE,
        fill_tokens=_no_extraction,
        fill_tokens_batch=_no_extraction,
        fill_relations_dense=_no_extraction,
        fill_relations_dense_batch=_no_extraction,
        fill_relations_sparse=_no_extraction,
        fill_relations_sparse_batch=_no_extraction,
        new_selfplay_stats=EmptySelfPlayStats,
    )
