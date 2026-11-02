"""Rolling Stock Stars adapter over the existing engine (core, entities, phases)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.actions import enumerate_policy_actions_py, get_decision_phase_py
from core.attention_relations import (
    MAX_ATTENTION_RELATION_EDGES,
    NUM_ATTENTION_RELATIONS,
)
from core.data import PHASE_ACTION_SIZES, DecisionPhase, GamePhases
from core.driver import DRIVER
from core.relations import (
    get_relation_coord_data,
    get_relation_coord_data_batch,
    get_relation_data,
    get_relation_data_batch,
)
from core.state import (
    GameState,
    get_layout,
    get_storage_player_capacity,
    get_turn_fields,
)
from core.token_data import (
    get_num_tokens,
    get_token_data,
    get_token_data_batch,
    get_token_dim,
)
from entities.player import PLAYERS
from entities.turn import TURN
from games import Game
from nn.policy_layout import PolicyLayout

if TYPE_CHECKING:
    from train.config import TrainingConfig


POLICY_LAYOUT = PolicyLayout(
    phase_sizes=tuple(int(size) for size in PHASE_ACTION_SIZES),
    # DecisionPhase order, in train/policy_metrics.py's TensorBoard spellings.
    phase_names=(
        "invest", "bid", "acq_corp", "acq_offer", "close", "div", "issue",
        "ipo", "par", "acq_co", "acq_price",
    ),
    phases_with_pass_slot=tuple(int(phase) for phase in (
        DecisionPhase.DPHASE_INVEST,
        DecisionPhase.DPHASE_BID,
        DecisionPhase.DPHASE_ACQ_SELECT_CORP,
        DecisionPhase.DPHASE_CLOSING,
        DecisionPhase.DPHASE_IPO,
    )),
)

# Only total_size depends on the player count.
_NUM_PLAYERS_OFFSET = get_layout(0).turn_offset + get_turn_fields().num_players
_PHASE_GAME_OVER = int(GamePhases.PHASE_GAME_OVER)


def _state_size(max_players: int) -> int:
    return get_layout(max_players).total_size


def _num_players_from_row(row: np.ndarray) -> int:
    return int(row[_NUM_PLAYERS_OFFSET])


def _scratch_state(root: GameState, row: np.ndarray) -> GameState:
    # The runtime flags run_search copies from its root; allow_positive_income_closing
    # and step_mode keep their defaults, as they do in search today.
    scratch = GameState.from_buffer(
        row, _num_players_from_row(row),
        max_players=get_storage_player_capacity(row.shape[0]),
        v3_behavior=root.v3_behavior,
    )
    scratch.acq_same_president = root.acq_same_president
    scratch.acq_offer_limits = root.acq_offer_limits
    scratch.max_acq_price_actions = root.max_acq_price_actions
    return scratch


def _is_terminal(state: GameState) -> bool:
    return TURN.get_phase(state) == _PHASE_GAME_OVER


def _final_scores(state: GameState) -> list[int]:
    return [
        PLAYERS[player_id].get_net_worth(state)
        for player_id in range(TURN.get_num_players(state))
    ]


def make_game(config: TrainingConfig) -> Game:
    """Build the RSS adapter for ``config``'s engine options."""
    v3_behavior = config.v3_behavior
    acq_same_president = config.acq_same_president
    max_acq_price_actions = config.max_acq_price_actions

    def new_game(num_players: int, max_players: int, seed: int) -> GameState:
        state = GameState(
            num_players, max_players=max_players, v3_behavior=v3_behavior,
            acq_same_president=acq_same_president,
            max_acq_price_actions=max_acq_price_actions,
        )
        state.initialize_game(num_players, seed=seed, max_players=max_players)
        return state

    return Game(
        name="rss",
        policy=POLICY_LAYOUT,
        state_size=_state_size,
        storage_player_capacity=get_storage_player_capacity,
        num_players_from_row=_num_players_from_row,
        new_game=new_game,
        scratch_state=_scratch_state,
        apply_action=DRIVER.apply_action,
        active_player=TURN.get_active_player,
        is_terminal=_is_terminal,
        decision_phase=get_decision_phase_py,
        enumerate_policy_actions=enumerate_policy_actions_py,
        final_scores=_final_scores,
        num_tokens=get_num_tokens,
        token_dim=get_token_dim,
        num_relations=NUM_ATTENTION_RELATIONS,
        max_relation_edges=MAX_ATTENTION_RELATION_EDGES,
        fill_tokens=get_token_data,
        fill_tokens_batch=get_token_data_batch,
        fill_relations_dense=get_relation_data,
        fill_relations_dense_batch=get_relation_data_batch,
        fill_relations_sparse=get_relation_coord_data,
        fill_relations_sparse_batch=get_relation_coord_data_batch,
    )
