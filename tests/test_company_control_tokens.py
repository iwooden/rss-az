"""Actor-relative company control across phases, owners, and padded batches."""

import numpy as np
import pytest

from core.data import GamePhases
from core.state import GameState
from core.token_data import get_num_tokens, get_token_data, get_token_data_batch, get_token_dim
from entities.company import COMPANIES
from entities.player import PLAYERS
from entities.turn import TURN
from tests.phases.conftest import float_corp_for_test, setup_receivership_corp
from tests.phases.helpers.ownership import give_company_to_fi, give_company_to_player
from train.debug_trace import format_token_dump


def _state(n, v3_behavior):
    state = GameState(n, max_players=5, v3_behavior=v3_behavior)
    state.initialize_game(n, seed=42, max_players=5)
    float_corp_for_test(state, 0, company_id=0, player_id=n - 1, par_index=10)
    float_corp_for_test(state, 1, company_id=1, player_id=0, par_index=12)
    setup_receivership_corp(state, 2, [2], par_index=14)
    give_company_to_player(state, 3, n - 1)
    give_company_to_player(state, 4, 0)
    for company, corp in ((5, 0), (6, 1)):
        give_company_to_player(state, company, 0)
        COMPANIES[company].transfer_to_corp_acquisition(state, corp)
    give_company_to_fi(state, 7)
    for company, move in ((8, "remove_from_game"), (9, "move_to_auction"), (10, "mark_revealed")):
        give_company_to_player(state, company, 0)
        getattr(COMPANIES[company], move)(state)
    TURN.set_active_corp(state, 0)
    TURN.set_active_company(state, 3)
    return state


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("v3_behavior", [False, True])
def test_control_flag_tracks_actor_in_every_owner_kind_and_multiple_phases(n, v3_behavior):
    state = _state(n, v3_behavior)
    raw = np.empty((get_num_tokens(5), get_token_dim(3)), np.float32)
    arrays, expected_rows = [], []
    for phase in (
        GamePhases.PHASE_INVEST, GamePhases.PHASE_ACQ_SELECT_CORP,
        GamePhases.PHASE_ACQ_SELECT_COMPANY, GamePhases.PHASE_ACQ_SELECT_PRICE,
        GamePhases.PHASE_CLOSING, GamePhases.PHASE_INCOME,
    ):
        TURN.set_phase(state, int(phase))
        for actor, controlled in ((n - 1, [0, 3, 5]), (0, [1, 4, 6]), (1, [])):
            TURN.set_active_player(state, actor)
            get_token_data(state, raw, max_players=5, layout_version=3)
            expected = np.zeros(len(COMPANIES), np.float32)
            expected[controlled] = 1
            np.testing.assert_array_equal(raw[1:37, 15], expected)
            assert not raw[1:37, 30:].any()
            arrays.append(state._array.copy())
            expected_rows.append(raw.copy())
    batch = np.empty((len(arrays), *raw.shape), np.float32)
    get_token_data_batch(arrays, batch, max_players=5, layout_version=3)
    np.testing.assert_array_equal(batch, expected_rows)


def test_control_flag_updates_after_presidency_and_ownership_changes():
    state = _state(3, True)
    TURN.set_phase(state, int(GamePhases.PHASE_INVEST))
    TURN.set_active_player(state, 2)
    raw = np.empty((get_num_tokens(5), get_token_dim(3)), np.float32)
    get_token_data(state, raw, max_players=5, layout_version=3)
    np.testing.assert_array_equal(raw[[1, 4, 6], 15], [1, 1, 1])
    PLAYERS[2].set_shares(state, 0, 0)  # Corporation enters receivership.
    COMPANIES[3].transfer_to_fi(state)
    get_token_data(state, raw, max_players=5, layout_version=3)
    assert not raw[1:37, 15].any()
    PLAYERS[0].set_shares(state, 0, 1)  # A different player becomes president.
    TURN.set_active_player(state, 0)
    get_token_data(state, raw, max_players=5, layout_version=3)
    np.testing.assert_array_equal(raw[[1, 4, 6], 15], [1, 0, 1])
    dump = format_token_dump(state, layout_version=3)
    assert "actor_controls_company=1" in dump
    assert "actor_controls_company=0" in dump
    assert "actor_controls_company" not in format_token_dump(state, layout_version=2)
