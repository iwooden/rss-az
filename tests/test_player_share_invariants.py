import pytest

from core.state import GameState
from entities import COMPANIES, CORPS, PLAYERS


def _float_corp(state, corp_id=0, player_id=0, company_id=0, par_index=15, float_shares=1):
    COMPANIES[company_id].transfer_to_player(state, player_id)
    CORPS[corp_id].float_corp(state, player_id, company_id, par_index, float_shares)


def test_set_shares_rejects_negative_shares():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    _float_corp(state)

    with pytest.raises(AssertionError, match="non-negative"):
        PLAYERS[0].set_shares(state, 0, -1)


def test_set_shares_rejects_inactive_corp():
    state = GameState(3)
    state.initialize_game(3, seed=42)

    with pytest.raises(AssertionError, match="inactive corp"):
        PLAYERS[0].set_shares(state, 0, 1)


def test_set_shares_rejects_more_than_issued():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    _float_corp(state)

    with pytest.raises(AssertionError, match="exceed issued shares"):
        PLAYERS[1].set_shares(state, 0, 3)


def test_set_shares_rejects_bank_underflow():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    _float_corp(state)
    corp = CORPS[0]

    corp.set_issued_shares(state, 7)
    corp.set_unissued_shares(state, 0)
    corp.set_bank_shares(state, 1)

    with pytest.raises(AssertionError, match="bank_shares=-1"):
        PLAYERS[1].set_shares(state, 0, 2)
