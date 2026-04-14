from core.data import GamePhases
from entities.player import PLAYERS
from entities.corp import CORPS

from tests.phases.conftest import make_auto_phase_state, float_corp_for_test
from tests.phases.helpers.finance import (
    set_player_cashs,
    set_corp_cashs,
    prime_corp_income_for_test,
)


def test_set_player_cashs_updates_multiple_players():
    state = make_auto_phase_state(3, int(GamePhases.PHASE_INCOME))  # exact phase irrelevant here

    set_player_cashs(state, {0: 111, 2: 222})

    assert PLAYERS[0].get_cash(state) == 111
    assert PLAYERS[1].get_cash(state) != 111
    assert PLAYERS[2].get_cash(state) == 222


def test_set_corp_cashs_updates_multiple_corps():
    state = make_auto_phase_state(3, int(GamePhases.PHASE_INCOME))
    float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
    float_corp_for_test(state, corp_id=1, player_id=0, par_index=12)

    set_corp_cashs(state, {0: 77, 1: 88})

    assert CORPS[0].get_cash(state) == 77
    assert CORPS[1].get_cash(state) == 88


def test_prime_corp_income_for_test_overrides_cached_income_after_refresh():
    state = make_auto_phase_state(3, int(GamePhases.PHASE_INCOME))
    float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
    CORPS[0].set_cash(state, 20)

    prime_corp_income_for_test(state, corp_id=0, income=7)

    assert CORPS[0].get_income(state) == 7
