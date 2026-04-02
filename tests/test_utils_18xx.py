"""Tests for shared 18xx integration helpers."""

from core.data import COMPANY_NAMES, CORP_NAMES, GamePhases, get_company_low_price
from core.state import GameState
from entities.company import COMPANIES
from entities.corp import CORPS
from entities.player import PLAYERS
from entities.turn import TURN
from phases.acquisition import setup_acquisition_phase_py
from tests.phases.conftest import assert_invariants, float_corp_for_test
from utils_18xx.game_session import GameSession


def test_sync_acq_round_respects_declined_offer() -> None:
    """Declined ACQ offers should not transfer the company."""
    state = GameState(3)
    state.initialize_game()

    company_id = 0
    buyer_corp_id = 0
    seller_player_id = 0

    COMPANIES[company_id].transfer_to_player(state, seller_player_id)
    float_corp_for_test(state, buyer_corp_id, player_id=seller_player_id)
    CORPS[buyer_corp_id].set_cash(state, 40)

    TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
    setup_acquisition_phase_py(state)

    session = GameSession(3)
    price = get_company_low_price(company_id)
    actions = [
        {
            "type": "offer",
            "corporation": CORP_NAMES[buyer_corp_id],
            "company": COMPANY_NAMES[company_id],
            "price": price,
        },
        {
            "type": "respond",
            "accept": "false",
        },
    ]

    next_idx = session._sync_acq_round(state, actions, 0)

    assert next_idx == 2
    assert TURN.get_phase(state) != GamePhases.PHASE_ACQUISITION
    assert COMPANIES[company_id].is_owned_by_player(state, seller_player_id)
    assert not COMPANIES[company_id].is_owned_by_corp(state, buyer_corp_id)
    assert_invariants(state, "After declined ACQ replay")


def test_sync_acq_round_patches_cross_president_acceptance() -> None:
    """Accepted cross-president ACQ offers should patch at the ACQ boundary."""
    state = GameState(3)
    state.initialize_game()

    company_id = 0
    buyer_corp_id = 0
    buyer_president_id = 0
    seller_player_id = 1

    COMPANIES[company_id].transfer_to_player(state, seller_player_id)
    float_corp_for_test(state, buyer_corp_id, player_id=buyer_president_id)
    CORPS[buyer_corp_id].set_cash(state, 40)

    seller_cash_before = PLAYERS[seller_player_id].get_cash(state)
    TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
    setup_acquisition_phase_py(state)

    session = GameSession(3)
    price = get_company_low_price(company_id)
    actions = [
        {
            "type": "offer",
            "corporation": CORP_NAMES[buyer_corp_id],
            "company": COMPANY_NAMES[company_id],
            "price": price,
        },
        {
            "type": "respond",
            "accept": "true",
        },
    ]

    next_idx = session._sync_acq_round(state, actions, 0)

    assert next_idx == 2
    assert TURN.get_phase(state) == GamePhases.PHASE_CLOSING
    assert COMPANIES[company_id].is_owned_by_corp(state, buyer_corp_id)
    assert CORPS[buyer_corp_id].get_cash(state) == 40 - price
    assert PLAYERS[seller_player_id].get_cash(state) == seller_cash_before + price
    assert_invariants(state, "After cross-president ACQ patch replay")


def test_sync_clo_round_patches_positive_income_close_at_pause_boundary() -> None:
    """Positive-income closes should patch at the paused CLO boundary."""
    state = GameState(3)
    state.initialize_game()
    state.pause_before_closing_transition = True
    TURN.set_coo_level(state, 1)

    company_id = 0
    COMPANIES[company_id].transfer_to_player(state, 0)
    assert COMPANIES[company_id].get_adjusted_income(state) > 0

    TURN.set_phase(state, GamePhases.PHASE_CLOSING)

    session = GameSession(3)
    actions = [
        {
            "type": "sell_company",
            "company": COMPANY_NAMES[company_id],
        },
    ]

    next_idx = session._sync_clo_round(state, actions, 0)

    assert next_idx == 1
    assert TURN.get_phase(state) == GamePhases.PHASE_INCOME
    assert COMPANIES[company_id].is_removed(state)
    assert_invariants(state, "After positive-income CLO patch replay")
