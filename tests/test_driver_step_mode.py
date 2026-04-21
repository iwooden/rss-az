from core.actions import (
    ACTION_ACQ_OFFER_ACCEPT_PY as ACTION_ACQ_OFFER_ACCEPT,
    ACTION_AUCTION_PY as ACTION_AUCTION,
    ACTION_RAISE_PY as ACTION_RAISE,
)
from core.data import GamePhases
from core.driver import (
    DRIVER,
    STATUS_INVALID_PY as STATUS_INVALID,
    STATUS_OK_PY as STATUS_OK,
    STATUS_PAUSED_PY as STATUS_PAUSED,
)
from core.state import GameState
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES

from tests.phases.conftest import (
    apply_and_verify,
    draw_to_fi,
    find_legal_action,
    float_corp_for_test,
    get_legal_actions,
    make_auto_phase_state,
)


PHASE_WRAP_UP = int(GamePhases.PHASE_WRAP_UP)
PHASE_ACQ_SELECT_CORP = int(GamePhases.PHASE_ACQ_SELECT_CORP)
PHASE_CLOSING = int(GamePhases.PHASE_CLOSING)
PHASE_ACQ_OFFER = int(GamePhases.PHASE_ACQ_OFFER)


def _make_acq_offer_accept_state():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    # Needed for ``test_apply_action_without_step_mode_auto_chains_to_closing``
    # to pause in CLOSING. The CoO=1 default would leave every private with
    # positive adjusted income, so the training-default legality gate would
    # short-circuit CLOSING straight through to INCOME.
    state.allow_positive_income_closing = True

    fi_co = draw_to_fi(state)
    # Use a valid FI-preemption setup: preemptor and original buyer have
    # different presidents.
    float_corp_for_test(state, corp_id=0, player_id=1, par_index=10)
    CORPS[0].set_cash(state, 200)
    float_corp_for_test(state, corp_id=1, player_id=0, par_index=12)
    CORPS[1].set_cash(state, 200)

    price = COMPANIES[fi_co].get_high_price()
    TURN.enter_acq_offer(
        state,
        offered_corp=0,
        company_id=fi_co,
        price=price,
        original_corp=1,
        deciding_player=1,
    )
    accept_id = find_legal_action(state, action_type=ACTION_ACQ_OFFER_ACCEPT)
    return state, accept_id


def _enter_bid_phase_with_opening_bid(state, offset=0):
    """Select a company in INVEST and have the starter place the opening bid."""
    actions = get_legal_actions(state)
    for action_id, info in actions:
        if info.action_type == ACTION_AUCTION:
            company_id = info.company_id
            apply_and_verify(state, action_id)
            # Opening bid at face_value + offset.
            raise_id = find_legal_action(state, action_type=ACTION_RAISE, amount=offset)
            apply_and_verify(state, raise_id)
            bid_price = COMPANIES[company_id].get_face_value() + offset
            return company_id, bid_price
    raise AssertionError("No auction action found in INVEST")


def _make_forced_bid_state():
    """Make a BID state with exactly one legal action.

    Enters BID via INVEST auction-select, has the starter place the opening
    bid at face_value (offset 0), then pins the next bidder's cash to the
    bid price — they can't raise (need cash > bid_price), so only leave
    remains legal.
    """
    state = GameState(3)
    state.initialize_game(3, seed=42)
    _, bid_price = _enter_bid_phase_with_opening_bid(state)
    active = TURN.get_active_player(state)
    PLAYERS[active].set_cash(state, bid_price)
    actions = get_legal_actions(state)
    assert len(actions) == 1
    return state


def test_apply_action_in_step_mode_pauses_before_auto_chain():
    state, accept_id = _make_acq_offer_accept_state()
    state.step_mode = True
    history = []

    status = DRIVER.apply_action(state, accept_id, history=history)

    assert status == STATUS_PAUSED
    assert TURN.get_phase(state) == PHASE_ACQ_SELECT_CORP
    assert TURN.get_active_player(state) == 0
    assert TURN.get_active_corp(state) == -1
    assert len(history) == 1


def test_apply_action_without_step_mode_auto_chains_to_closing():
    state, accept_id = _make_acq_offer_accept_state()
    history = []

    status = DRIVER.apply_action(state, accept_id, history=history)

    assert status == STATUS_OK
    assert TURN.get_phase(state) == PHASE_CLOSING
    assert TURN.get_active_player(state) == 1
    assert TURN.get_active_corp(state) == -1
    assert len(history) > 1


def test_is_non_player_phase_false_for_multi_choice_decision_state():
    state = GameState(3)
    state.initialize_game(3, seed=42)

    assert TURN.get_phase(state) != PHASE_WRAP_UP
    assert not DRIVER.is_non_player_phase(state)


def test_is_non_player_phase_true_for_forced_single_action_state():
    state = _make_forced_bid_state()

    assert TURN.get_phase(state) != PHASE_ACQ_OFFER
    assert DRIVER.is_non_player_phase(state)


def test_advance_phase_executes_one_automated_phase_and_pauses():
    state = make_auto_phase_state(3, PHASE_WRAP_UP)
    float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
    state.step_mode = True
    history = []

    status = DRIVER.advance_phase(state, history=history)

    assert status == STATUS_PAUSED
    assert TURN.get_phase(state) == PHASE_ACQ_SELECT_CORP
    assert TURN.get_active_player(state) == 0
    assert TURN.get_active_corp(state) == -1
    assert len(history) == 1
    assert history[0][1] == -1
    assert history[0][2] == PHASE_WRAP_UP


def test_advance_phase_rejects_real_player_decision_state():
    state = GameState(3)
    state.initialize_game(3, seed=42)

    assert DRIVER.advance_phase(state) == STATUS_INVALID
