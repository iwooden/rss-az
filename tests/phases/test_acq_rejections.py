"""Negotiation makes finite progress in v3 while replay retains legal prices."""

import numpy as np
import pytest

from core.actions import enumerate_policy_actions_py
from core.data import GameConstants, GamePhases, MAX_ACTION_SIZE
from core.driver import DRIVER, STATUS_INVALID_PY
from core.state import GameState
from entities.company import COMPANIES, CompanyLocation
from entities.corp import CORPS
from entities.player import PLAYERS
from entities.turn import TURN
from phases.acq_select_corp import setup_acquisition_phase_py
from tests.phases.conftest import float_corp_for_test, get_legal_actions
from tests.phases.helpers.ownership import (
    give_company_to_corp, give_company_to_fi, give_company_to_player,
)

TARGET = 14


def negotiation_state(n=3, v3=True, seller_kind="player"):
    state = GameState(n, acq_same_president=False, v3_behavior=v3)
    state.initialize_game(n, seed=42)
    state.step_mode = True
    state.allow_positive_income_closing = True
    # Two buyers with the same president must share their rejection history.
    for corp, company, par in ((0, 0, 10), (1, 1, 12)):
        float_corp_for_test(state, corp_id=corp, company_id=company, player_id=n - 1, par_index=par)
        CORPS[corp].set_cash(state, COMPANIES[TARGET].get_high_price())
    if seller_kind == "corp":
        float_corp_for_test(state, corp_id=2, company_id=2, player_id=0, par_index=14)
        give_company_to_corp(state, TARGET, 2)
    else:
        give_company_to_player(state, TARGET, 0)
    setup_acquisition_phase_py(state)
    TURN.set_active_player(state, n - 1)
    return state


def select_target(state, corp=0, target=TARGET):
    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_CORP)
    assert 1 + corp in [aid for aid, _ in get_legal_actions(state)]
    DRIVER.apply_action(state, 1 + corp)
    assert target in [aid for aid, _ in get_legal_actions(state)]
    DRIVER.apply_action(state, target)
    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_PRICE)


@pytest.mark.parametrize("v3", [False, True])
@pytest.mark.parametrize("seller_kind", ["player", "corp"])
@pytest.mark.parametrize("same_president", [False, True])
@pytest.mark.parametrize("cash_limit", ["minimum", "middle", "below_maximum", "maximum", "above_maximum"])
def test_cross_president_prices_require_full_company_maximum(
    v3, seller_kind, same_president, cash_limit,
):
    state = negotiation_state(v3=v3, seller_kind=seller_kind)
    company = COMPANIES[TARGET]
    low, high = company.get_low_price(), company.get_high_price()
    cash = {"minimum": low, "middle": low + 3, "below_maximum": high - 1,
            "maximum": high, "above_maximum": high + 20}[cash_limit]
    CORPS[0].set_cash(state, cash)
    # Locked acquisition proceeds are not available to fund an offer.
    CORPS[0].set_acquisition_proceeds(state, 100)
    if same_president:
        if seller_kind == "corp":
            give_company_to_corp(state, TARGET, 1)
        else:
            give_company_to_player(state, TARGET, 2)
    restricted = v3 and not same_president
    if restricted and cash < high:
        if seller_kind == "corp":
            # Retain the seller's second company, but make it unavailable so
            # this buyer has no cheaper alternative cross-president target.
            COMPANIES[2].transfer_to_corp_acquisition(state, 2)
        assert 1 not in [aid for aid, _ in get_legal_actions(state)]
        # A legal alternative makes company selection reachable; the foreign
        # target must still be filtered out, despite ample locked proceeds.
        give_company_to_player(state, 3, 2)
        assert 1 in [aid for aid, _ in get_legal_actions(state)]
        DRIVER.apply_action(state, 1)
        assert TARGET not in [aid for aid, _ in get_legal_actions(state)]
        before = state._array.copy()
        assert DRIVER.apply_action(state, TARGET) == STATUS_INVALID_PY
        np.testing.assert_array_equal(state._array, before)
        return
    select_target(state)
    max_offset = min(cash, high) - low
    expected = [max_offset] if restricted else list(range(max_offset + 1))
    assert [aid for aid, _ in get_legal_actions(state)] == expected
    buf = np.empty(MAX_ACTION_SIZE, dtype=np.uint16)
    for cap in (0, 8):
        count = enumerate_policy_actions_py(state, buf, cap)
        capped = expected if cap == 0 or len(expected) <= cap else expected[:4] + expected[-4:]
        np.testing.assert_array_equal(buf[:count], capped)
    if restricted and max_offset > 0:
        before = state._array.copy()
        assert DRIVER.apply_action(state, max_offset - 1) == STATUS_INVALID_PY
        np.testing.assert_array_equal(state._array, before)
    if restricted:
        DRIVER.apply_action(state, max_offset)
        assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_OFFER)
        assert TURN.get_acq_offer_price(state) == high
        assert TURN.get_active_player(state) == 0


@pytest.mark.parametrize("seller_kind", ["player", "corp"])
def test_cross_president_maximum_price_auto_chains_to_seller(seller_kind):
    state = negotiation_state(seller_kind=seller_kind)
    CORPS[0].set_cash(state, COMPANIES[TARGET].get_high_price() + 20)
    DRIVER.apply_action(state, 1)  # Select buyer while single-step mode is on.
    state.step_mode = False
    DRIVER.apply_action(state, TARGET)
    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_OFFER)
    assert TURN.get_active_player(state) == 0
    assert TURN.get_acq_offer_price(state) == COMPANIES[TARGET].get_high_price()


@pytest.mark.parametrize("buyer", [0, 2])  # OS is corporation 2.
@pytest.mark.parametrize("seller_kind", ["player", "fi"])
@pytest.mark.parametrize("cash_at_high", [False, True])
def test_only_os_buying_from_fi_gets_face_value_discount(buyer, seller_kind, cash_at_high):
    state = negotiation_state()
    company = COMPANIES[TARGET]
    if buyer == 2:
        float_corp_for_test(state, corp_id=2, company_id=2, player_id=2, par_index=14)
    cash = company.get_high_price() if cash_at_high else company.get_face_value()
    CORPS[buyer].set_cash(state, cash)
    CORPS[1].set_cash(state, 0)  # No FI intervention from the other buyer.
    if seller_kind == "fi":
        give_company_to_fi(state, TARGET)
    give_company_to_player(state, 3, 2)  # Always an affordable alternate target.
    DRIVER.apply_action(state, 1 + buyer)
    discount = buyer == 2 and seller_kind == "fi"
    assert (TARGET in [aid for aid, _ in get_legal_actions(state)]) == (cash_at_high or discount)
    if cash_at_high or discount:
        DRIVER.apply_action(state, TARGET)
        if seller_kind == "fi":
            price = company.get_face_value() if discount else company.get_high_price()
            assert CORPS[buyer].get_cash(state) == cash - price
            assert company.get_location(state) == int(CompanyLocation.LOC_CORP_ACQ)
        else:
            assert [aid for aid, _ in get_legal_actions(state)] == [
                company.get_high_price() - company.get_low_price(),
            ]


@pytest.mark.parametrize("n", [3, 6])
@pytest.mark.parametrize("v3", [False, True])
@pytest.mark.parametrize("seller_kind", ["player", "corp"])
def test_rejections_track_buyer_share_across_corps_and_gate_only_v3(n, v3, seller_kind):
    state = negotiation_state(n, v3, seller_kind)
    company = COMPANIES[TARGET]
    price = company.get_high_price() if v3 else company.get_low_price() + 1
    CORPS[0].set_cash(state, price)
    CORPS[1].set_cash(state, company.get_high_price() + 20 if v3 else company.get_low_price() + 3)
    select_target(state)
    DRIVER.apply_action(state, price - company.get_low_price())
    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_OFFER)
    assert TURN.get_active_player(state) == 0
    DRIVER.apply_action(state, 0)  # Seller rejects.
    assert company.get_max_rejected_price(state, n - 1) == price
    assert company.get_max_rejected_price(state, 0) == 0
    assert PLAYERS[n - 1].get_acq_rejections(state) == 1
    assert PLAYERS[0].get_acq_rejections(state) == 0
    assert TURN.get_active_player(state) == n - 1

    for clone in (
        GameState.from_array(state._array.copy(), n),
        GameState.from_buffer(state._array.copy(), n),
    ):
        assert company.get_max_rejected_price(clone, n - 1) == price
        assert PLAYERS[n - 1].get_acq_rejections(clone) == 1
    if v3:
        # Even a richer corporation with the same president cannot retry a
        # rejected full-price offer for this company in the same phase.
        if seller_kind == "corp":
            DRIVER.apply_action(state, 2)  # Seller's other company is still available.
            assert TARGET not in [aid for aid, _ in get_legal_actions(state)]
        else:
            assert [aid for aid, _ in get_legal_actions(state)] == [0]
    else:
        select_target(state, corp=1)
        assert [aid for aid, _ in get_legal_actions(state)] == [0, 1, 2, 3]
        # A lower historical offer remains legal, and cannot lower the maximum.
        DRIVER.apply_action(state, 0)
        DRIVER.apply_action(state, 0)
        assert company.get_max_rejected_price(state, n - 1) == price
        assert PLAYERS[n - 1].get_acq_rejections(state) == 2


def test_exhausted_prices_remove_targets_and_corps_then_cleanup_resets_history():
    state = negotiation_state()
    company = COMPANIES[TARGET]
    rejected_price = company.get_high_price()
    for corp in (0, 1):
        CORPS[corp].set_cash(state, rejected_price)
    select_target(state)
    DRIVER.apply_action(state, rejected_price - company.get_low_price())
    DRIVER.apply_action(state, 0)
    assert company.get_max_rejected_price(state, 2) == rejected_price
    assert [aid for aid, _ in get_legal_actions(state)] == [0]

    # Both selection levels must agree with the price mask.
    TURN.set_active_corp(state, 0)
    TURN.set_phase(state, int(GamePhases.PHASE_ACQ_SELECT_COMPANY))
    # Give this buyer another reachable target, so company enumeration is nonempty.
    give_company_to_player(state, 3, 0)
    assert TARGET not in [aid for aid, _ in get_legal_actions(state)]
    TURN.clear_active_corp(state)
    TURN.set_phase(state, int(GamePhases.PHASE_ACQ_SELECT_CORP))
    DRIVER.apply_action(state, 0)
    assert TURN.get_phase(state) == int(GamePhases.PHASE_CLOSING)
    assert all(c.get_max_rejected_price(state, p) == 0
               for c in COMPANIES for p in range(3))
    assert all(p.get_acq_rejections(state) == 0 for p in PLAYERS[:3])
    setup_acquisition_phase_py(state)
    select_target(state)
    assert [aid for aid, _ in get_legal_actions(state)] == [rejected_price - company.get_low_price()]


def test_accepted_offer_does_not_record_rejection_and_company_cannot_be_resold():
    state = negotiation_state()
    select_target(state)
    DRIVER.apply_action(state, COMPANIES[TARGET].get_high_price() - COMPANIES[TARGET].get_low_price())
    DRIVER.apply_action(state, 1)
    company = COMPANIES[TARGET]
    assert company.get_max_rejected_price(state, 2) == 0
    assert PLAYERS[2].get_acq_rejections(state) == 0
    assert company.get_location(state) == int(CompanyLocation.LOC_CORP_ACQ)
    DRIVER.apply_action(state, 2)  # Other corp can buy an older company, but not this one.
    assert TARGET not in [aid for aid, _ in get_legal_actions(state)]


def test_fi_preemption_decline_does_not_record_negotiation_rejection():
    state = negotiation_state()
    float_corp_for_test(state, corp_id=2, company_id=2, player_id=0, par_index=20)
    for corp in (0, 2):
        CORPS[corp].set_cash(state, 200)
    give_company_to_fi(state, TARGET)
    for _ in range(int(GameConstants.ACQ_REJECTION_CAP)):
        PLAYERS[0].increment_acq_rejections(state)
        PLAYERS[2].increment_acq_rejections(state)
    DRIVER.apply_action(state, 1)
    DRIVER.apply_action(state, TARGET)
    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_OFFER)
    assert TURN.get_active_player(state) == 0
    DRIVER.apply_action(state, 0)
    assert PLAYERS[0].get_acq_rejections(state) == int(GameConstants.ACQ_REJECTION_CAP)
    assert PLAYERS[2].get_acq_rejections(state) == int(GameConstants.ACQ_REJECTION_CAP)
    assert all(c.get_max_rejected_price(state, p) == 0
               for c in COMPANIES for p in range(3))


def test_new_game_has_zero_history_for_all_six_players():
    state = GameState(int(GameConstants.MAX_PLAYERS))
    for p in range(int(GameConstants.MAX_PLAYERS)):
        assert PLAYERS[p].get_acq_rejections(state) == 0
        PLAYERS[p].increment_acq_rejections(state)
        assert COMPANIES[TARGET].get_max_rejected_price(state, p) == 0
        COMPANIES[TARGET].record_rejected_offer(state, p, p + 1)
    state.initialize_game(6, seed=42)
    assert all(COMPANIES[TARGET].get_max_rejected_price(state, p) == 0 for p in range(6))
    assert state._array.dtype == np.int16
    assert all(PLAYERS[p].get_acq_rejections(state) == 0 for p in range(6))


@pytest.mark.parametrize("n", [3, 6])
@pytest.mark.parametrize("v3", [False, True])
@pytest.mark.parametrize("seller_kind", ["player", "corp"])
def test_two_rejections_close_cross_player_negotiation_only_in_v3(n, v3, seller_kind):
    state = negotiation_state(n, v3, seller_kind)
    if seller_kind == "corp":
        give_company_to_corp(state, TARGET + 1, 2)
    else:
        give_company_to_player(state, TARGET + 1, 0)
    for corp, target in ((0, TARGET), (1, TARGET + 1)):
        CORPS[corp].set_cash(state, 200)
        select_target(state, corp, target)
        company = COMPANIES[target]
        DRIVER.apply_action(state, company.get_high_price() - company.get_low_price())
        DRIVER.apply_action(state, 0)
    assert PLAYERS[n - 1].get_acq_rejections(state) == int(GameConstants.ACQ_REJECTION_CAP)
    if v3:
        assert [aid for aid, _ in get_legal_actions(state)] == [0]
        # Even a different company is blocked. A different player can still offer.
        give_company_to_player(state, TARGET + 2, 0)
        assert [aid for aid, _ in get_legal_actions(state)] == [0]
        float_corp_for_test(state, corp_id=3, company_id=3, player_id=1, par_index=16)
        CORPS[3].set_cash(state, 200)
        TURN.set_active_player(state, 1)
        select_target(state, corp=3)
        assert [aid for aid, _ in get_legal_actions(state)] == [
            COMPANIES[TARGET].get_high_price() - COMPANIES[TARGET].get_low_price(),
        ]
    else:
        select_target(state)
        DRIVER.apply_action(state, 0)
        DRIVER.apply_action(state, 0)
        assert PLAYERS[n - 1].get_acq_rejections(state) == 3


def test_cap_preserves_same_president_player_and_corp_purchases():
    state = negotiation_state()
    give_company_to_player(state, TARGET + 1, 0)
    for corp, target in ((0, TARGET), (1, TARGET + 1)):
        CORPS[corp].set_cash(state, 200)
        select_target(state, corp, target)
        company = COMPANIES[target]
        DRIVER.apply_action(state, company.get_high_price() - company.get_low_price())
        DRIVER.apply_action(state, 0)
    give_company_to_player(state, 3, 2)
    give_company_to_corp(state, 4, 1)
    DRIVER.apply_action(state, 1)
    targets = [aid for aid, _ in get_legal_actions(state)]
    assert TARGET not in targets
    assert 3 in targets and 4 in targets
    DRIVER.apply_action(state, 3)
    DRIVER.apply_action(state, 0)
    assert COMPANIES[3].get_location(state) == int(CompanyLocation.LOC_CORP_ACQ)
    assert PLAYERS[2].get_acq_rejections(state) == 2
