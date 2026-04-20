"""Tests for the ACQ_SELECT_CORP phase (first leg of three-step Acquisition).

Covers: PASS action, corp-select legality (active player presides, not in
receivership, cash > 0, any legal target), action id encoding (``1 + corp_id``),
receivership forced FI buys, transition to CLOSING when all pass, stay-on-same-
player re-entry after a successful acquisition, and acquisition-zone merge on
phase exit.

SELECT_COMPANY and SELECT_PRICE concerns live in the sibling files
``test_acq_select_company.py`` / ``test_acq_select_price.py``. ACQ_OFFER
preemption lives in ``test_acq_offer.py``.
"""
import pytest

from core.actions import (
    ACTION_PASS_PY as ACTION_PASS,
    ACTION_ACQ_PRICE_PY as ACTION_ACQ_PRICE,
    ACTION_ACQ_FI_BUY_PY as ACTION_ACQ_FI_BUY,
    ACTION_ACQ_SELECT_CORP_PY as ACTION_ACQ_SELECT_CORP,
    ACTION_ACQ_SELECT_COMPANY_PY as ACTION_ACQ_SELECT_COMPANY,
)
from core.data import GamePhases, GameConstants, CorpIndices
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES, CompanyLocation
from entities.fi import FI
from phases.acq_select_corp import setup_acquisition_phase_py

from tests.phases.conftest import (
    apply_and_verify,
    get_legal_actions,
    find_legal_action,
    float_corp_for_test,
    setup_receivership_corp,
    draw_to_player,
    draw_to_fi,
    draw_to_corp,
)
from tests.phases.helpers.ownership import count_at_location, give_company_to_fi


CORP_OS = int(CorpIndices.CORP_OS)  # 2


# =============================================================================
# HELPERS
# =============================================================================

def _acquire(state, corp_id, company_id, amount=None):
    """Drive a single acquisition through SELECT_CORP → ... → execute.

    Picks the minimum legal offset if ``amount`` is omitted. Auto-chain may
    swallow SELECT_COMPANY / SELECT_PRICE when only one legal action exists
    (e.g. single target, FI target, single affordable offset) — this helper
    tolerates that.
    """
    aid = find_legal_action(
        state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=corp_id,
    )
    apply_and_verify(state, aid)

    if TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_COMPANY):
        aid = find_legal_action(
            state, action_type=ACTION_ACQ_SELECT_COMPANY, company_id=company_id,
        )
        apply_and_verify(state, aid)

    if TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_PRICE):
        loc = COMPANIES[company_id].get_location(state)
        if loc == int(CompanyLocation.LOC_FI):
            aid = find_legal_action(state, action_type=ACTION_ACQ_FI_BUY)
        else:
            aid = find_legal_action(
                state, action_type=ACTION_ACQ_PRICE,
                amount=0 if amount is None else amount,
            )
        apply_and_verify(state, aid)


def _pass_through_select_corp(state, max_steps=50):
    """Pass until the engine leaves PHASE_ACQ_SELECT_CORP."""
    for _ in range(max_steps):
        if TURN.get_phase(state) != int(GamePhases.PHASE_ACQ_SELECT_CORP):
            return
        pass_id = find_legal_action(state, action_type=ACTION_PASS)
        apply_and_verify(state, pass_id)


# =============================================================================
# ENUMERATION
# =============================================================================

class TestEnumeration:
    """Legal-action enumeration: PASS + corp-select per eligible corp."""

    def test_pass_always_legal(self, game_state):
        """PASS is legal whenever SELECT_CORP is a real decision."""
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        pass_actions = [info for _, info in get_legal_actions(game_state)
                        if info.action_type == ACTION_PASS]
        assert len(pass_actions) == 1

    def test_no_price_or_fi_buy_actions_enumerated(self, game_state):
        """SELECT_CORP never exposes ACQ_PRICE or ACQ_FI_BUY — those are SELECT_PRICE."""
        draw_to_fi(game_state)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        actions = get_legal_actions(game_state)
        assert not any(info.action_type == ACTION_ACQ_PRICE for _, info in actions)
        assert not any(info.action_type == ACTION_ACQ_FI_BUY for _, info in actions)

    def test_action_id_encoding(self, game_state):
        """SELECT_CORP corp-select action id is ``1 + corp_id``."""
        draw_to_fi(game_state)
        float_corp_for_test(game_state, corp_id=3, player_id=0, par_index=10)
        CORPS[3].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=3,
        )
        assert aid == 1 + 3

    def test_corp_excluded_when_no_legal_target(self, game_state):
        """Corp with no affordable target is not offered."""
        # Corp with cash but no companies to buy → only PASS remains.
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        setup_acquisition_phase_py(game_state)

        actions = get_legal_actions(game_state)
        assert len(actions) == 1
        assert actions[0][1].action_type == ACTION_PASS

    def test_receivership_corp_not_enumerated(self, game_state):
        """Corps in receivership are never selectable in SELECT_CORP."""
        # Player 0 owns a live corp plus a receivership corp.
        draw_to_fi(game_state)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        recv_co = draw_to_player(game_state, 0)
        setup_receivership_corp(game_state, corp_id=1, company_ids=[recv_co])
        CORPS[1].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        corps_offered = {
            info.corp_id
            for _, info in get_legal_actions(game_state)
            if info.action_type == ACTION_ACQ_SELECT_CORP
        }
        assert 1 not in corps_offered, "Receivership corp should not be selectable"

    def test_foreign_presidency_not_enumerated(self, game_state):
        """Corps presided by another player are not selectable by the active player."""
        # Player 1 presides corp 1; player 0 presides corp 0.
        draw_to_player(game_state, 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        float_corp_for_test(game_state, corp_id=1, player_id=1, par_index=12)
        CORPS[1].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        active = TURN.get_active_player(game_state)
        corps_offered = {
            info.corp_id
            for _, info in get_legal_actions(game_state)
            if info.action_type == ACTION_ACQ_SELECT_CORP
        }
        # Active player should only see their own corp.
        expected_corp = 0 if active == 0 else 1
        assert corps_offered == {expected_corp}

    def test_zero_cash_corp_not_enumerated(self, game_state):
        """Corp with 0 cash cannot acquire; only PASS remains."""
        draw_to_player(game_state, 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 0)

        setup_acquisition_phase_py(game_state)

        actions = get_legal_actions(game_state)
        assert len(actions) == 1
        assert actions[0][1].action_type == ACTION_PASS


# =============================================================================
# PASS ACTION
# =============================================================================

class TestPassAction:
    """PASS marks the active player passed and advances to next eligible."""

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_pass_marks_player_as_passed(self, game_state):
        """Passing sets the per-player has_passed flag."""
        draw_to_player(game_state, 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 100)

        draw_to_player(game_state, 1)
        float_corp_for_test(game_state, corp_id=1, player_id=1, par_index=12)
        CORPS[1].set_cash(game_state, 100)

        setup_acquisition_phase_py(game_state)

        active = TURN.get_active_player(game_state)
        assert not PLAYERS[active].has_passed(game_state)

        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)
        assert PLAYERS[active].has_passed(game_state)

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_pass_advances_to_next_player(self, game_state):
        """After pass, control moves to the next eligible player."""
        draw_to_player(game_state, 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 100)

        draw_to_player(game_state, 1)
        float_corp_for_test(game_state, corp_id=1, player_id=1, par_index=12)
        CORPS[1].set_cash(game_state, 100)

        setup_acquisition_phase_py(game_state)

        first_active = TURN.get_active_player(game_state)
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_SELECT_CORP)
        assert TURN.get_active_player(game_state) != first_active

    def test_all_pass_transitions_past_closing(self, game_state):
        """When everyone passes, phase chains through CLOSING/INCOME to DIVIDENDS."""
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 100)

        setup_acquisition_phase_py(game_state)
        _pass_through_select_corp(game_state)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_DIVIDENDS)
        assert TURN.get_active_corp(game_state) == 0
        assert TURN.get_active_company(game_state) == -1


# =============================================================================
# CORP SELECT
# =============================================================================

class TestCorpSelect:
    """Corp-select seeds active_corp and transitions to SELECT_COMPANY."""

    def test_corp_select_sets_active_corp(self, game_state):
        """Applying a SELECT_CORP action seeds active_corp."""
        draw_to_player(game_state, 0)
        draw_to_player(game_state, 0)  # 2nd target so SELECT_COMPANY doesn't auto-chain
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        setup_acquisition_phase_py(game_state)

        aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=0,
        )
        apply_and_verify(game_state, aid)

        assert TURN.get_active_corp(game_state) == 0

    def test_corp_select_switches_to_select_company(self, game_state):
        """Applying SELECT_CORP transitions the engine phase to SELECT_COMPANY."""
        draw_to_player(game_state, 0)
        draw_to_player(game_state, 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        setup_acquisition_phase_py(game_state)

        aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=0,
        )
        apply_and_verify(game_state, aid)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_SELECT_COMPANY)
        assert TURN.get_active_company(game_state) == -1


# =============================================================================
# RECEIVERSHIP FORCED BUYS
# =============================================================================

class TestReceivershipForcedBuys:
    """Automatic receivership FI buys at the start of ACQ_SELECT_CORP."""

    def test_receivership_corp_auto_buys_fi_company(self, game_state):
        """Receivership corp with cash automatically buys from FI during setup."""
        recv_co = 10
        fi_co = 11
        setup_receivership_corp(game_state, corp_id=0, company_ids=[recv_co])
        give_company_to_fi(game_state, fi_co)
        expected_price = COMPANIES[fi_co].get_high_price()
        CORPS[0].set_cash(game_state, expected_price)
        fi_cash_before = FI.get_cash(game_state)

        setup_acquisition_phase_py(game_state)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INCOME)
        loc = COMPANIES[fi_co].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[fi_co].get_owner_id(game_state) == 0
        assert FI.get_cash(game_state) == fi_cash_before + expected_price
        assert CORPS[0].get_cash(game_state) == 0

    def test_receivership_with_no_cash_skips(self, game_state):
        """Receivership corp with 0 cash does not buy from FI."""
        recv_co = draw_to_player(game_state, 0)
        setup_receivership_corp(game_state, corp_id=0, company_ids=[recv_co])
        CORPS[0].set_cash(game_state, 0)

        fi_co = draw_to_fi(game_state)

        setup_acquisition_phase_py(game_state)

        assert COMPANIES[fi_co].get_location(game_state) == int(CompanyLocation.LOC_FI)


# =============================================================================
# PHASE TRANSITIONS
# =============================================================================

class TestPhaseTransitions:
    """Exit paths from SELECT_CORP."""

    def test_no_active_corps_auto_advance_to_income(self, game_state):
        """With no active corps, setup hands off through CLOSING and stops in INCOME."""
        setup_acquisition_phase_py(game_state)
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INCOME)
        assert TURN.get_active_corp(game_state) == -1
        assert TURN.get_active_company(game_state) == -1

    def test_only_receivership_corps_auto_advance_to_income(self, game_state):
        """Receivership-only setup finishes deterministic work and stops in INCOME."""
        recv_co = draw_to_player(game_state, 0)
        setup_receivership_corp(game_state, corp_id=0, company_ids=[recv_co])
        CORPS[0].set_cash(game_state, 0)

        setup_acquisition_phase_py(game_state)
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INCOME)
        assert TURN.get_active_corp(game_state) == -1
        assert TURN.get_active_company(game_state) == -1

    def test_eventually_reaches_closing(self, game_state):
        """Pass-only play from SELECT_CORP reaches CLOSING or beyond."""
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 100)

        setup_acquisition_phase_py(game_state)
        _pass_through_select_corp(game_state)

        phase = TURN.get_phase(game_state)
        assert phase >= int(GamePhases.PHASE_CLOSING), (
            f"Expected CLOSING or later, got phase {phase}"
        )


# =============================================================================
# STAY ON SAME PLAYER
# =============================================================================

class TestStayOnSamePlayer:
    """After a resolved buy, the same player re-enters SELECT_CORP."""

    def test_active_player_unchanged_after_acq_price(self, game_state):
        """Direct-execution acquisition leaves the active player unchanged."""
        private_co = draw_to_player(game_state, 0)
        draw_to_player(game_state, 0)  # keep a second target remaining after buy
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 300)

        setup_acquisition_phase_py(game_state)
        active_before = TURN.get_active_player(game_state)

        _acquire(game_state, corp_id=0, company_id=private_co)

        if TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_SELECT_CORP):
            assert TURN.get_active_player(game_state) == active_before

    def test_active_player_unchanged_after_fi_buy(self, game_state):
        """FI buy (no preemption) leaves the active player unchanged."""
        fi_co = draw_to_fi(game_state)
        draw_to_fi(game_state)  # keep a 2nd FI target so SELECT_CORP returns as a decision
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        setup_acquisition_phase_py(game_state)
        active_before = TURN.get_active_player(game_state)

        _acquire(game_state, corp_id=0, company_id=fi_co)

        if TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_SELECT_CORP):
            assert TURN.get_active_player(game_state) == active_before

    def test_two_consecutive_buys(self, game_state):
        """One player can make multiple acquisitions before passing."""
        co1 = draw_to_player(game_state, 0)
        co2 = draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        setup_acquisition_phase_py(game_state)

        _acquire(game_state, corp_id=0, company_id=co1)
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_SELECT_CORP)

        _acquire(game_state, corp_id=0, company_id=co2)

        for co in [co1, co2]:
            loc = COMPANIES[co].get_location(game_state)
            assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
            assert COMPANIES[co].get_owner_id(game_state) == 0


# =============================================================================
# ACQUISITION ZONE MERGE
# =============================================================================

class TestAcquisitionZoneMerge:
    """Zone merge + proceeds flush on exit from SELECT_CORP."""

    def test_acq_zone_merged_after_phase(self, game_state):
        """After SELECT_CORP ends, no companies remain in LOC_CORP_ACQ."""
        private_co = draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        _acquire(game_state, corp_id=0, company_id=private_co)
        _pass_through_select_corp(game_state)

        assert count_at_location(game_state, CompanyLocation.LOC_CORP_ACQ) == 0

    def test_proceeds_flushed_after_phase(self, game_state):
        """After SELECT_CORP ends, all active corps have 0 acquisition_proceeds."""
        seller_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        draw_to_corp(game_state, 0)

        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)
        CORPS[1].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        _acquire(game_state, corp_id=1, company_id=seller_co)
        _pass_through_select_corp(game_state)

        for corp_id in range(int(GameConstants.NUM_CORPS)):
            if CORPS[corp_id].is_active(game_state):
                assert CORPS[corp_id].get_acquisition_proceeds(game_state) == 0, (
                    f"Corp {corp_id} still has nonzero acquisition_proceeds"
                )
