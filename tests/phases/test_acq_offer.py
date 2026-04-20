"""Tests for the ACQ_OFFER phase.

Covers: FI preemption (accept/pass/cascade), cross-president corp-to-corp and
corp-to-player offers (accept/decline), receivership auto-buy fallback,
enumeration (always exactly 2 actions), and return-to-ACQUISITION transitions.

ACQ_OFFER has no setup function -- it is entered mid-action from ACQUISITION.
Tests set up ACQ_OFFER state directly using TURN field setters to replicate
what _enter_acq_offer() does, avoiding coupling to ACQUISITION internals.
"""
import pytest

from core.actions import (
    ACTION_PASS_PY as ACTION_PASS,
    ACTION_ACQ_OFFER_ACCEPT_PY as ACTION_ACQ_OFFER_ACCEPT,
    ACTION_ACQ_PRICE_PY as ACTION_ACQ_PRICE,
    ACTION_ACQ_SELECT_CORP_PY as ACTION_ACQ_SELECT_CORP,
)
from core.data import GamePhases, CorpIndices
from core.state import GameState
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
    find_legal_action_with_info,
    float_corp_for_test,
    setup_receivership_corp,
    draw_to_fi,
    draw_to_player,
    draw_to_corp,
)
from tests.phases.helpers.ownership import give_company_to_fi


# =============================================================================
# HELPERS
# =============================================================================

CORP_OS = int(CorpIndices.CORP_OS)  # 2


class TestTurnAcqOfferHelpers:
    def test_enter_acq_offer_sets_grouped_context(self, game_state):
        TURN.enter_acq_offer(
            game_state,
            offered_corp=4,
            company_id=7,
            price=33,
            original_corp=2,
            deciding_player=1,
        )

        assert TURN.get_acq_offer_corp(game_state) == 2
        assert TURN.get_acq_offer_price(game_state) == 33
        assert TURN.get_active_corp(game_state) == 4
        assert TURN.get_active_company(game_state) == 7
        assert TURN.get_active_player(game_state) == 1
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_OFFER)

    def test_clear_acquisition_context_clears_offer_fields(self, game_state):
        TURN.set_active_player(game_state, 0)
        TURN.enter_acq_offer(
            game_state,
            offered_corp=4,
            company_id=7,
            price=33,
            original_corp=2,
            deciding_player=1,
        )

        TURN.clear_acquisition_context(game_state)

        assert TURN.get_active_corp(game_state) == -1
        assert TURN.get_active_company(game_state) == -1
        assert TURN.get_acq_offer_price(game_state) == 0
        assert TURN.get_acq_offer_corp(game_state) == -1
        assert TURN.get_active_player(game_state) == 1


def _enter_acq_offer_direct(state, offered_corp, company_id, price,
                            original_corp, deciding_player):
    """Set up ACQ_OFFER state through TURN's grouped production helper.

    This still avoids coupling tests to ACQUISITION internals, but no longer
    duplicates the grouped field writes in Python.
    """
    TURN.enter_acq_offer(
        state,
        offered_corp=offered_corp,
        company_id=company_id,
        price=price,
        original_corp=original_corp,
        deciding_player=deciding_player,
    )


def _assert_offer_context_cleared(state):
    assert TURN.get_active_company(state) == -1
    assert TURN.get_acq_offer_price(state) == 0
    assert TURN.get_acq_offer_corp(state) == -1


# =============================================================================
# ENUMERATION
# =============================================================================

class TestEnumeration:
    """ACQ_OFFER always has exactly 2 legal actions: PASS and ACCEPT."""

    def test_always_two_actions(self, game_state):
        """Verify exactly PASS + ACCEPT are enumerated."""
        fi_co = draw_to_fi(game_state)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        price = COMPANIES[fi_co].get_high_price()
        _enter_acq_offer_direct(
            game_state, offered_corp=0, company_id=fi_co, price=price,
            original_corp=3, deciding_player=0,
        )

        actions = get_legal_actions(game_state)
        assert len(actions) == 2

        types = sorted(info.action_type for _, info in actions)
        assert types == sorted([ACTION_PASS, ACTION_ACQ_OFFER_ACCEPT])


# =============================================================================
# FI PREEMPTION: ACCEPT
# =============================================================================

class TestFiPreemptionAccept:
    """Preempting corp accepts FI buy via ACQ_OFFER."""

    def test_accept_transfers_company_to_preemptor(self, game_state):
        """Accepting transfers the FI company to the preempting corp."""
        fi_co = draw_to_fi(game_state)

        # Preemptor (corp 0, player 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        # Original buyer (corp 1, player 0) -- just needs to be active
        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)
        CORPS[1].set_cash(game_state, 200)

        price = COMPANIES[fi_co].get_high_price()
        _enter_acq_offer_direct(
            game_state, offered_corp=0, company_id=fi_co, price=price,
            original_corp=1, deciding_player=0,
        )

        accept_id = find_legal_action(game_state, action_type=ACTION_ACQ_OFFER_ACCEPT)
        apply_and_verify(game_state, accept_id)

        # Company goes to preemptor (corp 0), not original buyer (corp 1)
        loc = COMPANIES[fi_co].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[fi_co].get_owner_id(game_state) == 0

    def test_accept_deducts_cash_from_preemptor(self, game_state):
        """Accepting deducts the FI price from the preemptor's cash."""
        fi_co = draw_to_fi(game_state)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)
        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)
        CORPS[1].set_cash(game_state, 200)

        price = COMPANIES[fi_co].get_high_price()
        _enter_acq_offer_direct(
            game_state, offered_corp=0, company_id=fi_co, price=price,
            original_corp=1, deciding_player=0,
        )

        corp_cash_before = CORPS[0].get_cash(game_state)
        fi_cash_before = FI.get_cash(game_state)

        accept_id = find_legal_action(game_state, action_type=ACTION_ACQ_OFFER_ACCEPT)
        apply_and_verify(game_state, accept_id)

        assert CORPS[0].get_cash(game_state) == corp_cash_before - price
        assert FI.get_cash(game_state) == fi_cash_before + price

    def test_accept_clears_offer_context_and_auto_advances_to_closing(self, game_state):
        """Valid FI-preemption accept resolves the offer and auto-chains to CLOSING."""
        fi_co = draw_to_fi(game_state)

        # Valid FI preemption requires a different-president preemptor.
        float_corp_for_test(game_state, corp_id=0, player_id=1, par_index=10)
        CORPS[0].set_cash(game_state, 200)
        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)
        CORPS[1].set_cash(game_state, 200)

        price = COMPANIES[fi_co].get_high_price()
        _enter_acq_offer_direct(
            game_state, offered_corp=0, company_id=fi_co, price=price,
            original_corp=1, deciding_player=1,
        )

        accept_id = find_legal_action(game_state, action_type=ACTION_ACQ_OFFER_ACCEPT)
        apply_and_verify(game_state, accept_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_CLOSING)
        assert TURN.get_active_player(game_state) == 1
        assert TURN.get_active_corp(game_state) == -1
        _assert_offer_context_cleared(game_state)

    def test_os_preemptor_pays_face_value(self, game_state):
        """OS (corp 2) pays face value, not high price, when preempting."""
        fi_co = draw_to_fi(game_state)

        float_corp_for_test(game_state, corp_id=CORP_OS, player_id=0, par_index=10)
        CORPS[CORP_OS].set_cash(game_state, 200)
        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)
        CORPS[1].set_cash(game_state, 200)

        face_val = COMPANIES[fi_co].get_face_value()
        _enter_acq_offer_direct(
            game_state, offered_corp=CORP_OS, company_id=fi_co, price=face_val,
            original_corp=1, deciding_player=0,
        )

        corp_cash_before = CORPS[CORP_OS].get_cash(game_state)
        fi_cash_before = FI.get_cash(game_state)

        accept_id = find_legal_action(game_state, action_type=ACTION_ACQ_OFFER_ACCEPT)
        apply_and_verify(game_state, accept_id)

        # _execute_fi_buy uses _get_fi_purchase_price (face for OS, high for others)
        assert CORPS[CORP_OS].get_cash(game_state) == corp_cash_before - face_val
        assert FI.get_cash(game_state) == fi_cash_before + face_val


# =============================================================================
# FI PREEMPTION: PASS
# =============================================================================

class TestFiPreemptionPass:
    """Preempting corp declines the FI buy."""

    def test_pass_single_preemptor_original_buys(self, game_state):
        """When the only preemptor passes, the original corp buys from FI."""
        fi_co = draw_to_fi(game_state)

        # Preemptor (corp 0) -- will pass
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=15)
        CORPS[0].set_cash(game_state, 200)

        # Original buyer (corp 1)
        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=10)
        CORPS[1].set_cash(game_state, 200)

        price = COMPANIES[fi_co].get_high_price()
        _enter_acq_offer_direct(
            game_state, offered_corp=0, company_id=fi_co, price=price,
            original_corp=1, deciding_player=0,
        )

        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        # Original corp (1) should have bought the company
        loc = COMPANIES[fi_co].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[fi_co].get_owner_id(game_state) == 1

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_pass_sets_corp_passed_flag(self, game_state):
        """Passing sets the per-corp passed_acq_offer flag."""
        fi_co = draw_to_fi(game_state)

        # Two preemptors so passing doesn't immediately end
        # Preemptor 1 (corp 3, player 1) -- higher share price
        float_corp_for_test(game_state, corp_id=3, player_id=1, par_index=20)
        CORPS[3].set_cash(game_state, 200)

        # Preemptor 2 (corp 4, player 2) -- lower share price, offered second
        float_corp_for_test(game_state, corp_id=4, player_id=2, par_index=10)
        CORPS[4].set_cash(game_state, 200)

        # Original buyer (corp 5, player 0)
        float_corp_for_test(game_state, corp_id=5, player_id=0, par_index=5)
        CORPS[5].set_cash(game_state, 200)

        price = COMPANIES[fi_co].get_high_price()
        _enter_acq_offer_direct(
            game_state, offered_corp=3, company_id=fi_co, price=price,
            original_corp=5, deciding_player=1,
        )

        assert not CORPS[3].has_passed_acq_offer(game_state)
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)
        assert CORPS[3].has_passed_acq_offer(game_state)

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_pass_cascades_to_next_preemptor(self, game_state):
        """When first preemptor passes, offer cascades to next by share price."""
        fi_co = draw_to_fi(game_state)

        # Preemptor 1: corp 3, player 1 -- higher share price, offered first
        float_corp_for_test(game_state, corp_id=3, player_id=1, par_index=20)
        CORPS[3].set_cash(game_state, 200)

        # Preemptor 2: corp 4, player 2 -- lower share price, offered second
        float_corp_for_test(game_state, corp_id=4, player_id=2, par_index=10)
        CORPS[4].set_cash(game_state, 200)

        # Original buyer: corp 5, player 0
        float_corp_for_test(game_state, corp_id=5, player_id=0, par_index=5)
        CORPS[5].set_cash(game_state, 200)

        price = COMPANIES[fi_co].get_high_price()
        _enter_acq_offer_direct(
            game_state, offered_corp=3, company_id=fi_co, price=price,
            original_corp=5, deciding_player=1,
        )

        # Corp 3 (player 1) passes
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        # Should still be in ACQ_OFFER, now offered to corp 4 (player 2)
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_OFFER)
        assert TURN.get_active_corp(game_state) == 4
        assert TURN.get_active_player(game_state) == 2

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_all_preemptors_pass_original_buys(self, game_state):
        """When all preemptors pass, the original corp gets the company."""
        fi_co = draw_to_fi(game_state)

        # Preemptor 1: corp 3, player 1
        float_corp_for_test(game_state, corp_id=3, player_id=1, par_index=20)
        CORPS[3].set_cash(game_state, 200)

        # Preemptor 2: corp 4, player 2
        float_corp_for_test(game_state, corp_id=4, player_id=2, par_index=10)
        CORPS[4].set_cash(game_state, 200)

        # Original buyer: corp 5, player 0
        float_corp_for_test(game_state, corp_id=5, player_id=0, par_index=5)
        CORPS[5].set_cash(game_state, 200)

        price = COMPANIES[fi_co].get_high_price()
        _enter_acq_offer_direct(
            game_state, offered_corp=3, company_id=fi_co, price=price,
            original_corp=5, deciding_player=1,
        )

        # Both preemptors pass
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_OFFER)

        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        # Original corp (5) should have bought the company
        loc = COMPANIES[fi_co].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[fi_co].get_owner_id(game_state) == 5

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_receivership_original_buys_before_lower_price_preemptor(self, game_state):
        """A receivership original corp re-enters the price order before lower-price corps."""
        fi_co = draw_to_fi(game_state)
        recv_co = draw_to_player(game_state, 0)

        # Higher-price player-controlled preemptor offered first.
        high_preemptor = 3
        float_corp_for_test(game_state, corp_id=high_preemptor, player_id=1, par_index=20)
        CORPS[high_preemptor].set_cash(game_state, 200)

        # Lower-price player-controlled corp that can also afford the company.
        low_preemptor = 4
        float_corp_for_test(game_state, corp_id=low_preemptor, player_id=2, par_index=5)
        low_cash_before = COMPANIES[fi_co].get_high_price()
        CORPS[low_preemptor].set_cash(game_state, low_cash_before)

        # Original buyer is a receivership corp with a mid-range share price.
        original_corp = 5
        setup_receivership_corp(game_state, corp_id=original_corp, company_ids=[recv_co], par_index=10)
        CORPS[original_corp].set_cash(game_state, COMPANIES[fi_co].get_high_price())

        price = COMPANIES[fi_co].get_high_price()
        _enter_acq_offer_direct(
            game_state,
            offered_corp=high_preemptor,
            company_id=fi_co,
            price=price,
            original_corp=original_corp,
            deciding_player=1,
        )

        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        # The receivership original corp (price 10) outranks the lower-price
        # player corp (price 5), so it auto-buys immediately instead of
        # cascading to another ACQ_OFFER for corp 4.
        loc = COMPANIES[fi_co].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[fi_co].get_owner_id(game_state) == original_corp
        assert not (
            TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_OFFER)
            and TURN.get_active_corp(game_state) == low_preemptor
        )

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_second_preemptor_accepts(self, game_state):
        """First preemptor passes, second accepts and gets the company."""
        fi_co = draw_to_fi(game_state)

        # Preemptor 1: corp 3, player 1 -- will pass
        float_corp_for_test(game_state, corp_id=3, player_id=1, par_index=20)
        CORPS[3].set_cash(game_state, 200)

        # Preemptor 2: corp 4, player 2 -- will accept
        float_corp_for_test(game_state, corp_id=4, player_id=2, par_index=10)
        CORPS[4].set_cash(game_state, 200)

        # Original buyer: corp 5, player 0
        float_corp_for_test(game_state, corp_id=5, player_id=0, par_index=5)
        CORPS[5].set_cash(game_state, 200)

        price = COMPANIES[fi_co].get_high_price()
        _enter_acq_offer_direct(
            game_state, offered_corp=3, company_id=fi_co, price=price,
            original_corp=5, deciding_player=1,
        )

        # Corp 3 passes -> cascades to corp 4
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        # Corp 4 accepts
        accept_id = find_legal_action(game_state, action_type=ACTION_ACQ_OFFER_ACCEPT)
        apply_and_verify(game_state, accept_id)

        # Company goes to corp 4, not corp 3 or original corp 5
        loc = COMPANIES[fi_co].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[fi_co].get_owner_id(game_state) == 4

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_receivership_preemptor_auto_buys_and_clears_offer_context(self, game_state):
        """Passing to a receivership preemptor auto-buys and clears offer scratch state."""
        fi_co = draw_to_fi(game_state)

        # Preemptor 1: corp 3, player 1 -- will pass
        float_corp_for_test(game_state, corp_id=3, player_id=1, par_index=20)
        CORPS[3].set_cash(game_state, 200)

        # Preemptor 2: corp 4, receivership -- auto-buy candidate
        recv_co = draw_to_player(game_state, 0)
        setup_receivership_corp(game_state, corp_id=4, company_ids=[recv_co])
        CORPS[4].set_cash(game_state, 200)
        # Receivership corp needs a share price to be a preemptor
        # setup_receivership_corp already floats it with par_index default

        # Original buyer: corp 5, player 0
        float_corp_for_test(game_state, corp_id=5, player_id=0, par_index=5)
        CORPS[5].set_cash(game_state, 200)

        price = COMPANIES[fi_co].get_high_price()
        _enter_acq_offer_direct(
            game_state, offered_corp=3, company_id=fi_co, price=price,
            original_corp=5, deciding_player=1,
        )

        # Corp 3 passes
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert COMPANIES[fi_co].get_location(game_state) in (
            int(CompanyLocation.LOC_CORP_ACQ),
            int(CompanyLocation.LOC_CORP),
        )
        assert COMPANIES[fi_co].get_owner_id(game_state) == 4
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_DIVIDENDS)
        assert TURN.get_active_player(game_state) == 1
        assert TURN.get_active_corp(game_state) == 3
        _assert_offer_context_cleared(game_state)
        assert CORPS[3].has_passed_acq_offer(game_state)
        assert not CORPS[4].has_passed_acq_offer(game_state)


# =============================================================================
# CROSS-PRESIDENT: ACCEPT
# =============================================================================

class TestCrossPresidentAccept:
    """Owner accepts a cross-president acquisition offer."""

    @staticmethod
    def _make_state(num_players):
        state = GameState(num_players, acq_same_president=False)
        state.initialize_game(num_players, seed=42)
        return state

    def test_corp_to_corp_accept_transfers_company(self):
        """Accepting corp-to-corp offer transfers company to buyer."""
        state = self._make_state(3)

        # Seller: corp 0, player 0, 2 companies (keeps 1 after sale)
        seller_co = float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        draw_to_corp(state, 0)

        # Buyer: corp 1, player 1
        float_corp_for_test(state, corp_id=1, player_id=1, par_index=12)
        CORPS[1].set_cash(state, 200)

        low_price = COMPANIES[seller_co].get_low_price()
        price = low_price + 5

        _enter_acq_offer_direct(
            state, offered_corp=1, company_id=seller_co, price=price,
            original_corp=1, deciding_player=0,  # owner (corp 0's president) decides
        )

        accept_id = find_legal_action(state, action_type=ACTION_ACQ_OFFER_ACCEPT)
        apply_and_verify(state, accept_id)

        loc = COMPANIES[seller_co].get_location(state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[seller_co].get_owner_id(state) == 1

    def test_corp_to_corp_accept_cash_flow(self):
        """Accepting corp-to-corp: buyer pays, seller accumulates proceeds."""
        state = self._make_state(3)

        seller_co = float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        draw_to_corp(state, 0)

        float_corp_for_test(state, corp_id=1, player_id=1, par_index=12)
        CORPS[1].set_cash(state, 200)

        low_price = COMPANIES[seller_co].get_low_price()
        price = low_price + 3

        _enter_acq_offer_direct(
            state, offered_corp=1, company_id=seller_co, price=price,
            original_corp=1, deciding_player=0,
        )

        buyer_cash_before = CORPS[1].get_cash(state)
        seller_proceeds_before = CORPS[0].get_acquisition_proceeds(state)

        accept_id = find_legal_action(state, action_type=ACTION_ACQ_OFFER_ACCEPT)
        apply_and_verify(state, accept_id)

        assert CORPS[1].get_cash(state) == buyer_cash_before - price
        # Proceeds accumulate (flushed at end of ACQUISITION, not per-action)
        phase = TURN.get_phase(state)
        if phase == int(GamePhases.PHASE_ACQ_SELECT_CORP):
            assert CORPS[0].get_acquisition_proceeds(state) == seller_proceeds_before + price

    def test_corp_to_player_accept_transfers_cash(self):
        """Accepting corp-to-player: buyer pays corp cash, player receives cash."""
        state = self._make_state(3)

        # Player 1 owns a private company
        private_co = draw_to_player(state, 1)

        # Buyer: corp 0, player 0
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(state, 200)

        low_price = COMPANIES[private_co].get_low_price()
        price = low_price + 2

        _enter_acq_offer_direct(
            state, offered_corp=0, company_id=private_co, price=price,
            original_corp=0, deciding_player=1,  # player 1 (owner) decides
        )

        player_cash_before = PLAYERS[1].get_cash(state)
        corp_cash_before = CORPS[0].get_cash(state)

        accept_id = find_legal_action(state, action_type=ACTION_ACQ_OFFER_ACCEPT)
        apply_and_verify(state, accept_id)

        assert PLAYERS[1].get_cash(state) == player_cash_before + price
        assert CORPS[0].get_cash(state) == corp_cash_before - price

    def test_accept_clears_offer_context_and_auto_advances_to_closing(self):
        """Direct cross-president accept resolves the offer and clears scratch state."""
        state = self._make_state(3)

        private_co = draw_to_player(state, 1)

        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(state, 200)

        low_price = COMPANIES[private_co].get_low_price()
        price = low_price + 1

        _enter_acq_offer_direct(
            state, offered_corp=0, company_id=private_co, price=price,
            original_corp=0, deciding_player=1,
        )

        accept_id = find_legal_action(state, action_type=ACTION_ACQ_OFFER_ACCEPT)
        apply_and_verify(state, accept_id)

        assert TURN.get_phase(state) == int(GamePhases.PHASE_CLOSING)
        assert TURN.get_active_player(state) == 0
        assert TURN.get_active_corp(state) == -1
        _assert_offer_context_cleared(state)


# =============================================================================
# CROSS-PRESIDENT: PASS (DECLINE)
# =============================================================================

class TestCrossPresidentDecline:
    """Owner declines a cross-president acquisition offer."""

    @staticmethod
    def _make_state(num_players):
        state = GameState(num_players, acq_same_president=False)
        state.initialize_game(num_players, seed=42)
        return state

    def test_decline_cancels_acquisition(self):
        """Declining cancels the acquisition -- company stays with owner."""
        state = self._make_state(3)

        private_co = draw_to_player(state, 1)

        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(state, 200)

        low_price = COMPANIES[private_co].get_low_price()
        price = low_price + 5

        _enter_acq_offer_direct(
            state, offered_corp=0, company_id=private_co, price=price,
            original_corp=0, deciding_player=1,
        )

        corp_cash_before = CORPS[0].get_cash(state)
        player_cash_before = PLAYERS[1].get_cash(state)

        pass_id = find_legal_action(state, action_type=ACTION_PASS)
        apply_and_verify(state, pass_id)

        # Company stays with player 1
        assert COMPANIES[private_co].get_location(state) == int(CompanyLocation.LOC_PLAYER)
        assert COMPANIES[private_co].get_owner_id(state) == 1

        # No cash changed hands
        assert CORPS[0].get_cash(state) == corp_cash_before
        assert PLAYERS[1].get_cash(state) == player_cash_before

    def test_decline_clears_offer_context_and_returns_to_acquisition(self):
        """Direct cross-president decline cancels the offer and clears scratch state."""
        state = self._make_state(3)

        private_co = draw_to_player(state, 1)

        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(state, 200)

        low_price = COMPANIES[private_co].get_low_price()
        price = low_price + 1

        _enter_acq_offer_direct(
            state, offered_corp=0, company_id=private_co, price=price,
            original_corp=0, deciding_player=1,
        )

        pass_id = find_legal_action(state, action_type=ACTION_PASS)
        apply_and_verify(state, pass_id)

        assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_CORP)
        assert TURN.get_active_player(state) == 0
        assert TURN.get_active_corp(state) == -1
        _assert_offer_context_cleared(state)

    def test_decline_corp_to_corp_company_stays(self):
        """Declining corp-to-corp offer leaves company with seller corp."""
        state = self._make_state(3)

        seller_co = float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        draw_to_corp(state, 0)

        float_corp_for_test(state, corp_id=1, player_id=1, par_index=12)
        CORPS[1].set_cash(state, 200)

        low_price = COMPANIES[seller_co].get_low_price()
        price = low_price + 5

        _enter_acq_offer_direct(
            state, offered_corp=1, company_id=seller_co, price=price,
            original_corp=1, deciding_player=0,
        )

        pass_id = find_legal_action(state, action_type=ACTION_PASS)
        apply_and_verify(state, pass_id)

        # Company stays with corp 0
        assert COMPANIES[seller_co].get_location(state) == int(CompanyLocation.LOC_CORP)
        assert COMPANIES[seller_co].get_owner_id(state) == 0


# =============================================================================
# INTEGRATION: FULL FLOW THROUGH ACQUISITION -> ACQ_OFFER
# =============================================================================

class TestIntegrationFiPreemption:
    """Test ACQ_OFFER entered via FI_BUY preemption in the SELECT_CORP flow."""

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_fi_buy_triggers_preemption_and_accept_resolves(self, game_state):
        """FI buy by lower-priority corp triggers offer to higher-priority corp.

        Under the split: SELECT_CORP(3) → auto-chains SELECT_COMPANY (fi_co is
        corp 3's only legal target) → SELECT_PRICE (FI_BUY is the only offset) →
        preemption triggers ACQ_OFFER for corp 4's president.
        """
        fi_co = 10
        give_company_to_fi(game_state, fi_co)

        # Low-price buyer (player 1) -- initiates FI buy
        float_corp_for_test(game_state, corp_id=3, company_id=11, player_id=1, par_index=5)
        CORPS[3].set_cash(game_state, 200)

        # High-price preemptor (player 2) -- gets offer
        float_corp_for_test(game_state, corp_id=4, company_id=12, player_id=2, par_index=15)
        CORPS[4].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_SELECT_CORP)
        assert TURN.get_active_player(game_state) == 1

        select_corp_id = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=3,
        )
        apply_and_verify(game_state, select_corp_id)

        # SELECT_COMPANY / SELECT_PRICE each had 1 legal action → auto-chained.
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_OFFER)
        assert TURN.get_active_player(game_state) == 2
        assert TURN.get_active_corp(game_state) == 4
        assert TURN.get_active_company(game_state) == fi_co

        # Accept the preemption
        accept_id = find_legal_action(game_state, action_type=ACTION_ACQ_OFFER_ACCEPT)
        apply_and_verify(game_state, accept_id)

        # Company should go to preemptor (corp 4)
        loc = COMPANIES[fi_co].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[fi_co].get_owner_id(game_state) == 4


class TestIntegrationCrossPresident:
    """Test ACQ_OFFER entered via cross-president ACQ_PRICE through the split flow."""

    @staticmethod
    def _make_state(num_players):
        state = GameState(num_players, acq_same_president=False)
        state.initialize_game(num_players, seed=42)
        return state

    def test_cross_president_full_flow_accept(self):
        """Full flow: SELECT_CORP → SELECT_COMPANY → SELECT_PRICE → ACQ_OFFER → accept."""
        state = self._make_state(3)

        private_co = draw_to_player(state, 1)

        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(state, 10_000)

        setup_acquisition_phase_py(state)

        player_cash_before = PLAYERS[1].get_cash(state)

        # SELECT_CORP(0) — only 1 target (private_co) so SELECT_COMPANY auto-chains.
        select_corp_id = find_legal_action(
            state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=0,
        )
        apply_and_verify(state, select_corp_id)

        # Land in SELECT_PRICE with multiple offsets thanks to generous cash.
        assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_PRICE)

        acq_id, info = find_legal_action_with_info(
            state, action_type=ACTION_ACQ_PRICE, amount=2,
        )
        price = COMPANIES[private_co].get_low_price() + info.amount
        apply_and_verify(state, acq_id)

        assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_OFFER)
        assert TURN.get_active_player(state) == 1

        accept_id = find_legal_action(state, action_type=ACTION_ACQ_OFFER_ACCEPT)
        apply_and_verify(state, accept_id)

        loc = COMPANIES[private_co].get_location(state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[private_co].get_owner_id(state) == 0
        assert PLAYERS[1].get_cash(state) == player_cash_before + price

    def test_cross_president_full_flow_decline(self):
        """Full flow: SELECT_CORP → ... → ACQ_OFFER → decline."""
        state = self._make_state(3)

        private_co = draw_to_player(state, 1)

        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(state, 10_000)

        setup_acquisition_phase_py(state)

        corp_cash_before = CORPS[0].get_cash(state)

        select_corp_id = find_legal_action(
            state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=0,
        )
        apply_and_verify(state, select_corp_id)
        assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_PRICE)

        acq_id = find_legal_action(state, action_type=ACTION_ACQ_PRICE, amount=1)
        apply_and_verify(state, acq_id)

        assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_OFFER)

        pass_id = find_legal_action(state, action_type=ACTION_PASS)
        apply_and_verify(state, pass_id)

        # Company stays with player 1, corp cash unchanged
        assert COMPANIES[private_co].get_location(state) == int(CompanyLocation.LOC_PLAYER)
        assert COMPANIES[private_co].get_owner_id(state) == 1
        assert CORPS[0].get_cash(state) == corp_cash_before
