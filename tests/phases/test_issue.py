"""Tests for the ISSUE_SHARES phase.

Covers: issue vs pass actions, share transfer and cash proceeds, price drop to
next lower available space, Stock Masters (CORP_SM) no-price-drop ability,
processing order (descending share price), receivership auto-issue,
auto-skip of corps with no unissued shares, bankruptcy on issue to index 0,
phase transitions through IPO to the next turn, and legal-action enumeration.
"""
import pytest

from core.actions import (
    ACTION_PASS_PY as ACTION_PASS,
    ACTION_ISSUE_PY as ACTION_ISSUE,
)
from core.data import GamePhases, GameConstants, CorpIndices
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.market import MARKET
from phases.issue import setup_issue_phase_py

from tests.phases.conftest import (
    apply_and_verify,
    get_legal_actions,
    find_legal_action,
    float_corp_for_test,
    setup_receivership_corp,
)


# =============================================================================
# HELPERS
# =============================================================================

# BME: 1 star, face=1. Lowest-impact company for floating scenarios.
CO_A = 0
# BSE: 1 star, face=2. Second 1-star company for multi-corp setups.
CO_B = 1
# CIN: 1 star, face=3. Third 1-star company.
CO_C = 2

CORP_SM = int(CorpIndices.CORP_SM)  # 3


def _enter_issue(state):
    """Transition state into ISSUE_SHARES phase.

    If no active corp with unissued shares is found, setup will cascade
    through the auto-transition into IPO / INVEST.
    """
    setup_issue_phase_py(state)


# =============================================================================
# ENUMERATION
# =============================================================================

class TestEnumeration:
    """ISSUE always has exactly 2 legal actions when a decision is pending."""

    def test_exactly_two_actions(self, game_state):
        """With an active player-controlled corp, only PASS and ISSUE are legal."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        _enter_issue(game_state)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ISSUE_SHARES)
        actions = get_legal_actions(game_state)
        assert len(actions) == 2

        types = sorted(info.action_type for _, info in actions)
        assert types == sorted([ACTION_PASS, ACTION_ISSUE])

    def test_pass_action_id_zero(self, game_state):
        """PASS encodes as action id 0."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        _enter_issue(game_state)

        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        assert pass_id == 0

    def test_issue_action_id_one(self, game_state):
        """ISSUE encodes as action id 1."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        _enter_issue(game_state)

        issue_id = find_legal_action(game_state, action_type=ACTION_ISSUE)
        assert issue_id == 1


# =============================================================================
# ISSUE ACTION TESTS (share transfer + cash + price)
# =============================================================================

class TestIssueAction:
    """Issuing transfers one share to the bank and the corp receives cash."""

    def test_issue_transfers_one_share(self, game_state):
        """Issuing decrements unissued and increments both issued and bank."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        _enter_issue(game_state)

        corp = CORPS[0]
        unissued_before = corp.get_unissued_shares(game_state)
        issued_before = corp.get_issued_shares(game_state)
        bank_before = corp.get_bank_shares(game_state)

        issue_id = find_legal_action(game_state, action_type=ACTION_ISSUE)
        apply_and_verify(game_state, issue_id)

        assert corp.get_unissued_shares(game_state) == unissued_before - 1
        assert corp.get_issued_shares(game_state) == issued_before + 1
        assert corp.get_bank_shares(game_state) == bank_before + 1

    def test_issue_gives_corp_new_price_cash(self, game_state):
        """Normal corp receives the NEW (lower) price as cash proceeds."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        CORPS[0].set_cash(game_state, 50)
        _enter_issue(game_state)

        corp = CORPS[0]
        current_index = corp.get_price_index(game_state)
        new_index = MARKET.find_next_lower_space(game_state, current_index)
        expected_proceeds = MARKET.get_price_at_index(new_index)

        cash_before = corp.get_cash(game_state)
        issue_id = find_legal_action(game_state, action_type=ACTION_ISSUE)
        apply_and_verify(game_state, issue_id)

        assert corp.get_cash(game_state) == cash_before + expected_proceeds

    def test_issue_drops_price_one_space(self, game_state):
        """With all lower spaces free, issue drops price by exactly one index."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        _enter_issue(game_state)

        corp = CORPS[0]
        index_before = corp.get_price_index(game_state)

        issue_id = find_legal_action(game_state, action_type=ACTION_ISSUE)
        apply_and_verify(game_state, issue_id)

        assert corp.get_price_index(game_state) == index_before - 1

    def test_issue_slides_past_occupied_space(self, game_state):
        """Price slides past occupied target to next available lower space."""
        # Corp 1 at index 14 keeps that space occupied during corp 0's issue.
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        float_corp_for_test(game_state, corp_id=1, player_id=0,
                            company_id=CO_B, par_index=14)
        _enter_issue(game_state)

        # Corp 0 (index 15, higher) is processed first; target 14 is occupied,
        # so it slides to 13.
        assert TURN.get_active_corp(game_state) == 0
        issue_id = find_legal_action(game_state, action_type=ACTION_ISSUE)
        apply_and_verify(game_state, issue_id)

        assert CORPS[0].get_price_index(game_state) == 13

    def test_issue_frees_old_space(self, game_state):
        """Old market space becomes available after the price drop."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        _enter_issue(game_state)

        corp = CORPS[0]
        old_index = corp.get_price_index(game_state)
        assert not MARKET.is_space_available(game_state, old_index)

        issue_id = find_legal_action(game_state, action_type=ACTION_ISSUE)
        apply_and_verify(game_state, issue_id)

        assert MARKET.is_space_available(game_state, old_index)

    def test_issue_occupies_new_space(self, game_state):
        """New market space becomes occupied after the price drop."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        _enter_issue(game_state)

        corp = CORPS[0]
        old_index = corp.get_price_index(game_state)
        new_index = MARKET.find_next_lower_space(game_state, old_index)

        issue_id = find_legal_action(game_state, action_type=ACTION_ISSUE)
        apply_and_verify(game_state, issue_id)

        assert not MARKET.is_space_available(game_state, new_index)


# =============================================================================
# PASS ACTION TESTS
# =============================================================================

class TestPassAction:
    """Passing skips issue without touching shares, cash, or price."""

    def test_pass_does_not_transfer_shares(self, game_state):
        """PASS leaves the share counts unchanged."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        _enter_issue(game_state)

        corp = CORPS[0]
        unissued_before = corp.get_unissued_shares(game_state)
        issued_before = corp.get_issued_shares(game_state)
        bank_before = corp.get_bank_shares(game_state)

        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert corp.get_unissued_shares(game_state) == unissued_before
        assert corp.get_issued_shares(game_state) == issued_before
        assert corp.get_bank_shares(game_state) == bank_before

    def test_pass_does_not_change_cash(self, game_state):
        """PASS leaves corp cash unchanged."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        CORPS[0].set_cash(game_state, 50)
        _enter_issue(game_state)

        cash_before = CORPS[0].get_cash(game_state)
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert CORPS[0].get_cash(game_state) == cash_before

    def test_pass_does_not_change_price(self, game_state):
        """PASS leaves the corp's market price index unchanged."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        _enter_issue(game_state)

        index_before = CORPS[0].get_price_index(game_state)
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert CORPS[0].get_price_index(game_state) == index_before


# =============================================================================
# STOCK MASTERS (CORP_SM) SPECIAL RULE
# =============================================================================

class TestStockMasters:
    """CORP_SM's price does not drop when issuing."""

    def test_sm_price_unchanged(self, game_state):
        """Stock Masters keeps its market index after issuing."""
        float_corp_for_test(game_state, corp_id=CORP_SM, player_id=0,
                            company_id=CO_A, par_index=15)
        _enter_issue(game_state)

        corp = CORPS[CORP_SM]
        index_before = corp.get_price_index(game_state)

        issue_id = find_legal_action(game_state, action_type=ACTION_ISSUE)
        apply_and_verify(game_state, issue_id)

        assert corp.get_price_index(game_state) == index_before

    def test_sm_receives_current_price(self, game_state):
        """Stock Masters receives the (unchanged) current price as cash."""
        float_corp_for_test(game_state, corp_id=CORP_SM, player_id=0,
                            company_id=CO_A, par_index=15)
        CORPS[CORP_SM].set_cash(game_state, 0)
        _enter_issue(game_state)

        current_price = CORPS[CORP_SM].get_share_price(game_state)
        issue_id = find_legal_action(game_state, action_type=ACTION_ISSUE)
        apply_and_verify(game_state, issue_id)

        assert CORPS[CORP_SM].get_cash(game_state) == current_price

    def test_sm_keeps_market_space_occupied(self, game_state):
        """SM's market space remains occupied (no free + claim dance)."""
        float_corp_for_test(game_state, corp_id=CORP_SM, player_id=0,
                            company_id=CO_A, par_index=15)
        _enter_issue(game_state)

        issue_id = find_legal_action(game_state, action_type=ACTION_ISSUE)
        apply_and_verify(game_state, issue_id)

        assert not MARKET.is_space_available(game_state, 15)

    def test_sm_at_index_one_does_not_bankrupt(self, game_state):
        """SM at index 1 survives issuing — normal corps would bankrupt.

        A normal corp at index 1 issuing would drop to 0 and go bankrupt.
        SM's ability prevents the drop, so it stays active at index 1.
        """
        float_corp_for_test(game_state, corp_id=CORP_SM, player_id=0,
                            company_id=CO_A, par_index=10)
        CORPS[CORP_SM].move_to_price_index(game_state, 1)
        CORPS[CORP_SM].set_cash(game_state, 0)

        _enter_issue(game_state)

        issue_id = find_legal_action(game_state, action_type=ACTION_ISSUE)
        apply_and_verify(game_state, issue_id)

        assert CORPS[CORP_SM].is_active(game_state)
        assert CORPS[CORP_SM].get_price_index(game_state) == 1
        assert CORPS[CORP_SM].get_cash(game_state) == MARKET.get_price_at_index(1)


# =============================================================================
# PROCESSING ORDER
# =============================================================================

class TestProcessingOrder:
    """Corps are processed in descending share-price order."""

    def test_highest_price_corp_first(self, game_state):
        """The highest-priced corp is offered the decision first."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=5)
        float_corp_for_test(game_state, corp_id=1, player_id=0,
                            company_id=CO_B, par_index=15)
        _enter_issue(game_state)

        assert TURN.get_active_corp(game_state) == 1

    def test_second_corp_after_first_resolved(self, game_state):
        """After the top corp passes, the next-highest corp becomes active."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=5)
        float_corp_for_test(game_state, corp_id=1, player_id=0,
                            company_id=CO_B, par_index=15)
        _enter_issue(game_state)

        assert TURN.get_active_corp(game_state) == 1
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        # Second corp picked up before transitioning out.
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ISSUE_SHARES)
        assert TURN.get_active_corp(game_state) == 0

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_three_corps_descending_order(self, game_state):
        """Three corps at distinct prices are processed highest → lowest."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=5)
        float_corp_for_test(game_state, corp_id=1, player_id=1,
                            company_id=CO_B, par_index=15)
        float_corp_for_test(game_state, corp_id=2, player_id=2,
                            company_id=CO_C, par_index=10)
        _enter_issue(game_state)

        expected_order = [1, 2, 0]
        for expected_corp in expected_order:
            assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ISSUE_SHARES)
            assert TURN.get_active_corp(game_state) == expected_corp
            pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, pass_id)

    def test_active_player_is_president(self, game_state):
        """Active player matches the president of the active corp."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=5)
        float_corp_for_test(game_state, corp_id=1, player_id=1,
                            company_id=CO_B, par_index=15)
        _enter_issue(game_state)

        assert TURN.get_active_corp(game_state) == 1
        assert TURN.get_active_player(game_state) == 1


# =============================================================================
# RECEIVERSHIP
# =============================================================================

class TestReceivership:
    """Receivership corps auto-issue (or auto-skip) without a player decision."""

    def test_receivership_auto_issues(self, game_state):
        """Receivership corp with unissued shares auto-issues during advance."""
        setup_receivership_corp(game_state, corp_id=0,
                                company_ids=[CO_A], par_index=15)
        corp = CORPS[0]
        unissued_before = corp.get_unissued_shares(game_state)
        issued_before = corp.get_issued_shares(game_state)
        bank_before = corp.get_bank_shares(game_state)

        _enter_issue(game_state)

        # Auto-issued exactly once; no decision was offered to any player.
        assert corp.get_unissued_shares(game_state) == unissued_before - 1
        assert corp.get_issued_shares(game_state) == issued_before + 1
        assert corp.get_bank_shares(game_state) == bank_before + 1

    def test_receivership_with_no_unissued_auto_skipped(self, game_state):
        """Receivership corp with 0 unissued shares is skipped silently."""
        setup_receivership_corp(game_state, corp_id=0,
                                company_ids=[CO_A], par_index=15)

        corp = CORPS[0]
        # Empty the treasury: move all unissued into the bank.
        total = corp.get_total_shares()
        corp.set_unissued_shares(game_state, 0)
        corp.set_issued_shares(game_state, total)
        corp.set_bank_shares(game_state, total)
        cash_before = corp.get_cash(game_state)

        _enter_issue(game_state)

        # No issue performed, cash untouched, corp still active.
        assert corp.get_unissued_shares(game_state) == 0
        assert corp.get_cash(game_state) == cash_before
        assert corp.is_active(game_state)

    def test_receivership_skipped_for_player_decision(self, game_state):
        """Player-controlled corp waits for decision; receivership was pre-resolved.

        Corp 1 (receivership, higher price) auto-issues first; corp 0 (player)
        then becomes active for a decision.
        """
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=5)
        setup_receivership_corp(game_state, corp_id=1,
                                company_ids=[CO_B], par_index=15)

        corp1_unissued_before = CORPS[1].get_unissued_shares(game_state)
        corp0_unissued_before = CORPS[0].get_unissued_shares(game_state)

        _enter_issue(game_state)

        # Corp 1 auto-issued; corp 0 is now the active decision.
        assert CORPS[1].get_unissued_shares(game_state) == corp1_unissued_before - 1
        assert CORPS[0].get_unissued_shares(game_state) == corp0_unissued_before
        assert TURN.get_active_corp(game_state) == 0
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ISSUE_SHARES)


# =============================================================================
# AUTO-SKIP (PLAYER CORPS WITH NO UNISSUED)
# =============================================================================

class TestAutoSkip:
    """Player-controlled corps with 0 unissued shares are skipped silently."""

    def test_player_corp_with_no_unissued_skipped(self, game_state):
        """Player corp with 0 unissued shares never becomes active."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=5)
        float_corp_for_test(game_state, corp_id=1, player_id=0,
                            company_id=CO_B, par_index=15)
        # Drain corp 1's treasury — move all unissued into the bank, leaving
        # player 0's one floating share intact.
        total = CORPS[1].get_total_shares()
        player_held = PLAYERS[0].get_shares(game_state, 1)
        CORPS[1].set_unissued_shares(game_state, 0)
        CORPS[1].set_issued_shares(game_state, total)
        CORPS[1].set_bank_shares(game_state, total - player_held)

        _enter_issue(game_state)

        # Corp 1 skipped (no unissued); corp 0 is the first/only decision.
        assert TURN.get_active_corp(game_state) == 0


# =============================================================================
# PHASE TRANSITIONS
# =============================================================================

class TestPhaseTransitions:
    """Transitions out of ISSUE_SHARES into IPO / INVEST."""

    def test_single_corp_transitions_after_action(self, game_state):
        """After the only corp's decision, ISSUE cascades through IPO into INVEST."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        _enter_issue(game_state)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ISSUE_SHARES)
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)
        assert TURN.get_active_corp(game_state) == -1
        assert TURN.get_active_company(game_state) == -1
        assert TURN.get_active_player(game_state) == TURN.find_player_at_position(game_state, 0)

    def test_no_active_corps_immediate_transition(self, game_state):
        """If no active corps exist, setup immediately cascades through IPO into INVEST."""
        _enter_issue(game_state)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)
        assert TURN.get_active_corp(game_state) == -1
        assert TURN.get_active_company(game_state) == -1
        assert TURN.get_active_player(game_state) == TURN.find_player_at_position(game_state, 0)

    def test_all_receivership_transitions_past_issue(self, game_state):
        """All-receivership setup auto-processes and then cascades through IPO into INVEST."""
        setup_receivership_corp(game_state, corp_id=0,
                                company_ids=[CO_A], par_index=10)

        _enter_issue(game_state)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)
        assert TURN.get_active_corp(game_state) == -1
        assert TURN.get_active_company(game_state) == -1
        assert TURN.get_active_player(game_state) == TURN.find_player_at_position(game_state, 0)

    def test_two_corps_transition_after_both(self, game_state):
        """After the second corp's decision, ISSUE cascades through IPO into INVEST."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=5)
        float_corp_for_test(game_state, corp_id=1, player_id=0,
                            company_id=CO_B, par_index=15)
        _enter_issue(game_state)

        # First corp (higher price).
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ISSUE_SHARES)
        assert TURN.get_active_corp(game_state) == 0
        assert TURN.get_active_player(game_state) == 0
        assert TURN.is_issue_remaining(game_state, 0)
        assert not TURN.is_issue_remaining(game_state, 1)

        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)
        assert TURN.get_active_corp(game_state) == -1
        assert TURN.get_active_company(game_state) == -1
        assert TURN.get_active_player(game_state) == TURN.find_player_at_position(game_state, 0)


# =============================================================================
# REMAINING FLAGS
# =============================================================================

class TestRemainingFlags:
    """Per-corp remaining flags track which corps still need processing."""

    def test_remaining_cleared_after_processing(self, game_state):
        """After a corp's decision, its remaining flag is cleared."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        _enter_issue(game_state)

        assert TURN.is_issue_remaining(game_state, 0)
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert not TURN.is_issue_remaining(game_state, 0)

    def test_inactive_corps_not_remaining(self, game_state):
        """Inactive corps have the remaining flag set to False at setup."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=15)
        _enter_issue(game_state)

        # Only corp 0 is active; every other corp slot is inactive.
        assert TURN.is_issue_remaining(game_state, 0)
        for corp_id in range(1, int(GameConstants.NUM_CORPS)):
            assert not TURN.is_issue_remaining(game_state, corp_id)

    def test_all_active_corps_marked_remaining(self, game_state):
        """Every active corp with unissued shares is marked remaining at setup."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=5)
        float_corp_for_test(game_state, corp_id=2, player_id=0,
                            company_id=CO_B, par_index=10)
        _enter_issue(game_state)

        remaining_corps = {
            corp_id
            for corp_id in range(int(GameConstants.NUM_CORPS))
            if TURN.is_issue_remaining(game_state, corp_id)
        }
        expected_remaining = {
            corp_id
            for corp_id in range(int(GameConstants.NUM_CORPS))
            if CORPS[corp_id].is_active(game_state)
            and CORPS[corp_id].get_unissued_shares(game_state) > 0
        }

        assert remaining_corps == expected_remaining
        assert remaining_corps == {0, 2}
        assert TURN.get_active_corp(game_state) in remaining_corps


# =============================================================================
# BANKRUPTCY ON ISSUE
# =============================================================================

class TestBankruptcy:
    """Normal corps go bankrupt if issuing would drop price to 0."""

    def test_corp_at_index_one_goes_bankrupt(self, game_state):
        """Corp at index 1 issuing lands on index 0 → bankruptcy."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=10)
        # Move corp 0 to index 1 (next lower to 0 triggers bankruptcy).
        CORPS[0].move_to_price_index(game_state, 1)

        _enter_issue(game_state)

        assert CORPS[0].is_active(game_state)
        issue_id = find_legal_action(game_state, action_type=ACTION_ISSUE)
        apply_and_verify(game_state, issue_id)

        assert not CORPS[0].is_active(game_state)

    def test_bankrupt_corp_companies_removed(self, game_state):
        """A corp bankrupted by issue releases its companies from ownership."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_A, par_index=10)
        CORPS[0].move_to_price_index(game_state, 1)

        _enter_issue(game_state)
        assert CORPS[0].count_companies(game_state) == 1

        issue_id = find_legal_action(game_state, action_type=ACTION_ISSUE)
        apply_and_verify(game_state, issue_id)

        assert not CORPS[0].is_active(game_state)
        assert CORPS[0].count_companies(game_state) == 0
