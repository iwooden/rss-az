# cython: language_level=3
"""Consolidated tests for bankruptcy procedure.

Bankruptcy can be triggered by:
1. INVEST phase - selling shares drops price to 0
2. DIVIDENDS phase - dividend price adjustment drops to 0
3. INCOME phase - corp cash goes negative
4. ISSUE phase - issuing shares drops price to 0

Core bankruptcy behavior (tested once):
- Corp becomes inactive
- All companies removed from game
- All shares return to unissued
- Corp cash cleared
- Market space freed
- President flags cleared
- All player net worths updated

This file consolidates bankruptcy tests from:
- tests/phases/test_invest.py (TestBankruptcy)
- tests/phases/test_dividends.py (TestBankruptcy)
- tests/phases/test_income.py (TestCorpBankruptcy, TestMultipleBankruptcies)
- tests/phases/test_issue.py (TestBankruptcy)
"""
import pytest
from core.state import GameState
from core.data import GamePhases, get_corp_share_count, COMPANY_NAME_TO_ID
from core.actions import get_action_layout
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES
from entities.turn import TURN
from entities.market import MARKET
from phases.dividends import setup_dividends_phase_py, apply_dividend_action_py
from phases.income import apply_income_py
from phases.issue import setup_issue_phase_py, apply_issue_action_py
from tests.phases.conftest import apply_and_verify_all, assert_invariants, float_corp_for_test


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def bankruptcy_state():
    """State where one sell triggers bankruptcy (price index 1 -> 0).

    Uses company_id=3 explicitly because company 0 is excluded from 3-player games.
    """
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # Float corp 0 with company 3 (company 0 is excluded from 3-player games)
    float_corp_for_test(state, corp_id=0, company_id=3, par_index=1, float_shares=2)
    # float_shares=2 gives: player=2, bank=2, issued=4

    PLAYERS[0].set_cash(state, 100)

    return state


@pytest.fixture
def dividend_bankruptcy_state():
    """State where paying $0 dividend triggers bankruptcy."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # Float corp 0 at low price
    float_corp_for_test(state, corp_id=0, par_index=1)

    corp = CORPS[0]
    corp.set_stars(state, 0)  # No stars, will drop on $0 dividend
    corp.set_cash(state, 0)  # No cash bonus
    # Adjust shares for test scenario: player=3, bank=0, issued=3, unissued=4
    # set_shares auto-adjusts bank by -(3-1)=-2, so start bank at 2
    corp.set_issued_shares(state, 3)
    corp.set_unissued_shares(state, 4)
    corp.set_bank_shares(state, 2)
    PLAYERS[0].set_shares(state, 0, 3)  # bank: 2-2=0

    MARKET.set_space_available(state, 0, True)  # Ensure space 0 is open

    return state


@pytest.fixture
def income_bankruptcy_state():
    """State where negative income triggers bankruptcy."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # High CoO for negative income
    TURN.set_coo_level(state, 6)

    # Float corp 0
    float_corp_for_test(state, corp_id=0, par_index=10)
    CORPS[0].set_cash(state, 1)  # Not enough to cover negative income

    # Give corp a negative-income company
    # KK: income=5, stars=3. At CoO level 6, 3-star CoO=7. Adjusted = -2
    kk = COMPANY_NAME_TO_ID["KK"]
    COMPANIES[kk].transfer_to_corp(state, 0)

    return state


@pytest.fixture
def issue_bankruptcy_state():
    """State where issuing shares triggers bankruptcy."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # Float corp 0 at price index 1 (one above bankruptcy)
    float_corp_for_test(state, corp_id=0, par_index=1, float_shares=2)
    # float_shares=2 gives: player=2, bank=2, issued=4, unissued=3

    corp = CORPS[0]
    corp.set_cash(state, 50)

    # Give player 1 some shares too (auto-adjusts bank: 2-1=1)
    PLAYERS[1].set_shares(state, 0, 1)
    # Final: player0=2, player1=1, bank=0, issued=3, unissued=4
    corp.set_issued_shares(state, 3)
    corp.set_unissued_shares(state, 4)
    corp.set_bank_shares(state, 0)

    return state


# =============================================================================
# CORE BANKRUPTCY BEHAVIOR (tested once, via sell trigger)
# =============================================================================

class TestCoreBankruptcyBehavior:
    """Core bankruptcy procedure - tested via INVEST sell trigger.

    These tests verify the common bankruptcy effects that are identical
    regardless of which phase triggers the bankruptcy.
    """

    def test_bankruptcy_deactivates_corp(self, bankruptcy_state):
        """Corp becomes inactive after bankruptcy."""
        corp = CORPS[0]
        assert corp.is_active(bankruptcy_state)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(bankruptcy_state, sell_idx)

        assert not corp.is_active(bankruptcy_state)
        assert corp.get_price_index(bankruptcy_state) == 0

    def test_bankruptcy_removes_companies(self, bankruptcy_state):
        """Bankruptcy removes all corp's companies from game."""
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(bankruptcy_state, sell_idx)

        # Corp 0 was floated with company 3 (see bankruptcy_state fixture)
        assert COMPANIES[3].is_removed(bankruptcy_state)

    def test_bankruptcy_removes_multiple_companies(self, bankruptcy_state):
        """Bankruptcy removes ALL companies owned by corp."""
        # Add second company to corp (company 6 is in the deck for 3-player games)
        COMPANIES[6].transfer_to_corp(bankruptcy_state, 0)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(bankruptcy_state, sell_idx)

        # Corp 0 was floated with company 3, and we added company 6
        assert COMPANIES[3].is_removed(bankruptcy_state)
        assert COMPANIES[6].is_removed(bankruptcy_state)

    def test_bankruptcy_returns_shares_to_unissued(self, bankruptcy_state):
        """All shares return to unissued pool."""
        corp = CORPS[0]

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(bankruptcy_state, sell_idx)

        assert corp.get_unissued_shares(bankruptcy_state) == get_corp_share_count(0)
        assert corp.get_issued_shares(bankruptcy_state) == 0
        assert corp.get_bank_shares(bankruptcy_state) == 0

    def test_bankruptcy_clears_player_shares(self, bankruptcy_state):
        """All players' shares in bankrupt corp are zeroed."""
        # bankruptcy_state has player0=2, bank=2, issued=4 (float_shares=2)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(bankruptcy_state, sell_idx)

        assert PLAYERS[0].get_shares(bankruptcy_state, 0) == 0
        assert PLAYERS[1].get_shares(bankruptcy_state, 0) == 0

    def test_bankruptcy_clears_corp_cash(self, bankruptcy_state):
        """Corp cash returned to bank (set to 0)."""
        corp = CORPS[0]
        corp.set_cash(bankruptcy_state, 50)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(bankruptcy_state, sell_idx)

        assert corp.get_cash(bankruptcy_state) == 0

    def test_bankruptcy_frees_market_space(self, bankruptcy_state):
        """Market space freed for future use."""
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(bankruptcy_state, sell_idx)

        assert MARKET.is_space_available(bankruptcy_state, 1)

    def test_bankruptcy_clears_receivership(self, bankruptcy_state):
        """Bankrupt corp is no longer in receivership."""
        corp = CORPS[0]
        # Move player shares to bank, triggering auto-receivership via _recalculate_presidency
        # set_shares auto-adjusts bank shares by inverse delta
        PLAYERS[0].set_shares(bankruptcy_state, 0, 0)
        assert corp.is_in_receivership(bankruptcy_state)

        # Trigger bankruptcy through dividends (sell requires player shares)
        TURN.set_phase(bankruptcy_state, GamePhases.PHASE_DIVIDENDS)
        corp.set_stars(bankruptcy_state, 0)
        corp.set_cash(bankruptcy_state, 0)
        setup_dividends_phase_py(bankruptcy_state)
        apply_dividend_action_py(bankruptcy_state, 0)
        assert_invariants(bankruptcy_state, "After dividend-triggered bankruptcy (receivership)")

        assert not corp.is_in_receivership(bankruptcy_state)

    def test_bankruptcy_corp_available_for_ipo(self, bankruptcy_state):
        """Bankrupt corp can be IPO'd again."""
        corp = CORPS[0]

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(bankruptcy_state, sell_idx)

        assert not corp.is_active(bankruptcy_state)
        assert corp.get_unissued_shares(bankruptcy_state) > 0
        assert not corp.is_in_receivership(bankruptcy_state)

    def test_bankruptcy_updates_all_shareholders_net_worth(self, bankruptcy_state):
        """Bankruptcy updates net worth for all players who held shares."""
        corp = CORPS[0]

        # Give multiple players shares
        # bankruptcy_state starts with player0=2, bank=2, issued=4
        # set_shares auto-adjusts bank: 2->1->0
        PLAYERS[1].set_shares(bankruptcy_state, 0, 1)
        PLAYERS[2].set_shares(bankruptcy_state, 0, 1)
        # player0=2 + player1=1 + player2=1 + bank=0 = issued=4

        PLAYERS[1].set_cash(bankruptcy_state, 50)
        PLAYERS[2].set_cash(bankruptcy_state, 50)

        # Update net worth before
        PLAYERS[1].update_net_worth(bankruptcy_state)
        initial_net_worth_p1 = PLAYERS[1].get_net_worth(bankruptcy_state)
        share_value = corp.get_share_price(bankruptcy_state)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(bankruptcy_state, sell_idx)

        # Player 1 lost their shares
        assert PLAYERS[1].get_shares(bankruptcy_state, 0) == 0
        new_net_worth_p1 = PLAYERS[1].get_net_worth(bankruptcy_state)
        assert new_net_worth_p1 == initial_net_worth_p1 - share_value

    @pytest.mark.parametrize("num_players", [3, 6])
    def test_bankruptcy_different_player_counts(self, num_players):
        """Bankruptcy procedure works for all player counts."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Float corp 0 at bankruptcy-prone price
        float_corp_for_test(state, corp_id=0, par_index=1, float_shares=2)
        PLAYERS[0].set_cash(state, 100)

        layout = get_action_layout(num_players)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(state, sell_idx)

        assert not CORPS[0].is_active(state)


# =============================================================================
# TRIGGER: INVEST PHASE (sell drops price to 0)
# =============================================================================

class TestBankruptcyFromSell:
    """Bankruptcy triggered by selling shares in INVEST phase."""

    def test_sell_triggers_bankruptcy_at_price_zero(self, bankruptcy_state):
        """Sell that drops price to 0 triggers bankruptcy."""
        corp = CORPS[0]
        assert corp.get_price_index(bankruptcy_state) == 1

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(bankruptcy_state, sell_idx)

        assert not corp.is_active(bankruptcy_state)
        assert corp.get_price_index(bankruptcy_state) == 0


# =============================================================================
# TRIGGER: DIVIDENDS PHASE (price adjustment drops to 0)
# =============================================================================

class TestBankruptcyFromDividend:
    """Bankruptcy triggered by dividend price adjustment."""

    def test_dividend_triggers_bankruptcy_at_price_zero(self, dividend_bankruptcy_state):
        """Dividend price adjustment dropping to 0 triggers bankruptcy."""
        state = dividend_bankruptcy_state
        corp = CORPS[0]

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        # Pay $0 dividend to trigger price adjustment
        apply_dividend_action_py(state, 0)
        assert_invariants(state, "After apply_dividend_action_py($0)")

        assert not corp.is_active(state)

    def test_dividend_bankruptcy_clears_shares(self, dividend_bankruptcy_state):
        """Dividend-triggered bankruptcy clears all player shares."""
        state = dividend_bankruptcy_state

        # Add another shareholder
        # dividend_bankruptcy_state: player0=3, bank=0, issued=3, unissued=4
        # Increase issued/bank to make room, then set_shares auto-adjusts bank
        CORPS[0].set_issued_shares(state, 5)
        CORPS[0].set_unissued_shares(state, 2)
        CORPS[0].set_bank_shares(state, 2)
        PLAYERS[1].set_shares(state, 0, 2)  # bank: 2-2=0
        # Final: player0=3, player1=2, bank=0, issued=5, unissued=2

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        apply_dividend_action_py(state, 0)
        assert_invariants(state, "After apply_dividend_action_py($0)")

        assert PLAYERS[0].get_shares(state, 0) == 0
        assert PLAYERS[1].get_shares(state, 0) == 0

    def test_bankruptcy_from_price_2_with_minus_2_move(self):
        """Corp at price index 2 with -2 move goes bankrupt (issue l4g)."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp 0 at price index 2
        float_corp_for_test(state, corp_id=0, par_index=2)

        corp = CORPS[0]
        corp.set_stars(state, 0)  # Very low stars -> move will be -2 or worse
        corp.set_cash(state, 0)
        # Adjust shares: player=3, bank=0, issued=3, unissued=4
        # set_shares auto-adjusts bank by -(3-1)=-2, so start bank at 2
        corp.set_issued_shares(state, 3)
        corp.set_unissued_shares(state, 4)
        corp.set_bank_shares(state, 2)
        PLAYERS[0].set_shares(state, 0, 3)  # bank: 2-2=0

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        # Pay $0 dividend - with 0 stars and 0 cash, move should be negative
        apply_dividend_action_py(state, 0)
        assert_invariants(state, "After apply_dividend_action_py($0)")

        # Corp should be bankrupt (price dropped to 0)
        assert not corp.is_active(state)


# =============================================================================
# TRIGGER: INCOME PHASE (negative cash)
# =============================================================================

class TestBankruptcyFromIncome:
    """Bankruptcy triggered by negative cash during INCOME phase.

    This is unique - bankruptcy is from cash going negative,
    not from price dropping to 0.
    """

    def test_negative_income_triggers_bankruptcy(self, income_bankruptcy_state):
        """Corp with insufficient cash for negative income goes bankrupt."""
        state = income_bankruptcy_state
        corp = CORPS[0]

        # Verify setup - income should be negative
        income = corp.calculate_income(state)
        assert income < 0, f"Expected negative income, got {income}"
        assert corp.get_cash(state) + income < 0

        TURN.set_phase(state, GamePhases.PHASE_INCOME)
        apply_income_py(state)
        assert_invariants(state, "After apply_income_py")

        assert not corp.is_active(state)

    def test_corp_survives_with_sufficient_cash(self, income_bankruptcy_state):
        """Corp with enough cash to cover negative income survives."""
        state = income_bankruptcy_state
        corp = CORPS[0]
        corp.set_cash(state, 100)  # Plenty of cash

        income = corp.calculate_income(state)
        starting_cash = corp.get_cash(state)

        TURN.set_phase(state, GamePhases.PHASE_INCOME)
        apply_income_py(state)
        assert_invariants(state, "After apply_income_py (survives)")

        assert corp.is_active(state)
        assert corp.get_cash(state) == starting_cash + income

    def test_bankruptcy_check_immediate_after_income(self):
        """Bankruptcy is checked immediately after each corp's income."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        TURN.set_coo_level(state, 6)

        # Corp 0: will go bankrupt (low cash)
        float_corp_for_test(state, corp_id=0, par_index=10)
        CORPS[0].set_cash(state, 0)

        # Corp 1: will survive (high cash)
        float_corp_for_test(state, corp_id=1, player_id=1, par_index=15)
        CORPS[1].set_cash(state, 100)

        # Give both negative income companies
        kk = COMPANY_NAME_TO_ID["KK"]
        dr = COMPANY_NAME_TO_ID["DR"]
        COMPANIES[kk].transfer_to_corp(state, 0)
        COMPANIES[dr].transfer_to_corp(state, 1)

        TURN.set_phase(state, GamePhases.PHASE_INCOME)
        apply_income_py(state)
        assert_invariants(state, "After apply_income_py (immediate check)")

        assert not CORPS[0].is_active(state), "Corp 0 should be bankrupt"
        assert CORPS[1].is_active(state), "Corp 1 should survive"

    def test_two_corps_go_bankrupt_simultaneously(self):
        """Two corps with negative income both go bankrupt."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        TURN.set_coo_level(state, 6)

        # Set up two corps that will both go bankrupt
        companies = ["KK", "DR"]
        for corp_id, company_name in enumerate(companies):
            float_corp_for_test(state, corp_id=corp_id, player_id=corp_id, par_index=10 + corp_id)
            CORPS[corp_id].set_cash(state, 1)  # Not enough for negative income

            cid = COMPANY_NAME_TO_ID[company_name]
            COMPANIES[cid].transfer_to_corp(state, corp_id)

        TURN.set_phase(state, GamePhases.PHASE_INCOME)
        apply_income_py(state)
        assert_invariants(state, "After apply_income_py (two bankruptcies)")

        assert not CORPS[0].is_active(state)
        assert not CORPS[1].is_active(state)

    def test_bankruptcy_order_is_corp_id_order(self):
        """Corps are processed in corp_id order (0-7)."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        TURN.set_coo_level(state, 6)

        # Set up corps 0, 2, 4 with bankruptcy conditions
        companies = ["KK", "DR", "BY"]
        corp_ids = [0, 2, 4]
        for corp_id, company_name in zip(corp_ids, companies):
            float_corp_for_test(state, corp_id=corp_id, player_id=corp_id % 3, par_index=10 + corp_id)
            CORPS[corp_id].set_cash(state, 0)  # No cash for negative income
            cid = COMPANY_NAME_TO_ID[company_name]
            COMPANIES[cid].transfer_to_corp(state, corp_id)

        TURN.set_phase(state, GamePhases.PHASE_INCOME)
        apply_income_py(state)
        assert_invariants(state, "After apply_income_py (corp_id order)")

        for corp_id in corp_ids:
            assert not CORPS[corp_id].is_active(state), f"Corp {corp_id} should be bankrupt"


# =============================================================================
# TRIGGER: ISSUE PHASE (issuing drops price to 0)
# =============================================================================

class TestBankruptcyFromIssue:
    """Bankruptcy triggered by issuing shares in ISSUE phase."""

    def test_issue_at_low_price_causes_bankruptcy(self, issue_bankruptcy_state):
        """Corp at price index 1 goes bankrupt when issuing."""
        state = issue_bankruptcy_state
        corp = CORPS[0]

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)
        setup_issue_phase_py(state)
        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After apply_issue_action_py (bankruptcy)")

        assert not corp.is_active(state)

    def test_issue_bankruptcy_clears_player_shares(self, issue_bankruptcy_state):
        """Issue-triggered bankruptcy clears all player shares."""
        state = issue_bankruptcy_state

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)
        setup_issue_phase_py(state)
        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After apply_issue_action_py (clears shares)")

        assert PLAYERS[0].get_shares(state, 0) == 0
        assert PLAYERS[1].get_shares(state, 0) == 0


# =============================================================================
# EDGE CASES
# =============================================================================

class TestBankruptcyEdgeCases:
    """Edge cases and boundary conditions for bankruptcy."""

    def test_bankruptcy_at_price_1_not_price_0(self):
        """Corp at price 1 survives if price doesn't drop to 0."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp 0 at low price
        float_corp_for_test(state, corp_id=0, par_index=1)

        corp = CORPS[0]
        corp.set_stars(state, 10)  # High stars - won't drop
        corp.set_cash(state, 100)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        # Pay max dividend - with high stars, should move up or stay
        apply_dividend_action_py(state, 25)
        assert_invariants(state, "After apply_dividend_action_py($25)")

        # Corp should still be active
        assert corp.is_active(state)

    def test_receivership_corp_can_go_bankrupt(self):
        """Corp in receivership can still go bankrupt."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp 0 at low price
        float_corp_for_test(state, corp_id=0, par_index=1)

        corp = CORPS[0]
        corp.set_in_receivership(state, True)  # Put into receivership
        PLAYERS[0].set_shares(state, 0, 0)  # Remove player shares (auto-adjusts bank: 1+1=2)
        corp.set_stars(state, 0)
        corp.set_cash(state, 0)
        # Adjust shares: all in bank. player0=0, bank=3, issued=3, unissued=4
        corp.set_issued_shares(state, 3)
        corp.set_bank_shares(state, 3)
        corp.set_unissued_shares(state, 4)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        # Receivership pays $0 automatically
        apply_dividend_action_py(state, 0)
        assert_invariants(state, "After apply_dividend_action_py($0 receivership)")

        assert not corp.is_active(state)
        assert not corp.is_in_receivership(state)
