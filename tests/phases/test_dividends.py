"""Tests for DIVIDENDS phase (Phase 6).

Requirements covered:
- DIV-01: Dividend payment to shareholders (single/multiple players, bank shares)
- DIV-02: Max dividend constraint (price card limit AND affordability)
- DIV-03: Star calculation (companies, cash bonus, SI ability)
- DIV-04: Price movement (up 1/2, stay, down 1/2, skip occupied)
- DIV-05: Processing order (descending price)
- DIV-06: Receivership auto-handling (pay $0, adjust price)
- DIV-07: Bankruptcy (price drops to 0)
- DIV-08: Phase transitions (to IPO/INVEST or GAME_OVER)
"""
import pytest
from core.state import GameState
from core.data import (
    GamePhases, GameConstants, CorpIndices,
    get_required_stars, get_max_dividend, get_market_price,
    COMPANY_NAME_TO_ID, get_company_stars,
)
from core.actions import get_valid_action_mask, get_action_layout
from core.driver import DRIVER, STATUS_OK_PY as STATUS_OK
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES
from entities.market import MARKET
from phases.dividends import (
    setup_dividends_phase_py,
    apply_dividend_action_py,
    calculate_owned_stars_py,
    find_next_dividend_corp_py,
    calculate_price_move_py,
    find_target_index_py,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def dividend_state():
    """Create game state ready for DIVIDENDS phase with one active corp."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # Activate corp 0 (JS) with reasonable setup
    corp = CORPS[0]
    corp.set_active(state, True)
    corp.set_price_index(state, 10)  # Price ~$15
    corp.set_unissued_shares(state, 3)
    corp.set_bank_shares(state, 1)
    corp.set_issued_shares(state, 4)  # bank(1) + player(3) + player(1) = 5 issued? No: issued=4 means 4 issued
    corp.set_cash(state, 100)
    corp.set_stars(state, 5)

    # Give 2 shares to player 0, 2 to player 1
    PLAYERS[0].set_shares(state, 0, 2)
    PLAYERS[1].set_shares(state, 0, 2)
    PLAYERS[0].set_president_of(state, 0, True)

    # Mark market space as occupied
    MARKET.set_space_available(state, 10, False)

    # Transition to DIVIDENDS phase
    TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
    setup_dividends_phase_py(state)

    return state


@pytest.fixture
def multi_corp_dividend_state():
    """Create state with multiple active corps for processing order tests."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # Set up 3 corps at different prices
    # Corp 0 at price index 10 (~$15)
    CORPS[0].set_active(state, True)
    CORPS[0].set_price_index(state, 10)
    CORPS[0].set_unissued_shares(state, 3)
    CORPS[0].set_bank_shares(state, 0)
    CORPS[0].set_issued_shares(state, 4)
    CORPS[0].set_cash(state, 100)
    CORPS[0].set_stars(state, 10)
    PLAYERS[0].set_shares(state, 0, 4)
    PLAYERS[0].set_president_of(state, 0, True)
    MARKET.set_space_available(state, 10, False)

    # Corp 1 at price index 15 (~$23 - higher, processed first)
    CORPS[1].set_active(state, True)
    CORPS[1].set_price_index(state, 15)
    CORPS[1].set_unissued_shares(state, 2)
    CORPS[1].set_bank_shares(state, 0)
    CORPS[1].set_issued_shares(state, 5)
    CORPS[1].set_cash(state, 150)
    CORPS[1].set_stars(state, 12)
    PLAYERS[1].set_shares(state, 1, 5)
    PLAYERS[1].set_president_of(state, 1, True)
    MARKET.set_space_available(state, 15, False)

    # Corp 2 at price index 5 (~$8 - lowest, processed last)
    CORPS[2].set_active(state, True)
    CORPS[2].set_price_index(state, 5)
    CORPS[2].set_unissued_shares(state, 4)
    CORPS[2].set_bank_shares(state, 0)
    CORPS[2].set_issued_shares(state, 3)
    CORPS[2].set_cash(state, 50)
    CORPS[2].set_stars(state, 3)
    PLAYERS[2].set_shares(state, 2, 3)
    PLAYERS[2].set_president_of(state, 2, True)
    MARKET.set_space_available(state, 5, False)

    return state


# =============================================================================
# DIV-01: Dividend Payment Tests
# =============================================================================


class TestDividendPayment:
    """DIV-01: Dividend payment to shareholders."""

    def test_single_player_receives_dividend(self, dividend_state):
        """Single shareholder receives dividend × shares."""
        state = dividend_state
        corp = CORPS[0]

        # Player 0 has 2 shares, player 1 has 2 shares
        initial_cash_p0 = PLAYERS[0].get_cash(state)
        initial_cash_p1 = PLAYERS[1].get_cash(state)
        initial_corp_cash = corp.get_cash(state)

        # Pay $5 per share
        dividend_amount = 5
        apply_dividend_action_py(state, dividend_amount)

        # Player 0 receives 2 × $5 = $10
        assert PLAYERS[0].get_cash(state) == initial_cash_p0 + (2 * dividend_amount)
        # Player 1 receives 2 × $5 = $10
        assert PLAYERS[1].get_cash(state) == initial_cash_p1 + (2 * dividend_amount)

    def test_multiple_players_receive_dividend(self, dividend_state):
        """Multiple shareholders each receive dividend × their shares."""
        state = dividend_state
        corp = CORPS[0]

        # Set up: P0 has 1 share, P1 has 2 shares, P2 has 1 share
        PLAYERS[0].set_shares(state, 0, 1)
        PLAYERS[1].set_shares(state, 0, 2)
        PLAYERS[2].set_shares(state, 0, 1)
        corp.set_issued_shares(state, 4)

        initial_cash = [PLAYERS[i].get_cash(state) for i in range(3)]

        dividend_amount = 3
        apply_dividend_action_py(state, dividend_amount)

        assert PLAYERS[0].get_cash(state) == initial_cash[0] + 3  # 1 × $3
        assert PLAYERS[1].get_cash(state) == initial_cash[1] + 6  # 2 × $3
        assert PLAYERS[2].get_cash(state) == initial_cash[2] + 3  # 1 × $3

    def test_bank_shares_deduct_from_corp(self, dividend_state):
        """Bank shares receive dividend (deducted from corp but not paid to anyone)."""
        state = dividend_state
        corp = CORPS[0]

        # Set up: P0 has 2 shares, bank has 2 shares, issued = 4
        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[1].set_shares(state, 0, 0)
        corp.set_bank_shares(state, 2)
        corp.set_issued_shares(state, 4)
        corp.set_cash(state, 100)

        initial_corp_cash = corp.get_cash(state)

        dividend_amount = 5
        apply_dividend_action_py(state, dividend_amount)

        # Corp pays 4 × $5 = $20 total (including bank shares)
        expected_corp_cash = initial_corp_cash - (4 * dividend_amount)
        assert corp.get_cash(state) == expected_corp_cash

    def test_zero_dividend_no_payment(self, dividend_state):
        """Zero dividend results in no cash changes."""
        state = dividend_state
        corp = CORPS[0]

        initial_cash_p0 = PLAYERS[0].get_cash(state)
        initial_cash_p1 = PLAYERS[1].get_cash(state)
        initial_corp_cash = corp.get_cash(state)

        apply_dividend_action_py(state, 0)

        assert PLAYERS[0].get_cash(state) == initial_cash_p0
        assert PLAYERS[1].get_cash(state) == initial_cash_p1
        assert corp.get_cash(state) == initial_corp_cash


# =============================================================================
# DIV-02: Max Dividend Constraint Tests
# =============================================================================


class TestMaxDividendConstraint:
    """DIV-02: Max dividend is min(price_card_max, affordability)."""

    def test_max_dividend_from_price_card(self, dividend_state):
        """Max dividend limited by price card (price // 3)."""
        state = dividend_state
        corp = CORPS[0]

        # At price index 10, price = $14, max_dividend = 14 // 3 = 4
        price_index = corp.get_price_index(state)
        expected_max = get_max_dividend(price_index)
        assert expected_max == 4

        # Verify mask only allows 0-4
        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        for amount in range(5):  # 0-4 should be valid
            assert mask[layout['dividend_base'] + amount] == 1.0

        for amount in range(5, 26):  # 5-25 should be invalid
            assert mask[layout['dividend_base'] + amount] == 0.0

    def test_max_dividend_from_affordability(self, dividend_state):
        """Max dividend limited by corp cash when less than price card allows."""
        state = dividend_state
        corp = CORPS[0]

        # Corp has only $8, with 4 issued shares
        # Affordability max = 8 // 4 = 2
        corp.set_cash(state, 8)
        corp.set_issued_shares(state, 4)

        # Need to re-setup since we changed state
        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # Should only allow 0, 1, 2
        for amount in range(3):
            assert mask[layout['dividend_base'] + amount] == 1.0
        for amount in range(3, 26):
            assert mask[layout['dividend_base'] + amount] == 0.0

    def test_max_dividend_takes_minimum(self):
        """Max dividend is min(price_card, affordability)."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_price_index(state, 20)  # Price = $45, max = 45 // 3 = 15
        corp.set_issued_shares(state, 3)
        corp.set_cash(state, 12)  # Affordability = 12 // 3 = 4

        # Minimum of 15 and 4 = 4
        PLAYERS[0].set_shares(state, 0, 3)
        PLAYERS[0].set_president_of(state, 0, True)
        MARKET.set_space_available(state, 20, False)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        for amount in range(5):  # 0-4 valid
            assert mask[layout['dividend_base'] + amount] == 1.0
        for amount in range(5, 26):
            assert mask[layout['dividend_base'] + amount] == 0.0


# =============================================================================
# DIV-03: Star Calculation Tests
# =============================================================================


class TestStarCalculation:
    """DIV-03: Owned stars = company_stars + cash/10 + SI_bonus."""

    def test_company_stars_counted(self, dividend_state):
        """Stars from owned companies are counted."""
        state = dividend_state
        corp = CORPS[0]

        # Corp has 5 stars from companies
        corp.set_stars(state, 5)
        corp.set_cash(state, 0)  # No cash bonus

        owned_stars = calculate_owned_stars_py(state, 0)
        assert owned_stars == 5

    def test_cash_bonus_stars(self, dividend_state):
        """1 star per $10 cash."""
        state = dividend_state
        corp = CORPS[0]

        corp.set_stars(state, 0)  # No company stars
        corp.set_cash(state, 45)  # 45 // 10 = 4 bonus stars

        owned_stars = calculate_owned_stars_py(state, 0)
        assert owned_stars == 4

    def test_combined_stars(self, dividend_state):
        """Company stars + cash bonus combined."""
        state = dividend_state
        corp = CORPS[0]

        corp.set_stars(state, 7)
        corp.set_cash(state, 33)  # 3 bonus stars

        owned_stars = calculate_owned_stars_py(state, 0)
        assert owned_stars == 10  # 7 + 3

    def test_si_bonus_stars(self):
        """Stars, Inc. (SI) gets +2 additional stars."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up SI (corp index 7)
        si = CORPS[CorpIndices.CORP_SI]
        si.set_active(state, True)
        si.set_stars(state, 5)
        si.set_cash(state, 20)  # 2 bonus stars

        # SI should have 5 + 2 + 2 = 9 stars
        owned_stars = calculate_owned_stars_py(state, CorpIndices.CORP_SI)
        assert owned_stars == 9  # 5 company + 2 cash + 2 SI bonus

    def test_non_si_no_bonus(self, dividend_state):
        """Non-SI corps don't get the +2 bonus."""
        state = dividend_state
        corp = CORPS[0]  # JS, not SI

        corp.set_stars(state, 5)
        corp.set_cash(state, 20)

        owned_stars = calculate_owned_stars_py(state, 0)
        assert owned_stars == 7  # 5 + 2, no SI bonus


# =============================================================================
# DIV-04: Price Movement Tests
# =============================================================================


class TestPriceMovement:
    """DIV-04: Price moves based on owned vs required stars."""

    def test_move_up_two_when_diff_ge_2(self):
        """Stars >= required + 2: move up 2 tiers."""
        move = calculate_price_move_py(10, 8)  # diff = +2
        assert move == 2

        move = calculate_price_move_py(15, 10)  # diff = +5
        assert move == 2

    def test_move_up_one_when_diff_eq_1(self):
        """Stars = required + 1: move up 1 tier."""
        move = calculate_price_move_py(10, 9)  # diff = +1
        assert move == 1

    def test_no_move_when_diff_eq_0(self):
        """Stars = required: no movement."""
        move = calculate_price_move_py(10, 10)  # diff = 0
        assert move == 0

    def test_move_down_one_when_diff_eq_minus_1(self):
        """Stars = required - 1: move down 1 tier."""
        move = calculate_price_move_py(9, 10)  # diff = -1
        assert move == -1

    def test_move_down_two_when_diff_le_minus_2(self):
        """Stars <= required - 2: move down 2 tiers."""
        move = calculate_price_move_py(8, 10)  # diff = -2
        assert move == -2

        move = calculate_price_move_py(5, 10)  # diff = -5
        assert move == -2

    def test_skip_occupied_spaces_up(self):
        """Moving up skips occupied spaces."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Current index 10, moving up 2
        # Space 11 is occupied, 12 is free, 13 is free
        MARKET.set_space_available(state, 11, False)
        MARKET.set_space_available(state, 12, True)
        MARKET.set_space_available(state, 13, True)

        # Move +2 from 10 should skip 11 and land on 13
        target = find_target_index_py(state, 10, 2)
        assert target == 13

    def test_skip_occupied_spaces_down(self):
        """Moving down skips occupied spaces."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Current index 10, moving down 1
        # Space 9 is occupied
        MARKET.set_space_available(state, 9, False)
        MARKET.set_space_available(state, 8, True)

        # Move -1 from 10 should skip 9 and land on 8
        target = find_target_index_py(state, 10, -1)
        assert target == 8

    def test_move_to_max_price(self):
        """Moving up at high price goes to 26 (max, shared)."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # All spaces available
        for i in range(27):
            MARKET.set_space_available(state, i, True)

        # Move +2 from 25 should go to 26 (max)
        target = find_target_index_py(state, 25, 2)
        assert target == 26


# =============================================================================
# DIV-05: Processing Order Tests
# =============================================================================


class TestProcessingOrder:
    """DIV-05: Corps processed in descending share price order."""

    def test_highest_price_first(self, multi_corp_dividend_state):
        """Corp with highest price processes first."""
        state = multi_corp_dividend_state

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        # Corp 1 has highest price (index 15), should be first
        current_corp = TURN.get_dividend_corp(state)
        assert current_corp == 1

    def test_processing_order_sequence(self, multi_corp_dividend_state):
        """Corps process in order: highest -> middle -> lowest price."""
        state = multi_corp_dividend_state

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        # Verify processing order by checking find_next_dividend_corp
        order = []

        # First corp (Corp 1, price 15)
        corp_id = find_next_dividend_corp_py(state)
        order.append(corp_id)
        assert corp_id == 1
        TURN.set_dividend_remaining(state, corp_id, False)

        # Second corp (Corp 0, price 10)
        corp_id = find_next_dividend_corp_py(state)
        order.append(corp_id)
        assert corp_id == 0
        TURN.set_dividend_remaining(state, corp_id, False)

        # Third corp (Corp 2, price 5)
        corp_id = find_next_dividend_corp_py(state)
        order.append(corp_id)
        assert corp_id == 2
        TURN.set_dividend_remaining(state, corp_id, False)

        # No more corps
        assert find_next_dividend_corp_py(state) == -1

        assert order == [1, 0, 2]  # Descending price order


# =============================================================================
# DIV-06: Receivership Auto-Handling Tests
# =============================================================================


class TestReceivershipHandling:
    """DIV-06: Receivership corps pay $0 and auto-process."""

    def test_receivership_auto_processed(self):
        """Corp in receivership is auto-processed without player input."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up receivership corp at highest price
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_price_index(state, 15)
        corp.set_in_receivership(state, True)
        corp.set_stars(state, 10)
        corp.set_cash(state, 100)
        corp.set_issued_shares(state, 4)
        corp.set_bank_shares(state, 4)  # All issued shares in bank
        MARKET.set_space_available(state, 15, False)

        # Set up player-controlled corp at lower price
        corp1 = CORPS[1]
        corp1.set_active(state, True)
        corp1.set_price_index(state, 10)
        corp1.set_stars(state, 5)
        corp1.set_cash(state, 50)
        corp1.set_issued_shares(state, 3)
        PLAYERS[0].set_shares(state, 1, 3)
        PLAYERS[0].set_president_of(state, 1, True)
        MARKET.set_space_available(state, 10, False)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        # Should skip receivership corp and go to player-controlled corp
        current_corp = TURN.get_dividend_corp(state)
        assert current_corp == 1  # Corp 0 was auto-processed

    def test_receivership_pays_zero_dividend(self):
        """Receivership corp pays $0 dividend (cash unchanged)."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_price_index(state, 10)
        corp.set_in_receivership(state, True)
        corp.set_stars(state, 10)  # Plenty of stars, should rise
        corp.set_cash(state, 100)
        corp.set_issued_shares(state, 4)
        corp.set_bank_shares(state, 4)
        MARKET.set_space_available(state, 10, False)
        MARKET.set_space_available(state, 11, True)
        MARKET.set_space_available(state, 12, True)

        initial_cash = corp.get_cash(state)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        # Cash should be unchanged (no dividend paid)
        assert corp.get_cash(state) == initial_cash

    def test_receivership_price_adjusts(self):
        """Receivership corp still adjusts price based on stars."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_price_index(state, 10)
        corp.set_in_receivership(state, True)
        corp.set_stars(state, 50)  # Way more than needed, should rise 2
        corp.set_cash(state, 100)
        corp.set_issued_shares(state, 3)
        corp.set_bank_shares(state, 3)

        # Make spaces available
        for i in range(27):
            MARKET.set_space_available(state, i, True)
        MARKET.set_space_available(state, 10, False)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        # Price should have risen
        new_price_index = corp.get_price_index(state)
        assert new_price_index > 10


# =============================================================================
# DIV-07: Bankruptcy Tests
# =============================================================================


class TestBankruptcy:
    """DIV-07: Price drop to 0 triggers bankruptcy."""

    def test_bankruptcy_on_price_zero(self):
        """Corp goes bankrupt when price drops to 0."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_price_index(state, 1)  # Very low, will drop to 0
        corp.set_in_receivership(state, False)
        corp.set_stars(state, 0)  # No stars, will drop
        corp.set_cash(state, 0)  # No cash bonus
        corp.set_issued_shares(state, 3)
        PLAYERS[0].set_shares(state, 0, 3)
        PLAYERS[0].set_president_of(state, 0, True)
        MARKET.set_space_available(state, 1, False)
        MARKET.set_space_available(state, 0, True)

        # Give corp a company so bankruptcy has something to liquidate
        COMPANIES[0].transfer_to_corp(state, 0)
        corp.set_owns_company(state, 0, True)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        # Pay $0 dividend to trigger price adjustment
        apply_dividend_action_py(state, 0)

        # Corp should now be bankrupt (inactive)
        assert not corp.is_active(state)

    def test_bankruptcy_clears_shares(self):
        """Bankruptcy removes all shares from players."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_price_index(state, 1)
        corp.set_stars(state, 0)
        corp.set_cash(state, 0)
        corp.set_issued_shares(state, 4)
        corp.set_bank_shares(state, 1)
        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[1].set_shares(state, 0, 2)
        PLAYERS[0].set_president_of(state, 0, True)
        MARKET.set_space_available(state, 1, False)

        COMPANIES[0].transfer_to_corp(state, 0)
        corp.set_owns_company(state, 0, True)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        apply_dividend_action_py(state, 0)

        # Shares should be cleared
        assert PLAYERS[0].get_shares(state, 0) == 0
        assert PLAYERS[1].get_shares(state, 0) == 0


# =============================================================================
# DIV-08: Phase Transition Tests
# =============================================================================


class TestPhaseTransitions:
    """DIV-08: Phase transitions after all corps processed."""

    def test_transition_to_end_card(self, dividend_state):
        """After all corps processed, transitions to END_CARD."""
        state = dividend_state

        # Only one corp, pay dividend and phase should transition
        apply_dividend_action_py(state, 0)

        assert state.get_phase() == GamePhases.PHASE_END_CARD

    def test_transition_to_end_card_when_flipped(self, dividend_state):
        """Even if end card flipped, DIVIDENDS transitions to END_CARD (which handles GAME_OVER)."""
        state = dividend_state

        # Set end card as flipped
        TURN.set_end_card_flipped(state, True)

        apply_dividend_action_py(state, 0)

        # DIVIDENDS always transitions to END_CARD; END_CARD handles the GAME_OVER logic
        assert state.get_phase() == GamePhases.PHASE_END_CARD

    def test_no_active_corps_immediate_transition(self):
        """If no active corps, immediately transitions to END_CARD."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # No corps are active
        for i in range(8):
            CORPS[i].set_active(state, False)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        # Should immediately transition to END_CARD
        assert state.get_phase() == GamePhases.PHASE_END_CARD


# =============================================================================
# Integration Tests
# =============================================================================


class TestDividendsIntegration:
    """Integration tests for complete dividend scenarios."""

    def test_full_dividend_cycle(self, multi_corp_dividend_state):
        """Complete dividend cycle with multiple corps."""
        state = multi_corp_dividend_state

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        # Process all three corps
        assert TURN.get_dividend_corp(state) == 1  # Highest price first
        apply_dividend_action_py(state, 2)

        assert TURN.get_dividend_corp(state) == 0
        apply_dividend_action_py(state, 3)

        assert TURN.get_dividend_corp(state) == 2
        apply_dividend_action_py(state, 1)

        # Should transition to END_CARD
        assert state.get_phase() == GamePhases.PHASE_END_CARD

    def test_dividend_affects_price_adjustment(self, dividend_state):
        """Dividend reduces corp cash which affects cash bonus stars."""
        state = dividend_state
        corp = CORPS[0]

        # Set up corp with stars exactly meeting requirement
        # Price index 10, issued 4 -> required = round(4 * 15 / 10) = 6
        corp.set_stars(state, 5)  # Need 1 more star from cash
        corp.set_cash(state, 15)  # 1 bonus star -> total 6, exactly meets

        # Make spaces available
        for i in range(27):
            MARKET.set_space_available(state, i, True)
        MARKET.set_space_available(state, 10, False)

        initial_price = corp.get_price_index(state)

        # Re-setup to recalculate
        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        # Pay $3 per share × 4 shares = $12 total
        # After payment: cash = 15 - 12 = 3, only 0 bonus stars
        # Stars = 5 + 0 = 5, required = 6, diff = -1 -> down 1
        apply_dividend_action_py(state, 3)

        # Price should have dropped
        new_price = corp.get_price_index(state)
        assert new_price < initial_price

    def test_required_stars_formula(self):
        """Verify required stars formula: round(issued * price / 10)."""
        # Price index 10 = $14, issued = 4
        # Required = round(4 * 14 / 10) = round(5.6) = 6
        assert get_required_stars(10, 4) == 6

        # Price index 15 = $24, issued = 5
        # Required = round(5 * 24 / 10) = round(12) = 12
        assert get_required_stars(15, 5) == 12

        # Price index 5 = $9, issued = 3
        # Required = round(3 * 9 / 10) = round(2.7) = 3
        assert get_required_stars(5, 3) == 3
