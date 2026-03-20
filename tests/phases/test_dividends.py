"""Tests for DIVIDENDS phase (Phase 6)."""
import pytest
from core.state import GameState, get_layout
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
from tests.phases.conftest import float_corp_for_test, assert_invariants
from phases.dividends import (
    setup_dividends_phase_py,
    apply_dividend_action_py,
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

    # Float corp 0 with 2 shares each to player and bank
    float_corp_for_test(state, corp_id=0, float_shares=2)

    # Adjust share distribution: give shares to player 1
    # float_corp gave: unissued=3, issued=4, bank=2, player0=2
    # We want: bank=1, player0=2, player1=2 -> issued=5, unissued=2
    CORPS[0].set_issued_shares(state, 5)
    CORPS[0].set_unissued_shares(state, 2)
    CORPS[0].set_bank_shares(state, 3)  # set_shares below will subtract 2
    PLAYERS[1].set_shares(state, 0, 2)  # bank: 3 - 2 = 1

    # Set cash (stars auto-calculated: 1★ AKE + 100//10 = 11)
    CORPS[0].set_cash(state, 100)

    # Transition to DIVIDENDS phase
    TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
    setup_dividends_phase_py(state)
    assert_invariants(state, "After setup_dividends_phase_py in dividend_state fixture")

    return state


@pytest.fixture
def multi_corp_dividend_state():
    """Create state with multiple active corps for processing order tests."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # Float 3 corps at different prices (price order matters for this test)
    # Corp 0 at price index 10 (total=7)
    # After float(float_shares=2): player0=2, bank=2, issued=4, unissued=3
    # Want: player0=4, bank=0, issued=4, unissued=3
    float_corp_for_test(state, corp_id=0, player_id=0, par_index=10, float_shares=2)
    PLAYERS[0].set_shares(state, 0, 4)  # delta=+2, bank: 2-2=0
    CORPS[0].set_unissued_shares(state, 3)
    CORPS[0].set_cash(state, 100)

    # Corp 1 at price index 15 (higher, processed first, total=7)
    # After float(float_shares=2): player1=2, bank=2, issued=4, unissued=3
    # Want: player1=5, bank=0, issued=5, unissued=2
    float_corp_for_test(state, corp_id=1, player_id=1, par_index=15, float_shares=2)
    CORPS[1].set_issued_shares(state, 5)
    CORPS[1].set_unissued_shares(state, 2)
    CORPS[1].set_bank_shares(state, 3)  # set_shares below will subtract 3
    PLAYERS[1].set_shares(state, 1, 5)  # delta=+3, bank: 3-3=0
    CORPS[1].set_cash(state, 150)

    # Corp 2 at price index 5 (lowest, processed last, total=6)
    # After float(float_shares=1): player2=1, bank=1, issued=2, unissued=4
    # Want: player2=3, bank=0, issued=3, unissued=3
    float_corp_for_test(state, corp_id=2, player_id=2, par_index=5)
    CORPS[2].set_issued_shares(state, 3)
    CORPS[2].set_unissued_shares(state, 3)
    CORPS[2].set_bank_shares(state, 2)  # set_shares below will subtract 2
    PLAYERS[2].set_shares(state, 2, 3)  # delta=+2, bank: 2-2=0
    CORPS[2].set_cash(state, 50)

    return state


# =============================================================================
# Dividend Payment Tests
# =============================================================================


class TestDividendPayment:
    """Dividend payment to shareholders."""

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
        assert_invariants(state, "After dividend action")

        # Player 0 receives 2 × $5 = $10
        assert PLAYERS[0].get_cash(state) == initial_cash_p0 + (2 * dividend_amount)
        # Player 1 receives 2 × $5 = $10
        assert PLAYERS[1].get_cash(state) == initial_cash_p1 + (2 * dividend_amount)

    def test_multiple_players_receive_dividend(self, dividend_state):
        """Multiple shareholders each receive dividend × their shares."""
        state = dividend_state
        corp = CORPS[0]

        # Set up: P0 has 1 share, P1 has 2 shares, P2 has 1 share, bank=0
        # Fixture state: P0=2, P1=2, bank=1, issued=5, unissued=2
        # Want: P0=1, P1=2, P2=1, bank=0, issued=4, unissued=3
        corp.set_issued_shares(state, 4)
        corp.set_unissued_shares(state, 3)
        corp.set_bank_shares(state, 0)  # pre-set; set_shares deltas net to 0
        PLAYERS[0].set_shares(state, 0, 1)  # delta=-1, bank: 0+1=1
        PLAYERS[1].set_shares(state, 0, 2)  # delta=0, bank stays 1
        PLAYERS[2].set_shares(state, 0, 1)  # delta=+1, bank: 1-1=0

        initial_cash = [PLAYERS[i].get_cash(state) for i in range(3)]

        dividend_amount = 3
        apply_dividend_action_py(state, dividend_amount)
        assert_invariants(state, "After dividend action")

        assert PLAYERS[0].get_cash(state) == initial_cash[0] + 3  # 1 × $3
        assert PLAYERS[1].get_cash(state) == initial_cash[1] + 6  # 2 × $3
        assert PLAYERS[2].get_cash(state) == initial_cash[2] + 3  # 1 × $3

    def test_bank_shares_deduct_from_corp(self, dividend_state):
        """Bank shares receive dividend (deducted from corp but not paid to anyone)."""
        state = dividend_state
        corp = CORPS[0]

        # Set up: P0 has 2 shares, bank has 2 shares, issued = 4
        # Fixture state: P0=2, P1=2, bank=1, issued=5, unissued=2
        # Want: P0=2, P1=0, bank=2, issued=4, unissued=3
        corp.set_issued_shares(state, 4)
        corp.set_unissued_shares(state, 3)
        corp.set_bank_shares(state, 0)  # set_shares(P1,0) will add 2
        PLAYERS[0].set_shares(state, 0, 2)  # delta=0, bank stays 0
        PLAYERS[1].set_shares(state, 0, 0)  # delta=-2, bank: 0+2=2
        corp.set_cash(state, 100)

        initial_corp_cash = corp.get_cash(state)

        dividend_amount = 5
        apply_dividend_action_py(state, dividend_amount)
        assert_invariants(state, "After dividend action")

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
        assert_invariants(state, "After dividend action")

        assert PLAYERS[0].get_cash(state) == initial_cash_p0
        assert PLAYERS[1].get_cash(state) == initial_cash_p1
        assert corp.get_cash(state) == initial_corp_cash


# =============================================================================
# Max Dividend Constraint Tests
# =============================================================================


class TestMaxDividendConstraint:
    """Max dividend is min(price_card_max, affordability)."""

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
        # Total = 7: unissued=3, issued=4 (0 bank + 2 p0 + 2 p1)
        corp.set_cash(state, 8)
        corp.set_bank_shares(state, 0)
        corp.set_issued_shares(state, 4)
        corp.set_unissued_shares(state, 3)

        # Need to re-setup since we changed state
        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        assert_invariants(state, "After setup_dividends_phase_py")

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

        # Float corp 0 at price index 20 (price matters: max = 45 // 3 = 15)
        # With float_shares=1: issued=2, but we need issued=3 for affordability test
        float_corp_for_test(state, corp_id=0, par_index=20)
        corp = CORPS[0]

        # Adjust shares: need 3 issued for affordability calc (12 // 3 = 4)
        # After float: player0=1, bank=1, issued=2, unissued=5
        # Want: player0=3, bank=0, issued=3, unissued=4
        corp.set_issued_shares(state, 3)
        corp.set_unissued_shares(state, 4)
        corp.set_bank_shares(state, 2)  # set_shares below will subtract 2
        PLAYERS[0].set_shares(state, 0, 3)  # delta=+2, bank: 2-2=0
        corp.set_cash(state, 12)  # Affordability = 12 // 3 = 4 (less than price max of 15)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        assert_invariants(state, "After setup_dividends_phase_py")

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        for amount in range(5):  # 0-4 valid
            assert mask[layout['dividend_base'] + amount] == 1.0
        for amount in range(5, 26):
            assert mask[layout['dividend_base'] + amount] == 0.0

    def test_max_dividend_at_highest_price(self):
        """At price_index 26 ($75), max dividend is 25 (highest in action space)."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp 0 at price index 26 (max price - this is what we're testing)
        float_corp_for_test(state, corp_id=0, par_index=26)
        corp = CORPS[0]

        # Index 26 is a shared space (multiple corps can occupy it), re-mark available
        MARKET.set_space_available(state, 26, True)

        # Need enough cash to afford max dividend (2 shares * 25 = 50)
        corp.set_cash(state, 100)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        assert_invariants(state, "After setup_dividends_phase_py")

        # Verify the formula
        assert get_max_dividend(26) == 25

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # All 26 dividend actions (0-25) should be valid
        for amount in range(26):
            assert mask[layout['dividend_base'] + amount] == 1.0, f"Dividend {amount} should be valid"

        # Verify action 25 is the last dividend action (26 total: 0-25)
        # There is no dividend action 26 - it would be in the ISSUE phase space
        assert layout['dividend_base'] + 25 < layout['issue_start']


# =============================================================================
# Star Calculation Tests
# =============================================================================


class TestStarCalculation:
    """Owned stars = company_stars + cash/10 + SI_bonus (eagerly maintained)."""

    def test_company_stars_counted(self, dividend_state):
        """Stars from owned companies are counted via recalculate_stars."""
        state = dividend_state
        corp = CORPS[0]  # JS, owns company 3 (AKE, 1★)

        corp.set_cash(state, 0)  # No cash bonus
        corp.recalculate_stars(state)

        assert corp.get_stars(state) == 1  # AKE = 1★

    def test_cash_bonus_stars(self, dividend_state):
        """1 star per $10 cash."""
        state = dividend_state
        corp = CORPS[0]  # Owns AKE (1★)

        corp.set_cash(state, 45)  # 45 // 10 = 4 bonus stars
        corp.recalculate_stars(state)

        assert corp.get_stars(state) == 5  # 1 company + 4 cash

    def test_combined_stars(self, dividend_state):
        """Company stars + cash bonus combined."""
        state = dividend_state
        corp = CORPS[0]  # Owns AKE (1★)

        corp.set_cash(state, 33)  # 3 bonus stars
        corp.recalculate_stars(state)

        assert corp.get_stars(state) == 4  # 1 company + 3 cash

    def test_si_bonus_stars(self):
        """Stars, Inc. (SI) gets +2 additional stars."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float SI (corp index 7) — draws company 3 (AKE, 1★)
        cid = float_corp_for_test(state, corp_id=CorpIndices.CORP_SI)
        si = CORPS[CorpIndices.CORP_SI]
        company_stars = get_company_stars(cid)

        si.set_cash(state, 20)  # 2 bonus stars
        si.recalculate_stars(state)

        # company_stars + 2 cash + 2 SI bonus
        assert si.get_stars(state) == company_stars + 2 + 2

    def test_non_si_no_bonus(self, dividend_state):
        """Non-SI corps don't get the +2 bonus."""
        state = dividend_state
        corp = CORPS[0]  # JS (not SI), owns AKE (1★)

        corp.set_cash(state, 20)  # 2 cash bonus
        corp.recalculate_stars(state)

        assert corp.get_stars(state) == 3  # 1 company + 2 cash, no SI bonus


# =============================================================================
# Price Movement Tests
# =============================================================================


class TestPriceMovement:
    """Price moves based on owned vs required stars."""

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

    def test_occupied_between_current_and_target_up(self):
        """Occupied spaces between current and target don't affect result."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Current index 10, moving up 2 → target = 12
        # Space 11 is occupied (between), 12 is free
        MARKET.set_space_available(state, 11, False)
        MARKET.set_space_available(state, 12, True)

        # Move +2 from 10: target=12 (free) → lands on 12
        target = find_target_index_py(state, 10, 2)
        assert target == 12

    def test_slide_past_occupied_target_up(self):
        """When target space itself is occupied, slide further up."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Current index 10, moving up 2 → target = 12
        # Space 12 is occupied, 13 is free
        MARKET.set_space_available(state, 12, False)
        MARKET.set_space_available(state, 13, True)

        # Move +2 from 10: target=12 (occupied) → slide to 13
        target = find_target_index_py(state, 10, 2)
        assert target == 13

    def test_occupied_between_current_and_target_down(self):
        """Occupied spaces between current and target don't affect result."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Current index 10, moving down 2 → target = 8
        # Space 9 is occupied (between), 8 is free
        MARKET.set_space_available(state, 9, False)
        MARKET.set_space_available(state, 8, True)

        # Move -2 from 10: target=8 (free) → lands on 8
        target = find_target_index_py(state, 10, -2)
        assert target == 8

    def test_slide_past_occupied_target_down(self):
        """When target space itself is occupied, slide further down."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Current index 10, moving down 1 → target = 9
        # Space 9 is occupied, 8 is free
        MARKET.set_space_available(state, 9, False)
        MARKET.set_space_available(state, 8, True)

        # Move -1 from 10: target=9 (occupied) → slide to 8
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
# Processing Order Tests
# =============================================================================


class TestProcessingOrder:
    """Corps processed in descending share price order."""

    def test_highest_price_first(self, multi_corp_dividend_state):
        """Corp with highest price processes first."""
        state = multi_corp_dividend_state

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        assert_invariants(state, "After setup_dividends_phase_py")

        # Corp 1 has highest price (index 15), should be first
        current_corp = TURN.get_dividend_corp(state)
        assert current_corp == 1

    def test_processing_order_sequence(self, multi_corp_dividend_state):
        """Corps process in order: highest -> middle -> lowest price."""
        state = multi_corp_dividend_state

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        assert_invariants(state, "After setup_dividends_phase_py")

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
# Receivership Auto-Handling Tests
# =============================================================================


class TestReceivershipHandling:
    """Receivership corps pay $0 and auto-process."""

    def test_receivership_auto_processed(self):
        """Corp in receivership is auto-processed without player input."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp 0 at highest price, then put in receivership
        # After float(float_shares=2): player0=2, bank=2, issued=4, unissued=3
        COMPANIES[0].transfer_to_player(state, 0)
        corp = CORPS[0]
        corp.float_corp(state, 0, 0, 15, 2)
        corp.set_in_receivership(state, True)
        PLAYERS[0].set_shares(state, 0, 0)  # delta=-2, bank: 2+2=4
        corp.set_cash(state, 100)

        # Float corp 1 at lower price, player-controlled
        # After float(float_shares=1): player0=1, bank=1, issued=2, unissued=5
        # Want: player0=3, bank=0, issued=3, unissued=4
        COMPANIES[1].transfer_to_player(state, 0)
        corp1 = CORPS[1]
        corp1.float_corp(state, 0, 1, 10, 1)
        corp1.set_issued_shares(state, 3)
        corp1.set_unissued_shares(state, 4)
        corp1.set_bank_shares(state, 2)  # set_shares below will subtract 2
        PLAYERS[0].set_shares(state, 1, 3)  # delta=+2, bank: 2-2=0
        corp1.set_cash(state, 50)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        assert_invariants(state, "After setup_dividends_phase_py")

        # Should skip receivership corp and go to player-controlled corp
        current_corp = TURN.get_dividend_corp(state)
        assert current_corp == 1  # Corp 0 was auto-processed

    def test_receivership_pays_zero_dividend(self):
        """Receivership corp pays $0 dividend (cash unchanged)."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp 0, then put in receivership
        # After float(float_shares=2): player0=2, bank=2, issued=4, unissued=3
        COMPANIES[0].transfer_to_player(state, 0)
        corp = CORPS[0]
        corp.float_corp(state, 0, 0, 10, 2)
        corp.set_in_receivership(state, True)
        PLAYERS[0].set_shares(state, 0, 0)  # delta=-2, bank: 2+2=4
        corp.set_cash(state, 100)  # Stars = 1★ + 100//10 = 11, plenty to rise
        MARKET.set_space_available(state, 11, True)
        MARKET.set_space_available(state, 12, True)

        initial_cash = corp.get_cash(state)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        assert_invariants(state, "After setup_dividends_phase_py")

        # Cash should be unchanged (no dividend paid)
        assert corp.get_cash(state) == initial_cash

    def test_receivership_price_adjusts(self):
        """Receivership corp still adjusts price based on stars."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp 0, then put in receivership
        # After float(float_shares=1): player0=1, bank=1, issued=2, unissued=5
        # Want: player0=0, bank=3, issued=3, unissued=4
        COMPANIES[0].transfer_to_player(state, 0)
        corp = CORPS[0]
        corp.float_corp(state, 0, 0, 10, 1)
        corp.set_in_receivership(state, True)
        corp.set_issued_shares(state, 3)
        corp.set_unissued_shares(state, 4)
        corp.set_bank_shares(state, 2)  # set_shares below will add 1
        PLAYERS[0].set_shares(state, 0, 0)  # delta=-1, bank: 2+1=3
        corp.set_cash(state, 100)  # Stars = 1★ + 100//10 = 11, way more than required

        # Make spaces available (except current position)
        for i in range(27):
            MARKET.set_space_available(state, i, True)
        MARKET.set_space_available(state, 10, False)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        assert_invariants(state, "After setup_dividends_phase_py")

        # Price should have risen
        new_price_index = corp.get_price_index(state)
        assert new_price_index > 10


# =============================================================================
# Phase Transition Tests
# =============================================================================


class TestPhaseTransitions:
    """Phase transitions after all corps processed."""

    def test_transition_to_end_card(self, dividend_state):
        """After all corps processed, transitions to END_CARD."""
        state = dividend_state

        # Only one corp, pay dividend and phase should transition
        apply_dividend_action_py(state, 0)
        assert_invariants(state, "After dividend action")

        assert state.get_phase() == GamePhases.PHASE_END_CARD

    def test_transition_to_end_card_when_flipped(self, dividend_state):
        """Even if end card flipped, DIVIDENDS transitions to END_CARD (which handles GAME_OVER)."""
        state = dividend_state

        # Set end card as flipped
        TURN.set_end_card_flipped(state, True)

        apply_dividend_action_py(state, 0)
        assert_invariants(state, "After dividend action")

        # DIVIDENDS always transitions to END_CARD; END_CARD handles the GAME_OVER logic
        assert state.get_phase() == GamePhases.PHASE_END_CARD

    def test_no_active_corps_immediate_transition(self):
        """If no active corps, immediately transitions to END_CARD."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        # All corps start inactive after initialize_game()

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        assert_invariants(state, "After setup_dividends_phase_py")

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
        assert_invariants(state, "After setup_dividends_phase_py")

        # Process all three corps
        assert TURN.get_dividend_corp(state) == 1  # Highest price first
        apply_dividend_action_py(state, 2)
        assert_invariants(state, "After dividend action for corp 1")

        assert TURN.get_dividend_corp(state) == 0
        apply_dividend_action_py(state, 3)
        assert_invariants(state, "After dividend action for corp 0")

        assert TURN.get_dividend_corp(state) == 2
        apply_dividend_action_py(state, 1)
        assert_invariants(state, "After dividend action for corp 2")

        # Should transition to END_CARD
        assert state.get_phase() == GamePhases.PHASE_END_CARD

    def test_dividend_affects_price_adjustment(self, dividend_state):
        """Dividend reduces corp cash which affects cash bonus stars."""
        state = dividend_state
        corp = CORPS[0]

        # Corp owns AKE (1★), price_index=10 ($14), issued=5
        # Required = round(5 * 14 / 10) = 7
        # Set cash=65: stars = 1 + 65//10 = 7, exactly meets requirement
        corp.set_cash(state, 65)

        # Make spaces available
        for i in range(27):
            MARKET.set_space_available(state, i, True)
        MARKET.set_space_available(state, 10, False)

        initial_price = corp.get_price_index(state)

        # Re-setup to recalculate
        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        assert_invariants(state, "After setup_dividends_phase_py")

        # Pay $2/share × 5 issued = $10 total
        # After: cash = 55, bonus = 5, total = 1+5 = 6, required = 7, diff = -1 -> down 1
        apply_dividend_action_py(state, 2)
        assert_invariants(state, "After dividend action")

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


class TestActiveCorpDividends:
    """Test active corp block during DIVIDENDS phase."""

    @pytest.fixture
    def div_state(self):
        """State with a floated corp ready for dividends."""
        gs = GameState(3)
        gs.initialize_game(seed=42)
        float_corp_for_test(gs, corp_id=0, par_index=10, player_id=0)
        TURN.set_phase(gs, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(gs)
        return gs

    def test_active_corp_set_during_dividends(self, div_state):
        """Active corp info should be populated when dividend corp is set."""
        layout = get_layout(3)
        corp_id = TURN.get_dividend_corp(div_state)
        assert corp_id >= 0, "Expected a dividend corp to be set"

        # Check one-hot
        oh_base = layout.active_corp_offset
        assert div_state._array[oh_base + corp_id] == 1.0

        # Check info matches corp data
        corp_offset = layout.corps_offset + corp_id * layout.corp_stride
        assert abs(div_state._array[layout.active_corp_income_offset] - div_state._array[corp_offset + 5]) < 1e-6  # income
        assert abs(div_state._array[layout.active_corp_stars_offset] - div_state._array[corp_offset + 6]) < 1e-6  # stars
        assert abs(div_state._array[layout.active_corp_share_price_offset] - div_state._array[corp_offset + 7]) < 1e-6  # share_price

    def test_active_corp_cleared_after_dividends(self, div_state):
        """Active corp info should be cleared when transitioning out of DIVIDENDS."""
        layout = get_layout(3)
        # Apply a dividend of 0 to advance through dividends
        corp_id = TURN.get_dividend_corp(div_state)
        apply_dividend_action_py(div_state, 0)

        # After processing all corps, should transition out and clear
        phase = TURN.get_phase(div_state)
        if phase != GamePhases.PHASE_DIVIDENDS:
            # Transitioned out - active corp should be cleared
            for offset_name in ('active_corp_income_offset', 'active_corp_stars_offset',
                                'active_corp_share_price_offset'):
                assert div_state._array[getattr(layout, offset_name)] == 0.0, (
                    f"{offset_name} not cleared after transition"
                )

    def test_active_corp_updates_between_corps(self, div_state):
        """Active corp should update when advancing to next dividend corp."""
        layout = get_layout(3)
        # Float a second corp
        float_corp_for_test(div_state, corp_id=1, par_index=15, player_id=1)
        # Re-setup dividends to pick up both corps
        setup_dividends_phase_py(div_state)

        first_corp = TURN.get_dividend_corp(div_state)
        assert first_corp >= 0

        # Apply dividend to advance to next corp
        apply_dividend_action_py(div_state, 0)

        second_corp = TURN.get_dividend_corp(div_state)
        if second_corp >= 0 and second_corp != first_corp:
            oh_base = layout.active_corp_offset
            assert div_state._array[oh_base + second_corp] == 1.0
            assert div_state._array[oh_base + first_corp] == 0.0


# =============================================================================
# Dividend Impact Tests
# =============================================================================


class TestDividendImpact:
    """Dividend impact precomputation for NN visible state."""

    def test_impacts_populated_during_setup(self, dividend_state):
        """Impacts are set for valid dividend levels after setup."""
        state = dividend_state

        # dividend_state: corp 0, price_index=10 ($14), issued=5, cash=100
        # card_max = 14//3 = 4, afford_max = 100//5 = 20, max_div = 4
        # Levels 0-4 should have non-zero-or-zero impacts (but be populated)
        # Levels 5-25 should be exactly 0 (invalid)
        for level in range(5, 26):
            assert TURN.get_dividend_impact(state, level) == 0, (
                f"Invalid level {level} should have zero impact"
            )

    def test_impact_values_correct_varied(self):
        """Different dividend levels produce correct distinct price impacts."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp 0 at price_index=10 ($14)
        float_corp_for_test(state, corp_id=0, par_index=10, float_shares=2)
        corp = CORPS[0]

        # After float: unissued=3, issued=4, bank=2, player0=2
        # Want: issued=5, unissued=2, bank=1, player0=2, player1=2
        # set_shares auto-adjusts bank by -delta, so pre-set bank=3
        corp.set_issued_shares(state, 5)
        corp.set_unissued_shares(state, 2)
        corp.set_bank_shares(state, 3)  # set_shares below subtracts 2
        PLAYERS[1].set_shares(state, 0, 2)  # delta=+2, bank: 3-2=1
        corp.set_cash(state, 72)

        # Make market spaces available for movement
        for i in range(27):
            MARKET.set_space_available(state, i, True)
        MARKET.set_space_available(state, 10, False)  # Corp 0's current space

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        assert_invariants(state, "After setup")

        # Verify dividend corp is set
        assert TURN.get_dividend_corp(state) == 0

        # card_max = 14//3 = 4, afford_max = 72//5 = 14, max_div = 4
        # Level 0: cash=72, cash_stars=7, total=8, diff=+1, move=+1 → target=11, impact=+1
        # Level 1: cash=67, cash_stars=6, total=7, diff=0, move=0 → target=10, impact=0
        # Level 2: cash=62, cash_stars=6, total=7, diff=0, move=0 → target=10, impact=0
        # Level 3: cash=57, cash_stars=5, total=6, diff=-1, move=-1 → target=9, impact=-1
        # Level 4: cash=52, cash_stars=5, total=6, diff=-1, move=-1 → target=9, impact=-1
        assert TURN.get_dividend_impact(state, 0) == 1
        assert TURN.get_dividend_impact(state, 1) == 0
        assert TURN.get_dividend_impact(state, 2) == 0
        assert TURN.get_dividend_impact(state, 3) == -1
        assert TURN.get_dividend_impact(state, 4) == -1

        # Invalid levels are zero
        for level in range(5, 26):
            assert TURN.get_dividend_impact(state, level) == 0

    def test_impact_with_occupied_spaces(self):
        """Impact accounts for sliding past occupied market spaces."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        float_corp_for_test(state, corp_id=0, par_index=10, float_shares=2)
        corp = CORPS[0]

        # After float: unissued=3, issued=4, bank=2, player0=2
        # Want: issued=5, unissued=2, bank=1, player0=2, player1=2
        corp.set_issued_shares(state, 5)
        corp.set_unissued_shares(state, 2)
        corp.set_bank_shares(state, 3)  # set_shares below subtracts 2
        PLAYERS[1].set_shares(state, 0, 2)  # delta=+2, bank: 3-2=1
        corp.set_cash(state, 72)

        # Block space 11 (the natural +1 target) so it slides to 12
        for i in range(27):
            MARKET.set_space_available(state, i, True)
        MARKET.set_space_available(state, 10, False)  # Corp 0
        MARKET.set_space_available(state, 11, False)  # Blocked

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        assert_invariants(state, "After setup")

        # Level 0: move=+1, target=11 occupied → slides to 12, impact=+2
        assert TURN.get_dividend_impact(state, 0) == 2

    def test_impacts_cleared_on_phase_transition(self, dividend_state):
        """Impacts are cleared when transitioning out of DIVIDENDS."""
        state = dividend_state

        # Verify impacts are populated
        assert TURN.get_dividend_corp(state) == 0

        # Apply dividend to process the only corp → transitions out
        apply_dividend_action_py(state, 0)
        assert_invariants(state, "After dividend action")
        assert state.get_phase() == GamePhases.PHASE_END_CARD

        # All impacts should be cleared
        for level in range(26):
            assert TURN.get_dividend_impact(state, level) == 0, (
                f"Impact at level {level} not cleared after phase transition"
            )

    def test_impacts_update_between_corps(self, multi_corp_dividend_state):
        """Impacts are recomputed when advancing to next corp."""
        state = multi_corp_dividend_state

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        assert_invariants(state, "After setup")

        # First corp (corp 1, highest price)
        first_corp = TURN.get_dividend_corp(state)
        assert first_corp == 1
        first_impact_0 = TURN.get_dividend_impact(state, 0)

        # Apply dividend to advance to next corp
        apply_dividend_action_py(state, 0)
        assert_invariants(state, "After first dividend")

        # Second corp (corp 0, middle price)
        second_corp = TURN.get_dividend_corp(state)
        assert second_corp == 0

        # Impacts should be freshly computed for the new corp
        # The new corp has different cash/stars/price, so impacts differ
        # Just verify they're populated (at least level 0 exists since $0 is always valid)
        second_impact_0 = TURN.get_dividend_impact(state, 0)
        # Both corps have enough stars to rise, so impact should be positive
        assert second_impact_0 != 0 or first_impact_0 != 0, (
            "At least one corp should have non-zero impact at level 0"
        )

    def test_impact_normalization_roundtrip(self):
        """Values survive set→get roundtrip through normalization."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Test various impact values including negative
        for impact in [-5, -2, -1, 0, 1, 2, 5]:
            TURN.set_dividend_impact(state, 0, impact)
            assert TURN.get_dividend_impact(state, 0) == impact, (
                f"Roundtrip failed for impact={impact}"
            )

    def test_impact_zero_dividend_always_populated(self):
        """Level 0 (pay nothing) always has an impact since it's always valid."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        float_corp_for_test(state, corp_id=0, par_index=10)
        CORPS[0].set_cash(state, 50)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)
        assert_invariants(state, "After setup")

        # Level 0 should always be computed (paying $0 is always valid)
        # The impact reflects price movement from current stars vs required
        assert TURN.get_dividend_corp(state) == 0
        # get_dividend_impact returns the denormalized value; it should be an integer
        impact = TURN.get_dividend_impact(state, 0)
        assert isinstance(impact, int)
