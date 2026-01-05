"""Tests for Dividends phase."""

import pytest
from state import GameState
from phases.dividends import DividendsPhase, get_phase_handler
from data import py_get_company_stars, py_get_market_price

from tests.test_common import (
    StateBuilder, PHASE_INCOME, PHASE_DIVIDENDS, PHASE_END_CARD, MARKET_PRICES,
    CORP_JS, CORP_S, CORP_OS, CORP_SM, CORP_PR, CORP_DA, CORP_VM, CORP_SI
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def state():
    """Create a basic 3-player game state in DIVIDENDS phase."""
    s = GameState(3)
    s.phase = PHASE_DIVIDENDS
    s.coo_level = 1
    s.active_player = 0
    return s


@pytest.fixture
def handler():
    """Get dividends phase handler for 3 players."""
    return DividendsPhase(3)


@pytest.fixture
def builder(state):
    """Create a StateBuilder for test setup."""
    b = StateBuilder(state)
    b.init_all_market_available()
    return b


# =============================================================================
# HANDLER TESTS
# =============================================================================

class TestDividendsPhaseHandler:
    """Test handler creation and caching."""

    def test_get_phase_handler_creates_handler(self):
        """get_phase_handler creates a DividendsPhase instance."""
        handler = get_phase_handler(3)
        assert isinstance(handler, DividendsPhase)

    def test_get_phase_handler_caches(self):
        """get_phase_handler returns cached instance."""
        h1 = get_phase_handler(3)
        h2 = get_phase_handler(3)
        assert h1 is h2

    def test_different_player_counts_different_handlers(self):
        """Different player counts get different handlers."""
        h3 = get_phase_handler(3)
        h4 = get_phase_handler(4)
        assert h3 is not h4


# =============================================================================
# DIVIDEND VALIDATION TESTS
# =============================================================================

class TestDividendValidation:
    """Test dividend amount validation."""

    def test_can_pay_zero_dividend(self, state, handler, builder):
        """Zero dividend is always valid when corp is current."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 50)
        builder.set_corp_issued_shares(CORP_JS, 2)
        builder.set_corp_price_index(CORP_JS, 12)  # Price 18, max div = 6
        builder.set_player_president(0, CORP_JS, True)

        handler.setup_dividends(state)

        assert handler.can_do_dividend(state, 0)

    def test_cannot_exceed_max_dividend(self, state, handler, builder):
        """Cannot pay more than share_price // 3."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 100)
        builder.set_corp_issued_shares(CORP_JS, 2)
        builder.set_corp_price_index(CORP_JS, 12)  # Price 18, max div = 6
        builder.set_player_president(0, CORP_JS, True)

        handler.setup_dividends(state)

        assert handler.get_max_dividend(state) == 6
        assert handler.can_do_dividend(state, 6)
        assert not handler.can_do_dividend(state, 7)

    def test_cannot_pay_more_than_cash(self, state, handler, builder):
        """Cannot pay out more than corp has in cash."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 10)
        builder.set_corp_issued_shares(CORP_JS, 4)  # 4 shares * 3 = 12 needed
        builder.set_corp_price_index(CORP_JS, 12)  # Price 18, max div = 6
        builder.set_player_president(0, CORP_JS, True)

        handler.setup_dividends(state)

        # Can afford 2 per share (2 * 4 = 8)
        assert handler.can_do_dividend(state, 2)
        # Cannot afford 3 per share (3 * 4 = 12 > 10)
        assert not handler.can_do_dividend(state, 3)

    def test_get_valid_actions_returns_affordable_dividends(self, state, handler, builder):
        """get_valid_actions returns all affordable dividend amounts."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 15)
        builder.set_corp_issued_shares(CORP_JS, 3)
        builder.set_corp_price_index(CORP_JS, 12)  # Price 18, max div = 6
        builder.set_player_president(0, CORP_JS, True)

        handler.setup_dividends(state)

        valid = handler.get_valid_actions(state)
        # Can afford 0-5 (5*3=15), not 6 (6*3=18>15)
        assert valid == [0, 1, 2, 3, 4, 5]


# =============================================================================
# DIVIDEND PAYMENT TESTS
# =============================================================================

class TestDividendPayment:
    """Test dividend payment execution."""

    def test_dividend_deducts_from_corp_cash(self, state, handler, builder):
        """Paying dividend deducts from corp cash."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 50)
        builder.set_corp_issued_shares(CORP_JS, 4)
        builder.set_corp_price_index(CORP_JS, 12)
        builder.set_player_president(0, CORP_JS, True)
        builder.set_player_shares(0, CORP_JS, 2)

        handler.setup_dividends(state)
        handler.do_dividend(state, 3)  # 3 * 4 = 12

        assert builder.get_corp_cash(CORP_JS) == 38

    def test_dividend_pays_players_by_shares(self, state, handler, builder):
        """Players receive dividend based on shares owned."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 50)
        builder.set_corp_issued_shares(CORP_JS, 4)
        builder.set_corp_price_index(CORP_JS, 12)
        builder.set_player_president(0, CORP_JS, True)
        builder.set_player_shares(0, CORP_JS, 2)
        builder.set_player_shares(1, CORP_JS, 1)
        builder.set_player_cash(0, 10)
        builder.set_player_cash(1, 20)

        handler.setup_dividends(state)
        handler.do_dividend(state, 3)

        # Player 0 gets 3 * 2 = 6
        assert builder.get_player_cash(0) == 16
        # Player 1 gets 3 * 1 = 3
        assert builder.get_player_cash(1) == 23

    def test_zero_dividend_doesnt_change_cash(self, state, handler, builder):
        """Zero dividend doesn't change any cash."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 50)
        builder.set_corp_issued_shares(CORP_JS, 4)
        builder.set_corp_price_index(CORP_JS, 12)
        builder.set_player_president(0, CORP_JS, True)
        builder.set_player_shares(0, CORP_JS, 2)
        builder.set_player_cash(0, 10)

        handler.setup_dividends(state)
        handler.do_dividend(state, 0)

        assert builder.get_corp_cash(CORP_JS) == 50
        assert builder.get_player_cash(0) == 10


# =============================================================================
# RECEIVERSHIP TESTS
# =============================================================================

class TestReceivership:
    """Test receivership corps auto-pay 0."""

    def test_receivership_auto_pays_zero(self, state, handler, builder):
        """Corps in receivership automatically pay 0 dividend."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 100)
        builder.set_corp_issued_shares(CORP_JS, 2)
        builder.set_corp_price_index(CORP_JS, 12)
        builder.set_corp_in_receivership(CORP_JS, True)
        # No president set (in receivership)

        handler.setup_dividends(state)

        # Should auto-advance past this corp
        # Corp cash should remain unchanged
        assert builder.get_corp_cash(CORP_JS) == 100


# =============================================================================
# STAR CALCULATION TESTS
# =============================================================================

class TestStarCalculations:
    """Test star calculations for share price adjustment."""

    def test_corp_stars_from_companies(self, state, handler, builder):
        """Corp stars come from owned companies."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 0)
        builder.set_corp_owns_company(CORP_JS, 0, True)  # BME (1 star)
        builder.set_corp_owns_company(CORP_JS, 1, True)  # BSE (1 star)

        stars = handler.calculate_corp_stars(state, CORP_JS)
        assert stars == 2

    def test_corp_stars_include_cash_bonus(self, state, handler, builder):
        """Corp gets +1 star per $10 cash."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 35)  # +3 stars
        builder.set_corp_owns_company(CORP_JS, 0, True)  # BME (1 star)

        stars = handler.calculate_corp_stars(state, CORP_JS)
        assert stars == 4  # 1 company + 3 cash

    def test_si_gets_bonus_stars(self, state, handler, builder):
        """Stars Inc. gets +2 stars bonus."""
        builder.set_corp_active(CORP_SI, True)
        builder.set_corp_cash(CORP_SI, 0)
        builder.set_corp_owns_company(CORP_SI, 0, True)  # BME (1 star)

        stars = handler.calculate_corp_stars(state, CORP_SI)
        assert stars == 3  # 1 company + 2 SI bonus

    def test_target_stars_calculation(self, state, handler, builder):
        """Target stars = round(issued_shares * share_price / 10)."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_issued_shares(CORP_JS, 3)
        builder.set_corp_price_index(CORP_JS, 12)  # Price 18

        # Target = round(3 * 18 / 10) = round(5.4) = 5
        target = handler.calculate_target_stars(state, CORP_JS)
        assert target == 5


# =============================================================================
# SHARE PRICE ADJUSTMENT TESTS
# =============================================================================

class TestSharePriceAdjustment:
    """Test share price adjustment after dividend."""

    def test_no_change_when_stars_match(self, state, handler, builder):
        """Share price doesn't change when stars equal target."""
        builder.set_corp_active(CORP_JS, True)
        # Set cash so that after $0 dividend, corp_stars = target_stars
        # Target = round(2 * 10 / 10) = 2
        # Need exactly 2 stars: 2 from companies, 0 from cash
        builder.set_corp_cash(CORP_JS, 0)
        builder.set_corp_issued_shares(CORP_JS, 2)
        builder.set_corp_price_index(CORP_JS, 6)  # Price 10
        builder.set_player_president(0, CORP_JS, True)
        builder.set_player_shares(0, CORP_JS, 2)
        # Give 2 stars from companies to match target
        builder.set_corp_owns_company(CORP_JS, 0, True)  # 1 star
        builder.set_corp_owns_company(CORP_JS, 1, True)  # 1 star

        handler.setup_dividends(state)
        old_index = builder.get_corp_price_index(CORP_JS)
        handler.do_dividend(state, 0)

        assert builder.get_corp_price_index(CORP_JS) == old_index

    def test_price_increases_when_more_stars(self, state, handler, builder):
        """Share price increases when corp has more stars than target."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 50)
        builder.set_corp_issued_shares(CORP_JS, 2)
        builder.set_corp_price_index(CORP_JS, 6)  # Price 10
        builder.set_player_president(0, CORP_JS, True)
        builder.set_player_shares(0, CORP_JS, 2)
        # Target = round(2 * 10 / 10) = 2
        # Give 4 stars (2 more than target)
        builder.set_corp_owns_company(CORP_JS, 7, True)  # Orange: 2 stars
        builder.set_corp_owns_company(CORP_JS, 8, True)  # Orange: 2 stars

        handler.setup_dividends(state)
        old_index = builder.get_corp_price_index(CORP_JS)
        handler.do_dividend(state, 0)

        # Should move up 2 spaces
        new_index = builder.get_corp_price_index(CORP_JS)
        assert new_index == old_index + 2

    def test_price_decreases_when_fewer_stars(self, state, handler, builder):
        """Share price decreases when corp has fewer stars than target."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 0)  # No cash stars
        builder.set_corp_issued_shares(CORP_JS, 3)
        builder.set_corp_price_index(CORP_JS, 12)  # Price 18
        builder.set_player_president(0, CORP_JS, True)
        builder.set_player_shares(0, CORP_JS, 2)
        # Target = round(3 * 18 / 10) = 5
        # Give only 2 stars (3 fewer than target)
        builder.set_corp_owns_company(CORP_JS, 0, True)  # 1 star
        builder.set_corp_owns_company(CORP_JS, 1, True)  # 1 star

        # Verify our setup is correct before calling handler
        assert handler.calculate_corp_stars(state, CORP_JS) == 2
        assert handler.calculate_target_stars(state, CORP_JS) == 5

        handler.setup_dividends(state)
        handler.do_dividend(state, 0)

        # Should move down 2 spaces: from index 12 to index 10
        new_index = builder.get_corp_price_index(CORP_JS)
        assert new_index == 10  # 12 - 2 = 10

    def test_skips_unavailable_market_spaces(self, state, handler, builder):
        """Price adjustment skips spaces taken by other corps."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 50)
        builder.set_corp_issued_shares(CORP_JS, 2)
        builder.set_corp_price_index(CORP_JS, 6)  # Price 10
        builder.set_player_president(0, CORP_JS, True)
        builder.set_player_shares(0, CORP_JS, 2)
        # Target = 2, give 4 stars (+2 movement)
        builder.set_corp_owns_company(CORP_JS, 7, True)
        builder.set_corp_owns_company(CORP_JS, 8, True)
        # Mark adjacent spaces as taken
        builder.set_market_available(7, False)
        builder.set_market_available(8, False)

        handler.setup_dividends(state)
        handler.do_dividend(state, 0)

        # Should skip to index 9 (since 7,8 are taken)
        new_index = builder.get_corp_price_index(CORP_JS)
        assert new_index == 9


# =============================================================================
# CORP ORDERING TESTS
# =============================================================================

class TestCorpOrdering:
    """Test corps are processed in share price order."""

    def test_highest_price_first(self, state, handler, builder):
        """Corps are processed in descending share price order."""
        # Set up two corps at different prices
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 50)
        builder.set_corp_issued_shares(CORP_JS, 2)
        builder.set_corp_price_index(CORP_JS, 6)  # Price 10
        builder.set_player_president(0, CORP_JS, True)
        builder.set_corp_owns_company(CORP_JS, 0, True)

        builder.set_corp_active(CORP_S, True)
        builder.set_corp_cash(CORP_S, 50)
        builder.set_corp_issued_shares(CORP_S, 2)
        builder.set_corp_price_index(CORP_S, 12)  # Price 18 (higher)
        builder.set_player_president(1, CORP_S, True)
        builder.set_corp_owns_company(CORP_S, 1, True)

        handler.setup_dividends(state)

        # First corp should be S (higher price)
        assert handler.get_current_corp(state) == CORP_S


# =============================================================================
# PHASE TRANSITION TESTS
# =============================================================================

class TestPhaseTransition:
    """Test transition to END_CARD phase."""

    def test_transitions_after_all_corps(self, state, handler, builder):
        """Phase transitions to END_CARD after all corps pay."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 50)
        builder.set_corp_issued_shares(CORP_JS, 2)
        builder.set_corp_price_index(CORP_JS, 6)
        builder.set_player_president(0, CORP_JS, True)
        builder.set_player_shares(0, CORP_JS, 2)
        builder.set_corp_owns_company(CORP_JS, 0, True)
        builder.set_corp_owns_company(CORP_JS, 1, True)

        handler.setup_dividends(state)
        handler.do_dividend(state, 0)

        assert state.phase == PHASE_END_CARD

    def test_no_active_corps_transitions_immediately(self, state, handler, builder):
        """With no active corps, transitions to END_CARD immediately."""
        # No corps active
        handler.setup_dividends(state)

        assert state.phase == PHASE_END_CARD


# =============================================================================
# BANKRUPTCY TESTS
# =============================================================================

class TestBankruptcy:
    """Test bankruptcy when share price drops to 0."""

    def test_bankruptcy_on_price_zero(self, state, handler, builder):
        """Corp goes bankrupt if share price drops to 0."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 0)
        builder.set_corp_issued_shares(CORP_JS, 5)
        builder.set_corp_price_index(CORP_JS, 1)  # Price 5 (near bottom)
        builder.set_player_president(0, CORP_JS, True)
        builder.set_player_shares(0, CORP_JS, 3)
        # Target = round(5 * 5 / 10) = 3, give 0 stars
        # No companies = 0 stars, diff = -3 -> move -2

        handler.setup_dividends(state)
        handler.do_dividend(state, 0)

        # Corp should be bankrupt (inactive)
        assert not builder.is_corp_active(CORP_JS)
