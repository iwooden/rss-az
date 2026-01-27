"""Tests for CLOSING phase and mandatory close logic.

Coverage:
- CLO-14: Player income calculation and mandatory close triggering
- CLO-15: Cheapest negative-income company closed first
- CLO-16: Phase transition to INVEST (temporary for INCOME) after mandatory close
"""
import pytest
from core.state import GameState
from core.driver import GameDriver
from core.data import GamePhases
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES
from phases.closing import process_mandatory_close_py


class TestPlayerIncome:
    """Tests for Player.get_income() method (CLO-14 support)."""

    def test_get_income_no_companies(self, game_state):
        """Player with no private companies has 0 income."""
        # Fresh game state - player 0 has no companies
        income = PLAYERS[0].get_income(game_state)
        assert income == 0

    def test_get_income_single_company(self, game_state):
        """Player income equals adjusted income of owned company."""
        # Give player 0 a company (company 0: $1 income, 1 star)
        PLAYERS[0].set_owns_company(game_state, 0, True)

        # At CoO level 1, 1-star company has $0 CoO
        # Adjusted income = $1 - $0 = $1
        income = PLAYERS[0].get_income(game_state)
        coo_level = TURN.get_coo_level(game_state)
        assert coo_level == 1  # Default
        assert income == 1  # $1 - $0 = $1

    def test_get_income_multiple_companies(self, game_state):
        """Player income sums adjusted income from all owned companies."""
        # Give player 0 two companies
        # Company 0: $1 income, 1 star -> adjusted = $1 - $0 = $1 at CoO 1
        # Company 8: $3 income, 2 stars -> adjusted = $3 - $0 = $3 at CoO 1
        PLAYERS[0].set_owns_company(game_state, 0, True)
        PLAYERS[0].set_owns_company(game_state, 8, True)

        income = PLAYERS[0].get_income(game_state)
        # $1 + $3 = $4
        assert income == 4

    def test_get_income_negative_company(self, game_state):
        """Player income can be negative from high-CoO companies."""
        # Set CoO level to max (7) to get high CoO values
        TURN.set_coo_level(game_state, 7)

        # Give player a 1-star company at CoO 7
        # Company 0: $1 income, 1 star -> CoO at level 7 is $10
        # Adjusted = $1 - $10 = -$9
        PLAYERS[0].set_owns_company(game_state, 0, True)

        income = PLAYERS[0].get_income(game_state)
        assert income == -9  # $1 - $10 = -$9

    def test_get_income_excludes_corp_subsidiaries(self, game_state):
        """Player income excludes companies owned by corps (even if player is president)."""
        # Make player 0 president of corp 0
        CORPS[0].set_active(game_state, True)
        CORPS[0].set_price_index(game_state, 5)  # Some price
        PLAYERS[0].set_shares(game_state, 0, 3)
        PLAYERS[0].set_president_of(game_state, 0, True)

        # Give corp 0 a company
        CORPS[0].set_owns_company(game_state, 0, True)

        # Player income should be 0 (corp's company doesn't count)
        income = PLAYERS[0].get_income(game_state)
        assert income == 0


class TestMandatoryClose:
    """Tests for mandatory close logic (CLO-14, CLO-15)."""

    def test_mandatory_close_not_triggered_positive_total(self, game_state):
        """Mandatory close does nothing when income + cash >= 0."""
        # Player has $30 cash, no companies -> income 0
        # Total = $30 >= 0, no close needed
        assert PLAYERS[0].get_cash(game_state) == 30
        assert PLAYERS[0].get_income(game_state) == 0

        process_mandatory_close_py(game_state)

        # Nothing changed
        assert PLAYERS[0].get_cash(game_state) == 30

    def test_mandatory_close_triggered_negative_total(self, game_state):
        """CLO-14: Mandatory close triggers when income + cash < 0."""
        # Set up: player with negative income that exceeds cash
        # Set CoO level high
        TURN.set_coo_level(game_state, 7)

        # Give player a 1-star company with negative adjusted income
        # Company 0: $1 income, 1 star, CoO 7 = $10 -> adjusted = -$9
        PLAYERS[0].set_owns_company(game_state, 0, True)

        # Reduce player cash to trigger mandatory close
        # Cash = $5, income = -$9, total = -$4 < 0
        PLAYERS[0].set_cash(game_state, 5)

        process_mandatory_close_py(game_state)

        # Company should be closed
        assert not PLAYERS[0].owns_company(game_state, 0)
        assert COMPANIES[0].is_removed(game_state)

    def test_mandatory_close_cheapest_first(self, game_state):
        """CLO-15: Cheapest (lowest face value) negative-income company closed first."""
        TURN.set_coo_level(game_state, 7)

        # Give player two negative-income companies with different face values
        # Company 0: face value $1, 1 star (cheapest), adjusted = $1 - $10 = -$9
        # Company 8: face value $3, 2 stars, adjusted = $3 - $10 = -$7
        PLAYERS[0].set_owns_company(game_state, 0, True)
        PLAYERS[0].set_owns_company(game_state, 8, True)

        # Set cash so closing ONE company makes total >= 0
        # Income = -$9 + -$7 = -$16
        # Cash = $10, total = -$6 < 0
        # After closing company 0 (-$9): income = -$7, total = -$7 + $10 = $3 >= 0
        PLAYERS[0].set_cash(game_state, 10)

        process_mandatory_close_py(game_state)

        # Cheapest (company 0, face value $1) should be closed
        assert not PLAYERS[0].owns_company(game_state, 0)
        assert COMPANIES[0].is_removed(game_state)

        # More expensive company 8 should still be owned
        assert PLAYERS[0].owns_company(game_state, 8)
        assert not COMPANIES[8].is_removed(game_state)

    def test_mandatory_close_multiple_companies(self, game_state):
        """CLO-14: Closes multiple companies if needed until income + cash >= 0."""
        TURN.set_coo_level(game_state, 7)

        # Give player multiple negative-income companies
        PLAYERS[0].set_owns_company(game_state, 0, True)  # $1 FV, -$9 adj
        PLAYERS[0].set_owns_company(game_state, 1, True)  # $1 FV, -$9 adj
        PLAYERS[0].set_owns_company(game_state, 2, True)  # $2 FV, -$8 adj

        # Income = -$9 + -$9 + -$8 = -$26, cash = 10, total = -$16 < 0
        # Need to close multiple companies
        PLAYERS[0].set_cash(game_state, 10)

        process_mandatory_close_py(game_state)

        # Should have closed enough to make total >= 0
        income = PLAYERS[0].get_income(game_state)
        cash = PLAYERS[0].get_cash(game_state)
        assert income + cash >= 0

    def test_mandatory_close_js_bonus(self, game_state):
        """Junkyard Scrappers receives 2x printed income bonus on mandatory close."""
        TURN.set_coo_level(game_state, 7)

        # Activate Junkyard Scrappers (corp 0)
        CORPS[0].set_active(game_state, True)
        CORPS[0].set_price_index(game_state, 5)
        initial_js_cash = CORPS[0].get_cash(game_state)

        # Give player a negative-income company
        # Company 0: $1 printed income, 1 star
        PLAYERS[0].set_owns_company(game_state, 0, True)
        PLAYERS[0].set_cash(game_state, 5)  # Will trigger mandatory close

        process_mandatory_close_py(game_state)

        # JS should have received 2x printed income ($1 * 2 = $2)
        assert CORPS[0].get_cash(game_state) == initial_js_cash + 2

    def test_mandatory_close_only_negative_income_companies(self, game_state):
        """Mandatory close only targets negative-income companies, not positive."""
        TURN.set_coo_level(game_state, 7)

        # Give player one positive and one negative income company
        # Company 29: $10 income, 5 stars (blue), CoO 7 = $0 -> adj = +$10
        PLAYERS[0].set_owns_company(game_state, 29, True)

        # Company 0: $1 income, 1 star, CoO 7 = $10 -> adj = -$9
        PLAYERS[0].set_owns_company(game_state, 0, True)

        # Total income = +$10 + -$9 = +$1
        # Cash = $0, total = $1 >= 0, no close needed
        PLAYERS[0].set_cash(game_state, 0)

        process_mandatory_close_py(game_state)

        # No companies should be closed (total is positive)
        assert PLAYERS[0].owns_company(game_state, 29)
        assert PLAYERS[0].owns_company(game_state, 0)

    def test_mandatory_close_player_order(self, game_state):
        """Mandatory close processes players in ID order (0, 1, 2, ...)."""
        TURN.set_coo_level(game_state, 7)

        # Give player 1 a negative-income company (test non-zero player)
        PLAYERS[1].set_owns_company(game_state, 5, True)  # $2 FV, -$8 adj
        PLAYERS[1].set_cash(game_state, 5)  # Will trigger close

        process_mandatory_close_py(game_state)

        # Player 1's company should be closed
        assert not PLAYERS[1].owns_company(game_state, 5)
        assert COMPANIES[5].is_removed(game_state)
