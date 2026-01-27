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
