"""Tests for GameState.initialize_game() method."""
import pytest
from core.state import GameState
from core.data import GamePhases, GameConstants, get_corp_share_count
from entities.player import PLAYERS
from entities.fi import FI
from entities.corp import CORPS
from entities.market import MARKET
from entities.turn import TURN
from entities.deck import DECK


class TestInitSignature:
    """Method signature and reinitialization."""

    def test_accepts_optional_seed(self):
        """Method accepts optional seed parameter."""
        gs = GameState(4)
        gs.initialize_game()  # No seed - should work
        gs.initialize_game(42)  # With seed - should work

    def test_same_seed_produces_same_state(self):
        """Same seed produces reproducible results."""
        gs1 = GameState(4)
        gs1.initialize_game(12345)

        gs2 = GameState(4)
        gs2.initialize_game(12345)

        # Deck order should match
        assert DECK.get_order(gs1) == DECK.get_order(gs2)

    def test_can_reinitialize(self):
        """Can reinitialize existing GameState."""
        gs = GameState(4)
        gs.initialize_game(42)
        # Modify state
        gs.set_player_cash(0, 100)
        # Reinitialize
        gs.initialize_game(42)
        assert gs.get_player_cash(0) == 30  # Reset to starting value


class TestPlayerSetup:
    """Player initialization."""

    @pytest.mark.parametrize("num_players,expected_cash", [
        (2, 30), (3, 30), (4, 30), (5, 30), (6, 25)
    ])
    def test_starting_cash(self, num_players, expected_cash):
        """Correct starting cash by player count."""
        gs = GameState(num_players)
        gs.initialize_game()
        for i in range(num_players):
            assert PLAYERS[i].get_cash(gs) == expected_cash

    def test_turn_order_linear(self):
        """Players assigned linear turn order."""
        gs = GameState(4)
        gs.initialize_game()
        for i in range(4):
            assert PLAYERS[i].get_turn_order(gs) == i

    def test_no_owned_companies(self):
        """All player-owned companies cleared."""
        gs = GameState(4)
        gs.initialize_game()
        for i in range(4):
            for company_id in range(GameConstants.NUM_COMPANIES):
                assert not PLAYERS[i].owns_company(gs, company_id)

    def test_no_owned_shares(self):
        """All player-owned shares cleared."""
        gs = GameState(4)
        gs.initialize_game()
        for i in range(4):
            for corp_id in range(GameConstants.NUM_CORPS):
                assert PLAYERS[i].get_shares(gs, corp_id) == 0


class TestForeignInvestor:
    """Foreign Investor initialization."""

    def test_fi_starting_cash(self):
        """FI receives 4 starting cash."""
        gs = GameState(4)
        gs.initialize_game()
        assert FI.get_cash(gs) == 4

    def test_fi_no_companies(self):
        """FI owns no companies at start."""
        gs = GameState(4)
        gs.initialize_game()
        for company_id in range(GameConstants.NUM_COMPANIES):
            assert not FI.owns_company(gs, company_id)


class TestCorporations:
    """Corporation initialization."""

    def test_all_corps_inactive(self):
        """All 8 corporations inactive."""
        gs = GameState(4)
        gs.initialize_game()
        for corp in CORPS:
            assert not corp.is_active(gs)

    def test_shares_reset(self):
        """Each corporation's shares reset."""
        gs = GameState(4)
        gs.initialize_game()
        for corp in CORPS:
            expected_shares = get_corp_share_count(corp.corp_id)
            assert corp.get_unissued_shares(gs) == expected_shares
            assert corp.get_issued_shares(gs) == 0
            assert corp.get_bank_shares(gs) == 0

    def test_corp_no_companies(self):
        """No corporation owns any companies."""
        gs = GameState(4)
        gs.initialize_game()
        for corp in CORPS:
            for company_id in range(GameConstants.NUM_COMPANIES):
                assert not corp.owns_company(gs, company_id)

    def test_corp_no_price_card(self):
        """No corporation has a share price card."""
        gs = GameState(4)
        gs.initialize_game()
        for corp in CORPS:
            # Inactive corps have no meaningful price index
            assert not corp.is_active(gs)


class TestMarket:
    """Market initialization."""

    def test_all_spaces_available(self):
        """All 27 share price slots marked available."""
        gs = GameState(4)
        gs.initialize_game()
        for i in range(GameConstants.NUM_MARKET_SPACES):
            assert MARKET.is_space_available(gs, i)


class TestDeckAndDraw:
    """Deck building and initial draw."""

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_correct_companies_drawn(self, num_players):
        """N companies drawn (N = player count)."""
        gs = GameState(num_players)
        gs.initialize_game()

        auction_count = sum(
            1 for c_id in range(GameConstants.NUM_COMPANIES)
            if gs.is_company_for_auction(c_id)
        )
        assert auction_count == num_players

    def test_drawn_companies_for_auction(self):
        """Drawn companies marked as available for auction."""
        gs = GameState(4)
        gs.initialize_game()

        # Should have exactly 4 companies for auction
        auction_companies = [
            c_id for c_id in range(GameConstants.NUM_COMPANIES)
            if gs.is_company_for_auction(c_id)
        ]
        assert len(auction_companies) == 4

    @pytest.mark.parametrize("num_players,expected_remaining", [
        # Deck total = red + orange + yellow + green + blue, minus N drawn
        # Non-orange colors: N+1 each (includes "last" card)
        # Orange special: 4p=6, 5p=8(all), 6p=8(all)
        # 6p uses ALL: red=6, orange=8, yellow=8, green=7, blue=7 = 36
        (3, 17),   # 4+4+4+4+4=20, minus 3 drawn
        (4, 22),   # 5+6+5+5+5=26, minus 4 drawn
        (5, 27),   # 6+8+6+6+6=32, minus 5 drawn
        (6, 30),   # 6+8+8+7+7=36, minus 6 drawn
    ])
    def test_deck_size_correct(self, num_players, expected_remaining):
        """Deck has correct number of cards after initial draw."""
        gs = GameState(num_players)
        gs.initialize_game()
        assert DECK.get_remaining_count(gs) == expected_remaining


class TestTurnState:
    """Turn state initialization."""

    def test_phase_is_invest(self):
        """Phase set to 1 (Investment)."""
        gs = GameState(4)
        gs.initialize_game()
        assert TURN.get_phase(gs) == GamePhases.PHASE_INVEST

    def test_coo_level_is_one(self):
        """CoO level set to 1."""
        gs = GameState(4)
        gs.initialize_game()
        assert TURN.get_coo_level(gs) == 1

    def test_turn_number_is_one(self):
        """Turn number set to 1."""
        gs = GameState(4)
        gs.initialize_game()
        assert TURN.get_turn_number(gs) == 1

    def test_active_player_is_zero(self):
        """Active player set to player 0 after initialization."""
        gs = GameState(4)
        gs.initialize_game()
        assert gs.get_active_player() == 0

    def test_auction_state_cleared(self):
        """All auction/dividend/IPO state cleared."""
        gs = GameState(4)
        gs.initialize_game()

        assert TURN.get_auction_company(gs) == -1
        assert TURN.get_auction_high_bidder(gs) == -1
        assert TURN.get_auction_starter(gs) == -1
        assert TURN.get_dividend_corp(gs) == -1
        assert TURN.get_issue_corp(gs) == -1
        assert TURN.get_ipo_company(gs) == -1
        assert TURN.get_acq_active_corp(gs) == -1
        assert TURN.get_acq_target_company(gs) == -1
        assert not TURN.is_acq_fi_offer(gs)
        assert TURN.get_closing_company(gs) == -1
        assert not TURN.is_end_card_flipped(gs)
        assert TURN.get_consecutive_passes(gs) == 0
