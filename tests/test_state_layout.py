"""Tests for GameState layout sizes.

These tests verify that documented sizes in VECTORS.md and CLAUDE.md match actual
computed sizes. If a test fails, update the documentation to match the actual values.
"""
import pytest
from core.state import GameState
from core.data import GameConstants


# Constants from data.pxd (duplicated here for documentation verification)
NUM_PHASES = 11  # Phases 0-10 (INVEST through GAME_OVER)
NUM_COO_LEVELS = 7
NUM_COMPANIES = 36
NUM_CORPS = 8
NUM_MARKET_SPACES = 27
MAX_DECK_SIZE = 36
MAX_DIVIDEND = 26
OFFER_BUFFER_SIZE = 250
CLOSE_OFFER_BUFFER_SIZE = 100


def compute_player_stride(num_players: int) -> int:
    """Compute player stride for given player count."""
    return (
        1 +                 # cash
        1 +                 # net_worth
        num_players +       # turn_order one-hot
        1 +                 # is_auction_high_bidder
        NUM_COMPANIES +     # owned_companies (36)
        NUM_CORPS +         # owned_shares (8)
        NUM_CORPS +         # is_president (8)
        NUM_CORPS +         # share_buys (8)
        NUM_CORPS +         # share_sells (8)
        1                   # acquisition_proceeds
    )


def compute_corp_stride() -> int:
    """Compute corporation stride (fixed for all player counts)."""
    return (
        1 +                 # active
        1 +                 # cash
        1 +                 # unissued_shares
        1 +                 # issued_shares
        1 +                 # bank_shares
        1 +                 # income
        1 +                 # stars
        1 +                 # share_price
        1 +                 # acquisition_proceeds
        1 +                 # in_receivership
        NUM_MARKET_SPACES + # price_index one-hot (27)
        NUM_COMPANIES +     # owned_companies (36)
        NUM_COMPANIES       # acquisition_companies (36)
    )


def compute_turn_size(num_players: int) -> int:
    """Compute turn state size for given player count."""
    return (
        1 +                 # turn_number
        1 +                 # end_card_flipped
        1 +                 # consecutive_passes
        NUM_COMPANIES +     # auction_company (36)
        1 +                 # auction_price
        num_players +       # auction_high_bidder
        num_players +       # auction_starter
        num_players +       # auction_passed
        NUM_CORPS +         # dividend_corp (8)
        MAX_DIVIDEND +      # dividend_impact (26)
        NUM_CORPS +         # dividend_remaining (8)
        NUM_CORPS +         # issue_corp (8)
        NUM_CORPS +         # issue_remaining (8)
        NUM_COMPANIES +     # ipo_company (36)
        NUM_COMPANIES +     # ipo_remaining (36)
        NUM_CORPS +         # acq_active_corp (8)
        NUM_COMPANIES +     # acq_target_company (36)
        1 +                 # acq_is_fi_offer
        NUM_COMPANIES       # closing_company (36)
    )


def compute_visible_size(num_players: int) -> int:
    """Compute visible state size for given player count."""
    offset = 0

    # Phase one-hot (visible phases only)
    offset += NUM_PHASES  # 11

    # CoO one-hot
    offset += NUM_COO_LEVELS  # 7

    # Players
    player_stride = compute_player_stride(num_players)
    offset += player_stride * num_players

    # Foreign investor
    offset += 1 + NUM_COMPANIES  # cash + owned_companies (37)

    # Company locations
    offset += NUM_COMPANIES * 3  # auction, revealed, removed (108)

    # Company adjusted incomes
    offset += NUM_COMPANIES  # 36

    # Market availability
    offset += NUM_MARKET_SPACES  # 27

    # Corporations
    corp_stride = compute_corp_stride()
    offset += corp_stride * NUM_CORPS

    # Turn state
    offset += compute_turn_size(num_players)

    # Static company data: stars, low, face, high, synergies
    offset += NUM_COMPANIES * (4 + NUM_COMPANIES)  # 36 * 40 = 1440

    return offset


def compute_hidden_size() -> int:
    """Compute hidden state size (fixed for all player counts)."""
    offset = 0

    # Basic hidden fields
    offset += 1  # active_player
    offset += 1  # num_players
    offset += 1  # deck_top
    offset += MAX_DECK_SIZE  # deck_order (36)
    offset += 1  # phase (compact)
    offset += 1  # coo_level (compact)
    offset += 1  # auction_company (compact)
    offset += 1  # auction_high_bidder (compact)
    offset += 1  # auction_starter (compact)
    offset += NUM_CORPS  # corp_price_indices (8)

    # Acquisition offer buffer
    offset += 1  # offer_count
    offset += 1  # offer_index
    offset += OFFER_BUFFER_SIZE * 2  # offer_buffer (250 * 2 = 500)

    # Close offer buffer
    offset += 1  # close_offer_count
    offset += 1  # close_offer_index
    offset += CLOSE_OFFER_BUFFER_SIZE * 3  # close_offer_buffer (100 * 3 = 300)

    # O(1) access fields for one-hot values
    offset += 1  # acq_active_corp (compact)
    offset += 1  # acq_target_company (compact)
    offset += 1  # closing_company (compact)
    offset += 1  # dividend_corp (compact)
    offset += 1  # issue_corp (compact)
    offset += 1  # ipo_company (compact)

    # Company location tracking (O(1) clearing without scanning)
    offset += NUM_COMPANIES  # company_locations (36)
    offset += NUM_COMPANIES  # company_owner_ids (36)

    return offset


class TestStateLayoutSizes:
    """Verify state layout sizes match documentation."""

    # Expected sizes - these MUST match VECTORS.md and CLAUDE.md
    # If these tests fail, update the documentation to match!
    EXPECTED_SIZES = {
        2: {'visible': 2943, 'hidden': 934, 'total': 3877},
        3: {'visible': 3023, 'hidden': 934, 'total': 3957},
        4: {'visible': 3105, 'hidden': 934, 'total': 4039},
        5: {'visible': 3189, 'hidden': 934, 'total': 4123},
        6: {'visible': 3275, 'hidden': 934, 'total': 4209},
    }

    @pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
    def test_total_size_matches_actual(self, num_players):
        """Total computed size matches actual GameState array size."""
        gs = GameState(num_players)
        actual_total = gs._array.shape[0]

        visible = compute_visible_size(num_players)
        hidden = compute_hidden_size()
        computed_total = visible + hidden

        assert computed_total == actual_total, (
            f"{num_players} players: computed {computed_total} != actual {actual_total}"
        )

    @pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
    def test_sizes_match_expected(self, num_players):
        """Computed sizes match expected documented values."""
        expected = self.EXPECTED_SIZES[num_players]

        visible = compute_visible_size(num_players)
        hidden = compute_hidden_size()
        total = visible + hidden

        assert visible == expected['visible'], (
            f"{num_players} players visible: {visible} != expected {expected['visible']}"
        )
        assert hidden == expected['hidden'], (
            f"{num_players} players hidden: {hidden} != expected {expected['hidden']}"
        )
        assert total == expected['total'], (
            f"{num_players} players total: {total} != expected {expected['total']}"
        )


class TestComponentSizes:
    """Verify individual component sizes."""

    def test_player_stride_formula(self):
        """Player stride = 72 + num_players."""
        for num_players in [2, 3, 4, 5, 6]:
            stride = compute_player_stride(num_players)
            assert stride == 72 + num_players, (
                f"{num_players} players: stride {stride} != 72 + {num_players}"
            )

    def test_corp_stride_fixed(self):
        """Corp stride = 109 (fixed for all player counts)."""
        assert compute_corp_stride() == 109

    def test_turn_size_formula(self):
        """Turn size = 251 + 3*num_players."""
        for num_players in [2, 3, 4, 5, 6]:
            size = compute_turn_size(num_players)
            expected = 251 + 3 * num_players
            assert size == expected, (
                f"{num_players} players: turn size {size} != {expected}"
            )

    def test_hidden_size_fixed(self):
        """Hidden size = 934 (fixed for all player counts)."""
        assert compute_hidden_size() == 934

    def test_static_size(self):
        """Static company data = 36 * 40 = 1440."""
        static_size = NUM_COMPANIES * (4 + NUM_COMPANIES)
        assert static_size == 1440
