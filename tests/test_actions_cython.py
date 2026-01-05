"""Tests for Cython action vector module."""

import pytest
import numpy as np
from cython_core.state import GameState
from cython_core.actions import (
    get_total_action_count,
    get_action_layout,
    decode_action_py,
    get_valid_action_mask,
    get_constants,
)

from tests.test_common import (
    StateBuilder, PHASE_INVEST, PHASE_BID_IN_AUCTION, PHASE_ACQUISITION,
    PHASE_CLOSING, PHASE_DIVIDENDS, PHASE_ISSUE_SHARES, PHASE_IPO,
    NUM_CORPS, NUM_COMPANIES
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def state():
    """Create a basic 3-player game state."""
    s = GameState(3)
    s.phase = PHASE_INVEST
    s.coo_level = 1
    s.active_player = 0
    return s


@pytest.fixture
def builder(state):
    """Create a StateBuilder for test setup."""
    return StateBuilder(state)


@pytest.fixture
def layout():
    """Get action layout for 3 players."""
    return get_action_layout(3)


# =============================================================================
# CONSTANT AND LAYOUT TESTS
# =============================================================================

class TestConstants:
    """Test action constants."""

    def test_total_action_count_3_players(self):
        # 186 + (3 * 20) = 246
        assert get_total_action_count(3) == 246

    def test_total_action_count_4_players(self):
        # 186 + (4 * 20) = 266
        assert get_total_action_count(4) == 266

    def test_total_action_count_5_players(self):
        # 186 + (5 * 20) = 286
        assert get_total_action_count(5) == 286

    def test_total_action_count_6_players(self):
        # 186 + (6 * 20) = 306
        assert get_total_action_count(6) == 306

    def test_constants(self):
        consts = get_constants()
        assert consts['AUCTION_CAP'] == 20
        assert consts['MAX_PAR_SLOTS'] == 8
        assert consts['ACQ_PRICE_RANGE'] == 51
        assert consts['MAX_DIVIDEND'] == 26


class TestActionLayout:
    """Test action layout structure for 3 players."""

    def test_layout_total_size(self, layout):
        # 3 players: 186 + 60 = 246
        assert layout['total_size'] == 246

    def test_invest_phase_boundaries(self, layout):
        # INVEST: 1 + (3*20) + 8 + 8 = 77
        assert layout['invest_start'] == 0
        assert layout['pass_invest'] == 0
        assert layout['auction_base'] == 1
        assert layout['buy_share_base'] == 61  # 1 + 3*20
        assert layout['sell_share_base'] == 69  # 61 + 8
        assert layout['bid_start'] == 77  # 69 + 8

    def test_bid_phase_boundaries(self, layout):
        assert layout['bid_start'] == 77
        assert layout['leave_auction'] == 77
        assert layout['raise_bid_base'] == 78
        assert layout['acquisition_start'] == 97  # 77 + 20

    def test_acquisition_phase_boundaries(self, layout):
        assert layout['acquisition_start'] == 97
        assert layout['acq_price_base'] == 97
        assert layout['acq_fi_high'] == 148  # 97 + 51
        assert layout['acq_fi_face'] == 149
        assert layout['acq_pass'] == 150
        assert layout['closing_start'] == 151

    def test_closing_phase_boundaries(self, layout):
        assert layout['closing_start'] == 151
        assert layout['close_action'] == 151
        assert layout['close_pass'] == 152
        assert layout['dividends_start'] == 153

    def test_dividends_phase_boundaries(self, layout):
        assert layout['dividends_start'] == 153
        assert layout['dividend_base'] == 153
        assert layout['issue_start'] == 179  # 153 + 26

    def test_issue_phase_boundaries(self, layout):
        assert layout['issue_start'] == 179
        assert layout['issue_pass'] == 179
        assert layout['issue_action'] == 180
        assert layout['ipo_start'] == 181

    def test_ipo_phase_boundaries(self, layout):
        assert layout['ipo_start'] == 181
        assert layout['ipo_pass'] == 181
        assert layout['ipo_base'] == 182
        # Total: 182 + 64 = 246


class TestLayoutDifferentPlayerCounts:
    """Test that layouts differ based on player count."""

    def test_3_vs_6_player_layout(self):
        layout3 = get_action_layout(3)
        layout6 = get_action_layout(6)

        # Total size differs
        assert layout3['total_size'] == 246
        assert layout6['total_size'] == 306

        # INVEST phase boundaries differ
        assert layout3['buy_share_base'] == 61  # 1 + 60
        assert layout6['buy_share_base'] == 121  # 1 + 120

        # Later phases shift accordingly
        assert layout3['bid_start'] == 77
        assert layout6['bid_start'] == 137


# =============================================================================
# ACTION DECODING TESTS
# =============================================================================

class TestActionDecoding:
    """Test decode_action_py function for 3 players."""

    def test_decode_pass_invest(self, layout):
        phase, action_type, slot, corp_id, amount = decode_action_py(0, 3)
        assert phase == PHASE_INVEST
        assert action_type == 0  # ACTION_PASS
        assert slot == -1
        assert corp_id == -1

    def test_decode_auction_slot0_bid0(self, layout):
        phase, action_type, slot, corp_id, amount = decode_action_py(1, 3)
        assert phase == PHASE_INVEST
        assert action_type == 1  # ACTION_AUCTION
        assert slot == 0
        assert amount == 0

    def test_decode_auction_slot1_bid5(self, layout):
        # slot1, bid5 = 1 + 1*20 + 5 = 26
        phase, action_type, slot, corp_id, amount = decode_action_py(26, 3)
        assert phase == PHASE_INVEST
        assert action_type == 1  # ACTION_AUCTION
        assert slot == 1
        assert amount == 5

    def test_decode_buy_share_corp0(self, layout):
        # buy_share_base = 61 for 3 players
        phase, action_type, slot, corp_id, amount = decode_action_py(61, 3)
        assert phase == PHASE_INVEST
        assert action_type == 2  # ACTION_BUY_SHARE
        assert corp_id == 0

    def test_decode_buy_share_corp7(self, layout):
        phase, action_type, slot, corp_id, amount = decode_action_py(68, 3)
        assert phase == PHASE_INVEST
        assert action_type == 2  # ACTION_BUY_SHARE
        assert corp_id == 7

    def test_decode_sell_share_corp0(self, layout):
        # sell_share_base = 69 for 3 players
        phase, action_type, slot, corp_id, amount = decode_action_py(69, 3)
        assert phase == PHASE_INVEST
        assert action_type == 3  # ACTION_SELL_SHARE
        assert corp_id == 0

    def test_decode_leave_auction(self, layout):
        # bid_start = 77 for 3 players
        phase, action_type, slot, corp_id, amount = decode_action_py(77, 3)
        assert phase == PHASE_BID_IN_AUCTION
        assert action_type == 4  # ACTION_LEAVE_AUCTION

    def test_decode_raise_bid(self, layout):
        # Raise bid offset 5 = 78 + 5 = 83
        phase, action_type, slot, corp_id, amount = decode_action_py(83, 3)
        assert phase == PHASE_BID_IN_AUCTION
        assert action_type == 5  # ACTION_RAISE_BID
        assert amount == 5

    def test_decode_acq_price(self, layout):
        # acq_price_base = 97, offset 25 = 97 + 25 = 122
        phase, action_type, slot, corp_id, amount = decode_action_py(122, 3)
        assert phase == PHASE_ACQUISITION
        assert action_type == 6  # ACTION_ACQ_PRICE
        assert amount == 25

    def test_decode_acq_fi_high(self, layout):
        phase, action_type, slot, corp_id, amount = decode_action_py(148, 3)
        assert phase == PHASE_ACQUISITION
        assert action_type == 7  # ACTION_ACQ_FI_HIGH

    def test_decode_acq_fi_face(self, layout):
        phase, action_type, slot, corp_id, amount = decode_action_py(149, 3)
        assert phase == PHASE_ACQUISITION
        assert action_type == 8  # ACTION_ACQ_FI_FACE

    def test_decode_acq_pass(self, layout):
        phase, action_type, slot, corp_id, amount = decode_action_py(150, 3)
        assert phase == PHASE_ACQUISITION
        assert action_type == 0  # ACTION_PASS

    def test_decode_close_action(self, layout):
        phase, action_type, slot, corp_id, amount = decode_action_py(151, 3)
        assert phase == PHASE_CLOSING
        assert action_type == 9  # ACTION_CLOSE

    def test_decode_close_pass(self, layout):
        phase, action_type, slot, corp_id, amount = decode_action_py(152, 3)
        assert phase == PHASE_CLOSING
        assert action_type == 0  # ACTION_PASS

    def test_decode_dividend(self, layout):
        # Dividend amount 10 = 153 + 10 = 163
        phase, action_type, slot, corp_id, amount = decode_action_py(163, 3)
        assert phase == PHASE_DIVIDENDS
        assert action_type == 10  # ACTION_DIVIDEND
        assert amount == 10

    def test_decode_issue_pass(self, layout):
        phase, action_type, slot, corp_id, amount = decode_action_py(179, 3)
        assert phase == PHASE_ISSUE_SHARES
        assert action_type == 0  # ACTION_PASS

    def test_decode_issue_action(self, layout):
        phase, action_type, slot, corp_id, amount = decode_action_py(180, 3)
        assert phase == PHASE_ISSUE_SHARES
        assert action_type == 11  # ACTION_ISSUE

    def test_decode_ipo_pass(self, layout):
        phase, action_type, slot, corp_id, amount = decode_action_py(181, 3)
        assert phase == PHASE_IPO
        assert action_type == 0  # ACTION_PASS

    def test_decode_ipo_corp0_slot0(self, layout):
        phase, action_type, slot, corp_id, amount = decode_action_py(182, 3)
        assert phase == PHASE_IPO
        assert action_type == 12  # ACTION_IPO
        assert corp_id == 0
        assert slot == 0

    def test_decode_ipo_corp7_slot7(self, layout):
        # corp7, slot7 = 182 + 7*8 + 7 = 182 + 63 = 245
        phase, action_type, slot, corp_id, amount = decode_action_py(245, 3)
        assert phase == PHASE_IPO
        assert action_type == 12  # ACTION_IPO
        assert corp_id == 7
        assert slot == 7

    def test_decode_invalid_negative(self, layout):
        phase, action_type, slot, corp_id, amount = decode_action_py(-1, 3)
        assert phase == -1

    def test_decode_invalid_too_large(self, layout):
        phase, action_type, slot, corp_id, amount = decode_action_py(400, 3)
        assert phase == -1


# =============================================================================
# MASK GENERATION TESTS
# =============================================================================

class TestMaskGeneration:
    """Test get_valid_action_mask function."""

    def test_mask_shape(self, state):
        mask = get_valid_action_mask(state)
        # 3 players: 246 actions
        assert mask.shape == (246,)
        assert mask.dtype == np.float32

    def test_invest_phase_pass_always_valid(self, state):
        mask = get_valid_action_mask(state)
        assert mask[0] == 1.0  # pass_invest

    def test_invest_phase_auction_with_cash(self, state, builder):
        builder.set_player_cash(0, 100)
        builder.set_company_for_auction(0, True)  # Company 0 face value is $1

        mask = get_valid_action_mask(state)

        # Slot 0, bid offset 0 should be valid
        assert mask[1] == 1.0  # auction_base + 0*20 + 0

    def test_invest_phase_auction_slot_mapping(self, state, builder):
        builder.set_player_cash(0, 100)
        # Set companies 5 and 10 available (not 0)
        builder.set_company_for_auction(5, True)
        builder.set_company_for_auction(10, True)

        mask = get_valid_action_mask(state)

        # Slot 0 should map to company 5, slot 1 to company 10
        # Slot 0, bid 0 = index 1
        assert mask[1] == 1.0

    def test_invest_phase_buy_share(self, state, builder):
        builder.set_player_cash(0, 100)
        builder.set_corp_active(0, True)
        builder.set_corp_bank_shares(0, 5)
        builder.set_corp_price_index(0, 5)
        builder.set_market_available(6, True)

        mask = get_valid_action_mask(state)

        # Buy share corp 0 = index 61 for 3 players
        assert mask[61] == 1.0

    def test_invest_phase_sell_share(self, state, builder):
        builder.set_player_shares(0, 3, 2)  # Player 0 has 2 shares of corp 3
        builder.set_corp_active(3, True)
        builder.set_corp_price_index(3, 10)
        builder.set_market_available(9, True)

        mask = get_valid_action_mask(state)

        # Sell share corp 3 = index 69 + 3 = 72 for 3 players
        assert mask[72] == 1.0

    def test_wrong_phase_returns_empty_mask(self, state):
        state.phase = PHASE_BID_IN_AUCTION

        mask = get_valid_action_mask(state)

        # Should have leave_auction as valid (index 77 for 3 players)
        assert mask[77] == 1.0


class TestMaskInvestPhase:
    """Detailed tests for INVEST phase mask generation."""

    def test_no_cash_no_auctions(self, state, builder):
        builder.set_player_cash(0, 0)
        builder.set_company_for_auction(0, True)

        mask = get_valid_action_mask(state)

        # Pass is valid, but no auction actions
        assert mask[0] == 1.0
        assert mask[1] == 0.0  # Can't afford

    def test_multiple_auction_companies(self, state, builder):
        builder.set_player_cash(0, 50)
        # Set multiple companies for auction (max 3 for 3 players)
        for cid in [0, 1, 2]:
            builder.set_company_for_auction(cid, True)

        mask = get_valid_action_mask(state)

        # Should have actions for slots 0, 1, 2
        assert mask[1] == 1.0   # Slot 0
        assert mask[21] == 1.0  # Slot 1 (1 + 1*20)
        assert mask[41] == 1.0  # Slot 2 (1 + 2*20)


# =============================================================================
# MASK TESTS FOR OTHER PHASES
# =============================================================================

class TestMaskBidPhase:
    """Test BID_IN_AUCTION phase mask generation."""

    def test_leave_auction_always_valid(self, state, builder):
        state.phase = PHASE_BID_IN_AUCTION
        # Note: Would need to set up auction state properly
        # For now just test the phase check

        mask = get_valid_action_mask(state)
        # leave_auction = 77 for 3 players
        assert mask[77] == 1.0


class TestMaskClosingPhase:
    """Test CLOSING phase mask generation."""

    def test_closing_no_offer_empty_mask(self, state):
        state.phase = PHASE_CLOSING

        mask = get_valid_action_mask(state)

        # No closing company set, so no valid actions
        valid_count = np.sum(mask > 0)
        assert valid_count == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for action module."""

    def test_round_trip_decode_matches_layout(self, layout):
        """Verify decode_action_py is consistent with layout."""
        # Test pass invest
        phase, action_type, _, _, _ = decode_action_py(layout['pass_invest'], 3)
        assert phase == PHASE_INVEST
        assert action_type == 0

        # Test close action
        phase, action_type, _, _, _ = decode_action_py(layout['close_action'], 3)
        assert phase == PHASE_CLOSING
        assert action_type == 9

        # Test IPO pass
        phase, action_type, _, _, _ = decode_action_py(layout['ipo_pass'], 3)
        assert phase == PHASE_IPO
        assert action_type == 0

    def test_all_indices_decode_to_valid_phase(self, layout):
        """Verify all indices decode to a valid phase for 3 players."""
        total_actions = get_total_action_count(3)
        for idx in range(total_actions):
            phase, action_type, slot, corp_id, amount = decode_action_py(idx, 3)
            assert phase in [
                PHASE_INVEST, PHASE_BID_IN_AUCTION, PHASE_ACQUISITION,
                PHASE_CLOSING, PHASE_DIVIDENDS, PHASE_ISSUE_SHARES, PHASE_IPO
            ], f"Index {idx} decoded to invalid phase {phase}"

    def test_all_player_counts_have_valid_layout(self):
        """Test layouts for all supported player counts."""
        for num_players in [2, 3, 4, 5, 6]:
            layout = get_action_layout(num_players)
            total = get_total_action_count(num_players)
            assert layout['total_size'] == total
            assert total == 186 + (num_players * 20)

            # Verify all indices decode properly
            for idx in range(total):
                phase, action_type, slot, corp_id, amount = decode_action_py(idx, num_players)
                assert phase >= 0, f"Invalid phase for {num_players} players, index {idx}"
