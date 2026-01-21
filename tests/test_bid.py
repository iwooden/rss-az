"""Tests for BID_IN_AUCTION phase actions."""
import pytest
import numpy as np
from core.state import GameState
from core.driver import DRIVER
from core.actions import get_valid_action_mask, get_action_layout
from core.data import GamePhases
from entities.turn import TURN
from entities.player import PLAYERS
from entities.company import COMPANIES
from entities.deck import DECK

STATUS_OK = 0


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def auction_state():
    """Create game state with active auction (in BID phase)."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)
    assert state.get_phase() == GamePhases.PHASE_INVEST

    # Find and apply first valid auction action to enter BID phase
    mask = get_valid_action_mask(state)
    layout = get_action_layout(3)
    for i in range(layout['auction_base'], layout['buy_share_base']):
        if mask[i] == 1.0:
            result = DRIVER.apply_action(state, i)
            assert result == STATUS_OK
            break

    # Verify we're in BID phase with active auction
    assert state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION
    assert TURN.get_auction_company(state) >= 0
    return state


# =============================================================================
# LEAVE AUCTION TESTS
# =============================================================================

class TestLeaveAuction:
    """Test BID phase leave auction action behavior."""

    def test_leave_sets_passed_flag(self, auction_state):
        """BID-01: Leave auction sets passed flag for player."""
        player_id = auction_state.get_active_player()

        # Verify not passed initially
        assert not TURN.has_player_passed_auction(auction_state, player_id)

        # Apply leave auction
        layout = get_action_layout(3)
        result = DRIVER.apply_action(auction_state, layout['leave_auction'])
        assert result == STATUS_OK

        # Verify passed flag set
        assert TURN.has_player_passed_auction(auction_state, player_id)

    def test_leave_advances_to_next_bidder(self, auction_state):
        """BID-02: Leave auction advances to next non-passed bidder."""
        initial_player = auction_state.get_active_player()
        initial_position = PLAYERS[initial_player].get_turn_order(auction_state)

        # Apply leave auction
        layout = get_action_layout(3)
        result = DRIVER.apply_action(auction_state, layout['leave_auction'])
        assert result == STATUS_OK

        # Verify active player advanced (if auction didn't resolve)
        if auction_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            new_player = auction_state.get_active_player()
            new_position = PLAYERS[new_player].get_turn_order(auction_state)
            # Should have advanced to next position
            assert new_position == (initial_position + 1) % 3

    def test_leave_skips_passed_players(self, auction_state):
        """BID-02: Rotation skips players who have already left."""
        # Get current player and mark next player as passed
        current_player = auction_state.get_active_player()
        current_position = PLAYERS[current_player].get_turn_order(auction_state)
        next_position = (current_position + 1) % 3

        # Find player at next position and mark them as passed
        for player_id in range(3):
            if PLAYERS[player_id].get_turn_order(auction_state) == next_position:
                TURN.set_player_passed_auction(auction_state, player_id, True)
                break

        # Current player leaves
        layout = get_action_layout(3)
        result = DRIVER.apply_action(auction_state, layout['leave_auction'])
        assert result == STATUS_OK

        # Verify active player skipped the passed player (if auction didn't resolve)
        if auction_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            new_player = auction_state.get_active_player()
            new_position = PLAYERS[new_player].get_turn_order(auction_state)
            # Should have skipped to position after next
            assert new_position == (current_position + 2) % 3

    def test_last_leaver_triggers_resolution(self, auction_state):
        """BID-05: Auction resolves when only one bidder remains."""
        # Make all but one player leave
        layout = get_action_layout(3)

        # First player leaves
        DRIVER.apply_action(auction_state, layout['leave_auction'])
        assert auction_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION

        # Second player leaves - should trigger resolution
        DRIVER.apply_action(auction_state, layout['leave_auction'])

        # Verify auction resolved and returned to INVEST phase
        assert auction_state.get_phase() == GamePhases.PHASE_INVEST


# =============================================================================
# RAISE BID TESTS
# =============================================================================

class TestRaiseBid:
    """Test BID phase raise bid action behavior."""

    def test_raise_updates_price(self, auction_state):
        """BID-03: Raise bid updates auction price."""
        initial_price = TURN.get_auction_price(auction_state)

        # Find valid raise bid action
        mask = get_valid_action_mask(auction_state)
        layout = get_action_layout(3)
        raise_idx = None
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                raise_idx = i
                break

        if raise_idx is not None:
            result = DRIVER.apply_action(auction_state, raise_idx)
            assert result == STATUS_OK

            # Verify price increased
            new_price = TURN.get_auction_price(auction_state)
            assert new_price > initial_price

    def test_raise_updates_high_bidder(self, auction_state):
        """BID-03: Raise bid updates high bidder."""
        current_player = auction_state.get_active_player()

        # Find valid raise bid action
        mask = get_valid_action_mask(auction_state)
        layout = get_action_layout(3)
        raise_idx = None
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                raise_idx = i
                break

        if raise_idx is not None:
            result = DRIVER.apply_action(auction_state, raise_idx)
            assert result == STATUS_OK

            # Verify high bidder updated to current player
            high_bidder = TURN.get_auction_high_bidder(auction_state)
            assert high_bidder == current_player

    def test_raise_advances_to_next_bidder(self, auction_state):
        """Raise bid advances to next non-passed bidder."""
        initial_player = auction_state.get_active_player()
        initial_position = PLAYERS[initial_player].get_turn_order(auction_state)

        # Find valid raise bid action
        mask = get_valid_action_mask(auction_state)
        layout = get_action_layout(3)
        raise_idx = None
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                raise_idx = i
                break

        if raise_idx is not None:
            result = DRIVER.apply_action(auction_state, raise_idx)
            assert result == STATUS_OK

            # Verify active player advanced
            new_player = auction_state.get_active_player()
            new_position = PLAYERS[new_player].get_turn_order(auction_state)
            assert new_position == (initial_position + 1) % 3

    def test_raise_bid_values_correct(self, auction_state):
        """Verify raise bid calculates price correctly (face + amount + 1)."""
        from core.data import get_company_face_value

        company_id = TURN.get_auction_company(auction_state)
        face_value = get_company_face_value(company_id)

        # Find first valid raise bid action
        mask = get_valid_action_mask(auction_state)
        layout = get_action_layout(3)

        # The first raise bid action should be face + 0 + 1 = face + 1
        raise_idx = layout['raise_bid_base']
        if mask[raise_idx] == 1.0:
            result = DRIVER.apply_action(auction_state, raise_idx)
            assert result == STATUS_OK

            # Verify price is face + 1 (since amount=0 for first raise bid slot)
            new_price = TURN.get_auction_price(auction_state)
            assert new_price == face_value + 1


# =============================================================================
# AUCTION RESOLUTION TESTS
# =============================================================================

class TestAuctionResolution:
    """Test auction resolution behavior."""

    def test_winner_pays_bid_price(self, auction_state):
        """BID-06: Winner pays bid price to bank."""
        # Record winner and price before resolution
        winner_id = TURN.get_auction_high_bidder(auction_state)
        bid_price = TURN.get_auction_price(auction_state)
        initial_cash = PLAYERS[winner_id].get_cash(auction_state)

        # Make all others leave to trigger resolution
        layout = get_action_layout(3)
        for _ in range(2):  # Leave twice (2 of 3 players)
            if auction_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                DRIVER.apply_action(auction_state, layout['leave_auction'])

        # Verify winner paid
        final_cash = PLAYERS[winner_id].get_cash(auction_state)
        assert final_cash == initial_cash - bid_price

    def test_winner_receives_company(self, auction_state):
        """BID-07: Winner receives company ownership."""
        winner_id = TURN.get_auction_high_bidder(auction_state)
        company_id = TURN.get_auction_company(auction_state)

        # Verify company not owned by player initially (owner_id < 0)
        initial_owner = COMPANIES[company_id].get_owner_id(auction_state)
        assert initial_owner < 0  # Not owned by player

        # Make all others leave to trigger resolution
        layout = get_action_layout(3)
        for _ in range(2):
            if auction_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                DRIVER.apply_action(auction_state, layout['leave_auction'])

        # Verify winner owns company
        final_owner = COMPANIES[company_id].get_owner_id(auction_state)
        assert final_owner == winner_id

    def test_auction_state_cleared(self, auction_state):
        """BID-08: Auction resolution clears all auction state."""
        # Make all others leave to trigger resolution
        layout = get_action_layout(3)
        for _ in range(2):
            if auction_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                DRIVER.apply_action(auction_state, layout['leave_auction'])

        # Verify auction state cleared
        assert TURN.get_auction_company(auction_state) == -1
        assert TURN.get_auction_high_bidder(auction_state) == -1
        assert TURN.get_auction_starter(auction_state) == -1

        # Verify all passed flags cleared
        for player_id in range(3):
            assert not TURN.has_player_passed_auction(auction_state, player_id)

    def test_new_company_drawn(self, auction_state):
        """BID-09: New company drawn after auction resolution."""
        # At start of game, num_players companies are in auction row
        # When auction starts, one company is still marked "for auction" but is being bid on
        # After resolution: that company is transferred (cleared from auction)
        # and a new company is drawn (added to auction)
        # Net effect: auction row size stays constant

        # Record the company being auctioned
        auctioned_company = TURN.get_auction_company(auction_state)

        # Verify company is marked for auction initially
        assert auction_state.is_company_for_auction(auctioned_company)

        # Make all others leave to trigger resolution
        layout = get_action_layout(3)
        for _ in range(2):
            if auction_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                DRIVER.apply_action(auction_state, layout['leave_auction'])

        # Verify the auctioned company is no longer for auction (transferred to player)
        assert not auction_state.is_company_for_auction(auctioned_company)

        # Verify a new company was drawn (total count should be num_players companies)
        auction_count = sum(
            1 for cid in range(36)
            if auction_state.is_company_for_auction(cid)
        )
        assert auction_count == 3  # Should still have 3 companies in auction row

    def test_returns_to_invest_phase(self, auction_state):
        """BID-10: Auction resolution returns to INVEST phase."""
        # Make all others leave to trigger resolution
        layout = get_action_layout(3)
        for _ in range(2):
            if auction_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                DRIVER.apply_action(auction_state, layout['leave_auction'])

        # Verify phase transition
        assert auction_state.get_phase() == GamePhases.PHASE_INVEST

    def test_turn_goes_to_player_after_starter(self, auction_state):
        """BID-11: Next turn goes to player after auction starter."""
        starter_id = TURN.get_auction_starter(auction_state)
        starter_position = PLAYERS[starter_id].get_turn_order(auction_state)
        expected_next_position = (starter_position + 1) % 3

        # Make all others leave to trigger resolution
        layout = get_action_layout(3)
        for _ in range(2):
            if auction_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                DRIVER.apply_action(auction_state, layout['leave_auction'])

        # Verify active player is after starter
        active_player = auction_state.get_active_player()
        active_position = PLAYERS[active_player].get_turn_order(auction_state)
        assert active_position == expected_next_position

    def test_winner_net_worth_updated(self, auction_state):
        """BID-12: Winner's net worth updated after receiving company."""
        from core.data import get_company_face_value

        winner_id = TURN.get_auction_high_bidder(auction_state)
        company_id = TURN.get_auction_company(auction_state)
        bid_price = TURN.get_auction_price(auction_state)
        initial_net_worth = PLAYERS[winner_id].get_net_worth(auction_state)
        face_value = get_company_face_value(company_id)

        # Make all others leave to trigger resolution
        layout = get_action_layout(3)
        for _ in range(2):
            if auction_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                DRIVER.apply_action(auction_state, layout['leave_auction'])

        # Verify net worth updated: lost cash, gained company
        # Net worth change = -bid_price + face_value
        final_net_worth = PLAYERS[winner_id].get_net_worth(auction_state)
        expected_change = face_value - bid_price
        assert final_net_worth == initial_net_worth + expected_change


# =============================================================================
# FULL AUCTION CYCLE TESTS
# =============================================================================

class TestFullAuctionCycle:
    """Test complete auction workflows."""

    def test_complete_auction_cycle(self):
        """Test full flow: INVEST -> auction -> BID -> resolution -> INVEST."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Start in INVEST
        assert state.get_phase() == GamePhases.PHASE_INVEST

        # Start auction
        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                DRIVER.apply_action(state, i)
                break

        # Now in BID phase
        assert state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION

        # Two players leave
        for _ in range(2):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                DRIVER.apply_action(state, layout['leave_auction'])

        # Back to INVEST
        assert state.get_phase() == GamePhases.PHASE_INVEST

    def test_auction_with_immediate_resolution(self):
        """Test auction where all others leave immediately."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        starter_cash = PLAYERS[state.get_active_player()].get_cash(state)

        # Start auction
        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                DRIVER.apply_action(state, i)
                break

        winner_id = TURN.get_auction_high_bidder(state)

        # All others leave immediately
        for _ in range(2):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                DRIVER.apply_action(state, layout['leave_auction'])

        # Verify auction completed
        assert state.get_phase() == GamePhases.PHASE_INVEST
        # Verify winner paid
        assert PLAYERS[winner_id].get_cash(state) < starter_cash

    def test_auction_with_multiple_raises(self):
        """Test auction with several raises before resolution."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Start auction
        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                DRIVER.apply_action(state, i)
                break

        initial_price = TURN.get_auction_price(state)

        # First player raises
        mask = get_valid_action_mask(state)
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                DRIVER.apply_action(state, i)
                break

        # Verify price increased
        assert TURN.get_auction_price(state) > initial_price

        # Second player raises
        mask = get_valid_action_mask(state)
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                DRIVER.apply_action(state, i)
                break

        # Others leave to resolve
        for _ in range(2):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                DRIVER.apply_action(state, layout['leave_auction'])

        # Verify back to INVEST
        assert state.get_phase() == GamePhases.PHASE_INVEST


# =============================================================================
# MULTIPLE PLAYER COUNT TESTS
# =============================================================================

class TestMultiplePlayerCounts:
    """Test BID phase behavior across different player counts."""

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_auction_works_all_player_counts(self, num_players):
        """Auction flow works correctly for all player counts."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Start auction
        mask = get_valid_action_mask(state)
        layout = get_action_layout(num_players)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                DRIVER.apply_action(state, i)
                break

        assert state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION

        # All but one leave
        for _ in range(num_players - 1):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                DRIVER.apply_action(state, layout['leave_auction'])

        # Verify resolved
        assert state.get_phase() == GamePhases.PHASE_INVEST

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_bidder_rotation_correct_all_counts(self, num_players):
        """Bidder rotation works correctly for all player counts."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Start auction
        mask = get_valid_action_mask(state)
        layout = get_action_layout(num_players)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                DRIVER.apply_action(state, i)
                break

        # Track active players through leaves
        active_players = []
        for _ in range(num_players - 1):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                active_players.append(state.get_active_player())
                DRIVER.apply_action(state, layout['leave_auction'])

        # Verify we cycled through different players
        assert len(set(active_players)) >= 2  # At least 2 different players
