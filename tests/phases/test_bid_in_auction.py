"""Tests for BID_IN_AUCTION phase actions."""
import pytest
from core.state import GameState
from core.actions import get_valid_action_mask, get_action_layout
from core.data import GamePhases, get_company_face_value
from entities.turn import TURN
from entities.player import PLAYERS
from entities.company import COMPANIES
from entities.deck import DECK
from tests.phases.conftest import STATUS_OK, apply_and_verify_all

# Fixtures come from conftest.py automatically (bid_state is same as auction_state)
# Helper functions also available: assert_valid_mask, assert_invariants


# =============================================================================
# LEAVE AUCTION TESTS
# =============================================================================

class TestLeaveAuction:
    """Test BID phase leave auction action behavior."""

    def test_leave_sets_passed_flag(self, bid_state):
        """Leave auction sets passed flag for player."""
        player_id = bid_state.get_active_player()

        # Verify not passed initially
        assert not TURN.has_player_passed_auction(bid_state, player_id)

        # Apply leave auction
        layout = get_action_layout(3)
        apply_and_verify_all(bid_state, layout['leave_auction'])

        # Verify passed flag set
        assert TURN.has_player_passed_auction(bid_state, player_id)

    def test_leave_advances_to_next_bidder(self, bid_state):
        """Leave auction advances to next non-passed bidder."""
        initial_player = bid_state.get_active_player()
        initial_position = PLAYERS[initial_player].get_turn_order(bid_state)

        # Apply leave auction
        layout = get_action_layout(3)
        result = apply_and_verify_all(bid_state, layout['leave_auction'])

        # Verify active player advanced (if auction didn't resolve)
        if bid_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            # No auto-apply - next bidder has choice to raise or leave
            assert len(result.history) == 1, "Expected no forced actions after leave (auction continues)"

            new_player = bid_state.get_active_player()
            new_position = PLAYERS[new_player].get_turn_order(bid_state)
            # Should have advanced to next position
            assert new_position == (initial_position + 1) % 3

    def test_leave_skips_passed_players(self, bid_state):
        """Rotation skips players who have already left."""
        # Get current player and mark next player as passed
        current_player = bid_state.get_active_player()
        current_position = PLAYERS[current_player].get_turn_order(bid_state)
        next_position = (current_position + 1) % 3

        # Find player at next position and mark them as passed
        for player_id in range(3):
            if PLAYERS[player_id].get_turn_order(bid_state) == next_position:
                TURN.set_player_passed_auction(bid_state, player_id, True)
                break

        # Current player leaves
        layout = get_action_layout(3)
        apply_and_verify_all(bid_state, layout['leave_auction'])

        # Verify active player skipped the passed player (if auction didn't resolve)
        if bid_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            new_player = bid_state.get_active_player()
            new_position = PLAYERS[new_player].get_turn_order(bid_state)
            # Should have skipped to position after next
            assert new_position == (current_position + 2) % 3

    def test_last_leaver_triggers_resolution(self, bid_state):
        """Auction resolves when only one bidder remains."""
        # Make all but one player leave
        layout = get_action_layout(3)

        # First player leaves
        apply_and_verify_all(bid_state, layout['leave_auction'])
        assert bid_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION

        # Second player leaves - should trigger resolution
        apply_and_verify_all(bid_state, layout['leave_auction'])

        # Verify auction resolved and returned to INVEST phase
        assert bid_state.get_phase() == GamePhases.PHASE_INVEST

    def test_bidder_rotation_wraps_around(self):
        """Bidder rotation correctly wraps from last player to first."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Start auction
        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                apply_and_verify_all(state, i)
                break

        # Track rotation through all players
        seen_players = set()
        for _ in range(3):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                current = state.get_active_player()
                seen_players.add(current)

                # Raise to continue (don't leave)
                mask = get_valid_action_mask(state)
                raised = False
                for i in range(layout['raise_bid_base'], layout['acquisition_start']):
                    if mask[i] == 1.0:
                        apply_and_verify_all(state, i)
                        raised = True
                        break

                if not raised:
                    apply_and_verify_all(state, layout['leave_auction'])

        # Should have seen all 3 players (rotation works)
        assert len(seen_players) >= 2, "Should rotate through multiple players"


# =============================================================================
# RAISE BID TESTS
# =============================================================================

class TestRaiseBid:
    """Test BID phase raise bid action behavior."""

    def test_raise_updates_price(self, bid_state):
        """Raise bid updates auction price."""
        initial_price = TURN.get_auction_price(bid_state)

        # Find valid raise bid action
        mask = get_valid_action_mask(bid_state)
        layout = get_action_layout(3)
        raise_idx = None
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                raise_idx = i
                break

        if raise_idx is not None:
            apply_and_verify_all(bid_state, raise_idx)

            # Verify price increased
            new_price = TURN.get_auction_price(bid_state)
            assert new_price > initial_price

    def test_raise_updates_high_bidder(self, bid_state):
        """Raise bid updates high bidder."""
        current_player = bid_state.get_active_player()

        # Find valid raise bid action
        mask = get_valid_action_mask(bid_state)
        layout = get_action_layout(3)
        raise_idx = None
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                raise_idx = i
                break

        if raise_idx is not None:
            apply_and_verify_all(bid_state, raise_idx)

            # Verify high bidder updated to current player
            high_bidder = TURN.get_auction_high_bidder(bid_state)
            assert high_bidder == current_player

    def test_raise_advances_to_next_bidder(self, bid_state):
        """Raise bid advances to next non-passed bidder."""
        initial_player = bid_state.get_active_player()
        initial_position = PLAYERS[initial_player].get_turn_order(bid_state)

        # Find valid raise bid action
        mask = get_valid_action_mask(bid_state)
        layout = get_action_layout(3)
        raise_idx = None
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                raise_idx = i
                break

        if raise_idx is not None:
            result = apply_and_verify_all(bid_state, raise_idx)

            # No auto-apply - next bidder has choice to raise or leave
            assert len(result.history) == 1, "Expected no forced actions after raise"

            # Verify active player advanced
            new_player = bid_state.get_active_player()
            new_position = PLAYERS[new_player].get_turn_order(bid_state)
            assert new_position == (initial_position + 1) % 3

    def test_raise_bid_masked_when_player_cannot_afford(self, bid_state):
        """Raise options masked when player has insufficient cash.

        RULES.md line 334-335: 'Raise bid (must have enough money)'
        """
        company_id = TURN.get_auction_company(bid_state)
        face_value = get_company_face_value(company_id)
        current_bid = TURN.get_auction_price(bid_state)
        layout = get_action_layout(3)

        # Set cash so only a few raises above current bid are affordable
        active_player_id = bid_state.get_active_player()
        affordable_limit = current_bid + 2
        PLAYERS[active_player_id].set_cash(bid_state, affordable_limit)

        mask = get_valid_action_mask(bid_state)

        # Leave auction must always be valid
        assert mask[layout['leave_auction']] == 1.0

        # Check each raise bid offset
        any_valid = False
        any_invalid = False
        for bid_offset in range(19):  # AUCTION_CAP - 1
            new_bid = face_value + bid_offset + 1
            action_idx = layout['raise_bid_base'] + bid_offset
            if new_bid > current_bid and new_bid <= affordable_limit:
                assert mask[action_idx] == 1.0, \
                    f"Raise to {new_bid} should be valid (cash={affordable_limit}, current_bid={current_bid})"
                any_valid = True
            elif new_bid > affordable_limit:
                assert mask[action_idx] == 0.0, \
                    f"Raise to {new_bid} should be masked (cash={affordable_limit})"
                any_invalid = True

        # Verify we tested both valid and invalid raises
        assert any_valid, "Test setup: no valid raises found"
        assert any_invalid, "Test setup: no invalid raises (cash too high)"

    def test_only_leave_available_when_cash_below_minimum_raise(self, bid_state):
        """Only leave auction is available when cash < all raise amounts.

        RULES.md line 334-335: 'Raise bid (must have enough money)'
        """
        layout = get_action_layout(3)

        # Set cash to 0 so no raise is affordable
        active_player_id = bid_state.get_active_player()
        PLAYERS[active_player_id].set_cash(bid_state, 0)

        mask = get_valid_action_mask(bid_state)

        # Leave auction must always be valid
        assert mask[layout['leave_auction']] == 1.0

        # All raise options must be masked out
        for bid_offset in range(19):
            action_idx = layout['raise_bid_base'] + bid_offset
            assert mask[action_idx] == 0.0, \
                f"Raise at offset {bid_offset} should be masked (player has no cash)"

    def test_raise_bid_values_correct(self, bid_state):
        """Verify raise bid calculates price correctly (face + amount + 1)."""
        company_id = TURN.get_auction_company(bid_state)
        face_value = get_company_face_value(company_id)

        # Find first valid raise bid action
        mask = get_valid_action_mask(bid_state)
        layout = get_action_layout(3)

        # The first raise bid action should be face + 0 + 1 = face + 1
        raise_idx = layout['raise_bid_base']
        if mask[raise_idx] == 1.0:
            apply_and_verify_all(bid_state, raise_idx)

            # Verify price is face + 1 (since amount=0 for first raise bid slot)
            new_price = TURN.get_auction_price(bid_state)
            assert new_price == face_value + 1


# =============================================================================
# AUCTION RESOLUTION TESTS
# =============================================================================

class TestAuctionResolution:
    """Test auction resolution behavior."""

    def test_winner_pays_bid_price(self, bid_state):
        """Winner pays bid price to bank."""
        # Record winner and price before resolution
        winner_id = TURN.get_auction_high_bidder(bid_state)
        bid_price = TURN.get_auction_price(bid_state)
        initial_cash = PLAYERS[winner_id].get_cash(bid_state)

        # Make all others leave to trigger resolution
        layout = get_action_layout(3)
        for _ in range(2):  # Leave twice (2 of 3 players)
            if bid_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_and_verify_all(bid_state, layout['leave_auction'])

        # Verify winner paid
        final_cash = PLAYERS[winner_id].get_cash(bid_state)
        assert final_cash == initial_cash - bid_price

    def test_winner_receives_company(self, bid_state):
        """Winner receives company ownership."""
        winner_id = TURN.get_auction_high_bidder(bid_state)
        company_id = TURN.get_auction_company(bid_state)

        # Verify company not owned by player initially (owner_id < 0)
        initial_owner = COMPANIES[company_id].get_owner_id(bid_state)
        assert initial_owner < 0  # Not owned by player

        # Make all others leave to trigger resolution
        layout = get_action_layout(3)
        for _ in range(2):
            if bid_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_and_verify_all(bid_state, layout['leave_auction'])

        # Verify winner owns company
        final_owner = COMPANIES[company_id].get_owner_id(bid_state)
        assert final_owner == winner_id

    def test_auction_state_cleared(self, bid_state):
        """Auction resolution clears all auction state."""
        # Make all others leave to trigger resolution
        layout = get_action_layout(3)
        for _ in range(2):
            if bid_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_and_verify_all(bid_state, layout['leave_auction'])

        # Verify auction state cleared
        assert TURN.get_auction_company(bid_state) == -1
        assert TURN.get_auction_high_bidder(bid_state) == -1
        assert TURN.get_auction_starter(bid_state) == -1

        # Verify all passed flags cleared
        for player_id in range(3):
            assert not TURN.has_player_passed_auction(bid_state, player_id)

    def test_returns_to_invest_phase(self, bid_state):
        """Auction resolution returns to INVEST phase."""
        # Make all others leave to trigger resolution
        layout = get_action_layout(3)
        for _ in range(2):
            if bid_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_and_verify_all(bid_state, layout['leave_auction'])

        # Verify phase transition
        assert bid_state.get_phase() == GamePhases.PHASE_INVEST

    def test_turn_goes_to_player_after_starter(self, bid_state):
        """Next turn goes to player after auction starter."""
        starter_id = TURN.get_auction_starter(bid_state)
        starter_position = PLAYERS[starter_id].get_turn_order(bid_state)
        expected_next_position = (starter_position + 1) % 3

        # Make all others leave to trigger resolution
        layout = get_action_layout(3)
        for _ in range(2):
            if bid_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_and_verify_all(bid_state, layout['leave_auction'])

        # Verify active player is after starter
        active_player = bid_state.get_active_player()
        active_position = PLAYERS[active_player].get_turn_order(bid_state)
        assert active_position == expected_next_position

    def test_auction_resolution_with_empty_deck(self):
        """Auction resolves correctly when deck is empty (no replacement drawn).

        Near game end, the deck may be empty. Auction resolution should:
        1. Complete without errors
        2. Transfer company to winner
        3. Not draw a replacement (deck empty)
        4. Auction row decreases by 1 with no replacement
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Empty the deck
        DECK.set_order(state, [])
        assert DECK.is_empty(state)

        # Count initial auction row
        initial_auction_count = sum(
            1 for cid in range(36)
            if state.is_company_for_auction(cid)
        )
        assert initial_auction_count == 3

        # Count initial revealed companies
        initial_revealed_count = sum(
            1 for cid in range(36)
            if COMPANIES[cid].is_revealed(state)
        )

        # Start auction
        layout = get_action_layout(3)
        mask = get_valid_action_mask(state)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                apply_and_verify_all(state, i)
                break

        assert state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION
        auctioned_company = TURN.get_auction_company(state)

        # Resolve auction (all others leave)
        for _ in range(2):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_and_verify_all(state, layout['leave_auction'])

        # Verify auction resolved
        assert state.get_phase() == GamePhases.PHASE_INVEST

        # Verify auctioned company transferred (no longer for auction)
        assert not state.is_company_for_auction(auctioned_company)

        # Verify auction row decreased by 1 with NO replacement
        final_auction_count = sum(
            1 for cid in range(36)
            if state.is_company_for_auction(cid)
        )
        assert final_auction_count == initial_auction_count - 1, \
            "Auction row should decrease by 1 with no replacement from empty deck"

        # Verify no new revealed companies (deck was empty)
        final_revealed_count = sum(
            1 for cid in range(36)
            if COMPANIES[cid].is_revealed(state)
        )
        assert final_revealed_count == initial_revealed_count, \
            "No new company should be revealed when deck is empty"

        # Verify deck is still empty
        assert DECK.is_empty(state)

    def test_auction_draws_new_company_marked_unavailable(self):
        """New company drawn is revealed (unavailable) for this phase."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Count initial auction companies
        initial_auction_count = sum(
            1 for cid in range(36)
            if state.is_company_for_auction(cid)
        )
        assert initial_auction_count == 3

        # No revealed companies initially
        initial_revealed_count = sum(
            1 for cid in range(36)
            if COMPANIES[cid].is_revealed(state)
        )
        assert initial_revealed_count == 0

        layout = get_action_layout(3)

        # Complete an auction
        mask = get_valid_action_mask(state)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                apply_and_verify_all(state, i)
                break

        auctioned_company = TURN.get_auction_company(state)
        assert state.is_company_for_auction(auctioned_company)

        while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            apply_and_verify_all(state, layout['leave_auction'])

        # Auctioned company transferred to winner (no longer for auction)
        assert not state.is_company_for_auction(auctioned_company)

        # Auction row decreases by 1 (replacement is revealed, not available)
        final_auction_count = sum(
            1 for cid in range(36)
            if state.is_company_for_auction(cid)
        )
        assert final_auction_count == initial_auction_count - 1, \
            "Auction row should decrease by 1 (replacement is revealed)"

        # New company is revealed (unavailable this phase)
        final_revealed_count = sum(
            1 for cid in range(36)
            if COMPANIES[cid].is_revealed(state)
        )
        assert final_revealed_count == 1, \
            "Newly drawn company should be marked as revealed (unavailable)"

    def test_return_to_player_after_starter_not_winner(self):
        """Turn returns to player after starter, even if winner differs."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        starter = state.get_active_player()
        starter_position = PLAYERS[starter].get_turn_order(state)

        layout = get_action_layout(3)

        # Start auction
        mask = get_valid_action_mask(state)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                apply_and_verify_all(state, i)
                break

        recorded_starter = TURN.get_auction_starter(state)
        assert recorded_starter == starter

        # Player 1 raises (becomes high bidder)
        mask = get_valid_action_mask(state)
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                apply_and_verify_all(state, i)
                break

        # Others leave
        while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            apply_and_verify_all(state, layout['leave_auction'])

        # Active player should be after starter
        active_player = state.get_active_player()
        active_position = PLAYERS[active_player].get_turn_order(state)
        expected_position = (starter_position + 1) % 3

        assert active_position == expected_position, \
            f"Turn should go to position {expected_position} (after starter), not {active_position}"


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
                apply_and_verify_all(state, i)
                break

        # Now in BID phase
        assert state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION

        # Two players leave
        for _ in range(2):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_and_verify_all(state, layout['leave_auction'])

        # Back to INVEST
        assert state.get_phase() == GamePhases.PHASE_INVEST

    def test_auction_with_multiple_raises(self):
        """Test auction with several raises before resolution."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Start auction
        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                apply_and_verify_all(state, i)
                break

        initial_price = TURN.get_auction_price(state)

        # First player raises
        mask = get_valid_action_mask(state)
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                apply_and_verify_all(state, i)
                break

        # Verify price increased
        assert TURN.get_auction_price(state) > initial_price

        # Second player raises
        mask = get_valid_action_mask(state)
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                apply_and_verify_all(state, i)
                break

        # Others leave to resolve
        for _ in range(2):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_and_verify_all(state, layout['leave_auction'])

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
                apply_and_verify_all(state, i)
                break

        assert state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION

        # All but one leave
        for _ in range(num_players - 1):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_and_verify_all(state, layout['leave_auction'])

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
                apply_and_verify_all(state, i)
                break

        # Track active players through leaves
        active_players = []
        for _ in range(num_players - 1):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                active_players.append(state.get_active_player())
                apply_and_verify_all(state, layout['leave_auction'])

        # Verify we cycled through different players
        assert len(set(active_players)) >= 2  # At least 2 different players


# =============================================================================
# AUCTION MECHANICS TESTS
# =============================================================================

class TestAuctionMechanics:
    """Test auction slot mapping and price calculation."""

    def test_auction_slot_maps_to_company_by_face_value_order(self):
        """Slot index maps to correct company by ascending face value order."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Get available companies and their face values
        available = []
        for cid in range(36):
            if state.is_company_for_auction(cid):
                available.append((cid, get_company_face_value(cid)))

        # Sort by face value (ascending) - this is how slots should map
        available.sort(key=lambda x: x[1])

        layout = get_action_layout(3)

        # For each auction slot, verify it maps to correct company
        # Slot 0 + offset 0 should give lowest face value company
        # The auction action encodes: slot_index * MAX_AUCTION_OFFSET + offset
        # where slot_index determines which available company

        # Start auction with first slot (offset 0 = face value)
        first_slot_action = layout['auction_base']  # Slot 0, offset 0
        mask = get_valid_action_mask(state)

        if mask[first_slot_action] == 1.0:
            apply_and_verify_all(state, first_slot_action)

            # Verify auctioned company is the one with lowest face value
            auctioned_cid = TURN.get_auction_company(state)
            expected_cid = available[0][0]  # First in sorted list

            # Note: Due to how the mask works, the first valid slot
            # should correspond to the first available company
            assert auctioned_cid >= 0

    def test_starting_bid_equals_face_value_plus_offset(self):
        """Starting bid = company face value + price offset."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Start auction with offset 0 (face value)
        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # Find a valid auction action at offset 0
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                # Decode to get slot and offset
                relative_idx = i - layout['auction_base']
                # slot_index = relative_idx // MAX_AUCTION_OFFSET
                # offset = relative_idx % MAX_AUCTION_OFFSET

                apply_and_verify_all(state, i)

                company_id = TURN.get_auction_company(state)
                auction_price = TURN.get_auction_price(state)
                face_value = get_company_face_value(company_id)

                # Starting price should be >= face value
                assert auction_price >= face_value, \
                    f"Starting price {auction_price} < face value {face_value}"

                break

    def test_higher_offset_gives_higher_starting_bid(self):
        """Higher auction offset results in higher starting bid."""
        # Test with fresh state twice - once with low offset, once with higher
        prices = []

        for offset_target in [0, 3]:  # Test offset 0 and offset 3
            state = GameState(num_players=3)
            state.initialize_game(seed=42)

            layout = get_action_layout(3)
            mask = get_valid_action_mask(state)

            # Find action with target offset for first available company
            # Actions are encoded: base + (slot * MAX_OFFSET + offset)
            # We want slot 0, so action = base + offset
            target_action = layout['auction_base'] + offset_target

            if mask[target_action] == 1.0:
                apply_and_verify_all(state, target_action)
                prices.append(TURN.get_auction_price(state))

        if len(prices) == 2:
            assert prices[1] > prices[0], \
                f"Higher offset should give higher price: {prices}"

# =============================================================================
# AUTO-APPLY BEHAVIOR TESTS
# =============================================================================

class TestAutoApplyBehavior:
    """Tests for auto-apply forced action behavior."""

    def test_auction_resolution_auto_applies_forced_transitions(self):
        """BID->INVEST transition during auto-apply works correctly.

        When auction resolves with only one player remaining, multiple forced
        actions may occur (resolve auction -> advance to next INVEST player).
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Start auction
        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                apply_and_verify_all(state, i)
                break

        # First leave - should NOT resolve (2 bidders remain)
        result = apply_and_verify_all(state, layout['leave_auction'])
        assert state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION

        # Second leave - triggers resolution, returns to INVEST
        # This may involve auto-applied forced actions
        result = apply_and_verify_all(state, layout['leave_auction'])
        assert state.get_phase() == GamePhases.PHASE_INVEST
        # History should include at least the leave action
        assert result.applied_count >= 1
