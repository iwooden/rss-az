"""Tests for the BID phase.

Covers: LEAVE (pass), RAISE, auction resolution, next-bidder advancement,
phase transitions back to INVEST, and legal-action enumeration.
"""
import pytest

from core.actions import (
    ACTION_PASS_PY as ACTION_PASS,
    ACTION_RAISE_PY as ACTION_RAISE,
    ACTION_AUCTION_PY as ACTION_AUCTION,
)
from core.data import GamePhases, GameConstants
from entities.turn import TURN
from entities.player import PLAYERS
from entities.company import COMPANIES, CompanyLocation
from entities.deck import DECK

from tests.phases.conftest import (
    apply_and_verify,
    get_legal_actions,
    find_legal_action,
    find_legal_action_with_info,
)
from tests.phases.helpers.ownership import count_at_location


# =============================================================================
# HELPERS
# =============================================================================

def _enter_bid_phase(state, bid_offset=0):
    """Start an auction from INVEST to enter BID phase.

    Uses the first available auction action with the given bid_offset.
    Returns (starter_player_id, company_id, bid_price).
    """
    starter = TURN.get_active_player(state)
    actions = get_legal_actions(state)
    for aid, info in actions:
        if info.action_type == ACTION_AUCTION and info.amount == bid_offset:
            company_id = info.company_id
            face = COMPANIES[company_id].get_face_value()
            bid_price = face + bid_offset
            apply_and_verify(state, aid)
            return starter, company_id, bid_price

    pytest.fail(f"No auction action with offset={bid_offset} found")



# =============================================================================
# LEAVE (PASS) ACTION TESTS
# =============================================================================

class TestLeaveAction:
    """Test BID phase leave-auction action behavior."""

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_leave_marks_player_as_passed(self, game_state):
        """Leaving the auction marks the active player as passed.

        In 2p, a leave immediately resolves the auction (clearing all
        passed flags), so we need >= 3 players to observe the flag.
        """
        num_players = TURN.get_num_players(game_state)

        _enter_bid_phase(game_state)
        active = TURN.get_active_player(game_state)
        assert not PLAYERS[active].has_passed(game_state)

        leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, leave_id)
        assert PLAYERS[active].has_passed(game_state)

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_leave_advances_to_next_non_passed_bidder(self, game_state):
        """Leaving advances control to the next non-passed bidder."""
        num_players = TURN.get_num_players(game_state)

        _enter_bid_phase(game_state)
        first_bidder = TURN.get_active_player(game_state)

        leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, leave_id)

        next_bidder = TURN.get_active_player(game_state)
        assert next_bidder != first_bidder

    @pytest.mark.parametrize("game_state", [4, 5, 6], indirect=True)
    def test_leave_skips_already_passed_players(self, game_state):
        """After multiple leaves, advancing skips players who already left."""
        num_players = TURN.get_num_players(game_state)

        _enter_bid_phase(game_state)

        # First player leaves
        first_leaver = TURN.get_active_player(game_state)
        leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, leave_id)

        # Second player leaves
        second_leaver = TURN.get_active_player(game_state)
        leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, leave_id)

        # Third player should not be either of the leavers
        third_active = TURN.get_active_player(game_state)
        assert third_active != first_leaver
        assert third_active != second_leaver

    def test_last_leave_resolves_auction(self, game_state):
        """When all but one bidder leave, the auction resolves."""
        num_players = TURN.get_num_players(game_state)
        _enter_bid_phase(game_state)

        # All players except one leave (n-1 leaves needed; starter already
        # placed the opening bid and is skipped, so n-1 others remain).
        for _ in range(num_players - 1):
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        # Should be back in INVEST
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)


# =============================================================================
# RAISE ACTION TESTS
# =============================================================================

class TestRaiseAction:
    """Test BID phase raise action behavior."""

    def test_raise_updates_auction_price(self, game_state):
        """Raising sets auction_price to face_value + 1 + bid_offset."""
        _, company_id, _ = _enter_bid_phase(game_state)
        face = COMPANIES[company_id].get_face_value()

        raise_id, raise_info = find_legal_action_with_info(game_state, action_type=ACTION_RAISE)
        expected_price = face + 1 + raise_info.amount

        apply_and_verify(game_state, raise_id)
        assert TURN.get_auction_price(game_state) == expected_price

    def test_raise_updates_high_bidder(self, game_state):
        """Raising sets the raising player as the new high bidder."""
        _enter_bid_phase(game_state)
        raiser = TURN.get_active_player(game_state)

        raise_id = find_legal_action(game_state, action_type=ACTION_RAISE)
        apply_and_verify(game_state, raise_id)

        assert TURN.get_auction_high_bidder(game_state) == raiser

    def test_raise_does_not_mark_player_as_passed(self, game_state):
        """Raising keeps the player in the auction (not marked as passed)."""
        _enter_bid_phase(game_state)
        raiser = TURN.get_active_player(game_state)

        raise_id = find_legal_action(game_state, action_type=ACTION_RAISE)
        apply_and_verify(game_state, raise_id)

        assert not PLAYERS[raiser].has_passed(game_state)

    def test_raise_advances_to_next_bidder(self, game_state):
        """After raising, control advances to the next bidder."""
        _enter_bid_phase(game_state)
        raiser = TURN.get_active_player(game_state)

        raise_id = find_legal_action(game_state, action_type=ACTION_RAISE)
        apply_and_verify(game_state, raise_id)

        next_active = TURN.get_active_player(game_state)
        assert next_active != raiser

    def test_raise_stays_in_bid_phase(self, game_state):
        """Raising stays in BID phase (does not resolve the auction)."""
        _enter_bid_phase(game_state)

        raise_id = find_legal_action(game_state, action_type=ACTION_RAISE)
        apply_and_verify(game_state, raise_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_BID)

    @pytest.mark.parametrize("game_state", [2], indirect=True)
    def test_raise_then_leave_resolves_in_2p(self, game_state):
        """In 2p: raise then leave resolves the auction to the raiser."""
        num_players = TURN.get_num_players(game_state)

        _, company_id, _ = _enter_bid_phase(game_state)

        # Next bidder raises
        raiser = TURN.get_active_player(game_state)
        raise_id = find_legal_action(game_state, action_type=ACTION_RAISE)
        apply_and_verify(game_state, raise_id)

        # Starter leaves -> auction resolves, raiser wins
        leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, leave_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)
        # Winner should own the company
        assert COMPANIES[company_id].get_owner_id(game_state) == raiser

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_consecutive_raises_increase_price(self, game_state):
        """Multiple raises monotonically increase the auction price."""
        num_players = TURN.get_num_players(game_state)

        # Give all players plenty of cash
        for p in range(num_players):
            PLAYERS[p].set_cash(game_state, 200)

        _enter_bid_phase(game_state)
        first_price = TURN.get_auction_price(game_state)

        # First raise
        raise_id = find_legal_action(game_state, action_type=ACTION_RAISE)
        apply_and_verify(game_state, raise_id)
        second_price = TURN.get_auction_price(game_state)
        assert second_price > first_price

        # Second raise
        raise_id = find_legal_action(game_state, action_type=ACTION_RAISE)
        apply_and_verify(game_state, raise_id)
        third_price = TURN.get_auction_price(game_state)
        assert third_price > second_price


# =============================================================================
# AUCTION RESOLUTION TESTS
# =============================================================================

class TestAuctionResolution:
    """Test auction resolution when only one bidder remains."""

    def test_winner_pays_bid_price(self, game_state):
        """The auction winner's cash is reduced by the final bid price."""
        num_players = TURN.get_num_players(game_state)
        starter, _, bid_price = _enter_bid_phase(game_state)

        # Starter is the high bidder from the initial auction
        starter_cash_before = PLAYERS[starter].get_cash(game_state)

        # Everyone else leaves
        for _ in range(num_players - 1):
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        # Starter wins and pays the bid
        assert PLAYERS[starter].get_cash(game_state) == starter_cash_before - bid_price

    def test_winner_receives_company(self, game_state):
        """The auction winner receives the auctioned company."""
        num_players = TURN.get_num_players(game_state)
        starter, company_id, _ = _enter_bid_phase(game_state)

        for _ in range(num_players - 1):
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        assert PLAYERS[starter].owns_company(game_state, company_id)

    def test_raiser_wins_after_others_leave(self, game_state):
        """A player who raises and then everyone else leaves wins the auction."""
        num_players = TURN.get_num_players(game_state)
        _, company_id, _ = _enter_bid_phase(game_state)
        raiser = TURN.get_active_player(game_state)

        # Give raiser enough cash
        PLAYERS[raiser].set_cash(game_state, 200)

        raise_id, raise_info = find_legal_action_with_info(game_state, action_type=ACTION_RAISE)
        expected_price = COMPANIES[company_id].get_face_value() + 1 + raise_info.amount
        apply_and_verify(game_state, raise_id)

        # Everyone else leaves
        remaining_to_leave = num_players - 1  # raiser stays, starter+others must leave
        for _ in range(remaining_to_leave):
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)
        assert PLAYERS[raiser].owns_company(game_state, company_id)
        raiser_cash = PLAYERS[raiser].get_cash(game_state)
        assert raiser_cash == 200 - expected_price

    def test_replacement_card_drawn(self, game_state):
        """A replacement card is drawn from the deck on auction resolution."""
        num_players = TURN.get_num_players(game_state)
        deck_before = count_at_location(game_state, CompanyLocation.LOC_REVEALED)
        _enter_bid_phase(game_state)

        for _ in range(num_players - 1):
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        revealed_after = count_at_location(game_state, CompanyLocation.LOC_REVEALED)
        # One new card should be in LOC_REVEALED (or deck was empty)
        assert revealed_after >= deck_before

    def test_empty_deck_resolution_draws_no_replacement_and_still_resolves(self, game_state):
        """Auction resolution with an empty deck succeeds without creating a revealed card."""
        num_players = TURN.get_num_players(game_state)
        starter, company_id, bid_price = _enter_bid_phase(game_state)
        starter_cash_before = PLAYERS[starter].get_cash(game_state)

        DECK.set_order(game_state, [])
        assert TURN.get_cards_remaining(game_state) == 0
        assert DECK.is_empty(game_state)
        assert count_at_location(game_state, CompanyLocation.LOC_REVEALED) == 0

        for _ in range(num_players - 1):
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)
        assert PLAYERS[starter].owns_company(game_state, company_id)
        assert PLAYERS[starter].get_cash(game_state) == starter_cash_before - bid_price
        assert TURN.get_cards_remaining(game_state) == 0
        assert DECK.is_empty(game_state)
        assert count_at_location(game_state, CompanyLocation.LOC_REVEALED) == 0

    def test_auction_fields_cleared_after_resolution(self, game_state):
        """All auction scratch fields are cleared after resolution."""
        num_players = TURN.get_num_players(game_state)
        _enter_bid_phase(game_state)

        for _ in range(num_players - 1):
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        assert TURN.get_auction_price(game_state) == 0
        assert TURN.get_auction_high_bidder(game_state) == -1
        assert TURN.get_auction_starter(game_state) == -1
        assert TURN.get_active_company(game_state) == -1

    def test_passed_flags_cleared_after_resolution(self, game_state):
        """All player has_passed flags are cleared after resolution."""
        num_players = TURN.get_num_players(game_state)
        _enter_bid_phase(game_state)

        for _ in range(num_players - 1):
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        for p in range(num_players):
            assert not PLAYERS[p].has_passed(game_state)

    def test_control_goes_to_player_after_starter(self, game_state):
        """After resolution, active player is the one after the starter."""
        num_players = TURN.get_num_players(game_state)
        starter, _, _ = _enter_bid_phase(game_state)

        for _ in range(num_players - 1):
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        # Active player should be the one AFTER the starter in turn order
        active = TURN.get_active_player(game_state)
        starter_pos = PLAYERS[starter].get_turn_order(game_state)
        active_pos = PLAYERS[active].get_turn_order(game_state)
        assert active_pos == (starter_pos + 1) % num_players

    def test_winner_pays_raised_price(self, game_state):
        """Winner pays the final raised price, not the opening bid."""
        num_players = TURN.get_num_players(game_state)

        _, company_id, _ = _enter_bid_phase(game_state)

        # Give next bidder enough cash to raise
        raiser = TURN.get_active_player(game_state)
        PLAYERS[raiser].set_cash(game_state, 200)
        raiser_cash_before = 200

        # Get the raise action details before applying
        raise_actions = [(aid, info) for aid, info in get_legal_actions(game_state)
                        if info.action_type == ACTION_RAISE]
        # Pick the highest affordable raise for a clear test
        raise_aid, raise_info = raise_actions[-1]
        face = COMPANIES[company_id].get_face_value()
        final_price = face + 1 + raise_info.amount

        apply_and_verify(game_state, raise_aid)

        # Everyone else leaves (raiser becomes last remaining)
        remaining = num_players - 1
        for _ in range(remaining):
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        assert PLAYERS[raiser].owns_company(game_state, company_id)
        assert PLAYERS[raiser].get_cash(game_state) == raiser_cash_before - final_price


# =============================================================================
# ENUMERATION TESTS
# =============================================================================

class TestEnumeration:
    """Test legal-action enumeration for the BID phase."""

    def test_leave_always_legal(self, game_state):
        """Leave (pass) is always legal in BID phase."""
        _enter_bid_phase(game_state)
        actions = get_legal_actions(game_state)
        pass_actions = [info for _, info in actions if info.action_type == ACTION_PASS]
        assert len(pass_actions) == 1

    def test_raise_actions_are_strictly_above_current_bid(self, game_state):
        """All raise actions produce bids strictly greater than current bid."""
        _enter_bid_phase(game_state)
        company_id = TURN.get_active_company(game_state)
        current_bid = TURN.get_auction_price(game_state)
        face = COMPANIES[company_id].get_face_value()

        actions = get_legal_actions(game_state)
        for _, info in actions:
            if info.action_type == ACTION_RAISE:
                new_bid = face + 1 + info.amount
                assert new_bid > current_bid, (
                    f"raise offset={info.amount} produces bid {new_bid} "
                    f"not greater than current {current_bid}"
                )

    def test_raise_actions_limited_by_cash(self, game_state):
        """No raise action exceeds the active player's cash."""
        _enter_bid_phase(game_state)
        active = TURN.get_active_player(game_state)
        player_cash = PLAYERS[active].get_cash(game_state)
        company_id = TURN.get_active_company(game_state)
        face = COMPANIES[company_id].get_face_value()

        actions = get_legal_actions(game_state)
        for _, info in actions:
            if info.action_type == ACTION_RAISE:
                new_bid = face + 1 + info.amount
                assert new_bid <= player_cash, (
                    f"raise offset={info.amount} produces bid {new_bid} "
                    f"exceeding cash {player_cash}"
                )

    def test_no_raise_when_cannot_beat_current_bid(self, game_state):
        """When cash equals the current bid, PASS is the only legal action."""
        _, _, bid_price = _enter_bid_phase(game_state)
        active = TURN.get_active_player(game_state)

        # Cash equal to the current bid cannot afford any strictly higher bid.
        PLAYERS[active].set_cash(game_state, bid_price)

        actions = get_legal_actions(game_state)
        raise_actions = [
            (action_id, info)
            for action_id, info in actions
            if info.action_type == ACTION_RAISE
        ]
        pass_actions = [
            (action_id, info)
            for action_id, info in actions
            if info.action_type == ACTION_PASS
        ]

        assert raise_actions == []
        assert len(pass_actions) == 1
        assert len(actions) == 1

    def test_raise_count_matches_affordable_offsets(self, game_state):
        """Number of raise actions equals the number of affordable bid offsets."""
        _enter_bid_phase(game_state)
        active = TURN.get_active_player(game_state)
        player_cash = PLAYERS[active].get_cash(game_state)
        company_id = TURN.get_active_company(game_state)
        face = COMPANIES[company_id].get_face_value()
        current_bid = TURN.get_auction_price(game_state)

        # Compute expected count
        expected_count = 0
        auction_cap = int(GameConstants.AUCTION_CAP)
        min_offset = max(0, current_bid - face)
        for offset in range(min_offset, auction_cap - 1):
            new_bid = face + offset + 1
            if new_bid > player_cash:
                break
            expected_count += 1

        actions = get_legal_actions(game_state)
        actual_count = sum(1 for _, info in actions if info.action_type == ACTION_RAISE)
        assert actual_count == expected_count

    def test_only_leave_when_cash_zero(self, game_state):
        """With 0 cash, only leave is legal (no raises affordable)."""
        _enter_bid_phase(game_state)
        active = TURN.get_active_player(game_state)
        PLAYERS[active].set_cash(game_state, 0)

        actions = get_legal_actions(game_state)
        assert len(actions) == 1
        assert actions[0][1].action_type == ACTION_PASS

    def test_raise_at_exact_cash_boundary(self, game_state):
        """Raise is legal when new bid exactly equals player cash."""
        _enter_bid_phase(game_state)
        active = TURN.get_active_player(game_state)
        company_id = TURN.get_active_company(game_state)
        face = COMPANIES[company_id].get_face_value()
        current_bid = TURN.get_auction_price(game_state)

        # Set cash so exactly one raise is affordable: the minimum raise
        min_offset = max(0, current_bid - face)
        min_raise = face + min_offset + 1
        PLAYERS[active].set_cash(game_state, min_raise)

        actions = get_legal_actions(game_state)
        raise_actions = [info for _, info in actions if info.action_type == ACTION_RAISE]
        assert len(raise_actions) == 1
        assert face + 1 + raise_actions[0].amount == min_raise

    def test_max_raise_capped_at_auction_cap(self, game_state):
        """Even with unlimited cash, raises are capped at AUCTION_CAP - 1 offsets."""
        _enter_bid_phase(game_state)
        active = TURN.get_active_player(game_state)
        PLAYERS[active].set_cash(game_state, 10000)

        actions = get_legal_actions(game_state)
        raise_actions = [info for _, info in actions if info.action_type == ACTION_RAISE]
        auction_cap = int(GameConstants.AUCTION_CAP)
        # Maximum possible offsets is AUCTION_CAP - 1 (0..13 = 14 offsets)
        assert len(raise_actions) <= auction_cap - 1


# =============================================================================
# PHASE TRANSITION TESTS
# =============================================================================

class TestPhaseTransitions:
    """Test phase transitions from BID."""

    def test_resolution_returns_to_invest(self, game_state):
        """Auction resolution transitions back to PHASE_INVEST."""
        num_players = TURN.get_num_players(game_state)
        _enter_bid_phase(game_state)

        for _ in range(num_players - 1):
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)

    def test_raise_keeps_bid_phase(self, game_state):
        """Raising does not change the phase."""
        _enter_bid_phase(game_state)

        raise_id = find_legal_action(game_state, action_type=ACTION_RAISE)
        apply_and_verify(game_state, raise_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_BID)

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_leave_without_resolution_keeps_bid_phase(self, game_state):
        """A single leave when >2 bidders remain stays in BID."""
        num_players = TURN.get_num_players(game_state)

        _enter_bid_phase(game_state)

        leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, leave_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_BID)

    def test_multiple_auctions_in_one_invest_phase(self, game_state):
        """After auction resolution, a new auction can be started."""
        num_players = TURN.get_num_players(game_state)
        _enter_bid_phase(game_state)

        # Resolve first auction
        for _ in range(num_players - 1):
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)

        # Start a second auction (should be possible if auction-row companies exist)
        actions = get_legal_actions(game_state)
        auction_actions = [aid for aid, info in actions
                          if info.action_type == ACTION_AUCTION]
        if auction_actions:
            apply_and_verify(game_state, auction_actions[0])
            assert TURN.get_phase(game_state) == int(GamePhases.PHASE_BID)


# =============================================================================
# BIDDING SEQUENCE TESTS
# =============================================================================

class TestBiddingSequence:
    """Test multi-player bidding sequences."""

    @pytest.mark.parametrize("game_state", [2], indirect=True)
    def test_two_player_bidding_war(self, game_state):
        """Two players alternate raising until one leaves."""
        num_players = TURN.get_num_players(game_state)

        for p in range(num_players):
            PLAYERS[p].set_cash(game_state, 200)

        starter, company_id, _ = _enter_bid_phase(game_state)

        # Players alternate: raise, raise, leave
        p1 = TURN.get_active_player(game_state)
        raise_id = find_legal_action(game_state, action_type=ACTION_RAISE)
        apply_and_verify(game_state, raise_id)

        # Starter's turn again
        p2 = TURN.get_active_player(game_state)
        assert p2 == starter
        raise_id = find_legal_action(game_state, action_type=ACTION_RAISE)
        apply_and_verify(game_state, raise_id)

        # p1's turn — leave
        assert TURN.get_active_player(game_state) == p1
        leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, leave_id)

        # Starter wins
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)
        assert PLAYERS[starter].owns_company(game_state, company_id)

    def test_all_but_last_leave_sequentially(self, game_state):
        """Players leave one by one until only one remains."""
        num_players = TURN.get_num_players(game_state)
        starter, _, _ = _enter_bid_phase(game_state)

        leavers = []
        for _ in range(num_players - 1):
            leaver = TURN.get_active_player(game_state)
            leavers.append(leaver)
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        # All leavers are unique
        assert len(set(leavers)) == num_players - 1
        # The non-leaver is the starter (who opened the bid)
        assert starter not in leavers

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_raise_changes_winner_identity(self, game_state):
        """A raise followed by others leaving makes the raiser the winner."""
        num_players = TURN.get_num_players(game_state)

        for p in range(num_players):
            PLAYERS[p].set_cash(game_state, 200)

        starter, company_id, _ = _enter_bid_phase(game_state)

        # First non-starter raises
        raiser = TURN.get_active_player(game_state)
        assert raiser != starter
        raise_id = find_legal_action(game_state, action_type=ACTION_RAISE)
        apply_and_verify(game_state, raise_id)

        # Everyone else (including starter) leaves
        for _ in range(num_players - 1):
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        assert PLAYERS[raiser].owns_company(game_state, company_id)
