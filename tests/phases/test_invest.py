"""Tests for the INVEST phase.

Covers: PASS, AUCTION (start), BUY_SHARE, SELL_SHARE actions, round-trip
limits, legal-action enumeration, price movement, and phase transitions.
"""
from core.driver import STATUS_GAME_OVER_PY as STATUS_GAME_OVER
from core.actions import (
    ACTION_PASS_PY as ACTION_PASS,
    ACTION_AUCTION_PY as ACTION_AUCTION,
    ACTION_BUY_SHARE_PY as ACTION_BUY_SHARE,
    ACTION_SELL_SHARE_PY as ACTION_SELL_SHARE,
)
from core.data import GamePhases, GameConstants
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES
from entities.market import MARKET

from tests.phases.conftest import (
    apply_and_verify,
    get_legal_actions,
    find_legal_action,
    float_corp_for_test,
)


# =============================================================================
# HELPERS
# =============================================================================

def _make_trade_state(game_state, corp_id=0, player_id=0, par_index=10, float_shares=2):
    """Set up a state with a floated corp for buy/sell testing.

    Floats corp with ``float_shares`` issued (split evenly: player gets
    ``float_shares``, bank gets ``float_shares``). Default par_index=10 ($14).
    """
    float_corp_for_test(
        game_state, corp_id=corp_id, player_id=player_id,
        par_index=par_index, float_shares=float_shares,
    )
    TURN.set_phase(game_state, int(GamePhases.PHASE_INVEST))
    TURN.set_active_player(game_state, player_id)


# =============================================================================
# PASS ACTION TESTS
# =============================================================================

class TestPassAction:
    """Test INVEST phase pass action behavior."""

    def test_pass_increments_consecutive_passes(self, game_state):
        """Pass increments the consecutive-passes counter."""
        assert TURN.get_consecutive_passes(game_state) == 0
        action_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, action_id)
        assert TURN.get_consecutive_passes(game_state) == 1

    def test_pass_advances_active_player(self, game_state):
        """Pass advances to the next player in turn order."""
        num_players = TURN.get_num_players(game_state)
        p0 = TURN.get_active_player(game_state)
        action_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, action_id)
        p1 = TURN.get_active_player(game_state)
        assert p1 != p0

        pos0 = PLAYERS[p0].get_turn_order(game_state)
        pos1 = PLAYERS[p1].get_turn_order(game_state)
        assert pos1 == (pos0 + 1) % num_players

    def test_pass_follows_full_turn_order(self, game_state):
        """Consecutive passes cycle through all players."""
        num_players = TURN.get_num_players(game_state)
        visited = [TURN.get_active_player(game_state)]
        for _ in range(num_players - 1):
            pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, pass_id)
            visited.append(TURN.get_active_player(game_state))
        assert len(set(visited)) == num_players

    def test_all_players_pass_transitions_to_wrap_up(self, game_state):
        """All consecutive passes end INVEST (driver chains through to next INVEST turn)."""
        num_players = TURN.get_num_players(game_state)
        for _ in range(num_players):
            pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, pass_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)
        assert TURN.get_turn_number(game_state) == 2
        assert TURN.get_consecutive_passes(game_state) == 0

    def test_pass_clears_roundtrip_tracking_on_phase_end(self, game_state):
        """Round-trip counters are cleared when all players pass."""
        _make_trade_state(game_state, corp_id=0, player_id=0)
        num_players = TURN.get_num_players(game_state)

        buy_id = find_legal_action(game_state, action_type=ACTION_BUY_SHARE, corp_id=0)
        apply_and_verify(game_state, buy_id)
        assert PLAYERS[0].get_share_buys(game_state, 0) == 1

        for _ in range(num_players):
            pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, pass_id)

        assert PLAYERS[0].get_share_buys(game_state, 0) == 0
        assert PLAYERS[0].get_share_sells(game_state, 0) == 0


# =============================================================================
# START AUCTION TESTS
# =============================================================================

class TestStartAuction:
    """Test INVEST auction-start action behavior."""

    def test_auction_transitions_to_bid_phase(self, game_state):
        """Starting an auction transitions to PHASE_BID."""
        action_id = find_legal_action(game_state, action_type=ACTION_AUCTION)
        apply_and_verify(game_state, action_id)
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_BID)

    def test_auction_sets_active_company(self, game_state):
        """Auction sets active_company to the auctioned company."""
        actions = get_legal_actions(game_state)
        auction_actions = [(aid, info) for aid, info in actions
                          if info.action_type == ACTION_AUCTION]
        assert len(auction_actions) > 0
        aid, info = auction_actions[0]
        expected_company = info.company_id

        apply_and_verify(game_state, aid)
        assert TURN.get_active_company(game_state) == expected_company

    def test_auction_clears_price_and_high_bidder(self, game_state):
        """Selecting a company enters BID with no opening bid placed."""
        action_id = find_legal_action(game_state, action_type=ACTION_AUCTION)
        apply_and_verify(game_state, action_id)
        assert TURN.get_auction_price(game_state) == 0
        assert TURN.get_auction_high_bidder(game_state) == -1

    def test_auction_records_starter(self, game_state):
        """Auction starter is recorded, but no high bidder exists yet."""
        starter = TURN.get_active_player(game_state)
        action_id = find_legal_action(game_state, action_type=ACTION_AUCTION)
        apply_and_verify(game_state, action_id)
        assert TURN.get_auction_starter(game_state) == starter
        # No opening bid yet — high_bidder sentinel remains -1.
        assert TURN.get_auction_high_bidder(game_state) == -1

    def test_auction_keeps_starter_as_active_player(self, game_state):
        """Starter stays as the active player and places the opening bid."""
        starter = TURN.get_active_player(game_state)
        action_id = find_legal_action(game_state, action_type=ACTION_AUCTION)
        apply_and_verify(game_state, action_id)

        assert TURN.get_active_player(game_state) == starter

    def test_auction_clears_consecutive_passes(self, game_state):
        """Auction start resets the consecutive-passes counter."""
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)
        assert TURN.get_consecutive_passes(game_state) == 1

        action_id = find_legal_action(game_state, action_type=ACTION_AUCTION)
        apply_and_verify(game_state, action_id)
        assert TURN.get_consecutive_passes(game_state) == 0

    def test_auction_blocked_when_cannot_afford_face_value(self, game_state):
        """No auction actions when player has 0 cash."""
        active = TURN.get_active_player(game_state)
        PLAYERS[active].set_cash(game_state, 0)

        actions = get_legal_actions(game_state)
        auction_actions = [a for _, a in actions if a.action_type == ACTION_AUCTION]
        assert len(auction_actions) == 0
        # Pass must still be legal
        pass_actions = [a for _, a in actions if a.action_type == ACTION_PASS]
        assert len(pass_actions) == 1

    def test_auction_requires_face_value_affordable(self, game_state):
        """Select-company is legal iff player cash >= company face value."""
        actions = get_legal_actions(game_state)
        auction_actions = [(aid, info) for aid, info in actions
                          if info.action_type == ACTION_AUCTION]
        assert auction_actions, "Expected at least one auction action"

        # Pick a company and drop cash to one below its face value.
        company_id = auction_actions[0][1].company_id
        face = COMPANIES[company_id].get_face_value()
        active = TURN.get_active_player(game_state)
        PLAYERS[active].set_cash(game_state, face - 1)

        new_actions = get_legal_actions(game_state)
        selectable_companies = {
            info.company_id for _, info in new_actions
            if info.action_type == ACTION_AUCTION
        }
        assert company_id not in selectable_companies


# =============================================================================
# BUY SHARE TESTS
# =============================================================================

class TestBuyShare:
    """Test INVEST buy-share action behavior."""

    def test_buy_transfers_share(self, game_state):
        """Buy moves one share from bank to player."""
        _make_trade_state(game_state)
        initial_bank = CORPS[0].get_bank_shares(game_state)
        initial_player = PLAYERS[0].get_shares(game_state, 0)

        buy_id = find_legal_action(game_state, action_type=ACTION_BUY_SHARE, corp_id=0)
        apply_and_verify(game_state, buy_id)

        assert CORPS[0].get_bank_shares(game_state) == initial_bank - 1
        assert PLAYERS[0].get_shares(game_state, 0) == initial_player + 1

    def test_buy_pays_new_higher_price(self, game_state):
        """Player pays the new (higher) share price, not the old one."""
        _make_trade_state(game_state, par_index=10)
        initial_cash = PLAYERS[0].get_cash(game_state)
        old_index = CORPS[0].get_price_index(game_state)

        buy_id = find_legal_action(game_state, action_type=ACTION_BUY_SHARE, corp_id=0)
        apply_and_verify(game_state, buy_id)

        new_price = CORPS[0].get_share_price(game_state)
        assert CORPS[0].get_price_index(game_state) > old_index
        assert PLAYERS[0].get_cash(game_state) == initial_cash - new_price

    def test_buy_does_not_change_corp_cash(self, game_state):
        """Payment goes to bank, not the corporation."""
        _make_trade_state(game_state)
        corp_cash_before = CORPS[0].get_cash(game_state)

        buy_id = find_legal_action(game_state, action_type=ACTION_BUY_SHARE, corp_id=0)
        apply_and_verify(game_state, buy_id)

        assert CORPS[0].get_cash(game_state) == corp_cash_before

    def test_buy_moves_price_up(self, game_state):
        """Buy moves corp price index to the next available space."""
        _make_trade_state(game_state, par_index=10)
        old_index = CORPS[0].get_price_index(game_state)

        buy_id = find_legal_action(game_state, action_type=ACTION_BUY_SHARE, corp_id=0)
        apply_and_verify(game_state, buy_id)

        assert CORPS[0].get_price_index(game_state) > old_index

    def test_buy_frees_old_market_space(self, game_state):
        """Old market space is freed after buying."""
        _make_trade_state(game_state, par_index=10)
        old_index = CORPS[0].get_price_index(game_state)
        assert not MARKET.is_space_available(game_state, old_index)

        buy_id = find_legal_action(game_state, action_type=ACTION_BUY_SHARE, corp_id=0)
        apply_and_verify(game_state, buy_id)

        assert MARKET.is_space_available(game_state, old_index)

    def test_buy_occupies_new_market_space(self, game_state):
        """New market space is occupied after buying (unless $75 sentinel)."""
        _make_trade_state(game_state, par_index=10)

        buy_id = find_legal_action(game_state, action_type=ACTION_BUY_SHARE, corp_id=0)
        apply_and_verify(game_state, buy_id)

        new_index = CORPS[0].get_price_index(game_state)
        max_index = int(GameConstants.NUM_MARKET_SPACES) - 1
        if new_index != max_index:
            assert not MARKET.is_space_available(game_state, new_index)

    def test_buy_skips_occupied_space(self, game_state):
        """Price moves past occupied spaces to the next available one."""
        _make_trade_state(game_state, par_index=10)
        MARKET.set_space_available(game_state, 11, False)

        buy_id = find_legal_action(game_state, action_type=ACTION_BUY_SHARE, corp_id=0)
        apply_and_verify(game_state, buy_id)

        assert CORPS[0].get_price_index(game_state) > 11

    def test_buy_increments_share_buys(self, game_state):
        """Buy increments the per-corp share_buys counter."""
        _make_trade_state(game_state)
        assert PLAYERS[0].get_share_buys(game_state, 0) == 0

        buy_id = find_legal_action(game_state, action_type=ACTION_BUY_SHARE, corp_id=0)
        apply_and_verify(game_state, buy_id)

        assert PLAYERS[0].get_share_buys(game_state, 0) == 1

    def test_buy_clears_consecutive_passes(self, game_state):
        """Buy resets the consecutive-passes counter."""
        _make_trade_state(game_state)
        # One pass to get counter > 0 (stays in INVEST regardless of player count)
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)
        assert TURN.get_consecutive_passes(game_state) == 1

        buy_id = find_legal_action(game_state, action_type=ACTION_BUY_SHARE, corp_id=0)
        apply_and_verify(game_state, buy_id)
        assert TURN.get_consecutive_passes(game_state) == 0

    def test_buy_at_75_triggers_game_over(self, game_state):
        """Landing on market index 26 ($75) ends the game."""
        _make_trade_state(game_state, par_index=25)
        PLAYERS[0].set_cash(game_state, 200)

        buy_id = find_legal_action(game_state, action_type=ACTION_BUY_SHARE, corp_id=0)
        apply_and_verify(game_state, buy_id, expected_status=STATUS_GAME_OVER)
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_GAME_OVER)

    def test_buy_blocked_when_no_bank_shares(self, game_state):
        """Cannot buy when no bank shares remain."""
        _make_trade_state(game_state)
        CORPS[0].set_bank_shares(game_state, 0)

        actions = get_legal_actions(game_state)
        buy_actions = [a for _, a in actions
                      if a.action_type == ACTION_BUY_SHARE and a.corp_id == 0]
        assert len(buy_actions) == 0

    def test_buy_blocked_when_cannot_afford(self, game_state):
        """Cannot buy when player cash is 0."""
        _make_trade_state(game_state, par_index=10)
        PLAYERS[0].set_cash(game_state, 0)

        actions = get_legal_actions(game_state)
        buy_actions = [a for _, a in actions
                      if a.action_type == ACTION_BUY_SHARE and a.corp_id == 0]
        assert len(buy_actions) == 0


# =============================================================================
# SELL SHARE TESTS
# =============================================================================

class TestSellShare:
    """Test INVEST sell-share action behavior."""

    def test_sell_transfers_share_to_bank(self, game_state):
        """Sell moves one share from player to bank."""
        _make_trade_state(game_state)
        initial_bank = CORPS[0].get_bank_shares(game_state)
        initial_player = PLAYERS[0].get_shares(game_state, 0)

        sell_id = find_legal_action(game_state, action_type=ACTION_SELL_SHARE, corp_id=0)
        apply_and_verify(game_state, sell_id)

        assert CORPS[0].get_bank_shares(game_state) == initial_bank + 1
        assert PLAYERS[0].get_shares(game_state, 0) == initial_player - 1

    def test_sell_pays_new_lower_price(self, game_state):
        """Player receives the new (lower) share price."""
        _make_trade_state(game_state, par_index=10)
        initial_cash = PLAYERS[0].get_cash(game_state)

        sell_id = find_legal_action(game_state, action_type=ACTION_SELL_SHARE, corp_id=0)
        apply_and_verify(game_state, sell_id)

        # After sell, corp.get_share_price() returns the new (lower) price
        sell_price = CORPS[0].get_share_price(game_state)
        assert PLAYERS[0].get_cash(game_state) == initial_cash + sell_price

    def test_sell_moves_price_down(self, game_state):
        """Sell moves corp price index to the next lower available space."""
        _make_trade_state(game_state, par_index=10)
        old_index = CORPS[0].get_price_index(game_state)

        sell_id = find_legal_action(game_state, action_type=ACTION_SELL_SHARE, corp_id=0)
        apply_and_verify(game_state, sell_id)

        assert CORPS[0].get_price_index(game_state) < old_index

    def test_sell_frees_old_market_space(self, game_state):
        """Old market space is freed after selling."""
        _make_trade_state(game_state, par_index=10)
        old_index = CORPS[0].get_price_index(game_state)
        assert not MARKET.is_space_available(game_state, old_index)

        sell_id = find_legal_action(game_state, action_type=ACTION_SELL_SHARE, corp_id=0)
        apply_and_verify(game_state, sell_id)

        assert MARKET.is_space_available(game_state, old_index)

    def test_sell_occupies_new_market_space(self, game_state):
        """New market space is occupied after selling."""
        _make_trade_state(game_state, par_index=10)

        sell_id = find_legal_action(game_state, action_type=ACTION_SELL_SHARE, corp_id=0)
        apply_and_verify(game_state, sell_id)

        new_index = CORPS[0].get_price_index(game_state)
        if new_index != 0:  # $0 bankruptcy space is always available
            assert not MARKET.is_space_available(game_state, new_index)

    def test_sell_skips_occupied_space(self, game_state):
        """Price drops past occupied spaces to the next available one."""
        _make_trade_state(game_state, par_index=10)
        MARKET.set_space_available(game_state, 9, False)

        sell_id = find_legal_action(game_state, action_type=ACTION_SELL_SHARE, corp_id=0)
        apply_and_verify(game_state, sell_id)

        assert CORPS[0].get_price_index(game_state) < 9

    def test_sell_increments_share_sells(self, game_state):
        """Sell increments the per-corp share_sells counter."""
        _make_trade_state(game_state)
        assert PLAYERS[0].get_share_sells(game_state, 0) == 0

        sell_id = find_legal_action(game_state, action_type=ACTION_SELL_SHARE, corp_id=0)
        apply_and_verify(game_state, sell_id)

        assert PLAYERS[0].get_share_sells(game_state, 0) == 1

    def test_sell_clears_consecutive_passes(self, game_state):
        """Sell resets the consecutive-passes counter."""
        # Float with enough shares so every player can hold one
        num_players = TURN.get_num_players(game_state)
        _make_trade_state(game_state, float_shares=num_players)
        for p in range(num_players):
            PLAYERS[p].set_shares(game_state, 0, 1)

        # One pass to get counter > 0 (safe for all player counts)
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)
        assert TURN.get_consecutive_passes(game_state) == 1

        sell_id = find_legal_action(game_state, action_type=ACTION_SELL_SHARE, corp_id=0)
        apply_and_verify(game_state, sell_id)
        assert TURN.get_consecutive_passes(game_state) == 0

    def test_sell_to_zero_triggers_bankruptcy(self, game_state):
        """Selling when the next lower space is $0 triggers bankruptcy."""
        _make_trade_state(game_state, par_index=1)

        sell_id = find_legal_action(game_state, action_type=ACTION_SELL_SHARE, corp_id=0)
        apply_and_verify(game_state, sell_id)

        assert not CORPS[0].is_active(game_state)

    def test_sell_last_share_triggers_receivership(self, game_state):
        """Selling the last player-held share puts the corp in receivership."""
        _make_trade_state(game_state, par_index=10, float_shares=1)
        assert PLAYERS[0].get_shares(game_state, 0) == 1

        sell_id = find_legal_action(game_state, action_type=ACTION_SELL_SHARE, corp_id=0)
        apply_and_verify(game_state, sell_id)

        assert CORPS[0].is_active(game_state)
        assert CORPS[0].is_in_receivership(game_state)

    def test_sell_blocked_when_no_shares(self, game_state):
        """Cannot sell shares in a corp the active player doesn't own."""
        _make_trade_state(game_state)
        TURN.set_active_player(game_state, 1)

        actions = get_legal_actions(game_state)
        sell_actions = [a for _, a in actions
                       if a.action_type == ACTION_SELL_SHARE and a.corp_id == 0]
        assert len(sell_actions) == 0


# =============================================================================
# ROUND-TRIP LIMIT TESTS
# =============================================================================

class TestRoundTripLimits:
    """Test the per-corp round-trip constraint (max 2 per INVEST phase)."""

    def test_blocked_after_two_roundtrips(self, game_state):
        """Both buy and sell are blocked after 2 complete round-trips."""
        _make_trade_state(game_state)
        for _ in range(2):
            PLAYERS[0].increment_share_buys(game_state, 0)
            PLAYERS[0].increment_share_sells(game_state, 0)

        actions = get_legal_actions(game_state)
        corp0_trades = [a for _, a in actions
                       if a.corp_id == 0 and a.action_type in (ACTION_BUY_SHARE, ACTION_SELL_SHARE)]
        assert len(corp0_trades) == 0

    def test_not_blocked_with_asymmetric_counts(self, game_state):
        """min(buys=1, sells=3) = 1 < 2, so trades are still allowed."""
        _make_trade_state(game_state)
        PLAYERS[0].increment_share_buys(game_state, 0)
        for _ in range(3):
            PLAYERS[0].increment_share_sells(game_state, 0)

        actions = get_legal_actions(game_state)
        corp0_trades = [a for _, a in actions
                       if a.corp_id == 0 and a.action_type in (ACTION_BUY_SHARE, ACTION_SELL_SHARE)]
        assert len(corp0_trades) > 0

    def test_separate_corps_have_independent_limits(self, game_state):
        """Round-trip limits are per-corp, not global."""
        _make_trade_state(game_state, corp_id=0, player_id=0, par_index=10)
        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)

        for _ in range(2):
            PLAYERS[0].increment_share_buys(game_state, 0)
            PLAYERS[0].increment_share_sells(game_state, 0)

        actions = get_legal_actions(game_state)
        corp0_trades = [a for _, a in actions
                       if a.corp_id == 0 and a.action_type in (ACTION_BUY_SHARE, ACTION_SELL_SHARE)]
        assert len(corp0_trades) == 0

        corp1_trades = [a for _, a in actions
                       if a.corp_id == 1 and a.action_type in (ACTION_BUY_SHARE, ACTION_SELL_SHARE)]
        assert len(corp1_trades) > 0


# =============================================================================
# ENUMERATION TESTS
# =============================================================================

class TestEnumeration:
    """Test legal-action enumeration for the INVEST phase."""

    def test_initial_state_has_only_pass_and_auctions(self, game_state):
        """Fresh game: no active corps, so only pass and auction actions exist."""
        actions = get_legal_actions(game_state)
        action_types = {info.action_type for _, info in actions}
        assert ACTION_PASS in action_types
        assert ACTION_AUCTION in action_types
        assert ACTION_BUY_SHARE not in action_types
        assert ACTION_SELL_SHARE not in action_types

    def test_auction_count_matches_auction_row(self, game_state):
        """Number of auctionable companies equals num_players in initial state."""
        num_players = TURN.get_num_players(game_state)
        actions = get_legal_actions(game_state)
        auction_companies = {info.company_id for _, info in actions
                            if info.action_type == ACTION_AUCTION}
        assert len(auction_companies) == num_players

    def test_floated_corp_enables_buy_and_sell(self, game_state):
        """Floating a corp makes buy and sell legal for the president."""
        _make_trade_state(game_state)

        actions = get_legal_actions(game_state)
        action_types = {info.action_type for _, info in actions}
        assert ACTION_BUY_SHARE in action_types
        assert ACTION_SELL_SHARE in action_types

    def test_inactive_corp_not_tradeable(self, game_state):
        """No buy/sell actions exist when no corps are active."""
        actions = get_legal_actions(game_state)
        trade_actions = [a for _, a in actions
                        if a.action_type in (ACTION_BUY_SHARE, ACTION_SELL_SHARE)]
        assert len(trade_actions) == 0

    def test_pass_always_legal(self, game_state):
        """Pass is always legal, even with 0 cash."""
        active = TURN.get_active_player(game_state)
        PLAYERS[active].set_cash(game_state, 0)

        actions = get_legal_actions(game_state)
        pass_actions = [a for _, a in actions if a.action_type == ACTION_PASS]
        assert len(pass_actions) == 1

    def test_buy_legal_at_exact_cost(self, game_state):
        """Buy is legal when player cash exactly covers the next higher price."""
        _make_trade_state(game_state, par_index=10)
        next_index = MARKET.find_next_higher_space(game_state, 10)
        next_price = MARKET.get_price_at_index(next_index)

        PLAYERS[0].set_cash(game_state, next_price)

        actions = get_legal_actions(game_state)
        buy_actions = [a for _, a in actions
                      if a.action_type == ACTION_BUY_SHARE and a.corp_id == 0]
        assert len(buy_actions) == 1

    def test_buy_blocked_one_below_cost(self, game_state):
        """Buy is blocked when player cash is one below the next higher price."""
        _make_trade_state(game_state, par_index=10)
        next_index = MARKET.find_next_higher_space(game_state, 10)
        next_price = MARKET.get_price_at_index(next_index)

        PLAYERS[0].set_cash(game_state, next_price - 1)

        actions = get_legal_actions(game_state)
        buy_actions = [a for _, a in actions
                      if a.action_type == ACTION_BUY_SHARE and a.corp_id == 0]
        assert len(buy_actions) == 0


# =============================================================================
# PHASE TRANSITION TESTS
# =============================================================================

class TestPhaseTransitions:
    """Test phase transitions from INVEST."""

    def test_full_pass_cycle_returns_to_invest(self, game_state):
        """All passes chain through WRAP_UP and back to INVEST (next turn)."""
        num_players = TURN.get_num_players(game_state)
        initial_turn = TURN.get_turn_number(game_state)

        for _ in range(num_players):
            pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, pass_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)
        assert TURN.get_turn_number(game_state) == initial_turn + 1

    def test_auction_returns_to_invest_after_resolution(self, game_state):
        """Auction -> BID -> starter bids, others leave -> back to INVEST."""
        from core.actions import ACTION_RAISE_PY as ACTION_RAISE
        num_players = TURN.get_num_players(game_state)
        action_id = find_legal_action(game_state, action_type=ACTION_AUCTION)
        apply_and_verify(game_state, action_id)
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_BID)

        # Starter places the opening bid at face_value (offset 0). Pass is
        # not legal on the opening bid.
        opening = find_legal_action(game_state, action_type=ACTION_RAISE, amount=0)
        apply_and_verify(game_state, opening)

        # All other players leave the auction (pass in BID = leave).
        for _ in range(num_players - 1):
            leave_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, leave_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)
