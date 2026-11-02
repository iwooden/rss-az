"""The private auction: AUCTION and BID, a closed auction's included
(1889/engine-design.md §5.2, §6.1; Ruby's ``step/waterfall_auction.rb``).

Covers purchases of the topmost private, bids and their cash limits,
resolution chains, closed auctions, rounds of passes (A's discount down to the
free A, and payouts), priority, the pass cap, and the policy against the
engine-legal sets. ``tests/g1889/replay/test_auction_scripts.py`` checks the
same paths against Ruby.
"""

import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.bank import BANK
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES, topmost
from g1889.entities.turn import TURN

from tests.g1889.phases.conftest import (
    apply_and_verify, enter_auction, exact_id, find_legal_action, get_legal_actions, legal_ids, new_game,
    policy_ids,
)

TR, MF, ER, SMR, DR, SIR, UTF = range(D.NUM_PRIVATES)


# =============================================================================
# HELPERS
# =============================================================================

def buy(state):
    return apply_and_verify(state, K.AUCTION_BUY, 'buy the topmost')


def pass_turn(state):
    """Pass in AUCTION, or drop out of a closed auction in BID."""
    assert TURN.step(state) == K.STEP_AUCTION or TURN.auction_private(state) >= 0
    return apply_and_verify(state, 0, 'pass')


def passes(state, count):
    for _ in range(count):
        pass_turn(state)


def _amount_id(state, amount):
    """BID's grid point for ``amount`` when it is one, else its exact id."""
    p = TURN.auction_private(state) if TURN.auction_private(state) >= 0 else TURN.pending_private(state)
    k, off = divmod(amount - PRIVATES[p].min_bid(state), D.MIN_BID_INCREMENT)
    return K.BID_POINT + k if off == 0 and 0 <= k < K.BID_POINTS else exact_id(state, amount)


def bid(state, private, amount):
    """Select ``private`` in AUCTION and bid ``amount`` in BID (or let the
    driver bid when the minimum is the only point)."""
    player = TURN.active_player(state)
    result = apply_and_verify(state, K.AUCTION_BID + private, f'select {private}')
    if not result.forced:
        result = apply_and_verify(state, _amount_id(state, amount), f'bid {amount}')
    assert PLAYERS[player].bid(state, private) == amount
    return result


def raise_to(state, amount):
    """Raise in a closed auction."""
    assert TURN.step(state) == K.STEP_BID and TURN.auction_private(state) >= 0
    return apply_and_verify(state, _amount_id(state, amount), f'raise to {amount}')


def cash(state):
    return [PLAYERS[i].cash(state) for i in range(TURN.num_players(state))]


def owner(state, private):
    return PRIVATES[private].owner(state)


def bids_on(state, private):
    bids = (PLAYERS[i].bid(state, private) for i in range(TURN.num_players(state)))
    return {i: amount for i, amount in enumerate(bids) if amount}


@pytest.fixture
def state3():
    return new_game(3)


# =============================================================================
# THE OPENING
# =============================================================================

class TestOpening:
    def test_first_seat_opens_with_every_private_but_a_biddable(self, game_state):
        n = TURN.num_players(game_state)
        in_play = [p for p in range(D.NUM_PRIVATES) if owner(game_state, p) == K.OWNER_UNSOLD]
        assert in_play == list(range(5 if n == 2 else 6 if n == 3 else 7))
        assert TURN.active_player(game_state) == TURN.auction_turn(game_state) == 0
        assert TURN.priority_player(game_state) == 0
        assert legal_ids(game_state) == [K.AUCTION_PASS, K.AUCTION_BUY] + [K.AUCTION_BID + p for p in in_play]
        # Bidding on the topmost buys it at the bid; buying at its price dominates (§5.1).
        assert policy_ids(game_state) == [K.AUCTION_PASS, K.AUCTION_BUY] + [K.AUCTION_BID + p for p in in_play[1:]]


# =============================================================================
# BUYING THE TOPMOST
# =============================================================================

class TestBuyTopmost:
    def test_buy_pays_the_price_and_moves_priority_and_turn(self, game_state):
        n, bank = TURN.num_players(game_state), BANK.cash(game_state)
        before = cash(game_state)
        buy(game_state)
        assert owner(game_state, TR) == K.OWNER_PLAYER and topmost(game_state) == MF
        assert cash(game_state) == [before[0] - 20] + before[1:] and BANK.cash(game_state) == bank + 20
        assert TURN.step(game_state) == K.STEP_AUCTION
        assert TURN.active_player(game_state) == TURN.auction_turn(game_state) == 1 % n
        assert TURN.priority_player(game_state) == 1 % n

    def test_the_next_topmost_sells_at_face(self, state3):
        buy(state3)
        buy(state3)
        assert owner(state3, MF) == K.OWNER_PLAYER + 1 and cash(state3) == [400, 390, 420]

    def test_bid_on_the_topmost_buys_it_at_the_bid(self, state3):
        """Engine-legal but outside the policy: Ruby takes a bid on the topmost as a purchase."""
        apply_and_verify(state3, find_legal_action(state3, type=A.ACTION_SELECT_BID, private=TR))
        assert TURN.step(state3) == K.STEP_BID and PRIVATES[TR].min_bid(state3) == 20
        apply_and_verify(state3, K.BID_POINT + 2)
        assert owner(state3, TR) == K.OWNER_PLAYER and cash(state3)[0] == 390
        assert TURN.priority_player(state3) == 1 and bids_on(state3, TR) == {}

    def test_buy_needs_cash_beyond_other_bids(self, state3):
        """¥405 on SIR leaves ¥15: neither A nor a new bid is affordable, but raising SIR is."""
        bid(state3, SIR, 405)
        passes(state3, 2)
        assert legal_ids(state3) == [K.AUCTION_PASS, K.AUCTION_BID + SIR]

    def test_a_player_with_only_the_pass_is_passed_by_the_driver(self, state3):
        """Here the forced pass completes a round of passes."""
        bid(state3, SIR, 420)
        pass_turn(state3)
        result = pass_turn(state3)
        assert result.forced == [(K.DPHASE_AUCTION, K.AUCTION_PASS)]
        assert TURN.auction_discount(state3) == 5 and TURN.active_player(state3) == 1


# =============================================================================
# BIDS
# =============================================================================

class TestBids:
    def test_bid_holds_cash_without_paying(self, state3):
        bid(state3, ER, 55)
        assert bids_on(state3, ER) == {0: 55} and cash(state3) == [420] * 3
        assert PLAYERS[0].available(state3) == 365 and TURN.active_player(state3) == 1
        assert PRIVATES[ER].min_bid(state3) == 60

    def test_bid_grid_starts_at_the_minimum(self, state3):
        bid(state3, ER, 45)
        apply_and_verify(state3, K.AUCTION_BID + ER)
        infos = [info for _, info in get_legal_actions(state3)]
        assert [i.count for i in infos] == list(range(K.BID_POINTS))  # ¥50, ¥55, ... ¥125
        assert all(i.type == A.ACTION_PRICE_POINT for i in infos)
        apply_and_verify(state3, K.BID_POINT + 3)
        assert bids_on(state3, ER) == {0: 45, 1: 65}

    def test_bid_grid_and_range_stop_at_available_cash(self, state3):
        """Cash less the player's other bids caps a bid: ¥70 after ¥350 on SIR."""
        bid(state3, SIR, 350)
        passes(state3, 2)
        apply_and_verify(state3, K.AUCTION_BID + ER)
        assert legal_ids(state3) == [K.BID_POINT + k for k in range(6)]  # ¥45-70
        assert A.is_legal_action_py(state3, exact_id(state3, 70))
        assert not A.is_legal_action_py(state3, exact_id(state3, 71))
        assert not A.is_legal_action_py(state3, exact_id(state3, 44))

    def test_a_private_without_an_affordable_bid_cannot_be_selected(self, state3):
        bid(state3, SIR, 385)
        passes(state3, 2)
        assert K.AUCTION_BID + MF in legal_ids(state3)  # ¥35 of ¥35
        assert K.AUCTION_BID + ER not in legal_ids(state3)

    def test_raising_an_own_bid_reuses_its_cash(self, state3):
        bid(state3, SIR, 400)
        passes(state3, 2)
        bid(state3, SIR, 420)
        assert bids_on(state3, SIR) == {0: 420} and PLAYERS[0].available(state3) == 0

    def test_off_grid_amounts_are_exact_ids(self, state3):
        apply_and_verify(state3, K.AUCTION_BID + ER)
        apply_and_verify(state3, exact_id(state3, 47))
        assert bids_on(state3, ER) == {0: 47} and PRIVATES[ER].min_bid(state3) == 52

    def test_a_single_point_bid_is_forced(self, state3):
        """With cash for only the minimum, BID has one policy point, which the driver applies."""
        bid(state3, SIR, 375)
        passes(state3, 2)
        result = apply_and_verify(state3, K.AUCTION_BID + ER)
        assert result.forced == [(K.DPHASE_BID, K.BID_POINT)]
        assert bids_on(state3, ER) == {0: 45} and TURN.step(state3) == K.STEP_AUCTION

    def test_a_bid_clears_only_the_bidders_pass(self, state3):
        pass_turn(state3)
        bid(state3, MF, 35)
        assert [PLAYERS[i].passed(state3) for i in range(3)] == [True, False, False]
        pass_turn(state3)
        bid(state3, MF, 40)
        assert [PLAYERS[i].passed(state3) for i in range(3)] == [False, False, True]


# =============================================================================
# RESOLUTION CHAINS
# =============================================================================

class TestResolutionChain:
    def test_single_bids_sell_in_a_chain(self, state3):
        bid(state3, MF, 35)
        bid(state3, ER, 45)
        buy(state3)
        assert [owner(state3, p) for p in (TR, MF, ER)] == [K.OWNER_PLAYER + 2, K.OWNER_PLAYER, K.OWNER_PLAYER + 1]
        assert cash(state3) == [385, 375, 400] and topmost(state3) == SMR
        assert all(PLAYERS[i].committed(state3) == 0 for i in range(3))
        # A sale by resolution leaves priority with the player after the buyer.
        assert TURN.step(state3) == K.STEP_AUCTION
        assert TURN.active_player(state3) == TURN.priority_player(state3) == 0

    def test_a_chain_stops_at_a_private_without_bids(self, state3):
        bid(state3, ER, 45)
        buy(state3)
        assert topmost(state3) == MF and bids_on(state3, ER) == {0: 45}

    def test_a_chain_that_sells_everything_opens_the_stock_round(self):
        state = new_game(2)
        enter_auction(state, 1, bids=[(0, MF, 35), (1, ER, 45), (0, SMR, 55), (1, DR, 65)])
        pass_turn(state)
        buy(state)
        assert [owner(state, p) for p in range(5)] == [0, 0, 1, 0, 1]
        assert (TURN.round(state), TURN.step(state)) == (K.ROUND_STOCK, K.STEP_STOCK)
        # The turn player after the purchase opens the SR and holds priority.
        assert TURN.active_player(state) == TURN.priority_player(state) == TURN.auction_turn(state) == 1
        assert not any(PLAYERS[i].passed(state) for i in range(2))


# =============================================================================
# CLOSED AUCTIONS
# =============================================================================

@pytest.fixture
def closed4():
    """Four players: MF has bids of ¥35 (0), ¥40 (1), and ¥45 (2); player 3 buys A."""
    state = new_game(4)
    bid(state, MF, 35)
    bid(state, MF, 40)
    bid(state, MF, 45)
    buy(state)
    return state


class TestClosedAuction:
    def test_several_bids_open_a_closed_auction_led_by_the_lowest_bidder(self, closed4):
        """Ruby: the lowest bidder acts (the rulebook goes clockwise from the high bidder)."""
        assert (TURN.step(closed4), TURN.auction_private(closed4)) == (K.STEP_BID, MF)
        assert TURN.active_player(closed4) == 0
        assert TURN.auction_turn(closed4) == TURN.priority_player(closed4) == 0
        assert PRIVATES[MF].min_bid(closed4) == 50
        assert legal_ids(closed4) == [K.BID_DROP] + [K.BID_POINT + k for k in range(K.BID_POINTS)]
        assert policy_ids(closed4) == legal_ids(closed4)

    def test_a_raise_hands_the_turn_to_the_new_lowest_bidder(self, closed4):
        raise_to(closed4, 55)
        assert bids_on(closed4, MF) == {0: 55, 1: 40, 2: 45} and TURN.active_player(closed4) == 1
        raise_to(closed4, 62)  # off the grid
        assert TURN.active_player(closed4) == 2 and PRIVATES[MF].min_bid(closed4) == 67

    def test_dropping_out_loses_the_bid_and_the_last_bidder_buys(self, closed4):
        pass_turn(closed4)
        assert bids_on(closed4, MF) == {1: 40, 2: 45} and TURN.active_player(closed4) == 1
        pass_turn(closed4)
        assert owner(closed4, MF) == K.OWNER_PLAYER + 2 and cash(closed4) == [420, 420, 375, 400]
        # Normal turns resume where they were: the closed auction leaves the turn alone.
        assert (TURN.step(closed4), TURN.auction_private(closed4)) == (K.STEP_AUCTION, -1)
        assert TURN.active_player(closed4) == 0

    @staticmethod
    def _closed_after_passes():
        """MF bids of ¥35 (0) and ¥40 (1); players 2 and 0 pass; player 1 buys A."""
        state = new_game(3)
        bid(state, MF, 35)
        bid(state, MF, 40)
        passes(state, 2)
        buy(state)
        assert TURN.active_player(state) == 0 and PLAYERS[0].passed(state) and PLAYERS[2].passed(state)
        return state

    def test_a_raise_clears_the_raisers_pass(self):
        state = self._closed_after_passes()
        raise_to(state, 45)
        assert not PLAYERS[0].passed(state) and PLAYERS[2].passed(state)

    def test_a_drop_keeps_the_droppers_pass(self):
        state = self._closed_after_passes()
        pass_turn(state)
        assert owner(state, MF) == K.OWNER_PLAYER + 1 and PLAYERS[0].passed(state)

    def test_a_bidder_who_cannot_raise_drops_out_by_force(self):
        """¥360 on SIR caps player 0's MF bid at ¥60; player 1 raises to ¥60."""
        state = new_game(3)
        bid(state, SIR, 360)
        bid(state, MF, 35)
        pass_turn(state)
        bid(state, MF, 40)
        buy(state)
        assert (TURN.step(state), TURN.active_player(state)) == (K.STEP_BID, 1)
        result = raise_to(state, 60)
        assert result.forced == [(K.DPHASE_BID, K.BID_DROP)]
        assert owner(state, MF) == K.OWNER_PLAYER + 1 and cash(state)[1] == 420 - 20 - 60

    def test_a_closed_auction_chains_into_the_next(self):
        state = new_game(3)
        enter_auction(state, 0, sold=[(TR, 2, 20)],
                      bids=[(0, MF, 35), (1, MF, 40), (2, ER, 45), (0, SMR, 55), (2, SMR, 60)])
        assert (TURN.auction_private(state), TURN.active_player(state)) == (MF, 0)
        pass_turn(state)  # MF to player 1; ER to player 2; SMR has two bids
        assert [owner(state, p) for p in (MF, ER)] == [K.OWNER_PLAYER + 1, K.OWNER_PLAYER + 2]
        assert (TURN.auction_private(state), TURN.active_player(state)) == (SMR, 0)


# =============================================================================
# ROUNDS OF PASSES
# =============================================================================

class TestAllPassed:
    def test_a_round_of_passes_takes_five_off_a(self, game_state):
        n = TURN.num_players(game_state)
        passes(game_state, n)
        assert TURN.auction_discount(game_state) == K.AUCTION_DISCOUNT_STEP == 5
        assert PRIVATES[TR].price(game_state) == 15 and PRIVATES[MF].price(game_state) == 30
        # Play continues with the player after the last passer; the passes clear.
        assert TURN.active_player(game_state) == 0 and not any(PLAYERS[i].passed(game_state) for i in range(n))
        assert TURN.auction_pass_payouts(game_state) == 0

    def test_a_reaching_zero_goes_free_to_the_player_after_the_last_passer(self, state3):
        """Not to the auction's first player, as the rulebook has it: rounds of
        passes from player 1 end with player 0. Ruby moves the turn to A's
        taker, then on again; priority stays with the first seat."""
        bid(state3, MF, 35)
        bank = BANK.cash(state3)
        passes(state3, 3 * 3)
        assert TURN.auction_discount(state3) == 15 and TURN.active_player(state3) == 1
        passes(state3, 3)
        assert owner(state3, TR) == K.OWNER_PLAYER + 1 and TURN.auction_discount(state3) == 20
        assert owner(state3, MF) == K.OWNER_PLAYER and BANK.cash(state3) == bank + 35  # A was free
        assert TURN.active_player(state3) == 2 and TURN.priority_player(state3) == 0 and topmost(state3) == ER

    def test_the_free_a_resolves_the_next_private(self):
        state = new_game(2)
        bid(state, MF, 35)
        bid(state, MF, 40)
        passes(state, 8)
        assert owner(state, TR) == K.OWNER_PLAYER  # the player after the last passer (1)
        assert (TURN.step(state), TURN.auction_private(state), TURN.active_player(state)) == (K.STEP_BID, MF, 0)
        assert TURN.auction_turn(state) == 1

    def test_a_free_a_whose_chain_sells_everything_ends_the_auction(self):
        """No purchase was made, yet priority leaves the first seat: it goes to the
        player after A's taker (``reorder_players``)."""
        state = new_game(2)
        enter_auction(state, 0, bids=[(0, MF, 35), (1, ER, 45), (0, SMR, 55), (1, DR, 65)])
        assert TURN.priority_player(state) == 0
        passes(state, 8)
        assert owner(state, TR) == K.OWNER_PLAYER and TURN.round(state) == K.ROUND_STOCK
        assert TURN.priority_player(state) == TURN.active_player(state) == 1

    def test_passes_with_a_sold_pay_revenue_to_owners(self, state3):
        buy(state3)
        buy(state3)
        bid(state3, SIR, 85)
        before, bank = cash(state3), BANK.cash(state3)
        passes(state3, 3)
        assert cash(state3) == [before[0] + 5, before[1] + 5, before[2]] and BANK.cash(state3) == bank - 10
        assert TURN.auction_pass_payouts(state3) == 1 and PRIVATES[ER].price(state3) == 40
        assert TURN.auction_discount(state3) == 0 and TURN.active_player(state3) == 0
        passes(state3, 3)
        assert TURN.auction_pass_payouts(state3) == 2 and cash(state3)[0] == before[0] + 10

    def test_passes_persist_across_other_players_turns(self, state3):
        """A round of passes is every flag set, not n passes in a row: a purchase
        clears only the buyer's flag (``process_bid``)."""
        pass_turn(state3)
        buy(state3)
        pass_turn(state3)
        assert [PLAYERS[i].passed(state3) for i in range(3)] == [True, False, True]
        bank = BANK.cash(state3)
        pass_turn(state3)  # player 0 again
        pass_turn(state3)  # player 1: everyone has passed
        assert BANK.cash(state3) == bank - 5 and TURN.auction_pass_payouts(state3) == 1


# =============================================================================
# PRIORITY
# =============================================================================

class TestPriority:
    def test_priority_follows_the_last_turn_purchase_only(self, state3):
        """Bids and sales by resolution leave it alone."""
        pass_turn(state3)
        bid(state3, MF, 35)
        assert TURN.priority_player(state3) == 0
        buy(state3)  # player 2 buys A; MF sells to player 1 by resolution
        assert owner(state3, MF) == K.OWNER_PLAYER + 1 and TURN.priority_player(state3) == 0
        buy(state3)  # player 0 buys ER
        assert TURN.priority_player(state3) == 1
        pass_turn(state3)
        assert TURN.priority_player(state3) == 1

    def test_priority_survives_padding(self):
        state = new_game(3, max_players=6)
        buy(state)
        assert TURN.priority_player(state) == 1 and state.max_players == 6


# =============================================================================
# THE PASS CAP (policy only)
# =============================================================================

def _after_payouts(cap, payouts):
    """Three players, A sold to player 0, then ``payouts`` rounds of passes."""
    state = new_game(3, auction_pass_payout_cap=cap)
    buy(state)
    passes(state, 3 * payouts)
    return state


class TestPassCap:
    def test_the_cap_removes_the_pass_of_a_player_who_can_buy(self):
        state = _after_payouts(1, 1)
        assert K.AUCTION_PASS in legal_ids(state) and K.AUCTION_PASS not in policy_ids(state)
        assert K.AUCTION_BUY in policy_ids(state)
        apply_and_verify(state, K.AUCTION_PASS)  # the engine still takes it

    def test_the_pass_stays_before_the_cap(self):
        assert K.AUCTION_PASS in policy_ids(_after_payouts(1, 0))
        assert K.AUCTION_PASS in policy_ids(_after_payouts(2, 1))
        assert K.AUCTION_PASS not in policy_ids(_after_payouts(2, 2))

    def test_the_cap_off_keeps_the_pass(self):
        state = _after_payouts(0, 2)
        assert policy_ids(state) == [i for i in legal_ids(state) if i != K.AUCTION_BID + MF]

    def test_a_player_who_cannot_buy_may_only_pass(self):
        """¥400 on SIR leaves player 0 ¥20, short of MF's ¥30. Once capped, raising
        SIR would keep them short, so the pass is all the policy has."""
        state = new_game(3, auction_pass_payout_cap=1)
        bid(state, SIR, 400)
        buy(state)
        passes(state, 3)  # players 2, 0, and 1: a payout
        result = bid(state, ER, 45)  # player 2, whose pass the cap removed
        assert TURN.auction_pass_payouts(state) == 1
        assert result.forced == [(K.DPHASE_AUCTION, K.AUCTION_PASS)] and TURN.active_player(state) == 1
        assert PLAYERS[0].passed(state)

    def test_the_cap_leaves_a_closed_auction_alone(self):
        """Its private is the topmost, on which any raise is a purchase, and
        dropping out pays nobody: the policy keeps the drop and every raise."""
        state = _after_payouts(1, 1)  # player 1's turn
        bid(state, ER, 45)
        bid(state, ER, 50)
        buy(state)  # player 0 buys MF: ER's two bids close its auction
        assert (TURN.step(state), TURN.auction_private(state), TURN.active_player(state)) == (K.STEP_BID, ER, 1)
        assert policy_ids(state) == legal_ids(state) == [K.BID_DROP] + [K.BID_POINT + k for k in range(K.BID_POINTS)]

    @staticmethod
    def _capped_with_cash_on_sir():
        """Player 1 holds ¥350 on SIR, leaving ¥70 against MF's ¥30, after a payout."""
        state = new_game(3, auction_pass_payout_cap=1)
        buy(state)
        bid(state, SIR, 350)
        passes(state, 3)
        bid(state, ER, 45)
        bid(state, DR, 65)
        assert TURN.active_player(state) == 1 and TURN.auction_pass_payouts(state) == 1
        return state

    def test_a_capped_bid_must_leave_the_topmost_affordable(self):
        """Without this, players who outbid themselves on later privates never
        afford the topmost and the payouts go on forever (rules §12.9)."""
        state = self._capped_with_cash_on_sir()
        bids = [K.AUCTION_BID + p for p in (MF, ER, SMR, DR, SIR)]
        assert legal_ids(state) == [K.AUCTION_PASS, K.AUCTION_BUY] + bids
        assert policy_ids(state) == [K.AUCTION_BUY, K.AUCTION_BID + SIR]  # ER ¥50, SMR ¥55, DR ¥70 > ¥40
        apply_and_verify(state, K.AUCTION_BID + SIR)
        assert legal_ids(state) == [K.BID_POINT + k for k in range(14)]  # ¥355-420
        assert policy_ids(state) == [K.BID_POINT + k for k in range(8)]  # ¥355-390, which leave ¥30

    def test_a_bid_selected_outside_the_policy_keeps_its_points(self):
        state = self._capped_with_cash_on_sir()
        apply_and_verify(state, K.AUCTION_BID + ER)
        assert policy_ids(state) == legal_ids(state) == [K.BID_POINT + k for k in range(5)]  # ¥50-70

    @pytest.mark.parametrize('cap', [0, 1])
    def test_a_player_left_only_the_buy_is_made_to_buy(self, cap):
        """Player 1 outbids player 0's ¥385 on SIR with all of its cash, leaving
        player 0 ¥35: MF's ¥30 and nothing else. Once capped, the driver buys MF."""
        state = new_game(3, auction_pass_payout_cap=cap)
        bid(state, SIR, 385)
        bid(state, SIR, 420)
        buy(state)
        result = pass_turn(state)  # player 0; player 1 has only the pass
        assert result.forced == [(K.DPHASE_AUCTION, K.AUCTION_PASS)]
        result = pass_turn(state)  # player 2 completes the round: a payout
        assert TURN.auction_pass_payouts(state) == 1
        if cap:
            assert result.forced == [(K.DPHASE_AUCTION, K.AUCTION_BUY), (K.DPHASE_AUCTION, K.AUCTION_PASS)]
            assert owner(state, MF) == K.OWNER_PLAYER and TURN.active_player(state) == 2
        else:
            assert result.forced == [] and TURN.active_player(state) == 0
            assert policy_ids(state) == [K.AUCTION_PASS, K.AUCTION_BUY]


# =============================================================================
# POLICY AND ENGINE SETS
# =============================================================================

class TestPolicyAgainstEngine:
    def test_bid_policies_are_the_grid(self, closed4):
        assert policy_ids(closed4) == legal_ids(closed4)
        state = new_game(3)
        apply_and_verify(state, K.AUCTION_BID + ER)
        assert policy_ids(state) == legal_ids(state) == [K.BID_POINT + k for k in range(K.BID_POINTS)]

    def test_exact_amounts_are_legal_but_never_enumerated(self, closed4):
        size = K.ACTION_SIZE_BID
        assert all(i < size for i in legal_ids(closed4))
        assert A.is_legal_action_py(closed4, size + 51) and not A.is_legal_action_py(closed4, size + 49)
        assert A.is_legal_action_py(closed4, size + 420) and not A.is_legal_action_py(closed4, size + 421)

    def test_policy_drops_only_the_topmost_bid_without_the_cap(self, game_state):
        legal, policy = legal_ids(game_state), policy_ids(game_state)
        assert [i for i in legal if i not in policy] == [K.AUCTION_BID + topmost(game_state)]
