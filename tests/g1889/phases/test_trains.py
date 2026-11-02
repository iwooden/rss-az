"""DIVIDEND, BUY_TRAIN, TRAIN_PRICE, DISCARD, and the phases, as 6.7 builds
them (1889/engine-design.md §4-6; rules §8.7-8.9, §9; Ruby's
``step/dividend.rb``, ``step/buy_train.rb``, ``step/train.rb``,
``step/discard_train.rb``, ``depot.rb``, and ``phase.rb``).

Covers the payout and withhold amounts, the price moves at the market's
edges, the bank breaking on a payout; each BUY_TRAIN source (the depot's
lowest rank and D from phase 6, the pool, other corps) with the train limit
and the treasury; the must-buy without a pass, funded from the treasury,
and its emergency stop; TRAIN_PRICE's grid and exact amounts; the D
trade-in; each phase's effects (rusting, the pool's too; the limit; the
privates' closing, UTF's; D's availability); and the discards' order.
Everything runs through the driver, whose ``apply_and_verify`` also checks
that every train stays accounted for. ``tests/g1889/replay/`` checks the
same rules against Ruby at every fixture BuyTrain and DiscardTrain snapshot.

KO's home, Takamatsu (K4), has printed track to J3; J3's tile 8 at rotation
3 joins it to Sakaide & Okayama, a route of ¥50 for a 2-train in phase 2.
"""

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_INVALID
from g1889.entities.bank import BANK, BANK_ACCOUNT
from g1889.entities.corp import CORPS
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN

from tests.g1889.phases.conftest import (
    apply_and_verify, at_development_stop, at_or, enter_operating_round, enter_phase, exact_id, give_train, hold,
    lay, legal_ids, policy_ids, set_cash,
)

AR, IR, SR, KO, TR, KU, UR = range(D.NUM_CORPS)
PAR_65, PAR_100 = 0, 5
T2, T3, T4, T5, T6, TD = range(D.NUM_TRAIN_TYPES)
PHASE_3, PHASE_4, PHASE_5, PHASE_6, PHASE_D = D.GAME_PHASE_3, D.GAME_PHASE_4, D.GAME_PHASE_5, D.GAME_PHASE_6, \
    D.GAME_PHASE_D


def depot(t):
    return K.BUY_TRAIN_DEPOT + t


def pool(t):
    return K.BUY_TRAIN_POOL + t


def from_corp(c, t):
    return K.BUY_TRAIN_CORP + D.NUM_TRAIN_TYPES * c + t


def trade(t):
    return K.BUY_TRAIN_TRADE_IN + D.TRADE_IN_TYPE.index(t)


def private_buys(state):
    """BUY_TRAIN's buy-private slots: in phases 3-4 the corp may buy privates too."""
    return [K.BUY_TRAIN_BUY_PRIVATE + p for p in A.purchasable_privates_py(state)]


def trains_of(state, corp):
    return {D.TRAIN_NAMES[t]: n for t in range(D.NUM_TRAIN_TYPES) if (n := TRAINS.corp_trains(state, corp, t))}


def to_buy_train(state):
    """Pass every TRACK and withhold until the first corp's BUY_TRAIN."""
    while TURN.step(state) != K.STEP_BUY_TRAIN:
        apply_and_verify(state, K.DIVIDEND_WITHHOLD if TURN.step(state) == K.STEP_DIVIDEND else 0)


# =============================================================================
# DIVIDEND
# =============================================================================

def with_revenue(cell=None, pool_shares=1, num_players=3):
    """KO earns ¥50 with a 2-train: player 0 holds 6 shares (president), player
    1 two, the pool ``pool_shares``, and the IPO the rest. DIVIDEND waits."""
    state = at_or(num_players)
    hold(state, KO, {0: 6, 1: 2}, cell=cell, pool=pool_shares)
    give_train(state, KO, T2)
    lay(state, 'J3', '8', 3)
    enter_operating_round(state)
    apply_and_verify(state, K.TRACK_PASS)
    assert TURN.step(state) == K.STEP_DIVIDEND and TURN.route_revenue(state) == 50
    return state


class TestDividend:
    def test_a_payout_pays_each_share_a_tenth(self):
        """Players by their shares (the certificate counts 2), the corp for its
        Open Market share, nobody for the IPO's (rules §8.7)."""
        state = with_revenue()
        players, ko, bank = [PLAYERS[i].cash(state) for i in range(3)], CORPS[KO].cash(state), BANK.cash(state)
        col = MARKET.col(state, KO)
        apply_and_verify(state, K.DIVIDEND_PAYOUT)
        assert [PLAYERS[i].cash(state) - players[i] for i in range(3)] == [30, 10, 0]
        assert CORPS[KO].cash(state) - ko == 5 and bank - BANK.cash(state) == 45  # the IPO share's ¥5 stays
        assert MARKET.col(state, KO) == col + 1 and CORPS[KO].last_revenue(state) == 50
        assert TURN.route_revenue(state) == 0 and TURN.step(state) == K.STEP_BUY_TRAIN

    def test_a_withhold_pays_the_treasury(self):
        state = with_revenue()
        players, ko, col = [PLAYERS[i].cash(state) for i in range(3)], CORPS[KO].cash(state), MARKET.col(state, KO)
        apply_and_verify(state, K.DIVIDEND_WITHHOLD)
        assert [PLAYERS[i].cash(state) for i in range(3)] == players and CORPS[KO].cash(state) == ko + 50
        assert MARKET.col(state, KO) == col - 1 and CORPS[KO].last_revenue(state) == 50

    @pytest.mark.parametrize('cell, action, moved', [
        ((2, 11), K.DIVIDEND_PAYOUT, (1, 11)),    # right at a row's end goes up
        ((0, 14), K.DIVIDEND_PAYOUT, (0, 14)),    # ¥350 stays
        ((3, 0), K.DIVIDEND_WITHHOLD, (4, 0)),    # left at column 0 goes down
        ((10, 0), K.DIVIDEND_WITHHOLD, (10, 0)),  # ¥10 stays
    ])
    def test_the_price_moves_with_the_edge_rules(self, cell, action, moved):
        state = with_revenue(cell=cell)
        apply_and_verify(state, action)
        assert (MARKET.row(state, KO), MARKET.col(state, KO)) == moved

    def test_a_payout_the_bank_cannot_cover_breaks_it(self):
        """The bank pays in full and breaks (``bank.rb``); the game then ends
        after the OR set, which 6.8b handles."""
        state = with_revenue()
        BANK.transfer(state, BANK_ACCOUNT, K.OWNER_PLAYER + 2, BANK.cash(state) - 20)
        players = [PLAYERS[i].cash(state) for i in range(3)]
        apply_and_verify(state, K.DIVIDEND_PAYOUT)
        assert BANK.broken(state) and BANK.cash(state) == 20 - 45
        assert [PLAYERS[i].cash(state) - players[i] for i in range(3)] == [30, 10, 0]


# =============================================================================
# BUY_TRAIN
# =============================================================================

class TestBuyTrain:
    def test_the_depot_sells_its_lowest_rank_one_at_a_time(self):
        """KO buys 2s until it passes or reaches the limit, then may only pass."""
        state = at_or(3)
        hold(state, KO, {0: 6}, par=PAR_100)  # ¥1,000
        enter_operating_round(state)
        to_buy_train(state)
        cash, results = CORPS[KO].cash(state), []
        for n in range(1, D.PHASE_TRAIN_LIMIT[0] + 1):
            assert legal_ids(state) == [K.BUY_TRAIN_PASS, depot(T2)]
            results.append(apply_and_verify(state, depot(T2)))
            assert TRAINS.corp_trains(state, KO, T2) == n and TRAINS.depot(state, T2) == D.TRAIN_COUNT[T2] - n
        assert CORPS[KO].cash(state) == cash - 4 * 80 and TRAINS.corp_total(state, KO) == TRAINS.limit(state)
        assert results[-1].forced == [(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_PASS)]  # at the limit: nothing left to buy

    def test_the_depot_sells_d_from_phase_6_and_the_treasury_must_afford_it(self):
        state = at_or(3)
        enter_phase(state, PHASE_6)
        hold(state, KO, {0: 6}, par=PAR_100)
        enter_operating_round(state)
        to_buy_train(state)
        assert set(legal_ids(state)) >= {depot(T6)} and depot(TD) not in legal_ids(state)
        set_cash(state, KO, D.TRAIN_PRICE[TD])
        assert {depot(T6), depot(TD)} <= set(legal_ids(state))

    def test_the_pool_sells_any_rank_at_its_list_price(self):
        state = at_or(3)
        hold(state, KO, {0: 6})
        TRAINS.set_depot(state, T2, D.TRAIN_COUNT[T2] - 1)
        TRAINS.set_pool(state, T2, 1)
        enter_operating_round(state)
        to_buy_train(state)
        assert legal_ids(state) == [K.BUY_TRAIN_PASS, depot(T2), pool(T2)]
        cash, bank = CORPS[KO].cash(state), BANK.cash(state)
        apply_and_verify(state, pool(T2))
        assert TRAINS.pool(state, T2) == 0 and trains_of(state, KO) == {'2': 1}
        assert CORPS[KO].cash(state) == cash - 80 and BANK.cash(state) == bank + 80
        set_cash(state, KO, 79)
        assert legal_ids(state) == [K.BUY_TRAIN_PASS]

    def test_another_corp_sells_any_train_for_a_price_from_one_yen_to_the_treasury(self):
        """Its president needs no consent (``depot.rb:118-126``); TRAIN_PRICE
        follows, and the price goes to the seller."""
        state = at_or(3)
        hold(state, KO, {0: 6}, par=PAR_100)
        hold(state, IR, {1: 6})
        give_train(state, IR, T2, 2)
        enter_operating_round(state)
        to_buy_train(state)
        assert TURN.operating_corp(state) == KO and legal_ids(state) == [K.BUY_TRAIN_PASS, depot(T2), from_corp(IR, T2)]
        apply_and_verify(state, from_corp(IR, T2))
        assert TURN.step(state) == K.STEP_TRAIN_PRICE and TURN.pending_train_source(state) == IR * 6 + T2
        ko, ir, bank = CORPS[KO].cash(state), CORPS[IR].cash(state), BANK.cash(state)
        apply_and_verify(state, exact_id(state, 123))
        assert CORPS[KO].cash(state) == ko - 123 and CORPS[IR].cash(state) == ir + 123 and BANK.cash(state) == bank
        assert trains_of(state, KO) == {'2': 1} and trains_of(state, IR) == {'2': 1}
        assert TURN.step(state) == K.STEP_BUY_TRAIN and TURN.pending_train_source(state) == -1

    def test_a_treasury_without_cash_buys_from_no_corp(self):
        """``other_trains`` needs ¥1, so KO's one choice is the pass."""
        state = at_or(3)
        hold(state, KO, {0: 6}, par=PAR_100)
        hold(state, IR, {1: 6})
        give_train(state, IR, T2)
        set_cash(state, KO, 0)
        enter_operating_round(state)
        result = apply_and_verify(state, K.TRACK_PASS)
        assert legal_ids(result.state_before_decision(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_PASS)) == [K.BUY_TRAIN_PASS]

    def test_the_policy_buys_only_from_the_same_president(self):
        """``train_sale_same_president`` filters the policy, not legality (§7)."""
        state = at_or(3)
        hold(state, KO, {0: 6}, par=PAR_100)
        hold(state, IR, {1: 6})
        hold(state, SR, {0: 6})
        give_train(state, IR, T2)
        give_train(state, SR, T2)
        enter_operating_round(state)
        to_buy_train(state)
        assert TURN.operating_corp(state) == KO
        assert {from_corp(IR, T2), from_corp(SR, T2)} <= set(legal_ids(state)) == set(policy_ids(state))
        state.train_sale_same_president = True
        assert from_corp(SR, T2) in policy_ids(state) and from_corp(IR, T2) not in policy_ids(state)
        assert from_corp(IR, T2) in legal_ids(state)

    def test_a_corp_at_its_limit_buys_nothing_even_a_rusting_rank_away(self):
        """Rules §8.8: the first 4 would rust KO's 2s, but KO is at the limit."""
        state = at_or(3)
        enter_phase(state, PHASE_3)
        hold(state, KO, {0: 6}, par=PAR_100)
        give_train(state, KO, T2, 2)
        give_train(state, KO, T3, 2)
        TRAINS.set_depot(state, T3, 0)
        TRAINS.set_pool(state, T3, D.TRAIN_COUNT[T3] - 2)
        enter_operating_round(state)
        result = apply_and_verify(state, K.TRACK_PASS)
        assert TRAINS.depot_types(state) == 1 << T4 and (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_PASS) in result.forced
        assert trains_of(state, KO) == {'2': 2, '3': 2} and PHASE.current(state) == PHASE_3

    def test_a_must_buy_has_no_pass_and_may_buy_any_train_its_treasury_affords(self):
        """In phase 3 the depot sells 3s; KO may also take a cheaper pool 2 or
        another corp's train."""
        state = at_or(3)
        enter_phase(state, PHASE_3)
        lay(state, 'J3', '8', 3)
        hold(state, KO, {0: 6}, par=PAR_100)
        hold(state, IR, {1: 6})
        give_train(state, IR, T2)
        enter_operating_round(state)
        to_buy_train(state)
        assert TURN.operating_corp(state) == KO and A.must_buy_train_py(state)
        assert legal_ids(state) == [depot(T3), pool(T2), pool(T3), from_corp(IR, T2), *private_buys(state)]
        set_cash(state, KO, 100)  # the pool's 2 alone is in reach, and IR's train
        assert legal_ids(state) == [pool(T2), from_corp(IR, T2), *private_buys(state)]

    def test_a_must_buy_the_treasury_cannot_fund_needs_emergency_money(self):
        """TEMPORARY (6.8b): short of every depot and pool train, the president's
        cash and the buys from other corps that go with it are emergency money."""
        state = at_or(3)
        lay(state, 'J3', '8', 3)
        hold(state, KO, {0: 6})
        hold(state, IR, {1: 6}, par=PAR_100)
        give_train(state, IR, T2)
        set_cash(state, KO, 79)
        enter_operating_round(state)
        apply_and_verify(state, K.TRACK_PASS)
        apply_and_verify(state, K.BUY_TRAIN_PASS)  # IR's turn ends
        apply_and_verify(state, K.TRACK_PASS)
        assert TURN.operating_corp(state) == KO and TURN.step(state) == K.STEP_BUY_TRAIN
        assert at_development_stop(state)
        with pytest.raises(NotImplementedError, match='emergency money'):
            legal_ids(state)


# =============================================================================
# TRAIN_PRICE
# =============================================================================

def at_train_price(cash, train=T4):
    """KO, with ``cash``, has chosen IR's train of type ``train`` (phase 4)."""
    state = at_or(3)
    enter_phase(state, PHASE_4)
    hold(state, KO, {0: 6}, par=PAR_100)
    hold(state, IR, {1: 6})
    give_train(state, IR, train)
    enter_operating_round(state)
    to_buy_train(state)
    set_cash(state, KO, cash)
    apply_and_verify(state, from_corp(IR, train))
    assert TURN.step(state) == K.STEP_TRAIN_PRICE
    return state


class TestTrainPrice:
    @pytest.mark.parametrize('cash, points', [
        (500, [1, 300, 167, 334, 500]),    # ¥1, list, ⌈cash/3⌉, ⌈2·cash/3⌉, cash
        (300, [1, 300, 100, 200, None]),   # the cash equals the list price: dropped
        (200, [1, None, 67, 134, 200]),    # the list price is beyond the cash
        (2, [1, None, None, 2, None]),     # ⌈2/3⌉ = ¥1 and the cash, ¥2, repeat earlier points
    ])
    def test_the_grid(self, cash, points):
        state = at_train_price(cash)
        assert legal_ids(state) == [k for k, amount in enumerate(points) if amount]
        assert [A.price_amount_py(state, k) for k in legal_ids(state)] == [p for p in points if p]

    def test_exact_amounts_range_from_one_yen_to_the_treasury(self):
        state = at_train_price(250)
        assert [A.is_legal_action_py(state, exact_id(state, amount)) for amount in (0, 1, 249, 250, 251)] == \
            [False, True, True, True, False]
        ko, ir = CORPS[KO].cash(state), CORPS[IR].cash(state)
        apply_and_verify(state, exact_id(state, 249))
        assert (CORPS[KO].cash(state), CORPS[IR].cash(state)) == (ko - 249, ir + 249)

    def test_one_yen_is_forced_when_it_is_all_the_treasury_has(self):
        state = at_or(3)
        enter_phase(state, PHASE_4)
        hold(state, KO, {0: 6}, par=PAR_100)
        hold(state, IR, {1: 6})
        give_train(state, IR, T4)
        enter_operating_round(state)
        to_buy_train(state)
        set_cash(state, KO, 1)
        result = apply_and_verify(state, from_corp(IR, T4))
        assert result.forced[0] == (K.DPHASE_TRAIN_PRICE, 0) and trains_of(state, KO) == {'4': 1}
        assert CORPS[KO].cash(state) == 0


# =============================================================================
# THE D TRADE-IN
# =============================================================================

def in_phase_6(trains, cash=1000):
    """KO in phase 6 at BUY_TRAIN with ``trains`` (type -> count) and ``cash``;
    the 4s it does not hold wait in the pool."""
    state = at_or(3)
    enter_phase(state, PHASE_6)
    hold(state, KO, {0: 6}, par=PAR_100)
    for t, n in trains.items():
        give_train(state, KO, t, n)
    enter_operating_round(state)
    set_cash(state, KO, cash)
    to_buy_train(state)
    return state


class TestTradeIn:
    def test_a_corp_at_its_limit_may_trade_a_4_5_or_6_for_a_d(self):
        state = in_phase_6({T5: 1, T6: 1})
        assert legal_ids(state) == [K.BUY_TRAIN_PASS, trade(T5), trade(T6)]
        set_cash(state, KO, 799)
        assert legal_ids(state) == [K.BUY_TRAIN_PASS]

    def test_the_first_d_starts_phase_d_and_rusts_every_4_the_traded_one_too(self):
        state = in_phase_6({T4: 1, T5: 1})
        assert TRAINS.pool(state, T4) == D.TRAIN_COUNT[T4] - 1
        cash, bank = CORPS[KO].cash(state), BANK.cash(state)
        result = apply_and_verify(state, trade(T4))
        assert PHASE.current(state) == PHASE_D and trains_of(state, KO) == {'5': 1, 'D': 1}
        assert TRAINS.pool(state, T4) == 0 and TRAINS.depot(state, TD) == -1
        assert result.forced[0] == (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_PASS)  # at the limit, short of another D
        after = result.state_before_decision(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_PASS)
        assert CORPS[KO].cash(after) == cash - 800 and BANK.cash(after) == bank + 800

    def test_a_traded_5_or_6_goes_to_the_pool(self):
        state = in_phase_6({T6: 2})
        apply_and_verify(state, trade(T6))
        assert trains_of(state, KO) == {'6': 1, 'D': 1} and TRAINS.pool(state, T6) == 1


# =============================================================================
# PHASES
# =============================================================================

class TestPhases:
    def test_the_first_3_opens_phase_3_and_the_private_window(self):
        """Buying the depot's last 2 makes 3s available, and the first 3
        starts phase 3: nothing rusts, the limit stays 4, the set's length
        changes when the next SR ends, and BUY_PRIVATE follows the pass."""
        state = at_or(3)
        hold(state, KO, {0: 6}, par=PAR_100)
        TRAINS.set_depot(state, T2, 1)
        TRAINS.set_pool(state, T2, D.TRAIN_COUNT[T2] - 1)
        enter_operating_round(state)
        to_buy_train(state)
        apply_and_verify(state, depot(T2))
        assert TRAINS.depot_types(state) == 1 << T3
        apply_and_verify(state, depot(T3))
        assert PHASE.current(state) == PHASE_3 and TRAINS.limit(state) == 4 and trains_of(state, KO) == {'2': 1, '3': 1}
        assert TURN.or_count(state) == 1  # the set's length changes when the next SR ends
        apply_and_verify(state, K.BUY_TRAIN_PASS)
        assert TURN.step(state) == K.STEP_BUY_PRIVATE and K.BUY_PRIVATE_PRIVATE + D.PRIVATE_SIR in legal_ids(state)

    def test_the_first_4_rusts_every_2_in_the_pool_too(self):
        state = at_or(3)
        enter_phase(state, PHASE_3)
        hold(state, KO, {0: 6}, par=PAR_100)
        hold(state, IR, {1: 6})
        give_train(state, KO, T2)
        give_train(state, IR, T2, 2)
        TRAINS.set_depot(state, T3, 0)
        TRAINS.set_pool(state, T3, D.TRAIN_COUNT[T3])
        enter_operating_round(state)
        to_buy_train(state)
        assert TURN.operating_corp(state) == KO and TRAINS.pool(state, T2) == 3
        apply_and_verify(state, depot(T4))
        assert PHASE.current(state) == PHASE_4 and TRAINS.limit(state) == 3
        assert trains_of(state, KO) == {'4': 1} and trains_of(state, IR) == {} and TRAINS.pool(state, T2) == 0
        assert CORPS[KO].route_dirty(state) and CORPS[IR].route_dirty(state)

    @pytest.mark.parametrize('owner', ['player', 'corp'])
    def test_the_first_5_closes_the_privates_but_a_player_owned_utf(self, owner):
        """A player-owned UTF stays open and pays ¥50; a corp-owned one closes."""
        state = at_or(4, owners=[0, 1, 2, 3])
        enter_phase(state, PHASE_4)
        hold(state, KO, {0: 6}, par=PAR_100)
        give_train(state, KO, T4)
        TRAINS.set_depot(state, T4, 0)
        TRAINS.set_pool(state, T4, D.TRAIN_COUNT[T4] - 1)
        if owner == 'corp':
            PRIVATES[D.PRIVATE_UTF].set_owner(state, K.OWNER_CORP + KO)
        utf_owner = PRIVATES[D.PRIVATE_UTF].owner(state)
        enter_operating_round(state)
        to_buy_train(state)
        assert PRIVATES[D.PRIVATE_UTF].revenue(state) == D.PRIVATE_REVENUE[D.PRIVATE_UTF]
        apply_and_verify(state, depot(T5))
        assert PHASE.current(state) == PHASE_5 and TRAINS.limit(state) == 2
        closed = [p for p in range(D.NUM_PRIVATES) if PRIVATES[p].owner(state) == K.OWNER_CLOSED]
        if owner == 'player':
            assert closed == [p for p in range(D.NUM_PRIVATES) if p != D.PRIVATE_UTF]
            assert PRIVATES[D.PRIVATE_UTF].owner(state) == utf_owner and PRIVATES[D.PRIVATE_UTF].revenue(state) == 50
        else:
            assert closed == list(range(D.NUM_PRIVATES))
        assert not PRIVATES[D.PRIVATE_TR].blocks_hex(state, D.HEX_IDS.index('K4'))

    def test_the_first_6_rusts_every_3_and_makes_ds_available(self):
        state = at_or(3)
        enter_phase(state, PHASE_5)
        hold(state, KO, {0: 6}, par=PAR_100)
        give_train(state, KO, T3)
        TRAINS.set_depot(state, T5, 0)
        TRAINS.set_pool(state, T5, D.TRAIN_COUNT[T5])
        enter_operating_round(state)
        to_buy_train(state)
        assert TRAINS.depot_types(state) == 1 << T6
        apply_and_verify(state, depot(T6))
        assert PHASE.current(state) == PHASE_6 and trains_of(state, KO) == {'6': 1} and TRAINS.pool(state, T3) == 0
        assert TRAINS.depot_types(state) == 1 << T6 | 1 << TD


# =============================================================================
# DISCARD
# =============================================================================

def at_crowding_buy():
    """Phase 4: KO (player 0) holds a 3 and a 4, IR (player 1) a 3, a 3, and
    a 4, the rest of the 3s and 4s wait in the pool, and the depot's next
    train is the first 5, which KO, operating first, is about to buy."""
    state = at_or(3)
    enter_phase(state, PHASE_4)
    hold(state, KO, {0: 6}, par=PAR_100)
    hold(state, IR, {1: 6})
    give_train(state, KO, T3)
    give_train(state, KO, T4)
    give_train(state, IR, T3, 2)
    give_train(state, IR, T4)
    TRAINS.set_depot(state, T4, 0)
    TRAINS.set_pool(state, T4, D.TRAIN_COUNT[T4] - 2)
    enter_operating_round(state)
    to_buy_train(state)
    assert TURN.operating_corp(state) == KO
    return state


class TestDiscard:
    def test_crowded_corps_discard_in_corp_order_then_the_buyer_goes_on(self):
        """IR comes before KO in corp order (``crowded_corps``), whatever the
        stock prices; each discard goes to the pool."""
        state = at_crowding_buy()
        apply_and_verify(state, depot(T5))
        assert (TURN.step(state), TURN.return_step(state)) == (K.STEP_DISCARD, K.STEP_BUY_TRAIN)
        assert TRAINS.first_crowded(state) == IR and TURN.active_player(state) == 1
        assert legal_ids(state) == [T3, T4]
        pool = TRAINS.pool(state, T3)
        apply_and_verify(state, T3)
        assert trains_of(state, IR) == {'3': 1, '4': 1} and TRAINS.pool(state, T3) == pool + 1
        assert TRAINS.first_crowded(state) == KO and TURN.active_player(state) == 0
        assert legal_ids(state) == [T3, T4, T5]
        result = apply_and_verify(state, T4)
        assert trains_of(state, KO) == {'3': 1, '5': 1}
        after = result.state_before_decision(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_PASS)  # KO's BUY_TRAIN, at the limit
        assert (TURN.step(after), TURN.return_step(after), TURN.active_player(after)) == (K.STEP_BUY_TRAIN, -1, 0)
        assert result.forced[0] == (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_PASS)

    def test_a_discard_with_one_train_type_is_forced(self):
        state = at_crowding_buy()
        TRAINS.set_corp_trains(state, IR, T4, 0)
        TRAINS.set_pool(state, T4, TRAINS.pool(state, T4) + 1)
        TRAINS.set_corp_trains(state, IR, T3, 3)
        TRAINS.set_pool(state, T3, TRAINS.pool(state, T3) - 1)
        result = apply_and_verify(state, depot(T5))
        assert result.forced == [(K.DPHASE_DISCARD, T3)] and TRAINS.first_crowded(state) == KO

    def test_discard_is_illegal_outside_a_crowded_corp(self):
        state = at_crowding_buy()
        TURN.set_step(state, K.STEP_DISCARD)
        row = state._array.copy()
        assert legal_ids(state) == [] and DRIVER.apply_action(state, T3) == STATUS_INVALID
        np.testing.assert_array_equal(state._array, row)
