"""The operating round as 6.5 builds it: OR_START, TURN_START, TRACK, TILE,
TOKEN, ROUTE, DIVIDEND at zero revenue, BUY_TRAIN's pass and depot offers,
TURN_END, and OR_END (1889/engine-design.md §5.2-5.4, §6.1; rules §8; Ruby's
``round/operating.rb`` and the OR steps).

Covers the privates' revenue, home tokens, the OR order with its price and
stack ties, an OR with no floated corp, turns and their presidents, the next
OR of a set and the next SR, a lay's cost, tile, and reach, TILE's forced
single candidate, a token's price, the withhold's move at the market's edges,
must-buy against pass, and the driver's development stop wherever a later
sub-step takes over. ``tests/g1889/replay/`` checks the same paths against
Ruby on the fixtures; ``test_track_and_tokens.py`` pins the lay and token
rules.

Sides: 0 S, 1 SW, 2 NW, 3 N, 4 NE, 5 SE. KO's home, Takamatsu (K4), has
printed track to K6 (side 0), J5 (side 1), and J3 (side 2).
"""

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_INVALID
from g1889.entities.bank import BANK, BANK_ACCOUNT
from g1889.entities.corp import CORPS
from g1889.entities.graph import GRAPH
from g1889.entities.hexmap import MAP
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN

from tests.g1889.phases.conftest import (
    apply_and_verify, at_development_stop, end_auction, enter_operating_round, hold, legal_ids, new_game,
)

AR, IR, SR, KO, TR, KU, UR = range(D.NUM_CORPS)
PAR_65, PAR_70, PAR_100 = 0, 1, 5
HEX = {hex_id: h for h, hex_id in enumerate(D.HEX_IDS)}
TILE = {name: t for t, name in enumerate(D.TILE_NAMES)}
BUY_2 = K.BUY_TRAIN_DEPOT + D.TRAIN_2


def at_or(num_players=3, owners=()):
    """A game past the auction, the privates in play free to ``owners`` (seat 0
    by default); holdings, the map, and ``enter_operating_round`` follow."""
    state = new_game(num_players)
    end_auction(state, owners)
    return state


def lay(state, hex_id, tile, rotation):
    MAP.lay(state, HEX[hex_id], TILE[tile], rotation)


def set_cash(state, corp, cash):
    delta = CORPS[corp].cash(state) - cash
    BANK.transfer(state, *((K.OWNER_CORP + corp, BANK_ACCOUNT) if delta > 0 else (BANK_ACCOUNT, K.OWNER_CORP + corp)),
                  abs(delta))


def order(state):
    return [c for c in (TURN.or_order(state, i) for i in range(D.NUM_CORPS)) if c >= 0]


def turn_of(state):
    return TURN.operating_corp(state), TURN.or_index(state), TURN.active_player(state), TURN.step(state)


def pass_turn(state):
    """Pass every choice until the operating corp's turn is over, or the
    development stop; the results."""
    results = []
    while not at_development_stop(state):
        results.append(apply_and_verify(state, 0))
        if K.STEP_TURN_END in results[-1].steps:
            break
    return results


def steps(results):
    return [step for result in results for step in result.steps]


# =============================================================================
# OR_START
# =============================================================================

class TestOrStart:
    def test_privates_pay_their_owners_from_the_bank(self):
        state = at_or(3, owners=[0, 1, 2])
        hold(state, KO, {0: 6})
        PRIVATES[D.PRIVATE_SMR].set_owner(state, K.OWNER_CORP + KO)  # a corp's private pays its treasury
        players, ko, bank = [PLAYERS[i].cash(state) for i in range(3)], CORPS[KO].cash(state), BANK.cash(state)
        result = enter_operating_round(state)
        earned = [0, 0, 0]
        for p, private in enumerate(PRIVATES):
            owner = private.owner(state)
            if K.OWNER_PLAYER <= owner < K.OWNER_CORP:
                earned[owner] += D.PRIVATE_REVENUE[p]
        assert [PLAYERS[i].cash(state) - players[i] for i in range(3)] == earned and all(earned)
        assert CORPS[KO].cash(state) - ko == D.PRIVATE_REVENUE[D.PRIVATE_SMR]
        assert bank - BANK.cash(state) == sum(earned) + D.PRIVATE_REVENUE[D.PRIVATE_SMR]
        assert result.steps == [K.STEP_OR_START, K.STEP_TURN_START]

    def test_floated_corps_place_their_home_tokens_free(self):
        state = at_or(3)
        hold(state, KO, {0: 6})
        hold(state, IR, {1: 2})  # parred, not floated
        cash = CORPS[KO].cash(state)
        enter_operating_round(state, or_count=2)
        assert MAP.tokens(state, D.CORP_HOME[KO]) == 1 << KO and MAP.tokens(state, D.CORP_HOME[IR]) == 0
        assert CORPS[KO].tokens_left(state) == D.CORP_TOKENS[KO] - 1 and CORPS[KO].cash(state) == cash
        assert CORPS[IR].tokens_left(state) == D.CORP_TOKENS[IR] and order(state) == [KO]
        pass_turn(state)  # the next OR of the set leaves the placed token alone
        assert TURN.or_num(state) == 2 and CORPS[KO].tokens_left(state) == D.CORP_TOKENS[KO] - 1

    def test_the_order_is_stock_price_order(self):
        """Higher price first; at one price the column farther right
        (rules §6.7, ``Corporation#<=>``); in one cell, the earlier arrival."""
        state = at_or(3)
        hold(state, TR, {0: 6}, par=PAR_65)
        hold(state, AR, {1: 6}, par=PAR_70)          # ¥70 at (4, 3)
        hold(state, KO, {2: 6}, cell=(2, 1))         # ¥70 at (2, 1)
        hold(state, IR, {0: 6}, par=PAR_100)         # ¥100, first in the cell
        hold(state, SR, {1: 6}, par=PAR_100)         # beneath IR
        hold(state, UR, {2: 2}, par=PAR_100)         # parred, not floated
        assert MARKET.price(state, KO) == MARKET.price(state, AR) == 70
        enter_operating_round(state)
        assert order(state) == [IR, SR, AR, KO, TR]
        assert turn_of(state) == (IR, 0, 0, K.STEP_TRACK)

    def test_an_or_without_a_floated_corp_ends_at_once(self):
        state = at_or(3)
        hold(state, IR, {1: 2})
        TURN.set_priority_player(state, 1)
        bank = BANK.cash(state)
        result = enter_operating_round(state)
        assert result.steps == [K.STEP_OR_START, K.STEP_OR_END] and BANK.cash(state) < bank  # the privates paid
        assert (TURN.round(state), TURN.turn(state), TURN.or_num(state)) == (K.ROUND_STOCK, 2, 1)
        assert TURN.active_player(state) == 1


# =============================================================================
# TURNS AND ROUNDS
# =============================================================================

class TestTurns:
    def test_each_corp_operates_in_turn_with_its_president(self):
        state = at_or(3)
        hold(state, KO, {1: 6})
        hold(state, IR, {2: 6}, par=PAR_100)
        enter_operating_round(state)
        assert turn_of(state) == (IR, 0, 2, K.STEP_TRACK)
        results = pass_turn(state)
        assert steps(results)[-2:] == [K.STEP_TURN_END, K.STEP_TURN_START]
        assert turn_of(state) == (KO, 1, 1, K.STEP_TRACK)

    def test_a_turn_passes_through_token_route_and_dividend(self):
        """Ruby skips Token, Route, and Dividend here; the driver forces the
        token pass and the zero withhold, and runs ROUTE."""
        state = at_or(3)
        hold(state, KO, {0: 6})
        enter_operating_round(state)
        result = apply_and_verify(state, K.TRACK_PASS)
        assert result.forced == [(K.DPHASE_TOKEN, K.TOKEN_PASS), (K.DPHASE_DIVIDEND, K.DIVIDEND_WITHHOLD)]
        assert result.steps == [K.STEP_ROUTE] and TURN.step(state) == K.STEP_BUY_TRAIN

    def test_the_set_goes_on_to_its_next_or(self):
        state = at_or(3)
        hold(state, KO, {0: 6})
        enter_operating_round(state, or_count=2)
        cash = PLAYERS[0].cash(state)
        results = pass_turn(state)
        assert steps(results)[-4:] == [K.STEP_TURN_END, K.STEP_OR_END, K.STEP_OR_START, K.STEP_TURN_START]
        assert (TURN.round(state), TURN.or_num(state), TURN.turn(state)) == (K.ROUND_OPERATING, 2, 1)
        assert turn_of(state) == (KO, 0, 0, K.STEP_TRACK) and PLAYERS[0].cash(state) > cash  # paid again

    def test_the_sets_last_or_opens_the_next_sr(self):
        """Ruby reorders the players only after an SR: priority stays, and its
        holder opens the SR."""
        state = at_or(3)
        hold(state, KO, {0: 6})
        TURN.set_priority_player(state, 2)
        enter_operating_round(state)
        results = pass_turn(state)
        assert steps(results)[-2:] == [K.STEP_TURN_END, K.STEP_OR_END]
        assert (TURN.round(state), TURN.step(state), TURN.turn(state), TURN.or_num(state)) == \
            (K.ROUND_STOCK, K.STEP_STOCK, 2, 1)
        assert TURN.active_player(state) == TURN.priority_player(state) == 2
        assert order(state) == [] and TURN.operating_corp(state) == -1

    def test_a_broken_bank_rests_at_the_sets_end(self):
        """TEMPORARY (6.9 removes the development stop): the game ends after
        the set's last OR (``:full_or``), which lands with 6.8."""
        state = at_or(3)
        hold(state, KO, {0: 6})
        BANK.set_broken(state)
        enter_operating_round(state, or_count=2)
        pass_turn(state)
        assert TURN.or_num(state) == 2  # the set goes on
        results = pass_turn(state)
        assert steps(results)[-1] == K.STEP_TURN_END and TURN.step(state) == K.STEP_OR_END
        assert at_development_stop(state) and DRIVER.apply_action(state, 0) == STATUS_INVALID


# =============================================================================
# TRACK, TILE, TOKEN
# =============================================================================

class TestTrackTileToken:
    def test_a_lay_pays_its_cost_takes_its_tile_and_extends_the_network(self):
        state = at_or(3)
        hold(state, KO, {0: 6})
        hold(state, IR, {1: 6})
        enter_operating_round(state)
        assert order(state) == [KO, IR]
        GRAPH.refresh(state, IR)
        cash, bank, nine = CORPS[KO].cash(state), BANK.cash(state), TILE['9']
        assert GRAPH.sides_reached(state, KO, HEX['K8']) == 0
        apply_and_verify(state, K.TRACK_HEX + HEX['K6'])
        assert TURN.step(state) == K.STEP_TILE and TURN.pending_hex(state) == HEX['K6']
        candidates = A.lay_candidates_py(state, HEX['K6'])
        assert candidates == [(TILE['7'], 2), (TILE['8'], 3), (nine, 0)] and legal_ids(state) == [0, 1, 2]
        result = apply_and_verify(state, candidates.index((nine, 0)))
        assert (MAP.tile(state, HEX['K6']), MAP.rotation(state, HEX['K6'])) == (nine, 0)
        assert MAP.supply(state, nine) == D.TILE_COUNT[nine] - 1 and TURN.pending_hex(state) == -1
        assert CORPS[KO].cash(state) == cash - 80 and BANK.cash(state) == bank + 80  # K6's terrain
        assert GRAPH.dirty(result.state_at(1), IR)  # the lay made every corp's connectivity stale
        assert GRAPH.sides_reached(state, KO, HEX['K8']) == 1 << 3  # the straight reaches K8's north side

    def test_a_hex_with_one_candidate_forces_tile(self):
        state = at_or(3)
        hold(state, KO, {0: 6})
        for name in ('7', '8'):
            MAP.set_supply(state, TILE[name], 0)
        enter_operating_round(state)
        result = apply_and_verify(state, K.TRACK_HEX + HEX['K6'])
        assert result.forced[0] == (K.DPHASE_TILE, 0) and MAP.tile(state, HEX['K6']) == TILE['9']

    def test_one_lay_a_turn(self):
        state = at_or(3)
        hold(state, KO, {0: 6})
        enter_operating_round(state)
        apply_and_verify(state, K.TRACK_HEX + HEX['J3'])
        apply_and_verify(state, 0)
        assert TURN.step(state) != K.STEP_TRACK and A.lay_hexes_py(state, KO)  # lays are left, the turn's is used

    def test_a_token_costs_the_next_charter_price(self):
        """KO tokens Kochi (K8, AR's home with a second slot); one token a
        turn, so ROUTE follows. With K4 and K8 reached, KO must then buy."""
        state = at_or(3)
        hold(state, KO, {0: 6})
        lay(state, 'K6', '9', 0)
        lay(state, 'K8', '6', 1)
        lay(state, 'K8', '15', 1)
        enter_operating_round(state)
        apply_and_verify(state, K.TRACK_PASS)
        assert legal_ids(state) == [K.TOKEN_PASS, K.TOKEN_HEX + HEX['K8']]
        cash, bank = CORPS[KO].cash(state), BANK.cash(state)
        result = apply_and_verify(state, K.TOKEN_HEX + HEX['K8'])
        assert MAP.tokens(state, HEX['K8']) == 1 << KO and CORPS[KO].tokens_left(state) == 0
        assert CORPS[KO].cash(state) == cash - 40 and BANK.cash(state) == bank + 40
        assert result.steps[0] == K.STEP_ROUTE and TURN.step(state) == K.STEP_BUY_TRAIN
        assert A.must_buy_train_py(state) and at_development_stop(state)


# =============================================================================
# DIVIDEND
# =============================================================================

class TestDividend:
    @pytest.mark.parametrize('cell, moved', [
        ((4, 3), (4, 2)),     # left
        ((3, 0), (4, 0)),     # down at the left edge
        ((10, 0), (10, 0)),   # ¥10 stays
    ])
    def test_the_zero_withhold_moves_left_with_the_edge_rules(self, cell, moved):
        state = at_or(3)
        hold(state, KO, {0: 6}, cell=cell)
        hold(state, IR, {1: 6}, cell=moved)  # waits at the destination
        stamp = MARKET.stamp(state, KO)
        enter_operating_round(state)
        assert TURN.operating_corp(state) == KO
        result = apply_and_verify(state, K.TRACK_PASS)
        assert (K.DPHASE_DIVIDEND, K.DIVIDEND_WITHHOLD) in result.forced
        assert (MARKET.row(state, KO), MARKET.col(state, KO)) == moved
        assert CORPS[KO].last_revenue(state) == 0 and TURN.route_revenue(state) == 0
        if moved == cell:
            assert MARKET.stamp(state, KO) == stamp
        else:  # the bottom of the new cell's stack
            assert MARKET.stamp(state, KO) > MARKET.stamp(state, IR)

    def test_only_zero_revenue_is_settled_yet(self):
        state = at_or(3)
        hold(state, KO, {0: 6})
        enter_operating_round(state)
        TURN.set_step(state, K.STEP_DIVIDEND)
        assert legal_ids(state) == [K.DIVIDEND_WITHHOLD]
        TURN.set_route_revenue(state, 50)
        with pytest.raises(NotImplementedError, match='dividends'):
            legal_ids(state)


class TestMarketMoves:
    @pytest.mark.parametrize('cell, left, right', [
        ((4, 3), (4, 2), (4, 4)),
        ((3, 0), (4, 0), (3, 1)),
        ((10, 0), (10, 0), (10, 1)),
        ((2, 11), (2, 10), (1, 11)),   # right at the row's end goes up
        ((0, 14), (0, 13), (0, 14)),   # ¥350 stays
    ])
    def test_moves_follow_the_two_dimensional_edges(self, cell, left, right):
        state = at_or(3)
        for move, expected in ((MARKET.move_left, left), (MARKET.move_right, right)):
            MARKET.set_position(state, KO, *cell)
            move(state, KO)
            assert (MARKET.row(state, KO), MARKET.col(state, KO)) == expected


# =============================================================================
# ROUTE, BUY_TRAIN
# =============================================================================

def to_buy_train(state, cell=None):
    """KO operates alone and passes its lay; the driver takes it to BUY_TRAIN."""
    hold(state, KO, {0: 6}, cell=cell)
    enter_operating_round(state)
    return apply_and_verify(state, K.TRACK_PASS)


class TestBuyTrain:
    def test_a_corp_without_a_route_may_pass_or_buy(self):
        state = at_or(3)
        to_buy_train(state)
        assert TURN.step(state) == K.STEP_BUY_TRAIN and legal_ids(state) == [K.BUY_TRAIN_PASS, BUY_2]
        assert not A.must_buy_train_py(state)
        result = apply_and_verify(state, K.BUY_TRAIN_PASS)
        assert result.steps[0] == K.STEP_TURN_END and TURN.round(state) == K.ROUND_STOCK

    def test_a_corp_that_cannot_afford_a_train_passes(self):
        state = at_or(3)
        hold(state, KO, {0: 6})
        set_cash(state, KO, D.TRAIN_PRICE[D.TRAIN_2] - 1)
        enter_operating_round(state)
        result = apply_and_verify(state, K.TRACK_PASS)
        assert (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_PASS) in result.forced and TURN.round(state) == K.ROUND_STOCK

    def test_a_corp_with_a_route_must_buy(self):
        """No pass; the one train is forced, and buying arrives with 6.7, so
        the driver rests before it with nothing changed (the development stop)."""
        state = at_or(3)
        lay(state, 'J3', '8', 3)  # K4 to Okayama: two revenue centers
        result = to_buy_train(state)
        assert TURN.step(state) == K.STEP_BUY_TRAIN and A.must_buy_train_py(state)
        assert legal_ids(state) == [BUY_2] and at_development_stop(state)
        assert all(phase != K.DPHASE_BUY_TRAIN for phase, _ in result.decisions)
        row = state._array.copy()
        with pytest.raises(NotImplementedError, match='6.7'):
            DRIVER.apply_action(state, BUY_2)
        np.testing.assert_array_equal(state._array, row)

    def test_a_must_buy_beyond_the_treasury_needs_emergency_money(self):
        state = at_or(3)
        lay(state, 'J3', '8', 3)
        hold(state, KO, {0: 6})
        set_cash(state, KO, D.TRAIN_PRICE[D.TRAIN_2] - 1)
        enter_operating_round(state)
        apply_and_verify(state, K.TRACK_PASS)
        assert TURN.step(state) == K.STEP_BUY_TRAIN and at_development_stop(state)
        with pytest.raises(NotImplementedError, match='emergency money'):
            legal_ids(state)

    @pytest.mark.parametrize('where', ['pool', 'corp'])
    def test_trains_outside_the_depot_wait_for_their_sources(self, where):
        state = at_or(3)
        to_buy_train(state)
        if where == 'pool':
            TRAINS.set_pool(state, D.TRAIN_2, 1)
        else:
            TRAINS.set_corp_trains(state, IR, D.TRAIN_2, 1)
        with pytest.raises(NotImplementedError, match='pool or corps'):
            legal_ids(state)

    def test_the_depot_sells_its_lowest_rank_and_d_from_phase_6(self):
        state = at_or(3)
        assert TRAINS.depot_types(state) == 1 << D.TRAIN_2
        TRAINS.set_depot(state, D.TRAIN_2, 0)
        assert TRAINS.depot_types(state) == 1 << D.TRAIN_3
        for t in (D.TRAIN_3, D.TRAIN_4, D.TRAIN_5):
            TRAINS.set_depot(state, t, 0)
        assert TRAINS.depot_types(state) == 1 << D.TRAIN_6  # D waits for phase 6
        PHASE.set_current(state, D.GAME_PHASE_6)
        assert TRAINS.depot_types(state) == 1 << D.TRAIN_6 | 1 << D.TRAIN_D
        TRAINS.set_depot(state, D.TRAIN_6, 0)
        assert TRAINS.depot_types(state) == 1 << D.TRAIN_D

    def test_buys_offer_each_affordable_depot_type(self):
        """From phase 6 the depot sells 6s and Ds; a must-buy corp that can
        afford the 6 but not the D is offered the 6."""
        state = at_or(3)
        for t in (D.TRAIN_2, D.TRAIN_3, D.TRAIN_4, D.TRAIN_5):
            TRAINS.set_depot(state, t, 0)
        PHASE.set_current(state, D.GAME_PHASE_6)
        hold(state, KO, {0: 6}, par=PAR_100)  # ¥1,000
        enter_operating_round(state)
        apply_and_verify(state, K.TRACK_PASS)
        assert legal_ids(state) == [K.BUY_TRAIN_PASS, K.BUY_TRAIN_DEPOT + D.TRAIN_6]
        set_cash(state, KO, D.TRAIN_PRICE[D.TRAIN_D])
        assert legal_ids(state) == [K.BUY_TRAIN_PASS, K.BUY_TRAIN_DEPOT + D.TRAIN_6, K.BUY_TRAIN_DEPOT + D.TRAIN_D]

    def test_in_phases_3_and_4_the_buy_private_window_follows(self):
        """TEMPORARY (6.9 removes the development stop): BUY_PRIVATE lands with 6.8."""
        state = at_or(3)
        PHASE.set_current(state, D.GAME_PHASE_3)
        to_buy_train(state)
        apply_and_verify(state, K.BUY_TRAIN_PASS)
        assert TURN.step(state) == K.STEP_BUY_PRIVATE and at_development_stop(state)


class TestRoute:
    def test_a_corp_with_trains_rests_at_route(self):
        """TEMPORARY (6.9 removes the development stop): running trains lands with 6.6."""
        state = at_or(3)
        hold(state, KO, {0: 6})
        TRAINS.set_depot(state, D.TRAIN_2, D.TRAIN_COUNT[D.TRAIN_2] - 1)
        TRAINS.set_corp_trains(state, KO, D.TRAIN_2, 1)
        enter_operating_round(state)
        result = apply_and_verify(state, K.TRACK_PASS)
        assert result.forced == [(K.DPHASE_TOKEN, K.TOKEN_PASS)] and result.steps == []
        assert TURN.step(state) == K.STEP_ROUTE and at_development_stop(state)
        assert DRIVER.apply_action(state, 0) == STATUS_INVALID


class TestAdvance:
    def test_advance_at_a_choice_changes_nothing(self):
        state = at_or(3)
        hold(state, KO, {0: 6})
        enter_operating_round(state)
        row, history = state._array.copy(), []
        assert DRIVER.advance(state, history) == 0 and history == []
        np.testing.assert_array_equal(state._array, row)
