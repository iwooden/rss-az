"""The operating round's flow as 6.5 and 6.6 build it: OR_START, TURN_START,
TRACK, TILE, TOKEN, ROUTE, DIVIDEND at zero revenue, BUY_TRAIN's pass and
depot offers, TURN_END, and OR_END (1889/engine-design.md §5.2-5.4, §6.1, §8;
rules §8; Ruby's ``round/operating.rb`` and the OR steps).

Covers the privates' revenue, home tokens, the OR order with its price and
stack ties, an OR with no floated corp, turns and their presidents, the next
OR of a set and the next SR, a lay's cost, tile, and reach, TILE's forced
single candidate, a token's price, ROUTE's revenue and its replay override,
the withhold's move at the market's edges, must-buy against pass, and the
game's end after a broken bank's OR set.
``test_trains.py`` covers 6.7's dividends, train purchases, phases, and
discards; ``tests/g1889/replay/`` checks the same paths against Ruby on the
fixtures; ``test_track_and_tokens.py`` pins the lay and token rules.

Sides: 0 S, 1 SW, 2 NW, 3 N, 4 NE, 5 SE. KO's home, Takamatsu (K4), has
printed track to K6 (side 0), J5 (side 1), and J3 (side 2).
"""

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_GAME_OVER, STATUS_INVALID, STATUS_OK
from g1889.entities.bank import BANK
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
    HEX, TILE, apply_and_verify, at_or, depot, enter_operating_round, hold, lay, legal_ids, set_cash,
    train_choices, train_decisions,
)

AR, IR, SR, KO, TR, KU, UR = range(D.NUM_CORPS)
PAR_65, PAR_70, PAR_100 = 0, 1, 5
BUY_2 = train_decisions(depot(D.TRAIN_2))  # the depot, then its 2


def order(state):
    return [c for c in (TURN.or_order(state, i) for i in range(D.NUM_CORPS)) if c >= 0]


def turn_of(state):
    return TURN.operating_corp(state), TURN.or_index(state), TURN.active_player(state), TURN.step(state)


def pass_turn(state):
    """Pass every choice until the operating corp's turn is over, or the
    game; the results."""
    results = []
    while TURN.round(state) != K.ROUND_GAME_OVER:
        results.append(apply_and_verify(state, 0, expected_status=None))
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
        cash = PLAYERS[2].cash(state)  # the privates' owner
        results = pass_turn(state)
        assert steps(results)[-4:] == [K.STEP_TURN_END, K.STEP_OR_END, K.STEP_OR_START, K.STEP_TURN_START]
        assert (TURN.round(state), TURN.or_num(state), TURN.turn(state)) == (K.ROUND_OPERATING, 2, 1)
        assert turn_of(state) == (KO, 0, 0, K.STEP_TRACK) and PLAYERS[2].cash(state) > cash  # paid again

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

    def test_a_broken_bank_ends_the_game_after_its_set(self):
        """``GAME_END_CHECK``'s ``bank: :full_or``: the bank broke before the
        set (in the SR, say), so the set's ORs run, and after its last the game
        is over, keeping the last OR's order as Ruby keeps its round."""
        state = at_or(3)
        hold(state, KO, {0: 6})
        BANK.set_broken(state)
        enter_operating_round(state, or_count=2)
        pass_turn(state)
        assert TURN.or_num(state) == 2 and not A.is_terminal_py(state)  # the set goes on
        results = pass_turn(state)
        assert steps(results)[-2:] == [K.STEP_TURN_END, K.STEP_OR_END] and results[-1].status == STATUS_GAME_OVER
        assert A.is_terminal_py(state) and TURN.end_reason(state) == K.END_REASON_BANK
        assert (TURN.round(state), TURN.step(state)) == (K.ROUND_GAME_OVER, K.STEP_GAME_OVER)
        assert order(state) == [KO] and TURN.operating_corp(state) == -1 and TURN.or_num(state) == 2
        assert A.final_scores_py(state) == [PLAYERS[i].value(state) for i in range(3)]
        assert DRIVER.apply_action(state, 0) == STATUS_GAME_OVER

    def test_a_bank_that_breaks_in_the_sets_last_or_ends_the_game_there(self):
        """KO's payout breaks the bank in the set's only OR; the game ends at
        its end, IR's turn first."""
        state = at_or(3)
        lay(state, 'J3', '8', 3)
        hold(state, KO, {0: 6}, par=PAR_100)
        hold(state, IR, {1: 6})
        TRAINS.set_depot(state, D.TRAIN_2, TRAINS.depot(state, D.TRAIN_2) - 1)
        TRAINS.set_corp_trains(state, KO, D.TRAIN_2, 1)
        BANK.transfer(state, -1, K.OWNER_PLAYER + 1, BANK.cash(state) - 10)
        enter_operating_round(state)
        assert TURN.operating_corp(state) == KO
        apply_and_verify(state, K.TRACK_PASS)
        apply_and_verify(state, K.DIVIDEND_PAYOUT)  # ¥50 from a bank holding ¥10
        assert BANK.broken(state) and not A.is_terminal_py(state)
        assert TURN.operating_corp(state) == KO and TURN.step(state) == K.STEP_BUY_TRAIN
        apply_and_verify(state, K.BUY_TRAIN_PASS)
        assert TURN.operating_corp(state) == IR
        results = pass_turn(state)
        assert results[-1].status == STATUS_GAME_OVER and TURN.end_reason(state) == K.END_REASON_BANK


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
        turn, so ROUTE follows. With K4 and K8 reached, KO must then buy, and
        its one choice, a 2, is forced."""
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
        assert result.steps[0] == K.STEP_ROUTE and result.forced[-2:] == BUY_2
        assert CORPS[KO].cash(state) == cash - 40 - 80 and BANK.cash(state) == bank + 40 + 80
        assert TURN.step(state) == K.STEP_BUY_TRAIN and not A.must_buy_train_py(state)


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

    def test_revenue_offers_the_payout(self):
        state = at_or(3)
        hold(state, KO, {0: 6})
        enter_operating_round(state)
        TURN.set_step(state, K.STEP_DIVIDEND)
        assert legal_ids(state) == [K.DIVIDEND_WITHHOLD]
        TURN.set_route_revenue(state, 50)
        assert legal_ids(state) == [K.DIVIDEND_WITHHOLD, K.DIVIDEND_PAYOUT]


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
        assert TURN.step(state) == K.STEP_BUY_TRAIN and train_choices(state) == [K.BUY_TRAIN_PASS, depot(D.TRAIN_2)]
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
        """No pass; the one train, a 2, is forced, and then KO may pass or buy more."""
        state = at_or(3)
        lay(state, 'J3', '8', 3)  # K4 to Okayama: two revenue centers
        result = to_buy_train(state)
        assert result.forced[-2:] == BUY_2 and TRAINS.corp_trains(state, KO, D.TRAIN_2) == 1
        assert TURN.step(state) == K.STEP_BUY_TRAIN and not A.must_buy_train_py(state)
        assert train_choices(state) == [K.BUY_TRAIN_PASS, depot(D.TRAIN_2)]

    def test_a_must_buy_beyond_the_treasury_is_an_emergency(self):
        """The president pays what the treasury lacks of the cheapest train,
        KO's one choice, which is forced (``test_emergency.py`` has the rest)."""
        state = at_or(3)
        lay(state, 'J3', '8', 3)
        hold(state, KO, {0: 6})
        set_cash(state, KO, D.TRAIN_PRICE[D.TRAIN_2] - 1)
        enter_operating_round(state)
        cash = PLAYERS[0].cash(state)
        result = apply_and_verify(state, K.TRACK_PASS)
        assert result.forced[-3:-1] == BUY_2 and TRAINS.corp_trains(state, KO, D.TRAIN_2) == 1
        assert CORPS[KO].cash(state) == 0 and PLAYERS[0].cash(state) == cash - 1

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

    def test_in_phases_3_and_4_the_buy_private_window_follows(self):
        """BUY_PRIVATE offers each private the corp may buy, and "done" ends
        the turn; ``test_privates.py`` covers the purchases."""
        state = at_or(3)
        PHASE.set_current(state, D.GAME_PHASE_3)
        TRAINS.set_depot(state, D.TRAIN_2, 0)  # sold out, rusted at phase 4: given away for the test
        TRAINS.set_pool(state, D.TRAIN_2, D.TRAIN_COUNT[D.TRAIN_2])
        TRAINS.set_depot(state, D.TRAIN_3, D.TRAIN_COUNT[D.TRAIN_3] - 1)
        TRAINS.set_pool(state, D.TRAIN_3, 1)
        to_buy_train(state)
        apply_and_verify(state, K.BUY_TRAIN_PASS)
        assert TURN.step(state) == K.STEP_BUY_PRIVATE and legal_ids(state) == [K.BUY_PRIVATE_DONE] + [
            K.BUY_PRIVATE_PRIVATE + p for p in A.purchasable_privates_py(state)] and len(legal_ids(state)) > 1
        result = apply_and_verify(state, K.BUY_PRIVATE_DONE)
        assert result.steps[0] == K.STEP_TURN_END and TURN.round(state) == K.ROUND_OPERATING


class TestRoute:
    """ROUTE runs the corp's trains for the most they earn (``routes/solver.pyx``;
    ``tests/g1889/test_routes.py`` pins the route rules), and replay can pay a
    human's routes instead (``DRIVER.override_route_revenue``)."""

    @staticmethod
    def with_a_train(track=True):
        """KO with a 2-train at its first TRACK choice; ``track`` lays J3 from
        Takamatsu (¥30) to Sakaide & Okayama (¥20 in phase 2)."""
        state = at_or(3)
        hold(state, KO, {0: 6})
        TRAINS.set_depot(state, D.TRAIN_2, D.TRAIN_COUNT[D.TRAIN_2] - 1)
        TRAINS.set_corp_trains(state, KO, D.TRAIN_2, 1)
        if track:
            lay(state, 'J3', '8', 3)
        enter_operating_round(state)
        return state

    def test_a_corp_runs_its_trains_for_the_most_they_earn(self):
        """DIVIDEND then waits for the president's choice."""
        state = self.with_a_train()
        result = apply_and_verify(state, K.TRACK_PASS)
        assert result.forced == [(K.DPHASE_TOKEN, K.TOKEN_PASS)] and result.steps == [K.STEP_ROUTE]
        assert TURN.step(state) == K.STEP_DIVIDEND and TURN.route_revenue(state) == 50
        assert legal_ids(state) == [K.DIVIDEND_WITHHOLD, K.DIVIDEND_PAYOUT]

    def test_a_corp_without_a_route_earns_nothing_and_withholds(self):
        state = self.with_a_train(track=False)
        result = apply_and_verify(state, K.TRACK_PASS)
        assert result.forced == [(K.DPHASE_TOKEN, K.TOKEN_PASS), (K.DPHASE_DIVIDEND, K.DIVIDEND_WITHHOLD)]
        assert K.STEP_ROUTE in result.steps and TURN.step(state) == K.STEP_BUY_TRAIN

    def test_replay_pays_a_human_revenue_with_no_upper_bound(self):
        state = self.with_a_train()
        apply_and_verify(state, K.TRACK_PASS)
        history = []
        assert DRIVER.override_route_revenue(state, 40, history) == STATUS_OK and history == []
        assert TURN.step(state) == K.STEP_DIVIDEND and TURN.route_revenue(state) == 40
        # Ruby pays logged revenue unchecked (314 action 367), so it may exceed the most the trains earn.
        assert DRIVER.override_route_revenue(state, 90) == STATUS_OK and TURN.route_revenue(state) == 90

    def test_a_zero_override_forces_the_withhold(self):
        state = self.with_a_train()
        apply_and_verify(state, K.TRACK_PASS)
        col, history = MARKET.col(state, KO), []
        assert DRIVER.override_route_revenue(state, 0, history) == STATUS_OK
        assert [(phase, action_id) for _, phase, action_id in history if phase >= 0] == \
            [(K.DPHASE_DIVIDEND, K.DIVIDEND_WITHHOLD)]
        assert MARKET.col(state, KO) == col - 1 and TURN.step(state) == K.STEP_BUY_TRAIN

    def test_an_override_needs_the_corp_dividend_and_a_multiple_of_ten(self):
        state = self.with_a_train()
        assert DRIVER.override_route_revenue(state, 40) == STATUS_INVALID  # at TRACK
        apply_and_verify(state, K.TRACK_PASS)
        row = state._array.copy()
        for revenue in (-10, 45, 1 << 15):
            assert DRIVER.override_route_revenue(state, revenue) == STATUS_INVALID
        np.testing.assert_array_equal(state._array, row)


class TestAdvance:
    def test_advance_at_a_choice_changes_nothing(self):
        state = at_or(3)
        hold(state, KO, {0: 6})
        enter_operating_round(state)
        row, history = state._array.copy(), []
        assert DRIVER.advance(state, history) == 0 and history == []
        np.testing.assert_array_equal(state._array, row)
