"""Random play from new games through the auction, SRs, and ORs
(1889/engine-design.md §10.4): at 2-6 players, with the policy flags on
(policy ids only) and off (every legal id, and exact amounts in BID, RAISE,
and TRAIN_PRICE), each game runs until the driver's development stop or a
decision cap.

``apply_and_verify`` checks every state the driver passes through, forced
decisions and automated steps included: money totals ¥7,000, each corp's ten
shares are accounted for, each private has one owner, the supply is never
negative, every charter token is left or placed, every train is in the
depot, the pool, or a floated corp until its rank rusts, no corp is over the
limit outside DISCARD, the OR fields agree with the step, and every returned
decision is a choice. Here every state is also checked for caches that are
clean but stale (GRAPH and the route solver recomputed from scratch), and
every returned decision for its legal and policy sets on a rebound copy of
the row.

The players pass in an SR half the time they may, so rounds end, and a corp
that can buy from the depot buys its newest offer (a D by trade-in included),
which moves the phases on; every other choice is uniform. Uniform BUY_TRAIN
choices let corps spend their treasuries on each other's trains, and the
rusting then strands them in emergencies: only 4 of these 200 games reach
phase D that way, against about a fifth with the bias. The games run until a
corp must buy a train its treasury cannot fund (emergency money) or the bank
breaks and its OR set ends, both 6.8's.
"""

from collections import Counter

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.bank import BANK
from g1889.entities.corp import CORPS
from g1889.entities.graph import GRAPH
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.routes.solver import max_route_revenue
from g1889.state import GameState, get_corp_fields, get_global_fields, get_layout

from tests.g1889.phases.conftest import (
    apply_and_verify, at_development_stop, exact_id, legal_ids, new_game, policy_ids,
)

SEEDS = range(20)
MAX_DECISIONS = 3000
EXACT_SHARE = 0.25                 # of BID, RAISE, and TRAIN_PRICE decisions, with the flags off
UNCAPPED_PAYOUTS = 3               # payout rounds before the flags-off player turns the cap on
STOCK_PASS_SHARE = 0.5

GLOBALS, CORP_FIELDS = get_global_fields(), get_corp_fields()
ALL_CORPS = (1 << D.NUM_CORPS) - 1


def _choose(state, rng, flags):
    phase = A.get_decision_phase_py(state)
    if not flags and A.has_exact_amounts(phase) and rng.random() < EXACT_SHARE:
        if phase == K.DPHASE_TRAIN_PRICE:
            low, high = 1, CORPS[TURN.operating_corp(state)].cash(state)
        else:
            p = TURN.pending_private(state) if phase == K.DPHASE_BID else TURN.auction_private(state)
            low, high = PRIVATES[p].min_bid(state), PLAYERS[TURN.active_player(state)].available(state, p)
        return exact_id(state, int(rng.integers(low, high + 1)))
    ids = policy_ids(state) if flags else legal_ids(state)
    if phase == K.DPHASE_BUY_TRAIN and (depot := [i for i in ids if K.BUY_TRAIN_DEPOT <= i < K.BUY_TRAIN_POOL]):
        return depot[-1]
    if ids[0] == 0 and phase == K.DPHASE_STOCK and rng.random() < STOCK_PASS_SHARE:
        return 0
    return ids[int(rng.integers(len(ids)))]


def _assert_fresh_caches(state, msg):
    """Every connectivity and route cache marked clean equals a recomputation."""
    layout = get_layout(state.max_players)
    row = state._array
    fresh = GameState.from_array(row, TURN.num_players(state), max_players=state.max_players)
    for field in (GLOBALS.conn_dirty, GLOBALS.route_dirty):  # test setup: drop every cache
        fresh._array[layout.global_offset + field] = ALL_CORPS
    dirty = int(row[layout.global_offset + GLOBALS.conn_dirty])
    route_dirty = int(row[layout.global_offset + GLOBALS.route_dirty])
    for c in range(D.NUM_CORPS):
        if not route_dirty >> c & 1:
            cache = layout.corps_offset + c * CORP_FIELDS.size + CORP_FIELDS.route_cache
            assert row[cache] == max_route_revenue(fresh, c), f'{msg}: corp {c} route cache is stale'
        if dirty >> c & 1:
            continue
        GRAPH.refresh(fresh, c)
        graph = slice(layout.graph_offset + c * D.NUM_HEXES, layout.graph_offset + (c + 1) * D.NUM_HEXES)
        route = layout.corps_offset + c * CORP_FIELDS.size + CORP_FIELDS.route_available
        assert np.array_equal(row[graph], fresh._array[graph]) and row[route] == fresh._array[route], \
            f'{msg}: corp {c} graph cache is stale'


def _assert_rebound_sets(state, msg):
    """A wrapper rebound to a copy of the row, flags alike, offers the same ids."""
    n = TURN.num_players(state)
    other = GameState(n, max_players=state.max_players, auction_pass_payout_cap=state.auction_pass_payout_cap,
                      private_sale_same_president=state.private_sale_same_president,
                      train_sale_same_president=state.train_sale_same_president)
    other.rebind(state._array.copy(), n, state.max_players)
    assert (legal_ids(other), policy_ids(other)) == (legal_ids(state), policy_ids(state)), msg


def _play(state, rng, flags, msg, seen):
    """Random decisions until the development stop. ``seen`` counts the
    automated steps by name and the decisions by phase and action type."""
    for _ in range(MAX_DECISIONS):
        if TURN.round(state) == K.ROUND_AUCTION and TURN.auction_pass_payouts(state) >= UNCAPPED_PAYOUTS:
            state.auction_pass_payout_cap = 1
        result = apply_and_verify(state, _choose(state, rng, flags), msg)
        for i in range(len(result.history)):
            _assert_fresh_caches(result.state_at(i), f'{msg}: before entry {i}')
        _assert_fresh_caches(state, f'{msg}: after')
        for _, phase, action_id in result.history:
            seen[K.Step(action_id).name if phase < 0 else (phase, A.decode_action_py(phase, action_id).type)] += 1
        if at_development_stop(state):
            return
        _assert_rebound_sets(state, msg)
    pytest.fail(f'{msg}: no development stop within {MAX_DECISIONS} decisions')


def _stop(state):
    """Where a game rests: a must-buy short of every depot and pool train, or
    the end of a broken bank's OR set."""
    if TURN.step(state) == K.STEP_OR_END and BANK.broken(state) and TURN.or_num(state) == TURN.or_count(state):
        return 'bank'
    corp = TURN.operating_corp(state)
    assert TURN.step(state) == K.STEP_BUY_TRAIN and A.must_buy_train_py(state), K.Step(TURN.step(state)).name
    assert CORPS[corp].cash(state) < TRAINS.cheapest_price(state)
    return 'emergency'


# Of each setting's 100 games, at least this many reach phases 4 and D. As
# measured (6.7): phases 4, 5, 6, and D in 99, 73, 46, and 20 games with the
# flags off, and 100, 83, 35, and 22 with them on; 25 and 48 games end at a
# broken bank, the rest in emergencies.
MIN_REACH = {D.GAME_PHASE_4: 90, D.GAME_PHASE_D: 15}


@pytest.mark.parametrize('flags', [False, True], ids=['flags_off', 'flags_on'])
def test_random_games_run_to_the_development_stop(flags):
    """Every game gets into the ORs, buys trains through the phases, and stops
    at an emergency or a broken bank's game end (6.8); about a fifth reach
    phase D. Between them, the games lay track, place tokens, pay out and
    withhold, buy from the depot, the pool, and each other at grid and exact
    prices, trade in for Ds, and discard."""
    seen, phases = Counter(), Counter()
    for num_players in range(D.MIN_PLAYERS, D.MAX_PLAYERS + 1):
        for seed in SEEDS:
            msg = f'{num_players} players, seed {seed}'
            rng = np.random.default_rng(seed)
            state = new_game(num_players, max_players=D.MAX_PLAYERS if seed % 2 else 0, seed=seed,
                             priority_player=-1, auction_pass_payout_cap=1 if flags else 0, same_president=flags)
            _play(state, rng, flags, msg, seen)
            seen[_stop(state)] += 1
            phases.update(range(PHASE.current(state) + 1))
    assert all(phases[phase] >= n for phase, n in MIN_REACH.items()), phases
    assert seen['bank'] and seen['emergency'] and seen['STEP_OR_END'] and seen['STEP_TURN_END']
    assert seen[(K.DPHASE_TILE, A.ACTION_LAY_CANDIDATE)] and seen[(K.DPHASE_TOKEN, A.ACTION_PLACE_TOKEN)]
    for kind in (A.ACTION_WITHHOLD, A.ACTION_PAYOUT):
        assert seen[(K.DPHASE_DIVIDEND, kind)]
    for kind in (A.ACTION_BUY_DEPOT_TRAIN, A.ACTION_BUY_POOL_TRAIN, A.ACTION_BUY_CORP_TRAIN, A.ACTION_TRADE_IN):
        assert seen[(K.DPHASE_BUY_TRAIN, kind)], kind
    assert seen[(K.DPHASE_TRAIN_PRICE, A.ACTION_PRICE_POINT)] and seen[(K.DPHASE_DISCARD, A.ACTION_DISCARD)]
    assert seen[(K.DPHASE_BUY_PRIVATE, A.ACTION_PASS)]
    assert bool(seen[(K.DPHASE_TRAIN_PRICE, A.ACTION_EXACT_PRICE)]) != flags
