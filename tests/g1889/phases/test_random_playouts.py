"""Random play from new games through the auction, SRs, and ORs
(1889/engine-design.md §10.4): at 2-6 players, with the policy flags on
(policy ids only) and off (every legal id, and exact amounts in BID and
RAISE), each game runs until the driver's development stop or a decision cap.

``apply_and_verify`` checks every state the driver passes through, forced
decisions and automated steps included: money totals ¥7,000, each corp's ten
shares are accounted for, each private has one owner, the supply is never
negative, every charter token is left or placed, the OR fields agree with the
step, and every returned decision is a choice. Here every state is also
checked for caches that are clean but stale (GRAPH recomputed from scratch),
and every returned decision for its legal and policy sets on a rebound copy
of the row.

The players pass whenever they may at BUY_TRAIN, since train purchases land
with 6.7, and pass in an SR half the time they may, so rounds end. Games
then run until a corp must buy a train, which is where 6.5 stops.
"""

from collections import Counter

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.graph import GRAPH
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.turn import TURN
from g1889.state import GameState, get_corp_fields, get_global_fields, get_layout

from tests.g1889.phases.conftest import (
    apply_and_verify, at_development_stop, exact_id, legal_ids, new_game, policy_ids,
)

SEEDS = range(20)
MAX_DECISIONS = 2000
EXACT_SHARE = 0.25                 # of BID and RAISE decisions, with the flags off
UNCAPPED_PAYOUTS = 3               # payout rounds before the flags-off player turns the cap on
STOCK_PASS_SHARE = 0.5

GLOBALS, CORP_FIELDS = get_global_fields(), get_corp_fields()
ALL_CORPS = (1 << D.NUM_CORPS) - 1


def _choose(state, rng, flags):
    phase = A.get_decision_phase_py(state)
    if not flags and phase in (K.DPHASE_BID, K.DPHASE_RAISE) and rng.random() < EXACT_SHARE:
        p = TURN.pending_private(state) if phase == K.DPHASE_BID else TURN.auction_private(state)
        low, high = PRIVATES[p].min_bid(state), PLAYERS[TURN.active_player(state)].available(state, p)
        return exact_id(state, int(rng.integers(low, high + 1)))
    ids = policy_ids(state) if flags else legal_ids(state)
    if ids[0] == 0 and (phase == K.DPHASE_BUY_TRAIN or phase == K.DPHASE_STOCK and rng.random() < STOCK_PASS_SHARE):
        return 0
    return ids[int(rng.integers(len(ids)))]


def _assert_fresh_caches(state, msg):
    """Every connectivity cache marked clean equals a recomputation."""
    layout = get_layout(state.max_players)
    row = state._array
    fresh = GameState.from_array(row, TURN.num_players(state), max_players=state.max_players)
    fresh._array[layout.global_offset + GLOBALS.conn_dirty] = ALL_CORPS  # test setup: drop every cache
    dirty = int(row[layout.global_offset + GLOBALS.conn_dirty])
    for c in range(D.NUM_CORPS):
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
    automated steps by name and the decisions by action type."""
    for _ in range(MAX_DECISIONS):
        if TURN.round(state) == K.ROUND_AUCTION and TURN.auction_pass_payouts(state) >= UNCAPPED_PAYOUTS:
            state.auction_pass_payout_cap = 1
        result = apply_and_verify(state, _choose(state, rng, flags), msg)
        for i in range(len(result.history)):
            _assert_fresh_caches(result.state_at(i), f'{msg}: before entry {i}')
        _assert_fresh_caches(state, f'{msg}: after')
        for _, phase, action_id in result.history:
            seen[K.Step(action_id).name if phase < 0 else A.decode_action_py(phase, action_id).type] += 1
        if at_development_stop(state):
            return
        _assert_rebound_sets(state, msg)
    pytest.fail(f'{msg}: no development stop within {MAX_DECISIONS} decisions')


@pytest.mark.parametrize('flags', [False, True], ids=['flags_off', 'flags_on'])
def test_random_games_run_to_the_development_stop(flags):
    """Every game gets into the ORs and stops where a corp must buy its
    first train (6.7). Between them, the games lay track, place tokens, and
    withhold."""
    seen = Counter()
    for num_players in range(D.MIN_PLAYERS, D.MAX_PLAYERS + 1):
        for seed in SEEDS:
            msg = f'{num_players} players, seed {seed}'
            rng = np.random.default_rng(seed)
            state = new_game(num_players, max_players=D.MAX_PLAYERS if seed % 2 else 0, seed=seed,
                             priority_player=-1, auction_pass_payout_cap=1 if flags else 0, same_president=flags)
            _play(state, rng, flags, msg, seen)
            assert TURN.step(state) == K.STEP_BUY_TRAIN and A.must_buy_train_py(state), msg
    assert seen['STEP_OR_END'] and seen['STEP_TURN_END']  # whole turns and ORs, before the stops
    assert seen[A.ACTION_LAY_CANDIDATE] and seen[A.ACTION_PLACE_TOKEN] and seen[A.ACTION_WITHHOLD]
