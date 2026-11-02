"""Random play through the auction and the first SR (1889/engine-design.md
§10.4): from a fresh game at 2-6 players, with the policy flags on (policy
ids only) and off (every legal id, and exact amounts in BID and RAISE), the
first SR ends within a step cap. ``test_random_playouts.py`` plays on
through the operating rounds.

``apply_and_verify`` checks every state the driver passes through, forced
decisions and SR_END included: money is conserved, each corp's ten shares are
accounted for, presidents hold the most, floats follow the IPO, and every
returned decision is a choice. In the first SR nobody can exceed the
certificate or 60% limit, so those are checked too.

The first SR sells nothing, so a second one follows from the state at its
end, opened by hand: sales, presidency changes, and must-sell turns. Uniform
play would rarely see everyone pass in succession once sales refill cash, so
it passes half the time it may.
"""

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.market import MARKET
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.turn import TURN

from tests.g1889.phases.conftest import (
    apply_and_verify, enter_stock_round, exact_id, legal_ids, new_game, policy_ids,
)

SEEDS = range(40)
MAX_DECISIONS = 3000
EXACT_SHARE = 0.25                 # of BID and RAISE decisions, with the flags off
UNCAPPED_PAYOUTS = 3               # payout rounds before the flags-off player turns the cap on
SECOND_SR_PASS = 0.5


def _choose(state, rng, flags, pass_share=0.0):
    phase = A.get_decision_phase_py(state)
    if not flags and phase in (K.DPHASE_BID, K.DPHASE_RAISE) and rng.random() < EXACT_SHARE:
        p = TURN.pending_private(state) if phase == K.DPHASE_BID else TURN.auction_private(state)
        low, high = PRIVATES[p].min_bid(state), PLAYERS[TURN.active_player(state)].available(state, p)
        return exact_id(state, int(rng.integers(low, high + 1)))
    ids = policy_ids(state) if state.auction_pass_payout_cap else legal_ids(state)
    if phase == K.DPHASE_STOCK and ids[0] == K.STOCK_PASS and rng.random() < pass_share:
        return K.STOCK_PASS
    return ids[int(rng.integers(len(ids)))]


def _over_a_limit(state, player):
    """Over the certificate limit, or above 60% of a corp outside the orange zone."""
    n = TURN.num_players(state)
    return PLAYERS[player].cert_count(state) > D.CERT_LIMIT[n] or any(
        10 * PLAYERS[player].shares(state, c) > D.MAX_OWNERSHIP_PERCENT and not MARKET.unlimited(state, c)
        for c in range(D.NUM_CORPS))


def _play(state, rng, flags, msg, pass_share=0.0, check=None):
    """Random decisions until the SR ends; then the state goes back to where
    the OR that follows began, and every decision applied until then, forced
    ones included, is returned."""
    decisions = []
    for _ in range(MAX_DECISIONS):
        if TURN.round(state) == K.ROUND_AUCTION and TURN.auction_pass_payouts(state) >= UNCAPPED_PAYOUTS:
            state.auction_pass_payout_cap = 1
        result = apply_and_verify(state, _choose(state, rng, flags, pass_share), msg)
        end = next((i for i, (_, phase, step) in enumerate(result.history)
                    if phase < 0 and step == K.STEP_OR_START), None)
        decisions += [(phase, i) for _, phase, i in result.history[:end] if phase >= 0]
        if check is not None:
            for i in range(len(result.history[:end])):
                check(result.state_at(i))
        if end is not None:
            np.copyto(state._array, result.history[end][0])  # keeps the state's policy flags
            return decisions
        if check is not None:
            check(state)
    pytest.fail(f'{msg}: the SR did not end within {MAX_DECISIONS} decisions')


def _first_sr_limits(state):
    if TURN.round(state) == K.ROUND_STOCK:
        assert not any(_over_a_limit(state, i) for i in range(TURN.num_players(state)))


def _must_sell_offers_only_sales(state):
    if A.get_decision_phase_py(state) != K.DPHASE_STOCK:
        return
    legal = legal_ids(state)
    if K.STOCK_PASS not in legal:
        assert legal and all(i >= K.STOCK_SELL for i in legal) and _over_a_limit(state, TURN.active_player(state))


@pytest.mark.parametrize('flags', [False, True], ids=['flags_off', 'flags_on'])
@pytest.mark.parametrize('num_players', range(D.MIN_PLAYERS, D.MAX_PLAYERS + 1))
def test_random_play_reaches_the_first_or(num_players, flags):
    second_sr = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        state = new_game(num_players, max_players=D.MAX_PLAYERS if seed % 2 else 0, seed=seed, priority_player=-1,
                         auction_pass_payout_cap=1 if flags else 0, same_president=flags)
        _play(state, rng, flags, f'seed {seed}', check=_first_sr_limits)
        assert (TURN.round(state), TURN.step(state)) == (K.ROUND_OPERATING, K.STEP_OR_START)
        assert TURN.or_count(state) == D.PHASE_ORS[0]
        # A second SR from here, with sales.
        enter_stock_round(state, TURN.priority_player(state), turn=2)
        second_sr += _play(state, rng, flags, f'seed {seed}, SR 2', SECOND_SR_PASS, check=_must_sell_offers_only_sales)
    kinds = {A.decode_action_py(phase, i).type for phase, i in second_sr if phase == K.DPHASE_STOCK}
    assert A.ACTION_SELL in kinds  # the second SRs did sell
