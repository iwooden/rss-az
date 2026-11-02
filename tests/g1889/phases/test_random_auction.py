"""Random auctions (1889/engine-design.md §10.4): random play from a fresh game
reaches the stock round within a step cap at 2-6 players, with the policy
flags on (policy ids only, as in self-play) and off (every legal id, and exact
amounts in BID and RAISE). ``apply_and_verify`` checks money and the auction
invariants on every state the driver passes through, forced decisions
included, and that every returned decision is a choice.

Uniform legal play can keep rules §12.9's endless auction going: players who
outbid themselves on later privates never afford the topmost. Ruby allows
it, so with the flags off the player turns the pass cap on after a few
payout rounds; with them on, the cap ends it from the first.
"""

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.turn import TURN

from tests.g1889.phases.conftest import apply_and_verify, exact_id, legal_ids, new_game, policy_ids

SEEDS = range(100)
MAX_DECISIONS = 2000
EXACT_SHARE = 0.25                 # of BID and RAISE decisions, with the flags off
UNCAPPED_PAYOUTS = 3               # payout rounds before the flags-off player turns the cap on


def _choose(state, rng, flags):
    phase = A.get_decision_phase_py(state)
    if not flags and phase in (K.DPHASE_BID, K.DPHASE_RAISE) and rng.random() < EXACT_SHARE:
        p = TURN.pending_private(state) if phase == K.DPHASE_BID else TURN.auction_private(state)
        low, high = PRIVATES[p].min_bid(state), PLAYERS[TURN.active_player(state)].available(state, p)
        return exact_id(state, int(rng.integers(low, high + 1)))
    ids = policy_ids(state) if state.auction_pass_payout_cap else legal_ids(state)
    return ids[int(rng.integers(len(ids)))]


@pytest.mark.parametrize('flags', [False, True], ids=['flags_off', 'flags_on'])
@pytest.mark.parametrize('num_players', range(D.MIN_PLAYERS, D.MAX_PLAYERS + 1))
def test_random_auctions_reach_the_stock_round(num_players, flags):
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        state = new_game(num_players, max_players=D.MAX_PLAYERS if seed % 2 else 0, seed=seed, priority_player=-1,
                         auction_pass_payout_cap=1 if flags else 0, same_president=flags)
        for _ in range(MAX_DECISIONS):
            if TURN.round(state) != K.ROUND_AUCTION:
                break
            if TURN.auction_pass_payouts(state) >= UNCAPPED_PAYOUTS:
                state.auction_pass_payout_cap = 1
            apply_and_verify(state, _choose(state, rng, flags), f'seed {seed}')
        else:
            pytest.fail(f'seed {seed}: the auction did not end within {MAX_DECISIONS} decisions')
        # The SR opens with the auction's turn player, and the driver passes
        # for each player who can do nothing else, as Ruby skips them; when
        # nobody can, the SR ends at once and priority stays.
        assert TURN.priority_player(state) == TURN.auction_turn(state)
        if TURN.round(state) == K.ROUND_OPERATING:
            assert TURN.step(state) == K.STEP_OR_START
        else:
            assert (TURN.round(state), TURN.step(state)) == (K.ROUND_STOCK, K.STEP_STOCK)
            skipped = (TURN.active_player(state) - TURN.priority_player(state)) % num_players
            assert all(PLAYERS[(TURN.priority_player(state) + k) % num_players].passed(state)
                       for k in range(skipped))
        owners = [p.owner(state) for p in PRIVATES if p.owner(state) != K.OWNER_NOT_IN_PLAY]
        assert all(K.OWNER_PLAYER <= o < K.OWNER_PLAYER + num_players for o in owners)
