"""Shared fixtures and helpers for the 1889 phase tests (as tests/phases/conftest.py).

- ``game_state``: a fresh game at the auction, 2-6 players, policy flags off,
  the first seat first.
- ``new_game(num_players, ...)``: the same for one player count, with flags.
- ``get_legal_actions(state)``, ``get_policy_actions(state)``: decoded
  ``(id, info)`` lists of the engine-legal and policy ids.
- ``find_legal_action(state, **fields)``: the first legal id whose decoded
  fields (``type``, ``private``, ``count``, ...) match.
- ``apply_and_verify(state, action_id)``: apply through the driver with
  history, checking ``assert_invariants`` on every intermediate state and the
  result, and that a returned decision has two or more policy actions.
- ``enter_auction(state, turn_player, sold, bids)``: hand-built auction
  positions through ``setup_auction_py``.
"""

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_OK
from g1889.entities.bank import BANK
from g1889.entities.corp import CORPS
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES, topmost
from g1889.entities.turn import TURN
from g1889.phases.auction import setup_auction_py
from g1889.state import GameState

# Decision phases whose handlers have landed (6.3); later phases raise NotImplementedError.
IMPLEMENTED_PHASES = (K.DPHASE_AUCTION, K.DPHASE_BID, K.DPHASE_RAISE)


# =============================================================================
# STATES
# =============================================================================

def new_game(num_players, max_players=0, seed=42, priority_player=0, auction_pass_payout_cap=0,
             same_president=False):
    """A fresh game at the auction, by default with the first seat first;
    ``priority_player`` -1 lets the seed pick."""
    state = GameState(num_players, max_players=max_players, auction_pass_payout_cap=auction_pass_payout_cap,
                      private_sale_same_president=same_president, train_sale_same_president=same_president)
    state.initialize_game(num_players, seed=seed, priority_player=priority_player)
    return state


@pytest.fixture(params=[2, 3, 4, 5, 6])
def game_state(request):
    return new_game(request.param)


def enter_auction(state, turn_player, sold=(), bids=()):
    """Sell ``sold`` ``(private, player, price)`` and place ``bids``
    ``(player, private, amount)`` through the entity APIs, then enter the
    auction at ``turn_player``'s normal turn, resolving the topmost as after a
    sale (so bids on it sell it or open its closed auction)."""
    for private, player, price in sold:
        PRIVATES[private].sell_to_player(state, player, price)
    for player, private, amount in bids:
        PLAYERS[player].set_bid(state, private, amount)
    setup_auction_py(state, turn_player)
    assert_invariants(state, 'enter_auction')


# =============================================================================
# ACTIONS
# =============================================================================

def _decoded(state, enumerate_ids):
    phase = A.get_decision_phase_py(state)
    assert phase >= 0, f'not at a decision (step {K.Step(TURN.step(state)).name})'
    ids = np.zeros(K.MAX_ACTION_SIZE, dtype=np.uint16)
    count = enumerate_ids(state, ids)
    return [(int(i), A.decode_action_py(phase, int(i))) for i in ids[:count]]


def get_legal_actions(state):
    return _decoded(state, A.enumerate_legal_actions_py)


def get_policy_actions(state):
    return _decoded(state, A.enumerate_policy_actions_py)


def legal_ids(state):
    return [action_id for action_id, _ in get_legal_actions(state)]


def policy_ids(state):
    return [action_id for action_id, _ in get_policy_actions(state)]


def find_legal_action(state, **fields):
    """The first legal id whose decoded info matches every given field."""
    actions = get_legal_actions(state)
    for action_id, info in actions:
        if all(getattr(info, name) == value for name, value in fields.items()):
            return action_id
    raise AssertionError(f'no legal action with {fields}; legal: {[info for _, info in actions]}')


def exact_id(state, amount):
    """The exact-amount id of ``amount`` in the current BID or RAISE."""
    return K.PHASE_ACTION_SIZES[A.get_decision_phase_py(state)] + amount


# =============================================================================
# INVARIANTS
# =============================================================================

def assert_invariants(state, msg=''):
    """Money is conserved, bids sit on unsold privates within each player's
    cash, privates sell top-down, and the auction fields agree with the step."""
    n = TURN.num_players(state)
    total = BANK.cash(state) + sum(PLAYERS[i].cash(state) for i in range(n)) + sum(c.cash(state) for c in CORPS)
    assert total == D.BANK_CASH, f'{msg}: cash totals {total}'
    owners = [p.owner(state) for p in PRIVATES]
    in_play = [o for o in owners if o != K.OWNER_NOT_IN_PLAY]
    unsold_last = sorted(in_play, key=lambda o: o == K.OWNER_UNSOLD)
    assert in_play == unsold_last, f'{msg}: privates sold out of order {owners}'
    for i in range(n):
        player = PLAYERS[i]
        assert player.committed(state) <= player.cash(state), f'{msg}: player {i} bids beyond cash'
        for p in range(D.NUM_PRIVATES):
            if player.bid(state, p):
                assert owners[p] == K.OWNER_UNSOLD, f'{msg}: player {i} bids on sold private {p}'
                assert player.bid(state, p) >= D.PRIVATE_VALUE[p] + D.MIN_BID_INCREMENT, f'{msg}: low bid'

    step, top, active = TURN.step(state), topmost(state), TURN.active_player(state)
    if TURN.round(state) != K.ROUND_AUCTION:
        assert top == -1 and all(PLAYERS[i].committed(state) == 0 for i in range(n)), f'{msg}: auction left open'
        assert TURN.auction_private(state) == -1 and TURN.pending_private(state) == -1, msg
        return
    assert step in (K.STEP_AUCTION, K.STEP_BID, K.STEP_RAISE), f'{msg}: step {step} in the auction'
    if step == K.STEP_RAISE:
        auctioned = PRIVATES[TURN.auction_private(state)]
        assert TURN.auction_private(state) == top and auctioned.bid_count(state) >= 2, f'{msg}: closed auction'
        assert active == auctioned.lowest_bidder(state), f'{msg}: the lowest bidder acts in RAISE'
        return
    assert TURN.auction_private(state) == -1 and PRIVATES[top].bid_count(state) == 0, f'{msg}: topmost has bids'
    assert active == TURN.auction_turn(state), f'{msg}: the turn player acts'
    assert (TURN.pending_private(state) >= 0) == (step == K.STEP_BID), f'{msg}: pending private'


def assert_action_sets(state, msg=''):
    """Policy ids are legal, and present whenever legal ids are."""
    legal, policy = legal_ids(state), policy_ids(state)
    assert set(policy) <= set(legal) and (policy or not legal), f'{msg}: legal {legal}, policy {policy}'


# =============================================================================
# APPLY + VERIFY
# =============================================================================

class ApplyResult:
    """``history`` holds ``(row, decision phase, id)`` before each applied decision."""

    def __init__(self, state, history, status):
        self.state, self.history, self.status = state, history, status

    @property
    def decisions(self):
        return [(phase, action_id) for _, phase, action_id in self.history]

    @property
    def forced(self):
        """The decisions the driver applied after the given one."""
        return self.decisions[1:]

    def state_at(self, index):
        return GameState.from_array(self.history[index][0], TURN.num_players(self.state),
                                    max_players=self.state.max_players)


def apply_and_verify(state, action_id, msg='', expected_status=STATUS_OK):
    """Apply a legal id (enumerated, or an exact amount in range) through the
    driver and check every state it passed through."""
    assert A.is_legal_action_py(state, action_id), f'{msg}: id {action_id} is not legal; legal {legal_ids(state)}'
    history = []
    status = DRIVER.apply_action(state, action_id, history)
    assert status == expected_status, f'{msg}: status {status}, expected {expected_status}'
    result = ApplyResult(state, history, status)
    for i in range(len(history)):
        before = result.state_at(i)
        assert_invariants(before, f'{msg}: before decision {i} {history[i][1:]}')
        assert_action_sets(before, f'{msg}: before decision {i}')
    assert_invariants(state, f'{msg}: after')
    if A.get_decision_phase_py(state) in IMPLEMENTED_PHASES:
        assert_action_sets(state, f'{msg}: after')
        assert len(policy_ids(state)) >= 2, f'{msg}: returned a forced decision {get_policy_actions(state)}'
    return result
