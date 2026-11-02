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
  result, and that a returned decision is a choice: two or more policy
  actions, or an amount decision with exact amounts beyond its one point,
  unless the driver rests at its development stop (``at_development_stop``).
- ``enter_auction(state, turn_player, sold, bids)``: hand-built auction
  positions through ``setup_auction_py``.
- ``hold(state, corp, ...)`` and ``enter_stock_round(state, first, turn)``:
  hand-built SR positions through ``setup_stock_round_py``.
- ``enter_operating_round(state, or_num)``: an OR on hand-built holdings and
  map, through ``setup_operating_round_py`` and ``DRIVER.advance``, verified
  as ``apply_and_verify`` does.
"""

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_OK
from g1889.entities.bank import BANK
from g1889.entities.corp import CORPS, PRESIDENT_SHARES, SHARES_PER_CORP
from g1889.entities.hexmap import MAP
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES, topmost
from g1889.entities.turn import TURN
from g1889.phases.auction import setup_auction_py
from g1889.phases.operating import setup_operating_round_py
from g1889.phases.stock import setup_stock_round_py
from g1889.state import GameState

POOL_LIMIT = D.MARKET_SHARE_LIMIT // 10
# The steps of a corp's turn, from TURN_START to TURN_END, as 6.5 builds them.
TURN_STEPS = (K.STEP_TRACK, K.STEP_TILE, K.STEP_TOKEN, K.STEP_ROUTE, K.STEP_DIVIDEND, K.STEP_BUY_TRAIN,
              K.STEP_BUY_PRIVATE)
OR_STEPS = (K.STEP_OR_START, K.STEP_TURN_START, *TURN_STEPS, K.STEP_TURN_END, K.STEP_OR_END)


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


def end_auction(state, owners=()):
    """Give the privates in play to ``owners`` (a player per private, cycling;
    seat 0 by default) for free, leaving the auction behind."""
    owners = list(owners) or [0]
    in_play = [p for p in range(D.NUM_PRIVATES) if PRIVATES[p].owner(state) == K.OWNER_UNSOLD]
    for i, p in enumerate(in_play):
        PRIVATES[p].sell_to_player(state, owners[i % len(owners)], 0)


def hold(state, corp, holdings, par=0, cell=None, pool=0, president=None):
    """Hand-built holdings of a parred ``corp``: ``holdings`` maps players to
    shares (the president's include the certificate's two), ``par`` is the par
    index, ``cell`` the market cell (the par cell by default), and the IPO keeps
    the rest. The president defaults to the first largest holder; the corp
    floats, with its capital from the bank, once half its shares are out."""
    c = CORPS[corp]
    c.set_par_index(state, par)
    MARKET.set_position(state, corp, *(cell or (D.PAR_ROW[par], D.PAR_COL[par])))
    for player, shares in holdings.items():
        PLAYERS[player].set_shares(state, corp, shares)
    c.set_pool_shares(state, pool)
    c.set_ipo_shares(state, SHARES_PER_CORP - pool - sum(holdings.values()))
    if president is None:
        president = max(holdings, key=lambda p: (holdings[p], -p))
    c.set_president(state, president)
    assert holdings[president] >= PRESIDENT_SHARES
    c.float_if_ready(state)


def enter_stock_round(state, first, turn=2):
    """Open SR ``turn`` at ``first``'s turn on the holdings set up by hand."""
    TURN.set_turn(state, turn)
    setup_stock_round_py(state, first)
    assert_invariants(state, 'enter_stock_round')


def enter_operating_round(state, or_num=1, or_count=None, msg='enter_operating_round'):
    """Open OR ``or_num`` of a set of ``or_count`` (the phase's by default) on
    the holdings and map set up by hand, and run it to the first choice."""
    TURN.set_or_count(state, or_count or D.PHASE_ORS[PHASE.current(state)])
    setup_operating_round_py(state, or_num)
    history = []
    status = DRIVER.advance(state, history)
    return _verify(ApplyResult(state, history, status), msg)


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
    cash, privates sell top-down, and the auction fields agree with the step.
    Every corp's ten shares are somewhere, at most five in the pool; a parred
    corp sits on the market, its president holds the certificate's two shares
    and no fewer than anyone, and it has floated once half its shares are out."""
    n = TURN.num_players(state)
    total = BANK.cash(state) + sum(PLAYERS[i].cash(state) for i in range(n)) + sum(c.cash(state) for c in CORPS)
    assert total == D.BANK_CASH, f'{msg}: cash totals {total}'
    for c in CORPS:
        held = [PLAYERS[i].shares(state, c.corp_id) for i in range(n)]
        assert c.ipo_shares(state) + c.pool_shares(state) + sum(held) == SHARES_PER_CORP, f'{msg}: {c.sym} shares'
        assert c.pool_shares(state) <= POOL_LIMIT, f'{msg}: {c.sym} pool over the limit'
        if c.par_index(state) < 0:
            assert c.president(state) == -1 and MARKET.row(state, c.corp_id) == -1 and not c.floated(state), msg
            continue
        president = c.president(state)
        assert MARKET.row(state, c.corp_id) >= 0, f'{msg}: {c.sym} parred off the market'
        assert held[president] >= PRESIDENT_SHARES and held[president] == max(held), f'{msg}: {c.sym} president'
        assert c.floated(state) == (c.ipo_shares(state) <= SHARES_PER_CORP // 2), f'{msg}: {c.sym} float'
    owners = [p.owner(state) for p in PRIVATES]
    in_play = [o for o in owners if o != K.OWNER_NOT_IN_PLAY]
    unsold_last = sorted(in_play, key=lambda o: o == K.OWNER_UNSOLD)
    assert in_play == unsold_last, f'{msg}: privates sold out of order {owners}'
    assert all(K.OWNER_NOT_IN_PLAY <= o < K.OWNER_PLAYER + n or K.OWNER_CORP <= o < K.OWNER_CORP + D.NUM_CORPS
               for o in owners), f'{msg}: private owners {owners}'
    for i in range(n):
        player = PLAYERS[i]
        assert player.committed(state) <= player.cash(state), f'{msg}: player {i} bids beyond cash'
        for p in range(D.NUM_PRIVATES):
            if player.bid(state, p):
                assert owners[p] == K.OWNER_UNSOLD, f'{msg}: player {i} bids on sold private {p}'
                assert player.bid(state, p) >= D.PRIVATE_VALUE[p] + D.MIN_BID_INCREMENT, f'{msg}: low bid'
    _assert_map(state, msg)

    step, top, active = TURN.step(state), topmost(state), TURN.active_player(state)
    if TURN.round(state) != K.ROUND_AUCTION:
        assert top == -1 and all(PLAYERS[i].committed(state) == 0 for i in range(n)), f'{msg}: auction left open'
        assert TURN.auction_private(state) == -1 and TURN.pending_private(state) == -1, msg
        if TURN.round(state) == K.ROUND_STOCK:
            assert step in (K.STEP_STOCK, K.STEP_PAR, K.STEP_SR_END), f'{msg}: step {step} in the SR'
            assert (TURN.pending_corp(state) >= 0) == (step == K.STEP_PAR), f'{msg}: pending corp'
            sold = TURN.sr_sold_mask(state)
            assert sold & ~PLAYERS[active].sold_mask(state) == 0, f'{msg}: turn sales outside the SR sales'
            assert not TURN.sr_sold_then_bought(state) or (sold and TURN.sr_bought(state)), msg
            if TURN.turn(state) == 1:
                assert all(PLAYERS[i].sold_mask(state) == 0 for i in range(n)), f'{msg}: sales in the first SR'
        else:  # the SR's fields clear when it ends
            assert TURN.sr_bought(state) == TURN.sr_sold_mask(state) == TURN.sr_sold_then_bought(state) == 0, msg
            assert all(PLAYERS[i].sold_mask(state) == 0 for i in range(n)), f'{msg}: SR sales left over'
        _assert_operating(state, msg)
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


def _assert_map(state, msg):
    """No supply below zero, and every charter token is left or placed, at most one per hex."""
    assert all(MAP.supply(state, t) >= 0 for t in range(D.NUM_SUPPLY_TILES)), f'{msg}: negative supply'
    for c in CORPS:
        placed = sum(MAP.tokens(state, h) >> c.corp_id & 1 for h in range(D.NUM_HEXES))
        assert c.tokens_left(state) + placed == D.CORP_TOKENS[c.corp_id], f'{msg}: {c.sym} tokens'


def _assert_operating(state, msg):
    """The OR order lists distinct floated corps, the operating corp is the one
    at ``or_index`` and its president acts, TILE alone has a pending hex, and
    every OR field is clear outside the OR."""
    step, corp = TURN.step(state), TURN.operating_corp(state)
    order = [TURN.or_order(state, i) for i in range(D.NUM_CORPS)]
    corps = [c for c in order if c >= 0]
    assert order == corps + [-1] * (D.NUM_CORPS - len(corps)) and len(set(corps)) == len(corps), f'{msg}: {order}'
    assert (TURN.pending_hex(state) >= 0) == (step == K.STEP_TILE), f'{msg}: pending hex'
    assert TURN.route_revenue(state) == 0, f'{msg}: route revenue'  # until 6.6
    if TURN.round(state) != K.ROUND_OPERATING:
        assert corps == [] and corp == -1 and TURN.or_index(state) == 0, f'{msg}: OR fields outside the OR'
        return
    assert step in OR_STEPS, f'{msg}: step {step} in the OR'
    assert all(CORPS[c].floated(state) for c in corps), f'{msg}: an unfloated corp in {corps}'
    if step in TURN_STEPS or step == K.STEP_TURN_END:
        assert corp == order[TURN.or_index(state)] >= 0, f'{msg}: operating corp {corp}, order {order}'
        assert TURN.active_player(state) == CORPS[corp].president(state), f'{msg}: the president acts'
        assert MAP.tokens(state, D.CORP_HOME[corp]) >> corp & 1, f'{msg}: no home token'
    else:
        assert corp == -1, f'{msg}: operating corp {corp} between turns'


def assert_action_sets(state, msg=''):
    """Policy ids are legal, and present whenever legal ids are."""
    legal, policy = legal_ids(state), policy_ids(state)
    assert set(policy) <= set(legal) and (policy or not legal), f'{msg}: legal {legal}, policy {policy}'


def is_choice(state):
    """Whether the current decision is one the driver returns rather than
    applies: two or more policy actions, or an amount decision whose one legal
    grid point has legal exact amounts above it (engine-design §6)."""
    policy = get_policy_actions(state)
    if len(policy) >= 2:
        return True
    phase = A.get_decision_phase_py(state)
    if not A.has_exact_amounts(phase) or len(legal_ids(state)) != 1 or policy[0][1].type != A.ACTION_PRICE_POINT:
        return False
    return A.is_legal_action_py(state, exact_id(state, price_of(state, policy[0][1].count) + 1))


def price_of(state, point):
    """The amount of BID or RAISE grid point ``point``."""
    phase = A.get_decision_phase_py(state)
    p = TURN.pending_private(state) if phase == K.DPHASE_BID else TURN.auction_private(state)
    return PRIVATES[p].min_bid(state) + D.MIN_BID_INCREMENT * point


# =============================================================================
# APPLY + VERIFY
# =============================================================================

def at_development_stop(state):
    """Whether the driver rests where a later sub-step takes over (its TEMPORARY
    development stop, removed at 6.9): an automated step, a decision whose
    legal set raises NotImplementedError, or a forced decision, whose handler
    raised it."""
    if TURN.round(state) == K.ROUND_GAME_OVER:
        return False
    if A.get_decision_phase_py(state) < 0:
        return True
    try:
        return DRIVER.forced_action_py(state) >= 0
    except NotImplementedError:
        return True


class ApplyResult:
    """``history`` holds ``(row, decision phase, id)`` before each applied
    decision and ``(row, -1, step)`` before each automated step."""

    def __init__(self, state, history, status):
        self.state, self.history, self.status = state, history, status

    @property
    def decisions(self):
        return [(phase, action_id) for _, phase, action_id in self.history if phase >= 0]

    @property
    def steps(self):
        """The automated steps the driver ran."""
        return [step for _, phase, step in self.history if phase < 0]

    @property
    def forced(self):
        """The decisions the driver applied after the given one."""
        return self.decisions[1:]

    def state_at(self, index):
        return GameState.from_array(self.history[index][0], TURN.num_players(self.state),
                                    max_players=self.state.max_players)

    def state_before(self, step):
        """The state as automated step ``step`` first began, so the one before
        OR_START is the SR's end."""
        return self.state_at(next(i for i, (_, phase, s) in enumerate(self.history) if phase < 0 and s == step))


def _verify(result, msg):
    for i in range(len(result.history)):
        before = result.state_at(i)
        assert_invariants(before, f'{msg}: before entry {i} {result.history[i][1:]}')
        if result.history[i][1] >= 0:
            assert_action_sets(before, f'{msg}: before decision {i}')
    assert_invariants(result.state, f'{msg}: after')
    if not at_development_stop(result.state) and TURN.round(result.state) != K.ROUND_GAME_OVER:
        assert_action_sets(result.state, f'{msg}: after')
        assert is_choice(result.state), f'{msg}: returned a forced decision {get_policy_actions(result.state)}'
    return result


def apply_and_verify(state, action_id, msg='', expected_status=STATUS_OK):
    """Apply a legal id (enumerated, or an exact amount in range) through the
    driver and check every state it passed through."""
    assert A.is_legal_action_py(state, action_id), f'{msg}: id {action_id} is not legal; legal {legal_ids(state)}'
    history = []
    status = DRIVER.apply_action(state, action_id, history)
    assert status == expected_status, f'{msg}: status {status}, expected {expected_status}'
    return _verify(ApplyResult(state, history, status), msg)
