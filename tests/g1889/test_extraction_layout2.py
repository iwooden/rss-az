"""1889 token layout 2 (1889/v1-design.md §5; token-data-1889.md).

On the states of whole games (the layout-1 fixture's self-play, human, and
random games, and random playouts; ``tests/g1889/test_cache_purity.games``):

- **shape:** the sizes and widths, slot 0 and the padding, the selections,
  and batches against single states;
- **against layout 1,** which is pinned and audited: every slot layout 2
  keeps equals layout 1's, moved or rescaled as the spec says;
- **per-choice truth:** every new or changed feature that describes a choice
  equals what the state holds after that choice is really made (the action
  through the driver, or the purchase through the entity API where the
  feature is shown before it is affordable): the prices after sales,
  payouts, withholds, and a sold-out SR; the route max after a purchase,
  phase changes and trade-ins included, for the buyer and the seller; the
  shortfalls; what a float costs; the legal emergency sale; a candidate's Δ
  with the next train; and the rest of §5's additions;
- **ranges:** every slot stays within its bound, and every slot of layout 2
  is exercised.

``tests/g1889/test_cache_purity.py`` checks that layout 2 does not depend on
the caches, and ``tests/g1889/test_token_data_doc.py`` the slot tables.
"""

import math

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_INVALID
from g1889.entities.bank import BANK, BANK_ACCOUNT
from g1889.entities.corp import CORPS
from g1889.entities.hexmap import MAP
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.routes.solver import max_route_revenue
from g1889.state import GameState, get_global_fields, get_layout
from g1889.token_data import (
    TokenDataSize,
    TokenRow as R,
    TokenWidth2,
    get_num_tokens,
    get_token_data,
    get_token_data_batch,
    get_token_dim,
    get_token_widths,
)

from tests.g1889.phases.conftest import legal_ids
from tests.g1889.test_cache_purity import games
from tests.g1889.test_layout1_parity import apply_op, new_state

DIM = int(TokenDataSize.TOKEN_DIM_2)
GLOBALS = get_global_fields()
GLOBAL_OFFSET = get_layout(0).global_offset
WIDTHS = {'GLOBAL': 90, 'PRICE': 30, 'DIVIDEND': 13, 'CORP': 77, 'PRIVATE': 18, 'TRAIN': 15, 'HEX': 105,
          'CANDIDATE': 109, 'PLAYER': 40}
ROWS = {'GLOBAL': (R.ROW_GLOBAL, 1), 'PRICE': (R.ROW_PRICE, 1), 'DIVIDEND': (R.ROW_DIVIDEND, 1),
        'CORP': (R.ROW_CORP, D.NUM_CORPS), 'PRIVATE': (R.ROW_PRIVATE, D.NUM_PRIVATES),
        'TRAIN': (R.ROW_TRAIN, D.NUM_TRAIN_TYPES), 'HEX': (R.ROW_HEX, D.NUM_HEXES),
        'CANDIDATE': (R.ROW_CANDIDATE, D.MAX_LAY_CANDIDATES), 'PLAYER': (R.ROW_PLAYER, None)}
FLOAT_SHARES = 5
LOWEST_FLOAT = FLOAT_SHARES * min(D.PAR_PRICE)
SUB_PHASES = (K.DPHASE_PRIVATE_PRICE, K.DPHASE_ER_BONUS, K.DPHASE_PORT)
STRIDE = 3  # the states the general tests read; the decisions a test is about are all read


def f32(numerator, divisor):
    """A slot's value as the fill computes it, in float32."""
    return np.float32(numerator) / np.float32(divisor)


def log_yen(yen):
    return np.float32(math.log1p(max(yen, 0) / 100.0)) / np.float32(5)


def state_of(row):
    return GameState.from_buffer(row, int(row[GLOBAL_OFFSET + GLOBALS.num_players]),
                                 max_players=(len(row) - get_layout(0).players_offset) // get_layout(0).player_size)


def copy_of(state):
    return state_of(state._array.copy())


def tokens_of(state, layout=2):
    tokens = np.full((get_num_tokens(state.max_players), get_token_dim(layout)), np.nan, np.float32)
    get_token_data(copy_of(state), tokens, state.max_players, layout_version=layout)
    return tokens


def applied(state, action):
    """A copy with ``action`` applied, before the driver's automated steps
    and forced decisions."""
    after = copy_of(state)
    history = []
    assert DRIVER.apply_action(after, action, history) != STATUS_INVALID
    return state_of(history[1][0]) if len(history) > 1 else after


def rows(name, max_players):
    first, count = ROWS[name]
    return slice(first, first + (max_players if count is None else count))


def bought(state, corp, train_type, trade_in=None):
    """A copy after ``corp`` buys a ``train_type`` through the entity API (a
    depot train if the depot sells the type, else the pool's; a D for a traded
    ``trade_in``), with the cash for it, and the phase the purchase starts."""
    after = copy_of(state)
    BANK.transfer(after, BANK_ACCOUNT, K.OWNER_CORP + corp, D.TRAIN_PRICE[train_type])
    if trade_in is not None:
        TRAINS.trade_in_for_d(after, corp, trade_in)
    elif TRAINS.depot_types(after) >> train_type & 1:
        TRAINS.buy_from_depot(after, corp, train_type)
    else:
        TRAINS.buy_from_pool(after, corp, train_type)
    PHASE.on_train_bought(after, train_type)
    return after


def next_rust_phase(phase):
    """The first later phase whose train rusts a rank, or None."""
    return next((p for p in range(phase + 1, D.NUM_GAME_PHASES) if D.PHASE_ON_TRAIN[p] in D.TRAIN_RUSTS_ON), None)


def has_room(state, corp):
    return TRAINS.corp_total(state, corp) < TRAINS.limit(state)


class Sample:
    def __init__(self, kind, state):
        self.kind = kind
        self.state = copy_of(state)  # caches as the driver left them
        self.max_players = state.max_players
        self.phase = A.get_decision_phase_py(state)
        self.game_phase = PHASE.current(state)
        self.tokens = tokens_of(state)

    @property
    def corp(self):
        return TURN.operating_corp(self.state)

    def of(self, name):
        return self.tokens[rows(name, self.max_players)]


@pytest.fixture(scope='module')
def samples():
    """Every rest of every game, extracted in layout 2."""
    out = []
    for _, spec, calls in games():
        state = new_state(spec)
        out.append(Sample(spec['kind'], state))
        for call in calls:
            apply_op(state, *call)
            out.append(Sample(spec['kind'], state))
    return out


def at(samples, *phases):
    return [s for s in samples if s.phase in phases]


# =============================================================================
# SHAPE
# =============================================================================

def test_sizes_follow_the_row_map():
    assert get_token_dim(2) == DIM == 109 == max(WIDTHS.values())
    assert {name: int(TokenWidth2[f'TW2_{name}']) for name in WIDTHS} == WIDTHS
    for max_players in (3, 4, 5):
        expected = np.repeat(list(WIDTHS.values()), [1, 1, 1, 7, 7, 6, 52, 15, max_players])
        np.testing.assert_array_equal(get_token_widths(max_players, 2), expected)
        assert get_token_widths(max_players, 2).dtype == np.uint8


def test_a_buffer_of_the_other_layouts_width_is_refused():
    state = new_state({'num_players': 3, 'max_players': 3, 'seed': 1, 'priority_player': -1, 'flags': [False, False, 0]})
    with pytest.raises(AssertionError, match='buffer'):
        get_token_data(state, np.zeros((93, 108), np.float32), layout_version=2)
    with pytest.raises(AssertionError, match='buffer'):
        get_token_data(state, np.zeros((93, 109), np.float32), layout_version=1)


def test_visibility_padding_widths_and_selections(samples):
    """Slot 0 marks the rows layout 1 shows, hidden rows and the slots past a
    row's width are zero, every value is finite, and each selection slot is
    set on one row of its type at most."""
    seen = set()
    for s in samples[::STRIDE]:
        tokens, widths = s.tokens, get_token_widths(s.max_players, 2)
        assert np.isfinite(tokens).all()
        assert not any(tokens[i, widths[i]:].any() for i in range(len(tokens)))
        np.testing.assert_array_equal(tokens[:, 0], tokens_of(s.state, 1)[:, 0])
        assert not tokens[tokens[:, 0] == 0].any()
        for name, slots in {'CORP': (1, 2, 3, 57), 'PRIVATE': (1, 2), 'TRAIN': (1, 14), 'HEX': (1,),
                            'PLAYER': (1, 36)}.items():
            for slot in slots:
                assert s.of(name)[:, slot].sum() <= 1, (name, slot)
        seen.add((s.phase, s.game_phase))
    assert {phase for phase, _ in seen} == {-1, *range(K.NUM_DECISION_PHASES)}
    assert {game_phase for _, game_phase in seen} == set(range(D.NUM_GAME_PHASES))


def test_batches_match_single_states(samples):
    for capacity in (3, 5):
        picked = [s for s in samples[::7] if s.max_players == capacity]
        out = np.full((len(picked), get_num_tokens(capacity), DIM), np.nan, np.float32)
        get_token_data_batch([s.state._array.copy() for s in picked], out, capacity, layout_version=2)
        np.testing.assert_array_equal(out, np.stack([s.tokens for s in picked]))


def test_extraction_writes_only_the_caches(samples):
    """Layout 2 leaves every cache clean and changes nothing else in the row."""
    layout = get_layout(0)
    for s in samples[::STRIDE]:
        state = copy_of(s.state)
        tokens = np.empty_like(s.tokens)
        get_token_data(state, tokens, s.max_players, layout_version=2)
        fresh = copy_of(s.state)
        for field in ('conn_dirty', 'route_dirty', 'hyp_dirty'):
            fresh._array[GLOBAL_OFFSET + getattr(GLOBALS, field)] = (1 << D.NUM_CORPS) - 1  # test setup
        get_token_data(fresh, tokens, s.max_players, layout_version=2)
        np.testing.assert_array_equal(state._array, fresh._array)
        assert not any(state._array[GLOBAL_OFFSET + getattr(GLOBALS, f)] for f in ('conn_dirty', 'route_dirty', 'hyp_dirty'))
        changed = np.flatnonzero(state._array != s.state._array)
        caches = set(range(layout.graph_offset, layout.players_offset)) | set(range(layout.corps_offset, layout.privates_offset)) \
            | {GLOBAL_OFFSET + getattr(GLOBALS, f) for f in ('conn_dirty', 'route_dirty', 'hyp_dirty')}
        assert set(changed.tolist()) <= caches


# =============================================================================
# AGAINST LAYOUT 1
# =============================================================================

def test_the_slots_layout_2_keeps_are_layout_1s(samples):
    """GLOBAL 1-43 (39-40 only during the auction), PRICE, the DIVIDEND
    outcomes, CORP 1-37 and the moved 39-50, PRIVATE, TRAIN 1-9 and 12, HEX,
    CANDIDATE 1-107, and PLAYER, with the rescaled slots rescaled."""
    for s in samples[::STRIDE]:
        new, old = s.tokens, tokens_of(s.state, 1)
        auction = TURN.round(s.state) == K.ROUND_AUCTION

        def same(name, new_slots, old_slots, scale=1.0):
            got = new[rows(name, s.max_players)][:, new_slots]
            want = old[rows(name, s.max_players)][:, old_slots] * np.float32(scale)
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=0, err_msg=f'{name} {new_slots}')

        same('GLOBAL', slice(0, 39), slice(0, 39))
        same('GLOBAL', slice(41, 44), slice(41, 44))
        same('GLOBAL', slice(39, 41), slice(39, 41), 1.0 if auction else 0.0)
        same('PRICE', slice(0, 24), slice(0, 24))
        same('DIVIDEND', [0, 1, 2, 3, 4, 5, 6], [0, 1, 3, 4, 5, 6, 7])
        same('CORP', slice(0, 15), slice(0, 15))
        same('CORP', slice(20, 38), slice(20, 38))
        if TURN.round(s.state) == K.ROUND_OPERATING:
            same('CORP', [45], [39])
        same('CORP', slice(46, 49), slice(40, 43))
        same('CORP', [49], [43], 5 / 15)
        same('CORP', slice(50, 55), slice(44, 49))
        same('CORP', [56], [50], 3 / 7)
        same('PRIVATE', slice(0, 18), slice(0, 18))
        same('TRAIN', slice(0, 10), slice(0, 10))
        same('TRAIN', [12], [12])
        same('HEX', slice(0, 105), slice(0, 105))
        same('CANDIDATE', slice(0, 107), slice(0, 107))
        same('CANDIDATE', [107], [107], 25 / 50)
        same('PLAYER', slice(0, 11), slice(0, 11))
        same('PLAYER', slice(11, 25), slice(12, 26))
        same('PLAYER', slice(25, 31), slice(27, 33))
        same('PLAYER', [31, 32, 34], [33, 35, 37])
        same('PLAYER', [33], [36], 3 / 7)
        # Layout 1 fills the sale prices at the decisions that sell; layout 2 always, with the same values there.
        old_prices = old[rows('CORP', s.max_players)][:, 15:20]
        np.testing.assert_array_equal(new[rows('CORP', s.max_players)][:, 15:20][old_prices > 0], old_prices[old_prices > 0])


# =============================================================================
# PER-CHOICE TRUTH: THE MARKET
# =============================================================================

def test_market_prices_after_each_move(samples):
    """CORP 15-19 and 58-60 for every parred corp at every decision: the
    price after 1-5 shares sold (the bottom of a column repeats), after a
    payout, after a withhold, and after an SR ended sold out, each made with
    MARKET's move on a copy. Unparred corps have none."""
    for s in samples[::STRIDE]:
        for c in range(D.NUM_CORPS):
            row = s.of('CORP')[c]
            if CORPS[c].par_index(s.state) < 0:
                assert not row[15:20].any() and not row[58:61].any() and row[74] == 0
                continue
            sold = copy_of(s.state)
            for n in range(5):
                MARKET.move_down(sold, c)
                assert row[15 + n] == f32(MARKET.price(sold, c), 100)
            for slot, move in ((58, MARKET.move_right), (59, MARKET.move_left), (60, MARKET.move_up)):
                moved = copy_of(s.state)
                move(moved, c)
                assert row[slot] == f32(MARKET.price(moved, c), 100)
            assert row[74] == log_yen(MARKET.price(s.state, c))


def test_sales_payouts_and_withholds_move_the_price_as_shown(samples):
    """The same slots against the real actions: SELL_AMOUNT's and
    EMERGENCY_SELL_AMOUNT's blocks, and DIVIDEND's two choices."""
    seen = dict.fromkeys((K.DPHASE_SELL_AMOUNT, K.DPHASE_EMERGENCY_SELL_AMOUNT, K.DPHASE_DIVIDEND), 0)
    for s in at(samples, *seen):
        if s.phase == K.DPHASE_DIVIDEND:
            row = s.of('CORP')[s.corp]
            assert row[59] == f32(MARKET.price(applied(s.state, K.DIVIDEND_WITHHOLD), s.corp), 100)
            if K.DIVIDEND_PAYOUT in legal_ids(s.state):
                assert row[58] == f32(MARKET.price(applied(s.state, K.DIVIDEND_PAYOUT), s.corp), 100)
        else:
            sold = TURN.pending_corp(s.state)
            for action in legal_ids(s.state):
                assert s.of('CORP')[sold, 15 + action] == f32(MARKET.price(applied(s.state, action), sold), 100)
        seen[s.phase] += 1
    assert all(seen.values()), seen


def _passed_to_the_or(state):
    """The SR passed out on a copy, every player passing: the states just
    before and just after the next OR's OR_START (the SR's end has moved the
    sold-out corps up; OR_START sets the order), or None if a player may not
    pass or no OR follows."""
    after = copy_of(state)
    for _ in range(2 * TURN.num_players(state)):
        if A.get_decision_phase_py(after) != K.DPHASE_STOCK or K.STOCK_PASS not in legal_ids(after):
            return None
        history = []
        DRIVER.apply_action(after, K.STOCK_PASS, history)
        steps = [(phase, step) for _, phase, step in history]
        if (-1, K.STEP_OR_START) in steps:
            start = steps.index((-1, K.STEP_OR_START))
            return state_of(history[start][0]), state_of(history[start + 1][0]) if start + 1 < len(history) else after
    return None


def test_the_sold_out_price_and_the_operating_rank_are_the_next_ors(samples):
    """At STOCK, pass the SR out on a copy. Each sold-out corp is then at its
    CORP 60 price, and with no corp sold out the OR's order is CORP 38-44's:
    the floated corps by rank. A parred corp yet to float has the rank it
    would take: floated on a copy, it sorts there."""
    checked = ordered = unfloated = 0
    for s in at(samples, K.DPHASE_STOCK)[::2]:
        ranks = s.of('CORP')[:, 38:45]
        floated = [c for c in range(D.NUM_CORPS) if CORPS[c].floated(s.state)]
        assert ranks.sum(axis=1).tolist() == [float(CORPS[c].par_index(s.state) >= 0) for c in range(D.NUM_CORPS)]
        order = MARKET.sort_by_price(s.state, sum(1 << c for c in floated))
        assert [int(np.argmax(ranks[c])) for c in order] == list(range(len(order)))
        for c in range(D.NUM_CORPS):
            if CORPS[c].par_index(s.state) >= 0 and c not in floated:
                would = copy_of(s.state)
                CORPS[c].set_floated(would, True)  # test setup: as if it floated now
                assert MARKET.sort_by_price(would, sum(1 << f for f in [*floated, c])).index(c) == np.argmax(ranks[c])
                unfloated += 1
        passed = _passed_to_the_or(s.state)
        if passed is None:
            continue
        ended, started = passed
        sold_out = [c for c in range(D.NUM_CORPS) if CORPS[c].par_index(s.state) >= 0 and CORPS[c].sold_out(s.state)]
        for c in sold_out:
            assert s.of('CORP')[c, 60] == f32(MARKET.price(ended, c), 100)
        checked += 1
        if not sold_out:
            assert [TURN.or_order(started, i) for i in range(len(order))] == order
            ordered += 1
    assert checked > 100 and ordered > 20 and unfloated > 20


def test_the_operating_rank_in_an_or_is_the_ors_order(samples):
    for s in samples[::STRIDE]:
        if TURN.round(s.state) != K.ROUND_OPERATING:
            continue
        order = [TURN.or_order(s.state, i) for i in range(D.NUM_CORPS)]
        for c in range(D.NUM_CORPS):
            row = s.of('CORP')[c]
            assert row[38:45].tolist() == [float(order[i] == c) for i in range(D.NUM_CORPS)]
            assert row[45] == (c in order and order.index(c) < TURN.or_index(s.state))
            assert row[57] == (c == TURN.operating_corp(s.state))
    discards = at(samples, K.DPHASE_DISCARD)
    assert any(s.of('CORP')[:, 57].argmax() != s.of('CORP')[:, 1].argmax() for s in discards)  # the operator, not the discarder


# =============================================================================
# PER-CHOICE TRUTH: TRAINS
# =============================================================================

def test_the_hypothetical_route_maxes_are_those_after_the_rust_and_the_purchase(samples):
    """CORP 65 and 66, for every corp: its route max once the next rusting
    phase has started (PHASE.advance on a copy), and after it buys the next
    train (the entity purchase and ``on_train_bought`` on a copy) while it
    has room; its route max now otherwise. CORP 61-64 and 67 are the counts
    and the shortfall those copies show."""
    purchases = phase_changes = rusts = 0
    for s in samples[::STRIDE]:
        state = s.state
        limit, next_type = TRAINS.limit(state), TRAINS.cheapest_type(state)
        rust = next_rust_phase(s.game_phase)
        rusted = copy_of(state)
        while rust is not None and PHASE.current(rusted) < rust:
            PHASE.advance(rusted)
        later = copy_of(state)
        if s.game_phase + 1 < D.NUM_GAME_PHASES:
            PHASE.advance(later)
        for c in range(D.NUM_CORPS):
            row, held = s.of('CORP')[c], TRAINS.corp_total(state, c)
            now = max_route_revenue(copy_of(state), c)
            assert row[34] == f32(now, 100) and row[76] == log_yen(now)
            assert row[65] == f32(max_route_revenue(rusted, c), 100)
            rusts += row[65] != row[34]
            if held < limit:
                after = bought(state, c, next_type)
                assert row[66] == f32(max_route_revenue(after, c), 100)
                purchases += 1
                phase_changes += PHASE.current(after) != s.game_phase
            else:
                assert row[66] == row[34]
            assert row[61] == f32(held, limit) and row[62] == f32(limit - held, 4)
            assert row[63] == f32(TRAINS.corp_total(rusted, c), 4)
            assert row[64] == f32(TRAINS.limit(later) - TRAINS.corp_total(later, c), 4)
            assert row[67] == f32(max(D.TRAIN_PRICE[next_type] - CORPS[c].cash(state), 0), 1000)
            assert row[75] == log_yen(CORPS[c].cash(state)) and row[21] == f32(CORPS[c].cash(state), 1000)
        starts = s.of('TRAIN')[:, 14]
        assert starts.tolist() == [float(s.game_phase + 1 < D.NUM_GAME_PHASES and D.PHASE_ON_TRAIN[s.game_phase + 1] == t)
                                   for t in range(D.NUM_TRAIN_TYPES)]
        if starts.any() and TRAINS.depot_types(state) >> int(starts.argmax()) & 1:  # the depot sells it: buy one
            assert PHASE.current(bought(state, 0, int(starts.argmax()))) == s.game_phase + 1
    assert purchases > 1000 and phase_changes > 50 and rusts > 200


def _purchases(state):
    """SELECT_TRAIN's choices at a SELECT_TRAIN decision opened from
    ``state``'s BUY_TRAIN, by source: (source action, opened state)."""
    for action in legal_ids(state):
        if K.BUY_TRAIN_DEPOT <= action < K.BUY_TRAIN_SELL:
            opened = applied(state, action)
            if A.get_decision_phase_py(opened) == K.DPHASE_SELECT_TRAIN:
                yield action, opened


def _priced(state):
    """Past TRAIN_PRICE at its first price, if the state is there."""
    if A.get_decision_phase_py(state) == K.DPHASE_TRAIN_PRICE:
        return applied(state, legal_ids(state)[0])
    return state


def test_train_deltas_are_the_route_max_after_the_real_purchase(samples):
    """TRAIN 10, 11, and 13 at SELECT_TRAIN, opened from every BUY_TRAIN
    decision, and at TRAIN_PRICE: the buyer's and the seller's route max
    after the real action less before, the phase the purchase starts
    included. And TRAIN 12 at DISCARD."""
    seen = {'add': 0, 'trade': 0, 'corp': 0, 'price': 0, 'phase': 0, 'trade phase': 0, 'discard': 0}
    for s in at(samples, K.DPHASE_BUY_TRAIN):
        corp = s.corp
        for _, opened in _purchases(s.state):
            tokens = tokens_of(opened)[rows('TRAIN', s.max_players)]
            before = max_route_revenue(copy_of(opened), corp)
            source = TURN.pending_train_source(opened)
            seller = source - K.TRAIN_SOURCE_CORP if source >= K.TRAIN_SOURCE_CORP else None
            expected = np.zeros((D.NUM_TRAIN_TYPES, 4), np.float32)
            for action in legal_ids(opened):
                chosen = applied(opened, action)
                if seller is not None:  # TRAIN_PRICE shows the pending type's two deltas again
                    assert A.get_decision_phase_py(chosen) == K.DPHASE_TRAIN_PRICE
                    priced = tokens_of(chosen)[rows('TRAIN', s.max_players)]
                    assert priced[:, [10, 13]].any(axis=1).sum() <= 1 and not priced[:, [11, 12]].any()
                    np.testing.assert_array_equal(priced[action, [10, 13]], tokens[action, [10, 13]])
                    seen['price'] += 1
                after = _priced(chosen)
                delta = f32(max_route_revenue(after, corp) - before, 100)
                changed = PHASE.current(after) != s.game_phase
                if action < K.SELECT_TRAIN_TRADE_IN:
                    expected[action, 0] = delta
                    seen['add'] += 1
                    seen['phase'] += changed
                    if seller is not None:
                        expected[action, 3] = f32(max_route_revenue(after, seller)
                                                  - max_route_revenue(copy_of(opened), seller), 100)
                        seen['corp'] += 1
                else:
                    expected[D.TRADE_IN_TYPE[action - K.SELECT_TRAIN_TRADE_IN], 1] = delta
                    seen['trade'] += 1
                    seen['trade phase'] += changed
            np.testing.assert_array_equal(tokens[:, 10:14], expected)
    for s in at(samples, K.DPHASE_DISCARD):
        corp = TRAINS.first_crowded(s.state)
        expected = np.zeros((D.NUM_TRAIN_TYPES, 4), np.float32)
        for action in legal_ids(s.state):
            expected[D.DISCARD_TYPES[action], 2] = f32(max_route_revenue(applied(s.state, action), corp)
                                                       - max_route_revenue(copy_of(s.state), corp), 100)
            seen['discard'] += 1
        np.testing.assert_array_equal(s.of('TRAIN')[:, 10:14], expected)
    assert all(count > 5 for count in seen.values()), seen
    assert seen['phase'] >= 40  # the purchases layout 1's slot 10 got wrong


def test_train_10_at_buy_train_and_dividend_is_each_offered_types_gain(samples):
    """At BUY_TRAIN and DIVIDEND, while the decision corp has room: TRAIN 10
    for each type the depot sells or the pool holds is its route max after
    buying one (the entity purchase on a copy) less now, whatever the
    treasury; nothing at the other types, nor without room. Where the real
    purchase is legal at BUY_TRAIN, it gives the same."""
    seen = {K.DPHASE_BUY_TRAIN: 0, K.DPHASE_DIVIDEND: 0, 'real': 0, 'phase': 0, 'second type': 0}
    for s in at(samples, K.DPHASE_BUY_TRAIN, K.DPHASE_DIVIDEND):
        state, corp = s.state, s.corp
        deltas = s.of('TRAIN')[:, 10]
        assert not s.of('TRAIN')[:, [11, 12, 13]].any()
        now = max_route_revenue(copy_of(state), corp)
        offered = [t for t in range(D.NUM_TRAIN_TYPES)
                   if has_room(state, corp) and (TRAINS.depot_types(state) >> t & 1 or TRAINS.pool(state, t))]
        expected = np.zeros(D.NUM_TRAIN_TYPES, np.float32)
        for t in offered:
            after = bought(state, corp, t)
            expected[t] = f32(max_route_revenue(after, corp) - now, 100)
            seen['phase'] += PHASE.current(after) != s.game_phase
        np.testing.assert_array_equal(deltas, expected)
        seen[s.phase] += bool(offered)
        seen['second type'] += len(offered) > 1
        if s.phase == K.DPHASE_BUY_TRAIN:
            for source, opened in _purchases(state):
                if source >= K.BUY_TRAIN_CORP:
                    continue
                for action in legal_ids(opened):
                    if action < K.SELECT_TRAIN_TRADE_IN:
                        assert deltas[action] == f32(max_route_revenue(applied(opened, action), corp) - now, 100)
                        seen['real'] += 1
    assert all(count > 20 for count in seen.values()), seen


def test_the_dividend_row_shows_what_each_choice_leaves_toward_the_next_train(samples):
    """DIVIDEND 7-12: the next train's price, the shortfall from the treasury
    the real withhold and the real payout leave, the room, and the corp's
    CORP 66 and 65; no price or shortfall without room."""
    seen = {'room': 0, 'short': 0, 'none': 0}
    for s in at(samples, K.DPHASE_DIVIDEND):
        state, corp, div = s.state, s.corp, s.of('DIVIDEND')[0]
        room = TRAINS.limit(state) - TRAINS.corp_total(state, corp)
        price = D.TRAIN_PRICE[TRAINS.cheapest_type(state)]
        assert div[10] == f32(room, 4)
        assert div[11] == s.of('CORP')[corp, 66] and div[12] == s.of('CORP')[corp, 65]
        if room <= 0:
            assert not div[7:10].any()
            seen['none'] += 1
            continue
        kept = applied(state, K.DIVIDEND_WITHHOLD)
        assert div[7] == f32(price, 1000)
        assert div[8] == f32(max(price - CORPS[corp].cash(kept), 0), 1000)
        if K.DIVIDEND_PAYOUT in legal_ids(state):
            paid = applied(state, K.DIVIDEND_PAYOUT)
            assert div[9] == f32(max(price - CORPS[corp].cash(paid), 0), 1000)
        else:  # no revenue: both leave the treasury as it is
            assert div[9] == div[8]
        seen['room'] += 1
        seen['short'] += div[8] != div[9]
    assert all(count > 10 for count in seen.values()), seen


def test_a_candidates_delta_with_the_next_train(samples):
    """CANDIDATE 108 at TILE and ER_BONUS: after the real lay, the decision
    corp's route max with the next train bought (the entity purchase on a
    copy of the laid state) less the same on the state before; CANDIDATE 104
    without room."""
    seen = {'room': 0, 'full': 0, 'differs': 0, 'phase': 0}
    for s in at(samples, K.DPHASE_TILE, K.DPHASE_ER_BONUS):
        state, corp = s.state, s.corp
        er = s.phase == K.DPHASE_ER_BONUS
        room, next_type = has_room(state, corp), TRAINS.cheapest_type(state)
        before = max_route_revenue(bought(state, corp, next_type), corp) if room else None
        for k, action in enumerate(legal_ids(state)[er:]):
            row = s.of('CANDIDATE')[k]
            if not room:
                assert row[108] == row[104]
                seen['full'] += 1
                continue
            with_next = bought(applied(state, action), corp, next_type)
            assert row[108] == f32(max_route_revenue(with_next, corp) - before, 100)
            seen['room'] += 1
            seen['differs'] += row[108] != row[104]
            seen['phase'] += PHASE.current(with_next) != s.game_phase
        assert not s.of('CANDIDATE')[len(legal_ids(state)) - er:].any()
    assert all(count > 20 for count in seen.values()), seen


# =============================================================================
# PER-CHOICE TRUTH: CAPITAL
# =============================================================================

def test_par_affordability_is_the_shares_the_actor_can_buy(samples):
    """PRICE 24-29 at PAR: after the real par at each legal price, the shares
    out of the IPO and those the cash left would buy, up to the float's five."""
    seen = 0
    for s in at(samples, K.DPHASE_PAR):
        actor, price = TURN.active_player(s.state), s.of('PRICE')[0]
        expected = np.zeros(D.NUM_PAR_PRICES, np.float32)
        for action in legal_ids(s.state):
            parred = applied(s.state, action)
            corp = TURN.pending_corp(s.state)
            more = PLAYERS[actor].cash(parred) // D.PAR_PRICE[action]
            expected[action] = f32(min(10 - CORPS[corp].ipo_shares(parred) + more, FLOAT_SHARES), FLOAT_SHARES)
            seen += 1
        np.testing.assert_array_equal(price[24:30], expected)
    for s in at(samples, K.DPHASE_BID, K.DPHASE_TRAIN_PRICE, K.DPHASE_PRIVATE_PRICE):
        assert not s.of('PRICE')[0, 24:30].any()
    assert seen > 50


def test_float_slots_are_what_floating_takes(samples):
    """CORP 68-71: buying the shares CORP 68 counts from the IPO on a copy
    floats the corp at the last one, for CORP 69; 70 and 71 say whether the
    president's and the actor's cash covers it. An unparred corp needs five
    at the lowest par; a floated corp nothing. PLAYER 35 is the same for a
    corp yet to par."""
    seen = {'parred': 0, 'unparred': 0, 'floated': 0}
    for s in samples[::STRIDE]:
        state = s.state
        actor = TURN.active_player(state) if s.phase >= 0 else -1
        for c in range(D.NUM_CORPS):
            row = s.of('CORP')[c, 68:72]
            if CORPS[c].floated(state):
                assert not row.any()
                seen['floated'] += 1
            elif CORPS[c].par_index(state) < 0:  # five shares, less one DR's exchange took
                count = CORPS[c].ipo_shares(state) - 5
                cost = count * min(D.PAR_PRICE)
                assert count in (4, 5) and row.tolist() == [f32(count, 5), f32(cost, 1000), 0,
                                                            float(actor >= 0 and PLAYERS[actor].cash(state) >= cost)]
                seen['unparred'] += 1
            else:
                buyer, president = copy_of(state), CORPS[c].president(state)
                count = round(float(row[0]) * FLOAT_SHARES)
                assert count == round(float(s.of('CORP')[c, 22]) * 10) - 5 > 0
                BANK.transfer(buyer, BANK_ACCOUNT, K.OWNER_PLAYER + president, 2000)  # test setup: the cash to buy them
                paid = PLAYERS[president].cash(buyer)
                for i in range(count):
                    assert not CORPS[c].floated(buyer)
                    CORPS[c].buy_share(buyer, president, False)
                assert CORPS[c].floated(buyer)
                cost = paid - PLAYERS[president].cash(buyer)
                assert row[1] == f32(cost, 1000)
                assert row[2] == (PLAYERS[president].cash(state) >= cost)
                assert row[3] == (actor >= 0 and PLAYERS[actor].cash(state) >= cost)
                seen['parred'] += 1
        for i in range(TURN.num_players(state)):
            assert s.of('PLAYER')[i, 35] == (PLAYERS[i].cash(state) >= LOWEST_FLOAT)
    assert all(count > 100 for count in seen.values()), seen


def test_the_president_sold_flag_bars_the_buy_back(samples):
    """CORP 72: the president sold the corp this SR. At their STOCK turn the
    engine then offers them no share of it."""
    seen = barred = 0
    for s in samples[::STRIDE]:
        for c in range(D.NUM_CORPS):
            president = CORPS[c].president(s.state)
            sold = president >= 0 and bool(PLAYERS[president].sold_mask(s.state) >> c & 1)
            assert s.of('CORP')[c, 72] == sold
            seen += sold
            if sold and s.phase == K.DPHASE_STOCK and TURN.active_player(s.state) == president:
                assert not {K.STOCK_IPO + c, K.STOCK_POOL + c} & set(legal_ids(s.state))
                barred += 1
    assert seen > 20 and barred > 3


def test_emergency_slots_are_the_legal_sales_and_the_cash_to_raise(samples):
    """In an emergency (the decision corp must buy a train its treasury
    cannot): CORP 55 × 5 is the count of blocks EMERGENCY_SELL_AMOUNT offers
    for each corp BUY_TRAIN lets the president sell, and 0 for the others;
    CORP 73 on the decision corp is the cheapest train's price less the
    treasury and the president's cash. Both are 0 outside one."""
    seen = {'sale': 0, 'none': 0, 'need': 0, 'amount': 0}
    for s in samples:
        corps = s.of('CORP')
        if not corps[:, 37].any():
            assert not corps[:, 55].any() and not corps[:, 73].any()
            continue
        corp = int(corps[:, 37].argmax())
        president = CORPS[corp].president(s.state)
        need = TRAINS.cheapest_price(s.state) - CORPS[corp].cash(s.state) - PLAYERS[president].cash(s.state)
        assert corps[corp, 73] == f32(max(need, 0), 1000) and corps[:, 73].sum() == corps[corp, 73]
        seen['need'] += need > 0
        if s.phase == K.DPHASE_BUY_TRAIN:
            legal = legal_ids(s.state)
            for c in range(D.NUM_CORPS):
                blocks = len(legal_ids(applied(s.state, K.BUY_TRAIN_SELL + c))) if K.BUY_TRAIN_SELL + c in legal else 0
                assert round(float(corps[c, 55]) * 5) == blocks
                seen['sale' if blocks else 'none'] += 1
        elif s.phase == K.DPHASE_EMERGENCY_SELL_AMOUNT:
            assert round(float(corps[TURN.pending_corp(s.state), 55]) * 5) == len(legal_ids(s.state))
            seen['amount'] += 1
    assert all(seen.values()), seen


# =============================================================================
# THE REST OF §5
# =============================================================================

def test_global_additions(samples):
    """GLOBAL 39-40 are the auction's; 44-45 name the step PRIVATE_PRICE,
    ER_BONUS, and an OR's PORT return to (the real sub-phase, played out,
    resumes there); 46 the ORs left in the set; 47 the ORs the bank lasts;
    48-49 the train limits; 50-89 the supply."""
    seen = {'pre': 0, 'post': 0, 'sr port': 0, 'bank': 0}
    for s in samples[::STRIDE] + at(samples, *SUB_PHASES):
        state, row = s.state, s.of('GLOBAL')[0]
        if TURN.round(state) != K.ROUND_AUCTION:
            assert not row[39:41].any()
        else:
            assert row[39:41].tolist() == [f32(TURN.auction_pass_payouts(state), 3), f32(TURN.auction_discount(state), 20)]
        if s.phase in SUB_PHASES:
            after, history = copy_of(state), []
            while A.get_decision_phase_py(after) in SUB_PHASES:
                history.clear()
                DRIVER.apply_action(after, legal_ids(after)[0], history)
            resumed = [K.STEP_TO_DECISION_PHASE[TURN.step(state_of(r))] for r, _, _ in history[1:]] + [A.get_decision_phase_py(after)]
            resumed = next(p for p in resumed if p not in SUB_PHASES)
            assert row[44] == (resumed == K.DPHASE_BUY_PRIVATE_PRE) and row[45] == (resumed == K.DPHASE_BUY_PRIVATE_POST)
            seen['pre'] += row[44] == 1
            seen['post'] += row[45] == 1
            seen['sr port'] += not row[44:46].any()
        else:
            assert not row[44:46].any()
        in_or = TURN.round(state) == K.ROUND_OPERATING
        assert row[46] == (f32(TURN.or_count(state) - TURN.or_num(state), 2) if in_or else 0)
        drain = sum(max_route_revenue(copy_of(state), c) * (10 - CORPS[c].ipo_shares(state)) / 10
                    for c in range(D.NUM_CORPS) if CORPS[c].floated(state))
        cash = BANK.cash(state)
        lasts = 0 if cash <= 0 else 1 if not drain else min(cash / drain / 10, 1)
        assert abs(float(row[47]) - lasts) < 1e-6
        seen['bank'] += 0 < row[47] < 1
        assert row[48] == f32(TRAINS.limit(state), 4)
        assert row[49] == f32(D.PHASE_TRAIN_LIMIT[min(s.game_phase + 1, D.NUM_GAME_PHASES - 1)], 4)
        assert row[50:90].tolist() == [f32(MAP.supply(state, t), 5) for t in range(D.NUM_SUPPLY_TILES)]
    assert all(seen.values()), seen


def test_the_ors_left_count_down_to_the_next_sr(samples):
    """GLOBAL 46 × 2 at an OR decision is the count of later ORs before the
    next SR in the game played."""
    checked = 0
    for index, s in enumerate(samples):
        if TURN.round(s.state) != K.ROUND_OPERATING or index % 11:
            continue
        here = (TURN.turn(s.state), TURN.or_num(s.state))
        later = set()
        for other in samples[index + 1:]:
            if TURN.round(other.state) != K.ROUND_OPERATING:
                break
            later.add((TURN.turn(other.state), TURN.or_num(other.state)))
        else:
            continue  # the game's last set
        if other.kind != s.kind or TURN.round(other.state) != K.ROUND_STOCK:
            continue
        assert round(float(s.of('GLOBAL')[0, 46]) * 2) == len(later - {here})
        checked += 1
    assert checked > 100


def test_player_additions(samples):
    """PLAYER 36 marks, in a closed auction's BID, the player who acts when
    it resolves (played out on a copy: the auction's next turn, or the first
    SR's). PLAYER 37 is the income of an OR at the route maxes, which the
    real payout pays for its corp; 38-39 the log cash and worth."""
    seen = {'resumes': 0, 'opens sr': 0, 'payout': 0}
    for s in samples[::STRIDE] + at(samples, K.DPHASE_BID, K.DPHASE_DIVIDEND):
        state, players = s.state, s.of('PLAYER')[:TURN.num_players(s.state)]
        closed = s.phase == K.DPHASE_BID and TURN.auction_private(state) >= 0
        assert players[:, 36].sum() == closed
        if closed:
            private, after = TURN.auction_private(state), copy_of(state)
            while PRIVATES[private].owner(after) == K.OWNER_UNSOLD:  # drop until one bidder is left
                after = applied(after, K.BID_DROP if K.BID_DROP in legal_ids(after) else legal_ids(after)[0])
            if A.get_decision_phase_py(after) in (K.DPHASE_AUCTION, K.DPHASE_STOCK):  # no other closed auction
                assert players[TURN.active_player(after), 36] == 1
                seen['opens sr' if TURN.round(after) == K.ROUND_STOCK else 'resumes'] += 1
        scores = A.final_scores_py(state) if s.phase < 0 else None
        for i in range(len(players)):
            income = sum(-(-PLAYERS[i].shares(state, c) * max_route_revenue(copy_of(state), c) // 10)
                         for c in range(D.NUM_CORPS) if CORPS[c].floated(state))
            income += sum(PRIVATES[p].revenue(state) for p in range(D.NUM_PRIVATES)
                          if PRIVATES[p].owner(state) == K.OWNER_PLAYER + i)
            assert players[i, 37] == f32(income, 1000)
            assert players[i, 38] == log_yen(PLAYERS[i].cash(state))
            assert players[i, 39] == log_yen(round(float(players[i, 9]) * 1000))
            if scores is not None and not PLAYERS[i].bankrupt(state):
                assert players[i, 9] == f32(scores[i], 1000)
        if s.phase == K.DPHASE_DIVIDEND and K.DIVIDEND_PAYOUT in legal_ids(state) \
                and TURN.route_revenue(state) == max_route_revenue(copy_of(state), s.corp):
            paid = applied(state, K.DIVIDEND_PAYOUT)
            for i in range(len(players)):  # the payout is the corp's share of PLAYER 37
                others = copy_of(state)
                PLAYERS[i].set_shares(others, s.corp, 0)  # test setup: the income without this corp's shares
                rest = tokens_of(others)[R.ROW_PLAYER + i, 37]
                assert abs(float(players[i, 37] - rest) * 1000 - (PLAYERS[i].cash(paid) - PLAYERS[i].cash(state))) < 1e-3
            seen['payout'] += 1
    assert all(seen.values()), seen


def test_the_port_flag_is_mfs(samples):
    for s in samples[::STRIDE]:
        used = s.of('PRIVATE')[:, 15]
        assert used.sum() == used[D.PRIVATE_MF] == bool(PRIVATES[D.PRIVATE_MF].flags(s.state) & 1)


# =============================================================================
# RANGES
# =============================================================================

# Slots outside [0, 1]; every other slot is a flag, a one-hot, or a share of
# its natural maximum. MONEY is ¥/1000 (the largest seen on 88,019 states of
# human, self-play, and random games: a treasury of ¥4,046, cash of ¥6,394),
# WORTH a net worth /1000 (¥10,326), PRICE ¥/100 (share prices to ¥350, bids
# to ¥390, revenues and route maxes to ¥1,150), DELTA a signed route-max
# change /100, and ROOM a signed train count /4.
MONEY, WORTH, PRICE, DELTA, LOSS, ROOM = (0, 8), (0, 15), (0, 12), (-12, 12), (-12, 0), (-1, 1)
WIDE = {
    'GLOBAL': {34: (-4, 1), 39: (0, 4)},
    'PRICE': dict.fromkeys(range(5, 24), MONEY),
    'DIVIDEND': {10: ROOM, **dict.fromkeys((4, 7, 8, 9), MONEY), **dict.fromkeys((1, 2, 3, 5, 6, 11, 12), PRICE)},
    'CORP': {61: (0, 2.5), 62: ROOM, 64: ROOM, **dict.fromkeys((21, 67, 69, 73), MONEY),
             **dict.fromkeys((7, 8, 25, 33, 34, 48, 58, 59, 60, 65, 66, *range(15, 20)), PRICE)},
    'PRIVATE': dict.fromkeys((7, 8, 9, 10, 12, 13), PRICE),
    'TRAIN': {2: MONEY, 10: DELTA, 11: DELTA, 12: LOSS, 13: LOSS},
    'HEX': dict.fromkeys((2 + 71, 78, 84, 104), PRICE),
    'CANDIDATE': {108: DELTA, **dict.fromkeys((2 + 71, 99, 103, 104), PRICE)},
    'PLAYER': {8: MONEY, 9: WORTH, 10: (0, 1.5), 37: (0, 4), **dict.fromkeys(range(25, 31), PRICE)},
}
# Slots no decision fills. A candidate row's tile block (slots 2-77) never
# holds the port tile (10), a printed tile (42), a white, gray, or red tile
# (64, 68, 69), or an offboard (72), and the labelled K and T tiles (28, 29,
# 38-40, 76, 77) upgrade their city with one candidate, which the driver lays
# by itself.
NEVER = {'CANDIDATE': {10, 28, 29, 38, 39, 40, 42, 64, 68, 69, 72, 76, 77}}


def test_every_slot_is_within_its_range_and_exercised(samples):
    """Each slot's minimum and maximum over the games against its bound, and
    every slot of every row type takes two values or more, but those no
    decision fills."""
    for name, width in WIDTHS.items():
        values = np.concatenate([s.of(name)[s.of(name)[:, 0] == 1] for s in samples])
        low, high = values.min(axis=0), values.max(axis=0)
        for slot in range(1, width):
            lo, hi = WIDE.get(name, {}).get(slot, (0, 1))
            assert lo <= low[slot] and high[slot] <= hi, (name, slot, low[slot], high[slot])
            assert (high[slot] > low[slot]) == (slot not in NEVER.get(name, ())), (name, slot, low[slot], high[slot])
        assert not values[:, width:].any()
