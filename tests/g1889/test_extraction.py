"""1889 token extraction (1889/extraction-design.md §10 items 1, 2, and 6).

On random-playout states at 3-5 players, capacities 3 and 5, with the policy
flags on and off:

- every value is finite and within its divisor family's bound (§5.2);
- slot 0 follows §5.1, and padded rows and the slots past each row's width
  are zero;
- each ``is_selected`` group has at most one 1: the actor's PLAYER row at
  every decision, a CORP row exactly when there is a decision corp, and the
  pending hex at TILE;
- a row extracted twice, and its copy with every cache marked stale, give
  the same tokens;
- extraction changes only the caches, which then equal a recomputation, and
  CORP 34 reads the route solver's revenue; the hypotheticals behind the
  candidate rows leave the row as it was;
- candidate row k at TILE and ER_BONUS describes the state after applying
  its id to a copy: the hex's tile block, the cost paid, the supply left, and
  GRAPH's and the solver's deltas.

And the RSS guard: RSS's engine and its v2 and v3 models import nothing of
1889's, nor 1889's engine anything of RSS's, so no 1889 change can reach RSS
(``tests/test_v2_parity.py`` pins its outputs).
"""

from pathlib import Path
import re

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_INVALID
from g1889.entities.corp import CORPS
from g1889.entities.graph import GRAPH
from g1889.entities.hexmap import MAP
from g1889.entities.phase import PHASE
from g1889.entities.private import PRIVATES
from g1889.entities.turn import TURN
from g1889.routes.solver import max_route_revenue
from g1889.state import GameState, get_corp_fields, get_global_fields, get_layout, get_storage_player_capacity
from g1889.token_data import (
    TokenDataSize,
    TokenRow as R,
    get_num_tokens,
    get_token_data,
    get_token_data_batch,
    get_token_dim,
    get_token_widths,
)

from tests.g1889.phases.conftest import legal_ids, new_game, policy_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN_DIM = int(TokenDataSize.TOKEN_DIM)
GLOBALS, CORP_FIELDS = get_global_fields(), get_corp_fields()
LAYOUT = get_layout(0)  # every offset but the row width is the same at every capacity
ALL_CORPS = (1 << D.NUM_CORPS) - 1

SETUPS = ((3, 3), (3, 5), (4, 5), (5, 5))  # (num_players, max_players)
SEEDS = range(3)
MAX_DECISIONS = 3000
# All-pass payouts after which the auction's players keep to the capped
# policy set: uniform passes would loop the auction, paying out without end.
UNCAPPED_PAYOUTS = 3

# Row types: first row and rows (None: one per player), §5.1.
ROW_TYPES = {
    'GLOBAL': (R.ROW_GLOBAL, 1),
    'PRICE': (R.ROW_PRICE, 1),
    'DIVIDEND': (R.ROW_DIVIDEND, 1),
    'CORP': (R.ROW_CORP, D.NUM_CORPS),
    'PRIVATE': (R.ROW_PRIVATE, D.NUM_PRIVATES),
    'TRAIN': (R.ROW_TRAIN, D.NUM_TRAIN_TYPES),
    'HEX': (R.ROW_HEX, D.NUM_HEXES),
    'CANDIDATE': (R.ROW_CANDIDATE, D.MAX_LAY_CANDIDATES),
    'PLAYER': (R.ROW_PLAYER, None),
}
HIDDEN_UNTIL_7_3 = ('PRICE', 'DIVIDEND')

# Slots outside [0, 1], by §5.2's divisor families; every other slot is a
# flag, a one-hot, or a count by its natural maximum. SOFT counts are by a
# typical maximum: all-pass payouts, cities reached, privates, certificates.
MONEY, PRICE, SOFT, BANK = (0, 5), (0, 10), (0, 3), (-4, 1)
WIDE = {
    'GLOBAL': {33: BANK, 38: SOFT},
    'CORP': {21: MONEY, 43: SOFT, 50: SOFT, **dict.fromkeys((7, 8, 25, 33, 34, 42), PRICE)},
    'PRIVATE': dict.fromkeys((7, 8, 9, 10, 12, 13), PRICE),
    'TRAIN': {2: MONEY},
    'HEX': dict.fromkeys((2 + 71, 78, 84), PRICE),  # the tile block's revenue, D revenue, lay cost
    # The tile block's revenue, the lay cost, Δ reached revenue and route
    # max (a lay only adds track and slots), and the others' Δ centers.
    'CANDIDATE': {107: SOFT, **dict.fromkeys((2 + 71, 99, 103, 104), PRICE)},
    'PLAYER': {8: MONEY, 9: MONEY, 10: SOFT, 36: SOFT, **dict.fromkeys(range(26, 33), PRICE)},
}
# Selection slots: at most one row of the type sets each.
SELECTIONS = {'CORP': (1, 2, 3), 'PRIVATE': (1, 2), 'TRAIN': (1,), 'HEX': (1,), 'PLAYER': (1,)}


def _rows(name, max_players):
    first, count = ROW_TYPES[name]
    return slice(first, first + (max_players if count is None else count))


def _bounds(max_players):
    low = np.zeros((get_num_tokens(max_players), TOKEN_DIM), np.float32)
    high = np.ones_like(low)
    for name, slots in WIDE.items():
        for slot, (lo, hi) in slots.items():
            low[_rows(name, max_players), slot], high[_rows(name, max_players), slot] = lo, hi
    return low, high


def _tokens(state, max_players):
    tokens = np.full((get_num_tokens(max_players), TOKEN_DIM), np.nan, np.float32)
    get_token_data(state, tokens, max_players, layout_version=1)
    return tokens


def _choose(state, rng, flags):
    capped = TURN.round(state) == K.ROUND_AUCTION and state.auction_pass_payout_cap
    ids = policy_ids(state) if flags or capped else legal_ids(state)
    phase = A.get_decision_phase_py(state)
    if phase == K.DPHASE_BUY_TRAIN and K.BUY_TRAIN_DEPOT in ids and rng.random() < 0.8:
        return K.BUY_TRAIN_DEPOT
    if ids[0] == 0 and phase == K.DPHASE_STOCK and rng.random() < 0.5:
        return 0
    return ids[int(rng.integers(len(ids)))]


def _playouts():
    """(state, max_players) before each decision and at each game's end."""
    for flags in (False, True):
        for num_players, max_players in SETUPS:
            for seed in SEEDS:
                rng = np.random.default_rng(100 * num_players + seed)
                state = new_game(num_players, max_players=max_players, seed=seed, priority_player=-1,
                                 auction_pass_payout_cap=int(flags), same_president=flags)
                for _ in range(MAX_DECISIONS):
                    if TURN.auction_pass_payouts(state) >= UNCAPPED_PAYOUTS:
                        state.auction_pass_payout_cap = 1
                    yield state, max_players
                    if A.is_terminal_py(state):
                        break
                    assert DRIVER.apply_action(state, _choose(state, rng, flags)) != STATUS_INVALID
                else:
                    pytest.fail(f'{num_players} players, seed {seed}: no game end')


@pytest.fixture(scope='module')
def samples():
    """(row before extraction, state after it, max_players, tokens) at every
    decision of the playouts, extracted on the live state."""
    out = []
    for state, max_players in _playouts():
        row = state._array.copy()
        tokens = _tokens(state, max_players)
        out.append((row, _state(state._array.copy()), max_players, tokens))
    return out


def _candidate_count(state):
    """TILE's and ER_BONUS's lay candidates, by their legal ids: TILE's ids
    are its candidates, and ER_BONUS's follow its pass."""
    phase = A.get_decision_phase_py(state)
    if phase in (K.DPHASE_TILE, K.DPHASE_ER_BONUS):
        return len(legal_ids(state)) - (phase == K.DPHASE_ER_BONUS)
    return 0


def _cache_slots(row):
    """The slots extraction writes back: the graph block, each corp's route
    cache and ``route_available``, and the two dirty masks."""
    mask = np.zeros(row.shape, bool)
    mask[LAYOUT.graph_offset:LAYOUT.graph_offset + D.NUM_CORPS * D.NUM_HEXES] = True
    for c in range(D.NUM_CORPS):
        for field in (CORP_FIELDS.route_cache, CORP_FIELDS.route_available):
            mask[LAYOUT.corps_offset + c * CORP_FIELDS.size + field] = True
    for field in (GLOBALS.conn_dirty, GLOBALS.route_dirty):
        mask[LAYOUT.global_offset + field] = True
    return mask


def _stale(row):
    """A copy with every cache marked stale (test setup: raw dirty masks)."""
    stale = row.copy()
    for field in (GLOBALS.conn_dirty, GLOBALS.route_dirty):
        stale[LAYOUT.global_offset + field] = ALL_CORPS
    return stale


def _state(row):
    """A state over ``row``, without copying."""
    return GameState.from_buffer(row, int(row[LAYOUT.global_offset + GLOBALS.num_players]),
                                 max_players=get_storage_player_capacity(row.shape[0]))


# =============================================================================
# SIZES AND LAYOUT IDS
# =============================================================================

def test_sizes_follow_the_row_map():
    """§5.1: 90 + max_players rows of width 108, each type padded from its own width."""
    assert get_token_dim(1) == TOKEN_DIM == 108
    for max_players in (3, 4, 5):
        assert get_num_tokens(max_players) == 90 + max_players
        expected = np.repeat([43, 24, 8, 51, 18, 13, 105, 108, 38], [1, 1, 1, 7, 7, 6, 52, 15, max_players])
        np.testing.assert_array_equal(get_token_widths(max_players, 1), expected)
        assert get_token_widths(max_players, 1).dtype == np.uint8


@pytest.mark.parametrize('call', [
    lambda: get_token_dim(2),
    lambda: get_token_widths(5, 3),
    lambda: get_token_data(new_game(3), np.zeros((93, 108), np.float32), layout_version=0),
    lambda: get_token_data_batch([], np.zeros((0, 95, 108), np.float32), layout_version=2),
])
def test_other_layout_ids_raise(call):
    with pytest.raises(ValueError, match='Unsupported 1889 token layout version'):
        call()


# =============================================================================
# INVARIANTS
# =============================================================================

def test_values_are_finite_and_within_their_divisor_family(samples):
    bounds = {max_players: _bounds(max_players) for max_players in (3, 5)}
    for row, _, max_players, tokens in samples:
        low, high = bounds[max_players]
        assert np.isfinite(tokens).all()
        outside = np.argwhere((tokens < low) | (tokens > high))
        assert not len(outside), f'(row, slot) {outside[:5].tolist()} out of bounds at step {TURN.step(_state(row))}'


def test_visibility_padding_and_widths(samples):
    """Slot 0 per §5.1 (PRICE and DIVIDEND rows stay hidden until sub-step
    7.3), hidden and padded rows all zero, nothing past a row's width."""
    for row, after, max_players, tokens in samples:
        num_players = TURN.num_players(after)
        widths = get_token_widths(max_players, 1)
        assert not any(tokens[i, widths[i]:].any() for i in range(len(tokens)))
        visible = np.ones(len(tokens), bool)
        for name in HIDDEN_UNTIL_7_3:
            visible[_rows(name, max_players)] = False
        visible[R.ROW_CANDIDATE + _candidate_count(after):R.ROW_PLAYER] = False
        visible[_rows('PRIVATE', max_players)] = [p.owner(after) != K.OWNER_NOT_IN_PLAY for p in PRIVATES]
        visible[R.ROW_PLAYER + num_players:] = False
        np.testing.assert_array_equal(tokens[:, 0], visible)
        assert not tokens[~visible].any()


def test_selections_mark_at_most_one_row(samples):
    """Each selection slot is set on at most one row of its type; the actor's
    PLAYER row at every decision, a CORP row exactly at PAR and the OR's
    decisions, and at TILE the pending hex."""
    seen = set()
    for _, after, max_players, tokens in samples:
        for name, slots in SELECTIONS.items():
            for slot in slots:
                assert tokens[_rows(name, max_players), slot].sum() <= 1, (name, slot)
        phase = A.get_decision_phase_py(after)
        players = np.flatnonzero(tokens[_rows('PLAYER', max_players), 1])
        assert players.tolist() == ([TURN.active_player(after)] if phase >= 0 else [])
        has_corp = phase == K.DPHASE_PAR or phase >= 0 and TURN.round(after) == K.ROUND_OPERATING
        assert tokens[_rows('CORP', max_players), 1].sum() == has_corp, phase
        if phase == K.DPHASE_TILE:
            assert tokens[R.ROW_HEX + TURN.pending_hex(after), 1] == 1
        seen.add(phase)
    assert seen == set(range(K.NUM_DECISION_PHASES)) | {-1}, sorted(seen)  # every phase, and the game's end


def test_extraction_is_deterministic_whatever_the_caches(samples):
    """A row extracted twice, or with every cache stale, gives the same tokens."""
    for row, _, max_players, tokens in samples:
        copy = row.copy()
        np.testing.assert_array_equal(_tokens(_state(copy), max_players), tokens)
        np.testing.assert_array_equal(_tokens(_state(copy), max_players), tokens)
        np.testing.assert_array_equal(_tokens(_state(_stale(row)), max_players), tokens)


def test_extraction_writes_back_only_fresh_caches(samples):
    """Extraction changes only the caches, which are then all clean and equal
    a recomputation from scratch (as test_random_playouts' check), whatever
    the candidate rows tried on the scratch state; CORP 34 is the solver's
    revenue."""
    for row, after, _, tokens in samples:
        caches = _cache_slots(row)
        written = after._array
        np.testing.assert_array_equal(written[~caches], row[~caches])
        assert written[LAYOUT.global_offset + GLOBALS.conn_dirty] == written[LAYOUT.global_offset + GLOBALS.route_dirty] == 0
        fresh = _state(_stale(row))
        for c in range(D.NUM_CORPS):
            GRAPH.refresh(fresh, c)
            assert tokens[R.ROW_CORP + c, 34] == np.float32(max_route_revenue(fresh, c)) / np.float32(100)
        np.testing.assert_array_equal(written[caches], fresh._array[caches])


def test_batches_match_single_states_at_lay_decisions(samples):
    """A batch mixing lay decisions with others, and player counts, fills each
    row as a single state does, one scratch state serving its candidates."""
    for capacity in (3, 5):
        picked = [(row, tokens) for i, (row, after, max_players, tokens) in enumerate(samples)
                  if max_players == capacity and (_candidate_count(after) or i % 50 == 0)]
        out = np.full((len(picked), get_num_tokens(capacity), TOKEN_DIM), np.nan, np.float32)
        get_token_data_batch([row.copy() for row, _ in picked], out, capacity, layout_version=1)
        np.testing.assert_array_equal(out, np.stack([tokens for _, tokens in picked]))


# =============================================================================
# CANDIDATE OUTCOMES (§10 item 2)
# =============================================================================

def _f32(numerator, divisor):
    """A slot's value as the fill computes it, in float32."""
    return np.float32(numerator) / np.float32(divisor)


def _after_lay(row, action):
    """A copy of ``row`` with ``action`` applied, before the driver's
    automated steps and forced decisions."""
    state = _state(row.copy())
    history = []
    assert DRIVER.apply_action(state, action, history) != STATUS_INVALID
    return _state(history[1][0]) if len(history) > 1 else state


def _reach(state, corp):
    """GRAPH's reach for the corp: hexes with a reached side, reached centers,
    and their revenue now."""
    centers = [h for h in range(D.NUM_HEXES) if GRAPH.node_reached(state, corp, h)]
    hexes = sum(GRAPH.sides_reached(state, corp, h) != 0 for h in range(D.NUM_HEXES))
    return hexes, len(centers), sum(MAP.revenue(state, h, PHASE.current(state)) for h in centers)


def test_candidate_rows_describe_their_lays(samples):
    """Applying TILE k, or ER_BONUS 1 + k, to a copy lays what candidate row k
    describes: the hex's tile block afterwards is the row's and its new paths
    the row's added ones, the corp pays the lay cost and supply keeps the
    copies left, and the deltas are GRAPH's and the solver's after the lay
    less before; at ER_BONUS, the buying corp's."""
    seen = {K.DPHASE_TILE: 0, K.DPHASE_ER_BONUS: 0}
    for row, before, max_players, tokens in samples:
        phase = A.get_decision_phase_py(before)
        if phase not in seen:
            continue
        er = phase == K.DPHASE_ER_BONUS
        h, corp = D.ER_HEX if er else TURN.pending_hex(before), TURN.operating_corp(before)
        reach = [_reach(before, c) for c in range(D.NUM_CORPS)]
        old_paths = D.TILE_MASK[MAP.tile(before, h)][MAP.rotation(before, h)]
        assert (tokens[R.ROW_CANDIDATE:R.ROW_CANDIDATE + _candidate_count(before), 1] == er).all()
        for k, action in enumerate(legal_ids(before)[er:]):
            candidate, laid = tokens[R.ROW_CANDIDATE + k], _after_lay(row, action)
            tile = MAP.tile(laid, h)
            np.testing.assert_array_equal(candidate[2:78], _tokens(laid, max_players)[R.ROW_HEX + h, 2:78])
            added = D.TILE_MASK[tile][MAP.rotation(laid, h)] & ~old_paths
            assert candidate[78:99].tolist() == [added >> i & 1 for i in range(D.NUM_PAIRS)]
            assert candidate[99] == _f32(CORPS[corp].cash(before) - CORPS[corp].cash(laid), 100)
            assert candidate[100] == _f32(MAP.supply(laid, tile), D.TILE_COUNT[tile])
            deltas = [np.subtract(_reach(laid, c), reach[c]) for c in range(D.NUM_CORPS)]
            assert candidate[101:104].tolist() == [_f32(d, divisor) for d, divisor in zip(deltas[corp], (52, 25, 100))]
            assert candidate[104] == _f32(max_route_revenue(laid, corp) - max_route_revenue(before, corp), 100)
            assert candidate[105] == GRAPH.route_available(laid, corp)
            grown = [d[1] for c, d in enumerate(deltas) if c != corp and d[1] > 0]
            assert candidate[106:108].tolist() == [_f32(len(grown), 6), _f32(sum(grown), 25)]
            seen[phase] += 1
    assert all(seen.values()), seen


# =============================================================================
# RSS GUARD (§10 item 6)
# =============================================================================

IMPORTED = re.compile(r'^\s*(?:from|import|cimport)\s+([\w.]+)', re.MULTILINE)
RSS_ENGINE = ('core', 'entities', 'phases')


def _imports(paths):
    return {(path.relative_to(REPO_ROOT).as_posix(), name)
            for path in paths for name in IMPORTED.findall(path.read_text())}


def _sources(package):
    return [p for p in (REPO_ROOT / package).rglob('*') if p.suffix in ('.py', '.pyx', '.pxd')]


def test_rss_and_1889_engines_import_nothing_of_each_other():
    rss = _imports([p for package in RSS_ENGINE for p in _sources(package)]
                   + [REPO_ROOT / 'nn' / name for name in ('transformer-v2.py', 'transformer-v3.py')])
    assert not {(path, name) for path, name in rss if name.split('.')[0] == 'g1889' or name == 'games.g1889'}
    g1889 = _imports(_sources('g1889'))
    assert not {(path, name) for path, name in g1889 if name.split('.')[0] in RSS_ENGINE}
