"""The 1889 caches never go stale unmarked (1889/v1-design.md §5.9's purity test).

The row caches each corp's connectivity (the graph block and
``route_available``), its route max and the hexes of its best routes, and
its two hypothetical route maxes, behind ``conn_dirty``, ``route_dirty``, and
``hyp_dirty``. A lay or a token marks only the corps whose networks touch the
hex (``map_touch_mask``), and a change to the trains on sale only when the
cheapest changes type, so a missed invalidation would leave a stale value
marked clean, and the inputs would depend on the path a state was reached by.

Along whole games, at every decision the driver rests at and at the game's
end, with the caches as the engine left them:

- the tokens of each layout and the relation records equal those of a copy
  with every cache marked stale;
- after a full refresh, the row equals that copy's, refreshed: every cached
  value is what a recomputation from the rest of the row gives.

The refresh leaves every cache clean before the next move, as a search's
parent leaves them for its children, so each move's invalidation is tested
alone; a second state plays the same game refreshing only now and then,
which tests what stale marks do to later invalidations.

The games are the layout-1 fixture's (self-play, human with the replay
harness's overrides, and random; ``tests/g1889/test_layout1_parity.py``) and
random playouts under both flag settings.
"""

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.graph import GRAPH
from g1889.entities.phase import PHASE
from g1889.relations import RelationSize, get_relation_coord_data
from g1889.routes.solver import max_route_revenue, route_hypotheticals_py
from g1889.state import GameState, get_global_fields, get_layout
from g1889.token_data import get_num_tokens, get_token_data, get_token_dim

from tests.g1889.test_layout1_parity import apply_op, load_fixture, new_state

LAYOUT_VERSIONS = (1, 2)
GLOBALS = get_global_fields()
GLOBAL_OFFSET = get_layout(0).global_offset
DIRTY = [GLOBAL_OFFSET + getattr(GLOBALS, name) for name in GLOBALS._fields if name.endswith('_dirty')]
ALL_CORPS = (1 << D.NUM_CORPS) - 1
LAZY_STRIDE = 7  # the second state's refreshes, in decisions
# (num_players, max_players, seed, (private_sale, train_sale, payout cap), share of BUY_TRAINs that take the depot)
PLAYOUTS = tuple((n, capacity, 40 + 10 * n + i, flags, depot)
                 for n, capacity in ((3, 3), (4, 5), (5, 5))
                 for i, (flags, depot) in enumerate((((True, True, 1), 0.7), ((False, False, 1), 0.7),
                                                     ((True, True, 1), 0.0), ((False, False, 1), 0.95))))


def _copy(state, stale=False):
    copy = GameState.from_array(state._array, state._array[GLOBAL_OFFSET + GLOBALS.num_players],
                                max_players=state.max_players)
    if stale:
        copy._array[DIRTY] = ALL_CORPS  # test setup: drop every cache
    return copy


def _inputs(state):
    """Each layout's tokens and the relation records, extracted on a copy."""
    out = []
    for version in LAYOUT_VERSIONS:
        tokens = np.full((get_num_tokens(state.max_players), get_token_dim(version)), np.nan, np.float32)
        get_token_data(_copy(state), tokens, state.max_players, layout_version=version)
        out.append(tokens)
    records = np.zeros((int(RelationSize.MAX_RELATION_EDGES), 4), np.uint8)
    get_relation_coord_data(_copy(state), records, state.max_players)
    return [*out, records]


def _refresh(state):
    """Recompute whatever is marked stale, as extraction's prologue does."""
    for c in range(D.NUM_CORPS):
        GRAPH.refresh(state, c)
        max_route_revenue(state, c)
        route_hypotheticals_py(state, c)
    assert not state._array[DIRTY].any()


def check(state, where):
    """The inputs and the refreshed caches of ``state`` against those of a
    copy with every cache stale. Leaves ``state``'s caches fresh."""
    stale = _copy(state, stale=True)
    for mine, theirs in zip(_inputs(state), _inputs(stale)):
        np.testing.assert_array_equal(mine, theirs, err_msg=where)
    _refresh(state)
    _refresh(stale)
    differ = np.flatnonzero(state._array != stale._array)
    assert not len(differ), f'{where}: the caches differ from a recomputation at slots {differ[:10].tolist()}'


def _random_calls(state, rng, depot_share):
    """Uniform policy choices from ``state`` to the game's end, leaning to
    depot purchases so that phases change; the state is left at the start."""
    start = state._array.copy()
    ids = np.empty(K.MAX_ACTION_SIZE, np.uint16)
    calls = []
    while not A.is_terminal_py(state):
        count = A.enumerate_policy_actions_py(state, ids)
        choices = ids[:count].tolist()
        if (A.get_decision_phase_py(state) == K.DPHASE_BUY_TRAIN and K.BUY_TRAIN_DEPOT in choices
                and rng.random() < depot_share):
            action = K.BUY_TRAIN_DEPOT
        else:
            action = choices[int(rng.integers(count))]
        calls.append((0, action))
        apply_op(state, 0, action)
    state._array[:] = start
    return calls


def games():
    """(label, spec, calls) of every game: the fixture's, then the playouts."""
    fixture = load_fixture()
    for g, spec in enumerate(fixture['games']):
        yield f"{spec['kind']} {spec['label']}", spec, fixture[f'ops_{g}'].tolist()
    for num_players, max_players, seed, flags, depot_share in PLAYOUTS:
        spec = {'kind': 'playout', 'label': f'{num_players}p-{seed}', 'num_players': num_players,
                'max_players': max_players, 'seed': seed, 'priority_player': -1, 'flags': list(flags)}
        yield f'playout {spec["label"]}', spec, _random_calls(new_state(spec), np.random.default_rng(seed), depot_share)


def play(spec, calls, label, seen=None):
    """Check every rest of the game on a state refreshed at each, and every
    ``LAZY_STRIDE``-th on one refreshed only then. Returns the states checked."""
    state, lazy = new_state(spec), new_state(spec)
    for done in range(len(calls) + 1):
        where = f'{label} after {done} calls'
        check(state, where)
        if done % LAZY_STRIDE == 0 or done == len(calls):
            check(lazy, f'{where} (lazy)')
        if seen is not None:
            seen.add((spec['kind'], A.get_decision_phase_py(state), PHASE.current(state)))
        if done < len(calls):
            for each in (state, lazy):
                apply_op(each, *calls[done])
    return len(calls) + 1


def test_inputs_and_caches_do_not_depend_on_how_a_state_was_reached():
    seen, states = set(), 0
    for label, spec, calls in games():
        states += play(spec, calls, label, seen)
    kinds = {kind for kind, _, _ in seen}
    assert kinds == {'selfplay', 'human', 'random', 'playout'}
    for kind in ('human', 'playout'):  # v0's self-play stops at phase 5
        assert {phase for k, _, phase in seen if k == kind} == set(range(D.NUM_GAME_PHASES)), kind
    assert {phase for _, phase, _ in seen} == {-1, *range(K.NUM_DECISION_PHASES)}
    assert states > 10_000


@pytest.mark.parametrize('field', ['conn_dirty', 'route_dirty', 'hyp_dirty'])
def test_a_stale_cache_marked_clean_is_caught(field):
    """The check fails when a cache is stale and its mark is cleared: on the
    fixture's first human game, at the first decision where the mask is set
    after a move, clear it before the check."""
    fixture = load_fixture()
    g = next(g for g, spec in enumerate(fixture['games']) if spec['kind'] == 'human')
    spec, calls = fixture['games'][g], fixture[f'ops_{g}'].tolist()
    state = new_state(spec)
    slot = GLOBAL_OFFSET + getattr(GLOBALS, field)
    for done, call in enumerate(calls):
        check(state, f'{done}')
        apply_op(state, *call)
        stale = _copy(state, stale=True)
        _refresh(stale)
        marked = _copy(state)
        marked._array[slot] = 0  # test setup: a missed invalidation
        _refresh(marked)
        if state._array[slot] and (marked._array != stale._array).any():
            with pytest.raises(AssertionError):
                broken = _copy(state)
                broken._array[slot] = 0
                check(broken, 'missed')
            return
    pytest.fail(f'no move made {field} matter')
