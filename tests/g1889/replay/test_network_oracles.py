"""GRAPH, lay, and token legality against Ruby's oracles (6.5's acceptance).

At every OR snapshot of the four fixtures, the state comes from ``load_state``
and each corp with a token is checked against the extract's
``operating.graph``, ``tokens``, and ``lays`` (``extracts.py``): Ruby's own
``Graph``, ``City#tokenable?``, and ``Tracker`` code, with
``check_track_restrictions!`` run per candidate. The oracles ignore whose turn
it is, so every corp is checked, not only the operating one; at Track and
Token snapshots the operating corp's TRACK, TILE, and TOKEN ids are checked
too, and at BuyTrain snapshots ``must_buy_train`` (``operating.must_buy``).
They all agree exactly: the engine replays Ruby's walk step for step
(``g1889/entities/graph.pyx``). Random maps, denser in junctions and loops
than the fixtures, check GRAPH against Ruby's ``Graph`` as well.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.bank import BANK, BANK_ACCOUNT
from g1889.entities.corp import CORPS
from g1889.entities.graph import GRAPH
from g1889.entities.hexmap import MAP
from g1889.entities.phase import PHASE
from g1889.entities.turn import TURN
from g1889.state import GameState

from tests.g1889.replay.extracts import FIXTURES, load_fixture_extract, load_graph_extract
from tests.g1889.replay.snapshot import CORP, HEX, engine_lays, load_state, ruby_lays

# (OR snapshots, corp graphs, Track snapshots, corp lay sets, legal lays, rejected by the
# semi-restrictive rule, Token snapshots, corp token sets) per fixture.
COVERAGE = {
    '314': (197, 947, 43, 217, 4707, 1091, 16, 947),
    '962': (243, 1417, 57, 347, 9350, 3670, 15, 1417),
    'hs_jmmljkbw_17502': (2, 2, 1, 1, 6, 0, 0, 2),
    'hs_lhglbbiz_17502': (2, 2, 1, 1, 6, 0, 0, 2),
}


def _graph_differences(state, c: int, ruby: dict) -> list[str]:
    sides = {D.HEX_IDS[h]: [s for s in range(D.NUM_SIDES) if GRAPH.sides_reached(state, c, h) >> s & 1]
             for h in range(D.NUM_HEXES) if GRAPH.sides_reached(state, c, h)}
    nodes = sorted(D.HEX_IDS[h] for h in range(D.NUM_HEXES) if GRAPH.node_reached(state, c, h))
    return [f'{name}: engine {ours} != ruby {theirs}' for name, ours, theirs in (
        ('hexes', sides, ruby['hexes']), ('nodes', nodes, ruby['nodes']),
        ('route', GRAPH.route_available(state, c), ruby['route'])) if ours != theirs]


def _ids(state) -> list[int]:
    ids = np.zeros(K.MAX_ACTION_SIZE, dtype=np.uint16)
    return [int(i) for i in ids[:A.enumerate_legal_actions_py(state, ids)]]


def _or_snapshots(fixture):
    extract = load_fixture_extract(fixture)
    for record in extract['records']:
        snap = record['state']
        if snap['game']['round'] == 'OR' and not snap['game']['finished']:
            yield record['index'], snap, load_state(snap, len(extract['players']))


@pytest.mark.parametrize('fixture', FIXTURES)
def test_network_matches_ruby_at_every_or_snapshot(fixture):
    snapshots = graphs = tracks = lay_sets = legal = rejected = tokens = token_sets = 0
    for index, snap, state in _or_snapshots(fixture):
        snapshots += 1
        operating = snap['operating']
        for sym, ruby in operating['graph'].items():
            assert _graph_differences(state, CORP[sym], ruby) == [], (fixture, index, sym)
            graphs += 1
        for sym, ruby in operating['tokens'].items():
            assert [D.HEX_IDS[h] for h in A.token_hexes_py(state, CORP[sym])] == \
                sorted(ruby, key=HEX.__getitem__), (fixture, index, sym)
            token_sets += 1
        if operating['lays'] is not None:
            tracks += 1
            for sym, lays in operating['lays'].items():
                ours = engine_lays(state, CORP[sym])
                assert ours == ruby_lays(lays, snap['corporations'][sym]['cash']), (fixture, index, sym)
                lay_sets += 1
                legal += len(ours)
                rejected += sum(not ok for *_, ok in lays)
        tokens += snap['game']['step'] == 'Token'
    assert (snapshots, graphs, tracks, lay_sets, legal, rejected, tokens, token_sets) == COVERAGE[fixture]


@pytest.mark.parametrize('fixture', FIXTURES)
def test_operating_corp_track_tile_and_token_ids_match_ruby(fixture):
    """TRACK is pass and each hex with a Ruby lay; TILE at each indexes its
    candidates; TOKEN is pass and each city Ruby's Token step accepts."""
    checked = 0
    for index, snap, state in _or_snapshots(fixture):
        operating, step = snap['operating'], snap['game']['step']
        corp = TURN.operating_corp(state)
        sym = D.CORP_SYMS[corp]
        if step == 'Track' and operating['lays'] is not None:
            lays = ruby_lays(operating['lays'][sym], snap['corporations'][sym]['cash'])
            hexes = sorted({HEX[hex_id] for hex_id, _, _ in lays})
            assert _ids(state) == [K.TRACK_PASS] + [K.TRACK_HEX + h for h in hexes], (fixture, index)
            TURN.set_step(state, K.STEP_TILE)
            for h in hexes:
                TURN.set_pending_hex(state, h)
                candidates = A.lay_candidates_py(state, h)
                assert candidates == sorted(candidates)
                assert {(D.HEX_IDS[h], D.TILE_NAMES[t], r) for t, r in candidates} == \
                    {lay for lay in lays if lay[0] == D.HEX_IDS[h]}, (fixture, index, D.HEX_IDS[h])
                assert _ids(state) == list(range(len(candidates)))
            checked += 1
        elif step == 'Token':
            hexes = sorted(HEX[hex_id] for hex_id in operating['tokens'][sym])
            assert hexes, (fixture, index)  # Ruby skips the Token step when nothing can be tokened
            assert _ids(state) == [K.TOKEN_PASS] + [K.TOKEN_HEX + h for h in hexes], (fixture, index)
            checked += 1
    assert checked == {'314': 59, '962': 72}.get(fixture, 1)


# (BuyTrain snapshots before the game ends, must-buys among them) per fixture.
MUST_BUY_COVERAGE = {'314': (60, 15), '962': (52, 6), 'hs_jmmljkbw_17502': (1, 0), 'hs_lhglbbiz_17502': (1, 0)}


@pytest.mark.parametrize('fixture', FIXTURES)
def test_must_buy_matches_ruby_at_every_buy_train_snapshot(fixture):
    """``must_buy_train`` against ``must_buy_train?`` on a fresh graph. Ruby's
    own graph keeps ``route_train_purchase`` through ``Graph#clear``
    (engine-design §4), but here it had the operator's route cached at every
    snapshot and never read a stale value."""
    snapshots = must_buys = 0
    for index, snap, state in _or_snapshots(fixture):
        if (ruby := snap['operating']['must_buy']) is None:
            continue
        assert A.must_buy_train_py(state) == ruby['fresh'] == ruby['game'], (fixture, index)
        snapshots += 1
        must_buys += ruby['fresh']
    assert (snapshots, must_buys) == MUST_BUY_COVERAGE[fixture]


def random_map(rng: random.Random, num_lays: tuple[int, int] = (30, 120)) -> tuple[GameState, dict]:
    """A new game grown into a random network: a few tokens, then a number of
    lays in the range ``num_lays`` that the engine finds legal for random
    corps (any color, ample cash), with more tokens on the way. Also its ops
    for ``load_graph_extract``, and a probe of every upgrade over track that
    each corp's semi-restrictive rule decides."""
    state = GameState(4)
    state.initialize_game(4, priority_player=0)
    PHASE.set_current(state, D.GAME_PHASE_D)
    ops: list[list] = []

    def token(c: int, cities: list[int]) -> None:
        if cities and CORPS[c].tokens_left(state):
            h = rng.choice(cities)
            MAP.place_token(state, h, c)
            ops.append(['token', D.CORP_SYMS[c], D.HEX_IDS[h]])

    corps = rng.sample(range(D.NUM_CORPS), rng.randrange(2, D.NUM_CORPS + 1))
    for c in corps:
        BANK.transfer(state, BANK_ACCOUNT, K.OWNER_CORP + c, 900)
        token(c, [h for h in range(D.NUM_HEXES) if MAP.has_slot_for(state, h, c)])
    for _ in range(rng.randrange(*num_lays)):
        c = rng.choice(corps)
        lays = [(h, tile, rotation) for h in range(D.NUM_HEXES) for tile, rotation in A.lay_candidates_py(state, h, c)]
        if lays:
            h, tile, rotation = rng.choice(lays)
            MAP.lay(state, h, tile, rotation)
            ops.append(['lay', D.HEX_IDS[h], D.TILE_NAMES[tile], rotation])
        if rng.random() < 0.15:
            token(c, A.token_hexes_py(state, c))
    probe = {D.CORP_SYMS[c]: [[D.HEX_IDS[h], D.TILE_NAMES[t], r] for h, t, r in track_upgrades(state, c)]
             for c in corps if CORPS[c].tokens_left(state) < D.CORP_TOKENS[c]}
    return state, {'ops': ops, 'probe': probe}


def track_upgrades(state, c: int) -> list[tuple[int, int, int]]:
    """Every upgrade over track that the corp's other checks admit (exits open
    and meeting a reached side, the tile in supply): the lays the
    semi-restrictive rule decides, by a city's new value or by walking."""
    lays = []
    for h in range(D.NUM_HEXES):
        old, sides = MAP.tile(state, h), GRAPH.sides_reached(state, c, h)
        k = D.NUM_SIDES * old + MAP.rotation(state, h)
        if not sides or not D.TILE_MASK[old][MAP.rotation(state, h)]:
            continue
        for i in range(D.UPG_START[k], D.UPG_START[k + 1]):
            tile, rotation = D.UPG_TILE[i], D.UPG_ROT[i]
            exits = D.TILE_EXITS[tile][rotation]
            if MAP.supply(state, tile) and exits & sides and exits & ~D.HEX_EXIT_MASK[h] == 0:
                lays.append((h, tile, rotation))
    return lays


RANDOM_MAPS = 150


def test_graph_and_semi_restrictive_rule_match_ruby_on_random_maps():
    rng = random.Random(1889)
    maps = [random_map(rng) for _ in range(RANDOM_MAPS)]
    extract = load_graph_extract(f'random-{RANDOM_MAPS}', [spec for _, spec in maps])
    reached = probed = rejected = 0
    for i, ((state, spec), ruby) in enumerate(zip(maps, extract['graphs'])):
        assert set(ruby['graph']) == {sym for kind, sym, *_ in spec['ops'] if kind == 'token'}
        for sym, graph in ruby['graph'].items():
            assert _graph_differences(state, CORP[sym], graph) == [], (i, sym)
            reached += len(graph['hexes'])
        for sym, verdicts in ruby['restrictions'].items():
            c = CORP[sym]
            for (hex_id, name, rotation), ok in zip(spec['probe'][sym], verdicts):
                h, tile = HEX[hex_id], D.TILE_NAMES.index(name)
                assert ((tile, rotation) in A.lay_candidates_py(state, h, c)) == ok, (i, sym, hex_id, name, rotation)
                probed += 1
                rejected += not ok
    print(f'{reached} hexes reached, {probed} lays probed, {rejected} rejected')
    assert reached > 10_000 and probed > 1_000 and rejected > 100
