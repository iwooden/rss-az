"""The route solver against Ruby's ``AutoRouter`` and a brute-force search
(6.6's acceptance; 1889/engine-design.md §8, §11).

At every Route snapshot of the fixtures, where Ruby waits for the
president's routes, the state comes from ``load_state``. The solver's
maximum equals ``operating.auto_router``, AutoRouter's best on a fresh
graph, which searched every route there (no timeout, no list cut at
``route_limit``); the ROUTE step runs for it; and its routes are legal by
brute force. Where several combinations tie, the solver may pick another
than AutoRouter's. The maximum is at least every human's logged revenue but
314 action 367's, which Ruby paid unchecked (``test_extract_states.py``).

Brute force (``tests/g1889/route_brute.py``) agrees with the solver for
every corp with trains and a token at every OR snapshot, and on random
networks grown in phase D colors with random trains, D trains among them.
AutoRouter agrees on sparser random networks in phase D as well
(``extract_states.rb --graphs``), where a D train's diesel value and
identical trains come into play, which the fixtures never reach.
"""

from __future__ import annotations

import random

import pytest

import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER
from g1889.entities.graph import GRAPH
from g1889.entities.hexmap import MAP
from g1889.entities.phase import PHASE
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.routes.solver import best_routes_py, max_route_revenue

from tests.g1889.replay.extracts import FIXTURES, load_fixture_extract, load_graph_extract
from tests.g1889.replay.snapshot import CORP, HEX, load_state
from tests.g1889.replay.test_network_oracles import random_map
from tests.g1889.route_brute import best_revenue, legal_routes, normal, revenue, trains_of

# Route snapshots, and those where the solver picks AutoRouter's routes (the rest tie).
ROUTE_COVERAGE = {'314': (32, 25), '962': (48, 42), 'hs_jmmljkbw_17502': (0, 0), 'hs_lhglbbiz_17502': (0, 0)}
# Logged revenues below the maximum, and the one above it (314 action 367: IR's
# route passes through I2, which SR and AR fill).
HUMAN_COVERAGE = {'314': (4, {367}), '962': (1, set()), 'hs_jmmljkbw_17502': (0, set()),
                  'hs_lhglbbiz_17502': (0, set())}
# Corps with trains and a token, over every OR snapshot.
BRUTE_COVERAGE = {'314': 840, '962': 1331, 'hs_jmmljkbw_17502': 0, 'hs_lhglbbiz_17502': 0}


def check_routes(state, corp: int, routes: list[dict]) -> None:
    """``best_routes_py``'s routes are legal routes of the corp's trains, by
    brute force, that share no hex side and earn what they claim."""
    legal = legal_routes(state, corp)
    trains = trains_of(state, corp)
    used: frozenset = frozenset()
    for r in routes:
        route = legal[normal(r['hexes'])]
        assert r['stops'] == [h for h in r['hexes'] if D.HEX_RC[h] >= 0] and set(r['stops']) == set(route.stops)
        assert len(route.stops) <= D.TRAIN_DISTANCE[r['train']] and r['revenue'] == revenue(state, route.stops, r['train'])
        assert route.edges.isdisjoint(used)
        used |= route.edges
        trains.remove(r['train'])


def _route_snapshots(fixture):
    """(Route snapshot, the run_routes record after it, state, operating corp)."""
    extract = load_fixture_extract(fixture)
    records = extract['records']
    for before, record in zip(records, records[1:]):
        snap = before['state']
        if snap['game']['step'] == 'Route':
            yield snap, record, load_state(snap, len(extract['players'])), CORP[snap['operating']['operator']]


def _ruby_routes(router: dict) -> list:
    return sorted((r['train'].split('-')[0], r['revenue'], normal([HEX[h] for h in r['ordered_hexes']]))
                  for r in router['routes'])


@pytest.mark.parametrize('fixture', FIXTURES)
def test_the_solver_matches_auto_router_at_every_route_snapshot(fixture):
    snapshots = same = 0
    for snap, _, state, corp in _route_snapshots(fixture):
        router = snap['operating']['auto_router']
        routes = best_routes_py(state, corp)
        assert max_route_revenue(state, corp) == router['revenue'] == sum(r['revenue'] for r in routes), snap['game']
        check_routes(state, corp, routes)
        same += sorted((D.TRAIN_NAMES[r['train']], r['revenue'], normal(r['hexes'])) for r in routes) == \
            _ruby_routes(router)
        DRIVER.advance(state)  # ROUTE, then DIVIDEND, the president's choice
        assert TURN.step(state) == K.STEP_DIVIDEND and TURN.route_revenue(state) == router['revenue']
        snapshots += 1
    assert (snapshots, same) == ROUTE_COVERAGE[fixture]


@pytest.mark.parametrize('fixture', FIXTURES)
def test_the_maximum_covers_every_human_revenue_but_one(fixture):
    below, above = 0, set()
    for _, record, state, corp in _route_snapshots(fixture):
        logged, best = record['pre']['revenue'], max_route_revenue(state, corp)
        below += logged < best
        if logged > best:
            above.add(record['action']['id'])
    assert (below, above) == HUMAN_COVERAGE[fixture]


@pytest.mark.parametrize('fixture', FIXTURES)
def test_brute_force_agrees_at_every_or_snapshot(fixture):
    extract = load_fixture_extract(fixture)
    checked = 0
    for record in extract['records']:
        snap = record['state']
        if snap['game']['round'] != 'OR' or snap['game']['finished']:
            continue
        state = load_state(snap, len(extract['players']))
        for c in range(D.NUM_CORPS):
            if TRAINS.corp_total(state, c) and any(MAP.tokens(state, h) >> c & 1 for h in range(D.NUM_HEXES)):
                assert max_route_revenue(state, c) == best_revenue(state, c), (fixture, record['index'], c)
                checked += 1
    assert checked == BRUTE_COVERAGE[fixture]


RANDOM_NETWORKS = 40


def test_brute_force_agrees_on_random_networks():
    """Dense networks (``test_network_oracles.random_map``: phase D colors,
    junctions and loops, several corps' tokens), each corp with a token given
    one to three random trains, a D train first half the time."""
    rng = random.Random(66)
    cases = multi = d_trains = blocked = offboards = 0
    for _ in range(RANDOM_NETWORKS):
        state, _ = random_map(rng)
        assert PHASE.current(state) == D.GAME_PHASE_D
        for c in range(D.NUM_CORPS):
            if not any(MAP.tokens(state, h) >> c & 1 for h in range(D.NUM_HEXES)):
                continue
            trains = [rng.randrange(D.TRAIN_D + 1) for _ in range(rng.randint(1, 3))]
            if rng.random() < 0.5:
                trains[0] = D.TRAIN_D
            for t in range(D.NUM_TRAIN_TYPES):
                TRAINS.set_corp_trains(state, c, t, trains.count(t))
            routes = best_routes_py(state, c)
            assert max_route_revenue(state, c) == best_revenue(state, c) == sum(r['revenue'] for r in routes)
            check_routes(state, c, routes)
            cases += 1
            multi += len(routes) >= 2
            d_trains += D.TRAIN_D in trains
            blocked += any(GRAPH.node_reached(state, c, h) and MAP.blocked_for(state, h, c) for h in range(D.NUM_HEXES))
            offboards += any(D.TILE_RC[MAP.tile(state, h)] == D.RC_OFFBOARD for r in routes for h in r['stops'])
    print(f'{cases} corps, {multi} with two or more routes, {d_trains} with a D train, {blocked} reaching a '
          f'blocked city, {offboards} with an offboard on a route')
    assert cases > 150 and min(multi, d_trains, blocked, offboards) > 100


AUTO_ROUTER_NETWORKS = 60
AUTO_ROUTER_ROUTE_LIMIT = 10_000   # AutoRouter#path keeps each train's best route_limit routes
PHASE_D_TRAINS = (D.TRAIN_5, D.TRAIN_6, D.TRAIN_D)


def test_the_solver_matches_auto_router_on_random_networks():
    """Random networks in phase D, with 20 to 60 lays so Ruby's path walk
    stays short (a cold extract takes about 8 s): each corp with a token runs
    one or two trains of phase D's ranks (5, 6, D), and AutoRouter searches
    every route there."""
    rng = random.Random(188966)
    states, specs = [], []
    for _ in range(AUTO_ROUTER_NETWORKS):
        state, spec = random_map(rng, (20, 60))
        spec['phase'], spec['trains'] = 'D', {}
        del spec['probe']
        for c in range(D.NUM_CORPS):
            if any(MAP.tokens(state, h) >> c & 1 for h in range(D.NUM_HEXES)):
                trains = [rng.choice(PHASE_D_TRAINS) for _ in range(rng.randint(1, 2))]
                for t in range(D.NUM_TRAIN_TYPES):
                    TRAINS.set_corp_trains(state, c, t, trains.count(t))
                spec['trains'][D.CORP_SYMS[c]] = [D.TRAIN_NAMES[t] for t in trains]
        states.append(state)
        specs.append(spec)
    extract = load_graph_extract(f'routes-random-{AUTO_ROUTER_NETWORKS}', specs)
    cases = same = two = 0
    for state, ruby in zip(states, extract['graphs']):
        for sym, router in ruby['routes'].items():
            c = CORP[sym]
            assert not router['timed_out'] and max(router['candidates'].values(), default=0) < AUTO_ROUTER_ROUTE_LIMIT
            routes = best_routes_py(state, c)
            assert max_route_revenue(state, c) == router['revenue'], (sym, router)
            check_routes(state, c, routes)
            cases += 1
            two += len(routes) == 2
            same += sorted((D.TRAIN_NAMES[r['train']], r['revenue'], normal(r['hexes'])) for r in routes) == \
                _ruby_routes(router)
    print(f'{cases} corps, {two} running two trains, {same} with the routes AutoRouter picks')
    assert cases > 200 and two > 100 and same > cases // 2
