"""The route rules the solver applies (1889/engine-design.md §8; rules §8.6;
Ruby's ``route.rb`` and ``game/base.rb:1434-1625``), on small networks built
through the entity APIs, and the route cache.

``replay/test_route_oracles.py`` checks the solver against Ruby's
``AutoRouter`` on the fixtures and against brute force on fixture states and
random networks; ``phases/test_operating_round.py`` covers the ROUTE step and
``DRIVER.override_route_revenue``.

Sides: 0 S, 1 SW, 2 NW, 3 N, 4 NE, 5 SE. KO's home, Takamatsu (K4, ¥30), has
printed track to K6 (side 0), J5 (side 1), and J3 (side 2). Sakaide &
Okayama (J1) is an offboard paying ¥20, ¥40 from phase 5, and ¥80 to D trains.
"""

import itertools

import pytest

import g1889.data.tables as D
from g1889.entities.hexmap import MAP
from g1889.entities.phase import PHASE
from g1889.entities.trains import TRAINS
from g1889.routes.solver import (
    MAX_ROUTES, best_routes_py, max_route_revenue, route_hexes_py, route_hypotheticals_py,
)
from g1889.state import GameState, get_corp_fields, get_global_fields, get_layout

from tests.g1889.route_brute import normal

AR, IR, SR, KO, TR, KU, UR = range(D.NUM_CORPS)
HEX = {hex_id: h for h, hex_id in enumerate(D.HEX_IDS)}
TILE = {name: t for t, name in enumerate(D.TILE_NAMES)}
ALL_CORPS = (1 << D.NUM_CORPS) - 1

TO_SAKAIDE = (('J3', '8', 3),)                       # K4 - J3 - J1
VIA_MARUGAME = (('J3', '9', 2), ('I2', '5', 4))      # K4 - J3 - I2 (¥20, SR's home) - J1
TO_KOTOHIRA = (('J5', '58', 2), ('I4', '438', 3))    # K4 - J5 (town ¥10) - I4 (¥40)
CHAIN = (*TO_KOTOHIRA, ('I2', '6', 4))               # K4 - J5 - I4 - I2 - J1


def network(*lays, trains=(), phase=D.GAME_PHASE_2, tokens=((KO, 'K4'),)):
    """A new 4-player game in ``phase`` with the lays, the tokens, and KO's trains."""
    state = GameState(4)
    state.initialize_game(4, priority_player=0)
    PHASE.set_current(state, phase)
    for hex_id, tile, rotation in lays:
        MAP.lay(state, HEX[hex_id], TILE[tile], rotation)
    for corp, hex_id in tokens:
        MAP.place_token(state, HEX[hex_id], corp)
    set_trains(state, KO, trains)
    return state


def set_trains(state, corp, trains):
    for t in range(D.NUM_TRAIN_TYPES):
        TRAINS.set_corp_trains(state, corp, t, list(trains).count(t))


def routes(state, corp=KO):
    """(train, revenue, stops) per route, stops in their lower orientation."""
    return [(D.TRAIN_NAMES[r['train']], r['revenue'], [D.HEX_IDS[h] for h in normal(r['stops'])])
            for r in best_routes_py(state, corp)]


def stops(*hex_ids):
    return [D.HEX_IDS[h] for h in normal([HEX[hex_id] for hex_id in hex_ids])]


def test_every_path_of_a_revenue_center_tile_ends_at_it():
    """The solver's segments rest on this: track passes only through hexes
    without a revenue center. A plain tile also holds at most two paths with
    no side in common, so a route crosses a hex at most twice
    (``MAX_ROUTE_HEXES``), and the solver's train arrays fit the largest
    train limit."""
    for t in range(D.NUM_TILES):
        paths = [(D.PAIR_A[p], D.PAIR_B[p]) for p in D.TILE_PATHS[t][0][:D.TILE_PATH_COUNT[t]]]
        assert all((b == D.NODE) == (D.TILE_RC[t] != D.RC_NONE) for _, b in paths), D.TILE_NAMES[t]
        if D.TILE_RC[t] == D.RC_NONE:
            assert not any(len({*p, *q, *r}) == 6 for p, q, r in itertools.combinations(paths, 3)), D.TILE_NAMES[t]
    assert all((D.HEX_RC[h] >= 0) == (D.TILE_RC[D.NUM_SUPPLY_TILES + h] != D.RC_NONE) for h in range(D.NUM_HEXES))
    assert D.MAX_ROUTE_HEXES == 2 * D.NUM_HEXES and MAX_ROUTES == max(D.PHASE_TRAIN_LIMIT)


def test_no_train_or_no_token_earns_nothing():
    state = network(*TO_SAKAIDE)
    assert max_route_revenue(state, KO) == 0 and best_routes_py(state, KO) == []
    set_trains(state, IR, [D.TRAIN_D])
    assert max_route_revenue(state, IR) == 0


def test_a_route_needs_a_city_with_the_corps_token():
    """Imabari (F1, ¥60 in phase 5) to Matsuyama (E2, ¥20) pays more than
    KO's line but holds no KO token; IR, with a token there, runs it."""
    state = network(*TO_SAKAIDE, ('E2', '57', 1), trains=[D.TRAIN_2], phase=D.GAME_PHASE_5)
    assert routes(state) == [('2', 70, stops('K4', 'J1'))]
    MAP.place_token(state, HEX['E2'], IR)
    set_trains(state, IR, [D.TRAIN_2])
    assert routes(state, IR) == [('2', 80, stops('F1', 'E2'))]


@pytest.mark.parametrize('train, best', [
    (D.TRAIN_2, ('2', 50, stops('K4', 'J1'))),
    (D.TRAIN_3, ('3', 80, stops('K4', 'J5', 'I4'))),
    (D.TRAIN_4, ('4', 100, stops('J1', 'K4', 'J5', 'I4'))),
])
def test_a_train_counts_stops_up_to_its_distance_towns_included(train, best):
    """Every center a route visits is a stop, the town at J5 as well, so a
    2-train cannot reach Kotohira past it; a route may pass through its
    token city."""
    assert routes(network(*TO_SAKAIDE, *TO_KOTOHIRA, trains=[train])) == [best]


def test_a_d_train_has_no_distance_limit():
    state = network(*CHAIN, trains=[D.TRAIN_D], phase=D.GAME_PHASE_D)
    assert routes(state) == [('D', 180, stops('K4', 'J5', 'I4', 'I2', 'J1'))]
    set_trains(state, KO, [D.TRAIN_6])
    assert routes(state) == [('6', 140, stops('K4', 'J5', 'I4', 'I2', 'J1'))]


def test_an_offboard_ends_a_route():
    """Marugame's track meets Sakaide & Okayama, which no route may pass."""
    state = network(*TO_SAKAIDE, ('I2', '5', 4), trains=[D.TRAIN_3])
    assert routes(state) == [('3', 50, stops('K4', 'J1'))]


def test_a_blocked_city_ends_a_route_and_a_reservation_blocks_nothing():
    state = network(*VIA_MARUGAME, trains=[D.TRAIN_3])
    assert not MAP.blocked_for(state, HEX['I2'], KO)  # SR's home reservation holds its one slot
    assert routes(state) == [('3', 70, stops('K4', 'I2', 'J1'))]
    MAP.place_token(state, HEX['I2'], SR)
    assert MAP.blocked_for(state, HEX['I2'], KO)
    assert routes(state) == [('3', 50, stops('K4', 'I2'))]


def test_trains_share_no_track_but_may_share_stops():
    state = network(*TO_SAKAIDE, trains=[D.TRAIN_2, D.TRAIN_2])
    assert routes(state) == [('2', 50, stops('K4', 'J1'))]  # the line is used up
    for hex_id, tile, rotation in TO_KOTOHIRA:
        MAP.lay(state, HEX[hex_id], TILE[tile], rotation)
    assert sorted(routes(state)) == [('2', 40, stops('K4', 'J5')), ('2', 50, stops('K4', 'J1'))]


@pytest.mark.parametrize('phase, train, revenue', [
    (D.GAME_PHASE_2, D.TRAIN_2, 50), (D.GAME_PHASE_4, D.TRAIN_4, 50), (D.GAME_PHASE_5, D.TRAIN_5, 70),
    (D.GAME_PHASE_D, D.TRAIN_6, 70), (D.GAME_PHASE_D, D.TRAIN_D, 110),
])
def test_offboards_pay_by_phase_and_d_trains_their_d_value(phase, train, revenue):
    assert max_route_revenue(network(*TO_SAKAIDE, trains=[train], phase=phase), KO) == revenue


def test_best_routes_list_every_hex_in_route_order():
    """The hexes run from one end to the other, each stop once and the plain
    hex J3 between; ``stops`` are its revenue centers."""
    (route,) = best_routes_py(network(*TO_SAKAIDE, *TO_KOTOHIRA, trains=[D.TRAIN_4]), KO)
    assert normal(route['hexes']) == normal([HEX[h] for h in ('J1', 'J3', 'K4', 'J5', 'I4')])
    assert route['stops'] == [h for h in route['hexes'] if D.HEX_RC[h] >= 0] and route['train'] == D.TRAIN_4


def _cache(state, corp):
    layout, fields = get_layout(state.max_players), get_corp_fields()
    return layout.corps_offset + corp * fields.size + fields.route_cache


def _dirty(state):
    return int(state._array[get_layout(state.max_players).global_offset + get_global_fields().route_dirty])


def test_the_route_cache_goes_stale_with_the_map_trains_and_phase():
    state = network(*TO_SAKAIDE, trains=[D.TRAIN_2])
    assert _dirty(state) == ALL_CORPS
    assert [max_route_revenue(state, c) for c in range(D.NUM_CORPS)] == [0, 0, 0, 50, 0, 0, 0]
    assert _dirty(state) == 0 and state._array[_cache(state, KO)] == 50
    state._array[_cache(state, KO)] = 999  # test setup: a clean cache is read, not solved
    assert max_route_revenue(state, KO) == 999

    set_trains(state, KO, [D.TRAIN_2, D.TRAIN_3])  # the corp's trains: its own cache only
    assert _dirty(state) == 1 << KO and max_route_revenue(state, KO) == 50
    for hex_id, tile, rotation in TO_KOTOHIRA:      # lays: the corps whose track touches the hex
        MAP.lay(state, HEX[hex_id], TILE[tile], rotation)
    assert _dirty(state) == 1 << KO and max_route_revenue(state, KO) == 130
    MAP.place_token(state, HEX['I2'], SR)          # tokens: their corp, and those whose track faces the hex
    assert _dirty(state) == 1 << SR | 1 << KO and max_route_revenue(state, KO) == 130
    MAP.place_token(state, HEX['E2'], IR)          # far from KO's track: IR alone
    assert _dirty(state) == 1 << SR | 1 << IR
    assert (max_route_revenue(state, SR), max_route_revenue(state, IR), _dirty(state)) == (0, 0, 0)
    PHASE.set_current(state, D.GAME_PHASE_5)        # phases: every corp
    assert _dirty(state) == ALL_CORPS and max_route_revenue(state, KO) == 150


def _hyp_dirty(state):
    return int(state._array[get_layout(state.max_players).global_offset + get_global_fields().hyp_dirty])


def test_the_route_hexes_are_those_of_the_best_routes():
    """``route_hexes``, cached with the route max: the union of the routes' hexes."""
    state = network(*TO_SAKAIDE, *TO_KOTOHIRA, trains=[D.TRAIN_2, D.TRAIN_2])
    expected = sorted({h for route in best_routes_py(state, KO) for h in route['hexes']})
    assert route_hexes_py(state, KO) == expected == sorted(HEX[h] for h in ('J1', 'J3', 'K4', 'J5'))
    assert route_hexes_py(state, AR) == []
    set_trains(state, KO, [D.TRAIN_2])  # one train: the better route alone, to Sakaide
    assert route_hexes_py(state, KO) == sorted(HEX[h] for h in ('J1', 'J3', 'K4'))
    set_trains(state, KO, [D.TRAIN_3])  # a 3 reaches Kotohira past the town
    assert route_hexes_py(state, KO) == sorted(HEX[h] for h in ('K4', 'J5', 'I4'))


@pytest.mark.parametrize('phase, trains, rusted, rust_phase', [
    (D.GAME_PHASE_2, [D.TRAIN_2, D.TRAIN_2], [], D.GAME_PHASE_4),           # the 2s rust at phase 4
    (D.GAME_PHASE_3, [D.TRAIN_2, D.TRAIN_3], [D.TRAIN_3], D.GAME_PHASE_4),
    (D.GAME_PHASE_4, [D.TRAIN_3, D.TRAIN_4], [D.TRAIN_4], D.GAME_PHASE_6),  # the 3s at phase 6, past phase 5
    (D.GAME_PHASE_6, [D.TRAIN_4, D.TRAIN_6], [D.TRAIN_6], D.GAME_PHASE_D),  # the 4s at phase D
    (D.GAME_PHASE_D, [D.TRAIN_6, D.TRAIN_D], [D.TRAIN_6, D.TRAIN_D], D.GAME_PHASE_D),  # nothing left to rust
])
def test_the_route_max_after_the_next_rust(phase, trains, rusted, rust_phase):
    """``route_after_rust``: the trains that survive the next rust, at that
    phase's revenues (the offboard pays more from phase 5)."""
    state = network(*TO_SAKAIDE, *TO_KOTOHIRA, trains=trains, phase=phase)
    later = network(*TO_SAKAIDE, *TO_KOTOHIRA, trains=rusted, phase=rust_phase)
    assert route_hypotheticals_py(state, KO)[0] == max_route_revenue(later, KO)
    assert route_hypotheticals_py(state, AR) == (0, 0)


def test_the_route_max_with_the_next_train():
    """``route_with_next``: one more of the cheapest train on sale, with the
    phase its purchase starts; the route max now without room for it."""
    state = network(*TO_SAKAIDE, *TO_KOTOHIRA, trains=[D.TRAIN_2])
    more = network(*TO_SAKAIDE, *TO_KOTOHIRA, trains=[D.TRAIN_2, D.TRAIN_2])
    assert TRAINS.cheapest_type(state) == D.TRAIN_2
    assert route_hypotheticals_py(state, KO)[1] == max_route_revenue(more, KO) > max_route_revenue(state, KO)
    # The first 4 starts phase 4, where the 2 rusts: the 4 runs alone.
    state = network(*TO_SAKAIDE, *TO_KOTOHIRA, trains=[D.TRAIN_2, D.TRAIN_3], phase=D.GAME_PHASE_3)
    for t in (D.TRAIN_2, D.TRAIN_3):
        TRAINS.set_depot(state, t, 0)
    after = network(*TO_SAKAIDE, *TO_KOTOHIRA, trains=[D.TRAIN_3, D.TRAIN_4], phase=D.GAME_PHASE_4)
    assert TRAINS.cheapest_type(state) == D.TRAIN_4
    assert route_hypotheticals_py(state, KO)[1] == max_route_revenue(after, KO)
    # At the limit (two trains from phase 5) there is nothing to buy.
    state = network(*TO_SAKAIDE, *TO_KOTOHIRA, trains=[D.TRAIN_4, D.TRAIN_5], phase=D.GAME_PHASE_5)
    assert route_hypotheticals_py(state, KO)[1] == max_route_revenue(state, KO)


def test_the_hypothetical_route_maxes_go_stale_with_the_routes_and_the_next_train():
    state = network(*TO_SAKAIDE, *TO_KOTOHIRA, trains=[D.TRAIN_2])
    for c in range(D.NUM_CORPS):
        route_hypotheticals_py(state, c)
    assert _hyp_dirty(state) == _dirty(state) == 0
    set_trains(state, KO, [D.TRAIN_2, D.TRAIN_2])               # whatever makes a route stale
    assert _hyp_dirty(state) == _dirty(state) == 1 << KO
    route_hypotheticals_py(state, KO)
    TRAINS.set_depot(state, D.TRAIN_2, TRAINS.depot(state, D.TRAIN_2) - 1)  # a copy sold: the same next train
    assert _hyp_dirty(state) == 0
    with_two = route_hypotheticals_py(state, KO)[1]
    TRAINS.set_depot(state, D.TRAIN_2, 0)                        # the last: the next train is a 3
    assert TRAINS.cheapest_type(state) == D.TRAIN_3
    assert _hyp_dirty(state) == ALL_CORPS and _dirty(state) == 0
    assert route_hypotheticals_py(state, KO)[1] > with_two
    for c in range(D.NUM_CORPS):
        route_hypotheticals_py(state, c)
    TRAINS.set_pool(state, D.TRAIN_2, 1)                         # a discarded 2 is the cheapest again
    assert _hyp_dirty(state) == ALL_CORPS and route_hypotheticals_py(state, KO)[1] == with_two
