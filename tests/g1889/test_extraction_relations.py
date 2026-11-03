"""1889 relations (1889/extraction-design.md §6, §10 item 3).

On test_extraction's random-playout states, the relations extracted on the
live state, as the eval worker extracts them:

- the records are §6.1's edges, decoded from the entity APIs (``PLAYERS``,
  ``CORPS``, ``PRIVATES``, ``TRAINS``, ``MAP``, ``GRAPH``) with their values,
  and nothing else: the candidate relations stay empty until sub-step 7.2;
- they join only visible tokens, each group stays within its §6.1 bound, and
  the bounds sum to §6.3's 1,844, below MAX_RELATION_EDGES;
- the dense planes on a copy with every cache stale scatter the same records,
  and write back what the token fill does, so either fill may come first;
- the static planes are the tables' neighbors, homes, and ability hexes.

Conformance (tests/test_game_adapter.py) checks the batches and the scratch
states.
"""

import numpy as np
import pytest

import g1889.constants as K
import g1889.data.tables as D
from g1889.data import load_tables
from g1889.entities.corp import CORPS
from g1889.entities.graph import GRAPH
from g1889.entities.hexmap import MAP
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.relations import (
    RelationId as REL,
    RelationSize,
    StaticRelationId as SREL,
    get_relation_coord_data,
    get_relation_data,
    static_relation_planes,
)
from g1889.token_data import TokenRow as R, get_num_tokens
from games import RELATION_RECORD_WIDTH

from tests.g1889.test_extraction import _playouts, _stale, _state, _tokens

NUM_RELATIONS = int(REL.NUM_RELATIONS)
MAX_EDGES = int(RelationSize.MAX_RELATION_EDGES)
CANDIDATE_RELATIONS = range(int(REL.REL_CANDIDATE_AT), NUM_RELATIONS)  # sub-step 7.2

# §6.1's bound per group of relation ids, and §6.3's sum.
BOUNDS = {
    (REL.REL_PLAYER_HOLDS,): 35, (REL.REL_CORP_HELD_BY,): 35,
    (REL.REL_PLAYER_SHARES,): 35, (REL.REL_CORP_SHARES,): 35,
    (REL.REL_PRESIDES,): 7, (REL.REL_PRESIDENT,): 7,
    (REL.REL_PLAYER_OWNS, REL.REL_CORP_OWNS): 7, (REL.REL_OWNED_BY_PLAYER, REL.REL_OWNED_BY_CORP): 7,
    (REL.REL_BIDS_ON,): 35, (REL.REL_BID_BY,): 35,
    (REL.REL_CORP_TRAINS,): 28, (REL.REL_TRAIN_HELD_BY,): 28,
    (REL.REL_STATION,): 15, (REL.REL_STATION_OF,): 15,
    (REL.REL_REACHES_HEX,): 364, (REL.REL_HEX_REACHED_BY,): 364,
    (REL.REL_REACHES_CENTER,): 175, (REL.REL_CENTER_REACHED_BY,): 175,
    (REL.REL_TRACK_LINK,): 232,
    (REL.REL_CANDIDATE_AT,): 15, (REL.REL_CANDIDATE_OF,): 15,
    (REL.REL_CANDIDATE_EXIT,): 90, (REL.REL_EXIT_OF,): 90,
}


def _player(i):
    return R.ROW_PLAYER + i


def _corp(c):
    return R.ROW_CORP + c


def _private(p):
    return R.ROW_PRIVATE + p


def _hex(h):
    return R.ROW_HEX + h


@pytest.fixture(scope='module')
def samples():
    """(row before extraction, max_players, records, count) at every decision
    of the playouts, the records extracted on the live state."""
    out = []
    for state, max_players in _playouts():
        row = state._array.copy()
        records = np.full((MAX_EDGES, RELATION_RECORD_WIDTH), 0xFF, np.uint8)
        count = get_relation_coord_data(state, records, max_players)
        out.append((row, max_players, records, count))
    return out


def _dense(state, max_players):
    num_tokens = get_num_tokens(max_players)
    planes = np.full((NUM_RELATIONS, num_tokens, num_tokens), 0xFF, np.uint8)
    get_relation_data(state, planes, max_players)
    return planes


def _materialize(records, count, max_players):
    num_tokens = get_num_tokens(max_players)
    planes = np.zeros((NUM_RELATIONS, num_tokens, num_tokens), np.uint8)
    relation, query, key, value = records[:count].T
    planes[relation, query, key] = value
    return planes


def _track_at(state, h, side):
    return D.TILE_EXITS[MAP.tile(state, h)][MAP.rotation(state, h)] >> side & 1


def _edges(state):
    """§6.1's edges, {(relation, query, key): value}, from the entity APIs."""
    edges = {}

    def pair(relation, a, b, value=1):
        edges[int(relation), a, b] = edges[relation + 1, b, a] = value

    num_players = TURN.num_players(state)
    for c, corp in enumerate(CORPS):
        for i in range(num_players):
            if shares := PLAYERS[i].shares(state, c):
                pair(REL.REL_PLAYER_HOLDS, _player(i), _corp(c))
                pair(REL.REL_PLAYER_SHARES, _player(i), _corp(c), shares)
        if (president := corp.president(state)) >= 0:
            pair(REL.REL_PRESIDES, _player(president), _corp(c))
        for t in range(D.NUM_TRAIN_TYPES):
            if count := TRAINS.corp_trains(state, c, t):
                pair(REL.REL_CORP_TRAINS, _corp(c), R.ROW_TRAIN + t, count)
        for h in range(D.NUM_HEXES):
            if MAP.tokens(state, h) >> c & 1:
                pair(REL.REL_STATION, _corp(c), _hex(h))
            if GRAPH.sides_reached(state, c, h):
                pair(REL.REL_REACHES_HEX, _corp(c), _hex(h))
            if GRAPH.node_reached(state, c, h):
                pair(REL.REL_REACHES_CENTER, _corp(c), _hex(h))
    for p, private in enumerate(PRIVATES):
        owner = private.owner(state)
        if owner >= K.OWNER_CORP:
            pair(REL.REL_CORP_OWNS, _corp(owner - K.OWNER_CORP), _private(p))
        elif owner >= K.OWNER_PLAYER:
            pair(REL.REL_PLAYER_OWNS, _player(owner), _private(p))
        for i in range(num_players):
            if PLAYERS[i].bid(state, p):
                pair(REL.REL_BIDS_ON, _player(i), _private(p))
    for h in range(D.NUM_HEXES):
        for side, across in enumerate(D.HEX_NEIGHBOR[h]):
            if across >= 0 and _track_at(state, h, side) and _track_at(state, across, (side + 3) % D.NUM_SIDES):
                edges[int(REL.REL_TRACK_LINK), _hex(h), _hex(across)] = 1
    return edges


# =============================================================================
# DYNAMIC RELATIONS (§6.1)
# =============================================================================

def test_records_are_the_entity_relations(samples):
    """Each record is one of §6.1's edges, with its value, once, and every edge
    has its record; the tail is zero. Every relation but the candidates' occurs."""
    seen = set()
    for row, _, records, count in samples:
        decoded = {(r, q, k): v for r, q, k, v in records[:count].tolist()}
        assert len(decoded) == count and 0 not in decoded.values()
        assert decoded == _edges(_state(row.copy()))
        assert not records[count:].any()
        seen.update(r for r, _, _ in decoded)
    assert seen == set(range(NUM_RELATIONS)) - set(CANDIDATE_RELATIONS), sorted(seen)


def test_records_join_visible_tokens_within_their_bounds(samples):
    """Every record's query and key are visible rows, so the model drops none;
    each group stays within its §6.1 bound; and the bounds sum to §6.3's
    1,844, below MAX_RELATION_EDGES, which the largest count respects."""
    assert sum(BOUNDS.values()) == 1844 < MAX_EDGES == 2048
    assert sorted(r for group in BOUNDS for r in group) == list(range(NUM_RELATIONS))
    largest = 0
    for row, max_players, records, count in samples:
        visible = _tokens(_state(row.copy()), max_players)[:, 0] == 1
        assert visible[records[:count, 1]].all() and visible[records[:count, 2]].all()
        per_relation = np.bincount(records[:count, 0], minlength=NUM_RELATIONS)
        for group, bound in BOUNDS.items():
            assert per_relation[list(group)].sum() <= bound, (group, per_relation[list(group)].sum())
        largest = max(largest, count)
    assert 0 < largest <= MAX_EDGES


def test_dense_planes_match_the_records_whatever_the_caches(samples):
    """The dense planes of a copy with every cache stale hold the records, and
    the fill writes back the caches the token fill does, so the fills may come
    in either order."""
    for row, max_players, records, count in samples:
        stale = _stale(row)
        planes = _dense(_state(stale), max_players)
        assert np.count_nonzero(planes) == count
        np.testing.assert_array_equal(planes, _materialize(records, count, max_players))
        by_tokens = row.copy()
        _tokens(_state(by_tokens), max_players)
        np.testing.assert_array_equal(stale, by_tokens)


def test_track_links_are_symmetric_on_open_sides(samples):
    """TRACK_LINK is its own reverse, and links only sides open both ways."""
    open_both = np.zeros((D.NUM_HEXES, D.NUM_HEXES), bool)
    for h in range(D.NUM_HEXES):
        for side, across in enumerate(D.HEX_NEIGHBOR[h]):
            if across >= 0:
                open_both[h, across] = (D.HEX_EXIT_MASK[h] >> side & 1
                                        and D.HEX_EXIT_MASK[across] >> (side + 3) % D.NUM_SIDES & 1)
    assert open_both.sum() == BOUNDS[(REL.REL_TRACK_LINK,)]
    for _, max_players, records, count in samples[::10]:
        links = _materialize(records, count, max_players)[REL.REL_TRACK_LINK, _hex(0):_hex(D.NUM_HEXES),
                                                          _hex(0):_hex(D.NUM_HEXES)]
        np.testing.assert_array_equal(links, links.T)
        assert not (links & ~open_both).any()


# =============================================================================
# STATIC RELATIONS (§6.2)
# =============================================================================

@pytest.mark.parametrize('max_players', [3, 4, 5])
def test_static_planes_follow_the_tables(max_players):
    """HEX_SIDE s joins each hex to the one across side s, whose side s + 3
    faces back; HOME each corp to its home; ABILITY_HEX TR and ER to the hexes
    they block and MF to its ports; and each reverse is the transpose."""
    tables = load_tables()
    num_tokens = get_num_tokens(max_players)
    planes = static_relation_planes(max_players)
    assert planes.shape == (SREL.NUM_STATIC_RELATIONS, num_tokens, num_tokens) and planes.dtype == np.uint8

    def nonzero(plane):
        return set(zip(*np.nonzero(plane)))

    for side in range(D.NUM_SIDES):
        expected = {(_hex(h), _hex(entry['neighbors'][side]))
                    for h, entry in enumerate(tables['hexes']) if entry['neighbors'][side] >= 0}
        assert nonzero(planes[SREL.SREL_HEX_SIDE + side]) == expected
        np.testing.assert_array_equal(planes[SREL.SREL_HEX_SIDE + side].T,
                                      planes[SREL.SREL_HEX_SIDE + (side + 3) % D.NUM_SIDES])
    assert planes[SREL.SREL_HEX_SIDE:SREL.SREL_HEX_SIDE + D.NUM_SIDES].sum() == 244
    homes = {(_corp(c), _hex(corp['home'])) for c, corp in enumerate(tables['corporations'])}
    abilities = {(p, h) for p, private in enumerate(tables['privates'])
                 for ability in private['abilities'] for h in ability.get('hexes', [])}
    assert {(tables['privates'][p]['sym'], tables['hexes'][h]['id']) for p, h in abilities} == {
        ('TR', 'K4'), ('ER', 'C4'), ('MF', 'B11'), ('MF', 'G10'), ('MF', 'I12'), ('MF', 'J9')}
    abilities = {(_private(p), _hex(h)) for p, h in abilities}
    for relation, edges in ((SREL.SREL_HOME, homes), (SREL.SREL_ABILITY_HEX, abilities)):
        assert nonzero(planes[relation]) == edges
        np.testing.assert_array_equal(planes[relation + 1], planes[relation].T)
    assert planes.max() == 1 and not planes[:, R.ROW_PLAYER:].any() and not planes[:, :, R.ROW_PLAYER:].any()
