"""1889 extraction against Ruby's oracles (1889/extraction-design.md §10 item 5).

On ``load_state`` of each OR snapshot of the four fixtures, extraction's own
output is decoded, the token buffer and the sparse records, so these check
token indices and plumbing; ``test_network_oracles.py`` and
``test_route_oracles.py`` check the engine beneath:

- relations 16-19, both directions, for every corp: Ruby's
  ``connected_hexes`` and ``connected_nodes`` (``operating.graph``) for each
  corp with a token, and nothing for the others;
- at Track, HEX 102's candidate count per hex against Ruby's legal lays
  (``ruby_lays``); and after TRACK 1 + h on a copy, the candidate rows'
  tiles and rotations, their hex (relation 21), and the hexes across their
  exits (relation 23) against Ruby's lays at h, or the hex's own tile when
  the driver applies a lone lay;
- at TOKEN, HEX 103 against the cities Ruby's Token step accepts
  (``operating.tokens``);
- at Route, CORP 34 against AutoRouter's revenue (``operating.auto_router``).
"""

from __future__ import annotations

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_OK
from g1889.entities.turn import TURN
from g1889.relations import RelationId as REL, RelationSize, get_relation_coord_data
from g1889.state import GameState
from g1889.token_data import TileBlock, TokenDataSize, TokenRow as R, get_num_tokens, get_token_data

from tests.g1889.replay.extracts import FIXTURES, load_fixture_extract
from tests.g1889.replay.snapshot import CORP, HEX, TILE, load_state, ruby_lays

TB = int(TileBlock.TILE_BLOCK)
CANDIDATES, TOKENABLE, ROUTE_MAX = 102, 103, 34  # HEX 102, HEX 103, CORP 34

# (OR snapshots, corps with a token over them, Track snapshots, hexes with a
# lay, of which the driver applies a lone lay, TOKEN decisions, Route
# snapshots) per fixture.
COVERAGE = {
    '314': (197, 947, 43, 267, 62, 19, 32),
    '962': (243, 1417, 57, 446, 117, 15, 48),
    'hs_jmmljkbw_17502': (2, 2, 1, 1, 0, 0, 0),
    'hs_lhglbbiz_17502': (2, 2, 1, 1, 0, 0, 0),
}


def _extract(state: GameState) -> tuple[np.ndarray, set[tuple[int, int, int]]]:
    """The state's tokens, and its records as (relation, query, key) with value 1."""
    n = TURN.num_players(state)
    tokens = np.full((get_num_tokens(n), int(TokenDataSize.TOKEN_DIM)), np.nan, np.float32)
    records = np.empty((int(RelationSize.MAX_RELATION_EDGES), 4), np.uint8)
    get_token_data(state, tokens, layout_version=1)
    count = get_relation_coord_data(state, records)
    assert (records[:count, 3] > 0).all()
    return tokens, {(int(r), int(q), int(k)) for r, q, k, v in records[:count] if v == 1}


def _reach(records: set, c: int) -> tuple[set, set]:
    """Corp c's reached hexes and centers, after checking each against its reverse."""
    hexes, centers = set(), set()
    for relation, out in ((REL.REL_REACHES_HEX, hexes), (REL.REL_REACHES_CENTER, centers)):
        for r, q, k in records:
            if r == relation and q == R.ROW_CORP + c:
                assert (relation + 1, k, q) in records
                out.add(D.HEX_IDS[k - R.ROW_HEX])
        assert len(out) == sum(r == relation + 1 and k == R.ROW_CORP + c for r, q, k in records)
    return hexes, centers


def _tile(tokens: np.ndarray, row: int) -> tuple[str, int]:
    """The supply tile and canonical rotation in a HEX or CANDIDATE row's tile block."""
    block = tokens[row, TB:TB + 62]
    (tile,) = np.flatnonzero(block[:D.NUM_SUPPLY_TILES])
    mask = sum(1 << int(i) for i in np.flatnonzero(block[41:62]))
    return D.TILE_NAMES[tile], min(r for r in range(D.NUM_SIDES) if D.TILE_MASK[tile][r] == mask)


def _or_snapshots(extract):
    for record in extract['records']:
        snap = record['state']
        if snap['game']['round'] == 'OR' and not snap['game']['finished']:
            yield record['index'], snap, load_state(snap, len(extract['players']))


def _check_track(fixture, index, state, snap, tokens) -> tuple[int, int]:
    """HEX 102 and the candidate rows after each hex's TRACK choice; returns
    the hexes with a lay and how many of them the driver lays alone."""
    sym = D.CORP_SYMS[TURN.operating_corp(state)]
    lays = ruby_lays(snap['operating']['lays'][sym], snap['corporations'][sym]['cash'])
    by_hex = {h: {(tile, r) for hex_id, tile, r in lays if HEX[hex_id] == h} for h in {HEX[l[0]] for l in lays}}
    counts = np.rint(tokens[R.ROW_HEX:R.ROW_CANDIDATE, CANDIDATES] * D.MAX_LAY_CANDIDATES).astype(int)
    assert {h: n for h, n in enumerate(counts) if n} == {h: len(v) for h, v in by_hex.items()}, (fixture, index)
    forced = 0
    for h, ruby in by_hex.items():
        copy = GameState.from_array(state._array, TURN.num_players(state))
        assert DRIVER.apply_action(copy, K.TRACK_HEX + D.HEX_TRACK_SLOT[h]) == STATUS_OK
        after, records = _extract(copy)
        if A.get_decision_phase_py(copy) != K.DPHASE_TILE:  # a lone lay, applied
            assert {_tile(after, R.ROW_HEX + h)} == ruby, (fixture, index, D.HEX_IDS[h])
            forced += 1
            continue
        visible = int(after[R.ROW_CANDIDATE:R.ROW_PLAYER, 0].sum())
        ours = [_tile(after, R.ROW_CANDIDATE + k) for k in range(visible)]
        assert visible == len(ruby) == len(set(ours)) and set(ours) == ruby, (fixture, index, D.HEX_IDS[h])
        for k, (tile, r) in enumerate(ours):
            exits = {R.ROW_HEX + D.HEX_NEIGHBOR[h][s] for s in range(D.NUM_SIDES)
                     if D.TILE_EXITS[TILE[tile]][r] >> s & 1}
            assert {key for rel, q, key in records if rel == REL.REL_CANDIDATE_AT and q == R.ROW_CANDIDATE + k} \
                == {R.ROW_HEX + h}, (fixture, index, D.HEX_IDS[h], k)
            assert {key for rel, q, key in records if rel == REL.REL_CANDIDATE_EXIT and q == R.ROW_CANDIDATE + k} \
                == exits, (fixture, index, D.HEX_IDS[h], k)
    return len(by_hex), forced


def check_extraction(label, extract) -> tuple[int, ...]:
    """The counts of ``COVERAGE``. Extraction is for the model's 3-5 players."""
    snapshots = corps = tracks = lay_hexes = forced = token_steps = routes = 0
    for index, snap, state in _or_snapshots(extract):
        snapshots += 1
        step, operating = snap['game']['step'], snap['operating']
        tokens, records = _extract(state)
        for c, sym in enumerate(D.CORP_SYMS):
            hexes, centers = _reach(records, c)
            ruby = operating['graph'].get(sym, {'hexes': {}, 'nodes': []})
            assert (hexes, centers) == (set(ruby['hexes']), set(ruby['nodes'])), (label, index, sym)
            corps += sym in operating['graph']
        phase = A.get_decision_phase_py(state)
        if phase == K.DPHASE_TRACK and operating['lays'] is not None:
            tracks += 1
            n, f = _check_track(label, index, state, snap, tokens)
            lay_hexes, forced = lay_hexes + n, forced + f
        elif phase == K.DPHASE_TOKEN:
            sym = D.CORP_SYMS[TURN.operating_corp(state)]
            ours = {D.HEX_IDS[h] for h in np.flatnonzero(tokens[R.ROW_HEX:R.ROW_CANDIDATE, TOKENABLE])}
            assert ours == set(operating['tokens'][sym]), (label, index)
            token_steps += 1
        elif step == 'Route':
            c = CORP[operating['operator']]
            assert round(float(tokens[R.ROW_CORP + c, ROUTE_MAX]) * 100) == operating['auto_router']['revenue'], \
                (label, index)
            routes += 1
    return snapshots, corps, tracks, lay_hexes, forced, token_steps, routes


@pytest.mark.parametrize('fixture', FIXTURES)
def test_extraction_matches_ruby_at_every_or_snapshot(fixture):
    assert check_extraction(fixture, load_fixture_extract(fixture)) == COVERAGE[fixture]
