"""The checked-in 1889 tables against the 18xx.games Ruby engine.

The upgrade comparison asks the Ruby engine (``upgrade_oracle.rb``) the parts
of its tile-lay check that need no game state: ``Game::Base#upgrades_to?``
(color, preserved paths, label, revenue-center counts) and
``Step::Tracker#legal_tile_rotation?`` (exits onto the map, old paths kept in
place, city and slot preservation). It covers every hex's preprinted tile and
every supply tile at every rotation as the old tile, every supply tile and
rotation as the new one, and the MF and ER tile-lay abilities. 1889 overrides
neither method; Kouchi, Kotohira, and Takamatsu are handled by labels.

Left out, because they need game state or are not part of the check: phase
tile colors, supply counts, connectivity (the oracle reports every side as
reached) and the semi-restrictive new-track rule, blockers and ability owner
conditions, ``blocks_lay`` filtering of normal lays (437), terrain cost, and
token reservations.
"""

import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest

from g1889.data import TABLES_PATH, load_tables, mask_exits
from g1889.data import generate

RUBY = shutil.which('ruby')
ORACLE = Path(__file__).with_name('upgrade_oracle.rb')

pytestmark = pytest.mark.skipif(RUBY is None, reason='requires ruby')


def test_regenerating_leaves_tables_unchanged():
    text = generate.render(generate.build(generate.run_dump()))
    assert text == TABLES_PATH.read_text()
    pxd, pyx = generate.render_cython(json.loads(text))
    assert (pxd, pyx) == (generate.PXD_PATH.read_text(), generate.PYX_PATH.read_text())


@pytest.fixture(scope='module')
def oracle():
    assert RUBY is not None
    out = subprocess.run([RUBY, str(ORACLE)], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


@pytest.fixture(scope='module')
def legal():
    """legal(old tile, rotation, open sides) -> {(new name, rotation)} from the tables."""
    tables = load_tables()
    tiles = tables['tiles']
    targets = defaultdict(list)
    for row in tables['upgrades']:
        targets[tuple(row['from'])] = row['to']

    def lookup(old, rotation, exit_mask):
        return {(tiles[new]['name'], r) for new, r in targets[(old, rotation)]
                if mask_exits(tiles[new]['masks'][r]) & ~exit_mask == 0}
    return lookup


def test_preprinted_upgrades_match_ruby(oracle, legal):
    hexes = load_tables()['hexes']
    ours = {(h['id'], *target) for h in hexes for target in legal(h['tile'], 0, h['exit_mask'])}
    assert ours == {tuple(row) for row in oracle['preprinted']}


def test_supply_tile_upgrades_match_ruby(oracle, legal):
    tables = load_tables()
    supply = [(i, t['name']) for i, t in enumerate(tables['tiles']) if t['hex'] < 0]
    assert {sum(1 << int(s) for s in key) for key in oracle['supply']} == {h['exit_mask'] for h in tables['hexes']}
    for key, rows in oracle['supply'].items():
        exit_mask = sum(1 << int(s) for s in key)
        ours = {(name, r, *target) for i, name in supply for r in range(6) for target in legal(i, r, exit_mask)}
        assert ours == {tuple(row) for row in rows}, f'open sides {key}'


def test_ability_tile_lays_match_ruby(oracle, legal):
    tables = load_tables()
    hexes, tiles = tables['hexes'], tables['tiles']
    ours = set()
    for private in tables['privates']:
        for ability in private['abilities']:
            if ability['type'] != 'tile_lay':
                continue
            names = {tiles[t]['name'] for t in ability['tiles']}
            for h in ability['hexes']:
                ours |= {(private['sym'], hexes[h]['id'], name, r)
                         for name, r in legal(hexes[h]['tile'], 0, hexes[h]['exit_mask']) if name in names}
    assert ours == {tuple(row) for row in oracle['abilities']}
    assert {row[0] for row in oracle['abilities']} == {'MF', 'ER'}
