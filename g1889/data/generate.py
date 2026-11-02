"""Generate ``g1889/data/tables.json`` from the 18xx.games engine.

``dump_1889.rb`` reads the Ruby engine's 1889 game objects and constants;
this module indexes them, derives path masks and the upgrade table, and
writes one record per line so regenerating is byte-identical and diffs stay
readable.

    .venv/bin/python -m g1889.data.generate
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from g1889.data import (
    COLORS, LABELS, MARKET_TYPES, NUM_SIDES, RC_TYPES, TABLES_PATH, TERRAINS, mask_exits, path_mask,
)

DUMP_SCRIPT = Path(__file__).with_name('dump_1889.rb')
TRAIN_KEYS = {'name', 'distance', 'price', 'num', 'rusts_on', 'available_on', 'discount', 'events'}
PHASE_KEYS = {'name', 'on', 'train_limit', 'tiles', 'operating_rounds', 'status'}


def run_dump() -> dict[str, Any]:
    ruby = shutil.which('ruby')
    if ruby is None:
        raise RuntimeError('ruby is required to read the 18xx engine')
    out = subprocess.run([ruby, str(DUMP_SCRIPT)], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def _tile(t: dict, hex_index: int) -> dict:
    if any(t['unsupported'].values()) or len(t['revenue_centers']) > 1:
        raise ValueError(f"tile {t['name']} has parts the tables do not model: {t}")
    rc = t['revenue_centers'][0] if t['revenue_centers'] else None
    if rc and rc['type'] != 'offboard' and len(set(rc['phase_revenue'] + [rc['d_revenue']])) != 1:
        raise ValueError(f"tile {t['name']} has phase-dependent city/town revenue")
    paths = sorted(sorted(p) for p in t['paths'])
    masks = [path_mask(paths, r) for r in range(NUM_SIDES)]
    if mask_exits(masks[0]) != sum(1 << e for e in t['exits']):
        raise ValueError(f"tile {t['name']}: paths disagree with Ruby exits")
    terrains = {name for u in t['upgrades'] for name in u['terrains']}
    return {
        'name': t['name'],
        'color': COLORS.index(t['color']),
        'count': t.get('count', 0),
        'hex': hex_index,
        'label': LABELS.index(t['label'] or ''),
        'rc': RC_TYPES.index(rc['type']) if rc else 0,
        'revenue': rc['phase_revenue'][0] if rc and rc['type'] != 'offboard' else 0,
        'slots': rc['slots'] if rc else 0,
        'cost': sum(u['cost'] for u in t['upgrades']),
        'terrain': sum(1 << TERRAINS.index(name) for name in terrains),
        'blocks_lay': t['blocks_lay'],
        'paths': paths,
        'masks': masks,
    }


def upgrades_to(old: dict, rotation: int, new: dict, new_rotation: int) -> bool:
    """1889-rules.md 8.4.1-8.4.3: the next color, the same label and revenue-center
    type, and every old path kept in place. Slots never shrink on a 1889
    upgrade, so tokens always fit (checked in tests)."""
    return (new['color'] == old['color'] + 1
            and new['label'] == old['label']
            and new['rc'] == old['rc']
            and old['masks'][rotation] & ~new['masks'][new_rotation] == 0)


def _upgrade_rows(tiles: list[dict]) -> list[dict]:
    supply = [i for i, t in enumerate(tiles) if t['hex'] < 0]
    rows = []
    for i, old in enumerate(tiles):
        for rotation in range(NUM_SIDES) if old['hex'] < 0 else (0,):
            to = [[j, r] for j in supply for r in range(NUM_SIDES) if upgrades_to(old, rotation, tiles[j], r)]
            if to:
                rows.append({'from': [i, rotation], 'to': to})
    return rows


def build(dump: dict[str, Any]) -> dict[str, Any]:
    if dump['layout'] != 'flat':
        raise ValueError(f"unexpected layout {dump['layout']}")
    hexes = sorted(dump['hexes'], key=lambda h: (h['x'], h['y']))
    hex_index = {h['id']: i for i, h in enumerate(hexes)}
    corp_index = {c['sym']: i for i, c in enumerate(dump['corporations'])}
    private_index = {c['sym']: i for i, c in enumerate(dump['companies'])}
    train_index = {t['name']: i for i, t in enumerate(dump['trains'])}
    phase_index = {p['name']: i for i, p in enumerate(dump['phases'])}

    supply = sorted(dump['tiles'], key=lambda t: (COLORS.index(t['color']), int(t['name'])))
    tiles = [_tile(t, -1) for t in supply] + [_tile(h['tile'], i) for i, h in enumerate(hexes)]
    tile_index = {t['name']: i for i, t in enumerate(tiles[:len(supply)])}

    hex_rows, rc_rows = [], []
    for i, h in enumerate(hexes):
        rcs = h['tile']['revenue_centers']
        if rcs:
            rc = rcs[0]
            rc_rows.append({
                'hex': i,
                'type': RC_TYPES.index(rc['type']),
                'port': any(icon['image'].endswith('/port.svg') for icon in h['tile']['icons']),
                'slots': rc['slots'],
                'revenue': rc['phase_revenue'],
                'd_revenue': rc['d_revenue'],
            })
        hex_rows.append({
            'id': h['id'],
            'name': h['location_name'] or '',
            'x': h['x'],
            'y': h['y'],
            'tile': len(supply) + i,
            'neighbors': [hex_index[h['neighbors'][str(s)]] if str(s) in h['neighbors'] else -1
                          for s in range(NUM_SIDES)],
            'exit_mask': sum(1 << s for s in h['open_sides']),
            'rc': len(rc_rows) - 1 if rcs else -1,
            'reserved': [corp_index[sym] for sym in h['reservations']],
            'blockers': [private_index[sym] for sym in h['blockers']],
        })

    corporations = []
    for c in dump['corporations']:
        if c['city'] is not None or c['capitalization'] != 'full':
            raise ValueError(f"unexpected corporation setup: {c}")
        corporations.append({
            'sym': c['sym'],
            'name': c['name'],
            'home': hex_index[c['home']],
            'tokens': len(c['token_prices']),
            'token_prices': c['token_prices'],
            'float_percent': c['float_percent'],
            'max_ownership_percent': c['max_ownership_percent'],
        })

    # Ability parameters as Ruby gives them, with hex, tile, and corporation ids indexed.
    ability_index = {'hexes': hex_index, 'tiles': tile_index, 'corporations': corp_index}
    privates = [{
        'sym': c['sym'],
        'name': c['name'],
        'value': c['value'],
        'revenue': c['revenue'],
        'abilities': [{k: [ability_index[k][x] for x in v] if k in ability_index else v for k, v in a.items()}
                      for a in c.get('abilities', [])],
    } for c in dump['companies']]

    trains = []
    for t in dump['trains']:
        if set(t) - TRAIN_KEYS:
            raise ValueError(f"unmodeled train keys: {set(t) - TRAIN_KEYS}")
        trains.append({
            'name': t['name'],
            'distance': t['distance'],
            'price': t['price'],
            'count': -1 if t['num'] == 'unlimited' else t['num'],
            'rusts_on': train_index.get(t.get('rusts_on'), -1),
            'available_on': phase_index.get(t.get('available_on'), -1),
            'trade_in': [t.get('discount', {}).get(name, 0) for name in train_index],
            'events': [e['type'] for e in t.get('events', [])],
        })

    phases = []
    for p in dump['phases']:
        if set(p) - PHASE_KEYS:
            raise ValueError(f"unmodeled phase keys: {set(p) - PHASE_KEYS}")
        phases.append({
            'name': p['name'],
            'on': train_index.get(p.get('on'), -1),
            'train_limit': p['train_limit'],
            'tiles': [COLORS.index(c) for c in p['tiles']],
            'operating_rounds': p['operating_rounds'],
            'status': p.get('status', []),
        })

    market = dump['market']
    if any(cell is None or len(cell['types']) > 1 for row in market for cell in row):
        raise ValueError('market has gaps or multi-type cells')

    return {
        'source': 'Generated by g1889/data/generate.py from submodules/18xx '
                  '(lib/engine/game/g_1889). Do not edit; see g1889/data/__init__.py.',
        'enums': {'colors': COLORS, 'rc_types': RC_TYPES, 'labels': LABELS, 'terrains': TERRAINS,
                  'market_types': MARKET_TYPES},
        'rules': dump['rules'],
        'players': [{
            'players': int(n),
            'starting_cash': p['starting_cash'],
            'cert_limit': p['cert_limit'],
            'privates': [private_index[sym] for sym in p['privates']],
        } for n, p in dump['players'].items()],
        'hexes': hex_rows,
        'revenue_centers': rc_rows,
        'tiles': tiles,
        'upgrades': _upgrade_rows(tiles),
        'corporations': corporations,
        'privates': privates,
        'trains': trains,
        'phases': phases,
        'market_prices': [[cell['price'] for cell in row] for row in market],
        'market_types': [[MARKET_TYPES.index(cell['types'][0] if cell['types'] else '') for cell in row]
                         for row in market],
    }


def render(tables: dict[str, Any]) -> str:
    """JSON with one top-level key per line and one row per line beneath it."""
    def compact(value: Any) -> str:
        return json.dumps(value, separators=(', ', ': '))

    lines = ['{']
    for i, (key, value) in enumerate(tables.items()):
        end = ',' if i < len(tables) - 1 else ''
        if isinstance(value, (list, dict)) and value:
            items = value.items() if isinstance(value, dict) else enumerate(value)
            open_, close = ('{', '}') if isinstance(value, dict) else ('[', ']')
            lines.append(f'  {json.dumps(key)}: {open_}')
            rows = [compact(v) if isinstance(value, list) else f'{json.dumps(k)}: {compact(v)}' for k, v in items]
            lines += [f'    {row},' for row in rows[:-1]] + [f'    {rows[-1]}', f'  {close}{end}']
        else:
            lines.append(f'  {json.dumps(key)}: {compact(value)}{end}')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def main() -> None:
    TABLES_PATH.write_text(render(build(run_dump())))
    print(f'wrote {TABLES_PATH}')


if __name__ == '__main__':
    main()
