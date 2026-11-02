"""Invariants of the checked-in 1889 tables: facts from 1889/1889.md section 2
and 1889/1889-rules.md section 2, plus path, mask, rotation, and geometry
self-consistency."""

from collections import Counter, defaultdict

from g1889.data import (
    COLORS, LABELS, MARKET_TYPES, NODE, NUM_PAIRS, PAIR_INDEX, PAIRS, RC_TYPES, TERRAINS, load_tables,
    mask_exits, path_mask, rotate_endpoint,
)

T = load_tables()
HEXES, TILES, RCS = T['hexes'], T['tiles'], T['revenue_centers']
HEX = {h['id']: i for i, h in enumerate(HEXES)}
TILE = {t['name']: i for i, t in enumerate(TILES) if t['hex'] < 0}
SUPPLY = [TILES[i] for i in TILE.values()]
NONE, CITY, TOWN, OFFBOARD = range(len(RC_TYPES))
MOUNTAIN, WATER = (1 << TERRAINS.index(name) for name in ('mountain', 'water'))
SIDE_OFFSETS = ((0, 2), (-1, 1), (-1, -1), (0, -2), (1, -1), (1, 1))
UPGRADES = {tuple(row['from']): {tuple(t) for t in row['to']} for row in T['upgrades']}


def printed(hex_id):
    return TILES[HEXES[HEX[hex_id]]['tile']]


def ids(hex_indices):
    return {HEXES[h]['id'] for h in hex_indices}


def names(tiles):
    return {TILES[t]['name'] for t, _ in tiles}


def rotate_mask(mask, k):
    return sum(1 << PAIR_INDEX[rotate_endpoint(a, k)][rotate_endpoint(b, k)]
               for i, (a, b) in enumerate(PAIRS) if mask >> i & 1)


def legal_on(h, old, rotation=0):
    """Targets for old tile on hex h, with exits filtered by the hex's open sides."""
    return {(t, r) for t, r in UPGRADES.get((old, rotation), ())
            if mask_exits(TILES[t]['masks'][r]) & ~HEXES[h]['exit_mask'] == 0}


def test_enums_match_data_module():
    assert T['enums'] == {'colors': list(COLORS), 'rc_types': list(RC_TYPES), 'labels': list(LABELS),
                          'terrains': list(TERRAINS), 'market_types': list(MARKET_TYPES)}


def test_hexes_and_revenue_centers():
    assert len(HEXES) == 52
    assert Counter(COLORS[printed(h['id'])['color']] for h in HEXES) == {
        'white': 42, 'yellow': 2, 'green': 1, 'gray': 4, 'red': 3}
    assert len(RCS) == 25
    assert Counter(RC_TYPES[rc['type']] for rc in RCS) == {'city': 15, 'town': 7, 'offboard': 3}
    assert ids(rc['hex'] for rc in RCS if rc['port']) == {'B11', 'G10', 'I12', 'J9'}
    assert all(rc['type'] == TOWN for rc in RCS if rc['port'])
    assert ids(rc['hex'] for rc in RCS if rc['type'] == OFFBOARD) == {'F1', 'J1', 'L7'}
    rc_hexes = [rc['hex'] for rc in RCS]
    assert rc_hexes == sorted(set(rc_hexes))
    for i, h in enumerate(HEXES):
        rc_type = TILES[h['tile']]['rc']
        assert h['rc'] == (rc_hexes.index(i) if rc_type else -1)
        assert rc_type == NONE or RCS[h['rc']]['type'] == rc_type


def test_at_most_one_revenue_center_per_hex_and_tile():
    # The generator rejects Ruby tiles with more than one revenue center. Here:
    # paths reach the single node (endpoint 6) exactly when the tile has a
    # revenue center and track.
    for t in TILES:
        endpoints = {e for path in t['paths'] for e in path}
        assert endpoints <= set(range(NODE + 1))
        assert (NODE in endpoints) == (t['rc'] != NONE and bool(t['paths'])), t['name']
        assert (t['slots'] > 0) == (t['rc'] == CITY), t['name']


def test_offboards_and_printed_revenue():
    offboards = {HEXES[rc['hex']]['id']: (rc['revenue'], rc['d_revenue']) for rc in RCS if rc['type'] == OFFBOARD}
    assert offboards == {
        'F1': ([30, 30, 30, 60, 60, 60], 100),
        'J1': ([20, 20, 20, 40, 40, 40], 80),
        'L7': ([20, 20, 20, 40, 40, 40], 80)}
    assert all(t['revenue'] == 0 for t in TILES if t['rc'] == OFFBOARD)
    # Printed cities without a tile count as one slot and no revenue.
    for h in HEXES:
        t = TILES[h['tile']]
        if t['color'] == COLORS.index('white') and t['rc'] == CITY:
            assert (t['revenue'], t['slots']) == (0, 1)


def test_terrain_and_costs():
    by_terrain = defaultdict(set)
    for h in HEXES:
        t = printed(h['id'])
        if t['cost']:
            by_terrain[t['terrain']].add(h['id'])
            assert t['cost'] == 80
        else:
            assert t['terrain'] == 0
    assert len(by_terrain[MOUNTAIN]) == 14
    assert by_terrain[WATER] == {'K6'}
    assert by_terrain[WATER | MOUNTAIN] == {'H5', 'I6'}
    assert by_terrain[0] == {'I4', 'F9'}  # Kotohira and Kouchi: cost without terrain
    assert {t['name']: (t['cost'], t['terrain']) for t in SUPPLY if t['cost']} == {'438': (80, 0), '439': (80, 0)}


def test_labels():
    assert {h['id']: LABELS[printed(h['id'])['label']] for h in HEXES if printed(h['id'])['label']} == {
        'I4': 'H', 'K4': 'T', 'F9': 'K'}
    assert {t['name']: LABELS[t['label']] for t in SUPPLY if t['label']} == {
        '438': 'H', '439': 'H', '492': 'H', '440': 'T', '466': 'T', '465': 'K'}


def test_preprinted_track():
    def summary(hex_id):
        t = printed(hex_id)
        return COLORS[t['color']], RC_TYPES[t['rc']], t['revenue'], t['slots'], LABELS[t['label']], t['paths']

    assert summary('C4') == ('yellow', 'city', 20, 1, '', [[2, NODE]])
    assert HEXES[HEX['C4']]['neighbors'][2] == HEX['B3']  # toward Yawatahama
    assert summary('K4') == ('yellow', 'city', 30, 1, 'T', [[0, NODE], [1, NODE], [2, NODE]])
    assert summary('F9') == ('green', 'city', 30, 2, 'K', [[2, NODE], [3, NODE], [4, NODE], [5, NODE]])
    assert summary('B3') == ('gray', 'town', 20, 0, '', [[0, NODE], [5, NODE]])
    assert summary('B7') == ('gray', 'city', 40, 2, '', [[1, NODE], [3, NODE], [5, NODE]])
    assert summary('G14') == ('gray', 'town', 20, 0, '', [[3, NODE], [4, NODE]])
    assert summary('J7') == ('gray', 'none', 0, 0, '', [[1, 5]])


def test_tile_supply():
    assert len(SUPPLY) == 40
    assert sum(t['count'] for t in SUPPLY) == 63
    assert sum(t['count'] == 1 for t in SUPPLY) == 27
    assert all(t['count'] == 0 for t in TILES if t['hex'] >= 0)
    assert [t['hex'] for t in TILES[40:]] == list(range(52))
    assert [h['tile'] for h in HEXES] == list(range(40, 92))
    assert [t['name'] for t in SUPPLY if t['blocks_lay']] == ['437']
    # Families from 1889-rules.md 2.7: (color, revenue center, revenue, slots, label).
    families = {
        ('yellow', 'none', 0, 0, ''): {'7', '8', '9'},
        ('green', 'none', 0, 0, ''): {'16', '19', '20', '23', '24', '25', '26', '27', '28', '29'},
        ('brown', 'none', 0, 0, ''): {'39', '40', '41', '42', '45', '46', '47'},
        ('yellow', 'town', 10, 0, ''): {'3', '58'},
        ('yellow', 'town', 30, 0, ''): {'437'},
        ('yellow', 'city', 20, 1, ''): {'5', '6', '57'},
        ('green', 'city', 30, 1, ''): {'12', '13', '205', '206'},
        ('green', 'city', 30, 2, ''): {'14', '15'},
        ('brown', 'city', 40, 2, ''): {'448', '611'},
        ('yellow', 'city', 40, 1, 'H'): {'438'},
        ('green', 'city', 60, 2, 'H'): {'439'},
        ('brown', 'city', 80, 3, 'H'): {'492'},
        ('green', 'city', 40, 2, 'T'): {'440'},
        ('brown', 'city', 60, 2, 'T'): {'466'},
        ('brown', 'city', 60, 3, 'K'): {'465'},
    }
    ours = defaultdict(set)
    for t in SUPPLY:
        ours[COLORS[t['color']], RC_TYPES[t['rc']], t['revenue'], t['slots'], LABELS[t['label']]].add(t['name'])
    assert ours == families


def test_pairs():
    assert NUM_PAIRS == len(PAIRS) == 21
    for a in range(NODE + 1):
        assert PAIR_INDEX[a][a] == -1
        for b in range(NODE + 1):
            if a != b:
                assert PAIR_INDEX[a][b] == PAIR_INDEX[b][a]
                assert PAIRS[PAIR_INDEX[a][b]] == (min(a, b), max(a, b))


def test_paths_masks_and_rotations():
    for t in TILES:
        paths = t['paths']
        assert all(a < b for a, b in paths) and paths == sorted(paths) and len(set(map(tuple, paths))) == len(paths)
        for r in range(6):
            assert t['masks'][r] == path_mask(paths, r)
            assert t['masks'][r] < 1 << NUM_PAIRS
            assert bin(t['masks'][r]).count('1') == len(paths)
            for k in range(6):
                assert rotate_mask(t['masks'][r], k) == t['masks'][(r + k) % 6]


def test_neighbors_and_open_sides():
    at = {(h['x'], h['y']): i for i, h in enumerate(HEXES)}
    for i, h in enumerate(HEXES):
        for s, (dx, dy) in enumerate(SIDE_OFFSETS):
            n = h['neighbors'][s]
            assert n == at.get((h['x'] + dx, h['y'] + dy), -1)
            open_side = bool(h['exit_mask'] >> s & 1)
            if n < 0:
                assert not open_side
                continue
            assert HEXES[n]['neighbors'][(s + 3) % 6] == i
            # Gray and red hexes close the sides they have no track on.
            other = TILES[HEXES[n]['tile']]
            fixed = COLORS[other['color']] in ('gray', 'red')
            assert open_side == (not fixed or bool(mask_exits(other['masks'][0]) >> (s + 3) % 6 & 1))
        # Preprinted track never leaves through a closed side.
        assert mask_exits(TILES[h['tile']]['masks'][0]) & ~h['exit_mask'] == 0


def test_upgrade_table():
    froms = [tuple(row['from']) for row in T['upgrades']]
    assert froms == sorted(set(froms))
    for (old, rotation), targets in UPGRADES.items():
        o = TILES[old]
        assert o['hex'] < 0 or rotation == 0
        for new, new_rotation in targets:
            n = TILES[new]
            assert n['hex'] < 0 and n['color'] == o['color'] + 1
            assert (n['label'], n['rc']) == (o['label'], o['rc'])
            assert o['masks'][rotation] & ~n['masks'][new_rotation] == 0
            assert n['slots'] >= o['slots']  # tokens always fit, so the table needs no slot check
        if o['hex'] < 0:
            for k in range(6):
                assert UPGRADES[old, (rotation + k) % 6] == {(t, (r + k) % 6) for t, r in targets}
    assert all(COLORS[TILES[old]['color']] in ('white', 'yellow', 'green') for old, _ in UPGRADES)
    # Every white, yellow, or green printed hex can be upgraded on its own hex.
    for i, h in enumerate(HEXES):
        upgradable = COLORS[TILES[h['tile']]['color']] in ('white', 'yellow', 'green')
        assert bool(legal_on(i, h['tile'])) == upgradable, h['id']


def test_named_upgrades():
    def on(hex_id):
        return legal_on(HEX[hex_id], HEXES[HEX[hex_id]]['tile'])

    assert on('K4') == {(TILE['440'], 0)}
    assert on('F9') == {(TILE['465'], 2)}
    assert names(on('I4')) == {'438'}
    assert names(on('C4')) == {'12', '13', '14', '15', '205', '206'}
    for h in HEXES:
        t = TILES[h['tile']]
        if COLORS[t['color']] == 'white' and not t['label']:
            assert names(UPGRADES[h['tile'], 0]) == {
                NONE: {'7', '8', '9'}, CITY: {'5', '6', '57'}, TOWN: {'3', '58', '437'}}[t['rc']], h['id']
    assert names(legal_on(HEX['A8'], HEXES[HEX['A8']]['tile'])) == {'7', '8'}  # open sides 0, 4, 5: no straight
    assert not any(TILES[old]['rc'] == TOWN and TILES[old]['hex'] < 0 for old, _ in UPGRADES)  # towns never upgrade
    # Upgrades the rulebook manifest omits (1889-rules.md 12.3).
    for old, new in (('28', '39'), ('29', '39'), ('57', '205'), ('57', '206')):
        assert TILE[new] in {t for r in range(6) for t, _ in UPGRADES.get((TILE[old], r), ())}


def test_corporations():
    corps = T['corporations']
    assert {c['sym']: (HEXES[c['home']]['id'], c['tokens']) for c in corps} == {
        'AR': ('K8', 2), 'IR': ('E2', 2), 'SR': ('I2', 2), 'KO': ('K4', 2),
        'TR': ('F9', 3), 'KU': ('C10', 1), 'UR': ('B7', 3)}
    for i, c in enumerate(corps):
        assert c['token_prices'] == [0] + [40] * (c['tokens'] - 1)
        assert (c['float_percent'], c['max_ownership_percent']) == (50, 60)
        assert TILES[HEXES[c['home']]['tile']]['rc'] == CITY
        assert HEXES[c['home']]['reserved'] == [i]
    assert sum(len(h['reserved']) for h in HEXES) == len(corps)


def test_privates():
    privates = T['privates']
    assert [(p['sym'], p['value'], p['revenue']) for p in privates] == [
        ('TR', 20, 5), ('MF', 30, 5), ('ER', 40, 10), ('SMR', 50, 15), ('DR', 60, 15), ('SIR', 80, 20),
        ('UTF', 150, 30)]
    abilities = {p['sym']: {a['type']: a for a in p['abilities']} for p in privates}
    assert ids(abilities['TR']['blocks_hexes']['hexes']) == {'K4'}
    assert ids(abilities['ER']['blocks_hexes']['hexes']) == {'C4'}
    assert {h['id']: [privates[p]['sym'] for p in h['blockers']] for h in HEXES if h['blockers']} == {
        'K4': ['TR'], 'C4': ['ER']}
    mf = abilities['MF']['tile_lay']
    assert (ids(mf['hexes']), names((t, 0) for t in mf['tiles']), mf['owner_type'], mf['count']) == (
        {'B11', 'G10', 'I12', 'J9'}, {'437'}, 'player', 1)
    er = abilities['ER']['tile_lay']
    unlabeled_green_cities = {t['name'] for t in SUPPLY
                              if COLORS[t['color']] == 'green' and t['rc'] == CITY and not t['label']}
    assert (ids(er['hexes']), names((t, 0) for t in er['tiles']), er['when'], er['count']) == (
        {'C4'}, unlabeled_green_cities, 'sold', 1)
    smr = abilities['SMR']['tile_discount']
    assert (smr['discount'], smr['terrain'], smr['owner_type']) == (80, 'mountain', 'corporation')
    dr = abilities['DR']['exchange']
    assert ([T['corporations'][c]['sym'] for c in dr['corporations']], dr['from'], dr['owner_type']) == (
        ['IR'], 'ipo', 'player')
    assert abilities['SIR'] == {}
    assert abilities['UTF']['close']['on_phase'] == 'never'
    assert (abilities['UTF']['revenue_change']['revenue'], abilities['UTF']['revenue_change']['on_phase']) == (50, '5')


def test_players():
    # Private letters A-G are the privates in face-value order (test_privates).
    assert {p['players']: (p['starting_cash'], p['cert_limit'], ''.join('ABCDEFG'[i] for i in p['privates']))
            for p in T['players']} == {
        2: (420, 25, 'ABCDE'), 3: (420, 19, 'ABCDEF'), 4: (420, 14, 'ABCDEFG'), 5: (390, 12, 'ABCDEFG'),
        6: (390, 11, 'ABCDEFG')}
    assert T['rules']['BANK_CASH'] == 7000


def test_trains_and_phases():
    trains, phases = T['trains'], T['phases']
    names_ = [t['name'] for t in trains]
    assert [(t['name'], t['count'], t['price'], names_[t['rusts_on']] if t['rusts_on'] >= 0 else None)
            for t in trains] == [
        ('2', 6, 80, '4'), ('3', 5, 180, '6'), ('4', 4, 300, 'D'), ('5', 3, 450, None), ('6', 2, 630, None),
        ('D', -1, 1100, None)]
    d = trains[names_.index('D')]
    assert phases[d['available_on']]['name'] == '6'
    assert dict(zip(names_, d['trade_in'])) == {'2': 0, '3': 0, '4': 300, '5': 300, '6': 300, 'D': 0}
    assert [t['name'] for t in trains if 'close_companies' in t['events']] == ['5']
    assert [p['name'] for p in phases] == names_
    assert [p['on'] for p in phases] == [-1, 1, 2, 3, 4, 5]
    assert [p['train_limit'] for p in phases] == [4, 4, 3, 2, 2, 2]
    assert [p['operating_rounds'] for p in phases] == [1, 2, 2, 3, 3, 3]
    assert [[COLORS[c] for c in p['tiles']] for p in phases] == (
        [['yellow']] + [['yellow', 'green']] * 2 + [['yellow', 'green', 'brown']] * 3)
    assert [p['name'] for p in phases if 'can_buy_companies' in p['status']] == ['3', '4']


def test_market():
    prices, types = T['market_prices'], T['market_types']
    assert len(prices) == 11 and [len(row) for row in types] == [len(row) for row in prices]
    assert [len(row) for row in prices] == sorted((len(row) for row in prices), reverse=True)
    for r, row in enumerate(prices):
        assert row == sorted(set(row))
        if r:
            assert all(p < above for p, above in zip(row, prices[r - 1]))
    assert (prices[0][-1], prices[-1][0]) == (350, 10)
    cells = {MARKET_TYPES[t]: [] for t in range(len(MARKET_TYPES))}
    for r, row in enumerate(types):
        for c, t in enumerate(row):
            cells[MARKET_TYPES[t]].append((r, c, prices[r][c]))
    assert cells['par'] == [(r, 3, p) for r, p in enumerate([100, 90, 80, 75, 70, 65])]
    assert {p for _, _, p in cells['no_cert_limit']} == {40, 45, 50}
    assert {p for _, _, p in cells['unlimited']} == {10, 20, 30}
