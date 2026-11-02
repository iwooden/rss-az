"""Stage 2 codegen (tables.json -> tables.pxd/.pyx) and the compiled
g1889.data.tables mirrors against load_tables(). Runs without Ruby."""

import g1889.data.tables as C
from g1889.data import PAIRS, load_tables, mask_exits
from g1889.data import generate

T = load_tables()
HEXES, RCS, TILES = T['hexes'], T['revenue_centers'], T['tiles']
CORPS, PRIVATES, TRAINS, PHASES = T['corporations'], T['privates'], T['trains'], T['phases']


def column(rows, key):
    return [int(row[key]) for row in rows]


def targets(tile, rotation):
    k = tile * C.NUM_SIDES + rotation
    return [(C.UPG_TILE[i], C.UPG_ROT[i]) for i in range(C.UPG_START[k], C.UPG_START[k + 1])]


def test_cython_regenerates_from_json():
    assert generate.render_cython(T) == (generate.PXD_PATH.read_text(), generate.PYX_PATH.read_text())


def test_dimensions():
    # engine-design.md section 2, checked against the JSON.
    assert {name: getattr(C, name) for name in (
        'NUM_HEXES', 'NUM_RCS', 'NUM_SUPPLY_TILES', 'NUM_TILES', 'NUM_CORPS', 'NUM_PRIVATES', 'NUM_TRAIN_TYPES',
        'NUM_GAME_PHASES', 'MARKET_ROWS', 'MARKET_COLS', 'NUM_PAR_PRICES', 'NUM_PAIRS', 'NODE', 'NUM_PORTS',
        'MAX_CITY_SLOTS', 'MAX_LAY_CANDIDATES', 'MAX_ROUTE_HEXES')} == {
        'NUM_HEXES': 52, 'NUM_RCS': 25, 'NUM_SUPPLY_TILES': 40, 'NUM_TILES': 92, 'NUM_CORPS': 7, 'NUM_PRIVATES': 7,
        'NUM_TRAIN_TYPES': 6, 'NUM_GAME_PHASES': 6, 'MARKET_ROWS': 11, 'MARKET_COLS': 15, 'NUM_PAR_PRICES': 6,
        'NUM_PAIRS': 21, 'NODE': 6, 'NUM_PORTS': 4, 'MAX_CITY_SLOTS': 3, 'MAX_LAY_CANDIDATES': 15,
        'MAX_ROUTE_HEXES': 104}
    assert (C.NUM_HEXES, C.NUM_RCS, C.NUM_TILES, C.NUM_CORPS, C.NUM_PRIVATES, C.NUM_TRAIN_TYPES) == (
        len(HEXES), len(RCS), len(TILES), len(CORPS), len(PRIVATES), len(TRAINS))
    assert C.NUM_UPGRADE_TARGETS == C.UPG_START[-1] == len(C.UPG_TILE)


def test_map_and_tile_mirrors_equal_json():
    assert C.HEX_IDS == tuple(h['id'] for h in HEXES)
    assert C.TILE_NAMES == tuple(t['name'] for t in TILES)
    assert C.HEX_NEIGHBOR == [h['neighbors'] for h in HEXES]
    assert C.HEX_EXIT_MASK == column(HEXES, 'exit_mask')
    assert C.HEX_RC == column(HEXES, 'rc')
    assert C.HEX_HOME_CORP == [h['reserved'][0] if h['reserved'] else -1 for h in HEXES]
    assert C.HEX_BLOCKER == [h['blockers'][0] if h['blockers'] else -1 for h in HEXES]
    for name, key in (('RC_HEX', 'hex'), ('RC_TYPE', 'type'), ('RC_PORT', 'port'), ('RC_D_REVENUE', 'd_revenue')):
        assert getattr(C, name) == column(RCS, key), name
    assert C.RC_REVENUE == [rc['revenue'] for rc in RCS]
    for name in ('COLOR', 'LABEL', 'RC', 'SLOTS', 'TERRAIN', 'BLOCKS_LAY', 'COUNT', 'REVENUE', 'COST'):
        assert getattr(C, f'TILE_{name}') == column(TILES, name.lower()), name
    assert C.TILE_MASK == [t['masks'] for t in TILES]
    assert C.PAIR_A == [a for a, _ in PAIRS] and C.PAIR_B == [b for _, b in PAIRS]


def test_derived_tile_tables():
    for t in range(C.NUM_TILES):
        masks = C.TILE_MASK[t]
        assert C.TILE_EXITS[t] == [mask_exits(m) for m in masks]
        assert C.TILE_CANON_ROT[t] == [masks.index(m) for m in masks]


def test_upgrade_csr_holds_canonical_json_targets():
    rows = {tuple(row['from']): row['to'] for row in T['upgrades']}
    assert len(C.UPG_START) == C.NUM_TILES * C.NUM_SIDES + 1
    for t in range(C.NUM_TILES):
        for r in range(C.NUM_SIDES):
            ours = targets(t, r)
            assert ours == sorted(set(ours))
            assert all(C.TILE_CANON_ROT[n][nr] == nr for n, nr in ours)
            assert set(ours) == {(n, C.TILE_CANON_ROT[n][nr]) for n, nr in rows.get((t, r), [])}


def test_max_lay_candidates():
    def candidates(h, tile, rotation):
        return [(n, r) for n, r in targets(tile, rotation)
                if not C.TILE_BLOCKS_LAY[n] and C.TILE_EXITS[n][r] & ~C.HEX_EXIT_MASK[h] == 0]

    counts = [len(candidates(h, old, r)) for h in range(C.NUM_HEXES)
              for old, r in [(C.NUM_SUPPLY_TILES + h, 0)]
              + [(t, r) for t in range(C.NUM_SUPPLY_TILES) for r in range(C.NUM_SIDES)]]
    assert max(counts) == C.MAX_LAY_CANDIDATES
    # ER's bonus lays on C4 fit ER_BONUS's candidate slots.
    er = candidates(C.ER_HEX, C.NUM_SUPPLY_TILES + C.ER_HEX, 0)
    assert {n for n, _ in er} == set(C.ER_TILE) and len(er) <= C.MAX_LAY_CANDIDATES


def test_game_data_mirrors_equal_json():
    rules = T['rules']
    assert (C.BANK_CASH, C.MIN_BID_INCREMENT, C.MARKET_SHARE_LIMIT) == (
        rules['BANK_CASH'], rules['MIN_BID_INCREMENT'], rules['MARKET_SHARE_LIMIT'])
    assert (C.MAX_OWNERSHIP_PERCENT, C.FLOAT_PERCENT, C.SMR_DISCOUNT, C.UTF_LATE_REVENUE) == (60, 50, 80, 50)
    assert C.UTF_LATE_PHASE == C.CLOSE_PRIVATES_PHASE == C.GAME_PHASE_5
    assert (C.HEX_IDS[C.ER_HEX], C.TILE_NAMES[C.PORT_TILE]) == ('C4', '437')
    assert (C.CORP_SYMS[C.DR_EXCHANGE_CORP], C.TRAIN_NAMES[C.D_TRAIN]) == ('IR', 'D')
    assert C.CORP_SYMS == tuple(c['sym'] for c in CORPS) and C.PRIVATE_SYMS == tuple(p['sym'] for p in PRIVATES)
    assert C.CORP_HOME == column(CORPS, 'home') and C.CORP_TOKENS == column(CORPS, 'tokens')
    assert C.CORP_TOKEN_PRICE == [c['token_prices'] + [0] * (3 - c['tokens']) for c in CORPS]
    assert C.PRIVATE_VALUE == column(PRIVATES, 'value') and C.PRIVATE_REVENUE == column(PRIVATES, 'revenue')
    abilities = {(p['sym'], a['type']): a for p in PRIVATES for a in p['abilities']}
    assert C.MF_PORT_HEX == abilities['MF', 'tile_lay']['hexes']
    assert C.ER_TILE == abilities['ER', 'tile_lay']['tiles']
    for p in T['players']:
        n = p['players']
        assert (C.START_CASH[n], C.CERT_LIMIT[n]) == (p['starting_cash'], p['cert_limit'])
        assert C.PRIVATES_IN_PLAY[n] == sum(1 << i for i in p['privates'])
    for name in ('DISTANCE', 'PRICE', 'COUNT', 'RUSTS_ON', 'AVAILABLE_ON'):
        assert getattr(C, f'TRAIN_{name}') == column(TRAINS, name.lower()), name
    assert C.TRAIN_TRADE_IN == [t['trade_in'] for t in TRAINS]
    assert [C.TRAIN_NAMES[t] for t in C.TRADE_IN_TYPE] == ['4', '5', '6']
    assert C.PHASE_ON_TRAIN == column(PHASES, 'on') and C.PHASE_ORS == column(PHASES, 'operating_rounds')
    assert C.PHASE_TRAIN_LIMIT == column(PHASES, 'train_limit')
    assert C.PHASE_TILE_COLORS == [sum(1 << c for c in p['tiles']) for p in PHASES]
    assert C.PHASE_CAN_BUY_PRIVATES == [int('can_buy_companies' in p['status']) for p in PHASES]
    prices = T['market_prices']
    assert C.MARKET_ROW_LEN == [len(row) for row in prices]
    assert [row[:n] for row, n in zip(C.MARKET_PRICE, C.MARKET_ROW_LEN)] == prices
    assert [row[:n] for row, n in zip(C.MARKET_TYPE, C.MARKET_ROW_LEN)] == T['market_types']
    assert all(p == 0 for row, n in zip(C.MARKET_PRICE, C.MARKET_ROW_LEN) for p in row[n:])
    assert C.PAR_PRICE == [65, 70, 75, 80, 90, 100]
    assert all(C.MARKET_TYPE[r][c] == C.MARKET_PAR and C.MARKET_PRICE[r][c] == p
               for r, c, p in zip(C.PAR_ROW, C.PAR_COL, C.PAR_PRICE))
