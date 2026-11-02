"""g1889.constants against 1889/engine-design.md sections 5.2 and 6.1."""

import g1889.constants as K
import g1889.data.tables as D

# Section 5.2, in decision-phase order.
SIZES = {
    'AUCTION': 9, 'BID': 17, 'STOCK': 31, 'PAR': 6, 'SELL_AMOUNT': 5, 'TRACK': 62, 'TILE': 15, 'TOKEN': 61,
    'DIVIDEND': 10, 'DISCARD': 6, 'BUY_TRAIN': 26, 'SELECT_TRAIN': 9, 'TRAIN_PRICE': 5, 'EMERGENCY_SELL_AMOUNT': 5,
    'BUY_PRIVATE': 9, 'PRIVATE_PRICE': 4, 'ER_BONUS': 16, 'PORT': 24,
}
OFFSETS = {
    'AUCTION_PASS': 0, 'AUCTION_BUY': 1, 'AUCTION_BID': 2,
    'BID_DROP': 0, 'BID_POINT': 1, 'BID_EXACT': 17,
    'STOCK_PASS': 0, 'STOCK_IPO': 1, 'STOCK_POOL': 8, 'STOCK_PAR': 15, 'STOCK_SELL': 22, 'STOCK_DR': 29,
    'STOCK_MF': 30,
    'TRACK_PASS': 0, 'TRACK_HEX': 1, 'TRACK_BUY_PRIVATE': 53, 'TRACK_DR': 60, 'TRACK_MF': 61,
    'TOKEN_PASS': 0, 'TOKEN_HEX': 1, 'TOKEN_BUY_PRIVATE': 53, 'TOKEN_DR': 60,
    'DIVIDEND_WITHHOLD': 0, 'DIVIDEND_PAYOUT': 1, 'DIVIDEND_BUY_PRIVATE': 2, 'DIVIDEND_DR': 9,
    'TRAIN_SOURCE_DEPOT': 0, 'TRAIN_SOURCE_POOL': 1, 'TRAIN_SOURCE_CORP': 2, 'NUM_TRAIN_SOURCES': 9,
    'BUY_TRAIN_PASS': 0, 'BUY_TRAIN_DEPOT': 1, 'BUY_TRAIN_POOL': 2, 'BUY_TRAIN_CORP': 3, 'BUY_TRAIN_SELL': 10,
    'BUY_TRAIN_BANKRUPT': 17, 'BUY_TRAIN_BUY_PRIVATE': 18, 'BUY_TRAIN_DR': 25,
    'SELECT_TRAIN_TYPE': 0, 'SELECT_TRAIN_TRADE_IN': 6,
    'TRAIN_PRICE_EXACT': 5,
    'BUY_PRIVATE_DONE': 0, 'BUY_PRIVATE_PRIVATE': 1, 'BUY_PRIVATE_DR': 8,
    'PRIVATE_PRICE_EXACT': 4,
    'ER_BONUS_PASS': 0, 'ER_BONUS_CANDIDATE': 1,
    'PORT_LAY': 0,
}
AUTOMATED_STEPS = {'SR_END', 'OR_START', 'TURN_START', 'ROUTE', 'TURN_END', 'OR_END', 'GAME_OVER'}


def names(prefix):
    """Int constant names with prefix, in value order."""
    return sorted((n for n in dir(K) if n.startswith(prefix) and isinstance(getattr(K, n), int)),
                  key=lambda n: getattr(K, n))


def test_phase_sizes():
    assert [n.removeprefix('DPHASE_') for n in names('DPHASE_')] == list(SIZES)
    assert K.NUM_DECISION_PHASES == len(SIZES)
    assert K.PHASE_ACTION_SIZES == list(SIZES.values())
    assert [getattr(K, f'ACTION_SIZE_{name}') for name in SIZES] == list(SIZES.values())
    assert K.UNIFIED_ACTION_DIM == sum(SIZES.values()) == 320
    assert K.MAX_ACTION_SIZE == max(SIZES.values()) == 62


def test_phase_local_offsets():
    assert {name: getattr(K, name) for name in OFFSETS} == OFFSETS


def test_phase_blocks_tile_each_phase():
    # (first id, length) of every block; together they cover [0, size) exactly.
    sale = (0, K.MAX_SALE_SHARES)
    blocks = {
        'AUCTION': [(K.AUCTION_PASS, 1), (K.AUCTION_BUY, 1), (K.AUCTION_BID, D.NUM_PRIVATES)],
        'BID': [(K.BID_DROP, 1), (K.BID_POINT, K.BID_POINTS)],
        'STOCK': [(K.STOCK_PASS, 1), (K.STOCK_IPO, D.NUM_CORPS), (K.STOCK_POOL, D.NUM_CORPS),
                  (K.STOCK_PAR, D.NUM_CORPS), (K.STOCK_SELL, D.NUM_CORPS), (K.STOCK_DR, 1), (K.STOCK_MF, 1)],
        'PAR': [(0, D.NUM_PAR_PRICES)],
        'SELL_AMOUNT': [sale],
        'TRACK': [(K.TRACK_PASS, 1), (K.TRACK_HEX, D.NUM_HEXES), (K.TRACK_BUY_PRIVATE, D.NUM_PRIVATES),
                  (K.TRACK_DR, 1), (K.TRACK_MF, 1)],
        'TILE': [(0, D.MAX_LAY_CANDIDATES)],
        'TOKEN': [(K.TOKEN_PASS, 1), (K.TOKEN_HEX, D.NUM_HEXES), (K.TOKEN_BUY_PRIVATE, D.NUM_PRIVATES),
                  (K.TOKEN_DR, 1)],
        'DIVIDEND': [(K.DIVIDEND_WITHHOLD, 1), (K.DIVIDEND_PAYOUT, 1), (K.DIVIDEND_BUY_PRIVATE, D.NUM_PRIVATES),
                     (K.DIVIDEND_DR, 1)],
        'DISCARD': [(0, D.NUM_TRAIN_TYPES)],
        'BUY_TRAIN': [(K.BUY_TRAIN_PASS, 1), (K.BUY_TRAIN_DEPOT, 1), (K.BUY_TRAIN_POOL, 1),
                      (K.BUY_TRAIN_CORP, D.NUM_CORPS), (K.BUY_TRAIN_SELL, D.NUM_CORPS), (K.BUY_TRAIN_BANKRUPT, 1),
                      (K.BUY_TRAIN_BUY_PRIVATE, D.NUM_PRIVATES), (K.BUY_TRAIN_DR, 1)],
        'SELECT_TRAIN': [(K.SELECT_TRAIN_TYPE, D.NUM_TRAIN_TYPES), (K.SELECT_TRAIN_TRADE_IN, D.NUM_TRADE_IN_TYPES)],
        'TRAIN_PRICE': [(0, K.TRAIN_PRICE_POINTS)],
        'EMERGENCY_SELL_AMOUNT': [sale],
        'BUY_PRIVATE': [(K.BUY_PRIVATE_DONE, 1), (K.BUY_PRIVATE_PRIVATE, D.NUM_PRIVATES), (K.BUY_PRIVATE_DR, 1)],
        'PRIVATE_PRICE': [(0, K.PRIVATE_PRICE_POINTS)],
        'ER_BONUS': [(K.ER_BONUS_PASS, 1), (K.ER_BONUS_CANDIDATE, D.MAX_LAY_CANDIDATES)],
        'PORT': [(K.PORT_LAY, D.NUM_PORTS * D.NUM_SIDES)],
    }
    assert list(blocks) == list(SIZES)
    for phase, spans in blocks.items():
        end = 0
        for first, length in spans:
            assert first == end, phase
            end += length
        assert end == SIZES[phase], phase
    # Exact amounts start at the phase size, outside the policy layout.
    assert (K.BID_EXACT, K.TRAIN_PRICE_EXACT, K.PRIVATE_PRICE_EXACT) == (
        SIZES['BID'], SIZES['TRAIN_PRICE'], SIZES['PRIVATE_PRICE'])
    assert K.MAX_SALE_SHARES * 10 == D.MARKET_SHARE_LIMIT
    # A source's BUY_TRAIN slot is BUY_TRAIN_DEPOT + source.
    assert (K.BUY_TRAIN_POOL, K.BUY_TRAIN_CORP, K.BUY_TRAIN_SELL) == tuple(
        K.BUY_TRAIN_DEPOT + s for s in (K.TRAIN_SOURCE_POOL, K.TRAIN_SOURCE_CORP, K.NUM_TRAIN_SOURCES))


def test_steps_map_to_decision_phases():
    steps = names('STEP_')
    assert len(steps) == K.NUM_STEPS == len(K.STEP_TO_DECISION_PHASE)
    assert [getattr(K, n) for n in steps] == list(range(K.NUM_STEPS))
    assert all(-1 <= p < K.NUM_DECISION_PHASES for p in K.STEP_TO_DECISION_PHASE)
    for name in steps:
        phase = K.STEP_TO_DECISION_PHASE[getattr(K, name)]
        step = name.removeprefix('STEP_')
        assert phase == (-1 if step in AUTOMATED_STEPS else getattr(K, f'DPHASE_{step}')), name
    # Every decision phase is reached by exactly one step.
    assert sorted(p for p in K.STEP_TO_DECISION_PHASE if p >= 0) == list(range(K.NUM_DECISION_PHASES))


def test_rounds_owners_and_pass_slots():
    assert [n.removeprefix('ROUND_') for n in names('ROUND_')] == ['AUCTION', 'STOCK', 'OPERATING', 'GAME_OVER']
    assert (K.OWNER_NOT_IN_PLAY, K.OWNER_CLOSED, K.OWNER_UNSOLD, K.OWNER_PLAYER, K.OWNER_CORP) == (-3, -2, -1, 0, 8)
    assert K.OWNER_PLAYER + D.MAX_PLAYERS <= K.OWNER_CORP
    assert (K.END_REASON_NONE, K.END_REASON_BANKRUPT, K.END_REASON_BANK, K.NUM_END_REASONS) == (0, 1, 2, 3)
    assert K.PHASES_WITH_PASS_SLOT == tuple(getattr(K, f'DPHASE_{name}') for name in (
        'AUCTION', 'BID', 'STOCK', 'TRACK', 'TOKEN', 'BUY_TRAIN', 'BUY_PRIVATE', 'ER_BONUS'))
