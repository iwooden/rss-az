"""g1889.entities: cash transfers, bids, private ownership and revenue, the
turn fields, and the raw accessors' cache invalidation (1889/engine-design.md §4)."""

import pytest

import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.bank import BANK, BANK_ACCOUNT
from g1889.entities.corp import CORPS, CORPS_BY_SYM
from g1889.entities.hexmap import MAP
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES, PRIVATES_BY_SYM
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.state import GameState

ALL_CORPS = (1 << D.NUM_CORPS) - 1


@pytest.fixture
def state():
    s = GameState(4, max_players=5)
    s.initialize_game(4, seed=1, priority_player=0)
    return s


def corp_account(corp):
    return K.OWNER_CORP + corp


# BANK


def test_transfers_conserve_cash(state):
    BANK.transfer(state, 0, corp_account(D.CORP_IR), 300)
    BANK.transfer(state, corp_account(D.CORP_IR), BANK_ACCOUNT, 80)
    BANK.transfer(state, BANK_ACCOUNT, 3, 15)
    BANK.transfer(state, 1, 2, 0)
    assert PLAYERS[0].cash(state) == 120 and CORPS[D.CORP_IR].cash(state) == 220
    assert PLAYERS[3].cash(state) == 435 and BANK.cash(state) == 7000 - 4 * 420 + 80 - 15
    assert BANK.cash(state) + sum(PLAYERS[i].cash(state) for i in range(4)) + sum(
        c.cash(state) for c in CORPS) == D.BANK_CASH


@pytest.mark.parametrize('src, dst, amount', [
    (0, 1, 421),                              # players cannot overdraw
    (corp_account(0), 1, 1),                  # nor corps
    (0, 0, 5),                                # to itself
    (0, 1, -5),                               # negative
    (4, 0, 5),                                # a padded seat
    (0, corp_account(D.NUM_CORPS), 5),        # no such corp
    (0, 7, 5),                                # between the player and corp codes
])
def test_bad_transfers_raise(state, src, dst, amount):
    with pytest.raises(AssertionError):
        BANK.transfer(state, src, dst, amount)


def test_bank_breaks_below_zero_and_stays_broken(state):
    BANK.transfer(state, BANK_ACCOUNT, corp_account(0), BANK.cash(state))
    assert BANK.cash(state) == 0 and not BANK.broken(state)
    BANK.transfer(state, BANK_ACCOUNT, corp_account(1), 30)
    assert BANK.cash(state) == -30 and BANK.broken(state)
    BANK.transfer(state, corp_account(1), BANK_ACCOUNT, 30)
    assert BANK.cash(state) == 0 and BANK.broken(state)
    with pytest.raises(AssertionError):  # int16 bound
        BANK.transfer(state, BANK_ACCOUNT, corp_account(2), 32769)


# PLAYERS


def test_bids_and_committed_cash(state):
    player = PLAYERS[2]
    player.set_bid(state, D.PRIVATE_ER, 45)
    player.set_bid(state, D.PRIVATE_UTF, 155)
    assert (player.bid(state, D.PRIVATE_ER), player.committed(state)) == (45, 200)
    assert PLAYERS[1].committed(state) == 0
    player.clear_bids(state)
    assert player.committed(state) == 0
    with pytest.raises(AssertionError):
        PLAYERS[4].set_bid(state, D.PRIVATE_ER, 45)  # padded seat
    with pytest.raises(AssertionError):
        player.set_bid(state, D.NUM_PRIVATES, 45)


def test_player_flags_and_shares(state):
    player = PLAYERS[1]
    player.set_passed(state, True)
    player.set_sold_mask(state, 0b101)
    player.set_shares(state, D.CORP_KO, 2)
    player.set_bankrupt(state, True)
    assert (player.passed(state), player.sold_mask(state), player.shares(state, D.CORP_KO),
            player.bankrupt(state)) == (True, 0b101, 2, True)
    assert not PLAYERS[0].passed(state)
    with pytest.raises(AssertionError):
        player.set_shares(state, D.CORP_KO, -1)


# PRIVATES, CORPS


def test_private_ownership(state):
    er = PRIVATES_BY_SYM['ER']
    for owner in (K.OWNER_PLAYER + 3, K.OWNER_CORP + D.CORP_UR, K.OWNER_CLOSED, K.OWNER_UNSOLD):
        er.set_owner(state, owner)
        assert er.owner(state) == owner
    for owner in (K.OWNER_PLAYER + 4, K.OWNER_NOT_IN_PLAY - 1, K.OWNER_CORP + D.NUM_CORPS, 6):
        with pytest.raises(AssertionError):
            er.set_owner(state, owner)
    PRIVATES[D.PRIVATE_MF].set_flags(state, 1)
    assert PRIVATES[D.PRIVATE_MF].flags(state) == 1


def test_private_revenue_and_late_utf(state):
    assert [p.revenue(state) for p in PRIVATES] == D.PRIVATE_REVENUE
    utf = PRIVATES[D.PRIVATE_UTF]
    utf.set_owner(state, K.OWNER_PLAYER + 1)
    PHASE.set_current(state, D.GAME_PHASE_4)
    assert utf.revenue(state) == D.PRIVATE_REVENUE[D.PRIVATE_UTF]
    PHASE.set_current(state, D.GAME_PHASE_5)
    assert utf.revenue(state) == D.UTF_LATE_REVENUE == 50
    utf.set_owner(state, K.OWNER_CORP + D.CORP_AR)
    assert utf.revenue(state) == D.PRIVATE_REVENUE[D.PRIVATE_UTF]


def test_symbols_follow_the_tables():
    assert [c.sym for c in CORPS] == ['AR', 'IR', 'SR', 'KO', 'TR', 'KU', 'UR']
    assert [p.sym for p in PRIVATES] == ['TR', 'MF', 'ER', 'SMR', 'DR', 'SIR', 'UTF']
    assert CORPS_BY_SYM['KU'].corp_id == D.CORP_KU


def test_corp_accessors(state):
    corp = CORPS[D.CORP_TR]
    corp.set_par_index(state, 2)
    corp.set_floated(state, True)
    corp.set_ipo_shares(state, 4)
    corp.set_pool_shares(state, 1)
    corp.set_president(state, 3)
    corp.set_tokens_left(state, 1)
    corp.set_last_revenue(state, 120)
    assert (corp.par_index(state), corp.floated(state), corp.ipo_shares(state), corp.pool_shares(state),
            corp.president(state), corp.tokens_left(state), corp.last_revenue(state)) == (2, True, 4, 1, 3, 1, 120)
    for setter, value in ((corp.set_par_index, D.NUM_PAR_PRICES), (corp.set_ipo_shares, 11),
                          (corp.set_president, 4), (corp.set_tokens_left, D.CORP_TOKENS[D.CORP_TR] + 1)):
        with pytest.raises(AssertionError):
            setter(state, value)


# MARKET, MAP, TRAINS, PHASE


def test_market_positions_stack_by_arrival(state):
    row, col = D.PAR_ROW[0], D.PAR_COL[0]
    MARKET.set_position(state, D.CORP_SR, row, col)
    MARKET.set_position(state, D.CORP_AR, row, col)
    assert MARKET.price(state, D.CORP_SR) == MARKET.price(state, D.CORP_AR) == D.PAR_PRICE[0]
    assert MARKET.stamp(state, D.CORP_SR) < MARKET.stamp(state, D.CORP_AR) < MARKET.next_stamp(state)
    for bad_row, bad_col in ((0, D.MARKET_COLS), (D.MARKET_ROWS - 1, D.MARKET_ROW_LEN[-1]), (-1, 0)):
        with pytest.raises(AssertionError):
            MARKET.set_position(state, D.CORP_SR, bad_row, bad_col)


def test_map_writes_canonicalize_and_invalidate(state):
    tile = next(t for t in range(D.NUM_SUPPLY_TILES) if D.TILE_CANON_ROT[t][3] == 0)
    MAP.set_tile(state, 10, tile, 3)
    assert (MAP.tile(state, 10), MAP.rotation(state, 10)) == (tile, 0)
    assert MAP.conn_dirty(state) == ALL_CORPS and all(c.route_dirty(state) for c in CORPS)
    MAP.set_tokens(state, 10, 0b11)
    assert MAP.tokens(state, 10) == 0b11
    MAP.set_supply(state, tile, 0)
    assert MAP.supply(state, tile) == 0
    MAP.set_tile(state, 10, D.NUM_SUPPLY_TILES + 10, 0)  # its printed tile
    with pytest.raises(AssertionError):
        MAP.set_tile(state, 10, D.NUM_SUPPLY_TILES + 11, 0)
    with pytest.raises(AssertionError):
        MAP.set_tokens(state, 10, 1 << D.NUM_CORPS)


def test_train_and_phase_writes_invalidate_routes(state):
    TRAINS.set_corp_trains(state, D.CORP_KO, D.TRAIN_3, 2)
    assert TRAINS.corp_trains(state, D.CORP_KO, D.TRAIN_3) == 2
    assert [c.route_dirty(state) for c in CORPS] == [c.corp_id == D.CORP_KO for c in CORPS]
    assert MAP.conn_dirty(state) == 0
    TRAINS.set_depot(state, D.TRAIN_2, 5)
    TRAINS.set_pool(state, D.TRAIN_2, 1)
    assert (TRAINS.depot(state, D.TRAIN_2), TRAINS.pool(state, D.TRAIN_2), TRAINS.depot(state, D.TRAIN_D)) == (5, 1, -1)
    with pytest.raises(AssertionError):
        TRAINS.set_pool(state, D.TRAIN_2, -1)
    PHASE.set_current(state, D.GAME_PHASE_3)
    assert PHASE.current(state) == D.GAME_PHASE_3 and all(c.route_dirty(state) for c in CORPS)
    with pytest.raises(AssertionError):
        PHASE.set_current(state, D.NUM_GAME_PHASES)


# TURN


def test_turn_fields_round_trip(state):
    values = {
        'round': K.ROUND_OPERATING, 'step': K.STEP_TRACK, 'return_step': K.STEP_TOKEN, 'turn': 3,
        'or_num': 2, 'or_count': 2, 'active_player': 3, 'priority_player': 1, 'end_reason': 1,
        'auction_turn': 2, 'auction_private': D.PRIVATE_SMR, 'auction_discount': 10,
        'auction_pass_payouts': 1, 'sr_bought': 1, 'sr_sold_mask': 0b110, 'sr_sold_then_bought': 1,
        'operating_corp': D.CORP_UR, 'or_index': 4, 'route_revenue': 230, 'pending_hex': 51,
        'pending_corp': D.CORP_KU, 'pending_private': D.PRIVATE_ER, 'pending_train_source': 17,
        'pending_seller': 3, 'ebuy_sold_mask': 0b1, 'ebuy_sold_any': 1,
    }
    for name, value in values.items():
        getattr(TURN, f'set_{name}')(state, value)
    assert {name: getattr(TURN, name)(state) for name in values} == values
    TURN.set_or_order(state, 0, D.CORP_TR)
    assert [TURN.or_order(state, i) for i in range(2)] == [D.CORP_TR, -1]


@pytest.mark.parametrize('name, value', [
    ('round', K.NUM_ROUNDS), ('step', K.NUM_STEPS), ('return_step', -2), ('active_player', 4),
    ('priority_player', -1), ('end_reason', K.NUM_END_REASONS), ('auction_turn', 4),
    ('auction_private', D.NUM_PRIVATES),
    ('operating_corp', D.NUM_CORPS), ('pending_hex', D.NUM_HEXES), ('pending_seller', 4),
])
def test_turn_setters_check_ranges(state, name, value):
    with pytest.raises(AssertionError):
        getattr(TURN, f'set_{name}')(state, value)


def test_next_player_wraps(state):
    assert [TURN.next_player(state, p) for p in range(4)] == [1, 2, 3, 0]
