"""g1889.entities: cash transfers, bids, private ownership and revenue, the
auction's queries and sales, the turn fields, and the raw accessors' cache
invalidation (1889/engine-design.md §4)."""

import pytest

import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.bank import BANK, BANK_ACCOUNT
from g1889.entities.corp import CORPS, CORPS_BY_SYM
from g1889.entities.hexmap import MAP
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS, all_passed, clear_passed
from g1889.entities.private import PRIVATES, PRIVATES_BY_SYM, pay_revenue, topmost
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


def test_available_cash_excludes_other_bids(state):
    """``max_bid``: a bid may use the player's cash less their bids elsewhere."""
    player = PLAYERS[2]
    player.set_bid(state, D.PRIVATE_ER, 45)
    player.set_bid(state, D.PRIVATE_UTF, 155)
    assert player.available(state, D.PRIVATE_ER) == 420 - 155
    assert player.available(state, D.PRIVATE_UTF) == 420 - 45
    assert player.available(state, D.PRIVATE_SMR) == player.available(state) == 420 - 200
    player.set_bid(state, D.PRIVATE_UTF, 160)  # raising replaces the bid
    assert player.committed(state) == 205


def test_passed_flags_of_every_player(state):
    for i in range(3):
        PLAYERS[i].set_passed(state, True)
    assert not all_passed(state)
    PLAYERS[3].set_passed(state, True)
    assert all_passed(state)
    clear_passed(state)
    assert not any(PLAYERS[i].passed(state) for i in range(4))


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


# The auction's queries and sales


def test_topmost_and_price_follow_the_discount(state):
    a, mf = PRIVATES[D.PRIVATE_TR], PRIVATES[D.PRIVATE_MF]
    assert topmost(state) == D.PRIVATE_TR and (a.price(state), a.min_bid(state)) == (20, 20)
    TURN.set_auction_discount(state, 15)
    assert (a.price(state), a.min_bid(state), mf.price(state)) == (5, 5, 30)  # only A is discounted
    a.sell_to_player(state, 1, 5)
    assert topmost(state) == D.PRIVATE_MF and mf.min_bid(state) == 30
    for p in range(D.PRIVATE_MF, D.NUM_PRIVATES):
        PRIVATES[p].sell_to_player(state, 0, 0)
    assert topmost(state) == -1


def test_bid_queries(state):
    """Without bids the minimum is face + ¥5; with them, the high bid + ¥5. The
    topmost's minimum is its price, except in its closed auction."""
    er = PRIVATES[D.PRIVATE_ER]
    assert (er.high_bid(state), er.bid_count(state), er.lowest_bidder(state), er.min_bid(state)) == (0, 0, -1, 45)
    for player, amount in ((3, 50), (1, 45), (0, 60)):
        PLAYERS[player].set_bid(state, D.PRIVATE_ER, amount)
    assert (er.high_bid(state), er.bid_count(state), er.lowest_bidder(state), er.min_bid(state)) == (60, 3, 1, 65)
    PRIVATES[D.PRIVATE_TR].sell_to_player(state, 2, 20)
    PRIVATES[D.PRIVATE_MF].sell_to_player(state, 2, 30)
    TURN.set_auction_private(state, D.PRIVATE_ER)
    assert er.min_bid(state) == 65
    TURN.set_auction_private(state, -1)
    PLAYERS[0].set_bid(state, D.PRIVATE_ER, 0)
    PLAYERS[1].set_bid(state, D.PRIVATE_ER, 0)
    PLAYERS[3].set_bid(state, D.PRIVATE_ER, 0)
    assert er.min_bid(state) == 40


def test_sale_to_a_player_pays_the_bank_and_ends_the_bids(state):
    sir = PRIVATES[D.PRIVATE_SIR]
    PLAYERS[1].set_bid(state, D.PRIVATE_SIR, 85)
    PLAYERS[1].set_bid(state, D.PRIVATE_DR, 65)
    bank = BANK.cash(state)
    sir.sell_to_player(state, 1, 85)
    assert sir.owner(state) == K.OWNER_PLAYER + 1 and BANK.cash(state) == bank + 85
    assert (PLAYERS[1].cash(state), PLAYERS[1].bid(state, D.PRIVATE_SIR), PLAYERS[1].committed(state)) == (335, 0, 65)
    with pytest.raises(AssertionError):  # sold
        sir.sell_to_player(state, 2, 85)
    PRIVATES[D.PRIVATE_TR].sell_to_player(state, 3, 0)  # the free A moves no cash
    assert PRIVATES[D.PRIVATE_TR].owner(state) == K.OWNER_PLAYER + 3 and BANK.cash(state) == bank + 85
    with pytest.raises(AssertionError):  # beyond the player's cash
        PRIVATES[D.PRIVATE_UTF].sell_to_player(state, 0, 421)
    assert PRIVATES[D.PRIVATE_UTF].owner(state) == K.OWNER_UNSOLD


def test_revenue_goes_to_owners_from_the_bank(state):
    """Players and corps collect; unsold, closed, and absent privates pay nothing."""
    PRIVATES[D.PRIVATE_TR].set_owner(state, K.OWNER_PLAYER + 2)
    PRIVATES[D.PRIVATE_SMR].set_owner(state, K.OWNER_PLAYER + 2)
    PRIVATES[D.PRIVATE_DR].set_owner(state, corp_account(D.CORP_KO))
    PRIVATES[D.PRIVATE_ER].set_owner(state, K.OWNER_CLOSED)
    PRIVATES[D.PRIVATE_SIR].set_owner(state, K.OWNER_NOT_IN_PLAY)
    bank = BANK.cash(state)
    pay_revenue(state)
    assert PLAYERS[2].cash(state) == 420 + 5 + 15 and CORPS[D.CORP_KO].cash(state) == 15
    assert BANK.cash(state) == bank - 35 and PLAYERS[0].cash(state) == 420
    BANK.transfer(state, BANK_ACCOUNT, corp_account(D.CORP_AR), BANK.cash(state) - 10)
    pay_revenue(state)  # the bank breaks and still pays
    assert BANK.cash(state) == -25 and BANK.broken(state)


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


@pytest.mark.parametrize('cell, down, up', [
    ((5, 3), (6, 3), (4, 3)),
    ((10, 3), (10, 3), (9, 3)),       # the bottom of a column: no move down
    ((4, 7), (4, 7), (3, 7)),         # row 5 stops at column 6, so (4, 7) is a column's bottom
    ((0, 14), (1, 14), (0, 14)),      # the top row: no move up
    ((10, 0), (10, 0), (9, 0)),
])
def test_market_moves_stop_at_the_edges(state, cell, down, up):
    for move, target in ((MARKET.move_down, down), (MARKET.move_up, up)):
        MARKET.set_position(state, D.CORP_IR, *cell)
        stamp = MARKET.stamp(state, D.CORP_IR)
        move(state, D.CORP_IR)
        assert (MARKET.row(state, D.CORP_IR), MARKET.col(state, D.CORP_IR)) == target
        # A move goes to the bottom of its new cell; a blocked one keeps its place.
        assert (MARKET.stamp(state, D.CORP_IR) == stamp) == (target == cell)


def test_market_moves_join_the_bottom_of_a_stack(state):
    MARKET.set_position(state, D.CORP_KO, 5, 3)
    MARKET.set_position(state, D.CORP_AR, 4, 3)
    MARKET.set_position(state, D.CORP_IR, 4, 3)
    MARKET.move_up(state, D.CORP_KO)
    MARKET.move_down(state, D.CORP_AR)
    MARKET.move_up(state, D.CORP_AR)
    assert MARKET.sort_by_price(state, ALL_CORPS) == [D.CORP_IR, D.CORP_KO, D.CORP_AR]


def test_stock_price_order(state):
    """Higher price, then farther right, then higher in the stack (rules §6.7)."""
    for corp, cell in ((D.CORP_AR, (5, 3)), (D.CORP_IR, (2, 1)), (D.CORP_SR, (3, 0)), (D.CORP_KO, (5, 3)),
                       (D.CORP_TR, (0, 0)), (D.CORP_UR, (2, 0))):
        MARKET.set_position(state, corp, *cell)
    # ¥75 (TR), ¥70 (IR), three at ¥65 (AR over KO in column 3, then UR in column 0), ¥60 (SR).
    assert MARKET.sort_by_price(state, ALL_CORPS) == [D.CORP_TR, D.CORP_IR, D.CORP_AR, D.CORP_KO, D.CORP_UR,
                                                    D.CORP_SR]
    assert MARKET.sort_by_price(state, 1 << D.CORP_KO | 1 << D.CORP_AR | 1 << D.CORP_KU) == [D.CORP_AR, D.CORP_KO]
    keys = sorted(MARKET.order_key(state, c) for c in (D.CORP_AR, D.CORP_KO))
    assert keys[0] == MARKET.order_key(state, D.CORP_AR)
    MARKET.set_position(state, D.CORP_KU, 4, 2)  # ¥65 again, but farther left than (5, 3)
    assert MARKET.sort_by_price(state, 1 << D.CORP_KO | 1 << D.CORP_KU) == [D.CORP_KO, D.CORP_KU]


@pytest.mark.parametrize('cell, counts, unlimited', [
    ((5, 3), True, False), ((5, 0), False, False), ((8, 3), False, False), ((8, 0), False, True),
    ((10, 2), False, True), ((10, 3), False, False), ((4, 0), True, False),
])
def test_market_zones(state, cell, counts, unlimited):
    assert MARKET.counts_for_limit(state, D.CORP_IR) and not MARKET.unlimited(state, D.CORP_IR)  # unparred
    MARKET.set_position(state, D.CORP_IR, *cell)
    assert (MARKET.counts_for_limit(state, D.CORP_IR), MARKET.unlimited(state, D.CORP_IR)) == (counts, unlimited)


# Share operations


def _parred(state, corp=D.CORP_IR, player=0, par_index=0):
    CORPS[corp].par(state, player, par_index)
    return CORPS[corp]


def test_par_buys_the_certificate_and_places_the_token(state):
    corp = _parred(state, par_index=2)  # ¥75
    assert (corp.par_price(state), corp.president(state), corp.ipo_shares(state)) == (75, 0, 8)
    assert PLAYERS[0].shares(state, D.CORP_IR) == 2 and PLAYERS[0].cash(state) == 420 - 150
    assert (MARKET.row(state, D.CORP_IR), MARKET.col(state, D.CORP_IR)) == (D.PAR_ROW[2], D.PAR_COL[2])
    with pytest.raises(AssertionError):
        corp.par(state, 1, 0)  # parred once


def test_buys_pay_the_bank_and_float_at_half(state):
    corp = _parred(state)
    corp.buy_share(state, 1, False)
    corp.buy_share(state, 2, False)
    assert not corp.floated(state) and corp.cash(state) == 0
    corp.buy_share(state, 3, False)
    assert corp.floated(state) and corp.cash(state) == 650 and corp.ipo_shares(state) == 5
    assert [PLAYERS[i].cash(state) for i in range(4)] == [290, 355, 355, 355]
    assert not corp.float_if_ready(state)  # once


def test_pool_buys_cost_the_market_price(state):
    corp = _parred(state)
    corp.buy_share(state, 1, False)
    corp.buy_share(state, 1, False)
    MARKET.set_position(state, D.CORP_IR, 7, 3)  # ¥55
    corp.sell_shares(state, 1, 2)
    assert PLAYERS[1].cash(state) == 420 - 130 + 110 and corp.pool_shares(state) == 2
    assert (MARKET.row(state, D.CORP_IR), MARKET.price(state, D.CORP_IR)) == (9, 45)
    corp.buy_share(state, 2, True)
    assert PLAYERS[2].cash(state) == 420 - 45 and corp.pool_shares(state) == 1


def test_presidency_follows_the_largest_holding(state):
    corp = _parred(state)
    for player in (1, 1):
        corp.buy_share(state, player, False)
    assert corp.president(state) == 0  # 2 to 2
    corp.buy_share(state, 1, False)
    assert corp.president(state) == 1  # 3 beats 2
    corp.buy_share(state, 0, False)
    corp.buy_share(state, 0, False)
    assert corp.president(state) == 0  # 4 beats 3
    corp.sell_shares(state, 0, 2)
    assert corp.president(state) == 1 and [PLAYERS[i].shares(state, D.CORP_IR) for i in range(2)] == [2, 3]


def test_successor_ties_break_clockwise_from_the_president(state):
    corp = _parred(state, player=2)
    for player in (2, 3, 3, 1, 1):
        corp.buy_share(state, player, False)
    corp.sell_shares(state, 2, 2)  # seats 3 and 1 hold two each; 3 is first clockwise from 2
    assert corp.president(state) == 3


def test_max_sale(state):
    corp = _parred(state)
    corp.buy_share(state, 0, False)
    corp.buy_share(state, 1, False)
    assert corp.max_sale(state, 0) == 1  # the certificate needs a two-share holder
    assert corp.max_sale(state, 1) == 1
    corp.buy_share(state, 1, False)
    assert corp.max_sale(state, 0) == 3
    corp.sell_shares(state, 1, 2)
    assert corp.max_sale(state, 0) == 1 and corp.pool_shares(state) == 2  # nobody holds two now
    CORPS[D.CORP_KO].set_ipo_shares(state, 9)
    PLAYERS[2].set_shares(state, D.CORP_KO, 1)
    assert CORPS[D.CORP_KO].max_sale(state, 2) == 0  # unparred


def test_sold_out(state):
    corp = _parred(state)
    for i in range(8):
        corp.buy_share(state, 1 + i % 3, False)
    assert corp.sold_out(state)
    corp.sell_shares(state, 1, 1)
    assert not corp.sold_out(state)


# Certificates and limits


def test_cert_count_counts_privates_and_one_per_certificate(state):
    PRIVATES[D.PRIVATE_ER].set_owner(state, K.OWNER_PLAYER)
    PRIVATES[D.PRIVATE_MF].set_owner(state, K.OWNER_CORP)
    corp = _parred(state)
    corp.buy_share(state, 0, False)
    assert PLAYERS[0].cert_count(state) == 1 + 2  # ER, the certificate, one share
    MARKET.set_position(state, D.CORP_IR, 7, 1)  # yellow
    assert PLAYERS[0].cert_count(state) == 1


def test_can_gain_checks_sixty_percent_and_the_limit(state):
    corp = _parred(state)
    for _ in range(3):
        corp.buy_share(state, 0, False)
    assert PLAYERS[0].can_gain(state, D.CORP_IR) and not PLAYERS[0].can_gain(state, D.CORP_IR, 2)
    corp.buy_share(state, 0, False)
    assert not PLAYERS[0].can_gain(state, D.CORP_IR)  # 60% held
    MARKET.set_position(state, D.CORP_IR, 8, 0)  # orange
    assert PLAYERS[0].can_gain(state, D.CORP_IR)
    for p in range(D.NUM_PRIVATES):
        PRIVATES[p].set_owner(state, K.OWNER_PLAYER + 1)
    BANK.transfer(state, BANK_ACCOUNT, 1, 200)
    for c in (D.CORP_KO, D.CORP_SR):
        _parred(state, c, player=1)
        for _ in range(2):
            CORPS[c].buy_share(state, 1, False)
    assert PLAYERS[1].cert_count(state) == 13 and PLAYERS[1].can_gain(state, D.CORP_UR)
    CORPS[D.CORP_KO].buy_share(state, 1, False)
    assert PLAYERS[1].cert_count(state) == D.CERT_LIMIT[4] and not PLAYERS[1].can_gain(state, D.CORP_UR)
    assert PLAYERS[1].can_gain(state, D.CORP_IR)  # orange shares do not count


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
    ('auction_private', D.NUM_PRIVATES), ('auction_discount', -5), ('auction_discount', 25),
    ('auction_pass_payouts', -1), ('sr_bought', 2), ('sr_sold_mask', 1 << D.NUM_CORPS), ('sr_sold_then_bought', -1),
    ('operating_corp', D.NUM_CORPS), ('pending_hex', D.NUM_HEXES), ('pending_seller', 4),
])
def test_turn_setters_check_ranges(state, name, value):
    with pytest.raises(AssertionError):
        getattr(TURN, f'set_{name}')(state, value)


def test_next_player_wraps(state):
    assert [TURN.next_player(state, p) for p in range(4)] == [1, 2, 3, 0]
