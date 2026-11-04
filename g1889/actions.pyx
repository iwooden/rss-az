"""Phase-local action encoding and legality for the 1889 engine; see actions.pxd."""

from collections import namedtuple

cimport numpy as cnp
import numpy as np
from libc.stdint cimport int8_t, uint16_t, uint32_t
from libc.stdio cimport fprintf, stderr
from libc.stdlib cimport abort

from g1889.constants cimport (
    ACTION_SIZE_BUY_PRIVATE_PRE, AUCTION_BID, AUCTION_BUY, AUCTION_PASS, BID_DROP, BID_POINT, BID_POINTS,
    BUY_PRIVATE_POST_DONE, BUY_PRIVATE_POST_DR, BUY_PRIVATE_POST_PRIVATE, BUY_PRIVATE_PRE_DONE, BUY_PRIVATE_PRE_DR,
    BUY_PRIVATE_PRE_MF, BUY_PRIVATE_PRE_PRIVATE, BUY_TRAIN_BANKRUPT, BUY_TRAIN_CORP, BUY_TRAIN_DEPOT, BUY_TRAIN_PASS,
    BUY_TRAIN_POOL, BUY_TRAIN_SELL, DIVIDEND_PAYOUT, DIVIDEND_WITHHOLD, DPHASE_AUCTION, DPHASE_BID,
    DPHASE_BUY_PRIVATE_POST, DPHASE_BUY_PRIVATE_PRE, DPHASE_BUY_TRAIN, DPHASE_DISCARD, DPHASE_DIVIDEND,
    DPHASE_EMERGENCY_SELL_AMOUNT, DPHASE_ER_BONUS, DPHASE_PAR, DPHASE_PORT, DPHASE_PRIVATE_PRICE,
    DPHASE_SELECT_TRAIN, DPHASE_SELL_AMOUNT, DPHASE_STOCK, DPHASE_TILE, DPHASE_TOKEN, DPHASE_TRACK,
    DPHASE_TRAIN_PRICE, ER_BONUS_CANDIDATE, ER_BONUS_PASS, MAX_ACTION_SIZE, MAX_SALE_SHARES, NUM_DECISION_PHASES,
    NUM_STEPS, NUM_TRAIN_SOURCES, OWNER_CORP, OWNER_PLAYER, OWNER_UNSOLD, PHASE_ACTION_SIZE, PORT_LAY,
    PRIVATE_PRICE_POINTS, ROUND_GAME_OVER, ROUND_OPERATING, SELECT_TRAIN_TRADE_IN, SELECT_TRAIN_TYPE, STEP_AUCTION,
    STEP_BUY_TRAIN, STEP_DIVIDEND, STEP_TOKEN, STEP_TO_DECISION_PHASE, STEP_TRACK, STOCK_DR, STOCK_IPO, STOCK_MF,
    STOCK_PAR, STOCK_PASS, STOCK_POOL, STOCK_SELL, TOKEN_HEX, TOKEN_PASS, TRACK_HEX, TRACK_PASS, TRAIN_PRICE_POINTS,
    TRAIN_SOURCE_CORP, TRAIN_SOURCE_POOL,
)
from g1889.data.tables cimport (
    CERT_LIMIT, D_TRAIN, DISCARD_TYPES, DR_EXCHANGE_CORP, ENDPOINT_PAIRS, ER_HEX, ER_TILE, HEX_EXIT_MASK, HEX_NEIGHBOR,
    HEX_TOKEN_SLOT, HEX_TRACK_SLOT, MAX_LAY_CANDIDATES, MAX_PLAYERS, MF_PORT_HEX, MIN_BID_INCREMENT, NODE, NUM_CORPS,
    NUM_DISCARD_TYPES, NUM_ER_CANDIDATES, NUM_ER_TILES, NUM_HEXES, NUM_PAR_PRICES, NUM_PORTS, NUM_PORT_LAYS,
    NUM_PRIVATES, NUM_SIDES, NUM_TOKEN_HEXES, NUM_TRACK_HEXES, NUM_TRADE_IN_TYPES, NUM_TRAIN_TYPES, PAR_PRICE,
    PHASE_CAN_BUY_PRIVATES, PHASE_TILE_COLORS, PORT_LAY_HEX, PORT_LAY_ROT, PORT_LAY_SLOT, PORT_TILE, PRIVATE_DR,
    PRIVATE_MF, PRIVATE_TR, PRIVATE_VALUE, TILE_BLOCKS_LAY, TILE_COLOR, TILE_EXITS, TILE_MASK, TILE_REVENUE,
    TOKEN_HEXES, TRACK_HEXES, TRADE_IN_TYPE, TRAIN_DISCARD_SLOT, TRAIN_PRICE, UPG_ROT, UPG_START, UPG_TILE,
)
from g1889.entities.corp cimport (
    PRESIDENT_SHARES, corp_cash, corp_emergency_sale, corp_ipo_shares, corp_max_sale, corp_par_index,
    corp_par_price, corp_pool_shares, corp_president, corp_token_price, corp_tokens_left,
)
from g1889.entities.graph cimport graph_node, graph_route_available, graph_sides, graph_walks_new_track
from g1889.entities.hexmap cimport hex_paths, hex_rotation, hex_tile, map_has_slot_for, map_lay_cost, supply_count
from g1889.entities.market cimport market_price
from g1889.entities.player cimport (
    player_available, player_can_gain, player_can_hold, player_cash, player_cert_count, player_shares,
    player_sold_mask, player_value,
)
from g1889.entities.phase cimport game_phase
from g1889.entities.private cimport (
    MF_PORT_USED, private_blocks_hex, private_flags, private_min_bid, private_owner, private_price,
    topmost_private,
)
from g1889.entities.trains cimport (
    corp_train_count, corp_train_total, pool_count, trade_in_price, trains_cheapest_in_pool, trains_cheapest_price,
    trains_cheapest_type, trains_depot_types, trains_first_crowded, trains_limit,
)
from g1889.entities.turn cimport (
    turn_active_player, turn_auction_pass_payouts, turn_auction_private, turn_ebuy_sold_any, turn_ebuy_sold_mask,
    turn_num_players, turn_operating_corp, turn_pending_corp, turn_pending_hex, turn_pending_private,
    turn_pending_train_source, turn_pending_train_type, turn_round, turn_route_revenue, turn_sr_bought,
    turn_sr_sold_mask, turn_sr_sold_then_bought, turn_step, turn_turn,
)
from g1889.state cimport GameState

from g1889.constants import DecisionPhase

cnp.import_array()

ActionInfoTuple = namedtuple('ActionInfoTuple', (
    'phase', 'type', 'corp', 'private', 'hex', 'train_type', 'count', 'amount'))

_PHASE_NAMES = tuple(DecisionPhase(p).name.removeprefix('DPHASE_') for p in range(<int>NUM_DECISION_PHASES))

# Exact ids are ``size + amount``; amounts stop where ids leave uint16.
cdef enum:
    MAX_ACTION_ID = 0xFFFF


# =============================================================================
# ENCODE / DECODE
# =============================================================================

cdef inline int _index(int value, int count) except -1 nogil:
    assert 0 <= value < count, f"index {value} out of range [0, {count})"
    return value


cdef inline int _exact(int phase, int amount) except -1 nogil:
    cdef int base = PHASE_ACTION_SIZE[phase]
    assert 0 <= amount <= MAX_ACTION_ID - base, f"exact amount {amount} out of range"
    return base + amount


cdef int _find(const int8_t* values, int count, int value) except -1 nogil:
    cdef int i
    for i in range(count):
        if values[i] == value:
            return i
    assert False, f"{value} is not one of the {count} encodable values"
    return 0


cdef int encode_action(ActionInfo info) except -1 nogil:
    """Pack ``info`` into its phase-local id: the inverse of ``decode_action``."""
    cdef int phase = info.phase
    cdef int t = info.type
    if phase == DPHASE_AUCTION:
        if t == ACTION_PASS:
            return AUCTION_PASS
        if t == ACTION_BUY_TOPMOST:
            return AUCTION_BUY
        if t == ACTION_SELECT_BID:
            return AUCTION_BID + _index(info.private - 1, <int>NUM_PRIVATES - 1)
    elif phase == DPHASE_BID:
        if t == ACTION_PASS:
            return BID_DROP
        if t == ACTION_PRICE_POINT:
            return BID_POINT + _index(info.count, <int>BID_POINTS)
        if t == ACTION_EXACT_PRICE:
            return _exact(phase, info.amount)
    elif phase == DPHASE_TRAIN_PRICE or phase == DPHASE_PRIVATE_PRICE:
        if t == ACTION_PRICE_POINT:
            return _index(info.count, PHASE_ACTION_SIZE[phase])
        if t == ACTION_EXACT_PRICE:
            return _exact(phase, info.amount)
    elif phase == DPHASE_STOCK:
        if t == ACTION_PASS:
            return STOCK_PASS
        if t == ACTION_BUY_IPO:
            return STOCK_IPO + _index(info.corp, <int>NUM_CORPS)
        if t == ACTION_BUY_POOL:
            return STOCK_POOL + _index(info.corp, <int>NUM_CORPS)
        if t == ACTION_SELECT_PAR:
            return STOCK_PAR + _index(info.corp, <int>NUM_CORPS)
        if t == ACTION_SELECT_SALE:
            return STOCK_SELL + _index(info.corp, <int>NUM_CORPS)
        if t == ACTION_DR_EXCHANGE:
            return STOCK_DR
        if t == ACTION_MF_PORT:
            return STOCK_MF
    elif phase == DPHASE_PAR:
        if t == ACTION_PAR:
            return _index(info.count, PHASE_ACTION_SIZE[phase])
    elif phase == DPHASE_SELL_AMOUNT or phase == DPHASE_EMERGENCY_SELL_AMOUNT:
        if t == ACTION_SELL:
            return _index(info.count - 1, <int>MAX_SALE_SHARES)
    elif phase == DPHASE_BUY_PRIVATE_PRE:
        if t == ACTION_PASS:
            return BUY_PRIVATE_PRE_DONE
        if t == ACTION_BUY_PRIVATE:
            return BUY_PRIVATE_PRE_PRIVATE + _index(info.private, <int>NUM_PRIVATES)
        if t == ACTION_DR_EXCHANGE:
            return BUY_PRIVATE_PRE_DR
        if t == ACTION_MF_PORT:
            return BUY_PRIVATE_PRE_MF
    elif phase == DPHASE_TRACK:
        if t == ACTION_PASS:
            return TRACK_PASS
        if t == ACTION_SELECT_HEX:
            return TRACK_HEX + _index(HEX_TRACK_SLOT[_index(info.hex, <int>NUM_HEXES)], <int>NUM_TRACK_HEXES)
    elif phase == DPHASE_TILE:
        if t == ACTION_LAY_CANDIDATE:
            return _index(info.count, PHASE_ACTION_SIZE[phase])
    elif phase == DPHASE_TOKEN:
        if t == ACTION_PASS:
            return TOKEN_PASS
        if t == ACTION_PLACE_TOKEN:
            return TOKEN_HEX + _index(HEX_TOKEN_SLOT[_index(info.hex, <int>NUM_HEXES)], <int>NUM_TOKEN_HEXES)
    elif phase == DPHASE_DIVIDEND:
        if t == ACTION_WITHHOLD:
            return DIVIDEND_WITHHOLD
        if t == ACTION_PAYOUT:
            return DIVIDEND_PAYOUT
    elif phase == DPHASE_DISCARD:
        if t == ACTION_DISCARD:
            return _index(TRAIN_DISCARD_SLOT[_index(info.train_type, <int>NUM_TRAIN_TYPES)], <int>NUM_DISCARD_TYPES)
    elif phase == DPHASE_BUY_TRAIN:
        if t == ACTION_PASS:
            return BUY_TRAIN_PASS
        if t == ACTION_BUY_DEPOT_TRAIN:
            return BUY_TRAIN_DEPOT
        if t == ACTION_BUY_POOL_TRAIN:
            return BUY_TRAIN_POOL
        if t == ACTION_BUY_CORP_TRAIN:
            return BUY_TRAIN_CORP + _index(info.corp, <int>NUM_CORPS)
        if t == ACTION_SELECT_SALE:
            return BUY_TRAIN_SELL + _index(info.corp, <int>NUM_CORPS)
        if t == ACTION_BANKRUPT:
            return BUY_TRAIN_BANKRUPT
    elif phase == DPHASE_SELECT_TRAIN:
        if t == ACTION_TRAIN_TYPE:
            return SELECT_TRAIN_TYPE + _index(info.train_type, <int>NUM_TRAIN_TYPES)
        if t == ACTION_TRADE_IN:
            return SELECT_TRAIN_TRADE_IN + _find(TRADE_IN_TYPE, <int>NUM_TRADE_IN_TYPES, info.train_type)
    elif phase == DPHASE_BUY_PRIVATE_POST:
        if t == ACTION_PASS:
            return BUY_PRIVATE_POST_DONE
        if t == ACTION_BUY_PRIVATE:
            return BUY_PRIVATE_POST_PRIVATE + _index(info.private, <int>NUM_PRIVATES)
        if t == ACTION_DR_EXCHANGE:
            return BUY_PRIVATE_POST_DR
    elif phase == DPHASE_ER_BONUS:
        if t == ACTION_PASS:
            return ER_BONUS_PASS
        if t == ACTION_LAY_CANDIDATE:
            return ER_BONUS_CANDIDATE + _index(info.count, PHASE_ACTION_SIZE[phase] - <int>ER_BONUS_CANDIDATE)
    elif phase == DPHASE_PORT:
        if t == ACTION_PORT_LAY:
            return PORT_LAY + _index(
                PORT_LAY_SLOT[_find(MF_PORT_HEX, <int>NUM_PORTS, info.hex)][_index(info.count, <int>NUM_SIDES)],
                <int>NUM_PORT_LAYS)
    assert False, f"encode_action: no action of type {t} in decision phase {phase}"
    return 0


cdef inline void _decode_price(ActionInfo* info, int action_id, int first_point, int exact) noexcept nogil:
    if action_id >= exact:
        info.type = ACTION_EXACT_PRICE
        info.amount = action_id - exact
    else:
        info.type = ACTION_PRICE_POINT
        info.count = action_id - first_point


cdef ActionInfo decode_action(int phase, int action_id) noexcept nogil:
    """Unpack a phase-local id; exact ids (at or above the phase size) decode to
    ACTION_EXACT_PRICE. The id must be valid for the phase."""
    cdef ActionInfo info
    info.phase = phase
    info.type = ACTION_PASS
    info.corp = info.private = info.hex = info.train_type = info.count = info.amount = -1

    if phase == DPHASE_AUCTION:
        if action_id == AUCTION_BUY:
            info.type = ACTION_BUY_TOPMOST
        elif action_id >= AUCTION_BID:
            info.type = ACTION_SELECT_BID
            info.private = action_id - AUCTION_BID + 1
    elif phase == DPHASE_BID:
        if action_id != BID_DROP:
            _decode_price(&info, action_id, BID_POINT, PHASE_ACTION_SIZE[phase])
    elif phase == DPHASE_TRAIN_PRICE or phase == DPHASE_PRIVATE_PRICE:
        _decode_price(&info, action_id, 0, PHASE_ACTION_SIZE[phase])
    elif phase == DPHASE_STOCK:
        if action_id == STOCK_PASS:
            pass
        elif action_id < STOCK_POOL:
            info.type = ACTION_BUY_IPO
            info.corp = action_id - STOCK_IPO
        elif action_id < STOCK_PAR:
            info.type = ACTION_BUY_POOL
            info.corp = action_id - STOCK_POOL
        elif action_id < STOCK_SELL:
            info.type = ACTION_SELECT_PAR
            info.corp = action_id - STOCK_PAR
        elif action_id < STOCK_DR:
            info.type = ACTION_SELECT_SALE
            info.corp = action_id - STOCK_SELL
        else:
            info.type = ACTION_MF_PORT if action_id == STOCK_MF else ACTION_DR_EXCHANGE
    elif phase == DPHASE_PAR:
        info.type = ACTION_PAR
        info.count = action_id
        info.amount = PAR_PRICE[action_id]
    elif phase == DPHASE_SELL_AMOUNT or phase == DPHASE_EMERGENCY_SELL_AMOUNT:
        info.type = ACTION_SELL
        info.count = action_id + 1
    elif phase == DPHASE_BUY_PRIVATE_PRE:
        if action_id == BUY_PRIVATE_PRE_DONE:
            pass
        elif action_id < BUY_PRIVATE_PRE_DR:
            info.type = ACTION_BUY_PRIVATE
            info.private = action_id - BUY_PRIVATE_PRE_PRIVATE
        else:
            info.type = ACTION_MF_PORT if action_id == BUY_PRIVATE_PRE_MF else ACTION_DR_EXCHANGE
    elif phase == DPHASE_TRACK:
        if action_id != TRACK_PASS:
            info.type = ACTION_SELECT_HEX
            info.hex = TRACK_HEXES[action_id - TRACK_HEX]
    elif phase == DPHASE_TILE:
        info.type = ACTION_LAY_CANDIDATE
        info.count = action_id
    elif phase == DPHASE_TOKEN:
        if action_id != TOKEN_PASS:
            info.type = ACTION_PLACE_TOKEN
            info.hex = TOKEN_HEXES[action_id - TOKEN_HEX]
    elif phase == DPHASE_DIVIDEND:
        info.type = ACTION_WITHHOLD if action_id == DIVIDEND_WITHHOLD else ACTION_PAYOUT
    elif phase == DPHASE_DISCARD:
        info.type = ACTION_DISCARD
        info.train_type = DISCARD_TYPES[action_id]
    elif phase == DPHASE_BUY_TRAIN:
        if action_id == BUY_TRAIN_PASS:
            pass
        elif action_id == BUY_TRAIN_DEPOT:
            info.type = ACTION_BUY_DEPOT_TRAIN
        elif action_id == BUY_TRAIN_POOL:
            info.type = ACTION_BUY_POOL_TRAIN
        elif action_id < BUY_TRAIN_SELL:
            info.type = ACTION_BUY_CORP_TRAIN
            info.corp = action_id - BUY_TRAIN_CORP
        elif action_id < BUY_TRAIN_BANKRUPT:
            info.type = ACTION_SELECT_SALE
            info.corp = action_id - BUY_TRAIN_SELL
        else:
            info.type = ACTION_BANKRUPT
    elif phase == DPHASE_SELECT_TRAIN:
        if action_id < SELECT_TRAIN_TRADE_IN:
            info.type = ACTION_TRAIN_TYPE
            info.train_type = action_id - SELECT_TRAIN_TYPE
        else:
            info.type = ACTION_TRADE_IN
            info.train_type = TRADE_IN_TYPE[action_id - SELECT_TRAIN_TRADE_IN]
    elif phase == DPHASE_BUY_PRIVATE_POST:
        if action_id == BUY_PRIVATE_POST_DONE:
            pass
        elif action_id < BUY_PRIVATE_POST_DR:
            info.type = ACTION_BUY_PRIVATE
            info.private = action_id - BUY_PRIVATE_POST_PRIVATE
        else:
            info.type = ACTION_DR_EXCHANGE
    elif phase == DPHASE_ER_BONUS:
        if action_id != ER_BONUS_PASS:
            info.type = ACTION_LAY_CANDIDATE
            info.count = action_id - ER_BONUS_CANDIDATE
    elif phase == DPHASE_PORT:
        info.type = ACTION_PORT_LAY
        info.hex = PORT_LAY_HEX[action_id - PORT_LAY]
        info.count = PORT_LAY_ROT[action_id - PORT_LAY]
    return info


# =============================================================================
# DECISION PHASE AND LEGALITY
# =============================================================================

cdef int get_decision_phase(GameState state) noexcept nogil:
    """The decision phase of the current step, or -1 at an automated or
    terminal step (or a corrupt one, which the driver rejects)."""
    cdef int step = turn_step(state)
    if step < 0 or step >= <int>NUM_STEPS:
        return -1
    return STEP_TO_DECISION_PHASE[step]


cdef bint is_terminal(GameState state) noexcept nogil:
    """Whether the game is over: a bankruptcy, or a broken bank's OR set,
    has ended it (``end_reason`` tells which)."""
    return turn_round(state) == ROUND_GAME_OVER


cdef int final_scores(GameState state, int* scores) noexcept nogil:
    """Write each canonical player's score, Ruby's ``Player#value``
    (``player_value``), to ``scores`` and return the player count. Ruby's
    ``result`` ranks the same values once the game is over; they are defined
    at any state, a bankrupt player's included."""
    cdef int n = turn_num_players(state)
    cdef int i
    for i in range(n):
        scores[i] = player_value(state, i)
    return n


cpdef bint has_exact_amounts(int phase) noexcept nogil:
    """Whether ``phase`` takes exact amounts, at ids from its size up."""
    return phase == DPHASE_BID or phase == DPHASE_TRAIN_PRICE or phase == DPHASE_PRIVATE_PRICE


# =============================================================================
# AUCTION, BID
# =============================================================================
#
# BID names the amount of a bid on the private AUCTION selected
# (``pending_private``), or in a closed auction on the topmost private
# (``auction_private``), whose lowest bidder raises or drops out (slot 0, legal
# only there). A bid ranges from the private's minimum bid
# (``private_min_bid``: in a closed auction the high bid + ¥5) to the bidder's
# cash less their other bids (``player_available``). Grid point k is the
# minimum + ¥5k; points past the range are not offered.

cdef inline int _bid_private(GameState state) noexcept nogil:
    """The private BID is for: the closed auction's, else the selected one."""
    cdef int p = turn_auction_private(state)
    return p if p >= 0 else turn_pending_private(state)


cdef int price_amount(GameState state, const ActionInfo* info) noexcept nogil:
    """The yen of a BID, TRAIN_PRICE, or PRIVATE_PRICE decision: its exact
    amount, or grid point ``count``."""
    cdef int points[5]  # TRAIN_PRICE_POINTS; PRIVATE_PRICE_POINTS fits
    if info.type == ACTION_EXACT_PRICE:
        return info.amount
    if info.phase == DPHASE_TRAIN_PRICE:
        _train_price_points(state, points)
        return points[info.count]
    if info.phase == DPHASE_PRIVATE_PRICE:
        _private_price_points(state, points)
        return points[info.count]
    return private_min_bid(state, _bid_private(state)) + <int>MIN_BID_INCREMENT * info.count


cdef inline bint _amount_in_range(GameState state, int phase, int amount) noexcept nogil:
    cdef int p, low, high
    if phase == DPHASE_TRAIN_PRICE:
        return 1 <= amount <= _train_price_cash(state)
    if phase == DPHASE_PRIVATE_PRICE:
        _private_price_range(state, &low, &high)
        return low <= amount <= high
    p = _bid_private(state)
    return private_min_bid(state, p) <= amount <= player_available(state, turn_active_player(state), p)


cdef int _enumerate_bid(GameState state, uint16_t* ids) noexcept nogil:
    """The drop in a closed auction, then the grid points the bidder covers."""
    cdef int p = _bid_private(state)
    cdef int low = private_min_bid(state, p)
    cdef int high = player_available(state, turn_active_player(state), p)
    cdef int count = 0
    cdef int k
    if turn_auction_private(state) >= 0:
        ids[0] = BID_DROP
        count = 1
    for k in range(<int>BID_POINTS):
        if low + <int>MIN_BID_INCREMENT * k > high:
            break
        ids[count] = <uint16_t>(BID_POINT + k)
        count += 1
    return count


cdef inline bint _can_bid(GameState state, int player, int private) noexcept nogil:
    """Whether ``private`` is unsold and ``player`` can cover its minimum bid."""
    return (private_owner(state, private) == <int>OWNER_UNSOLD
            and private_min_bid(state, private) <= player_available(state, player, private))


cdef int _enumerate_auction(GameState state, uint16_t* ids) noexcept nogil:
    """Pass; buy the topmost private at its price; select each unsold private
    but private 0 whose minimum bid the player can cover. On the topmost,
    BID's amount buys it. Private 0 is the topmost while it is unsold, so a bid
    on it is always that purchase, which buying at the price dominates: it has
    no slot, and replay takes a human's with ``DRIVER.override_topmost_bid``
    (§12 item 8)."""
    cdef int player = turn_active_player(state)
    cdef int top = topmost_private(state)
    cdef int count = 1
    cdef int p
    ids[0] = AUCTION_PASS
    if private_price(state, top) <= player_available(state, player, top):
        ids[count] = AUCTION_BUY
        count += 1
    for p in range(max(top, 1), <int>NUM_PRIVATES):
        if _can_bid(state, player, p):
            ids[count] = <uint16_t>(AUCTION_BID + p - 1)
            count += 1
    return count


cdef bint override_bid_legal(GameState state) noexcept nogil:
    """Whether replay may take a bid on private 0 now: the turn player's normal
    turn in the auction, as AUCTION would offer another private's slot."""
    return turn_step(state) == STEP_AUCTION and _can_bid(state, turn_active_player(state), <int>PRIVATE_TR)


# The pass cap (§7). Once ``auction_pass_payout_cap`` rounds of passes have paid
# out, a player who can buy the topmost private may not pass, nor bid so much
# that they no longer could. The second half closes rules §12.9's loop for
# good: without it, players who outbid themselves on later privates never
# afford the topmost, so the cap never binds and payouts go on forever. Once
# capped, a player who cannot buy may only pass, and one who can must buy
# within finitely many bids.

cdef inline bint _capped(GameState state) noexcept nogil:
    cdef int cap = state._auction_pass_payout_cap
    return cap > 0 and turn_auction_pass_payouts(state) >= cap


cdef inline int _capped_bid_limit(GameState state, int private) noexcept nogil:
    """The most the actor may bid on ``private`` and still afford the topmost."""
    return (player_available(state, turn_active_player(state), private)
            - private_price(state, topmost_private(state)))


cdef int _filter_auction(GameState state, uint16_t* ids, int count) noexcept nogil:
    """Drop the bid on the topmost private, which buying it at its price
    dominates (§5.1), and apply the pass cap."""
    cdef int top = topmost_private(state)
    cdef bint capped = _capped(state)
    cdef bint no_pass = capped and private_price(state, top) <= player_available(state, turn_active_player(state), top)
    cdef int i, p
    cdef int kept = 0
    for i in range(count):
        if ids[i] == AUCTION_PASS and no_pass:  # the actor can buy the topmost
            continue
        if ids[i] >= AUCTION_BID:
            p = ids[i] - AUCTION_BID + 1
            if p == top or (capped and private_min_bid(state, p) > _capped_bid_limit(state, p)):
                continue
        ids[kept] = ids[i]
        kept += 1
    return kept


cdef int _filter_bid(GameState state, int count) noexcept nogil:
    """Once capped, keep the points (ids 1, 2, ...) that leave the bidder able
    to buy the topmost; keep all of them if none does, which only a selection
    outside the policy can bring about. Bids on the topmost, a closed
    auction's included, are for the private the cap protects, so it leaves
    them alone."""
    cdef int p = _bid_private(state)
    cdef int room
    if not _capped(state) or p == topmost_private(state):
        return count
    room = _capped_bid_limit(state, p) - private_min_bid(state, p)
    if room < 0:
        return count
    return min(count, room // <int>MIN_BID_INCREMENT + 1)


# =============================================================================
# STOCK, PAR, SELL_AMOUNT
# =============================================================================
#
# Ruby's BuySellParShares (step/buy_sell_par_shares.rb). On a turn the player
# buys at most one certificate (a share, or a par through PAR) and sells any
# number of blocks, one per corp, until a purchase follows a sale
# (SELL_BUY_ORDER = :sell_buy_or_buy_sell). A sale is factored: STOCK picks
# the corp, and SELL_AMOUNT the block's size. Nothing is sold in the first SR
# (SELL_AFTER = :first), and nothing sold this SR is bought back. A player
# over a limit with something to sell has a must-sell turn: sales only, no
# pass. Slot 0 passes before acting and is "done" after; it is legal at every
# other turn. Every turn also offers the player's DR exchange (STOCK_DR) and
# MF port tile (STOCK_MF) while they are usable, which section INTERJECTIONS
# below has, so a player with nothing else to do passes through the forced
# rule, as Ruby skips them, only when neither private has a use.

cdef inline bint _can_buy(GameState state, int player, int corp, int price) noexcept nogil:
    """A 10% share at ``price`` (``can_buy?``): not sold this SR, affordable, and
    within the holding and certificate limits."""
    return (not (player_sold_mask(state, player) >> corp & 1) and price <= player_cash(state, player)
            and player_can_gain(state, player, corp, 1))


cdef inline bint _can_par(GameState state, int player, int corp) noexcept nogil:
    """Par an unparred corp: the President's Certificate at the lowest par must
    be affordable (Ruby prices it at ``min_price``) and within the limits."""
    return (corp_par_index(state, corp) < 0 and 2 * PAR_PRICE[0] <= player_cash(state, player)
            and player_can_gain(state, player, corp, 2))


cdef int sale_limit(GameState state, int player, int corp) noexcept nogil:
    """The largest block of ``corp`` the player may sell now; 0 for none. Each
    corp sells as one block per turn (MUST_SELL_IN_BLOCKS)."""
    if (turn_turn(state) <= 1 or turn_sr_sold_then_bought(state)
            or turn_sr_sold_mask(state) >> corp & 1):
        return 0
    return corp_max_sale(state, player, corp)


cdef bint _must_sell(GameState state, int player) noexcept nogil:
    """``must_sell?``: the player can sell, and is over the certificate limit or
    holds more than 60% of a corp they can sell outside the orange zone."""
    cdef bint can_sell = False
    cdef int c
    for c in range(<int>NUM_CORPS):
        if sale_limit(state, player, c) == 0:
            continue
        can_sell = True
        if not player_can_hold(state, player, c, 0):
            return True
    return can_sell and player_cert_count(state, player) > CERT_LIMIT[turn_num_players(state)]


cdef int _enumerate_stock(GameState state, uint16_t* ids) except -1 nogil:
    """Pass; buys from the IPO, from the pool, and pars, unless the turn has
    bought; then each corp the player may sell. A must-sell turn offers only
    the sales. Then DR's exchange and MF's port tile while the player may use
    them, whether or not Ruby asks (§12 item 4)."""
    cdef int player = turn_active_player(state)
    cdef int count = 0
    cdef int c
    if not _must_sell(state, player):
        ids[count] = STOCK_PASS
        count += 1
        if not turn_sr_bought(state):
            for c in range(<int>NUM_CORPS):
                if (corp_par_index(state, c) >= 0 and corp_ipo_shares(state, c) > 0
                        and _can_buy(state, player, c, corp_par_price(state, c))):
                    ids[count] = <uint16_t>(STOCK_IPO + c)
                    count += 1
            for c in range(<int>NUM_CORPS):
                if corp_pool_shares(state, c) > 0 and _can_buy(state, player, c, market_price(state, c)):
                    ids[count] = <uint16_t>(STOCK_POOL + c)
                    count += 1
            for c in range(<int>NUM_CORPS):
                if _can_par(state, player, c):
                    ids[count] = <uint16_t>(STOCK_PAR + c)
                    count += 1
    for c in range(<int>NUM_CORPS):
        if sale_limit(state, player, c):
            ids[count] = <uint16_t>(STOCK_SELL + c)
            count += 1
    count = _append_dr(state, ids, count, STOCK_DR)
    return _append_mf(state, ids, count, STOCK_MF)


cdef inline int _iota(uint16_t* ids, int count) noexcept nogil:
    """Write ids 0 to ``count`` - 1, a phase's indices, and return ``count``."""
    cdef int n
    for n in range(count):
        ids[n] = <uint16_t>n
    return count


cdef int _enumerate_sell_amount(GameState state, uint16_t* ids) noexcept nogil:
    """Each block of the pending corp the player may sell, from one share up."""
    cdef int corp = turn_pending_corp(state)
    return _iota(ids, sale_limit(state, turn_active_player(state), corp) if corp >= 0 else 0)


cdef int _enumerate_par(GameState state, uint16_t* ids) noexcept nogil:
    """The par prices whose President's Certificate (twice the par) the player can afford."""
    cdef int cash = player_cash(state, turn_active_player(state))
    cdef int count = 0
    cdef int k
    for k in range(<int>NUM_PAR_PRICES):
        if 2 * PAR_PRICE[k] <= cash:
            ids[count] = <uint16_t>k
            count += 1
    return count


# =============================================================================
# TRACK, TILE, TOKEN
# =============================================================================
#
# Ruby's Track and Token steps (step/tracker.rb, step/tokener.rb). A lay is
# factored (§5.4): TRACK picks a hex that has a candidate, and TILE an index
# into the hex's candidates, the upgrade table's canonical targets that pass
# Ruby's checks, sorted by (tile, rotation). With nothing to lay or token, the
# pass is forced.


cdef uint32_t _enterable_paths(GameState state, int hex) noexcept nogil:
    """The paths at the sides a walk could enter the hex by: those where the
    hex across has track facing it, through a side open to it."""
    cdef uint32_t paths = 0
    cdef int side, across, back
    for side in range(<int>NUM_SIDES):
        across = HEX_NEIGHBOR[hex][side]
        back = (side + 3) % <int>NUM_SIDES
        if across >= 0 and HEX_EXIT_MASK[across] >> back & 1 and hex_paths(state, across) & ENDPOINT_PAIRS[back]:
            paths |= ENDPOINT_PAIRS[side]
    return paths


cdef bint _semi_restrictive(GameState state, int corp, int hex, int old, uint32_t old_paths, int tile,
                            int rotation, uint32_t enterable) noexcept nogil:
    """The semi-restrictive rule (``check_track_restrictions!``, 1889's
    ``TRACK_RESTRICTION``): after the lay, the corp's walks must meet a path
    of the new tile that is new track, or that touches a revenue center whose
    value changed. A lay where there is no track passes.

    A lay that changes the value, with every path of the new tile touching
    the node, passes whenever its exits meet a side the corp reaches
    (``_lay_scan`` checks that first): the walks run as before the lay until
    they first enter the hex at a side the new tile has track on, and there
    they walk every path at that side. New track off the node fails when none
    of it ends where a walk could enter the hex (``enterable``). Any other
    lay is decided by walking with the tile laid (``graph_walks_new_track``),
    which checks only for new track. Two table facts make that complete,
    both asserted in ``tests/g1889/test_data_tables.py``: every path of a
    tile with a revenue center touches it, and every upgrade of a tile with
    track and a revenue center changes its value. So every lay on track at a
    revenue center takes the shortcut above, and the walk decides only lays
    where there is no revenue center, whose values cannot change."""
    cdef uint32_t paths = TILE_MASK[tile][rotation]
    cdef uint32_t added = paths & ~old_paths
    if old_paths == 0:
        return True
    if TILE_REVENUE[tile] != TILE_REVENUE[old] and paths & ~ENDPOINT_PAIRS[<int>NODE] == 0:
        return True
    if not (added & (ENDPOINT_PAIRS[<int>NODE] | enterable)):
        return False
    return graph_walks_new_track(state, corp, hex, tile, rotation)


cdef int _lay_scan(GameState state, int corp, int hex, LayCandidate* buf, bint first_only) noexcept nogil:
    """The corp's lay candidates at the hex, or with ``first_only`` whether
    there is one (0 or 1). Ruby's Track step offers a hex the corp reaches
    that no private blocks (``tracker_available_hex``), and the tiles in
    supply of the phase's colors that upgrade it (``upgradeable_tiles``) at
    rotations whose exits stay on open sides and meet a reached side
    (``legal_tile_rotation?``). The lay cost must be affordable, and the lay
    pass the semi-restrictive rule."""
    cdef int old = hex_tile(state, hex)
    cdef int k = old * <int>NUM_SIDES + hex_rotation(state, hex)
    cdef int count = 0
    cdef int sides, colors, i, tile, rotation, exits
    cdef uint32_t old_paths, enterable
    if UPG_START[k] == UPG_START[k + 1] or private_blocks_hex(state, hex):
        return 0
    sides = graph_sides(state, corp, hex)
    if sides == 0 or map_lay_cost(state, hex, corp) > corp_cash(state, corp):
        return 0
    colors = PHASE_TILE_COLORS[game_phase(state)]
    old_paths = hex_paths(state, hex)
    enterable = _enterable_paths(state, hex) if old_paths else 0
    for i in range(UPG_START[k], UPG_START[k + 1]):
        tile = UPG_TILE[i]
        rotation = UPG_ROT[i]
        exits = TILE_EXITS[tile][rotation]
        if (TILE_BLOCKS_LAY[tile] or not (colors >> TILE_COLOR[tile] & 1) or supply_count(state, tile) <= 0
                or exits & ~HEX_EXIT_MASK[hex] or not (exits & sides)
                or not _semi_restrictive(state, corp, hex, old, old_paths, tile, rotation, enterable)):
            continue
        if first_only:
            return 1
        if count == <int>MAX_LAY_CANDIDATES:
            fprintf(stderr, b"lay_candidates: hex %d has more than MAX_LAY_CANDIDATES %d\n",
                    hex, <int>MAX_LAY_CANDIDATES)
            abort()
        buf[count].tile = tile
        buf[count].rotation = rotation
        count += 1
    return count


cdef int lay_candidates(GameState state, int corp, int hex, LayCandidate* buf) noexcept nogil:
    """Write the corp's lay candidates at the hex to ``buf`` (MAX_LAY_CANDIDATES
    slots), sorted by (tile, rotation), and return the count."""
    return _lay_scan(state, corp, hex, buf, False)


cdef int enumerate_lay_candidates(GameState state, int hex, LayCandidate* buf) noexcept nogil:
    """The operating corp's lay candidates at the hex: TILE's ids index them."""
    return _lay_scan(state, turn_operating_corp(state), hex, buf, False)


cdef bint can_lay(GameState state, int corp, int hex) noexcept nogil:
    return _lay_scan(state, corp, hex, NULL, True) != 0


cdef bint can_token(GameState state, int corp, int hex) noexcept nogil:
    """Whether the corp may token the hex's city now (``Tokener``): it has a
    token left and can afford the next one's price (``City#tokenable?``), the
    city has a slot for it, and its walk reaches the city (``connected_nodes``;
    Ruby's ``reachable_hexes`` then holds too)."""
    return (corp_tokens_left(state, corp) > 0 and corp_token_price(state, corp) <= corp_cash(state, corp)
            and map_has_slot_for(state, hex, corp) and graph_node(state, corp, hex))


cdef int _enumerate_track(GameState state, uint16_t* ids) except -1 nogil:
    """Pass, then each hex with a lay candidate, by its slot (§12 item 8)."""
    cdef int corp = turn_operating_corp(state)
    cdef int count = 1
    cdef int h
    ids[0] = TRACK_PASS
    if corp < 0:
        return count
    for h in range(<int>NUM_HEXES):
        if can_lay(state, corp, h):
            assert HEX_TRACK_SLOT[h] >= 0, f"_enumerate_track: a lay on hex {h}, which has no slot"
            ids[count] = <uint16_t>(TRACK_HEX + HEX_TRACK_SLOT[h])
            count += 1
    return count


cdef int _enumerate_tile(GameState state, uint16_t* ids) noexcept nogil:
    """An index per candidate at the pending hex."""
    cdef LayCandidate candidates[15]  # MAX_LAY_CANDIDATES
    cdef int hex = turn_pending_hex(state)
    if hex < 0 or turn_operating_corp(state) < 0:
        return 0
    return _iota(ids, enumerate_lay_candidates(state, hex, candidates))


cdef int _enumerate_token(GameState state, uint16_t* ids) except -1 nogil:
    """Pass, then each city the corp may token, by its slot (§12 item 8)."""
    cdef int corp = turn_operating_corp(state)
    cdef int count = 1
    cdef int h
    ids[0] = TOKEN_PASS
    if corp < 0:
        return count
    for h in range(<int>NUM_HEXES):
        if can_token(state, corp, h):
            assert HEX_TOKEN_SLOT[h] >= 0, f"_enumerate_token: a token on hex {h}, which has no slot"
            ids[count] = <uint16_t>(TOKEN_HEX + HEX_TOKEN_SLOT[h])
            count += 1
    return count


# =============================================================================
# DIVIDEND, BUY_TRAIN, SELECT_TRAIN, TRAIN_PRICE, EMERGENCY_SELL_AMOUNT, DISCARD
# =============================================================================
#
# Ruby's Dividend, BuyTrain, and DiscardTrain steps (step/dividend.rb,
# step/buy_train.rb, step/train.rb, step/discard_train.rb). Ruby skips Dividend
# at zero revenue with a withhold, so DIVIDEND then offers only that, and the
# driver applies it. BUY_TRAIN buys one train at a time until the corp passes: it picks the source (the
# depot, the pool, or another corp), and SELECT_TRAIN the train; a buy from
# another corp names its price in TRAIN_PRICE. A must-buy the treasury cannot
# fund is an emergency (step/emergency_money.rb, step/bankrupt.rb): the
# president's cash, their share sales (BUY_TRAIN picks the corp and
# EMERGENCY_SELL_AMOUNT the block), and bankruptcy.


cdef bint must_buy_train(GameState state, int corp) noexcept nogil:
    """``must_buy_train?`` (``MUST_BUY_TRAIN = :route``): the corp has no train
    and some token's walk reaches two revenue centers (GRAPH's
    ``route_available``, computed fresh, as the extractor's oracle is; Ruby's
    own graph can keep a stale true, engine-design.md §4). Ruby also needs a
    train in the depot or pool, and the unlimited D keeps the depot stocked."""
    return corp_train_total(state, corp) == 0 and graph_route_available(state, corp)


cdef int _enumerate_dividend(GameState state, uint16_t* ids) noexcept nogil:
    """Withhold, then pay out when there is revenue (1889 has no half pay)."""
    ids[0] = DIVIDEND_WITHHOLD
    if turn_route_revenue(state) == 0:
        return 1
    ids[1] = DIVIDEND_PAYOUT
    return 2


cdef int _enumerate_buy_train(GameState state, uint16_t* ids) noexcept nogil:
    """Pass unless the corp must buy; each source with a train to offer
    (``_source_trains``); and in an emergency each corp whose shares the
    president may sell, then bankruptcy (``_append_emergency``)."""
    cdef int corp = turn_operating_corp(state)
    cdef bint must_buy, emergency
    cdef int count = 0
    cdef int s
    if corp < 0:
        ids[0] = BUY_TRAIN_PASS
        return 1
    must_buy = must_buy_train(state, corp)
    emergency = must_buy and corp_cash(state, corp) < trains_cheapest_price(state)
    if not must_buy:
        ids[0] = BUY_TRAIN_PASS
        count = 1
    for s in range(<int>NUM_TRAIN_SOURCES):
        if _source_trains(state, corp, s, emergency, NULL):
            ids[count] = <uint16_t>(BUY_TRAIN_DEPOT + s)
            count += 1
    if emergency:
        count = _append_emergency(state, corp, ids, count)
    return count


cdef inline int _put(uint16_t* ids, int count, int action_id) noexcept nogil:
    if ids != NULL:
        ids[count] = <uint16_t>action_id
    return count + 1


cdef int _source_trains(GameState state, int corp, int source, bint emergency, uint16_t* ids) noexcept nogil:
    """SELECT_TRAIN's ids for a purchase by ``corp`` from ``source``, or with
    ``ids`` NULL their count. With room under the phase's limit (``room?``),
    each depot type its treasury affords (``depot_trains``), each pool type
    it affords, and with ¥1 or more every train of another corp
    (``other_trains``), priced in TRAIN_PRICE (``spend_minmax``: ¥1 to the
    treasury); and room or not, each 4, 5, or 6 it may trade with ¥800 for an
    available D, from the depot (``discountable_trains_for``). Ruby's
    ``buy_train_action`` checks no room, but at the limit its BuyTrain step
    opens only for a trade-in (``can_buy_train?``), and the UI then offers
    only that. In an emergency (§6.2; 1889's EBUY_* constants are Ruby's
    defaults but EBUY_PRES_SWAP = false and EBUY_FROM_OTHERS = :never):

    - only the cheapest depot or pool train (``min_depot_train``, the
      depot's on a tie), while the treasury and the president's cash cover
      it: the president pays the rest (``buy_train_action``), toward that
      train only (``check_for_cheapest_train``). A depot buy may change the
      phase;
    - every train of another corp, for ¥1 to the treasury (``spend_minmax``),
      while the treasury holds ¥1 and the president has sold nothing since
      the turn's last purchase (``last_share_sold_price``)."""
    cdef int cash = corp_cash(state, corp)
    cdef int seller = source - <int>TRAIN_SOURCE_CORP
    cdef bint room = corp_train_total(state, corp) < trains_limit(state)
    cdef int count = 0
    cdef int types, t, i
    if seller >= 0:
        if seller == corp or cash <= 0 or (turn_ebuy_sold_any(state) if emergency else not room):
            return 0
        for t in range(<int>NUM_TRAIN_TYPES):
            if corp_train_count(state, seller, t):
                count = _put(ids, count, SELECT_TRAIN_TYPE + t)
        return count
    if emergency:
        if _emergency_need(state, corp) > 0 or trains_cheapest_in_pool(state) != (source == TRAIN_SOURCE_POOL):
            return 0
        return _put(ids, 0, SELECT_TRAIN_TYPE + trains_cheapest_type(state))
    if source == TRAIN_SOURCE_POOL:
        for t in range(<int>NUM_TRAIN_TYPES):
            if room and pool_count(state, t) and TRAIN_PRICE[t] <= cash:
                count = _put(ids, count, SELECT_TRAIN_TYPE + t)
        return count
    types = trains_depot_types(state)
    for t in range(<int>NUM_TRAIN_TYPES):
        if room and types >> t & 1 and TRAIN_PRICE[t] <= cash:
            count = _put(ids, count, SELECT_TRAIN_TYPE + t)
    if types >> <int>D_TRAIN & 1:
        for i in range(<int>NUM_TRADE_IN_TYPES):
            t = TRADE_IN_TYPE[i]
            if corp_train_count(state, corp, t) and trade_in_price(t) <= cash:
                count = _put(ids, count, SELECT_TRAIN_TRADE_IN + i)
    return count


cdef int _enumerate_select_train(GameState state, uint16_t* ids) noexcept nogil:
    """The trains of the pending source (``_source_trains``)."""
    cdef int corp = turn_operating_corp(state)
    cdef int source = turn_pending_train_source(state)
    if corp < 0 or source < 0:
        return 0
    return _source_trains(state, corp, source, in_emergency(state, corp), ids)


# Emergency money (§6.2). The corp must buy, and its treasury is short of
# the cheapest depot or pool train (``buyable_trains`` with ``entity.cash <
# min_depot_price``). What the president must still raise is the need: that
# train's price less the treasury and their cash (``needed_cash`` -
# ``available_cash``). Ruby's BuyTrain also takes the president's share sales
# outside a must-buy, and its Bankrupt step a bankruptcy at any step of the
# turn, whenever the president's cash and the treasury fall short of the
# cheapest train; the UI offers both only in an emergency, as the engine does.


cdef inline int _emergency_need(GameState state, int corp) noexcept nogil:
    return trains_cheapest_price(state) - corp_cash(state, corp) - player_cash(state, corp_president(state, corp))


cdef bint in_emergency(GameState state, int corp) noexcept nogil:
    """Whether ``corp`` must buy a train its treasury cannot fund."""
    return must_buy_train(state, corp) and corp_cash(state, corp) < trains_cheapest_price(state)


cdef bint can_go_bankrupt(GameState state, int corp) noexcept nogil:
    """``can_go_bankrupt?``: the treasury, the president's cash, and every
    block they may still sell (``liquidity`` with ``emergency``, through
    BuyTrain's ``can_sell?``, each corp once) fall short of the cheapest
    train. Each corp's largest block counts in full, as every smaller one
    falls short then (``selling_minimum_shares?``)."""
    cdef int president = corp_president(state, corp)
    cdef int need = _emergency_need(state, corp)
    cdef int sold = turn_ebuy_sold_mask(state)
    cdef int c
    for c in range(<int>NUM_CORPS):
        if not (sold >> c & 1):
            need -= corp_emergency_sale(state, president, c) * market_price(state, c)
    return need > 0


cdef int _emergency_sale_limit(GameState state, int corp, int seller, int need) noexcept nogil:
    """The largest block of ``seller``'s shares the president of ``corp`` may
    sell toward ``need`` (``can_sell?``; every smaller block is legal too):
    while the need is unmet, of each corp at most once this turn
    (``MUST_SELL_IN_BLOCKS``), within ``corp_emergency_sale`` (the pool's
    room; no presidency changes), and only while the next smaller block
    falls short (``EBUY_SELL_MORE_THAN_NEEDED = false``). The operating
    corp's own shares are sellable too."""
    cdef int most
    if need <= 0 or turn_ebuy_sold_mask(state) >> seller & 1:
        return 0
    most = corp_emergency_sale(state, corp_president(state, corp), seller)
    return min(most, (need - 1) // market_price(state, seller) + 1) if most else 0


cdef int _append_emergency(GameState state, int corp, uint16_t* ids, int count) noexcept nogil:
    """An emergency's other choices, after the trains: each corp whose shares
    the president may sell (``_emergency_sale_limit``), then bankruptcy when
    even all of that falls short (``can_go_bankrupt``)."""
    cdef int need = _emergency_need(state, corp)
    cdef int c
    for c in range(<int>NUM_CORPS):
        if _emergency_sale_limit(state, corp, c, need):
            ids[count] = <uint16_t>(BUY_TRAIN_SELL + c)
            count += 1
    if can_go_bankrupt(state, corp):
        ids[count] = BUY_TRAIN_BANKRUPT
        count += 1
    return count


cdef int _enumerate_emergency_sell_amount(GameState state, uint16_t* ids) noexcept nogil:
    """Each block of the pending corp the president may sell, from one share up."""
    cdef int corp = turn_operating_corp(state)
    cdef int seller = turn_pending_corp(state)
    if corp < 0 or seller < 0:
        return 0
    return _iota(ids, _emergency_sale_limit(state, corp, seller, _emergency_need(state, corp)))


cdef int _filter_private_buys(GameState state, uint16_t* ids, int count, int first) noexcept nogil:
    """``private_sale_same_president`` (§7): the corp buys only privates its
    own president owns (slots ``first`` + p)."""
    cdef int owner, i, id_
    cdef int kept = 0
    if not state.private_sale_same_president:
        return count
    owner = <int>OWNER_PLAYER + corp_president(state, turn_operating_corp(state))
    for i in range(count):
        id_ = ids[i]
        if first <= id_ < first + <int>NUM_PRIVATES and private_owner(state, id_ - first) != owner:
            continue
        ids[kept] = ids[i]
        kept += 1
    return kept


cdef int _filter_buy_train(GameState state, uint16_t* ids, int count) noexcept nogil:
    """``train_sale_same_president`` (§7): only corps with the buyer's
    president are sources."""
    cdef int president, i, id_
    cdef int kept = 0
    if not state.train_sale_same_president:
        return count
    president = corp_president(state, turn_operating_corp(state))
    for i in range(count):
        id_ = ids[i]
        if BUY_TRAIN_CORP <= id_ < BUY_TRAIN_SELL and corp_president(state, id_ - BUY_TRAIN_CORP) != president:
            continue
        ids[kept] = ids[i]
        kept += 1
    return kept


cdef inline void _offer_points(int* points, int count, int low, int high) noexcept nogil:
    """Set to -1, not offered, each grid point outside ``low`` to ``high`` or
    equal to an earlier one (§5.2)."""
    cdef int k, j
    for k in range(count):
        if not low <= points[k] <= high:
            points[k] = -1
        for j in range(k):
            if points[j] == points[k]:
                points[k] = -1


cdef inline int _train_price_cash(GameState state) noexcept nogil:
    """The most TRAIN_PRICE may pay: the buyer's treasury, 0 without a pending train."""
    cdef int corp = turn_operating_corp(state)
    return corp_cash(state, corp) if corp >= 0 and turn_pending_train_type(state) >= 0 else 0


cdef void _train_price_points(GameState state, int* points) noexcept nogil:
    """TRAIN_PRICE's grid for the pending train (§5.2): ¥1, its list price,
    ⌈cash/3⌉, ⌈2·cash/3⌉, and the buyer's cash. A point outside ¥1 to the
    cash (``spend_minmax``), or equal to an earlier one, is -1: not offered."""
    cdef int cash = _train_price_cash(state)
    cdef int train = turn_pending_train_type(state)
    points[0] = 1
    points[1] = TRAIN_PRICE[train] if train >= 0 else 0
    points[2] = (cash + 2) // 3
    points[3] = (2 * cash + 2) // 3
    points[4] = cash
    _offer_points(points, <int>TRAIN_PRICE_POINTS, 1, cash)


cdef int _enumerate_train_price(GameState state, uint16_t* ids) noexcept nogil:
    """The grid points offered."""
    cdef int points[5]  # TRAIN_PRICE_POINTS
    cdef int count = 0
    cdef int k
    _train_price_points(state, points)
    for k in range(<int>TRAIN_PRICE_POINTS):
        if points[k] > 0:
            ids[count] = <uint16_t>k
            count += 1
    return count


cdef int _enumerate_discard(GameState state, uint16_t* ids) except -1 nogil:
    """Each train type of the first crowded corp (``DiscardTrain``), whose
    president chooses, by its slot (§12 item 8)."""
    cdef int corp = trains_first_crowded(state)
    cdef int count = 0
    cdef int t
    if corp < 0:
        return 0
    for t in range(<int>NUM_TRAIN_TYPES):
        if corp_train_count(state, corp, t):
            assert TRAIN_DISCARD_SLOT[t] >= 0, f"_enumerate_discard: corp {corp} holds type {t}, which has no slot"
            ids[count] = <uint16_t>TRAIN_DISCARD_SLOT[t]
            count += 1
    return count


# =============================================================================
# INTERJECTIONS: BUY_PRIVATE_PRE, BUY_PRIVATE_POST, PRIVATE_PRICE, ER_BONUS, PORT, DR
# =============================================================================
#
# Ruby's BuyCompany, Exchange, and SpecialTrack steps are non-blocking: their
# actions join whatever step Ruby stops at (``Round::Base#actions_for``). The
# action space offers them in three places only (§5.1), and replay takes a
# human's others through ``DRIVER.override_private_slot``:
#
# - BUY_PRIVATE_PRE opens the corp's turn when any slot besides "done" is
#   legal: each private the corp may buy, DR, and MF.
# - BUY_PRIVATE_POST is Ruby's blocking BuyCompany after BuyTrain, in phases
#   3-4: "done", each private, and with a private to buy, DR. Ruby skips it
#   with nothing to do (``skip_steps``), and "done" is then forced.
# - STOCK offers DR and MF on every turn of their owner (§12 item 4): Ruby's
#   player may use them during another player's turn too (DR's exchange is
#   ``when: 'any'``), which the engine does not offer, so a turn Ruby skips or
#   ends with nothing else to do is the owner's one chance. The OR loses
#   nothing that way, as a president with a use for either gets PRE.
#
# The slots:
#
# - Buy private p (phases 3-4, the operating corp's treasury): PRIVATE_PRICE
#   names the price, then ER_BONUS follows a sale of ER, and the step resumes
#   (``return_step``).
# - DR: the active player exchanges DR for an IR share in place.
# - MF: PORT lays the port tile, then the step resumes. Ruby allows it at the
#   owner's SR turn and in an OR at the Track step of a corp the owner is
#   president of, or before any corp's turn has acted; the engine offers the
#   owner's own turns only (§12 item 4), at STOCK and at PRE, before the lay.
#   Ruby keeps Track open after the lay for it (§10.2), which the engine
#   does not offer.
#
# Every 1889 private and its abilities are open in phases 2-4 only, so none
# of this reaches phase 5.


cdef inline int _private_price_low(int private) noexcept nogil:
    """``Company#min_price``: half the face value, rounded up."""
    return (PRIVATE_VALUE[private] + 1) // 2


cdef bint can_buy_private(GameState state, int corp, int private) noexcept nogil:
    """Whether ``corp`` may buy ``private`` now (``BuyCompany#can_buy_company?``
    and ``buy_company``): in phases 3-4 (``can_buy_companies``), from a player
    (``purchasable_companies``; ``company_sellable`` keeps a corp's), for at
    least ⌈face/2⌉ (``min_price``) from its treasury. 1889 has no ``no_buy``
    private outside the beginner game, and Ruby checks no consent."""
    cdef int owner = private_owner(state, private)
    return (PHASE_CAN_BUY_PRIVATES[game_phase(state)] and <int>OWNER_PLAYER <= owner < <int>OWNER_CORP
            and _private_price_low(private) <= corp_cash(state, corp))


cdef int _append_private_buys(GameState state, uint16_t* ids, int count, int first) noexcept nogil:
    """Append the operating corp's buy-private slots, ``first`` + p."""
    cdef int corp = turn_operating_corp(state)
    cdef int p
    if corp < 0:
        return count
    for p in range(<int>NUM_PRIVATES):
        if can_buy_private(state, corp, p):
            ids[count] = <uint16_t>(first + p)
            count += 1
    return count


cdef bint can_exchange_dr(GameState state, int player) noexcept nogil:
    """Whether ``player`` may exchange DR for an IR share now
    (``Exchange#can_exchange?``; the ability's ``when: any`` sets no time):
    they own DR, IR's IPO holds a share besides the President's Certificate
    (``available_share``; IR need not be parred), and the share keeps them
    within 60% of IR outside the orange zone (``holding_ok?``). An exchange
    skips the certificate limit (``can_gain?`` with ``exchange``): a
    certificate goes for the one that comes."""
    cdef int corp = <int>DR_EXCHANGE_CORP
    if private_owner(state, <int>PRIVATE_DR) != <int>OWNER_PLAYER + player:
        return False
    if corp_ipo_shares(state, corp) <= (0 if corp_par_index(state, corp) >= 0 else <int>PRESIDENT_SHARES):
        return False
    return player_can_hold(state, player, corp, 1)


cdef int _append_dr(GameState state, uint16_t* ids, int count, int dr_id) noexcept nogil:
    if can_exchange_dr(state, turn_active_player(state)):
        ids[count] = <uint16_t>dr_id
        count += 1
    return count


cdef int port_lays(GameState state, uint16_t* ids) except -1 nogil:
    """PORT's ids (§12 item 8), or with ``ids`` NULL their count:
    while a player owns MF and its one lay is unused, the port tile, in supply,
    on each port hex it upgrades (only an empty port town: the tile is yellow)
    at each canonical rotation whose exits stay on open sides (Ruby's
    SpecialTrack: ``potential_tiles``, ``legal_tile_rotation?`` with every
    neighbor reachable, as the ability needs no route; the port hexes cost
    nothing, and ``check_track_restrictions!`` binds only operators)."""
    cdef int owner = private_owner(state, <int>PRIVATE_MF)
    cdef int count = 0
    cdef int i, j, h, k, rotation
    if (not (<int>OWNER_PLAYER <= owner < <int>OWNER_CORP) or private_flags(state, <int>PRIVATE_MF) & <int>MF_PORT_USED
            or supply_count(state, <int>PORT_TILE) <= 0
            or not (PHASE_TILE_COLORS[game_phase(state)] >> TILE_COLOR[<int>PORT_TILE] & 1)):
        return 0
    for i in range(<int>NUM_PORTS):
        h = MF_PORT_HEX[i]
        k = hex_tile(state, h) * <int>NUM_SIDES + hex_rotation(state, h)
        for j in range(UPG_START[k], UPG_START[k + 1]):
            rotation = UPG_ROT[j]
            if UPG_TILE[j] != <int>PORT_TILE or TILE_EXITS[<int>PORT_TILE][rotation] & ~HEX_EXIT_MASK[h]:
                continue
            assert PORT_LAY_SLOT[i][rotation] >= 0, f"port_lays: port hex {i} at rotation {rotation} has no slot"
            if ids != NULL:
                ids[count] = <uint16_t>(PORT_LAY + PORT_LAY_SLOT[i][rotation])
            count += 1
    return count


cdef int _append_mf(GameState state, uint16_t* ids, int count, int mf_id) except -1 nogil:
    """The MF slot, for MF's owner when PORT has a lay."""
    if (private_owner(state, <int>PRIVATE_MF) == <int>OWNER_PLAYER + turn_active_player(state)
            and port_lays(state, NULL)):
        ids[count] = <uint16_t>mf_id
        count += 1
    return count


cdef int er_candidates(GameState state, LayCandidate* buf) noexcept nogil:
    """ER_BONUS's lays on C4, sorted like TILE's by (tile, rotation): the
    upgrade table's targets of C4's tile among ER's green city tiles
    (``ER_TILE``), in supply, of a phase color, with exits on open sides
    (Ruby's SpecialTrack for ER's ``sold`` ability: ``potential_tiles``,
    ``legal_tile_rotation?`` with every neighbor reachable, as the ability
    needs no route). A private's lay skips ``upgrades_to?``'s label and city
    checks, which no ER tile needs, and ``check_track_restrictions!``, which
    binds only operators. C4 costs nothing to lay on."""
    cdef int h = <int>ER_HEX
    cdef int k = hex_tile(state, h) * <int>NUM_SIDES + hex_rotation(state, h)
    cdef int colors = PHASE_TILE_COLORS[game_phase(state)]
    cdef int count = 0
    cdef int i, j, tile, rotation
    cdef bint listed
    for i in range(UPG_START[k], UPG_START[k + 1]):
        tile = UPG_TILE[i]
        rotation = UPG_ROT[i]
        listed = False
        for j in range(<int>NUM_ER_TILES):
            listed = listed or ER_TILE[j] == tile
        if (not listed or supply_count(state, tile) <= 0 or not (colors >> TILE_COLOR[tile] & 1)
                or TILE_EXITS[tile][rotation] & ~HEX_EXIT_MASK[h]):
            continue
        if count == <int>MAX_LAY_CANDIDATES:
            fprintf(stderr, b"er_candidates: more than MAX_LAY_CANDIDATES %d\n", <int>MAX_LAY_CANDIDATES)
            abort()
        buf[count].tile = tile
        buf[count].rotation = rotation
        count += 1
    return count


cdef int _enumerate_buy_private_pre(GameState state, uint16_t* ids) except -1 nogil:
    """The turn's opening window: done, each private the corp may buy, DR, and MF."""
    cdef int count = _append_private_buys(state, ids, 1, BUY_PRIVATE_PRE_PRIVATE)
    ids[0] = BUY_PRIVATE_PRE_DONE
    count = _append_dr(state, ids, count, BUY_PRIVATE_PRE_DR)
    return _append_mf(state, ids, count, BUY_PRIVATE_PRE_MF)


cdef bint buy_private_pre_open(GameState state) except -1 nogil:
    """Whether the operating corp's turn opens at BUY_PRIVATE_PRE: some slot
    besides "done" is engine-legal, whatever the policy filter leaves."""
    cdef uint16_t ids[10]  # ACTION_SIZE_BUY_PRIVATE_PRE
    return _enumerate_buy_private_pre(state, ids) > 1


cdef bint override_slot_legal(GameState state, int action_id) except -1 nogil:
    """Whether replay may take BUY_PRIVATE_PRE's slot ``action_id``, not "done",
    at this step: one of the operating corp's decisions that offers no such
    slot, where the same slot would be legal at PRE. The sub-phases sit inside
    one Ruby action, and PRE and POST have their own slots."""
    cdef uint16_t ids[10]  # ACTION_SIZE_BUY_PRIVATE_PRE
    cdef int step = turn_step(state)
    cdef int count, i
    if turn_round(state) != ROUND_OPERATING or not (
            step == STEP_TRACK or step == STEP_TOKEN or step == STEP_DIVIDEND or step == STEP_BUY_TRAIN):
        return False
    count = _enumerate_buy_private_pre(state, ids)
    for i in range(1, count):
        if ids[i] == action_id:
            return True
    return False


cdef int _enumerate_buy_private_post(GameState state, uint16_t* ids) noexcept nogil:
    """Ruby's blocking BuyCompany: done, each private the corp may buy, and,
    with one to buy, where Ruby asks, DR."""
    cdef int count = _append_private_buys(state, ids, 1, BUY_PRIVATE_POST_PRIVATE)
    ids[0] = BUY_PRIVATE_POST_DONE
    if count > 1:
        count = _append_dr(state, ids, count, BUY_PRIVATE_POST_DR)
    return count


cdef void _private_price_range(GameState state, int* low, int* high) noexcept nogil:
    """PRIVATE_PRICE's legal range for the pending private: ⌈face/2⌉ to
    2·face (``Company#min_price``, ``max_price``) and the buyer's treasury
    (``spend``); empty (``high`` < ``low``) without a pending private."""
    cdef int p = turn_pending_private(state)
    cdef int corp = turn_operating_corp(state)
    if p < 0 or corp < 0:
        low[0] = 1
        high[0] = 0
        return
    low[0] = _private_price_low(p)
    high[0] = min(2 * PRIVATE_VALUE[p], corp_cash(state, corp))


cdef void _private_price_points(GameState state, int* points) noexcept nogil:
    """PRIVATE_PRICE's grid (§5.2): ⌈face/2⌉, face, 1½·face, and the most
    the corp may pay. A point outside the legal range, or equal to an earlier
    one, is -1: not offered."""
    cdef int p = turn_pending_private(state)
    cdef int face = PRIVATE_VALUE[p] if p >= 0 else 0
    cdef int low, high
    _private_price_range(state, &low, &high)
    points[0] = low
    points[1] = face
    points[2] = (3 * face + 1) // 2
    points[3] = high
    _offer_points(points, <int>PRIVATE_PRICE_POINTS, low, high)


cdef int _enumerate_private_price(GameState state, uint16_t* ids) noexcept nogil:
    """The grid points offered."""
    cdef int points[4]  # PRIVATE_PRICE_POINTS
    cdef int count = 0
    cdef int k
    _private_price_points(state, points)
    for k in range(<int>PRIVATE_PRICE_POINTS):
        if points[k] > 0:
            ids[count] = <uint16_t>k
            count += 1
    return count


cdef int _enumerate_er_bonus(GameState state, uint16_t* ids) except -1 nogil:
    """Pass, which Ruby always offers (``TrackLayWhenCompanySold``), then each
    candidate on C4, whose printed tile bounds them (§12 item 8)."""
    cdef LayCandidate candidates[15]  # MAX_LAY_CANDIDATES
    cdef int count = er_candidates(state, candidates)
    cdef int k
    assert count <= <int>NUM_ER_CANDIDATES, f"_enumerate_er_bonus: {count} candidates on C4"
    ids[0] = ER_BONUS_PASS
    for k in range(count):
        ids[1 + k] = <uint16_t>(ER_BONUS_CANDIDATE + k)
    return 1 + count


cdef int enumerate_legal_actions(GameState state, uint16_t* action_ids) except -1 nogil:
    """Write the engine-legal ids of the current decision in a fixed order and
    return the count, 0 at an automated step. The buffer holds MAX_ACTION_SIZE."""
    cdef int phase = get_decision_phase(state)
    cdef int count
    if phase < 0:
        return 0
    if phase == DPHASE_AUCTION:
        count = _enumerate_auction(state, action_ids)
    elif phase == DPHASE_BID:
        count = _enumerate_bid(state, action_ids)
    elif phase == DPHASE_STOCK:
        count = _enumerate_stock(state, action_ids)
    elif phase == DPHASE_PAR:
        count = _enumerate_par(state, action_ids)
    elif phase == DPHASE_SELL_AMOUNT:
        count = _enumerate_sell_amount(state, action_ids)
    elif phase == DPHASE_BUY_PRIVATE_PRE:
        count = _enumerate_buy_private_pre(state, action_ids)
    elif phase == DPHASE_TRACK:
        count = _enumerate_track(state, action_ids)
    elif phase == DPHASE_TILE:
        count = _enumerate_tile(state, action_ids)
    elif phase == DPHASE_TOKEN:
        count = _enumerate_token(state, action_ids)
    elif phase == DPHASE_DIVIDEND:
        count = _enumerate_dividend(state, action_ids)
    elif phase == DPHASE_BUY_TRAIN:
        count = _enumerate_buy_train(state, action_ids)
    elif phase == DPHASE_SELECT_TRAIN:
        count = _enumerate_select_train(state, action_ids)
    elif phase == DPHASE_TRAIN_PRICE:
        count = _enumerate_train_price(state, action_ids)
    elif phase == DPHASE_EMERGENCY_SELL_AMOUNT:
        count = _enumerate_emergency_sell_amount(state, action_ids)
    elif phase == DPHASE_DISCARD:
        count = _enumerate_discard(state, action_ids)
    elif phase == DPHASE_BUY_PRIVATE_POST:
        count = _enumerate_buy_private_post(state, action_ids)
    elif phase == DPHASE_PRIVATE_PRICE:
        count = _enumerate_private_price(state, action_ids)
    elif phase == DPHASE_ER_BONUS:
        count = _enumerate_er_bonus(state, action_ids)
    else:  # DPHASE_PORT
        count = port_lays(state, action_ids)
    if count > <int>MAX_ACTION_SIZE:
        fprintf(stderr, b"enumerate_legal_actions: %d ids exceed MAX_ACTION_SIZE %d\n",
                count, <int>MAX_ACTION_SIZE)
        abort()
    return count


cdef int enumerate_policy_actions(GameState state, uint16_t* action_ids) except -1 nogil:
    """The legal ids minus the flag restrictions and permanent filters (§5.1, §7)."""
    cdef int count = enumerate_legal_actions(state, action_ids)
    cdef int phase
    if count == 0:
        return 0
    phase = get_decision_phase(state)
    if phase == DPHASE_AUCTION:
        return _filter_auction(state, action_ids, count)
    if phase == DPHASE_BID:
        return _filter_bid(state, count)
    if phase == DPHASE_BUY_PRIVATE_PRE:
        return _filter_private_buys(state, action_ids, count, BUY_PRIVATE_PRE_PRIVATE)
    if phase == DPHASE_BUY_TRAIN:
        return _filter_buy_train(state, action_ids, count)
    if phase == DPHASE_BUY_PRIVATE_POST:
        return _filter_private_buys(state, action_ids, count, BUY_PRIVATE_POST_PRIVATE)
    return count  # no flag restricts the others


cdef bint is_legal_action(GameState state, int action_id) except -1 nogil:
    """Engine legality: a scan of the enumerated ids, or a range check for an
    exact amount. False at an automated step."""
    cdef uint16_t ids[46]  # MAX_ACTION_SIZE
    cdef int phase = get_decision_phase(state)
    cdef int count, i
    if phase < 0 or action_id < 0:
        return False
    if action_id >= PHASE_ACTION_SIZE[phase]:
        if not has_exact_amounts(phase) or action_id > MAX_ACTION_ID:
            return False
        return _amount_in_range(state, phase, action_id - PHASE_ACTION_SIZE[phase])
    count = enumerate_legal_actions(state, ids)
    for i in range(count):
        if ids[i] == action_id:
            return True
    return False


cdef bint wide_amount_range(GameState state, int phase) except -1 nogil:
    """Whether the current amount decision's legal range holds more than one
    amount, so exact amounts may lie beyond its grid points."""
    cdef int p, low, high
    if phase == DPHASE_BID:
        p = _bid_private(state)
        return player_available(state, turn_active_player(state), p) > private_min_bid(state, p)
    if phase == DPHASE_TRAIN_PRICE:
        return _train_price_cash(state) > 1
    if phase == DPHASE_PRIVATE_PRICE:
        _private_price_range(state, &low, &high)
        return high > low
    assert False, f"wide_amount_range: {_PHASE_NAMES[phase]} takes no exact amounts"
    return False


# =============================================================================
# PYTHON WRAPPERS
# =============================================================================

cdef int _check_id(int phase, int action_id) except -1:
    assert 0 <= phase < <int>NUM_DECISION_PHASES, f"invalid decision phase {phase}"
    assert 0 <= action_id <= (MAX_ACTION_ID if has_exact_amounts(phase) else PHASE_ACTION_SIZE[phase] - 1), \
        f"action id {action_id} out of range for {_PHASE_NAMES[phase]} (size {PHASE_ACTION_SIZE[phase]})"
    return 0


cpdef object decode_action_py(int phase, int action_id):
    _check_id(phase, action_id)
    cdef ActionInfo info = decode_action(phase, action_id)
    return ActionInfoTuple(info.phase, info.type, info.corp, info.private, info.hex, info.train_type,
                           info.count, info.amount)


cpdef int encode_action_py(object info) except -1:
    """Encode any object with ActionInfoTuple's fields."""
    cdef ActionInfo c_info
    assert 0 <= info.phase < <int>NUM_DECISION_PHASES, f"invalid decision phase {info.phase}"
    c_info.phase = info.phase
    c_info.type = info.type
    c_info.corp = info.corp
    c_info.private = info.private
    c_info.hex = info.hex
    c_info.train_type = info.train_type
    c_info.count = info.count
    c_info.amount = info.amount
    return encode_action(c_info)


cpdef int get_decision_phase_py(GameState state):
    return get_decision_phase(state)


cdef uint16_t* _id_buffer(cnp.ndarray action_ids) except NULL:
    assert action_ids.dtype == np.uint16 and action_ids.ndim == 1 and action_ids.flags['C_CONTIGUOUS'], \
        "action_ids must be a contiguous uint16 vector"
    assert action_ids.shape[0] >= <int>MAX_ACTION_SIZE, f"action_ids needs {MAX_ACTION_SIZE} slots"
    return <uint16_t*>cnp.PyArray_DATA(action_ids)


cpdef int enumerate_legal_actions_py(GameState state, cnp.ndarray action_ids) except -1:
    """Fill a uint16 buffer of at least MAX_ACTION_SIZE with the legal ids; return the count."""
    return enumerate_legal_actions(state, _id_buffer(action_ids))


cpdef int enumerate_policy_actions_py(GameState state, cnp.ndarray action_ids) except -1:
    """Fill a uint16 buffer of at least MAX_ACTION_SIZE with the policy ids; return the count."""
    return enumerate_policy_actions(state, _id_buffer(action_ids))


cpdef bint is_legal_action_py(GameState state, int action_id) except -1:
    return is_legal_action(state, action_id)


cpdef int price_amount_py(GameState state, int action_id) except -1:
    """The yen of ``action_id`` (a grid point or an exact amount) at the current
    BID, TRAIN_PRICE, or PRIVATE_PRICE decision."""
    cdef int phase = get_decision_phase(state)
    assert phase >= 0 and has_exact_amounts(phase), \
        f"not at a BID, TRAIN_PRICE, or PRIVATE_PRICE decision (phase {phase})"
    _check_id(phase, action_id)
    cdef ActionInfo info = decode_action(phase, action_id)
    assert info.type != ACTION_PASS, "BID's drop has no amount"
    return price_amount(state, &info)


cdef int _corp_or_operating(GameState state, int corp) except -2:
    if corp < 0:
        corp = turn_operating_corp(state)
    assert 0 <= corp < <int>NUM_CORPS, f"corp {corp} out of range (no operating corp?)"
    return corp


cpdef list lay_candidates_py(GameState state, int hex, int corp=-1):
    """``(tile, rotation)`` of each lay candidate at ``hex``, in TILE's order,
    for ``corp`` (the operating corp by default)."""
    cdef LayCandidate buf[15]  # MAX_LAY_CANDIDATES
    assert 0 <= hex < <int>NUM_HEXES, f"hex {hex} out of range"
    cdef int count = lay_candidates(state, _corp_or_operating(state, corp), hex, buf)
    return [(buf[k].tile, buf[k].rotation) for k in range(count)]


cpdef list lay_hexes_py(GameState state, int corp=-1):
    """The hexes ``corp`` (the operating corp by default) may lay on: TRACK's hexes."""
    corp = _corp_or_operating(state, corp)
    return [h for h in range(<int>NUM_HEXES) if can_lay(state, corp, h)]


cpdef bint must_buy_train_py(GameState state, int corp=-1):
    """Whether ``corp`` (the operating corp by default) must buy a train."""
    return must_buy_train(state, _corp_or_operating(state, corp))


cpdef bint in_emergency_py(GameState state, int corp=-1):
    """Whether ``corp`` (the operating corp by default) must buy a train its treasury cannot fund."""
    return in_emergency(state, _corp_or_operating(state, corp))


cpdef bint can_go_bankrupt_py(GameState state, int corp=-1):
    """Whether the president of ``corp`` (the operating corp by default) may go
    bankrupt, were it in an emergency (Ruby's ``can_go_bankrupt?``)."""
    return can_go_bankrupt(state, _corp_or_operating(state, corp))


cpdef bint is_terminal_py(GameState state):
    return is_terminal(state)


cpdef list final_scores_py(GameState state):
    """One score per canonical player: Ruby's ``Player#value``."""
    cdef int scores[6]  # MAX_PLAYERS
    cdef int n = final_scores(state, scores)
    return [scores[i] for i in range(n)]


cpdef list purchasable_privates_py(GameState state, int corp=-1):
    """The privates ``corp`` (the operating corp by default) may buy now."""
    corp = _corp_or_operating(state, corp)
    return [p for p in range(<int>NUM_PRIVATES) if can_buy_private(state, corp, p)]


cpdef bint can_exchange_dr_py(GameState state, int player):
    """Whether ``player`` may exchange DR for an IR share now, whoever's turn it is."""
    assert 0 <= player < turn_num_players(state), f"player {player} out of range"
    return can_exchange_dr(state, player)


cpdef list port_lays_py(GameState state):
    """``(hex, rotation)`` of each port tile lay MF's owner may make now, in PORT's order."""
    cdef uint16_t ids[10]  # ACTION_SIZE_PORT
    cdef int count = port_lays(state, ids)
    cdef ActionInfo info
    cdef int k
    lays = []
    for k in range(count):
        info = decode_action(DPHASE_PORT, ids[k])
        lays.append((info.hex, info.count))
    return lays


cpdef list er_candidates_py(GameState state):
    """``(tile, rotation)`` of each lay ER's seller may make on C4, in ER_BONUS's order."""
    cdef LayCandidate buf[15]  # MAX_LAY_CANDIDATES
    cdef int count = er_candidates(state, buf)
    return [(buf[k].tile, buf[k].rotation) for k in range(count)]


cpdef list token_hexes_py(GameState state, int corp=-1):
    """The hexes ``corp`` (the operating corp by default) may token: TOKEN's hexes."""
    corp = _corp_or_operating(state, corp)
    return [h for h in range(<int>NUM_HEXES) if can_token(state, corp, h)]


# Python mirrors: a bare assignment would be shadowed by the cdef names.
for _name, _member in ActionType.__members__.items():
    globals()[_name] = int(_member)
del _name, _member


# =============================================================================
# IMPORT-TIME CHECKS
# =============================================================================

# Local id, candidate, price, and score buffers are sized with the literals 46, 15, 10, 5, 4, and 6.
assert <int>MAX_ACTION_SIZE == 46 and <int>MAX_LAY_CANDIDATES == 15 and <int>MAX_PLAYERS == 6
assert <int>ACTION_SIZE_BUY_PRIVATE_PRE == 10 and <int>NUM_PORT_LAYS == 10
assert <int>TRAIN_PRICE_POINTS == 5 and <int>PRIVATE_PRICE_POINTS == 4


def _check_round_trip():
    cdef int phase, action_id
    for phase in range(<int>NUM_DECISION_PHASES):
        for action_id in range(PHASE_ACTION_SIZE[phase]):
            assert encode_action(decode_action(phase, action_id)) == action_id, (phase, action_id)
        if has_exact_amounts(phase):
            action_id = PHASE_ACTION_SIZE[phase] + 1889
            assert encode_action(decode_action(phase, action_id)) == action_id, (phase, action_id)


_check_round_trip()
del _check_round_trip
