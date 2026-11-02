"""Phase-local action encoding and legality for the 1889 engine; see actions.pxd."""

from collections import namedtuple

cimport numpy as cnp
import numpy as np
from libc.stdint cimport int8_t, uint16_t, uint32_t
from libc.stdio cimport fprintf, stderr
from libc.stdlib cimport abort

from g1889.constants cimport (
    ACTION_SIZE_RAISE, AUCTION_BID, AUCTION_BUY, AUCTION_PASS, BID_POINT, BID_POINTS,
    BUY_PRIVATE_DONE, BUY_PRIVATE_DR, BUY_PRIVATE_PRIVATE, BUY_TRAIN_BANKRUPT,
    BUY_TRAIN_BUY_PRIVATE, BUY_TRAIN_CORP, BUY_TRAIN_DEPOT, BUY_TRAIN_DR, BUY_TRAIN_PASS,
    BUY_TRAIN_POOL, BUY_TRAIN_SELL, BUY_TRAIN_TRADE_IN, DIVIDEND_BUY_PRIVATE, DIVIDEND_DR,
    DIVIDEND_PAYOUT, DIVIDEND_WITHHOLD, DPHASE_AUCTION, DPHASE_BID, DPHASE_BUY_PRIVATE,
    DPHASE_BUY_TRAIN, DPHASE_DISCARD, DPHASE_DIVIDEND, DPHASE_ER_BONUS, DPHASE_PAR, DPHASE_PORT,
    DPHASE_PRIVATE_PRICE, DPHASE_RAISE, DPHASE_STOCK, DPHASE_TILE, DPHASE_TOKEN, DPHASE_TRACK,
    DPHASE_TRAIN_PRICE, ER_BONUS_CANDIDATE, ER_BONUS_PASS, MAX_ACTION_SIZE, MAX_SALE_SHARES,
    NUM_DECISION_PHASES, NUM_STEPS, OWNER_UNSOLD, PHASE_ACTION_SIZE, PORT_LAY, RAISE_DROP,
    RAISE_POINT, RAISE_POINTS, STEP_TO_DECISION_PHASE, STOCK_DR, STOCK_IPO, STOCK_MF, STOCK_PAR,
    STOCK_PASS, STOCK_POOL, STOCK_SELL, TOKEN_BUY_PRIVATE, TOKEN_DR, TOKEN_HEX, TOKEN_PASS,
    TRACK_BUY_PRIVATE, TRACK_DR, TRACK_HEX, TRACK_MF, TRACK_PASS, TRAIN_PRICE_POINTS,
)
from g1889.data.tables cimport (
    CERT_LIMIT, D_TRAIN, ENDPOINT_PAIRS, HEX_EXIT_MASK, HEX_NEIGHBOR, MAX_LAY_CANDIDATES, MAX_OWNERSHIP_PERCENT,
    MF_PORT_HEX, MIN_BID_INCREMENT, NODE, NUM_CORPS, NUM_HEXES, NUM_PAR_PRICES, NUM_PORTS, NUM_PRIVATES,
    NUM_SIDES, NUM_TRADE_IN_TYPES, NUM_TRAIN_TYPES, PAR_PRICE, PHASE_TILE_COLORS,
    TILE_BLOCKS_LAY, TILE_COLOR, TILE_EXITS, TILE_MASK, TILE_REVENUE, TRADE_IN_TYPE, TRAIN_PRICE, UPG_ROT,
    UPG_START, UPG_TILE,
)
from g1889.entities.corp cimport (
    corp_cash, corp_ipo_shares, corp_max_sale, corp_par_index, corp_par_price, corp_pool_shares, corp_president,
    corp_token_price, corp_tokens_left,
)
from g1889.entities.graph cimport graph_node, graph_route_available, graph_sides, graph_walks_new_track
from g1889.entities.hexmap cimport hex_paths, hex_rotation, hex_tile, map_has_slot_for, map_lay_cost, supply_count
from g1889.entities.market cimport market_price, market_unlimited
from g1889.entities.player cimport (
    player_available, player_can_gain, player_cash, player_cert_count, player_shares, player_sold_mask,
)
from g1889.entities.phase cimport game_phase
from g1889.entities.private cimport (
    private_blocks_hex, private_min_bid, private_owner, private_price, topmost_private,
)
from g1889.entities.trains cimport (
    corp_train_count, corp_train_total, pool_count, trade_in_price, trains_cheapest_price, trains_depot_types,
    trains_first_crowded, trains_limit,
)
from g1889.entities.turn cimport (
    turn_active_player, turn_auction_pass_payouts, turn_auction_private, turn_num_players, turn_operating_corp,
    turn_pending_hex, turn_pending_private, turn_pending_train_source, turn_route_revenue, turn_sr_bought,
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


cdef int _unimplemented(int phase, const char* what) except -1 nogil:
    with gil:
        raise NotImplementedError(
            f"1889 {_PHASE_NAMES[phase]}: {what.decode('ascii')} not implemented yet (engine-design.md §11)")


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


cdef inline int _sale(int first, int corp, int count) except -1 nogil:
    return first + <int>MAX_SALE_SHARES * _index(corp, <int>NUM_CORPS) + _index(count - 1, <int>MAX_SALE_SHARES)


cdef int _find(const int8_t* values, int count, int value) except -1 nogil:
    cdef int i
    for i in range(count):
        if values[i] == value:
            return i
    with gil:
        raise ValueError(f"{value} is not one of the {count} encodable values")


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
            return AUCTION_BID + _index(info.private, <int>NUM_PRIVATES)
    elif phase == DPHASE_BID or phase == DPHASE_TRAIN_PRICE or phase == DPHASE_PRIVATE_PRICE:
        if t == ACTION_PRICE_POINT:
            return _index(info.count, PHASE_ACTION_SIZE[phase])
        if t == ACTION_EXACT_PRICE:
            return _exact(phase, info.amount)
    elif phase == DPHASE_RAISE:
        if t == ACTION_PASS:
            return RAISE_DROP
        if t == ACTION_PRICE_POINT:
            return RAISE_POINT + _index(info.count, <int>ACTION_SIZE_RAISE - <int>RAISE_POINT)
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
        if t == ACTION_SELL:
            return _sale(STOCK_SELL, info.corp, info.count)
        if t == ACTION_DR_EXCHANGE:
            return STOCK_DR
        if t == ACTION_MF_PORT:
            return STOCK_MF
    elif phase == DPHASE_PAR:
        if t == ACTION_PAR:
            return _index(info.count, PHASE_ACTION_SIZE[phase])
    elif phase == DPHASE_TRACK:
        if t == ACTION_PASS:
            return TRACK_PASS
        if t == ACTION_SELECT_HEX:
            return TRACK_HEX + _index(info.hex, <int>NUM_HEXES)
        if t == ACTION_BUY_PRIVATE:
            return TRACK_BUY_PRIVATE + _index(info.private, <int>NUM_PRIVATES)
        if t == ACTION_DR_EXCHANGE:
            return TRACK_DR
        if t == ACTION_MF_PORT:
            return TRACK_MF
    elif phase == DPHASE_TILE:
        if t == ACTION_LAY_CANDIDATE:
            return _index(info.count, PHASE_ACTION_SIZE[phase])
    elif phase == DPHASE_TOKEN:
        if t == ACTION_PASS:
            return TOKEN_PASS
        if t == ACTION_PLACE_TOKEN:
            return TOKEN_HEX + _index(info.hex, <int>NUM_HEXES)
        if t == ACTION_BUY_PRIVATE:
            return TOKEN_BUY_PRIVATE + _index(info.private, <int>NUM_PRIVATES)
        if t == ACTION_DR_EXCHANGE:
            return TOKEN_DR
    elif phase == DPHASE_DIVIDEND:
        if t == ACTION_WITHHOLD:
            return DIVIDEND_WITHHOLD
        if t == ACTION_PAYOUT:
            return DIVIDEND_PAYOUT
        if t == ACTION_BUY_PRIVATE:
            return DIVIDEND_BUY_PRIVATE + _index(info.private, <int>NUM_PRIVATES)
        if t == ACTION_DR_EXCHANGE:
            return DIVIDEND_DR
    elif phase == DPHASE_DISCARD:
        if t == ACTION_DISCARD:
            return _index(info.train_type, <int>NUM_TRAIN_TYPES)
    elif phase == DPHASE_BUY_TRAIN:
        if t == ACTION_PASS:
            return BUY_TRAIN_PASS
        if t == ACTION_BUY_DEPOT_TRAIN:
            return BUY_TRAIN_DEPOT + _index(info.train_type, <int>NUM_TRAIN_TYPES)
        if t == ACTION_TRADE_IN:
            return BUY_TRAIN_TRADE_IN + _find(TRADE_IN_TYPE, <int>NUM_TRADE_IN_TYPES, info.train_type)
        if t == ACTION_BUY_POOL_TRAIN:
            return BUY_TRAIN_POOL + _index(info.train_type, <int>NUM_TRAIN_TYPES)
        if t == ACTION_BUY_CORP_TRAIN:
            return (BUY_TRAIN_CORP + <int>NUM_TRAIN_TYPES * _index(info.corp, <int>NUM_CORPS)
                    + _index(info.train_type, <int>NUM_TRAIN_TYPES))
        if t == ACTION_SELL:
            return _sale(BUY_TRAIN_SELL, info.corp, info.count)
        if t == ACTION_BANKRUPT:
            return BUY_TRAIN_BANKRUPT
        if t == ACTION_BUY_PRIVATE:
            return BUY_TRAIN_BUY_PRIVATE + _index(info.private, <int>NUM_PRIVATES)
        if t == ACTION_DR_EXCHANGE:
            return BUY_TRAIN_DR
    elif phase == DPHASE_BUY_PRIVATE:
        if t == ACTION_PASS:
            return BUY_PRIVATE_DONE
        if t == ACTION_BUY_PRIVATE:
            return BUY_PRIVATE_PRIVATE + _index(info.private, <int>NUM_PRIVATES)
        if t == ACTION_DR_EXCHANGE:
            return BUY_PRIVATE_DR
    elif phase == DPHASE_ER_BONUS:
        if t == ACTION_PASS:
            return ER_BONUS_PASS
        if t == ACTION_LAY_CANDIDATE:
            return ER_BONUS_CANDIDATE + _index(info.count, PHASE_ACTION_SIZE[phase] - <int>ER_BONUS_CANDIDATE)
    elif phase == DPHASE_PORT:
        if t == ACTION_PORT_LAY:
            return (PORT_LAY + <int>NUM_SIDES * _find(MF_PORT_HEX, <int>NUM_PORTS, info.hex)
                    + _index(info.count, <int>NUM_SIDES))
    with gil:
        raise ValueError(f"no action of type {t} in decision phase {phase}")


cdef inline void _decode_price(ActionInfo* info, int action_id, int first_point, int exact) noexcept nogil:
    if action_id >= exact:
        info.type = ACTION_EXACT_PRICE
        info.amount = action_id - exact
    else:
        info.type = ACTION_PRICE_POINT
        info.count = action_id - first_point


cdef inline void _decode_sale(ActionInfo* info, int offset) noexcept nogil:
    info.type = ACTION_SELL
    info.corp = offset // <int>MAX_SALE_SHARES
    info.count = offset % <int>MAX_SALE_SHARES + 1


cdef ActionInfo decode_action(int phase, int action_id) noexcept nogil:
    """Unpack a phase-local id; exact ids (at or above the phase size) decode to
    ACTION_EXACT_PRICE. The id must be valid for the phase."""
    cdef ActionInfo info
    cdef int k
    info.phase = phase
    info.type = ACTION_PASS
    info.corp = info.private = info.hex = info.train_type = info.count = info.amount = -1

    if phase == DPHASE_AUCTION:
        if action_id == AUCTION_BUY:
            info.type = ACTION_BUY_TOPMOST
        elif action_id >= AUCTION_BID:
            info.type = ACTION_SELECT_BID
            info.private = action_id - AUCTION_BID
    elif phase == DPHASE_BID or phase == DPHASE_TRAIN_PRICE or phase == DPHASE_PRIVATE_PRICE:
        _decode_price(&info, action_id, 0, PHASE_ACTION_SIZE[phase])
    elif phase == DPHASE_RAISE:
        if action_id != RAISE_DROP:
            _decode_price(&info, action_id, RAISE_POINT, PHASE_ACTION_SIZE[phase])
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
            _decode_sale(&info, action_id - STOCK_SELL)
        else:
            info.type = ACTION_MF_PORT if action_id == STOCK_MF else ACTION_DR_EXCHANGE
    elif phase == DPHASE_PAR:
        info.type = ACTION_PAR
        info.count = action_id
        info.amount = PAR_PRICE[action_id]
    elif phase == DPHASE_TRACK:
        if action_id == TRACK_PASS:
            pass
        elif action_id < TRACK_BUY_PRIVATE:
            info.type = ACTION_SELECT_HEX
            info.hex = action_id - TRACK_HEX
        elif action_id < TRACK_DR:
            info.type = ACTION_BUY_PRIVATE
            info.private = action_id - TRACK_BUY_PRIVATE
        else:
            info.type = ACTION_MF_PORT if action_id == TRACK_MF else ACTION_DR_EXCHANGE
    elif phase == DPHASE_TILE:
        info.type = ACTION_LAY_CANDIDATE
        info.count = action_id
    elif phase == DPHASE_TOKEN:
        if action_id == TOKEN_PASS:
            pass
        elif action_id < TOKEN_BUY_PRIVATE:
            info.type = ACTION_PLACE_TOKEN
            info.hex = action_id - TOKEN_HEX
        elif action_id < TOKEN_DR:
            info.type = ACTION_BUY_PRIVATE
            info.private = action_id - TOKEN_BUY_PRIVATE
        else:
            info.type = ACTION_DR_EXCHANGE
    elif phase == DPHASE_DIVIDEND:
        if action_id == DIVIDEND_WITHHOLD:
            info.type = ACTION_WITHHOLD
        elif action_id == DIVIDEND_PAYOUT:
            info.type = ACTION_PAYOUT
        elif action_id < DIVIDEND_DR:
            info.type = ACTION_BUY_PRIVATE
            info.private = action_id - DIVIDEND_BUY_PRIVATE
        else:
            info.type = ACTION_DR_EXCHANGE
    elif phase == DPHASE_DISCARD:
        info.type = ACTION_DISCARD
        info.train_type = action_id
    elif phase == DPHASE_BUY_TRAIN:
        if action_id == BUY_TRAIN_PASS:
            pass
        elif action_id < BUY_TRAIN_TRADE_IN:
            info.type = ACTION_BUY_DEPOT_TRAIN
            info.train_type = action_id - BUY_TRAIN_DEPOT
        elif action_id < BUY_TRAIN_POOL:
            info.type = ACTION_TRADE_IN
            info.train_type = TRADE_IN_TYPE[action_id - BUY_TRAIN_TRADE_IN]
        elif action_id < BUY_TRAIN_CORP:
            info.type = ACTION_BUY_POOL_TRAIN
            info.train_type = action_id - BUY_TRAIN_POOL
        elif action_id < BUY_TRAIN_SELL:
            k = action_id - BUY_TRAIN_CORP
            info.type = ACTION_BUY_CORP_TRAIN
            info.corp = k // <int>NUM_TRAIN_TYPES
            info.train_type = k % <int>NUM_TRAIN_TYPES
        elif action_id < BUY_TRAIN_BANKRUPT:
            _decode_sale(&info, action_id - BUY_TRAIN_SELL)
        elif action_id == BUY_TRAIN_BANKRUPT:
            info.type = ACTION_BANKRUPT
        elif action_id < BUY_TRAIN_DR:
            info.type = ACTION_BUY_PRIVATE
            info.private = action_id - BUY_TRAIN_BUY_PRIVATE
        else:
            info.type = ACTION_DR_EXCHANGE
    elif phase == DPHASE_BUY_PRIVATE:
        if action_id == BUY_PRIVATE_DONE:
            pass
        elif action_id < BUY_PRIVATE_DR:
            info.type = ACTION_BUY_PRIVATE
            info.private = action_id - BUY_PRIVATE_PRIVATE
        else:
            info.type = ACTION_DR_EXCHANGE
    elif phase == DPHASE_ER_BONUS:
        if action_id != ER_BONUS_PASS:
            info.type = ACTION_LAY_CANDIDATE
            info.count = action_id - ER_BONUS_CANDIDATE
    elif phase == DPHASE_PORT:
        info.type = ACTION_PORT_LAY
        info.hex = MF_PORT_HEX[(action_id - PORT_LAY) // <int>NUM_SIDES]
        info.count = (action_id - PORT_LAY) % <int>NUM_SIDES
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


cdef inline bint _has_exact(int phase) noexcept nogil:
    return (phase == DPHASE_BID or phase == DPHASE_RAISE or phase == DPHASE_TRAIN_PRICE
            or phase == DPHASE_PRIVATE_PRICE)


# =============================================================================
# AUCTION, BID, RAISE
# =============================================================================
#
# A bid on private p ranges from its minimum bid (``private_min_bid``) to the
# bidder's cash less their other bids (``player_available``). Grid point k is
# the minimum + ¥5k; points past the range are not offered.

cdef inline int _bid_private(GameState state, int phase) noexcept nogil:
    """The private a BID or RAISE amount is for."""
    return turn_pending_private(state) if phase == DPHASE_BID else turn_auction_private(state)


cdef int price_amount(GameState state, const ActionInfo* info) noexcept nogil:
    """The yen of a BID, RAISE, or TRAIN_PRICE decision: its exact amount, or
    grid point ``count``."""
    cdef int points[5]  # TRAIN_PRICE_POINTS
    if info.type == ACTION_EXACT_PRICE:
        return info.amount
    if info.phase == DPHASE_TRAIN_PRICE:
        _train_price_points(state, points)
        return points[info.count]
    return private_min_bid(state, _bid_private(state, info.phase)) + <int>MIN_BID_INCREMENT * info.count


cdef inline bint _amount_in_range(GameState state, int phase, int amount) noexcept nogil:
    cdef int p
    if phase == DPHASE_TRAIN_PRICE:
        return 1 <= amount <= _train_price_cash(state)
    p = _bid_private(state, phase)
    return private_min_bid(state, p) <= amount <= player_available(state, turn_active_player(state), p)


cdef int _bid_points(GameState state, int phase, uint16_t* ids, int count, int first_id,
                     int num_points) noexcept nogil:
    """Append the grid points of the actor's bid in ``phase``, from ``first_id``."""
    cdef int p = _bid_private(state, phase)
    cdef int low = private_min_bid(state, p)
    cdef int high = player_available(state, turn_active_player(state), p)
    cdef int k
    for k in range(num_points):
        if low + <int>MIN_BID_INCREMENT * k > high:
            break
        ids[count] = <uint16_t>(first_id + k)
        count += 1
    return count


cdef int _enumerate_auction(GameState state, uint16_t* ids) noexcept nogil:
    """Pass; buy the topmost private at its price; select each unsold private
    whose minimum bid the player can cover. On the topmost, BID's amount buys it."""
    cdef int player = turn_active_player(state)
    cdef int top = topmost_private(state)
    cdef int count = 1
    cdef int p
    ids[0] = AUCTION_PASS
    if private_price(state, top) <= player_available(state, player, top):
        ids[count] = AUCTION_BUY
        count += 1
    for p in range(top, <int>NUM_PRIVATES):
        if (private_owner(state, p) == <int>OWNER_UNSOLD
                and private_min_bid(state, p) <= player_available(state, player, p)):
            ids[count] = <uint16_t>(AUCTION_BID + p)
            count += 1
    return count


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
    cdef bint can_buy = count > 1 and ids[1] == AUCTION_BUY
    cdef int i, p
    cdef int kept = 0
    for i in range(count):
        if ids[i] == AUCTION_PASS and capped and can_buy:
            continue
        if ids[i] >= AUCTION_BID:
            p = ids[i] - AUCTION_BID
            if p == top or (capped and private_min_bid(state, p) > _capped_bid_limit(state, p)):
                continue
        ids[kept] = ids[i]
        kept += 1
    return kept


cdef int _filter_bid(GameState state, int count) noexcept nogil:
    """Once capped, keep the points (ids 0, 1, ...) that leave the bidder able
    to buy the topmost; keep all of them if none does, which only a selection
    outside the policy can bring about."""
    cdef int p = turn_pending_private(state)
    cdef int room
    if not _capped(state) or p == topmost_private(state):
        return count
    room = _capped_bid_limit(state, p) - private_min_bid(state, p)
    if room < 0:
        return count
    return min(count, room // <int>MIN_BID_INCREMENT + 1)


# =============================================================================
# STOCK, PAR
# =============================================================================
#
# Ruby's BuySellParShares (step/buy_sell_par_shares.rb). On a turn the player
# buys at most one certificate (a share, or a par through PAR) and sells any
# number of blocks, one per corp, until a purchase follows a sale
# (SELL_BUY_ORDER = :sell_buy_or_buy_sell). Nothing is sold in the first SR
# (SELL_AFTER = :first), and nothing sold this SR is bought back. A player
# over a limit with something to sell has a must-sell turn: sales only, no
# pass. Slot 0 passes before acting and is "done" after; it is legal at every
# other turn, so a player with nothing else to do passes through the forced
# rule, as Ruby skips them. DR (STOCK_DR) and MF (STOCK_MF) arrive with 6.8.

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


cdef inline int _sale_limit(GameState state, int player, int corp) noexcept nogil:
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
        if _sale_limit(state, player, c) == 0:
            continue
        can_sell = True
        if 10 * player_shares(state, player, c) > <int>MAX_OWNERSHIP_PERCENT and not market_unlimited(state, c):
            return True
    return can_sell and player_cert_count(state, player) > CERT_LIMIT[turn_num_players(state)]


cdef int _enumerate_stock(GameState state, uint16_t* ids) noexcept nogil:
    """Pass; buys from the IPO, from the pool, and pars, unless the turn has
    bought; then every legal block of every corp. A must-sell turn offers only
    the sales."""
    cdef int player = turn_active_player(state)
    cdef int count = 0
    cdef int c, n
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
        for n in range(_sale_limit(state, player, c)):
            ids[count] = <uint16_t>(STOCK_SELL + <int>MAX_SALE_SHARES * c + n)
            count += 1
    return count


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
# Ruby's checks, sorted by (tile, rotation). Buying a private, DR, and MF
# arrive with 6.8.


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

    Every 1889 city upgrade changes the city's value, and every path of a
    city tile touches the city. Such a lay passes whenever its exits meet a
    side the corp reaches (``_lay_scan`` checks that first): the walks run as
    before the lay until they first enter the hex at a side the new tile has
    track on, and there they walk every path at that side. New track off the
    node fails when none of it ends where a walk could enter the hex
    (``enterable``). Any other lay is decided by walking with the tile laid
    (``graph_walks_new_track``)."""
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


cdef int _enumerate_track(GameState state, uint16_t* ids) noexcept nogil:
    """Pass, then each hex with a lay candidate."""
    cdef int corp = turn_operating_corp(state)
    cdef int count = 1
    cdef int h
    ids[0] = TRACK_PASS
    if corp < 0:
        return count
    for h in range(<int>NUM_HEXES):
        if can_lay(state, corp, h):
            ids[count] = <uint16_t>(TRACK_HEX + h)
            count += 1
    return count


cdef int _enumerate_tile(GameState state, uint16_t* ids) noexcept nogil:
    """An index per candidate at the pending hex."""
    cdef LayCandidate candidates[15]  # MAX_LAY_CANDIDATES
    cdef int hex = turn_pending_hex(state)
    cdef int count = 0
    cdef int k
    if hex >= 0 and turn_operating_corp(state) >= 0:
        count = enumerate_lay_candidates(state, hex, candidates)
    for k in range(count):
        ids[k] = <uint16_t>k
    return count


cdef int _enumerate_token(GameState state, uint16_t* ids) noexcept nogil:
    """Pass, then each city the corp may token."""
    cdef int corp = turn_operating_corp(state)
    cdef int count = 1
    cdef int h
    ids[0] = TOKEN_PASS
    if corp < 0:
        return count
    for h in range(<int>NUM_HEXES):
        if can_token(state, corp, h):
            ids[count] = <uint16_t>(TOKEN_HEX + h)
            count += 1
    return count


# =============================================================================
# DIVIDEND, BUY_TRAIN, TRAIN_PRICE, DISCARD, BUY_PRIVATE
# =============================================================================
#
# Ruby's Dividend, BuyTrain, DiscardTrain, and BuyCompany steps
# (step/dividend.rb, step/buy_train.rb, step/train.rb, step/discard_train.rb,
# step/buy_company.rb). Ruby skips Dividend at zero revenue with a withhold,
# so DIVIDEND then offers only that, and the driver applies it. BUY_TRAIN
# buys one train at a time until the corp passes; a buy from another corp
# names its price in TRAIN_PRICE. A must-buy the treasury cannot fund needs
# emergency money (6.8), and BUY_TRAIN raises there rather than offer an
# incomplete set. Buying a private, DR, and MF, which the OR decisions lack
# until 6.8, are the exception: BUY_PRIVATE offers only "done".


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


cdef int _enumerate_buy_train(GameState state, uint16_t* ids) except -1 nogil:
    """Pass unless the corp must buy. Then, with room under the phase's limit
    (``room?``), each depot type its treasury affords (``depot_trains``);
    each 4, 5, or 6 it may trade with ¥800 for an available D, room or not
    (``discountable_trains_for``); with room again, each pool type it affords
    and, with ¥1 or more, every train of another corp (``other_trains``),
    priced in TRAIN_PRICE (``spend_minmax``: ¥1 to the treasury). Ruby's
    ``buy_train_action`` checks no room, but at the limit its BuyTrain step
    opens only for a trade-in (``can_buy_train?``), and the UI then offers
    only that. A must-buy whose
    treasury is short of the cheapest depot or pool train raises: the
    president's money, share sales, and the buys from other corps that go
    with them are emergency money (6.8)."""
    cdef int corp = turn_operating_corp(state)
    cdef int count = 0
    cdef int cash, types, t, c, i
    cdef bint room
    if corp < 0:
        ids[0] = BUY_TRAIN_PASS
        return 1
    cash = corp_cash(state, corp)
    if not must_buy_train(state, corp):
        ids[count] = BUY_TRAIN_PASS
        count += 1
    elif cash < trains_cheapest_price(state):
        return _unimplemented(DPHASE_BUY_TRAIN, b"emergency money")
    room = corp_train_total(state, corp) < trains_limit(state)
    types = trains_depot_types(state)
    if room:
        for t in range(<int>NUM_TRAIN_TYPES):
            if types >> t & 1 and TRAIN_PRICE[t] <= cash:
                ids[count] = <uint16_t>(BUY_TRAIN_DEPOT + t)
                count += 1
    if types >> <int>D_TRAIN & 1:
        for i in range(<int>NUM_TRADE_IN_TYPES):
            t = TRADE_IN_TYPE[i]
            if corp_train_count(state, corp, t) and trade_in_price(t) <= cash:
                ids[count] = <uint16_t>(BUY_TRAIN_TRADE_IN + i)
                count += 1
    if not room:
        return count
    for t in range(<int>NUM_TRAIN_TYPES):
        if pool_count(state, t) and TRAIN_PRICE[t] <= cash:
            ids[count] = <uint16_t>(BUY_TRAIN_POOL + t)
            count += 1
    if cash > 0:
        for c in range(<int>NUM_CORPS):
            for t in range(<int>NUM_TRAIN_TYPES):
                if c != corp and corp_train_count(state, c, t):
                    ids[count] = <uint16_t>(BUY_TRAIN_CORP + <int>NUM_TRAIN_TYPES * c + t)
                    count += 1
    return count


cdef int _filter_buy_train(GameState state, uint16_t* ids, int count) noexcept nogil:
    """``train_sale_same_president`` (§7): only corps with the buyer's
    president sell it trains."""
    cdef int president, i, id_
    cdef int kept = 0
    if not state.train_sale_same_president:
        return count
    president = corp_president(state, turn_operating_corp(state))
    for i in range(count):
        id_ = ids[i]
        if (BUY_TRAIN_CORP <= id_ < BUY_TRAIN_SELL
                and corp_president(state, (id_ - BUY_TRAIN_CORP) // <int>NUM_TRAIN_TYPES) != president):
            continue
        ids[kept] = ids[i]
        kept += 1
    return kept


cdef inline int _train_price_cash(GameState state) noexcept nogil:
    """The most TRAIN_PRICE may pay: the buyer's treasury, 0 without a pending train."""
    cdef int corp = turn_operating_corp(state)
    return corp_cash(state, corp) if corp >= 0 and turn_pending_train_source(state) >= 0 else 0


cdef void _train_price_points(GameState state, int* points) noexcept nogil:
    """TRAIN_PRICE's grid for the pending train (§5.2): ¥1, its list price,
    ⌈cash/3⌉, ⌈2·cash/3⌉, and the buyer's cash. A point outside ¥1 to the
    cash (``spend_minmax``), or equal to an earlier one, is -1: not offered."""
    cdef int cash = _train_price_cash(state)
    cdef int source = turn_pending_train_source(state)
    cdef int k, j
    points[0] = 1
    points[1] = TRAIN_PRICE[source % <int>NUM_TRAIN_TYPES] if source >= 0 else 0
    points[2] = (cash + 2) // 3
    points[3] = (2 * cash + 2) // 3
    points[4] = cash
    for k in range(<int>TRAIN_PRICE_POINTS):
        if not 1 <= points[k] <= cash:
            points[k] = -1
        for j in range(k):
            if points[j] == points[k]:
                points[k] = -1


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


cdef int _enumerate_discard(GameState state, uint16_t* ids) noexcept nogil:
    """Each train type of the first crowded corp (``DiscardTrain``), whose
    president chooses."""
    cdef int corp = trains_first_crowded(state)
    cdef int count = 0
    cdef int t
    if corp < 0:
        return 0
    for t in range(<int>NUM_TRAIN_TYPES):
        if corp_train_count(state, corp, t):
            ids[count] = <uint16_t>t
            count += 1
    return count


cdef int enumerate_legal_actions(GameState state, uint16_t* action_ids) except -1 nogil:
    """Write the engine-legal ids of the current decision in a fixed order and
    return the count, 0 at an automated step. The buffer holds MAX_ACTION_SIZE."""
    cdef int phase = get_decision_phase(state)
    cdef int count
    if phase < 0:
        return 0
    # Each phase's _enumerate_<phase> joins this dispatch in its sub-step.
    if phase == DPHASE_AUCTION:
        count = _enumerate_auction(state, action_ids)
    elif phase == DPHASE_BID:
        count = _bid_points(state, phase, action_ids, 0, BID_POINT, BID_POINTS)
    elif phase == DPHASE_RAISE:
        action_ids[0] = RAISE_DROP
        count = _bid_points(state, phase, action_ids, 1, RAISE_POINT, RAISE_POINTS)
    elif phase == DPHASE_STOCK:
        count = _enumerate_stock(state, action_ids)
    elif phase == DPHASE_PAR:
        count = _enumerate_par(state, action_ids)
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
    elif phase == DPHASE_TRAIN_PRICE:
        count = _enumerate_train_price(state, action_ids)
    elif phase == DPHASE_DISCARD:
        count = _enumerate_discard(state, action_ids)
    elif phase == DPHASE_BUY_PRIVATE:
        action_ids[0] = BUY_PRIVATE_DONE
        count = 1
    else:
        count = _unimplemented(phase, b"legal actions")
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
    # Each phase's policy filter lands with its enumerator.
    phase = get_decision_phase(state)
    if phase == DPHASE_AUCTION:
        return _filter_auction(state, action_ids, count)
    if phase == DPHASE_BID:
        return _filter_bid(state, count)
    if phase == DPHASE_BUY_TRAIN:
        return _filter_buy_train(state, action_ids, count)
    if (phase == DPHASE_RAISE or phase == DPHASE_STOCK or phase == DPHASE_PAR or phase == DPHASE_TRACK
            or phase == DPHASE_TILE or phase == DPHASE_TOKEN or phase == DPHASE_DIVIDEND
            or phase == DPHASE_TRAIN_PRICE or phase == DPHASE_DISCARD or phase == DPHASE_BUY_PRIVATE):
        return count  # no flag restricts these
    return _unimplemented(phase, b"policy filter")


cdef bint is_legal_action(GameState state, int action_id) except -1 nogil:
    """Engine legality: a scan of the enumerated ids, or a range check for an
    exact amount. False at an automated step."""
    cdef uint16_t ids[102]  # MAX_ACTION_SIZE
    cdef int phase = get_decision_phase(state)
    cdef int count, i
    if phase < 0 or action_id < 0:
        return False
    if action_id >= PHASE_ACTION_SIZE[phase]:
        if not _has_exact(phase) or action_id > MAX_ACTION_ID:
            return False
        # Each amount phase's range check lands with its enumerator.
        if phase == DPHASE_BID or phase == DPHASE_RAISE or phase == DPHASE_TRAIN_PRICE:
            return _amount_in_range(state, phase, action_id - PHASE_ACTION_SIZE[phase])
        return _unimplemented(phase, b"exact amounts")
    count = enumerate_legal_actions(state, ids)
    for i in range(count):
        if ids[i] == action_id:
            return True
    return False


cdef bint has_exact(int phase) noexcept nogil:
    return _has_exact(phase)


cdef bint wide_amount_range(GameState state, int phase) except -1 nogil:
    """Whether the current amount decision's legal range holds more than one
    amount, so exact amounts may lie beyond its grid points."""
    cdef int p
    if phase == DPHASE_BID or phase == DPHASE_RAISE:
        p = _bid_private(state, phase)
        return player_available(state, turn_active_player(state), p) > private_min_bid(state, p)
    if phase == DPHASE_TRAIN_PRICE:
        return _train_price_cash(state) > 1
    return _unimplemented(phase, b"exact amounts")


# =============================================================================
# PYTHON WRAPPERS
# =============================================================================

cpdef bint has_exact_amounts(int phase):
    """Whether ``phase`` takes exact amounts, at ids from its size up."""
    return _has_exact(phase)


cdef int _check_id(int phase, int action_id) except -1:
    assert 0 <= phase < <int>NUM_DECISION_PHASES, f"invalid decision phase {phase}"
    assert 0 <= action_id < PHASE_ACTION_SIZE[phase] or (_has_exact(phase) and action_id <= MAX_ACTION_ID), \
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
    BID, RAISE, or TRAIN_PRICE decision."""
    cdef int phase = get_decision_phase(state)
    assert phase == DPHASE_BID or phase == DPHASE_RAISE or phase == DPHASE_TRAIN_PRICE, \
        f"not at a BID, RAISE, or TRAIN_PRICE decision (phase {phase})"
    _check_id(phase, action_id)
    cdef ActionInfo info = decode_action(phase, action_id)
    assert info.type != ACTION_PASS, "RAISE's drop has no amount"
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

# Local id and candidate buffers are sized with the literals 102 and 15.
assert <int>MAX_ACTION_SIZE == 102 and <int>MAX_LAY_CANDIDATES == 15


def _check_round_trip():
    cdef int phase, action_id
    for phase in range(<int>NUM_DECISION_PHASES):
        for action_id in range(PHASE_ACTION_SIZE[phase]):
            assert encode_action(decode_action(phase, action_id)) == action_id, (phase, action_id)
        if _has_exact(phase):
            action_id = PHASE_ACTION_SIZE[phase] + 1889
            assert encode_action(decode_action(phase, action_id)) == action_id, (phase, action_id)


_check_round_trip()
del _check_round_trip
