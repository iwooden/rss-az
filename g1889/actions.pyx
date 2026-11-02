"""Phase-local action encoding and legality for the 1889 engine; see actions.pxd."""

from collections import namedtuple

cimport numpy as cnp
import numpy as np
from libc.stdint cimport int8_t, uint16_t
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
    TRACK_BUY_PRIVATE, TRACK_DR, TRACK_HEX, TRACK_MF, TRACK_PASS,
)
from g1889.data.tables cimport (
    MF_PORT_HEX, MIN_BID_INCREMENT, NUM_CORPS, NUM_HEXES, NUM_PORTS, NUM_PRIVATES, NUM_SIDES,
    NUM_TRADE_IN_TYPES, NUM_TRAIN_TYPES, PAR_PRICE, TRADE_IN_TYPE,
)
from g1889.entities.player cimport player_available
from g1889.entities.private cimport private_min_bid, private_owner, private_price, topmost_private
from g1889.entities.turn cimport (
    turn_active_player, turn_auction_pass_payouts, turn_auction_private, turn_pending_private, turn_step,
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
    """The yen of a BID or RAISE decision: its exact amount, or grid point ``count``."""
    if info.type == ACTION_EXACT_PRICE:
        return info.amount
    return private_min_bid(state, _bid_private(state, info.phase)) + <int>MIN_BID_INCREMENT * info.count


cdef inline bint _amount_in_range(GameState state, int phase, int amount) noexcept nogil:
    cdef int p = _bid_private(state, phase)
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
    if phase == DPHASE_RAISE:
        return count
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
        if phase == DPHASE_BID or phase == DPHASE_RAISE:
            return _amount_in_range(state, phase, action_id - PHASE_ACTION_SIZE[phase])
        return _unimplemented(phase, b"exact amounts")
    count = enumerate_legal_actions(state, ids)
    for i in range(count):
        if ids[i] == action_id:
            return True
    return False


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


# Python mirrors: a bare assignment would be shadowed by the cdef names.
for _name, _member in ActionType.__members__.items():
    globals()[_name] = int(_member)
del _name, _member


# =============================================================================
# IMPORT-TIME CHECKS
# =============================================================================

# Local id buffers are sized with the literal 102.
assert <int>MAX_ACTION_SIZE == 102


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
