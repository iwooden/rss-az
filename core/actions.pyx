# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Action vector implementation.

Defines the action space for the neural network output layer:
- Dynamic action count based on player count: 186 + (num_players * 20)
  - 3 players: 246 actions
  - 4 players: 266 actions
  - 5 players: 286 actions
  - 6 players: 306 actions
- Valid action mask generation
- Action decoding (index -> ActionInfo)
"""

cimport cython
import numpy as np
cimport numpy as cnp

from core.state cimport GameState
from core.data cimport (
    GameConstants, GamePhases,
    get_company_face_value, get_company_low_price, get_company_high_price,
    get_company_stars, get_par_price, get_market_index, get_market_price,
    is_valid_par_price, get_par_index_for_slot, NUM_PAR_PRICES, CORP_OS
)

# Import types and functions from our own pxd
from core.actions cimport (
    ActionLayout, ActionInfo, ActionType,
    ACTION_PASS, ACTION_AUCTION, ACTION_BUY_SHARE, ACTION_SELL_SHARE,
    ACTION_LEAVE_AUCTION, ACTION_RAISE_BID, ACTION_ACQ_PRICE, ACTION_ACQ_FI_HIGH,
    ACTION_ACQ_FI_FACE, ACTION_CLOSE, ACTION_DIVIDEND, ACTION_ISSUE, ACTION_IPO,
    AUCTION_CAP, MAX_PAR_SLOTS, ACQ_PRICE_RANGE, MAX_DIVIDEND,
    NUM_CORPS as ACTION_NUM_CORPS, NUM_COMPANIES as ACTION_NUM_COMPANIES,
    NUM_PAR_PRICES as ACTION_NUM_PAR_PRICES
)

# Constants from data module
DEF NUM_COMPANIES = 36
DEF NUM_CORPS = 8
DEF NUM_MARKET_SPACES = 27
DEF PHASE_INVEST = 0
DEF PHASE_BID_IN_AUCTION = 1
DEF PHASE_ACQUISITION = 3
DEF PHASE_CLOSING = 4
DEF PHASE_DIVIDENDS = 6
DEF PHASE_ISSUE_SHARES = 8
DEF PHASE_IPO = 9

# Import low-level functions from entities for direct access
from entities.player cimport (
    get_player_shares, PlayerOffsets, get_player_offsets,
    get_roundtrips, get_share_buys, get_share_sells
)
from entities.company cimport get_auction_company_for_slot


cdef int get_total_actions_for_players(int num_players) noexcept nogil:
    """
    Calculate total action count for a given player count.

    Layout:
    - INVEST: 1 + (num_players * 20) + 8 + 8 = 17 + (num_players * 20)
    - BID: 20
    - ACQUISITION: 54
    - CLOSING: 2
    - DIVIDENDS: 26
    - ISSUE: 2
    - IPO: 65

    Total = 186 + (num_players * 20)
    """
    return 186 + (num_players * AUCTION_CAP)


cdef ActionLayout compute_action_layout(int num_players) noexcept nogil:
    """Compute action layout with all offsets based on player count."""
    cdef ActionLayout layout
    cdef int offset = 0
    cdef int auction_slots = num_players  # One slot per player (max companies for auction)

    layout.total_size = get_total_actions_for_players(num_players)

    # INVEST phase: dynamic based on num_players
    layout.invest_start = offset
    layout.pass_invest = offset
    offset += 1
    layout.auction_base = offset  # slot * AUCTION_CAP + bid_offset
    offset += auction_slots * AUCTION_CAP  # num_players * 20
    layout.buy_share_base = offset  # +corp_id
    offset += NUM_CORPS  # 8
    layout.sell_share_base = offset  # +corp_id
    offset += NUM_CORPS  # 8
    # Total invest: 1 + (num_players * 20) + 8 + 8

    # BID phase: 137-156 (20 actions)
    layout.bid_start = offset
    layout.leave_auction = offset
    offset += 1
    layout.raise_bid_base = offset  # +bid_offset (0-18)
    offset += AUCTION_CAP - 1  # 19
    # Total bid: 1 + 19 = 20

    # ACQUISITION phase: 157-210 (54 actions)
    layout.acquisition_start = offset
    layout.acq_price_base = offset  # +price_offset (0-50)
    offset += ACQ_PRICE_RANGE  # 51
    layout.acq_fi_high = offset
    offset += 1
    layout.acq_fi_face = offset
    offset += 1
    layout.acq_pass = offset
    offset += 1
    # Total acquisition: 51 + 1 + 1 + 1 = 54

    # CLOSING phase: 211-212 (2 actions)
    layout.closing_start = offset
    layout.close_action = offset
    offset += 1
    layout.close_pass = offset
    offset += 1
    # Total closing: 2

    # DIVIDENDS phase: 213-238 (26 actions)
    layout.dividends_start = offset
    layout.dividend_base = offset  # +amount (0-25)
    offset += MAX_DIVIDEND
    # Total dividends: 26

    # ISSUE phase: 239-240 (2 actions)
    layout.issue_start = offset
    layout.issue_pass = offset
    offset += 1
    layout.issue_action = offset
    offset += 1
    # Total issue: 2

    # IPO phase: 241-305 (65 actions)
    layout.ipo_start = offset
    layout.ipo_pass = offset
    offset += 1
    layout.ipo_base = offset  # corp_id * MAX_PAR_SLOTS + par_slot
    offset += NUM_CORPS * MAX_PAR_SLOTS  # 64
    # Total IPO: 1 + 64 = 65

    return layout


# =============================================================================
# ACTION DECODING
# =============================================================================

cdef ActionInfo decode_action(ActionLayout* layout, int action_idx) noexcept nogil:
    """Decode an action index into an ActionInfo struct."""
    cdef ActionInfo info
    cdef int auction_offset, ipo_offset

    info.phase = -1
    info.action_type = ACTION_PASS
    info.slot = -1
    info.corp_id = -1
    info.amount = -1

    # Check bounds
    if action_idx < 0 or action_idx >= layout.total_size:
        return info

    # INVEST phase
    if action_idx < layout.bid_start:
        info.phase = PHASE_INVEST
        if action_idx == layout.pass_invest:
            info.action_type = ACTION_PASS
        elif action_idx < layout.buy_share_base:
            # Auction action
            info.action_type = ACTION_AUCTION
            auction_offset = action_idx - layout.auction_base
            info.slot = auction_offset // AUCTION_CAP
            info.amount = auction_offset % AUCTION_CAP  # bid_offset
        elif action_idx < layout.sell_share_base:
            # Buy share
            info.action_type = ACTION_BUY_SHARE
            info.corp_id = action_idx - layout.buy_share_base
        else:
            # Sell share
            info.action_type = ACTION_SELL_SHARE
            info.corp_id = action_idx - layout.sell_share_base
        return info

    # BID phase
    if action_idx < layout.acquisition_start:
        info.phase = PHASE_BID_IN_AUCTION
        if action_idx == layout.leave_auction:
            info.action_type = ACTION_LEAVE_AUCTION
        else:
            info.action_type = ACTION_RAISE_BID
            info.amount = action_idx - layout.raise_bid_base  # 0-18 -> bid face+1 to face+19
        return info

    # ACQUISITION phase
    if action_idx < layout.closing_start:
        info.phase = PHASE_ACQUISITION
        if action_idx < layout.acq_fi_high:
            info.action_type = ACTION_ACQ_PRICE
            info.amount = action_idx - layout.acq_price_base  # 0-50
        elif action_idx == layout.acq_fi_high:
            info.action_type = ACTION_ACQ_FI_HIGH
        elif action_idx == layout.acq_fi_face:
            info.action_type = ACTION_ACQ_FI_FACE
        else:
            info.action_type = ACTION_PASS
        return info

    # CLOSING phase
    if action_idx < layout.dividends_start:
        info.phase = PHASE_CLOSING
        if action_idx == layout.close_action:
            info.action_type = ACTION_CLOSE
        else:
            info.action_type = ACTION_PASS
        return info

    # DIVIDENDS phase
    if action_idx < layout.issue_start:
        info.phase = PHASE_DIVIDENDS
        info.action_type = ACTION_DIVIDEND
        info.amount = action_idx - layout.dividend_base  # 0-25
        return info

    # ISSUE phase
    if action_idx < layout.ipo_start:
        info.phase = PHASE_ISSUE_SHARES
        if action_idx == layout.issue_pass:
            info.action_type = ACTION_PASS
        else:
            info.action_type = ACTION_ISSUE
        return info

    # IPO phase
    info.phase = PHASE_IPO
    if action_idx == layout.ipo_pass:
        info.action_type = ACTION_PASS
    else:
        info.action_type = ACTION_IPO
        ipo_offset = action_idx - layout.ipo_base
        info.corp_id = ipo_offset // MAX_PAR_SLOTS
        info.slot = ipo_offset % MAX_PAR_SLOTS  # par_slot
    return info


# =============================================================================
# MASK GENERATION HELPERS
# =============================================================================

cdef void _fill_invest_mask(GameState state, ActionLayout* layout, float* mask) noexcept:
    """Fill mask for INVEST phase actions."""
    cdef int slot, company_id, bid_offset, corp_id
    cdef int face_value, player_cash, bank_shares, player_shares
    cdef int player_id = state._get_active_player()
    cdef float* player = state._player_ptr(player_id)
    cdef PlayerOffsets po = get_player_offsets(state._num_players)
    cdef int num_auction_slots = state._num_players  # Dynamic based on player count
    # Round-trip tracking variables (INV-17)
    cdef int buys, sells, roundtrips
    cdef bint roundtrip_blocked

    # Pass is always valid
    mask[layout.pass_invest] = 1.0

    # Auction: slot-based indexing (num_players slots)
    player_cash = state.get_player_cash(player_id)
    for slot in range(num_auction_slots):
        company_id = get_auction_company_for_slot(state, slot)
        if company_id < 0:
            break  # No more slots
        face_value = get_company_face_value(company_id)
        for bid_offset in range(AUCTION_CAP):
            if face_value + bid_offset <= player_cash:
                mask[layout.auction_base + slot * AUCTION_CAP + bid_offset] = 1.0
            else:
                break  # Can't afford higher bids

    # Buy/Sell share: corp_id indexing
    cdef int current_price_index, buy_index, buy_price
    cdef float* market_ptr = state._data + state._layout.market_offset
    for corp_id in range(NUM_CORPS):
        # Round-trip limit check (INV-17)
        buys = get_share_buys(player, &po, corp_id)
        sells = get_share_sells(player, &po, corp_id)
        roundtrips = (buys + sells) // 2
        roundtrip_blocked = roundtrips >= 2  # MAX_ROUNDTRIPS

        # Buy: corp is active, has bank shares, player can afford, not roundtrip blocked
        if state.is_corp_active(corp_id) and not roundtrip_blocked:
            bank_shares = state.get_corp_bank_shares(corp_id)
            if bank_shares > 0:
                # Check if player can afford the buy price (next higher index)
                current_price_index = state.get_corp_price_index(corp_id)
                # Find next higher available market space
                buy_index = current_price_index + 1
                while buy_index < NUM_MARKET_SPACES and market_ptr[buy_index] != 1.0:
                    buy_index += 1
                if buy_index >= NUM_MARKET_SPACES:
                    buy_index = NUM_MARKET_SPACES - 1  # Max index
                buy_price = get_market_price(buy_index)
                if player_cash >= buy_price:
                    mask[layout.buy_share_base + corp_id] = 1.0

        # Sell: player owns shares, not roundtrip blocked
        if not roundtrip_blocked:
            player_shares = get_player_shares(player, &po, corp_id)
            if player_shares > 0:
                mask[layout.sell_share_base + corp_id] = 1.0


cdef void _fill_bid_mask(GameState state, ActionLayout* layout, float* mask) noexcept:
    """Fill mask for BID_IN_AUCTION phase actions."""
    cdef int player_id = state._get_active_player()
    cdef int player_cash = state.get_player_cash(player_id)
    cdef int company_id = state.get_auction_company()
    cdef int current_bid = state.get_auction_price()
    cdef int face_value, bid_offset, new_bid

    # Leave auction is always valid
    mask[layout.leave_auction] = 1.0

    if company_id < 0:
        return

    face_value = get_company_face_value(company_id)

    # Raise bid: offset 0-18 represents bid face+1 to face+19
    # Must beat current bid
    for bid_offset in range(AUCTION_CAP - 1):
        new_bid = face_value + bid_offset + 1  # +1 because min raise is face+1
        if new_bid > current_bid and new_bid <= player_cash:
            mask[layout.raise_bid_base + bid_offset] = 1.0


cdef void _fill_acquisition_mask(GameState state, ActionLayout* layout, float* mask) noexcept:
    """Fill mask for ACQUISITION phase actions."""
    cdef int corp_id = state.get_acq_active_corp()
    cdef int company_id = state.get_acq_target_company()
    cdef int low_price, high_price, corp_cash, offset, price

    if corp_id < 0 or company_id < 0:
        return

    # Pass is always valid
    mask[layout.acq_pass] = 1.0

    if state.is_acq_fi_offer():
        # FI offer: only specific buy actions valid
        if corp_id == CORP_OS:
            # OS buys at face value
            if state.get_corp_cash(corp_id) >= get_company_face_value(company_id):
                mask[layout.acq_fi_face] = 1.0
        else:
            # Others buy at high price
            if state.get_corp_cash(corp_id) >= get_company_high_price(company_id):
                mask[layout.acq_fi_high] = 1.0
    else:
        # General acquisition: price offsets based on affordability
        low_price = get_company_low_price(company_id)
        high_price = get_company_high_price(company_id)
        corp_cash = state.get_corp_cash(corp_id)
        for offset in range(high_price - low_price + 1):
            price = low_price + offset
            if price <= corp_cash:
                mask[layout.acq_price_base + offset] = 1.0


cdef void _fill_closing_mask(GameState state, ActionLayout* layout, float* mask) noexcept:
    """Fill mask for CLOSING phase actions."""
    cdef int company_id = state.get_current_closing_company()
    if company_id >= 0:
        mask[layout.close_action] = 1.0
        mask[layout.close_pass] = 1.0


cdef void _fill_dividends_mask(GameState state, ActionLayout* layout, float* mask) noexcept:
    """Fill mask for DIVIDENDS phase actions."""
    # Get current corp being processed (from dividend_corp one-hot)
    cdef float* turn = state._turn_ptr()
    cdef int corp_id, amount
    cdef int max_dividend

    # Find active dividend corp
    for corp_id in range(NUM_CORPS):
        if turn[state._turn.dividend_corp + corp_id] == 1.0:
            break
    else:
        return  # No active corp

    # Calculate max dividend (simplified - corp cash / issued shares)
    cdef int corp_cash = state.get_corp_cash(corp_id)
    cdef int issued_shares = state.get_corp_issued_shares(corp_id)
    if issued_shares > 0:
        max_dividend = corp_cash // issued_shares
    else:
        max_dividend = 0

    # Mark valid dividend amounts
    for amount in range(min(max_dividend + 1, MAX_DIVIDEND)):
        mask[layout.dividend_base + amount] = 1.0


cdef void _fill_issue_mask(GameState state, ActionLayout* layout, float* mask) noexcept:
    """Fill mask for ISSUE_SHARES phase actions."""
    cdef float* turn = state._turn_ptr()
    cdef int corp_id

    # Find active issue corp
    for corp_id in range(NUM_CORPS):
        if turn[state._turn.issue_corp + corp_id] == 1.0:
            break
    else:
        return

    # Check if corp has unissued shares
    if state.get_corp_unissued_shares(corp_id) > 0:
        mask[layout.issue_action] = 1.0

    # Pass is always valid (unless in receivership with unissued shares)
    if not state.is_corp_in_receivership(corp_id) or state.get_corp_unissued_shares(corp_id) == 0:
        mask[layout.issue_pass] = 1.0


cdef void _fill_ipo_mask(GameState state, ActionLayout* layout, float* mask) noexcept:
    """Fill mask for IPO phase actions."""
    cdef float* turn = state._turn_ptr()
    cdef int company_id, corp_id, par_slot, par_index, par_price, market_index
    cdef int star_tier, face_value, player_cash, cost, player_shares

    # Pass is always valid
    mask[layout.ipo_pass] = 1.0

    # Find current IPO company
    for company_id in range(NUM_COMPANIES):
        if turn[state._turn.ipo_company + company_id] == 1.0:
            break
    else:
        return  # No active company

    star_tier = get_company_stars(company_id)
    face_value = get_company_face_value(company_id)
    player_cash = state.get_player_cash(state._get_active_player())

    for corp_id in range(NUM_CORPS):
        if state.is_corp_active(corp_id):
            continue  # Skip active corps

        for par_slot in range(MAX_PAR_SLOTS):
            par_index = get_par_index_for_slot(star_tier, par_slot)
            if par_index < 0:
                break  # No more valid par prices for this tier

            par_price = get_par_price(par_index)
            market_index = get_market_index(par_price)

            # Check if market space is available
            if market_index < 0 or not state.is_market_space_available(market_index):
                continue

            # Calculate cost: (player_shares * par_price) - face_value
            if par_price >= face_value:
                player_shares = 1
            else:
                player_shares = 2
            cost = (player_shares * par_price) - face_value

            if cost <= player_cash:
                mask[layout.ipo_base + corp_id * MAX_PAR_SLOTS + par_slot] = 1.0


cdef void _fill_mask_for_phase(GameState state, int phase, ActionLayout* layout, float* mask) noexcept:
    """Fill mask based on current phase. Central dispatch to avoid duplication."""
    if phase == PHASE_INVEST:
        _fill_invest_mask(state, layout, mask)
    elif phase == PHASE_BID_IN_AUCTION:
        _fill_bid_mask(state, layout, mask)
    elif phase == PHASE_ACQUISITION:
        _fill_acquisition_mask(state, layout, mask)
    elif phase == PHASE_CLOSING:
        _fill_closing_mask(state, layout, mask)
    elif phase == PHASE_DIVIDENDS:
        _fill_dividends_mask(state, layout, mask)
    elif phase == PHASE_ISSUE_SHARES:
        _fill_issue_mask(state, layout, mask)
    elif phase == PHASE_IPO:
        _fill_ipo_mask(state, layout, mask)


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

cpdef int get_total_action_count(int num_players):
    """Return total action space size for given player count."""
    return get_total_actions_for_players(num_players)


cpdef object get_valid_action_mask(GameState state):
    """
    Generate valid action mask for current game state.

    Returns numpy float32 array where:
    - 1.0 = valid action
    - 0.0 = invalid action

    Size depends on player count: 186 + (num_players * 20)
    """
    cdef int num_players = state._num_players
    cdef ActionLayout layout = compute_action_layout(num_players)
    cdef int total_actions = layout.total_size
    cdef cnp.ndarray mask = np.zeros(total_actions, dtype=np.float32)
    cdef float* mask_ptr = <float*>cnp.PyArray_DATA(mask)

    cdef int phase = state.get_phase()
    _fill_mask_for_phase(state, phase, &layout, mask_ptr)

    return mask


cpdef tuple get_forced_action(GameState state):
    """
    Check if there's only one valid action (forced move).

    Returns:
        (action_idx, True) if exactly one valid action (forced)
        (-1, False) if zero or multiple valid actions
    """
    cdef int num_players = state._num_players
    cdef ActionLayout layout = compute_action_layout(num_players)
    cdef int total_actions = layout.total_size
    cdef cnp.ndarray mask = np.zeros(total_actions, dtype=np.float32)
    cdef float* mask_ptr = <float*>cnp.PyArray_DATA(mask)
    cdef int phase = state.get_phase()
    cdef int i, count, single_action

    # Fill the mask based on phase
    _fill_mask_for_phase(state, phase, &layout, mask_ptr)

    # Count valid actions
    count = 0
    single_action = -1
    for i in range(total_actions):
        if mask_ptr[i] == 1.0:
            count += 1
            if count == 1:
                single_action = i
            elif count > 1:
                return (-1, False)  # Multiple valid actions

    if count == 1:
        return (single_action, True)
    return (-1, False)


cpdef tuple decode_action_py(int action_idx, int num_players):
    """
    Python-accessible action decoding.

    Returns tuple: (phase, action_type, slot, corp_id, amount)
    """
    cdef ActionLayout layout = compute_action_layout(num_players)
    cdef ActionInfo info = decode_action(&layout, action_idx)
    return (info.phase, info.action_type, info.slot, info.corp_id, info.amount)


cpdef dict get_action_layout(int num_players):
    """Get action layout as a dictionary for Python access."""
    cdef ActionLayout layout = compute_action_layout(num_players)
    return {
        'total_size': layout.total_size,
        'invest_start': layout.invest_start,
        'bid_start': layout.bid_start,
        'acquisition_start': layout.acquisition_start,
        'closing_start': layout.closing_start,
        'dividends_start': layout.dividends_start,
        'issue_start': layout.issue_start,
        'ipo_start': layout.ipo_start,
        'pass_invest': layout.pass_invest,
        'auction_base': layout.auction_base,
        'buy_share_base': layout.buy_share_base,
        'sell_share_base': layout.sell_share_base,
        'leave_auction': layout.leave_auction,
        'raise_bid_base': layout.raise_bid_base,
        'acq_price_base': layout.acq_price_base,
        'acq_fi_high': layout.acq_fi_high,
        'acq_fi_face': layout.acq_fi_face,
        'acq_pass': layout.acq_pass,
        'close_action': layout.close_action,
        'close_pass': layout.close_pass,
        'dividend_base': layout.dividend_base,
        'issue_pass': layout.issue_pass,
        'issue_action': layout.issue_action,
        'ipo_pass': layout.ipo_pass,
        'ipo_base': layout.ipo_base,
    }


def get_constants():
    """Get constants for Python tests."""
    return {
        'AUCTION_CAP': AUCTION_CAP,
        'MAX_PAR_SLOTS': MAX_PAR_SLOTS,
        'ACQ_PRICE_RANGE': ACQ_PRICE_RANGE,
        'MAX_DIVIDEND': MAX_DIVIDEND,
    }


# Python-accessible action type constants for testing
ACTION_PASS_PY = ACTION_PASS
ACTION_AUCTION_PY = ACTION_AUCTION
ACTION_BUY_SHARE_PY = ACTION_BUY_SHARE
ACTION_SELL_SHARE_PY = ACTION_SELL_SHARE
ACTION_LEAVE_AUCTION_PY = ACTION_LEAVE_AUCTION
ACTION_RAISE_BID_PY = ACTION_RAISE_BID
ACTION_ACQ_PRICE_PY = ACTION_ACQ_PRICE
ACTION_ACQ_FI_HIGH_PY = ACTION_ACQ_FI_HIGH
ACTION_ACQ_FI_FACE_PY = ACTION_ACQ_FI_FACE
ACTION_CLOSE_PY = ACTION_CLOSE
ACTION_DIVIDEND_PY = ACTION_DIVIDEND
ACTION_ISSUE_PY = ACTION_ISSUE
ACTION_IPO_PY = ACTION_IPO
