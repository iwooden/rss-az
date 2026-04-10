"""
Per-phase action encoding — skeleton implementation.

See ``core/actions.pxd`` for the full contract. This file supplies:

  - concrete encode/decode arithmetic for every decision phase
  - the engine → decision phase bridge
  - *stub* legal-action enumeration (returns 0, asserts if called for now)
  - Python-accessible wrappers for tests

The legal-enumeration helpers (``_enumerate_invest``, ``_enumerate_bid``, …)
are intentionally empty. They are the second-pass work that ports mask
generation out of ``actions-old.pyx`` into phase-local sparse form. The
encoding layer defined here is frozen against the ``ActionSize`` enum in
``core/data.pxd``; the roundtrip self-check below catches drift between the
``encode_*`` arithmetic and those sizes.
"""

cimport cython
import numpy as np
cimport numpy as cnp
from collections import namedtuple

from libc.stdint cimport int16_t, uint16_t

from core.state cimport (
    GameState,
    LAYOUT,
    TURN_OFFSETS,
    PLAYER_FIELDS,
    CORP_FIELDS,
    COMPANY_OFFSETS,
)
from core.data cimport (
    GameConstants,
    GamePhases,
    ACTION_SIZE_INVEST,
    ACTION_SIZE_BID,
    ACTION_SIZE_ACQUISITION,
    ACTION_SIZE_ACQ_OFFER,
    ACTION_SIZE_CLOSING,
    ACTION_SIZE_DIVIDENDS,
    ACTION_SIZE_ISSUE,
    ACTION_SIZE_IPO,
    DPHASE_INVEST,
    DPHASE_BID,
    DPHASE_ACQUISITION,
    DPHASE_ACQ_OFFER,
    DPHASE_CLOSING,
    DPHASE_DIVIDENDS,
    DPHASE_ISSUE,
    DPHASE_IPO,
    ENGINE_TO_DECISION_PHASE,
    COMPANY_FACE_VALUE,
    MARKET_PRICES,
)
from entities.company cimport LOC_AUCTION

cnp.import_array()

# Python-accessible namedtuple mirror of the C-level ActionInfo struct.
ActionInfoTuple = namedtuple('ActionInfoTuple', [
    'phase', 'action_type', 'corp_id', 'company_id', 'amount',
])


# =============================================================================
# PER-PHASE SIZE LOOKUP TABLE
# =============================================================================

# Populated at module import by ``_init_phase_action_sizes``. The pxd
# declares this as ``cdef int _PHASE_ACTION_SIZES_C[8]``; the init function
# fills it from the ``ActionSize`` enum members cimported from ``core.data``
# so phase sizes are readable via a single indexed lookup rather than a
# switch. The table name is distinct from ``core.data.PHASE_ACTION_SIZES``
# (the Python list) to avoid confusion.
cdef void _init_phase_action_sizes() noexcept nogil:
    _PHASE_ACTION_SIZES_C[<int>DPHASE_INVEST] = ACTION_SIZE_INVEST
    _PHASE_ACTION_SIZES_C[<int>DPHASE_BID] = ACTION_SIZE_BID
    _PHASE_ACTION_SIZES_C[<int>DPHASE_ACQUISITION] = ACTION_SIZE_ACQUISITION
    _PHASE_ACTION_SIZES_C[<int>DPHASE_ACQ_OFFER] = ACTION_SIZE_ACQ_OFFER
    _PHASE_ACTION_SIZES_C[<int>DPHASE_CLOSING] = ACTION_SIZE_CLOSING
    _PHASE_ACTION_SIZES_C[<int>DPHASE_DIVIDENDS] = ACTION_SIZE_DIVIDENDS
    _PHASE_ACTION_SIZES_C[<int>DPHASE_ISSUE] = ACTION_SIZE_ISSUE
    _PHASE_ACTION_SIZES_C[<int>DPHASE_IPO] = ACTION_SIZE_IPO


_init_phase_action_sizes()


# =============================================================================
# ENCODE FROM ActionInfo (inverse of decode_action)
# =============================================================================

cdef int encode_action(ActionInfo info) noexcept nogil:
    """Repack an ``ActionInfo`` into its phase-local action id.

    Used by the encode/decode roundtrip test and by replay diagnostics. The
    decoder is the canonical direction in production code; this function
    exists to let tests verify the inverse is well-defined.

    Asserts on illegal combinations — an ``ActionInfo`` with a phase and
    action_type that do not correspond (e.g. phase=INVEST, type=CLOSE) is a
    programmer error and should crash loudly.
    """
    cdef int phase = info.phase

    if phase == DPHASE_INVEST:
        if info.action_type == ACTION_PASS:
            return 0
        if info.action_type == ACTION_AUCTION:
            return encode_invest_auction(info.company_id, info.amount)
        if info.action_type == ACTION_BUY_SHARE:
            return encode_invest_buy(info.corp_id)
        if info.action_type == ACTION_SELL_SHARE:
            return encode_invest_sell(info.corp_id)

    elif phase == DPHASE_BID:
        if info.action_type == ACTION_PASS:
            return 0
        if info.action_type == ACTION_RAISE:
            return encode_bid_raise(info.amount)

    elif phase == DPHASE_ACQUISITION:
        if info.action_type == ACTION_PASS:
            return 0
        if info.action_type == ACTION_ACQ_PRICE:
            return encode_acquisition_price(info.corp_id, info.company_id, info.amount)
        if info.action_type == ACTION_ACQ_FI_BUY:
            return encode_acquisition_fi_buy(info.corp_id, info.company_id)

    elif phase == DPHASE_ACQ_OFFER:
        if info.action_type == ACTION_PASS:
            return 0
        if info.action_type == ACTION_ACQ_OFFER_BUY:
            return 1

    elif phase == DPHASE_CLOSING:
        if info.action_type == ACTION_PASS:
            return 0
        if info.action_type == ACTION_CLOSE:
            return encode_closing_close(info.company_id)

    elif phase == DPHASE_DIVIDENDS:
        if info.action_type == ACTION_DIVIDEND:
            return info.amount

    elif phase == DPHASE_ISSUE:
        if info.action_type == ACTION_PASS:
            return 0
        if info.action_type == ACTION_ISSUE:
            return 1

    elif phase == DPHASE_IPO:
        if info.action_type == ACTION_PASS:
            return 0
        if info.action_type == ACTION_IPO:
            return encode_ipo(info.corp_id, info.amount)

    assert False, f"encode_action: illegal (phase={phase}, type={info.action_type})"


# =============================================================================
# DECODE
# =============================================================================

cdef ActionInfo decode_action(int phase_id, int action_id) noexcept nogil:
    """Decode a phase-local action id into an ``ActionInfo`` struct.

    Inverse of the ``encode_*`` helpers. The returned struct has all fields
    populated; fields that don't apply for the action type are set to -1.
    """
    cdef ActionInfo info
    cdef int idx, pair, k

    info.phase = phase_id
    info.action_type = ACTION_PASS
    info.corp_id = -1
    info.company_id = -1
    info.amount = -1

    # --- INVEST -----------------------------------------------------------
    if phase_id == DPHASE_INVEST:
        if action_id == 0:
            info.action_type = ACTION_PASS
        elif action_id < 541:
            # Auction: 1 + company_id * 15 + bid_offset
            info.action_type = ACTION_AUCTION
            idx = action_id - 1
            info.company_id = idx // 15
            info.amount = idx % 15
        else:
            # Trade: 541 + corp_id * 2 + {0=buy, 1=sell}
            idx = action_id - 541
            info.corp_id = idx // 2
            if (idx & 1) == 0:
                info.action_type = ACTION_BUY_SHARE
            else:
                info.action_type = ACTION_SELL_SHARE
        return info

    # --- BID --------------------------------------------------------------
    if phase_id == DPHASE_BID:
        if action_id == 0:
            # "Leave the auction" is a pass-class action.
            info.action_type = ACTION_PASS
        else:
            info.action_type = ACTION_RAISE
            info.amount = action_id - 1
        return info

    # --- ACQUISITION ------------------------------------------------------
    if phase_id == DPHASE_ACQUISITION:
        if action_id == 0:
            info.action_type = ACTION_PASS
            return info
        idx = action_id - 1
        pair = idx // 52            # corp * 36 + company
        k = idx % 52
        info.corp_id = pair // 36
        info.company_id = pair % 36
        if k < 51:
            info.action_type = ACTION_ACQ_PRICE
            info.amount = k
        else:
            info.action_type = ACTION_ACQ_FI_BUY
        return info

    # --- ACQ_OFFER --------------------------------------------------------
    if phase_id == DPHASE_ACQ_OFFER:
        if action_id == 0:
            info.action_type = ACTION_PASS
        else:
            info.action_type = ACTION_ACQ_OFFER_BUY
        return info

    # --- CLOSING ----------------------------------------------------------
    if phase_id == DPHASE_CLOSING:
        if action_id == 0:
            info.action_type = ACTION_PASS
        else:
            info.action_type = ACTION_CLOSE
            info.company_id = action_id - 1
        return info

    # --- DIVIDENDS --------------------------------------------------------
    if phase_id == DPHASE_DIVIDENDS:
        info.action_type = ACTION_DIVIDEND
        info.amount = action_id
        return info

    # --- ISSUE ------------------------------------------------------------
    if phase_id == DPHASE_ISSUE:
        if action_id == 0:
            info.action_type = ACTION_PASS
        else:
            info.action_type = ACTION_ISSUE
        return info

    # --- IPO --------------------------------------------------------------
    if phase_id == DPHASE_IPO:
        if action_id == 0:
            info.action_type = ACTION_PASS
            return info
        idx = action_id - 1
        info.action_type = ACTION_IPO
        info.corp_id = idx // 14
        info.amount = idx % 14
        return info

    assert False, f"decode_action: unknown phase {phase_id}"


# =============================================================================
# DECISION PHASE BRIDGE
# =============================================================================

cdef int get_decision_phase(GameState state) noexcept nogil:
    """Read the engine phase from ``state`` and map it to a DecisionPhase.

    Returns -1 for automated/terminal engine phases (WRAP_UP, INCOME,
    END_CARD, GAME_OVER). Callers are expected to only invoke enumeration /
    decoding for phases where the driver actually asks for an action.

    Reads the phase slot directly from the state buffer and indexes the
    ``ENGINE_TO_DECISION_PHASE`` table cimported from ``core.data`` — no
    Python singleton access, the whole function stays nogil.
    """
    cdef int engine_phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]
    assert 0 <= engine_phase < 12, f"corrupt engine phase: {engine_phase}"
    return ENGINE_TO_DECISION_PHASE[engine_phase]


# =============================================================================
# SPARSE LEGAL-ACTION ENUMERATION (skeleton)
# =============================================================================
#
# The actual rule-level legality logic is deferred to a follow-up pass. For
# now each helper is a stub that writes nothing and returns 0. The driver
# and eval server should treat "0 legal actions" as "engine rewrite for
# this phase is incomplete" and refuse to proceed, rather than mistaking it
# for a legitimate stuck state.
#
# When the second pass lands, these stubs become the central point of
# truth for legality. Port the mask-generation logic out of
# ``core/actions-old.pyx`` into these helpers, re-keyed to the new sparse
# encoding. Keep the enumeration order deterministic and documented per
# phase (replay targets depend on it).

cdef int _enumerate_invest(
    GameState state, uint16_t* ids,
) noexcept nogil:
    """Emit every legal DPHASE_INVEST action in deterministic order.

    Ordering (phase-local IDs match the ``encode_invest_*`` helpers in
    ``core/actions.pxd``):

      1. ``id 0``: pass (always legal)
      2. ``ids 1..540``: auction starts, ``1 + company_id * 15 + bid_offset``.
         Emitted for each LOC_AUCTION company, for each affordable bid offset.
         Offsets are monotone in cost so the inner loop breaks as soon as
         ``face_value + offset`` exceeds the active player's cash.
      3. ``ids 541..556``: trade actions, ``541 + corp_id * 2 + {0=buy,1=sell}``.
         Emitted for each active corp, gated on the round-trip limit
         (per-player ``min(buys, sells) < 2`` over that corp), then on
         buy/sell-specific legality:
           - BUY: bank has a share AND player can afford the *next higher*
             market price after price movement.
           - SELL: player owns a share.

    Returns the number of IDs written.

    Practical upper bound: ``1 + num_players * 15 + 16``. Worst case at
    the game's maximum of 6 players is 107 — well under
    ``MAX_LEGAL_ACTIONS = 256``. The naive bound of 557 is unreachable
    because only ``num_players`` companies are LOC_AUCTION at any time.

    Reads state via direct slot arithmetic against the module-level
    ``LAYOUT`` / ``PLAYER_FIELDS`` / ``CORP_FIELDS`` / ``COMPANY_OFFSETS``
    structs cimported from ``core.state``. This is the intentional
    escape hatch for the per-phase enumerators: they run at peak MCTS
    hot-path frequency, so they bypass the entity-handle dispatch layer
    to stay in one nogil-clean C translation unit. Phase *handlers* in
    ``phases/`` still go through entity handles — the layout-constant
    imports are scoped to ``core/actions.pyx``.
    """
    cdef int count = 0
    cdef int max_index = <int>GameConstants.NUM_MARKET_SPACES - 1  # 26
    cdef int player_id = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_player
    ]
    cdef int player_base = LAYOUT.players_offset + player_id * PLAYER_FIELDS.size
    cdef int player_cash = <int>state._data[player_base + PLAYER_FIELDS.cash]
    cdef int16_t* market_ptr = state._data + LAYOUT.market_offset

    cdef int company_id, bid_offset, face_value, loc
    cdef int corp_id, corp_base, buys, sells, current_index, buy_index, i

    # --- id 0: pass ---------------------------------------------------------
    ids[count] = 0
    count += 1

    # --- ids 1..540: auctions ----------------------------------------------
    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        loc = <int>state._data[
            LAYOUT.companies_offset + COMPANY_OFFSETS.locations + company_id
        ]
        if loc != <int>LOC_AUCTION:
            continue
        face_value = COMPANY_FACE_VALUE[company_id]
        if face_value > player_cash:
            # Can't afford even the opening bid; no legal offset exists.
            continue
        for bid_offset in range(15):
            if face_value + bid_offset > player_cash:
                # Monotone in offset — once unaffordable, every higher
                # offset is also unaffordable.
                break
            ids[count] = <uint16_t>encode_invest_auction(company_id, bid_offset)
            count += 1

    # --- ids 541..556: buy / sell ------------------------------------------
    for corp_id in range(<int>GameConstants.NUM_CORPS):
        # Round-trip gate applies to *both* buy and sell: a player who has
        # already completed 2 paired buy/sells against this corp is locked
        # out of further trades on it for the rest of the INVEST phase.
        buys = <int>state._data[
            player_base + PLAYER_FIELDS.share_buys + corp_id
        ]
        sells = <int>state._data[
            player_base + PLAYER_FIELDS.share_sells + corp_id
        ]
        if (buys if buys < sells else sells) >= 2:
            continue

        corp_base = LAYOUT.corps_offset + corp_id * CORP_FIELDS.size
        if <int>state._data[corp_base + CORP_FIELDS.active] != 1:
            # Inactive corps have no tradable shares.
            continue

        # --- BUY ------------------------------------------------------------
        if <int>state._data[corp_base + CORP_FIELDS.bank_shares] > 0:
            # Inlined next-higher scan. Matches Market._find_next_higher_space
            # exactly: walk from current_index+1 up to but not including
            # max_index, fall through to max_index as the always-available
            # $75 sentinel.
            current_index = <int>state._data[corp_base + CORP_FIELDS.price_index]
            buy_index = max_index
            for i in range(current_index + 1, max_index):
                if market_ptr[i] == 1:
                    buy_index = i
                    break
            if player_cash >= MARKET_PRICES[buy_index]:
                ids[count] = <uint16_t>encode_invest_buy(corp_id)
                count += 1

        # --- SELL -----------------------------------------------------------
        if <int>state._data[
            player_base + PLAYER_FIELDS.owned_shares + corp_id
        ] > 0:
            ids[count] = <uint16_t>encode_invest_sell(corp_id)
            count += 1

    return count


cdef int _enumerate_bid(
    GameState state, uint16_t* ids,
) noexcept nogil:
    """Emit every legal DPHASE_BID action in deterministic order.

    Ordering (phase-local IDs match the ``encode_bid_*`` helpers in
    ``core/actions.pxd``):

      1. ``id 0``: leave the auction (always legal — decodes to
         ``ACTION_PASS``).
      2. ``ids 1..14``: raise to ``face_value + 1 + offset``, emitted for
         each legal ``offset in 0..13``. Both the strictly-greater-than
         current bid gate and the affordability gate are monotone in
         ``offset``, so the inner loop starts at the smallest legal
         offset and breaks as soon as the new bid exceeds the active
         player's cash.

    Returns the number of IDs written. Worst case: 1 + 14 = 15 — well
    under ``MAX_LEGAL_ACTIONS = 256``.

    Reads state via direct slot arithmetic against the module-level
    layout constants, matching the ``_enumerate_invest`` escape-hatch
    pattern — this module is the one place allowed to cimport
    ``LAYOUT`` / ``PLAYER_FIELDS`` / ``TURN_OFFSETS`` / ``COMPANY_OFFSETS``.
    """
    cdef int count = 0
    cdef int player_id = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_player
    ]
    cdef int player_base = LAYOUT.players_offset + player_id * PLAYER_FIELDS.size
    cdef int player_cash = <int>state._data[player_base + PLAYER_FIELDS.cash]
    cdef int company_id = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_company
    ]
    cdef int current_bid = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.auction_price
    ]
    cdef int face, offset, new_bid, min_offset

    # --- id 0: leave the auction (always legal) ----------------------------
    ids[count] = 0
    count += 1

    # A BID state with no active_company is a driver bug — nothing to
    # auction, nothing to raise on.
    assert company_id >= 0, "_enumerate_bid: active_company unset"

    face = COMPANY_FACE_VALUE[company_id]
    # Smallest legal offset: the first ``offset`` where
    # ``face + offset + 1 > current_bid``. Solving gives
    # ``offset >= current_bid - face``; clamp to 0 since negative offsets
    # are not part of the encoding.
    min_offset = current_bid - face
    if min_offset < 0:
        min_offset = 0
    for offset in range(min_offset, 14):
        new_bid = face + offset + 1
        if new_bid > player_cash:
            # Affordability is monotone in offset — once we can't afford
            # the next step, we can't afford any higher one.
            break
        ids[count] = <uint16_t>encode_bid_raise(offset)
        count += 1

    return count


cdef int _enumerate_acquisition(
    GameState state, uint16_t* ids,
) noexcept nogil:
    return 0


cdef int _enumerate_acq_offer(
    GameState state, uint16_t* ids,
) noexcept nogil:
    return 0


cdef int _enumerate_closing(
    GameState state, uint16_t* ids,
) noexcept nogil:
    """Emit every legal DPHASE_CLOSING action in deterministic order.

    Ordering:
      1. ``id 0``: pass (always legal)
      2. ``ids 1..36``: close company_id, ``1 + company_id``.
         Player-owned privates (LOC_PLAYER, owner == active_player) are
         always closable. Corp subsidiaries (LOC_CORP) are closable if
         the active player presides the corp, the corp is not in
         receivership, and the corp retains at least 2 companies.

    Returns the number of IDs written. Worst case: 37 (pass + 36 companies).
    """
    cdef int count = 0
    cdef int active_player = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_player
    ]
    cdef int company_base = LAYOUT.companies_offset
    cdef int loc, owner
    cdef int corp_id, corp_base, company_id
    cdef int player_base = LAYOUT.players_offset + active_player * PLAYER_FIELDS.size
    cdef int corp_company_count[8]

    # --- id 0: pass (always legal) -------------------------------------------
    ids[count] = 0
    count += 1

    # --- Pre-scan: count companies per corp -----------------------------------
    for corp_id in range(8):
        corp_company_count[corp_id] = 0
    for company_id in range(36):
        loc = <int>state._data[company_base + COMPANY_OFFSETS.locations + company_id]
        if loc == 5:  # LOC_CORP
            owner = <int>state._data[company_base + COMPANY_OFFSETS.owner_ids + company_id]
            corp_company_count[owner] += 1

    # --- Player-owned privates ------------------------------------------------
    for company_id in range(36):
        loc = <int>state._data[company_base + COMPANY_OFFSETS.locations + company_id]
        if loc != 3:  # LOC_PLAYER
            continue
        owner = <int>state._data[company_base + COMPANY_OFFSETS.owner_ids + company_id]
        if owner != active_player:
            continue
        ids[count] = <uint16_t>encode_closing_close(company_id)
        count += 1

    # --- Corp subsidiaries (presided by active_player, not in receivership) ---
    for corp_id in range(8):
        corp_base = LAYOUT.corps_offset + corp_id * CORP_FIELDS.size
        if <int>state._data[corp_base + CORP_FIELDS.active] != 1:
            continue
        if <int>state._data[corp_base + CORP_FIELDS.in_receivership] == 1:
            continue
        if <int>state._data[player_base + PLAYER_FIELDS.is_president + corp_id] != 1:
            continue
        if corp_company_count[corp_id] <= 1:
            continue  # Can't close the last company
        for company_id in range(36):
            loc = <int>state._data[company_base + COMPANY_OFFSETS.locations + company_id]
            if loc != 5:  # LOC_CORP
                continue
            owner = <int>state._data[company_base + COMPANY_OFFSETS.owner_ids + company_id]
            if owner != corp_id:
                continue
            ids[count] = <uint16_t>encode_closing_close(company_id)
            count += 1

    return count


cdef int _enumerate_dividends(
    GameState state, uint16_t* ids,
) noexcept nogil:
    return 0


cdef int _enumerate_issue(
    GameState state, uint16_t* ids,
) noexcept nogil:
    return 0


cdef int _enumerate_ipo(
    GameState state, uint16_t* ids,
) noexcept nogil:
    return 0


cdef int enumerate_legal_actions(
    GameState state,
    uint16_t* action_ids,
) noexcept nogil:
    """Fill ``action_ids`` with legal phase-local IDs; return the count.

    Reads the decision phase from ``state`` via ``get_decision_phase``,
    then dispatches to a phase-specific ``_enumerate_*`` helper. Returns
    0 for automated/terminal engine phases. The helpers own the per-phase
    enumeration order contract and all run ``nogil`` via direct
    state-buffer slot arithmetic.
    """
    cdef int phase_id = get_decision_phase(state)
    if phase_id < 0:
        return 0

    cdef int count = 0

    if phase_id == DPHASE_INVEST:
        count = _enumerate_invest(state, action_ids)
    elif phase_id == DPHASE_BID:
        count = _enumerate_bid(state, action_ids)
    elif phase_id == DPHASE_ACQUISITION:
        count = _enumerate_acquisition(state, action_ids)
    elif phase_id == DPHASE_ACQ_OFFER:
        count = _enumerate_acq_offer(state, action_ids)
    elif phase_id == DPHASE_CLOSING:
        count = _enumerate_closing(state, action_ids)
    elif phase_id == DPHASE_DIVIDENDS:
        count = _enumerate_dividends(state, action_ids)
    elif phase_id == DPHASE_ISSUE:
        count = _enumerate_issue(state, action_ids)
    elif phase_id == DPHASE_IPO:
        count = _enumerate_ipo(state, action_ids)

    # Overflow is a bug, not a recoverable condition. Kmax is sized to
    # the worst-case legal set across all phases; exceeding it means the
    # enumerator or the bound is wrong.
    assert count <= MAX_LEGAL_ACTIONS, \
        "legal action count exceeds MAX_LEGAL_ACTIONS — bump Kmax or fix enumerator"
    return count


# =============================================================================
# PYTHON-ACCESSIBLE WRAPPERS
# =============================================================================

cpdef int get_phase_action_size(int phase_id):
    """Return the action-space size for a given DecisionPhase."""
    assert 0 <= phase_id < <int>GameConstants.NUM_DECISION_PHASES, \
        f"invalid decision phase: {phase_id}"
    return _PHASE_ACTION_SIZES_C[phase_id]


cpdef object decode_action_py(int phase_id, int action_id):
    """Python wrapper around ``decode_action``.

    Returns a tuple ``(phase, action_type, corp_id, company_id, amount)``.
    """
    assert 0 <= phase_id < <int>GameConstants.NUM_DECISION_PHASES, \
        f"invalid decision phase: {phase_id}"
    assert 0 <= action_id < _PHASE_ACTION_SIZES_C[phase_id], \
        f"action_id {action_id} out of range for phase {phase_id} " \
        f"(size {_PHASE_ACTION_SIZES_C[phase_id]})"
    cdef ActionInfo info = decode_action(phase_id, action_id)
    return ActionInfoTuple(info.phase, info.action_type, info.corp_id, info.company_id, info.amount)


cpdef int get_decision_phase_py(GameState state):
    """Python wrapper around ``get_decision_phase``."""
    return get_decision_phase(state)


cpdef int enumerate_legal_actions_py(GameState state, cnp.ndarray action_ids):
    """Python wrapper around ``enumerate_legal_actions``.

    Caller supplies a pre-allocated uint16 numpy array of at least
    ``MAX_LEGAL_ACTIONS`` elements. Returns the number of legal action
    IDs written into the buffer.
    """
    cdef uint16_t* buf_ptr = <uint16_t*>cnp.PyArray_DATA(action_ids)
    return enumerate_legal_actions(state, buf_ptr)


# =============================================================================
# MODULE-LEVEL PYTHON CONSTANTS (for tests / external inspection)
# =============================================================================
#
# ``DecisionPhase`` lives on ``core.data`` — Python consumers should use
# ``from core.data import DecisionPhase`` and reference ``DPHASE_*`` as
# attributes. Only constants that are specific to the actions API (action
# type tags, legal-action buffer width) are mirrored here.

MAX_LEGAL_ACTIONS_PY = MAX_LEGAL_ACTIONS

ACTION_PASS_PY = ACTION_PASS
ACTION_AUCTION_PY = ACTION_AUCTION
ACTION_BUY_SHARE_PY = ACTION_BUY_SHARE
ACTION_SELL_SHARE_PY = ACTION_SELL_SHARE
ACTION_RAISE_PY = ACTION_RAISE
ACTION_ACQ_PRICE_PY = ACTION_ACQ_PRICE
ACTION_ACQ_FI_BUY_PY = ACTION_ACQ_FI_BUY
ACTION_ACQ_OFFER_BUY_PY = ACTION_ACQ_OFFER_BUY
ACTION_CLOSE_PY = ACTION_CLOSE
ACTION_DIVIDEND_PY = ACTION_DIVIDEND
ACTION_ISSUE_PY = ACTION_ISSUE
ACTION_IPO_PY = ACTION_IPO


# =============================================================================
# IMPORT-TIME SELF-CHECK
# =============================================================================
#
# Catch encoding drift immediately: verify the computed per-phase max
# action id equals ``size - 1`` for every phase. Any mismatch means the
# ``encode_*`` formulae and the ``ActionSize`` enum in ``core/data.pxd``
# have fallen out of sync, which would silently corrupt replay alignment.
# ``core.data`` is the single source of truth — no cross-module comparison
# to ``nn/transformer.py`` is needed anymore.

assert encode_invest_sell(7) == ACTION_SIZE_INVEST - 1
assert encode_bid_raise(13) == ACTION_SIZE_BID - 1
assert encode_acquisition_fi_buy(7, 35) == ACTION_SIZE_ACQUISITION - 1
assert 1 == ACTION_SIZE_ACQ_OFFER - 1
assert encode_closing_close(35) == ACTION_SIZE_CLOSING - 1
assert 25 == ACTION_SIZE_DIVIDENDS - 1
assert 1 == ACTION_SIZE_ISSUE - 1
assert encode_ipo(7, 13) == ACTION_SIZE_IPO - 1
