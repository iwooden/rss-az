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
from libc.stdio cimport fprintf, stderr
from libc.stdlib cimport abort

from core.state cimport (
    GameState,
    LAYOUT,
    TURN_OFFSETS,
    PLAYER_FIELDS,
)
from core.data cimport (
    GameConstants,
    GamePhases,
    CorpIndices,
    AUCTION_CAP,
    ACTION_SIZE_INVEST,
    ACTION_SIZE_BID,
    ACTION_SIZE_ACQ_SELECT_CORP,
    ACTION_SIZE_ACQ_SELECT_COMPANY,
    ACTION_SIZE_ACQ_SELECT_PRICE,
    ACTION_SIZE_ACQ_OFFER,
    ACTION_SIZE_CLOSING,
    ACTION_SIZE_DIVIDENDS,
    ACTION_SIZE_ISSUE,
    ACTION_SIZE_IPO,
    ACTION_SIZE_PAR,
    MAX_ACTION_SIZE,
    DPHASE_INVEST,
    DPHASE_BID,
    DPHASE_ACQ_SELECT_CORP,
    DPHASE_ACQ_SELECT_COMPANY,
    DPHASE_ACQ_SELECT_PRICE,
    DPHASE_ACQ_OFFER,
    DPHASE_CLOSING,
    DPHASE_DIVIDENDS,
    DPHASE_ISSUE,
    DPHASE_IPO,
    DPHASE_PAR,
    ENGINE_TO_DECISION_PHASE,
    COMPANY_FACE_VALUE,
    COMPANY_LOW_PRICE,
    COMPANY_HIGH_PRICE,
    COMPANY_STARS,
    ALL_PAR_PRICES,
    PAR_PRICE_VALID,
    PRICE_TO_MARKET_INDEX,
    MARKET_PRICES,
)
from entities.company cimport (
    LOC_AUCTION,
    LOC_FI,
    LOC_CORP,
    LOC_PLAYER,
    company_adjusted_income,
    company_location,
    company_owner_id,
    company_owned_by_player,
)
from entities.corp cimport (
    count_corp_companies,
    corp_is_active,
    corp_cash,
    corp_issued_shares,
    corp_bank_shares,
    corp_price_index,
    corp_is_in_receivership,
    corp_president_id,
)
from phases.closing cimport _corp_closable_by_player

cnp.import_array()

# Python-accessible namedtuple mirror of the C-level ActionInfo struct.
ActionInfoTuple = namedtuple('ActionInfoTuple', [
    'phase', 'action_type', 'corp_id', 'company_id', 'amount',
])


# =============================================================================
# PER-PHASE SIZE LOOKUP TABLE
# =============================================================================

# Populated at module import by ``_init_phase_action_sizes``. The pxd
# declares this as ``cdef int _PHASE_ACTION_SIZES_C[11]``; the init function
# fills it from the ``ActionSize`` enum members cimported from ``core.data``
# so phase sizes are readable via a single indexed lookup rather than a
# switch. The table name is distinct from ``core.data.PHASE_ACTION_SIZES``
# (the Python list) to avoid confusion.
cdef void _init_phase_action_sizes() noexcept nogil:
    _PHASE_ACTION_SIZES_C[<int>DPHASE_INVEST] = ACTION_SIZE_INVEST
    _PHASE_ACTION_SIZES_C[<int>DPHASE_BID] = ACTION_SIZE_BID
    _PHASE_ACTION_SIZES_C[<int>DPHASE_ACQ_SELECT_CORP] = ACTION_SIZE_ACQ_SELECT_CORP
    _PHASE_ACTION_SIZES_C[<int>DPHASE_ACQ_SELECT_COMPANY] = ACTION_SIZE_ACQ_SELECT_COMPANY
    _PHASE_ACTION_SIZES_C[<int>DPHASE_ACQ_SELECT_PRICE] = ACTION_SIZE_ACQ_SELECT_PRICE
    _PHASE_ACTION_SIZES_C[<int>DPHASE_ACQ_OFFER] = ACTION_SIZE_ACQ_OFFER
    _PHASE_ACTION_SIZES_C[<int>DPHASE_CLOSING] = ACTION_SIZE_CLOSING
    _PHASE_ACTION_SIZES_C[<int>DPHASE_DIVIDENDS] = ACTION_SIZE_DIVIDENDS
    _PHASE_ACTION_SIZES_C[<int>DPHASE_ISSUE] = ACTION_SIZE_ISSUE
    _PHASE_ACTION_SIZES_C[<int>DPHASE_IPO] = ACTION_SIZE_IPO
    _PHASE_ACTION_SIZES_C[<int>DPHASE_PAR] = ACTION_SIZE_PAR


_init_phase_action_sizes()


cdef inline void _require_action_capacity(int count, const char* context) noexcept nogil:
    """Abort if the sparse legal-action buffer would overflow.

    ``MAX_ACTION_SIZE`` is the tight per-phase upper bound (max over all
    ``ACTION_SIZE_*``). Exceeding it is a configuration bug that must fail
    in optimized builds too, before writing past the caller's fixed-size
    scratch buffer.
    """
    if count >= MAX_ACTION_SIZE:
        fprintf(
            stderr,
            b"enumerate_legal_actions overflow in %s: MAX_ACTION_SIZE=%d\n",
            context,
            <int>MAX_ACTION_SIZE,
        )
        abort()


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
            return encode_invest_auction(info.company_id)
        if info.action_type == ACTION_BUY_SHARE:
            return encode_invest_buy(info.corp_id)
        if info.action_type == ACTION_SELL_SHARE:
            return encode_invest_sell(info.corp_id)

    elif phase == DPHASE_BID:
        if info.action_type == ACTION_PASS:
            return 0
        if info.action_type == ACTION_RAISE:
            return encode_bid_raise(info.amount)

    elif phase == DPHASE_ACQ_SELECT_CORP:
        if info.action_type == ACTION_PASS:
            return 0
        if info.action_type == ACTION_ACQ_SELECT_CORP:
            return encode_acq_select_corp(info.corp_id)

    elif phase == DPHASE_ACQ_SELECT_COMPANY:
        if info.action_type == ACTION_ACQ_SELECT_COMPANY:
            return encode_acq_select_company(info.company_id)

    elif phase == DPHASE_ACQ_SELECT_PRICE:
        if info.action_type == ACTION_ACQ_PRICE:
            return encode_acq_select_price(info.amount)

    elif phase == DPHASE_ACQ_OFFER:
        if info.action_type == ACTION_PASS:
            return 0
        if info.action_type == ACTION_ACQ_OFFER_ACCEPT:
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
            return encode_ipo(info.corp_id)

    elif phase == DPHASE_PAR:
        if info.action_type == ACTION_PAR:
            return encode_par(info.amount)

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
    cdef int idx

    info.phase = phase_id
    info.action_type = ACTION_PASS
    info.corp_id = -1
    info.company_id = -1
    info.amount = -1

    # --- INVEST -----------------------------------------------------------
    if phase_id == DPHASE_INVEST:
        if action_id == 0:
            info.action_type = ACTION_PASS
        elif action_id < 1 + <int>GameConstants.NUM_COMPANIES:
            # Select company: 1 + company_id. Price is chosen in BID.
            info.action_type = ACTION_AUCTION
            info.company_id = action_id - 1
        else:
            # Trade: 1 + NUM_COMPANIES + corp_id * 2 + {0=buy, 1=sell}
            idx = action_id - (1 + <int>GameConstants.NUM_COMPANIES)
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

    # --- ACQ_SELECT_CORP --------------------------------------------------
    if phase_id == DPHASE_ACQ_SELECT_CORP:
        if action_id == 0:
            info.action_type = ACTION_PASS
        else:
            info.action_type = ACTION_ACQ_SELECT_CORP
            info.corp_id = action_id - 1
        return info

    # --- ACQ_SELECT_COMPANY -----------------------------------------------
    if phase_id == DPHASE_ACQ_SELECT_COMPANY:
        info.action_type = ACTION_ACQ_SELECT_COMPANY
        info.company_id = action_id
        return info

    # --- ACQ_SELECT_PRICE -------------------------------------------------
    if phase_id == DPHASE_ACQ_SELECT_PRICE:
        info.action_type = ACTION_ACQ_PRICE
        info.amount = action_id
        return info

    # --- ACQ_OFFER --------------------------------------------------------
    if phase_id == DPHASE_ACQ_OFFER:
        if action_id == 0:
            info.action_type = ACTION_PASS
        else:
            info.action_type = ACTION_ACQ_OFFER_ACCEPT
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
        else:
            info.action_type = ACTION_IPO
            info.corp_id = action_id - 1
        return info

    # --- PAR --------------------------------------------------------------
    if phase_id == DPHASE_PAR:
        info.action_type = ACTION_PAR
        info.amount = action_id
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
    assert 0 <= engine_phase < <int>GameConstants.NUM_PHASES, \
        f"corrupt engine phase: {engine_phase}"
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
      2. ``ids 1..36``: select company, ``1 + company_id``. Emitted for each
         LOC_AUCTION company that the active player can afford at face value
         (the minimum legal opening bid in BID). Price selection itself is
         deferred to the BID phase.
      3. ``ids 37..52``: trade actions, ``37 + corp_id * 2 + {0=buy,1=sell}``.
         Emitted for each active corp, gated on the round-trip limit
         (per-player ``min(buys, sells) < 2`` over that corp), then on
         buy/sell-specific legality:
           - BUY: bank has a share AND player can afford the *next higher*
             market price after price movement.
           - SELL: player owns a share.

    Returns the number of IDs written.

    Practical upper bound: ``1 + num_players + 16``. Worst case at the game's
    maximum of 6 players is 23 — well under ``MAX_ACTION_SIZE``.

    Reads state via direct slot arithmetic and entity-owned primitives
    where available. This is the intentional
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

    cdef int company_id, face_value
    cdef int corp_id, buys, sells, current_index, buy_index, i

    # --- id 0: pass ---------------------------------------------------------
    ids[count] = 0
    count += 1

    # --- ids 1..36: select company ------------------------------------------
    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_location(state, company_id) != <int>LOC_AUCTION:
            continue
        face_value = COMPANY_FACE_VALUE[company_id]
        if face_value > player_cash:
            # Can't afford the minimum opening bid in BID — no legal select.
            continue
        _require_action_capacity(count, b"INVEST_AUCTION")
        ids[count] = <uint16_t>encode_invest_auction(company_id)
        count += 1

    # --- ids 37..52: buy / sell ---------------------------------------------
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
        if min(buys, sells) >= 2:
            continue

        if not corp_is_active(state, corp_id):
            # Inactive corps have no tradable shares.
            continue

        # --- BUY ------------------------------------------------------------
        if corp_bank_shares(state, corp_id) > 0:
            # Inlined next-higher scan. Matches Market._find_next_higher_space
            # exactly: walk from current_index+1 up to but not including
            # max_index, fall through to max_index as the always-available
            # $75 sentinel.
            current_index = corp_price_index(state, corp_id)
            buy_index = max_index
            for i in range(current_index + 1, max_index):
                if market_ptr[i] == 1:
                    buy_index = i
                    break
            if player_cash >= MARKET_PRICES[buy_index]:
                _require_action_capacity(count, b"INVEST_BUY")
                ids[count] = <uint16_t>encode_invest_buy(corp_id)
                count += 1

        # --- SELL -----------------------------------------------------------
        if <int>state._data[
            player_base + PLAYER_FIELDS.owned_shares + corp_id
        ] > 0:
            _require_action_capacity(count, b"INVEST_SELL")
            ids[count] = <uint16_t>encode_invest_sell(corp_id)
            count += 1

    return count


cdef int _enumerate_bid(
    GameState state, uint16_t* ids,
) noexcept nogil:
    """Emit every legal DPHASE_BID action in deterministic order.

    Ordering (phase-local IDs match the ``encode_bid_*`` helpers in
    ``core/actions.pxd``):

      1. ``id 0``: leave the auction — legal only *after* the opening bid
         has been placed. On the first bid (auction_high_bidder == -1) the
         starter is committed to this company and pass is omitted.
      2. ``ids 1..15``: bid ``face_value + offset`` for ``offset in 0..14``.
         - First bid: ``min_offset = 0`` (any price ≥ face_value).
         - Subsequent bid: ``min_offset = max(0, current_bid - face + 1)``
           so ``new_bid > current_bid``.
         Affordability gates ``max_offset`` at ``player_cash - face``.
         Both gates are monotone so the inner loop breaks on the first
         unaffordable offset.

    Returns the number of IDs written. Worst case: 1 + AUCTION_CAP = 16 —
    well under ``MAX_ACTION_SIZE``.

    Reads state via direct slot arithmetic against the module-level
    layout constants and entity-owned primitives, matching the
    ``_enumerate_invest`` escape-hatch pattern.
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
    cdef int high_bidder = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.auction_high_bidder
    ]
    cdef int face, offset, new_bid, min_offset
    cdef bint is_first_bid = high_bidder < 0

    # --- id 0: leave the auction (illegal on the opening bid) --------------
    if not is_first_bid:
        ids[count] = 0
        count += 1

    # A BID state with no active_company is a driver bug — nothing to
    # auction, nothing to bid on.
    assert company_id >= 0, "_enumerate_bid: active_company unset"

    face = COMPANY_FACE_VALUE[company_id]
    # Opening bid: any offset that the starter can afford (min = face).
    # Subsequent bid: offset must produce ``new_bid > current_bid``, so
    # ``offset >= current_bid - face + 1``. Clamp to 0 defensively.
    if is_first_bid:
        min_offset = 0
    else:
        min_offset = current_bid - face + 1
        if min_offset < 0:
            min_offset = 0
    for offset in range(min_offset, <int>AUCTION_CAP):
        new_bid = face + offset
        if new_bid > player_cash:
            # Affordability is monotone in offset — once we can't afford
            # the next step, we can't afford any higher one.
            break
        _require_action_capacity(count, b"BID_RAISE")
        ids[count] = <uint16_t>encode_bid_raise(offset)
        count += 1

    return count


cdef inline bint _acq_pair_has_legal_price(
    GameState state, int corp_id, int company_id,
    int player_id, int cash, bint same_pres,
) noexcept nogil:
    """Does ``(corp_id, company_id)`` admit at least one legal acquisition price?

    Shared predicate for SELECT_CORP (as part of the any-target hoist) and
    SELECT_COMPANY (per-company gate). Encodes the same legality as the
    per-offset loop in SELECT_PRICE, collapsed to a boolean:

      - LOC_FI:     corp can afford the fixed FI price (face_value for OS,
                    high_price otherwise).
      - LOC_CORP:   target isn't self-owned, seller isn't in receivership,
                    seller retains >= 2 companies after sale, and under
                    same-president gating the active player presides the
                    seller. Affordability: ``cash >= low_price``.
      - LOC_PLAYER: under same-president gating the seller is the active
                    player. Affordability: ``cash >= low_price``.

    All other locations (LOC_CORP_ACQ, LOC_DECK, LOC_AUCTION) are unreachable
    as acquisition targets.
    """
    cdef int loc = company_location(state, company_id)
    cdef int owner_id
    cdef int price, low_price, high_price, max_offset

    if loc == <int>LOC_FI:
        if corp_id == <int>CorpIndices.CORP_OS:
            price = COMPANY_FACE_VALUE[company_id]
        else:
            price = COMPANY_HIGH_PRICE[company_id]
        return cash >= price
    elif loc == <int>LOC_CORP:
        owner_id = company_owner_id(state, company_id)
        if owner_id == corp_id:
            return False
        if corp_is_in_receivership(state, owner_id):
            return False
        if same_pres and corp_president_id(state, owner_id) != player_id:
            return False
        if count_corp_companies(state, owner_id, True) <= 1:
            return False
        low_price = COMPANY_LOW_PRICE[company_id]
        high_price = COMPANY_HIGH_PRICE[company_id]
        max_offset = high_price - low_price
        if cash - low_price < max_offset:
            max_offset = cash - low_price
        return max_offset >= 0
    elif loc == <int>LOC_PLAYER:
        owner_id = company_owner_id(state, company_id)
        if same_pres and owner_id != player_id:
            return False
        low_price = COMPANY_LOW_PRICE[company_id]
        high_price = COMPANY_HIGH_PRICE[company_id]
        max_offset = high_price - low_price
        if cash - low_price < max_offset:
            max_offset = cash - low_price
        return max_offset >= 0
    return False


cdef inline bint _corp_has_legal_target(
    GameState state, int corp_id, int player_id, bint same_pres,
) noexcept nogil:
    """True iff ``corp_id`` has at least one legal acquisition target.

    Hoisted out of SELECT_CORP to prevent SELECT_COMPANY from hitting the
    zero-legal-actions assertion on a corp with no reachable target.
    """
    cdef int cash = corp_cash(state, corp_id)
    cdef int company_id
    if cash <= 0:
        return False
    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if _acq_pair_has_legal_price(
            state, corp_id, company_id, player_id, cash, same_pres,
        ):
            return True
    return False


cdef int _enumerate_acq_select_corp(
    GameState state, uint16_t* ids,
) noexcept nogil:
    """Emit every legal DPHASE_ACQ_SELECT_CORP action in deterministic order.

    Ordering:
      1. ``id 0``: pass — decline to acquire this turn.
      2. ``ids 1..8``: ``1 + corp_id``. Emitted for each active,
         non-receivership corp the active player presides over that has
         at least one legal (company, price) target.

    The any-legal-target hoist (``_corp_has_legal_target``) is required:
    without it, SELECT_COMPANY can be entered against a corp with zero
    reachable targets and would hit the zero-legal-actions trip.

    Returns the number of IDs written. Worst case: 9 (pass + 8 corps).
    """
    cdef int count = 0
    cdef int player_id = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_player
    ]
    cdef bint same_pres = state.acq_same_president
    cdef int corp_id

    # --- id 0: pass (decline to acquire) ------------------------------------
    ids[count] = 0
    count += 1

    # --- ids 1..8: corp-select ----------------------------------------------
    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if not corp_is_active(state, corp_id):
            continue
        if corp_is_in_receivership(state, corp_id):
            continue
        if corp_president_id(state, corp_id) != player_id:
            continue
        if not _corp_has_legal_target(state, corp_id, player_id, same_pres):
            continue
        _require_action_capacity(count, b"ACQ_SELECT_CORP")
        ids[count] = <uint16_t>encode_acq_select_corp(corp_id)
        count += 1

    return count


cdef int _enumerate_acq_select_company(
    GameState state, uint16_t* ids,
) noexcept nogil:
    """Emit every legal DPHASE_ACQ_SELECT_COMPANY action in deterministic order.

    Ordering:
      ``ids 0..35``: ``company_id``. Emitted for each company where
      ``(active_corp, company_id)`` has at least one legal price (or FI_BUY)
      under the current same-president gate. No pass — SELECT_CORP already
      committed the player to picking a target.

    Returns the number of IDs written. Worst case: 36.
    """
    cdef int count = 0
    cdef int player_id = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_player
    ]
    cdef int corp_id = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_corp
    ]
    assert corp_id >= 0, "_enumerate_acq_select_company: active_corp unset"

    cdef bint same_pres = state.acq_same_president
    cdef int cash = corp_cash(state, corp_id)
    cdef int company_id

    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if not _acq_pair_has_legal_price(
            state, corp_id, company_id, player_id, cash, same_pres,
        ):
            continue
        _require_action_capacity(count, b"ACQ_SELECT_COMPANY")
        ids[count] = <uint16_t>encode_acq_select_company(company_id)
        count += 1

    return count


cdef int _enumerate_acq_select_price(
    GameState state, uint16_t* ids,
) noexcept nogil:
    """Emit every legal DPHASE_ACQ_SELECT_PRICE action in deterministic order.

    Ordering:
      ``ids 0..max_offset`` where
      ``max_offset = min(high - low, cash - low, 50)``.

    Trusts SELECT_COMPANY's filter: by the time we're here, the
    (active_corp, active_company) pair has at least one legal price and
    the seller-side ownership gates have been validated. LOC_FI targets
    never reach SELECT_PRICE — they execute directly during SELECT_COMPANY.

    Returns the number of IDs written. Worst case: 51.
    """
    cdef int count = 0
    cdef int corp_id = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_corp
    ]
    cdef int company_id = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_company
    ]
    assert corp_id >= 0, "_enumerate_acq_select_price: active_corp unset"
    assert company_id >= 0, "_enumerate_acq_select_price: active_company unset"
    assert company_location(state, company_id) != <int>LOC_FI, \
        f"_enumerate_acq_select_price: LOC_FI target {company_id} should " \
        f"have executed in SELECT_COMPANY"

    cdef int cash = corp_cash(state, corp_id)
    cdef int low_price = COMPANY_LOW_PRICE[company_id]
    cdef int high_price = COMPANY_HIGH_PRICE[company_id]
    cdef int max_offset = high_price - low_price
    cdef int price_offset

    if cash - low_price < max_offset:
        max_offset = cash - low_price
    if max_offset > 50:
        max_offset = 50
    for price_offset in range(max_offset + 1):
        _require_action_capacity(count, b"ACQ_SELECT_PRICE")
        ids[count] = <uint16_t>encode_acq_select_price(price_offset)
        count += 1

    return count


cdef int _enumerate_acq_offer(
    GameState state, uint16_t* ids,
) noexcept nogil:
    """Enumerate legal ACQ_OFFER actions: always PASS + ACCEPT.

    We never enter ACQ_OFFER unless both actions are valid (affordability
    is pre-filtered before entry), so this always returns exactly 2.
    """
    ids[0] = 0  # PASS
    ids[1] = 1  # ACCEPT
    return 2


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

    When ``state.allow_positive_income_closing`` is False (the default),
    individual companies are only emitted if their CoO-adjusted income is
    strictly negative. This is the training-time gate: the model only gets
    to close companies that are losing money. The 18xx replay tests flip
    the flag to True to recover the unrestricted behavior.

    Returns the number of IDs written. Worst case: 37 (pass + 36 companies).
    """
    cdef int count = 0
    cdef int active_player = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_player
    ]
    cdef bint allow_positive = state.allow_positive_income_closing
    cdef int loc, owner
    cdef int company_id

    # --- id 0: pass (always legal) -------------------------------------------
    ids[count] = 0
    count += 1

    # --- Emit player-owned + valid corp-owned companies -----------------------
    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        loc = company_location(state, company_id)
        if loc == <int>LOC_PLAYER:
            if not company_owned_by_player(state, company_id, active_player):
                continue
        elif loc == <int>LOC_CORP:
            owner = company_owner_id(state, company_id)
            if not _corp_closable_by_player(state, owner, active_player):
                continue
        else:
            continue

        if (not allow_positive
                and company_adjusted_income(state, company_id) >= 0):
            continue

        _require_action_capacity(count, b"CLOSING")
        ids[count] = <uint16_t>encode_closing_close(company_id)
        count += 1

    return count


cdef int _enumerate_dividends(
    GameState state, uint16_t* ids,
) noexcept nogil:
    """Emit every legal DPHASE_DIVIDENDS action in deterministic order.

    Action IDs are literal dividend amounts: 0, 1, 2, ..., max_dividend.
    Legal range: 0 to min(price // 3, cash // issued_shares, 25).

    No pass action -- 0-dividend is always legal and plays that role.
    Worst case: 26 IDs (amounts 0-25).
    """
    cdef int active_corp = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_corp
    ]
    cdef int price_index = corp_price_index(state, active_corp)
    cdef int issued_shares = corp_issued_shares(state, active_corp)
    cdef int cash = corp_cash(state, active_corp)

    cdef int max_by_price = MARKET_PRICES[price_index] // 3
    cdef int max_by_afford = cash // issued_shares

    cdef int max_div = min(max_by_afford, max_by_price, 25)
    max_div = max(max_div, 0)

    cdef int count = 0
    cdef int level
    for level in range(max_div + 1):
        _require_action_capacity(count, b"DIVIDENDS")
        ids[count] = <uint16_t>level
        count += 1
    return count


cdef int _enumerate_issue(
    GameState state, uint16_t* ids,
) noexcept nogil:
    # Active corp always has unissued shares when this runs
    # (advance logic auto-skips corps with 0 unissued).
    ids[0] = 0  # pass
    ids[1] = 1  # issue
    return 2


cdef inline bint _is_par_legal(
    int par_index, int star_tier, int face_value,
    int player_cash, const int16_t* market_ptr,
) noexcept nogil:
    """Triple-gate predicate: par is star-valid, market-available, affordable.

    Single source of truth for IPO's any-affordable-par gate and PAR's
    per-index emission — sharing this avoids the class of bugs where IPO
    offers a corp with no legal PAR follow-up.
    """
    cdef int par_price, market_index, float_shares, player_payment
    if PAR_PRICE_VALID[star_tier - 1][par_index] == 0:
        return False
    par_price = ALL_PAR_PRICES[par_index]
    market_index = PRICE_TO_MARKET_INDEX[par_price]
    if market_ptr[market_index] == 0:
        return False
    float_shares = 2 if face_value > par_price else 1
    player_payment = (float_shares * par_price) - face_value
    if player_payment > player_cash:
        return False
    return True


cdef inline bint _any_par_affordable(
    int star_tier, int face_value, int player_cash, const int16_t* market_ptr,
) noexcept nogil:
    cdef int par_index
    for par_index in range(<int>GameConstants.NUM_PAR_PRICES):
        if _is_par_legal(par_index, star_tier, face_value, player_cash, market_ptr):
            return True
    return False


cdef int _enumerate_ipo(
    GameState state, uint16_t* ids,
) noexcept nogil:
    """Emit every legal DPHASE_IPO action in deterministic order.

    Ordering:
      1. ``id 0``: pass (always legal)
      2. ``ids 1..8``: ``1 + corp_id``. Emitted for each inactive corp
         when at least one par price is star-valid, market-available, and
         affordable. The any-affordable check is invariant in corp_id
         (depends on active_company + active_player), so it's hoisted out
         of the corp loop. The PAR handler re-checks per-par legality.

    Returns the number of IDs written. Worst case: 9 (pass + 8 corps).
    """
    cdef int16_t* d = &state._data[0]
    cdef int count = 0

    cdef int company_id = <int>d[LAYOUT.turn_offset + TURN_OFFSETS.active_company]
    assert company_id >= 0, "_enumerate_ipo: active_company unset"

    cdef int star_tier = COMPANY_STARS[company_id]
    cdef int face_value = COMPANY_FACE_VALUE[company_id]

    cdef int player_id = <int>d[LAYOUT.turn_offset + TURN_OFFSETS.active_player]
    cdef int player_base = LAYOUT.players_offset + player_id * PLAYER_FIELDS.size
    cdef int player_cash = <int>d[player_base + PLAYER_FIELDS.cash]
    cdef const int16_t* market_ptr = d + LAYOUT.market_offset

    cdef int corp_id

    # --- id 0: pass (always legal) -------------------------------------------
    ids[count] = 0
    count += 1

    # No affordable par for this (company, player, market) — only PASS.
    if not _any_par_affordable(star_tier, face_value, player_cash, market_ptr):
        return count

    # --- ids 1..8: corp-select -----------------------------------------------
    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if corp_is_active(state, corp_id):
            continue
        _require_action_capacity(count, b"IPO")
        ids[count] = <uint16_t>encode_ipo(corp_id)
        count += 1

    return count


cdef int _enumerate_par(
    GameState state, uint16_t* ids,
) noexcept nogil:
    """Emit every legal DPHASE_PAR action in deterministic order.

    Ordering:
      ``ids 0..13``: ``par_index``. Emitted for each par price that is
      valid for the active_company's star tier, has an available market
      slot, and is affordable by the active player. No pass.

    Returns the number of IDs written. Worst case: 14.
    """
    cdef int16_t* d = &state._data[0]
    cdef int count = 0

    cdef int company_id = <int>d[LAYOUT.turn_offset + TURN_OFFSETS.active_company]
    assert company_id >= 0, "_enumerate_par: active_company unset"

    cdef int star_tier = COMPANY_STARS[company_id]
    cdef int face_value = COMPANY_FACE_VALUE[company_id]

    cdef int player_id = <int>d[LAYOUT.turn_offset + TURN_OFFSETS.active_player]
    cdef int player_base = LAYOUT.players_offset + player_id * PLAYER_FIELDS.size
    cdef int player_cash = <int>d[player_base + PLAYER_FIELDS.cash]
    cdef const int16_t* market_ptr = d + LAYOUT.market_offset

    cdef int par_index

    for par_index in range(<int>GameConstants.NUM_PAR_PRICES):
        if not _is_par_legal(par_index, star_tier, face_value, player_cash, market_ptr):
            continue
        _require_action_capacity(count, b"PAR")
        ids[count] = <uint16_t>encode_par(par_index)
        count += 1

    return count


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
    elif phase_id == DPHASE_ACQ_SELECT_CORP:
        count = _enumerate_acq_select_corp(state, action_ids)
    elif phase_id == DPHASE_ACQ_SELECT_COMPANY:
        count = _enumerate_acq_select_company(state, action_ids)
    elif phase_id == DPHASE_ACQ_SELECT_PRICE:
        count = _enumerate_acq_select_price(state, action_ids)
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
    elif phase_id == DPHASE_PAR:
        count = _enumerate_par(state, action_ids)

    # Overflow is a configuration bug, not a recoverable condition. This is
    # intentionally checked in release builds, not only with Python asserts.
    if count > MAX_ACTION_SIZE:
        fprintf(
            stderr,
            b"enumerate_legal_actions returned count=%d > MAX_ACTION_SIZE=%d\n",
            count,
            <int>MAX_ACTION_SIZE,
        )
        abort()
    return count


cdef int _filter_acq_price_edge_actions(
    int phase_id,
    uint16_t* action_ids,
    int count,
    int max_acq_price_actions,
) noexcept nogil:
    """Keep only low/high ACQ price offsets for policy/search when configured."""
    cdef int edge_count
    cdef int high_start
    cdef int i

    if (
        max_acq_price_actions <= 0
        or phase_id != DPHASE_ACQ_SELECT_PRICE
        or count <= max_acq_price_actions
    ):
        return count

    edge_count = max_acq_price_actions // 2
    high_start = count - edge_count
    for i in range(edge_count):
        action_ids[edge_count + i] = action_ids[high_start + i]
    return max_acq_price_actions


cdef int enumerate_policy_actions(
    GameState state,
    uint16_t* action_ids,
    int max_acq_price_actions,
) noexcept nogil:
    """Enumerate legal actions, optionally capping ACQ_SELECT_PRICE to edges.

    This is a policy/search view only. Engine legality remains owned by
    ``enumerate_legal_actions`` so direct driver validation and external replay
    compatibility keep the full game action set.
    """
    cdef int phase_id = get_decision_phase(state)
    cdef int count = enumerate_legal_actions(state, action_ids)
    return _filter_acq_price_edge_actions(
        phase_id,
        action_ids,
        count,
        max_acq_price_actions,
    )


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


cpdef int encode_action_py(object info):
    """Python wrapper around ``encode_action``.

    Accepts an ``ActionInfoTuple`` (or any object with phase / action_type /
    corp_id / company_id / amount attributes, e.g. the tuple returned by
    ``decode_action_py``). Used by the encode/decode round-trip test.
    """
    cdef ActionInfo c_info
    c_info.phase = info.phase
    c_info.action_type = info.action_type
    c_info.corp_id = info.corp_id
    c_info.company_id = info.company_id
    c_info.amount = info.amount
    return encode_action(c_info)


cpdef int get_decision_phase_py(GameState state):
    """Python wrapper around ``get_decision_phase``."""
    return get_decision_phase(state)


cpdef int enumerate_legal_actions_py(GameState state, cnp.ndarray action_ids):
    """Python wrapper around ``enumerate_legal_actions``.

    Caller supplies a pre-allocated uint16 numpy array of at least
    ``MAX_ACTION_SIZE`` elements (the tight upper bound across all phases).
    Returns the number of legal action IDs written into the buffer.
    """
    cdef uint16_t* buf_ptr = <uint16_t*>cnp.PyArray_DATA(action_ids)
    return enumerate_legal_actions(state, buf_ptr)


cpdef int enumerate_policy_actions_py(
    GameState state,
    cnp.ndarray action_ids,
    int max_acq_price_actions=0,
):
    """Policy/search legal-action wrapper with optional ACQ price edge cap.

    ``max_acq_price_actions=0`` preserves the exact engine legal list. A
    positive value must be even; when ACQ_SELECT_PRICE would expose more than
    that many prices, the low half and high half are kept and middle offsets are
    omitted from the returned policy/search action set.
    """
    cdef uint16_t* buf_ptr
    assert max_acq_price_actions >= 0, (
        "max_acq_price_actions must be 0 or a positive even integer, "
        f"got {max_acq_price_actions}"
    )
    assert max_acq_price_actions == 0 or max_acq_price_actions % 2 == 0, (
        "max_acq_price_actions must be divisible by 2, "
        f"got {max_acq_price_actions}"
    )
    assert max_acq_price_actions <= ACTION_SIZE_ACQ_SELECT_PRICE, (
        "max_acq_price_actions must be <= ACTION_SIZE_ACQ_SELECT_PRICE "
        f"({ACTION_SIZE_ACQ_SELECT_PRICE}), got {max_acq_price_actions}"
    )
    buf_ptr = <uint16_t*>cnp.PyArray_DATA(action_ids)
    return enumerate_policy_actions(state, buf_ptr, max_acq_price_actions)


# =============================================================================
# MODULE-LEVEL PYTHON CONSTANTS (for tests / external inspection)
# =============================================================================
#
# ``DecisionPhase`` lives on ``core.data`` — Python consumers should use
# ``from core.data import DecisionPhase`` and reference ``DPHASE_*`` as
# attributes. The legal-action buffer width for Python callers is
# ``core.data.MAX_ACTION_SIZE``. Only action-type tags are mirrored here.

ACTION_PASS_PY = ACTION_PASS
ACTION_AUCTION_PY = ACTION_AUCTION
ACTION_BUY_SHARE_PY = ACTION_BUY_SHARE
ACTION_SELL_SHARE_PY = ACTION_SELL_SHARE
ACTION_RAISE_PY = ACTION_RAISE
ACTION_ACQ_PRICE_PY = ACTION_ACQ_PRICE
ACTION_ACQ_OFFER_ACCEPT_PY = ACTION_ACQ_OFFER_ACCEPT
ACTION_CLOSE_PY = ACTION_CLOSE
ACTION_DIVIDEND_PY = ACTION_DIVIDEND
ACTION_ISSUE_PY = ACTION_ISSUE
ACTION_IPO_PY = ACTION_IPO
ACTION_PAR_PY = ACTION_PAR
ACTION_ACQ_SELECT_CORP_PY = ACTION_ACQ_SELECT_CORP
ACTION_ACQ_SELECT_COMPANY_PY = ACTION_ACQ_SELECT_COMPANY


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
assert encode_bid_raise(<int>AUCTION_CAP - 1) == ACTION_SIZE_BID - 1
assert encode_acq_select_corp(7) == ACTION_SIZE_ACQ_SELECT_CORP - 1
assert encode_acq_select_company(35) == ACTION_SIZE_ACQ_SELECT_COMPANY - 1
assert encode_acq_select_price(50) == ACTION_SIZE_ACQ_SELECT_PRICE - 1
assert 1 == ACTION_SIZE_ACQ_OFFER - 1
assert encode_closing_close(35) == ACTION_SIZE_CLOSING - 1
assert 25 == ACTION_SIZE_DIVIDENDS - 1
assert 1 == ACTION_SIZE_ISSUE - 1
assert encode_ipo(7) == ACTION_SIZE_IPO - 1
assert encode_par(13) == ACTION_SIZE_PAR - 1
