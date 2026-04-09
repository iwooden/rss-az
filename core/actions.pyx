# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
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

from libc.stdint cimport uint16_t

from core.state cimport GameState
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
from entities.player cimport Player
from entities.corp cimport Corporation
from entities.company cimport Company, LOC_AUCTION
from entities.market cimport Market
from entities.turn cimport TurnState

cnp.import_array()


# =============================================================================
# CACHED TYPED ENTITY SINGLETON REFERENCES
# =============================================================================
#
# The legal-action enumerators below dispatch through entity handles for
# every state read. Cython does not allow C arrays of ``cdef class``
# instances ("Array element cannot be a Python object"), so we split the
# caching:
#
# - Single-instance entities (``TurnState``, ``Market``) live in
#   typed module-level ``cdef class`` variables — cheap to access and
#   safe to call ``cdef nogil`` methods on without GIL.
# - Multi-instance entities (players, corps, companies) are reached
#   through the entity module's own Python singleton lists inside the
#   ``with gil:`` block of each enumerator. Indexing those lists needs
#   GIL for refcount bookkeeping, but the per-entry cost is amortized
#   against the full enumeration pass.

cdef TurnState _TURN_REF
cdef Market _MARKET_REF


# Late Python-level imports of the entity modules so we can reach the
# singleton instances during ``_init_entity_refs`` and inside enumerator
# ``with gil:`` blocks. No cycle — no entity module imports from
# ``core.actions``.
from entities import player as _player_module
from entities import corp as _corp_module
from entities import company as _company_module
from entities import market as _market_module
from entities import turn as _turn_module


cdef void _init_entity_refs() except *:
    """Populate the single-instance entity reference caches.

    Runs once at module import. Assignment increments the Python refcount
    for each cached reference, so the singletons are kept alive for the
    process lifetime.
    """
    global _TURN_REF, _MARKET_REF
    _TURN_REF = <TurnState>_turn_module.TURN
    _MARKET_REF = <Market>_market_module.MARKET


_init_entity_refs()


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
            return encode_invest_pass()
        if info.action_type == ACTION_AUCTION:
            return encode_invest_auction(info.company_id, info.amount)
        if info.action_type == ACTION_BUY_SHARE:
            return encode_invest_buy(info.corp_id)
        if info.action_type == ACTION_SELL_SHARE:
            return encode_invest_sell(info.corp_id)

    elif phase == DPHASE_BID:
        if info.action_type == ACTION_PASS:
            return encode_bid_pass()
        if info.action_type == ACTION_RAISE:
            return encode_bid_raise(info.amount)

    elif phase == DPHASE_ACQUISITION:
        if info.action_type == ACTION_PASS:
            return encode_acquisition_pass()
        if info.action_type == ACTION_ACQ_PRICE:
            return encode_acquisition_price(info.corp_id, info.company_id, info.amount)
        if info.action_type == ACTION_ACQ_FI_BUY:
            return encode_acquisition_fi_buy(info.corp_id, info.company_id)

    elif phase == DPHASE_ACQ_OFFER:
        if info.action_type == ACTION_PASS:
            return encode_acq_offer_pass()
        if info.action_type == ACTION_ACQ_OFFER_BUY:
            return encode_acq_offer_buy()

    elif phase == DPHASE_CLOSING:
        if info.action_type == ACTION_PASS:
            return encode_closing_pass()
        if info.action_type == ACTION_CLOSE:
            return encode_closing_close(info.company_id)

    elif phase == DPHASE_DIVIDENDS:
        if info.action_type == ACTION_DIVIDEND:
            return encode_dividends(info.amount)

    elif phase == DPHASE_ISSUE:
        if info.action_type == ACTION_PASS:
            return encode_issue_pass()
        if info.action_type == ACTION_ISSUE:
            return encode_issue_issue()

    elif phase == DPHASE_IPO:
        if info.action_type == ACTION_PASS:
            return encode_ipo_pass()
        if info.action_type == ACTION_IPO:
            return encode_ipo(info.corp_id, info.amount)

    # Unreachable for a well-formed ActionInfo. The assert fires under
    # debug builds; under -O it falls through to -1 which callers treat
    # as a decoding error.
    with gil:
        raise AssertionError(
            f"encode_action: illegal (phase={phase}, type={info.action_type})"
        )


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

    # Unknown phase — caller error.
    with gil:
        raise AssertionError(f"decode_action: unknown phase {phase_id}")


# =============================================================================
# DECISION PHASE BRIDGE
# =============================================================================

cdef int get_decision_phase(GameState state) noexcept nogil:
    """Read the engine phase from ``state`` and map it to a DecisionPhase.

    Returns -1 for automated/terminal engine phases (WRAP_UP, INCOME,
    END_CARD, GAME_OVER). Callers are expected to only invoke enumeration /
    decoding for phases where the driver actually asks for an action.

    Dispatches through the cached ``TurnState`` reference so the engine
    phase read goes through the entity handle (``_get_phase``) instead of
    indexing the state buffer directly — the whole function stays nogil.
    """
    cdef int engine_phase = _TURN_REF._get_phase(state)
    if engine_phase < 0 or engine_phase >= 12:
        return -1
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

cdef int _enumerate_invest(GameState state, uint16_t* ids):
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

    All state access goes through entity handles: ``_TURN_REF`` and
    ``_MARKET_REF`` are module-level typed singletons; players / corps /
    companies are indexed out of the entity modules' singleton lists and
    cast to their handle type.

    This function is NOT declared ``nogil`` — Cython does not allow
    ``cdef class`` locals (needed to dispatch the entity methods) inside
    a nogil function, and it does not allow module-level C arrays of
    ``cdef class`` references either. Callers from nogil context
    (``enumerate_legal_actions``) acquire the GIL around the call. The
    body itself runs with the GIL held but every accessor method it
    invokes is ``cdef nogil``, so the method dispatch is still a direct
    C call with no Python-attribute-lookup overhead.
    """
    cdef int count = 0
    cdef Player active_player_handle
    cdef Corporation corp
    cdef Company company
    cdef int player_cash
    cdef int company_id, bid_offset, face_value
    cdef int corp_id, buys, sells, buy_index

    active_player_handle = <Player>_player_module.PLAYERS[
        _TURN_REF._get_active_player(state)
    ]
    player_cash = active_player_handle._get_cash(state)

    # --- id 0: pass ---------------------------------------------------------
    ids[count] = <uint16_t>encode_invest_pass()
    count += 1

    # --- ids 1..540: auctions ----------------------------------------------
    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        company = <Company>_company_module.COMPANIES[company_id]
        if company._get_location(state) != <int>LOC_AUCTION:
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
        corp = <Corporation>_corp_module.CORPS[corp_id]

        # Round-trip gate applies to *both* buy and sell: a player who has
        # already completed 2 paired buy/sells against this corp is locked
        # out of further trades on it for the rest of the INVEST phase.
        buys = active_player_handle._get_share_buys(state, corp_id)
        sells = active_player_handle._get_share_sells(state, corp_id)
        if (buys if buys < sells else sells) >= 2:
            continue

        if not corp._is_active(state):
            # Inactive corps have no tradable shares.
            continue

        # --- BUY ------------------------------------------------------------
        if corp._get_bank_shares(state) > 0:
            buy_index = _MARKET_REF._find_next_higher_space(
                state, corp._get_price_index(state)
            )
            if player_cash >= MARKET_PRICES[buy_index]:
                ids[count] = <uint16_t>encode_invest_buy(corp_id)
                count += 1

        # --- SELL -----------------------------------------------------------
        if active_player_handle._get_shares(state, corp_id) > 0:
            ids[count] = <uint16_t>encode_invest_sell(corp_id)
            count += 1

    return count


cdef int _enumerate_bid(
    GameState state, uint16_t* ids,
) noexcept nogil:
    return 0


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
    return 0


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
    int phase_id,
    uint16_t* action_ids,
) noexcept nogil:
    """Fill ``action_ids`` with legal phase-local IDs; return the count.

    Dispatches to a phase-specific ``_enumerate_*`` helper. The helpers
    own the per-phase enumeration order contract.

    The fully-implemented enumerators dispatch through entity handles and
    therefore can't be declared ``nogil`` (Cython forbids ``cdef class``
    locals inside nogil functions). For those phases the dispatcher
    acquires the GIL around the call. Stub enumerators still marked
    nogil are called directly.
    """
    cdef int count = 0

    if phase_id == DPHASE_INVEST:
        with gil:
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


cdef (int, bint) get_forced_action(GameState state) noexcept nogil:
    """Return ``(action_id, True)`` if exactly one action is legal.

    Used by the driver to fast-forward through single-choice decision
    points. Returns ``(-1, False)`` for zero or multiple legal actions.
    """
    cdef uint16_t scratch[256]   # MAX_LEGAL_ACTIONS
    cdef int phase_id = get_decision_phase(state)
    if phase_id < 0:
        return (-1, False)
    cdef int count = enumerate_legal_actions(state, phase_id, scratch)
    if count == 1:
        return (<int>scratch[0], True)
    return (-1, False)


# =============================================================================
# PYTHON-ACCESSIBLE WRAPPERS
# =============================================================================

cpdef int get_phase_action_size(int phase_id):
    """Return the action-space size for a given DecisionPhase."""
    if phase_id < 0 or phase_id >= <int>GameConstants.NUM_DECISION_PHASES:
        raise ValueError(f"invalid decision phase: {phase_id}")
    return _PHASE_ACTION_SIZES_C[phase_id]


cpdef tuple decode_action_py(int phase_id, int action_id):
    """Python wrapper around ``decode_action``.

    Returns a tuple ``(phase, action_type, corp_id, company_id, amount)``.
    """
    if phase_id < 0 or phase_id >= <int>GameConstants.NUM_DECISION_PHASES:
        raise ValueError(f"invalid decision phase: {phase_id}")
    if action_id < 0 or action_id >= _PHASE_ACTION_SIZES_C[phase_id]:
        raise ValueError(
            f"action_id {action_id} out of range for phase {phase_id} "
            f"(size {_PHASE_ACTION_SIZES_C[phase_id]})"
        )
    cdef ActionInfo info = decode_action(phase_id, action_id)
    return (info.phase, info.action_type, info.corp_id, info.company_id, info.amount)


cpdef object enumerate_legal_actions_py(GameState state, int phase_id=-1):
    """Python wrapper around ``enumerate_legal_actions``.

    If ``phase_id`` is -1 (default), the current decision phase is read
    from ``state`` via ``get_decision_phase``. Returns a tuple
    ``(phase_id, action_ids_array)`` where ``action_ids_array`` is a
    uint16 numpy array of length ``count``.
    """
    cdef int dphase
    if phase_id < 0:
        dphase = get_decision_phase(state)
    else:
        dphase = phase_id
    if dphase < 0:
        return (-1, np.empty(0, dtype=np.uint16))

    cdef cnp.ndarray buf = np.empty(MAX_LEGAL_ACTIONS, dtype=np.uint16)
    cdef uint16_t* buf_ptr = <uint16_t*>cnp.PyArray_DATA(buf)
    cdef int count = enumerate_legal_actions(state, dphase, buf_ptr)
    return (dphase, buf[:count].copy())


cpdef tuple get_forced_action_py(GameState state):
    """Python wrapper around ``get_forced_action``. Returns ``(action_id, found)``."""
    cdef int action_id
    cdef bint found
    (action_id, found) = get_forced_action(state)
    return (action_id, bool(found))


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
assert encode_acq_offer_buy() == ACTION_SIZE_ACQ_OFFER - 1
assert encode_closing_close(35) == ACTION_SIZE_CLOSING - 1
assert encode_dividends(25) == ACTION_SIZE_DIVIDENDS - 1
assert encode_issue_issue() == ACTION_SIZE_ISSUE - 1
assert encode_ipo(7, 13) == ACTION_SIZE_IPO - 1
