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
encoding layer defined here is frozen and must stay in lockstep with
``nn/transformer.py::PHASE_ACTION_SIZES``; a startup-time assertion guards
against drift.
"""

cimport cython
import numpy as np
cimport numpy as cnp

from libc.stdint cimport uint16_t

from core.state cimport GameState, LAYOUT, TURN_OFFSETS
from core.data cimport GameConstants, GamePhases

cnp.import_array()


# =============================================================================
# PHASE_ACTION_SIZES TABLE
# =============================================================================

# Populated at module import by ``_init_phase_action_sizes``. The pxd
# declares this as ``cdef int PHASE_ACTION_SIZES[8]``; the init function
# fills it from the ``ACTION_SIZE_*`` enum constants so phase sizes are
# readable via a single indexed lookup rather than a switch.
cdef void _init_phase_action_sizes() noexcept nogil:
    PHASE_ACTION_SIZES[<int>DPHASE_INVEST] = ACTION_SIZE_INVEST
    PHASE_ACTION_SIZES[<int>DPHASE_BID] = ACTION_SIZE_BID
    PHASE_ACTION_SIZES[<int>DPHASE_ACQUISITION] = ACTION_SIZE_ACQUISITION
    PHASE_ACTION_SIZES[<int>DPHASE_ACQ_OFFER] = ACTION_SIZE_ACQ_OFFER
    PHASE_ACTION_SIZES[<int>DPHASE_CLOSING] = ACTION_SIZE_CLOSING
    PHASE_ACTION_SIZES[<int>DPHASE_DIVIDENDS] = ACTION_SIZE_DIVIDENDS
    PHASE_ACTION_SIZES[<int>DPHASE_ISSUE] = ACTION_SIZE_ISSUE
    PHASE_ACTION_SIZES[<int>DPHASE_IPO] = ACTION_SIZE_IPO


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

# Lookup table from engine GamePhases -> DecisionPhase. -1 marks phases
# that are automated or terminal (driver fast-forwards, no action needed).
cdef int _ENGINE_TO_DECISION[12]

cdef void _init_engine_to_decision() noexcept nogil:
    cdef int i
    for i in range(12):
        _ENGINE_TO_DECISION[i] = -1
    _ENGINE_TO_DECISION[<int>GamePhases.PHASE_INVEST] = DPHASE_INVEST
    _ENGINE_TO_DECISION[<int>GamePhases.PHASE_BID_IN_AUCTION] = DPHASE_BID
    _ENGINE_TO_DECISION[<int>GamePhases.PHASE_ACQUISITION] = DPHASE_ACQUISITION
    _ENGINE_TO_DECISION[<int>GamePhases.PHASE_ACQ_OFFER] = DPHASE_ACQ_OFFER
    _ENGINE_TO_DECISION[<int>GamePhases.PHASE_CLOSING] = DPHASE_CLOSING
    _ENGINE_TO_DECISION[<int>GamePhases.PHASE_DIVIDENDS] = DPHASE_DIVIDENDS
    _ENGINE_TO_DECISION[<int>GamePhases.PHASE_ISSUE_SHARES] = DPHASE_ISSUE
    _ENGINE_TO_DECISION[<int>GamePhases.PHASE_IPO] = DPHASE_IPO


_init_engine_to_decision()


cdef int get_decision_phase(GameState state) noexcept nogil:
    """Read the engine phase from ``state`` and map it to a DecisionPhase.

    Returns -1 for automated/terminal engine phases (WRAP_UP, INCOME,
    END_CARD, GAME_OVER). Callers are expected to only invoke enumeration /
    decoding for phases where the driver actually asks for an action.

    Reads the phase slot directly from the state buffer — no Python
    singleton access — so the whole function stays nogil.
    """
    cdef int engine_phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]
    if engine_phase < 0 or engine_phase >= 12:
        return -1
    return _ENGINE_TO_DECISION[engine_phase]


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
    return 0


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
    """
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
    return PHASE_ACTION_SIZES[phase_id]


cpdef tuple decode_action_py(int phase_id, int action_id):
    """Python wrapper around ``decode_action``.

    Returns a tuple ``(phase, action_type, corp_id, company_id, amount)``.
    """
    if phase_id < 0 or phase_id >= <int>GameConstants.NUM_DECISION_PHASES:
        raise ValueError(f"invalid decision phase: {phase_id}")
    if action_id < 0 or action_id >= PHASE_ACTION_SIZES[phase_id]:
        raise ValueError(
            f"action_id {action_id} out of range for phase {phase_id} "
            f"(size {PHASE_ACTION_SIZES[phase_id]})"
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

# Expose DecisionPhase values as plain Python ints so tests can import them
# without cimporting the enum. Matches the naming in nn/transformer.py.
PHASE_INVEST = DPHASE_INVEST
PHASE_BID = DPHASE_BID
PHASE_ACQUISITION = DPHASE_ACQUISITION
PHASE_ACQ_OFFER = DPHASE_ACQ_OFFER
PHASE_CLOSING = DPHASE_CLOSING
PHASE_DIVIDENDS = DPHASE_DIVIDENDS
PHASE_ISSUE = DPHASE_ISSUE
PHASE_IPO = DPHASE_IPO

PHASE_ACTION_SIZES_PY = [
    ACTION_SIZE_INVEST,
    ACTION_SIZE_BID,
    ACTION_SIZE_ACQUISITION,
    ACTION_SIZE_ACQ_OFFER,
    ACTION_SIZE_CLOSING,
    ACTION_SIZE_DIVIDENDS,
    ACTION_SIZE_ISSUE,
    ACTION_SIZE_IPO,
]

MAX_LEGAL_ACTIONS_PY = MAX_LEGAL_ACTIONS
MAX_ACTION_SIZE_PY = MAX_ACTION_SIZE

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
# ``encode_*`` formulae and the pxd size constants have fallen out of
# sync, which will silently corrupt replay alignment later.

def _verify_encoding_consistency():
    from nn.transformer import PHASE_ACTION_SIZES as NN_SIZES

    local = list(PHASE_ACTION_SIZES_PY)
    nn = list(NN_SIZES)
    assert local == nn, (
        f"core/actions.pyx PHASE_ACTION_SIZES {local} disagrees with "
        f"nn/transformer.py {nn}. Keep them in lockstep or replay "
        f"targets will silently misalign with policy logits."
    )

    # Spot-check each phase's highest encodable id.
    assert encode_invest_sell(7) == ACTION_SIZE_INVEST - 1
    assert encode_bid_raise(13) == ACTION_SIZE_BID - 1
    assert encode_acquisition_fi_buy(7, 35) == ACTION_SIZE_ACQUISITION - 1
    assert encode_acq_offer_buy() == ACTION_SIZE_ACQ_OFFER - 1
    assert encode_closing_close(35) == ACTION_SIZE_CLOSING - 1
    assert encode_dividends(25) == ACTION_SIZE_DIVIDENDS - 1
    assert encode_issue_issue() == ACTION_SIZE_ISSUE - 1
    assert encode_ipo(7, 13) == ACTION_SIZE_IPO - 1


try:
    _verify_encoding_consistency()
except ImportError:
    # nn/transformer.py imports torch; environments without torch (early
    # CI / static analysis) still get the self-consistency checks above,
    # just not the cross-module comparison.
    assert encode_invest_sell(7) == ACTION_SIZE_INVEST - 1
    assert encode_bid_raise(13) == ACTION_SIZE_BID - 1
    assert encode_acquisition_fi_buy(7, 35) == ACTION_SIZE_ACQUISITION - 1
    assert encode_acq_offer_buy() == ACTION_SIZE_ACQ_OFFER - 1
    assert encode_closing_close(35) == ACTION_SIZE_CLOSING - 1
    assert encode_dividends(25) == ACTION_SIZE_DIVIDENDS - 1
    assert encode_issue_issue() == ACTION_SIZE_ISSUE - 1
    assert encode_ipo(7, 13) == ACTION_SIZE_IPO - 1
