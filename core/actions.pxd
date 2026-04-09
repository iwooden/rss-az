# cython: language_level=3
"""
Per-phase action encoding for the transformer refactor.

Replaces the old global action vector with **phase-local** action IDs. The
encoding defined here is the single source of truth for:

  - what integer represents what game action in each decision phase
  - how replay buffers serialize ``(phase_id, action_id)`` pairs
  - how the engine decodes a sparse legal candidate into a phase handler call
  - how ``nn/transformer.py`` interprets its policy head outputs

The per-phase action *counts* themselves live in ``core/data.pxd`` as the
``ActionSize`` ``cpdef enum`` — single source of truth shared with the model
module. The ``encode_*`` arithmetic below must stay consistent with those
counts; the roundtrip asserts in ``actions.pyx`` guard against drift.

See ``transformers.md`` and ``sparse-refactor.md`` for motivation.

Conventions
-----------

- **Decision phases** are the 8 phases the model sees, distinct from the
  engine's 12 ``GamePhases`` — notably the engine's ``ACQUISITION`` splits
  into ``DPHASE_ACQUISITION`` + ``DPHASE_ACQ_OFFER``, and ``IPO`` / ``PAR``
  collapse into ``DPHASE_IPO``. The enum itself lives in ``core/data.pxd``
  (``cpdef enum DecisionPhase``) alongside the engine → decision lookup
  table.
- Action IDs are **phase-local**: the same integer means different things in
  different phases. Callers must always carry the ``phase_id`` alongside the
  ``action_id``.
- Action IDs fit comfortably in ``uint16`` (max 14,977 for ``ACQUISITION``).
- The sparse legal-action path is the only public contract. There is no
  dense per-phase mask surface; any dense consumer should scatter from the
  sparse list.
"""

from libc.stdint cimport uint16_t

from core.state cimport GameState
from core.data cimport (
    ACTION_SIZE_INVEST,
    ACTION_SIZE_BID,
    ACTION_SIZE_ACQUISITION,
    ACTION_SIZE_ACQ_OFFER,
    ACTION_SIZE_CLOSING,
    ACTION_SIZE_DIVIDENDS,
    ACTION_SIZE_ISSUE,
    ACTION_SIZE_IPO,
    MAX_ACTION_SIZE,
    DPHASE_INVEST,
    DPHASE_BID,
    DPHASE_ACQUISITION,
    DPHASE_ACQ_OFFER,
    DPHASE_CLOSING,
    DPHASE_DIVIDENDS,
    DPHASE_ISSUE,
    DPHASE_IPO,
)


# =============================================================================
# SPARSE LEGAL-ACTION BUFFER WIDTH
# =============================================================================
#
# Per-phase sizes (``ACTION_SIZE_*``, ``MAX_ACTION_SIZE``) and the decision
# phase enum (``DPHASE_*``) are cimported from ``core.data`` above — that
# module is the single source of truth for both. ``MAX_LEGAL_ACTIONS`` is
# the pad width for the sparse legal-action buffer and lives here because
# it's a property of the enumeration API, not the policy head geometry.

cdef enum:
    # Padded-sparse buffer width. Every replay/IPC tensor pads to this size.
    # Legal counts above this are considered a bug (see sparse-refactor.md).
    # Kmax=256 is the Phase-1 default; revisit after legal-count profiling.
    MAX_LEGAL_ACTIONS = 256


# Module-private C array of per-phase sizes, indexed by ``DecisionPhase``.
# Filled in at import time from the ``ActionSize`` enum so callers can look
# sizes up without a per-phase switch. Named ``_PHASE_ACTION_SIZES_C`` to
# avoid confusion with the Python-level ``PHASE_ACTION_SIZES`` list exported
# by ``core.data`` — that list is the Python consumer surface; this table
# is the Cython-side lookup.
cdef int _PHASE_ACTION_SIZES_C[8]


# =============================================================================
# ACTION TYPE ENUM (decoded action semantics)
# =============================================================================

cdef enum ActionType:
    # ACTION_PASS is the universal "opt out" for every phase that has one:
    # INVEST pass, BID leave-auction, ACQUISITION pass, ACQ_OFFER pass,
    # CLOSING pass, ISSUE pass, IPO pass. They all decode to ACTION_PASS.
    ACTION_PASS = 0
    ACTION_AUCTION = 1         # INVEST: start auction on company_id at face+amount
    ACTION_BUY_SHARE = 2       # INVEST: buy corp_id
    ACTION_SELL_SHARE = 3      # INVEST: sell corp_id
    ACTION_RAISE = 4           # BID: raise current bid by (face + 1 + amount)
    ACTION_ACQ_PRICE = 5       # ACQUISITION: corp_id acquires company_id at low+amount
    ACTION_ACQ_FI_BUY = 6      # ACQUISITION: corp_id buys company_id from FI at fixed price
    ACTION_ACQ_OFFER_BUY = 7   # ACQ_OFFER: preempting corp takes the offered company
    ACTION_CLOSE = 8           # CLOSING: close company_id
    ACTION_DIVIDEND = 9        # DIVIDENDS: pay dividend of amount
    ACTION_ISSUE = 10          # ISSUE: issue one share
    ACTION_IPO = 11            # IPO: start corp_id at par index amount


# =============================================================================
# ACTION INFO STRUCT (result of decode_action)
# =============================================================================

cdef struct ActionInfo:
    int phase           # DecisionPhase
    int action_type     # ActionType
    int corp_id         # -1 if not applicable
    int company_id      # -1 if not applicable
    int amount          # bid_offset / price_offset / dividend / par_index / raise_offset / -1


# =============================================================================
# ENCODE HELPERS (pure arithmetic, no state)
# =============================================================================
#
# Each phase exposes one encode function per action kind. All inputs are
# assumed valid (asserts guard bounds). The functions are ``inline`` nogil
# so Cython can fold them away in hot paths.

cdef inline int encode_invest_pass() noexcept nogil:
    return 0

cdef inline int encode_invest_auction(int company_id, int bid_offset) noexcept nogil:
    return 1 + company_id * 15 + bid_offset

cdef inline int encode_invest_buy(int corp_id) noexcept nogil:
    return 541 + corp_id * 2

cdef inline int encode_invest_sell(int corp_id) noexcept nogil:
    return 541 + corp_id * 2 + 1

cdef inline int encode_bid_pass() noexcept nogil:
    # "Leave the current auction" is a pass-class action.
    return 0

cdef inline int encode_bid_raise(int raise_offset) noexcept nogil:
    return 1 + raise_offset

cdef inline int encode_acquisition_pass() noexcept nogil:
    return 0

cdef inline int encode_acquisition_price(int corp_id, int company_id, int price_offset) noexcept nogil:
    return 1 + (corp_id * 36 + company_id) * 52 + price_offset

cdef inline int encode_acquisition_fi_buy(int corp_id, int company_id) noexcept nogil:
    return 1 + (corp_id * 36 + company_id) * 52 + 51

cdef inline int encode_acq_offer_pass() noexcept nogil:
    return 0

cdef inline int encode_acq_offer_buy() noexcept nogil:
    return 1

cdef inline int encode_closing_pass() noexcept nogil:
    return 0

cdef inline int encode_closing_close(int company_id) noexcept nogil:
    return 1 + company_id

cdef inline int encode_dividends(int amount) noexcept nogil:
    return amount

cdef inline int encode_issue_pass() noexcept nogil:
    return 0

cdef inline int encode_issue_issue() noexcept nogil:
    return 1

cdef inline int encode_ipo_pass() noexcept nogil:
    return 0

cdef inline int encode_ipo(int corp_id, int par_index) noexcept nogil:
    return 1 + corp_id * 14 + par_index


# Reverse of decode_action: pack an ActionInfo back into its phase-local id.
# Used by the import-time roundtrip tests and by any code path that builds
# an ActionInfo programmatically (e.g. replay diagnostics). Asserts on
# invalid combinations.
cdef int encode_action(ActionInfo info) noexcept nogil


# =============================================================================
# DECODE
# =============================================================================
#
# The ``phase_id`` argument is required — action IDs are phase-local, so the
# decoder has to know which encoding to invert. Returns a fully-populated
# ``ActionInfo`` with unused fields set to -1.

cdef ActionInfo decode_action(int phase_id, int action_id) noexcept nogil


# =============================================================================
# DECISION PHASE BRIDGE
# =============================================================================
#
# Reads the engine phase from ``state`` and maps it to a ``DecisionPhase``
# via ``core.data.ENGINE_TO_DECISION_PHASE``. That table maps the 12 engine
# ``GamePhases`` into the 8-phase decision space; automated/terminal phases
# (WRAP_UP, INCOME, END_CARD, GAME_OVER) map to -1 and the driver fast-
# forwards through them without consulting this module.

cdef int get_decision_phase(GameState state) noexcept nogil


# =============================================================================
# SPARSE LEGAL-ACTION ENUMERATION
# =============================================================================
#
# Writes legal phase-local action IDs into ``action_ids`` in a deterministic,
# phase-specific order and returns the count. The buffer must hold at least
# ``MAX_LEGAL_ACTIONS`` slots. An assert fires if a phase produces more legal
# actions than ``MAX_LEGAL_ACTIONS`` — that is a configuration bug.
#
# The caller owns ``phase_id`` lookup (usually via ``get_decision_phase``).
# Keeping it explicit lets callers scope enumeration to a specific phase
# without re-reading the state.
#
# NOTE: This is the skeleton. Phase-specific legality logic lives in
# ``_enumerate_*`` helpers in ``actions.pyx`` and will be filled in a
# follow-up. The contract (return count, fill buffer in deterministic
# order) is the stable piece.

cdef int enumerate_legal_actions(
    GameState state,
    int phase_id,
    uint16_t* action_ids,
) noexcept nogil


# Forced-action helper. Enumerates into a temporary buffer; if exactly one
# legal action is found, returns ``(action_id, 1)``. Otherwise returns
# ``(-1, 0)``. Used by the driver to fast-forward through single-choice
# decision points.
#
# Declared as ``cdef`` (not ``cpdef``) and with a ``noexcept`` return so it
# can live on the nogil hot path. A Python wrapper is provided below.

cdef (int, bint) get_forced_action(GameState state) noexcept nogil


# =============================================================================
# PYTHON-ACCESSIBLE WRAPPERS
# =============================================================================

cpdef int get_phase_action_size(int phase_id)
cpdef tuple decode_action_py(int phase_id, int action_id)
cpdef object enumerate_legal_actions_py(GameState state, int phase_id=*)
cpdef tuple get_forced_action_py(GameState state)
