# cython: language_level=3
"""
Per-phase action encoding for the transformer refactor.

Replaces the old global action vector with **phase-local** action IDs. The
encoding defined here is the single source of truth for:

  - what integer represents what game action in each decision phase
  - how replay buffers serialize ``(phase_id, action_id)`` pairs
  - how the engine decodes a sparse legal candidate into a phase handler call
  - how ``nn/transformer.py`` interprets its policy head outputs

The per-phase action counts **must** match
``nn/transformer.py::PHASE_ACTION_SIZES`` exactly. Any divergence silently
misaligns replay targets with model logits. Encode/decode roundtrips are
tested at import time; changing the layout here requires a matching change
to the model module.

See ``transformers.md`` and ``sparse-refactor.md`` for motivation.

Conventions
-----------

- **Decision phases** are the 8 phases the model sees, distinct from the
  engine's 12 ``GamePhases`` — notably the engine's ``ACQUISITION`` splits
  into ``DPHASE_ACQUISITION`` + ``DPHASE_ACQ_OFFER``, and ``IPO`` / ``PAR``
  collapse into ``DPHASE_IPO``.
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


# =============================================================================
# DECISION PHASES (what the model sees)
# =============================================================================

cdef enum DecisionPhase:
    DPHASE_INVEST = 0
    DPHASE_BID = 1
    DPHASE_ACQUISITION = 2
    DPHASE_ACQ_OFFER = 3
    DPHASE_CLOSING = 4
    DPHASE_DIVIDENDS = 5
    DPHASE_ISSUE = 6
    DPHASE_IPO = 7
    NUM_DECISION_PHASES = 8


# =============================================================================
# ACTION SPACE SIZES (must match nn/transformer.py::PHASE_ACTION_SIZES)
# =============================================================================

cdef enum:
    ACTION_SIZE_INVEST = 557        # 1 pass + 36*15 auction + 8*2 trade
    ACTION_SIZE_BID = 15            # 1 leave + 14 raises
    ACTION_SIZE_ACQUISITION = 14977 # 1 pass + 8*36*52 corp x company x {51 price + FI_BUY}
    ACTION_SIZE_ACQ_OFFER = 2       # pass + buy
    ACTION_SIZE_CLOSING = 37        # 1 pass + 36 company closes
    ACTION_SIZE_DIVIDENDS = 26      # dividend amounts 0..25
    ACTION_SIZE_ISSUE = 2           # pass + issue
    ACTION_SIZE_IPO = 113           # 1 pass + 8*14 corp x par index

    MAX_ACTION_SIZE = 14977         # max over all phases (ACQUISITION)

    # Padded-sparse buffer width. Every replay/IPC tensor pads to this size.
    # Legal counts above this are considered a bug (see sparse-refactor.md).
    # Kmax=256 is the Phase-1 default; revisit after legal-count profiling.
    MAX_LEGAL_ACTIONS = 256


# Module-level C array of per-phase sizes, indexed by DecisionPhase. Filled
# in at import time from the constants above so callers can look sizes up
# without a switch.
cdef int PHASE_ACTION_SIZES[8]


# =============================================================================
# ACTION TYPE ENUM (decoded action semantics)
# =============================================================================

cdef enum ActionType:
    ACTION_PASS = 0
    ACTION_AUCTION = 1         # INVEST: start auction on company_id at face+amount
    ACTION_BUY_SHARE = 2       # INVEST: buy corp_id
    ACTION_SELL_SHARE = 3      # INVEST: sell corp_id
    ACTION_LEAVE = 4           # BID: leave the current auction
    ACTION_RAISE = 5           # BID: raise current bid by (face + 1 + amount)
    ACTION_ACQ_PRICE = 6       # ACQUISITION: corp_id acquires company_id at low+amount
    ACTION_ACQ_FI_BUY = 7      # ACQUISITION: corp_id buys company_id from FI at fixed price
    ACTION_ACQ_OFFER_BUY = 8   # ACQ_OFFER: preempting corp takes the offered company
    ACTION_CLOSE = 9           # CLOSING: close company_id
    ACTION_DIVIDEND = 10       # DIVIDENDS: pay dividend of amount
    ACTION_ISSUE = 11          # ISSUE: issue one share
    ACTION_IPO = 12            # IPO: start corp_id at par index amount


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

cdef inline int encode_bid_leave() noexcept nogil:
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
# Maps engine ``GamePhases`` to the 8-phase decision space. The engine now
# distinguishes ``ACQUISITION`` from ``ACQ_OFFER`` directly in its phase
# enum, so this is a straight 1:1 table lookup. Engine phases without a
# corresponding decision phase (WRAP_UP, INCOME, END_CARD, GAME_OVER) are
# automated/terminal and return -1 — the driver auto-applies those without
# consulting the action module.

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
