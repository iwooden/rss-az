"""
Game driver — implementation.

The driver routes a phase-local ``action_id`` from the caller through:

  1. ``apply_action`` — decision-phase / legality validation, single
     dispatch.
  2. ``_auto_chain`` — repeatedly fast-forwards through automated
     engine phases (WRAP_UP / INCOME / END_CARD) and forced decision
     points (decision phases with exactly one legal action) until
     either a multi-choice decision is reached or the engine hits
     ``PHASE_GAME_OVER``.

The legality check is a linear scan over the legal-action set produced
by ``core/actions.enumerate_legal_actions``. Worst case is bounded by
``MAX_LEGAL_ACTIONS = 1024``; INVEST's empirical worst case is ~107.

Phase handlers are ``cdef void apply_<phase>_action(state, ActionInfo*)
noexcept`` — they assume legality, which the driver guarantees. Phases
that have not yet been ported land on ``NotImplementedError`` stubs in
``_dispatch`` / ``_run_automated_phase``; replacing each stub is a
one-line edit when the corresponding ``phases/<phase>.pyx`` lands.

History tracking: callers may pass an optional ``history`` list. The
driver appends ``(state._array.copy(), phase_id, action_id)`` tuples
*before* applying each action. ``phase_id`` is recorded explicitly so
replay code does not have to re-derive it from the recorded state.
"""

from libc.stdint cimport uint16_t

from core.state cimport GameState
from core.actions cimport (
    ActionInfo,
    decode_action,
    enumerate_legal_actions,
    get_decision_phase,
)
from core.data cimport (
    GamePhases,
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
)
from core.driver cimport (
    ActionStatus,
    STATUS_OK,
    STATUS_GAME_OVER,
    STATUS_INVALID,
    STATUS_PAUSED,
)
from phases.invest cimport apply_invest_action
from phases.bid cimport apply_bid_action
from phases.wrap_up cimport apply_wrap_up
from phases.income cimport apply_income
from phases.end_card cimport apply_end_card
from phases.dividends cimport apply_dividend_action
from phases.issue cimport apply_issue_action
from phases.ipo cimport apply_ipo_action
from phases.par cimport apply_par_action
from phases.acq_select_corp cimport apply_acq_select_corp_action
from phases.acq_select_company cimport apply_acq_select_company_action
from phases.acq_select_price cimport apply_acq_select_price_action
from phases.acq_offer cimport apply_acq_offer_action
from phases.closing cimport apply_closing_action

# Late Python-level entity import — same pattern as ``phases/invest.pyx``
# and ``phases/bid.pyx``. The driver only needs ``TURN`` to read the raw
# engine phase between auto-chain steps; everything else flows through
# decode_action / phase handlers.
from entities import turn as turn_module


# =============================================================================
# CONSTANTS
# =============================================================================

# Auto-chain loop guard. Hitting this is a bug — either an enumerator is
# producing an infinite forced-action chain or an automated phase fails to
# advance. The old driver used the same value.
DEF MAX_AUTO_CHAIN_ITERATIONS = 1000


# =============================================================================
# DRIVER CLASS
# =============================================================================

cdef class GameDriver:
    """Stateless driver singleton — see ``driver-impl.md``."""

    def __cinit__(self):
        # Stateless — every method takes a ``GameState`` and reads/writes
        # against it directly. The class exists only to give Cython a
        # convenient namespace for the cdef hot-path helpers.
        pass

    cdef bint _is_automated_engine_phase(self, int engine_phase) noexcept nogil:
        return (
            engine_phase == GamePhases.PHASE_WRAP_UP
            or engine_phase == GamePhases.PHASE_INCOME
            or engine_phase == GamePhases.PHASE_END_CARD
        )

    cdef int _forced_action_or_negative_one(self, GameState state) except -2:
        cdef uint16_t scratch[1024]   # MAX_LEGAL_ACTIONS
        cdef int count = enumerate_legal_actions(state, scratch)
        assert count > 0, (
            f"_forced_action_or_negative_one: zero legal actions in engine phase "
            f"{turn_module.TURN.get_phase(state)} — enumerator bug or unported phase"
        )
        if count == 1:
            return <int>scratch[0]
        return -1

    # -------------------------------------------------------------------------
    # _dispatch — phase-handler switch
    # -------------------------------------------------------------------------

    cdef int _dispatch(
        self,
        GameState state,
        int action_id,
        object history,
    ) except -1:
        """Decode ``action_id`` and dispatch to the current phase's handler.

        Reads the decision phase from ``state`` via ``get_decision_phase``.
        Records ``(state._array.copy(), phase_id, action_id)`` to
        ``history`` *before* mutating state, so the recorded snapshot is
        the pre-action state — matching the convention the old driver used.
        Phase handlers are ``noexcept`` ``void`` and assume legality; the
        caller (``apply_action`` or ``_auto_chain``) is responsible for
        only passing legal ids.
        """
        cdef int phase_id = get_decision_phase(state)
        assert phase_id >= 0, f"_dispatch: state is in automated/terminal phase (decision_phase={phase_id})"

        if history is not None:
            history.append((state._array.copy(), phase_id, action_id))

        cdef ActionInfo info = decode_action(phase_id, action_id)

        if phase_id == DPHASE_INVEST:
            apply_invest_action(state, &info)
        elif phase_id == DPHASE_BID:
            apply_bid_action(state, &info)
        elif phase_id == DPHASE_ACQ_SELECT_CORP:
            apply_acq_select_corp_action(state, &info)
        elif phase_id == DPHASE_ACQ_SELECT_COMPANY:
            apply_acq_select_company_action(state, &info)
        elif phase_id == DPHASE_ACQ_SELECT_PRICE:
            apply_acq_select_price_action(state, &info)
        elif phase_id == DPHASE_ACQ_OFFER:
            apply_acq_offer_action(state, &info)
        elif phase_id == DPHASE_CLOSING:
            apply_closing_action(state, &info)
        elif phase_id == DPHASE_DIVIDENDS:
            apply_dividend_action(state, &info)
        elif phase_id == DPHASE_ISSUE:
            apply_issue_action(state, &info)
        elif phase_id == DPHASE_IPO:
            apply_ipo_action(state, &info)
        elif phase_id == DPHASE_PAR:
            apply_par_action(state, &info)
        else:
            assert False, f"_dispatch: unknown decision phase {phase_id}"

        return STATUS_OK

    # -------------------------------------------------------------------------
    # _run_automated_phase — non-decision phase switch
    # -------------------------------------------------------------------------

    cdef int _run_automated_phase(
        self,
        GameState state,
        int engine_phase,
        object history,
    ) except -1:
        """Run a single automated engine phase (WRAP_UP / INCOME / END_CARD).

        Recorded into ``history`` as ``(state._array.copy(), -1, engine_phase)``
        so replay code can distinguish automated transitions from real
        decisions: ``phase_id == -1`` flags the third tuple element as a
        ``GamePhases`` value rather than a ``DecisionPhase`` value.
        """
        if history is not None:
            history.append((state._array.copy(), -1, engine_phase))

        if engine_phase == GamePhases.PHASE_WRAP_UP:
            apply_wrap_up(state)
        elif engine_phase == GamePhases.PHASE_INCOME:
            apply_income(state)
        elif engine_phase == GamePhases.PHASE_END_CARD:
            apply_end_card(state)
        else:
            assert False, f"_run_automated_phase: phase {engine_phase} is not automated"

        return STATUS_OK

    # -------------------------------------------------------------------------
    # _auto_chain — fast-forward through automated phases & forced decisions
    # -------------------------------------------------------------------------

    cdef int _auto_chain(self, GameState state, object history) except -1:
        """Loop until the engine sits at a real (multi-choice) decision.

        Each iteration looks at the raw engine phase:

          - ``PHASE_GAME_OVER``: return ``STATUS_GAME_OVER``.
          - Automated phase (WRAP_UP / INCOME / END_CARD): run it and loop.
          - Decision phase: enumerate legal actions.
              * 0 actions  → enumerator bug (or unported); raise.
              * 1 action   → forced; dispatch and loop.
              * 2+ actions → caller's turn; return ``STATUS_OK``.

        ``_dispatch`` asserts ``get_decision_phase >= 0``, catching any
        engine phase that maps to neither a decision phase nor one of the
        three automated phases handled above.
        """
        cdef int iterations = 0
        cdef int engine_phase, action_id

        while iterations < MAX_AUTO_CHAIN_ITERATIONS:
            engine_phase = turn_module.TURN.get_phase(state)

            if engine_phase == GamePhases.PHASE_GAME_OVER:
                return STATUS_GAME_OVER

            if self._is_automated_engine_phase(engine_phase):
                self._run_automated_phase(state, engine_phase, history)
                iterations += 1
                continue

            action_id = self._forced_action_or_negative_one(state)
            if action_id < 0:
                # Real decision — hand control back to the caller.
                return STATUS_OK

            # Exactly one legal action: forced. Dispatch and loop.
            self._dispatch(state, action_id, history)
            iterations += 1

        assert False, f"_auto_chain: exceeded {MAX_AUTO_CHAIN_ITERATIONS} iterations"
        return STATUS_OK

    # -------------------------------------------------------------------------
    # Public driver inspection / stepping helpers
    # -------------------------------------------------------------------------

    cpdef bint is_non_player_phase(self, GameState state):
        cdef int engine_phase = turn_module.TURN.get_phase(state)
        if engine_phase == GamePhases.PHASE_GAME_OVER:
            return False
        if self._is_automated_engine_phase(engine_phase):
            return True
        return self._forced_action_or_negative_one(state) >= 0

    cpdef int advance_phase(
        self,
        GameState state,
        object history=None,
    ) except -1:
        cdef int engine_phase = turn_module.TURN.get_phase(state)
        cdef int action_id

        if engine_phase == GamePhases.PHASE_GAME_OVER:
            return STATUS_GAME_OVER

        if self._is_automated_engine_phase(engine_phase):
            self._run_automated_phase(state, engine_phase, history)
        else:
            action_id = self._forced_action_or_negative_one(state)
            if action_id < 0:
                return STATUS_INVALID
            self._dispatch(state, action_id, history)

        if turn_module.TURN.get_phase(state) == GamePhases.PHASE_GAME_OVER:
            return STATUS_GAME_OVER
        if state.step_mode:
            return STATUS_PAUSED
        return STATUS_OK

    # -------------------------------------------------------------------------
    # apply_action — public entry point
    # -------------------------------------------------------------------------

    cpdef int apply_action(
        self,
        GameState state,
        int action_id,
        object history=None,
    ) except -1:
        """Validate, dispatch, and auto-chain.

        Returns:
          * ``STATUS_OK``       — action applied; another decision pending.
          * ``STATUS_GAME_OVER`` — engine reached ``PHASE_GAME_OVER``
                                  during dispatch or auto-chain.
          * ``STATUS_INVALID``  — ``action_id`` is not legal in the current
                                  decision phase, or the engine is sitting
                                  in a non-decision phase (caller bug:
                                  the previous ``apply_action`` should
                                  have auto-chained past it, so this is
                                  only reachable on a brand-new state
                                  that was hand-built rather than coming
                                  from ``initialize_game``).
        """
        cdef int engine_phase = turn_module.TURN.get_phase(state)
        if engine_phase == GamePhases.PHASE_GAME_OVER:
            return STATUS_GAME_OVER

        cdef uint16_t scratch[1024]   # MAX_LEGAL_ACTIONS
        cdef int count = enumerate_legal_actions(state, scratch)
        cdef int i
        cdef bint legal = False
        for i in range(count):
            if scratch[i] == action_id:
                legal = True
                break
        if not legal:
            return STATUS_INVALID

        self._dispatch(state, action_id, history)
        if turn_module.TURN.get_phase(state) == GamePhases.PHASE_GAME_OVER:
            return STATUS_GAME_OVER
        if state.step_mode:
            return STATUS_PAUSED
        return self._auto_chain(state, history)

# =============================================================================
# MODULE-LEVEL SINGLETON + PYTHON CONSTANTS
# =============================================================================

DRIVER = GameDriver()

STATUS_OK_PY = STATUS_OK
STATUS_GAME_OVER_PY = STATUS_GAME_OVER
STATUS_INVALID_PY = STATUS_INVALID
STATUS_PAUSED_PY = STATUS_PAUSED
