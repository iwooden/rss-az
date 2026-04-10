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
``MAX_LEGAL_ACTIONS = 256``; INVEST's empirical worst case is ~107.

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
    DPHASE_ACQUISITION,
    DPHASE_ACQ_OFFER,
    DPHASE_CLOSING,
    DPHASE_DIVIDENDS,
    DPHASE_ISSUE,
    DPHASE_IPO,
)
from core.driver cimport (
    ActionStatus,
    STATUS_OK,
    STATUS_GAME_OVER,
    STATUS_INVALID,
)
from phases.invest cimport apply_invest_action
from phases.bid cimport apply_bid_action

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

    # -------------------------------------------------------------------------
    # _dispatch — phase-handler switch
    # -------------------------------------------------------------------------

    cdef int _dispatch(
        self,
        GameState state,
        int phase_id,
        int action_id,
        object history,
    ) except -1:
        """Decode ``action_id`` for ``phase_id`` and dispatch to its handler.

        Records ``(state._array.copy(), phase_id, action_id)`` to
        ``history`` *before* mutating state, so the recorded snapshot is
        the pre-action state — matching the convention the old driver used.
        Phase handlers are ``noexcept`` ``void`` and assume legality; the
        caller (``apply_action`` or ``_auto_chain``) is responsible for
        only passing legal ids.
        """
        if history is not None:
            history.append((state._array.copy(), phase_id, action_id))

        cdef ActionInfo info = decode_action(phase_id, action_id)

        if phase_id == DPHASE_INVEST:
            apply_invest_action(state, &info)
        elif phase_id == DPHASE_BID:
            apply_bid_action(state, &info)
        elif phase_id == DPHASE_ACQUISITION:
            raise NotImplementedError(
                "DPHASE_ACQUISITION handler not yet ported (rss-az-trvp)"
            )
        elif phase_id == DPHASE_ACQ_OFFER:
            raise NotImplementedError(
                "DPHASE_ACQ_OFFER handler not yet ported (rss-az-trvp)"
            )
        elif phase_id == DPHASE_CLOSING:
            raise NotImplementedError(
                "DPHASE_CLOSING handler not yet ported (rss-az-trvp)"
            )
        elif phase_id == DPHASE_DIVIDENDS:
            raise NotImplementedError(
                "DPHASE_DIVIDENDS handler not yet ported (rss-az-trvp)"
            )
        elif phase_id == DPHASE_ISSUE:
            raise NotImplementedError(
                "DPHASE_ISSUE handler not yet ported (rss-az-trvp)"
            )
        elif phase_id == DPHASE_IPO:
            raise NotImplementedError(
                "DPHASE_IPO handler not yet ported (rss-az-trvp)"
            )
        else:
            raise AssertionError(
                f"_dispatch: unknown decision phase {phase_id}"
            )

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
            raise NotImplementedError(
                "PHASE_WRAP_UP handler not yet ported (rss-az-trvp)"
            )
        elif engine_phase == GamePhases.PHASE_INCOME:
            raise NotImplementedError(
                "PHASE_INCOME handler not yet ported (rss-az-trvp)"
            )
        elif engine_phase == GamePhases.PHASE_END_CARD:
            raise NotImplementedError(
                "PHASE_END_CARD handler not yet ported (rss-az-trvp)"
            )
        else:
            raise AssertionError(
                f"_run_automated_phase: phase {engine_phase} is not automated"
            )

        # NOTE: every branch above currently raises, so a trailing
        # ``return STATUS_OK`` would be unreachable (Cython warns). When
        # the WRAP_UP / INCOME / END_CARD handlers land (rss-az-trvp),
        # each implemented branch should follow its ``apply_<phase>(state)``
        # call with ``return STATUS_OK``.

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

        ``get_decision_phase`` returning ``-1`` here would mean an engine
        phase that maps to neither a decision phase nor one of the three
        automated phases handled above — that's a programmer error in
        ``ENGINE_TO_DECISION_PHASE``.
        """
        cdef int iterations = 0
        cdef int engine_phase, decision_phase_id, count, i, action_id
        cdef uint16_t scratch[256]   # MAX_LEGAL_ACTIONS

        while iterations < MAX_AUTO_CHAIN_ITERATIONS:
            engine_phase = turn_module.TURN.get_phase(state)

            if engine_phase == GamePhases.PHASE_GAME_OVER:
                return STATUS_GAME_OVER

            if (engine_phase == GamePhases.PHASE_WRAP_UP
                    or engine_phase == GamePhases.PHASE_INCOME
                    or engine_phase == GamePhases.PHASE_END_CARD):
                self._run_automated_phase(state, engine_phase, history)
                iterations += 1
                continue

            decision_phase_id = get_decision_phase(state)
            if decision_phase_id < 0:
                raise AssertionError(
                    f"_auto_chain: engine phase {engine_phase} has no "
                    f"decision phase mapping and is not in the automated "
                    f"set — ENGINE_TO_DECISION_PHASE bug"
                )

            count = enumerate_legal_actions(state, decision_phase_id, scratch)
            if count == 0:
                raise RuntimeError(
                    f"_auto_chain: zero legal actions in decision phase "
                    f"{decision_phase_id} (engine phase {engine_phase}) — "
                    f"enumerator bug or unported phase"
                )
            if count >= 2:
                # Real decision — hand control back to the caller.
                return STATUS_OK

            # Exactly one legal action: forced. Dispatch and loop.
            action_id = <int>scratch[0]
            self._dispatch(state, decision_phase_id, action_id, history)
            iterations += 1

        raise RuntimeError(
            f"_auto_chain: exceeded {MAX_AUTO_CHAIN_ITERATIONS} iterations"
        )

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

        cdef int decision_phase_id = get_decision_phase(state)
        if decision_phase_id < 0:
            # Engine is in WRAP_UP / INCOME / END_CARD: caller jumped the
            # gun. Don't auto-chain on the caller's behalf — the contract
            # is "you call apply_action when it's your turn"; if the
            # engine isn't waiting on you, that's a caller bug.
            return STATUS_INVALID

        cdef uint16_t scratch[256]   # MAX_LEGAL_ACTIONS
        cdef int count = enumerate_legal_actions(state, decision_phase_id, scratch)
        cdef int i
        cdef bint legal = False
        for i in range(count):
            if scratch[i] == action_id:
                legal = True
                break
        if not legal:
            return STATUS_INVALID

        self._dispatch(state, decision_phase_id, action_id, history)
        return self._auto_chain(state, history)

    # -------------------------------------------------------------------------
    # Python convenience wrappers
    # -------------------------------------------------------------------------

    cpdef object legal_actions(self, GameState state):
        """Return ``(decision_phase, ndarray of legal action ids)``.

        Thin wrapper around ``core.actions.enumerate_legal_actions_py`` so
        callers (MCTS, tests) only need to import the driver to drive a
        game. The lazy import avoids pulling Python-level numpy bits into
        Cython's cimport graph at module load.
        """
        from core.actions import enumerate_legal_actions_py
        return enumerate_legal_actions_py(state)

    cpdef int decision_phase(self, GameState state):
        """Return the current ``DecisionPhase`` (or ``-1`` for automated/terminal)."""
        return get_decision_phase(state)


# =============================================================================
# MODULE-LEVEL SINGLETON + PYTHON CONSTANTS
# =============================================================================

DRIVER = GameDriver()

STATUS_OK_PY = STATUS_OK
STATUS_GAME_OVER_PY = STATUS_GAME_OVER
STATUS_INVALID_PY = STATUS_INVALID
