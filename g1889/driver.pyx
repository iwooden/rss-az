"""GameDriver for the 1889 engine; see driver.pxd.

Handlers are ``cdef void apply_<phase>(GameState, ActionInfo*)`` in
``g1889/phases/`` and assume legal actions. Each joins ``_dispatch`` or
``_run_automated_step`` in its sub-step (engine-design.md §11); until then the
driver raises NotImplementedError on reaching it.
"""

from libc.stdint cimport uint16_t

from g1889.actions cimport enumerate_policy_actions, get_decision_phase, is_legal_action
from g1889.constants cimport NUM_STEPS, ROUND_GAME_OVER, STEP_GAME_OVER, STEP_TO_DECISION_PHASE
from g1889.entities.turn cimport turn_round, turn_step
from g1889.state cimport GameState

from g1889.actions import decode_action_py
from g1889.constants import DecisionPhase, Step

# Reaching it means a step that never advances or an endless forced chain.
cdef enum:
    MAX_AUTO_CHAIN_ITERATIONS = 1000


cdef class GameDriver:
    """Stateless driver singleton."""

    cdef int _forced_action(self, GameState state) except -2:
        """The only policy action of the current decision, or -1 if there are more."""
        cdef uint16_t ids[102]  # MAX_ACTION_SIZE
        cdef int count = enumerate_policy_actions(state, ids)
        if count == 0:
            raise RuntimeError(f"no policy actions at step {Step(turn_step(state)).name}")
        return ids[0] if count == 1 else -1

    cdef int _dispatch(self, GameState state, int phase, int action_id, object history) except -1:
        if history is not None:
            history.append((state._array.copy(), phase, action_id))
        raise NotImplementedError(
            f"1889 {DecisionPhase(phase).name} handler not implemented yet: {decode_action_py(phase, action_id)}")

    cdef int _run_automated_step(self, GameState state, int step, object history) except -1:
        if not 0 <= step < <int>NUM_STEPS or STEP_TO_DECISION_PHASE[step] >= 0 or step == STEP_GAME_OVER:
            raise RuntimeError(f"step {step} is not an automated step")
        if history is not None:
            history.append((state._array.copy(), -1, step))
        raise NotImplementedError(f"1889 automated step {Step(step).name} not implemented yet")

    cdef int _auto_chain(self, GameState state, object history) except -1:
        cdef int i, step, phase, action_id
        for i in range(<int>MAX_AUTO_CHAIN_ITERATIONS):
            if turn_round(state) == ROUND_GAME_OVER:
                return STATUS_GAME_OVER
            step = turn_step(state)
            phase = get_decision_phase(state)
            if phase < 0:
                self._run_automated_step(state, step, history)
                continue
            action_id = self._forced_action(state)
            if action_id < 0:
                return STATUS_OK
            self._dispatch(state, phase, action_id, history)
        raise RuntimeError(f"auto-chain exceeded {MAX_AUTO_CHAIN_ITERATIONS} iterations")

    cpdef int apply_action(self, GameState state, int action_id, object history=None) except -1:
        """Validate ``action_id`` against engine legality, apply it, and auto-chain.

        Returns STATUS_OK, STATUS_GAME_OVER, or STATUS_INVALID. Exact amounts
        and policy-restricted actions are accepted whenever they are legal.
        """
        if turn_round(state) == ROUND_GAME_OVER:
            return STATUS_GAME_OVER
        cdef int phase = get_decision_phase(state)
        if phase < 0 or not is_legal_action(state, action_id):
            return STATUS_INVALID
        self._dispatch(state, phase, action_id, history)
        return self._auto_chain(state, history)


DRIVER = GameDriver()

# Python mirrors: a bare assignment would be shadowed by the cdef names.
for _name, _member in ActionStatus.__members__.items():
    globals()[_name] = int(_member)
del _name, _member
