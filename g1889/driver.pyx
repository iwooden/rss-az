"""GameDriver for the 1889 engine; see driver.pxd.

Handlers are ``cdef void apply_<phase>(GameState, const ActionInfo*) except *``
in ``g1889/phases/`` and assume legal actions; ``except *`` lets an entity
assertion reach the caller. Each joins ``_dispatch`` or ``_run_automated_step``
in its sub-step (engine-design.md §11); until then the driver raises
NotImplementedError on reaching it. So does a handler or step whose branch a
later sub-step owns, before it changes anything; history is recorded only
once a handler or step returns, so the state and the history are left as
they were.
"""

from libc.stdint cimport uint16_t

from g1889.actions cimport (
    ActionInfo, decode_action, enumerate_legal_actions, enumerate_policy_actions, get_decision_phase, has_exact,
    is_legal_action, wide_amount_range,
)
from g1889.constants cimport (
    DPHASE_AUCTION, DPHASE_BID, DPHASE_BUY_PRIVATE, DPHASE_BUY_TRAIN, DPHASE_DISCARD, DPHASE_DIVIDEND, DPHASE_PAR,
    DPHASE_RAISE, DPHASE_STOCK, DPHASE_TILE, DPHASE_TOKEN, DPHASE_TRACK, DPHASE_TRAIN_PRICE, NUM_STEPS,
    ROUND_GAME_OVER, ROUND_OPERATING, STEP_DIVIDEND, STEP_GAME_OVER, STEP_OR_END, STEP_OR_START, STEP_ROUTE,
    STEP_SR_END, STEP_TO_DECISION_PHASE, STEP_TURN_END, STEP_TURN_START,
)
from g1889.entities.turn cimport TurnState, turn_operating_corp, turn_round, turn_step
from g1889.phases.auction cimport apply_auction, apply_bid, apply_raise
from g1889.phases.buy_train cimport apply_buy_train, apply_discard, apply_train_price
from g1889.phases.dividend cimport apply_dividend
from g1889.phases.operating cimport (
    apply_buy_private, apply_tile, apply_token, apply_track, run_or_end, run_or_start, run_route, run_turn_end,
    run_turn_start,
)
from g1889.phases.stock cimport apply_par, apply_stock, run_sr_end
from g1889.state cimport GameState

from g1889.actions import decode_action_py
from g1889.constants import DecisionPhase, Step
from g1889.entities import turn as _turn

cdef TurnState TURN = _turn.TURN

# Reaching it means a step that never advances or an endless forced chain.
cdef enum:
    MAX_AUTO_CHAIN_ITERATIONS = 1000


cdef class GameDriver:
    """Stateless driver singleton."""

    cdef int _forced_action(self, GameState state) except -2:
        """The only policy action of the current decision, or -1 if there is a
        choice. An amount decision whose one legal grid point has exact amounts
        above it is a choice (§6): forcing it would hide them from a human whose
        off-grid bid left a range narrower than a grid step. When the policy
        drops legal grid points (the pass cap), self-play never uses exact
        amounts, so its one point is forced."""
        cdef uint16_t ids[102]  # MAX_ACTION_SIZE
        cdef int count = enumerate_policy_actions(state, ids)
        cdef int action_id = ids[0]
        cdef int phase
        if count == 0:
            raise RuntimeError(f"no policy actions at step {Step(turn_step(state)).name}")
        if count > 1:
            return -1
        phase = get_decision_phase(state)
        if has_exact(phase) and enumerate_legal_actions(state, ids) == 1 and wide_amount_range(state, phase):
            return -1
        return action_id

    cdef int _dispatch(self, GameState state, int phase, int action_id, object history) except -1:
        """Apply one decision. A handler a later sub-step owns raises
        NotImplementedError before touching the state or the history."""
        cdef object row = state._array.copy() if history is not None else None
        cdef ActionInfo info = decode_action(phase, action_id)
        if phase == DPHASE_AUCTION:
            apply_auction(state, &info)
        elif phase == DPHASE_BID:
            apply_bid(state, &info)
        elif phase == DPHASE_RAISE:
            apply_raise(state, &info)
        elif phase == DPHASE_STOCK:
            apply_stock(state, &info)
        elif phase == DPHASE_PAR:
            apply_par(state, &info)
        elif phase == DPHASE_TRACK:
            apply_track(state, &info)
        elif phase == DPHASE_TILE:
            apply_tile(state, &info)
        elif phase == DPHASE_TOKEN:
            apply_token(state, &info)
        elif phase == DPHASE_DIVIDEND:
            apply_dividend(state, &info)
        elif phase == DPHASE_BUY_TRAIN:
            apply_buy_train(state, &info)
        elif phase == DPHASE_TRAIN_PRICE:
            apply_train_price(state, &info)
        elif phase == DPHASE_DISCARD:
            apply_discard(state, &info)
        elif phase == DPHASE_BUY_PRIVATE:
            apply_buy_private(state, &info)
        else:
            raise NotImplementedError(
                f"1889 {DecisionPhase(phase).name} handler not implemented yet: {decode_action_py(phase, action_id)}")
        if history is not None:
            history.append((row, phase, action_id))
        return 0

    cdef int _run_automated_step(self, GameState state, int step, object history) except -1:
        """Run one automated step. A step from a later sub-step raises
        NotImplementedError before touching the state or the history."""
        if not 0 <= step < <int>NUM_STEPS or STEP_TO_DECISION_PHASE[step] >= 0 or step == STEP_GAME_OVER:
            raise RuntimeError(f"step {step} is not an automated step")
        cdef object row = state._array.copy() if history is not None else None
        if step == STEP_SR_END:
            run_sr_end(state)
        elif step == STEP_OR_START:
            run_or_start(state)
        elif step == STEP_TURN_START:
            run_turn_start(state)
        elif step == STEP_ROUTE:
            run_route(state)
        elif step == STEP_TURN_END:
            run_turn_end(state)
        elif step == STEP_OR_END:
            run_or_end(state)
        else:
            raise NotImplementedError(f"1889 automated step {Step(step).name} not implemented yet")
        if history is not None:
            history.append((row, -1, step))
        return 0

    cdef int _auto_chain(self, GameState state, object history) except -1:
        cdef int i, step, phase, action_id
        for i in range(<int>MAX_AUTO_CHAIN_ITERATIONS):
            if turn_round(state) == ROUND_GAME_OVER:
                return STATUS_GAME_OVER
            step = turn_step(state)
            phase = get_decision_phase(state)
            try:
                if phase < 0:
                    self._run_automated_step(state, step, history)
                    continue
                action_id = self._forced_action(state)
                if action_id < 0:
                    return STATUS_OK
                self._dispatch(state, phase, action_id, history)
            except NotImplementedError:
                # TEMPORARY development stop, removed at 6.9 (§11): rest where
                # a later sub-step takes over (a decision whose legal set it
                # owns, such as a must-buy that needs emergency money, or an
                # automated step such as OR_END after a broken bank's last
                # OR), so replay can compare the state there.
                # Nothing has changed since the last decision or step. The next
                # apply_action fails loudly: NotImplementedError at a decision,
                # STATUS_INVALID at an automated step.
                return STATUS_OK
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

    cpdef int advance(self, GameState state, object history=None) except -1:
        """Run automated steps and forced decisions from the current state, as
        ``apply_action`` does after its action: for a state set up by hand at
        an automated step (tests, ``setup_operating_round_py``). Returns
        STATUS_OK or STATUS_GAME_OVER."""
        return self._auto_chain(state, history)

    cpdef int override_route_revenue(self, GameState state, int revenue, object history=None) except -1:
        """Replay only: make ``revenue`` the operating corp's revenue this turn,
        as Ruby pays a human's logged routes, then auto-chain, since zero
        revenue forces the withhold. Legal at the corp's DIVIDEND decision for
        any multiple of ¥10 from 0 up, with no upper bound: Ruby pays logged
        revenue unchecked, and 314 action 367 pays ¥400 where the most IR's
        trains can earn is ¥370 (engine-design.md §8). History gets the
        auto-chain, not the override. The pipeline never calls it. Returns
        STATUS_INVALID elsewhere, else as ``apply_action``."""
        if (turn_round(state) != ROUND_OPERATING or turn_step(state) != STEP_DIVIDEND
                or turn_operating_corp(state) < 0 or not 0 <= revenue < 1 << 15 or revenue % 10):
            return STATUS_INVALID
        TURN.set_route_revenue(state, revenue)
        return self._auto_chain(state, history)

    def forced_action_py(self, GameState state):
        """The id the driver would apply by itself at the current decision, or -1."""
        return self._forced_action(state)


DRIVER = GameDriver()

# Python mirrors: a bare assignment would be shadowed by the cdef names.
for _name, _member in ActionStatus.__members__.items():
    globals()[_name] = int(_member)
del _name, _member
