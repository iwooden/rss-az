"""GameDriver for the 1889 engine; see driver.pxd.

Handlers are ``cdef void apply_<phase>(GameState, const ActionInfo*) except *``
in ``g1889/phases/`` and assume legal actions; ``except *`` lets an entity
assertion reach the caller. History is recorded once a handler or step returns.
"""

from libc.stdint cimport uint16_t

from g1889.actions cimport (
    ACTION_BUY_PRIVATE, ACTION_DR_EXCHANGE, ACTION_MF_PORT, ActionInfo, decode_action, enumerate_legal_actions,
    enumerate_policy_actions, get_decision_phase, has_exact_amounts, is_legal_action, out_of_turn_legal,
    override_bid_legal, override_slot_legal, wide_amount_range,
)
from g1889.constants cimport (
    DPHASE_AUCTION, DPHASE_BID, DPHASE_BUY_PRIVATE_POST, DPHASE_BUY_PRIVATE_PRE, DPHASE_BUY_TRAIN, DPHASE_DISCARD,
    DPHASE_DIVIDEND, DPHASE_EMERGENCY_SELL_AMOUNT, DPHASE_ER_BONUS, DPHASE_PAR, DPHASE_PORT, DPHASE_PRIVATE_PRICE,
    DPHASE_SELECT_TRAIN, DPHASE_SELL_AMOUNT, DPHASE_STOCK, DPHASE_TILE, DPHASE_TOKEN, DPHASE_TRACK,
    DPHASE_TRAIN_PRICE, MAX_ACTION_SIZE, NUM_STEPS, ROUND_GAME_OVER, ROUND_OPERATING, STEP_DIVIDEND,
    STEP_GAME_OVER, STEP_OR_END, STEP_OR_START, STEP_ROUTE, STEP_SR_END, STEP_TO_DECISION_PHASE, STEP_TURN_END,
    STEP_TURN_START,
)
from g1889.data.tables cimport PRIVATE_TR
from g1889.entities.turn cimport TurnState, turn_operating_corp, turn_round, turn_step
from g1889.phases.auction cimport apply_auction, apply_bid, select_bid
from g1889.phases.buy_train cimport (
    apply_buy_train, apply_discard, apply_emergency_sell_amount, apply_select_train, apply_train_price,
)
from g1889.phases.dividend cimport apply_dividend
from g1889.phases.operating cimport (
    apply_tile, apply_token, apply_track, run_or_end, run_or_start, run_route, run_turn_end, run_turn_start,
)
from g1889.phases.privates cimport (
    apply_buy_private_post, apply_buy_private_pre, apply_dr_out_of_turn, apply_er_bonus, apply_interjection,
    apply_port, apply_private_price,
)
from g1889.phases.stock cimport apply_par, apply_sell_amount, apply_stock, run_sr_end
from g1889.state cimport GameState

from g1889.constants import Step
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
        cdef uint16_t ids[46]  # MAX_ACTION_SIZE
        cdef int count = enumerate_policy_actions(state, ids)
        cdef int action_id, phase
        assert count > 0, f"_forced_action: no policy actions at step {Step(turn_step(state)).name}"
        if count != 1:
            return -1
        action_id = ids[0]
        phase = get_decision_phase(state)
        if has_exact_amounts(phase) and enumerate_legal_actions(state, ids) == 1 and wide_amount_range(state, phase):
            return -1
        return action_id

    cdef int _dispatch(self, GameState state, int phase, int action_id, object history) except -1:
        """Apply one decision. The privates' slots interject in several phases
        (§5.1), so they dispatch by action type."""
        cdef object row = state._array.copy() if history is not None else None
        cdef ActionInfo info = decode_action(phase, action_id)
        if info.type == ACTION_BUY_PRIVATE or info.type == ACTION_DR_EXCHANGE or info.type == ACTION_MF_PORT:
            apply_interjection(state, &info)
        elif phase == DPHASE_AUCTION:
            apply_auction(state, &info)
        elif phase == DPHASE_BID:
            apply_bid(state, &info)
        elif phase == DPHASE_STOCK:
            apply_stock(state, &info)
        elif phase == DPHASE_PAR:
            apply_par(state, &info)
        elif phase == DPHASE_SELL_AMOUNT:
            apply_sell_amount(state, &info)
        elif phase == DPHASE_BUY_PRIVATE_PRE:
            apply_buy_private_pre(state, &info)
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
        elif phase == DPHASE_SELECT_TRAIN:
            apply_select_train(state, &info)
        elif phase == DPHASE_TRAIN_PRICE:
            apply_train_price(state, &info)
        elif phase == DPHASE_EMERGENCY_SELL_AMOUNT:
            apply_emergency_sell_amount(state, &info)
        elif phase == DPHASE_DISCARD:
            apply_discard(state, &info)
        elif phase == DPHASE_BUY_PRIVATE_POST:
            apply_buy_private_post(state, &info)
        elif phase == DPHASE_PRIVATE_PRICE:
            apply_private_price(state, &info)
        elif phase == DPHASE_ER_BONUS:
            apply_er_bonus(state, &info)
        elif phase == DPHASE_PORT:
            apply_port(state, &info)
        else:
            assert False, f"_dispatch: unknown decision phase {phase}"
        if history is not None:
            history.append((row, phase, action_id))
        return 0

    cdef int _run_automated_step(self, GameState state, int step, object history) except -1:
        """Run one automated step."""
        assert 0 <= step < <int>NUM_STEPS and STEP_TO_DECISION_PHASE[step] < 0 and step != STEP_GAME_OVER, \
            f"_run_automated_step: step {step} is not an automated step"
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
            assert False, f"_run_automated_step: step {Step(step).name} has no handler"
        if history is not None:
            history.append((row, -1, step))
        return 0

    cdef int _auto_chain(self, GameState state, object history) except -1:
        cdef int i, phase, action_id
        for i in range(<int>MAX_AUTO_CHAIN_ITERATIONS):
            if turn_round(state) == ROUND_GAME_OVER:
                return STATUS_GAME_OVER
            phase = get_decision_phase(state)
            if phase < 0:
                self._run_automated_step(state, turn_step(state), history)
                continue
            action_id = self._forced_action(state)
            if action_id < 0:
                return STATUS_OK
            self._dispatch(state, phase, action_id, history)
        assert False, f"_auto_chain: exceeded {MAX_AUTO_CHAIN_ITERATIONS} iterations"
        return STATUS_OK

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

    cpdef int override_private_slot(self, GameState state, int action_id, object history=None) except -1:
        """Replay only: a human's purchase, DR exchange, or port tile at a
        decision of the corp's turn that offers no such slot (TRACK, TOKEN,
        DIVIDEND, or BUY_TRAIN; Ruby joins them to whatever step it stops at),
        or at ROUTE, which then runs on what the move left. ``action_id`` names
        it as BUY_PRIVATE_PRE's slot does, and it runs as that slot does:
        PRIVATE_PRICE, then ER_BONUS if due, or PORT, or the exchange in place,
        and back to this step. Then it auto-chains; the human's exact price goes
        through ``apply_action`` as usual. History gets the auto-chain, not the
        override. The pipeline never calls it. Returns STATUS_INVALID where the
        slot is not legal or the decision has its own (``override_slot_legal``),
        else as ``apply_action``."""
        cdef ActionInfo info
        if not override_slot_legal(state, action_id):
            return STATUS_INVALID
        info = decode_action(DPHASE_BUY_PRIVATE_PRE, action_id)
        apply_interjection(state, &info)
        return self._auto_chain(state, history)

    cpdef int apply_out_of_turn(self, GameState state, int private, object history=None) except -1:
        """Replay only: ``private``'s ability from its owner, whoever's decision
        this is, as Ruby's Exchange step joins whatever step it stops at; only
        DR's exchange (``out_of_turn_legal``). The decision goes on, then it
        auto-chains. History gets the auto-chain, not the exchange. The pipeline
        never calls it. Returns STATUS_INVALID where it is not legal, else as
        ``apply_action``."""
        if not out_of_turn_legal(state, private):
            return STATUS_INVALID
        apply_dr_out_of_turn(state)
        return self._auto_chain(state, history)

    cpdef int override_topmost_bid(self, GameState state, object history=None) except -1:
        """Replay only: a human's bid on private 0, which Ruby takes as a
        purchase at the bid. Private 0 is the topmost while it is unsold, so
        it is the one private with no bid slot (engine-design.md §12 item 8);
        this opens BID on it, as a later private's slot does, then auto-chains,
        and BID's amount goes through ``apply_action`` as usual. History gets
        the auto-chain, not the override. The pipeline never calls it. Returns
        STATUS_INVALID where AUCTION would not offer the bid
        (``override_bid_legal``), else as ``apply_action``."""
        if not override_bid_legal(state):
            return STATUS_INVALID
        select_bid(state, <int>PRIVATE_TR)
        return self._auto_chain(state, history)

    def forced_action_py(self, GameState state):
        """The id the driver would apply by itself at the current decision, or -1."""
        return self._forced_action(state)


cdef GameDriver _DRIVER = GameDriver()
DRIVER = _DRIVER


cpdef int apply_alone(GameState state, int action_id) except -1:
    """Apply one decision by its handler alone (``_dispatch``, interjections
    included), without the automated steps and forced decisions
    ``apply_action`` runs after it, and record no history: what extraction's
    choice table diffs (1889/v2-design.md §5.3). The state may be left at an
    automated step, mid-step, or with another player to act.

    The action must be engine-legal here (``is_legal_action``). Like the
    handlers, this trusts its caller, extraction, which applies the ids
    ``enumerate_legal_actions`` gives: checking each would enumerate the
    decision again. Returns 0."""
    cdef int phase = get_decision_phase(state)
    assert phase >= 0 and turn_round(state) != ROUND_GAME_OVER, \
        f"apply_alone: no decision at step {Step(turn_step(state)).name}"
    assert action_id >= 0, f"apply_alone: action id {action_id}"
    return _DRIVER._dispatch(state, phase, action_id, None)


# Python mirrors: a bare assignment would be shadowed by the cdef names.
for _name, _member in ActionStatus.__members__.items():
    globals()[_name] = int(_member)
del _name, _member

# _forced_action's buffer is sized with the literal 46.
assert <int>MAX_ACTION_SIZE == 46
