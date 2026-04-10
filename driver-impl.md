# `core/driver` rewrite — implementation plan

Beads: rss-az-t3b7. Depends on rss-az-848a (legal-action enumerators).

## What we have

- **Engine layer (built):** `core/state`, `core/data`, `core/actions`, `entities/*`.
- **Phase handlers (built):** `phases/invest.pyx`, `phases/bid.pyx`. Both expose `cdef void apply_<phase>_action(GameState state, ActionInfo* info) noexcept` — no return value, legality is assumed.
- **Action plumbing (built):** `decode_action`, `enumerate_legal_actions`, `get_forced_action`, `get_decision_phase` already live in `core/actions`. The driver does **not** need to duplicate any of these.
- **Engine→decision phase mapping (built):** `ENGINE_TO_DECISION_PHASE[12]` in `core/data`. Automated/terminal phases (`WRAP_UP`, `INCOME`, `END_CARD`, `GAME_OVER`) map to `-1`.
- **Enumerators (partial):** `_enumerate_invest` and `_enumerate_bid` are filled. The other six (`_enumerate_acquisition`, `_acq_offer`, `_closing`, `_dividends`, `_issue`, `_ipo`) return `0` — covered by 848a.
- **Phase handlers (missing):** `acquisition`, `acq_offer`, `closing`, `dividends`, `issue`, `ipo`, plus the three automated phases (`wrap_up`, `income`, `end_card`). Tracked separately under rss-az-trvp.

## Goal

Land a working `core/driver.{pxd,pyx}` that:
1. Accepts a phase-local `action_id` from the caller.
2. Validates it against the legal set for the current decision phase, decodes it, and dispatches to the right phase handler.
3. After every applied action, fast-forwards through forced decisions and automated phases until either (a) a real branching decision is reached, or (b) the game ends.
4. Returns a status code distinguishing OK / GAME_OVER / INVALID.

The driver is built to the new contract from day one. It is **only** end-to-end testable for INVEST → BID → INVEST loops until the rest of the phase handlers and enumerators land. That's fine — the driver itself doesn't change shape when more handlers come online; only the dispatch table and the not-yet-implemented phase imports do.

## API shape

```cython
# core/driver.pxd
from core.state cimport GameState

cdef enum ActionStatus:
    STATUS_OK = 0          # Action applied; another decision is pending
    STATUS_GAME_OVER = 1   # Engine reached PHASE_GAME_OVER
    STATUS_INVALID = 2     # action_id was not legal in the current phase

cdef class GameDriver:
    cpdef int apply_action(self, GameState state, int action_id) except -1
    cpdef object get_legal_actions(self, GameState state)
    cpdef int get_decision_phase(self, GameState state)
```

Module-level singleton: `DRIVER = GameDriver()` (matches the existing entity-handle pattern). All methods are stateless w.r.t. the driver.

### Why drop the old PAUSED status

`STATUS_PAUSED` existed only for the 18xx replay tests, which patched state at specific phase boundaries via `pause_before_acq_transition` / `pause_before_closing_transition` flags on `GameState`. Those flags don't exist on the new `GameState`, the replay tests are still broken on `core.driver`, and there are no current callers. Skip `PAUSED` for now; reintroduce if/when replay tests come back.

### Why drop `step_mode`

The old `state.step_mode` flag let callers disable the auto-forced-action loop. The new `GameState` still has the field (line 621 of `state.pyx`), but no current consumer needs it — MCTS will want forced-action skipping enabled by default, eval server too. Plan to keep the field unused on the driver path; if a future caller wants single-step semantics they can call the (proposed) `apply_action_no_chain` cdef helper directly. Don't expose it from the cpdef surface yet.

### Action-id semantics

Action IDs are phase-local. The caller is responsible for sampling/choosing the action against the **same** decision phase the driver is currently in. The driver reads `state`'s phase via `get_decision_phase(state)` at dispatch time and trusts the caller's id is for that phase. Mismatch is caught by the "is this id in the legal set?" check before dispatch.

The driver does **not** take an explicit `phase_id` argument. Rationale: it's redundant — phase is unambiguously determined by state, and threading two ints through MCTS just creates a way for them to drift.

## File layout

```
core/driver.pxd
core/driver.pyx
```

That's it. No package split. The phase-handler dispatch is a single switch in `_dispatch_action`.

`setup.py` adds `'core/driver.pyx'` to the explicit `pyx_files` list (right after `core/actions.pyx`).

## Implementation outline

### `core/driver.pxd`

```cython
# cython: language_level=3
from core.state cimport GameState

cdef enum ActionStatus:
    STATUS_OK = 0
    STATUS_GAME_OVER = 1
    STATUS_INVALID = 2

cdef class GameDriver:
    cdef int _dispatch(self, GameState state, int phase_id, int action_id) noexcept
    cdef int _auto_chain(self, GameState state) noexcept
    cpdef int apply_action(self, GameState state, int action_id) except -1
    cpdef object get_legal_actions(self, GameState state)
    cpdef int get_decision_phase(self, GameState state)
```

Keep `_dispatch` and `_auto_chain` `cdef noexcept` so the apply_action hot path stays mostly C.

### `core/driver.pyx` — sections

**1. Imports**

```cython
from core.state cimport GameState
from core.actions cimport (
    ActionInfo,
    decode_action,
    enumerate_legal_actions,
    get_forced_action,
    get_decision_phase,
    MAX_LEGAL_ACTIONS,
)
from core.data cimport (
    GamePhases,
    DecisionPhase,
    DPHASE_INVEST, DPHASE_BID, DPHASE_ACQUISITION, DPHASE_ACQ_OFFER,
    DPHASE_CLOSING, DPHASE_DIVIDENDS, DPHASE_ISSUE, DPHASE_IPO,
)
from phases.invest cimport apply_invest_action
from phases.bid cimport apply_bid_action
# Future:
# from phases.acquisition cimport apply_acquisition_action
# from phases.acq_offer cimport apply_acq_offer_action
# from phases.closing cimport apply_closing_action
# from phases.dividends cimport apply_dividends_action
# from phases.issue cimport apply_issue_action
# from phases.ipo cimport apply_ipo_action
# from phases.wrap_up cimport apply_wrap_up
# from phases.income cimport apply_income
# from phases.end_card cimport apply_end_card

from entities import turn as turn_module
```

The pre-refactor pattern (in `phases/invest.pyx` and `bid.pyx`) is to do entity access via `from entities import <module>` at module load. Driver follows the same pattern for the small bits where it needs entity reads (currently just reading the engine phase to detect GAME_OVER, which we can also do via `get_decision_phase` + a check on the raw phase via `turn_module.TURN`).

**2. Constants**

```cython
DEF MAX_AUTO_CHAIN_ITERATIONS = 1000
```

Same loop guard as the old driver. Hitting it = bug, raise `RuntimeError`.

**3. `_dispatch` — phase-handler switch**

```cython
cdef int _dispatch(self, GameState state, int phase_id, int action_id) noexcept:
    cdef ActionInfo info = decode_action(phase_id, action_id)

    if phase_id == DPHASE_INVEST:
        apply_invest_action(state, &info)
    elif phase_id == DPHASE_BID:
        apply_bid_action(state, &info)
    elif phase_id == DPHASE_ACQUISITION:
        # apply_acquisition_action(state, &info)
        with gil:
            raise NotImplementedError("ACQUISITION handler not yet ported")
    elif phase_id == DPHASE_ACQ_OFFER:
        with gil:
            raise NotImplementedError("ACQ_OFFER handler not yet ported")
    elif phase_id == DPHASE_CLOSING:
        with gil:
            raise NotImplementedError("CLOSING handler not yet ported")
    elif phase_id == DPHASE_DIVIDENDS:
        with gil:
            raise NotImplementedError("DIVIDENDS handler not yet ported")
    elif phase_id == DPHASE_ISSUE:
        with gil:
            raise NotImplementedError("ISSUE handler not yet ported")
    elif phase_id == DPHASE_IPO:
        with gil:
            raise NotImplementedError("IPO handler not yet ported")
    else:
        with gil:
            raise AssertionError(f"_dispatch: unknown phase {phase_id}")

    return STATUS_OK
```

As phase handlers land, replace each `NotImplementedError` block with the real cimport+call. Each replacement is a one-line edit.

**4. Automated-phase handling**

The three automated engine phases (`WRAP_UP`, `INCOME`, `END_CARD`) don't have handlers yet. The driver needs a `_run_automated_phase(state, engine_phase)` helper that mirrors `_dispatch` but for the automated set:

```cython
cdef int _run_automated_phase(self, GameState state, int engine_phase) noexcept:
    if engine_phase == GamePhases.PHASE_WRAP_UP:
        with gil:
            raise NotImplementedError("WRAP_UP not yet ported")
    elif engine_phase == GamePhases.PHASE_INCOME:
        with gil:
            raise NotImplementedError("INCOME not yet ported")
    elif engine_phase == GamePhases.PHASE_END_CARD:
        with gil:
            raise NotImplementedError("END_CARD not yet ported")
    return STATUS_OK
```

Same shape as `_dispatch`. As `phases/wrap_up.pyx` etc. land, swap each `raise` for the real call.

**5. `_auto_chain` — fast-forward through forced + automated phases**

```cython
cdef int _auto_chain(self, GameState state) noexcept:
    cdef int iterations = 0
    cdef int engine_phase, decision_phase, action_id
    cdef bint found

    while iterations < MAX_AUTO_CHAIN_ITERATIONS:
        engine_phase = turn_module.TURN.get_phase(state)

        if engine_phase == GamePhases.PHASE_GAME_OVER:
            return STATUS_GAME_OVER

        # Automated phases: run, loop again to see what comes next.
        if (engine_phase == GamePhases.PHASE_WRAP_UP
                or engine_phase == GamePhases.PHASE_INCOME
                or engine_phase == GamePhases.PHASE_END_CARD):
            self._run_automated_phase(state, engine_phase)
            iterations += 1
            continue

        # Decision phase: ask for forced action.
        decision_phase = get_decision_phase(state)
        # decision_phase < 0 here would mean a phase mapped to -1 that we
        # didn't handle in the automated branch above — bug.
        if decision_phase < 0:
            with gil:
                raise AssertionError(
                    f"_auto_chain: unhandled non-decision engine phase {engine_phase}"
                )

        (action_id, found) = get_forced_action(state)
        if not found:
            # Multiple legal actions (or zero — see below). Caller's turn.
            return STATUS_OK

        self._dispatch(state, decision_phase, action_id)
        iterations += 1

    with gil:
        raise RuntimeError(
            f"_auto_chain: exceeded {MAX_AUTO_CHAIN_ITERATIONS} iterations"
        )
```

Note: `get_forced_action` returns `(-1, False)` for both *zero* legal actions and *2+* legal actions. The old driver distinguished these and raised `ZeroLegalActionsError` for the zero case. **We should preserve that** so unimplemented enumerators (currently returning 0) crash loudly instead of looking like "caller's turn".

The cleanest fix is to expose a helper that returns the count, not just the forced bit. Two options:

- **Option A (preferred):** Add a `cdef int count_legal_actions(GameState state, int phase_id) noexcept nogil` to `core/actions.{pxd,pyx}` that runs `enumerate_legal_actions` into a scratch buffer and returns the count. Driver calls it after `get_forced_action` returns `not found`, raises if count == 0. One extra enumeration on the rare non-forced branch — fine.
- **Option B:** Have the driver run the enumeration inline (it can already cimport `enumerate_legal_actions`) and skip `get_forced_action` entirely; the driver becomes the single owner of "forced or not" logic. Simpler, one fewer helper, slightly more code in the driver.

Option B is what I'd actually do — `get_forced_action` is a thin wrapper that loses information the driver wants. Inline the scratch-buffer enumeration in `_auto_chain`. The pattern:

```cython
cdef uint16_t scratch[256]  # MAX_LEGAL_ACTIONS
cdef int count = enumerate_legal_actions(state, decision_phase, scratch)
if count == 0:
    with gil:
        raise RuntimeError(
            f"zero legal actions in decision phase {decision_phase} "
            f"(engine phase {engine_phase}) — enumerator bug"
        )
if count == 1:
    self._dispatch(state, decision_phase, <int>scratch[0])
    iterations += 1
    continue
# count >= 2: caller's turn.
return STATUS_OK
```

This also lets `get_legal_actions` (Python wrapper) reuse the same enumeration codepath via the existing `enumerate_legal_actions_py`.

**6. `apply_action` — public entry point**

```cython
cpdef int apply_action(self, GameState state, int action_id) except -1:
    cdef int engine_phase = turn_module.TURN.get_phase(state)
    if engine_phase == GamePhases.PHASE_GAME_OVER:
        return STATUS_GAME_OVER

    cdef int decision_phase = get_decision_phase(state)
    if decision_phase < 0:
        # apply_action was called while the engine sits in an automated
        # phase — caller bug. The driver auto-chains through automated
        # phases, so the only way this fires is if the caller invoked
        # apply_action on a freshly-initialized state without first
        # advancing past WRAP_UP. Treat as INVALID rather than asserting
        # so test code can recover.
        return STATUS_INVALID

    # Validate action_id against the legal set for this phase.
    cdef uint16_t scratch[256]
    cdef int count = enumerate_legal_actions(state, decision_phase, scratch)
    cdef int i
    cdef bint legal = False
    for i in range(count):
        if scratch[i] == action_id:
            legal = True
            break
    if not legal:
        return STATUS_INVALID

    # Dispatch + auto-chain.
    self._dispatch(state, decision_phase, action_id)
    return self._auto_chain(state)
```

Linear scan over the legal-set for validation is fine: `MAX_LEGAL_ACTIONS = 256` and INVEST's worst case is ~107. No need for a sorted/binary-search path.

**7. Python helpers**

```cython
cpdef object get_legal_actions(self, GameState state):
    """Return (decision_phase, ndarray of legal action ids)."""
    from core.actions import enumerate_legal_actions_py
    return enumerate_legal_actions_py(state)

cpdef int get_decision_phase(self, GameState state):
    return get_decision_phase(state)
```

Both wrap existing functionality. Drop them if MCTS doesn't end up needing them on the driver — they exist elsewhere already.

**8. Module-level singleton + status constants**

```cython
DRIVER = GameDriver()

STATUS_OK_PY = STATUS_OK
STATUS_GAME_OVER_PY = STATUS_GAME_OVER
STATUS_INVALID_PY = STATUS_INVALID
```

## Initial-state bootstrap

Confirmed via `core/state.pyx:703-789`: `initialize_game` draws `num_players` companies straight into `LOC_AUCTION` via `move_to_auction()` (step 7) and lands in `PHASE_INVEST` with `active_player=0` and `coo_level=1` already seeded. A freshly-initialized state is ready for an INVEST decision on the very first `apply_action` call — no WRAP_UP fast-forward required to reach a real decision point. INVEST↔BID is therefore fully exercisable from `initialize_game` alone.

## Testing strategy

Until the other phase handlers and enumerators land, only INVEST↔BID is exercisable end-to-end. Tests to write *now* (in a new `tests/test_driver.py`):

1. **Construct a state with one auctionable company, run `DRIVER.apply_action(state, encode_invest_pass())` for all 3 players.** Verify the engine sits in `PHASE_WRAP_UP` (or `GAME_OVER` if init produced no companies). With WRAP_UP unimplemented, this should currently raise `NotImplementedError` — test asserts that.
2. **Start an auction via INVEST → AUCTION.** Verify driver lands in `PHASE_BID` with `active_company` set, `auction_high_bidder` = starter, and the next bidder up.
3. **Resolve a single-bidder auction via successive BID-leave actions.** Verify driver returns to INVEST after the resolve, with the company in the winner's possession and a `set_active_player_after(starter)` advance.
4. **Validate STATUS_INVALID** by passing an action id that's out of phase (e.g. an INVEST id while in BID).
5. **Validate STATUS_GAME_OVER** by manually constructing a state mid-auction where the buy completes the $75 trigger, then calling `apply_action` and asserting `STATUS_GAME_OVER`.
6. **Loop guard**: hard to test without a deliberately-buggy enumerator. Skip for now; rely on the assert.

These tests can live alongside the driver immediately and form the integration baseline. As more enumerators/handlers land, extend the test list.

## Resolved design decisions

1. **History tracking** — `(state.copy(), phase_id, action_id)` tuples. Phase id is included explicitly so replay code never has to re-derive it from the recorded state. The `history=` parameter itself is deferred until a concrete caller (replay tests / training data dump) asks for it.
2. **`_run_automated_phase` is a separate `cdef` helper** rather than inlined into `_auto_chain` — keeps both dispatch tables side-by-side and easy to scan.
3. **Initial-state bootstrap is a non-issue** (see above) — `initialize_game` lands directly in `PHASE_INVEST` with auctionable companies seeded.

## Implementation notes still worth flagging

- **`apply_action` exception spec.** Use `except -1` so Cython propagates exceptions raised inside `_dispatch` / `_auto_chain` / `_run_automated_phase`. Status codes are all non-negative so `-1` is unambiguous as the error sentinel.
- **Reject `apply_action` from an automated phase explicitly** via `get_decision_phase < 0 → STATUS_INVALID`. Faster than running an enumeration on a phase with no decision space, and more diagnostic.
- **`phases/__init__.pyx` is empty.** When automated-phase handlers (`wrap_up.pyx` etc.) land, they need to be added to `setup.py::pyx_files`. Driver landing first means the driver compiles against `NotImplementedError` stubs, not against missing imports. This is intentional — keeps the driver mergeable independent of phase-handler progress.

## Sequencing

1. **Land driver with the dispatch table stubbed** (NotImplementedError for unported phases). Driver compiles, INVEST/BID end-to-end works.
2. **Wire 848a enumerators.** As each enumerator lands, the corresponding phase becomes legality-checked but still NotImplementedError on dispatch.
3. **Wire phase handlers (rss-az-trvp).** As each handler lands, swap the `NotImplementedError` for the real call. Each is a one-line change.
4. **Wire automated phases.** Same pattern — replace the three `NotImplementedError`s in `_run_automated_phase`.
5. **Driver is then complete** and the MCTS / eval-server rewrites can proceed against a working game loop.

## Out of scope

- MCTS rewrite (rss-az-t3d2, blocked on this).
- Phase handlers other than INVEST/BID (rss-az-trvp).
- Replay-test reintegration (depends on driver + handlers).
- `STATUS_PAUSED` revival.
- `step_mode` plumbing.
- History recording.
