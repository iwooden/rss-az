---
phase: 23
plan: 03
subsystem: phase-integration
tags: [cython, phase-transitions, temp-end-turn]
dependency-graph:
  requires: ["23-01"]
  provides:
    - TEMP_END_TURN phase handler
    - Updated CLOSING->INCOME transition
    - Roundtrip clearing at end of INVEST
  affects: []
tech-stack:
  added: []
  patterns:
    - Non-player phase pattern (0 valid actions, auto-executes)
key-files:
  created:
    - phases/temp_end_turn.pyx
    - phases/temp_end_turn.pxd
  modified:
    - phases/closing.pyx
    - phases/invest.pyx
    - phases/__init__.pyx
    - phases/__init__.pxd
    - core/driver.pyx
    - tests/phases/test_closing.py
    - tests/phases/test_wrap_up.py
    - tests/test_integration.py
decisions:
  - key: "phase-transition-order"
    value: "CLOSING->INCOME->TEMP_END_TURN->INVEST"
    context: "Complete phase chain per game rules"
metrics:
  duration: "10 minutes"
  completed: "2026-02-02"
---

# Phase 23 Plan 03: TEMP_END_TURN Phase Summary

TEMP_END_TURN phase and phase transition updates for complete end-of-turn bookkeeping.

## Completed Tasks

| Task | Name | Commit | Key Changes |
|------|------|--------|-------------|
| 1 | Update CLOSING to transition to INCOME | 1b55da7 | Removed turn increment and roundtrip clear from _transition_to_income |
| 2 | Create TEMP_END_TURN phase files | 1f9efec | Created temp_end_turn.pyx and temp_end_turn.pxd |
| 3 | Move roundtrip clear to INVEST | 1a93aab | Added roundtrip clearing before WRAP_UP transition |
| 4 | Update phases/__init__ exports | 0929448 | Added temp_end_turn to package exports |
| 5 | Update driver for TEMP_END_TURN | 746eb07 | Added phase dispatch with sentinel -104 |
| 6 | Update tests | d4409f0 | Fixed 49 tests expecting INVEST instead of INCOME |

## Implementation Details

### TEMP_END_TURN Phase Handler (phases/temp_end_turn.pyx)
```cython
cdef int apply_temp_end_turn(GameState state) noexcept:
    cdef int current_turn = turn_module.TURN.get_turn_number(state)
    turn_module.TURN.set_turn_number(state, current_turn + 1)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)
    return 0
```

### Phase Transition Chain
Complete flow: INVEST -> WRAP_UP -> ACQUISITION -> CLOSING -> INCOME -> TEMP_END_TURN -> INVEST

### Key Changes
- CLOSING no longer increments turn number (moved to TEMP_END_TURN)
- CLOSING no longer clears roundtrip tracking (moved to INVEST before WRAP_UP)
- Roundtrip clearing happens at end of INVEST phase when all players pass
- TEMP_END_TURN is a non-player phase with 0 valid actions

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] NUM_PHASES constant was 11 but PHASE_TEMP_END_TURN is value 11**
- **Found during:** Task 5 verification
- **Issue:** set_phase() bounds check `phase < NUM_PHASES` rejected phase 11
- **Fix:** Updated NUM_PHASES from 11 to 12 in core/data.pxd (done by 23-02 parallel task)
- **Commit:** Incorporated in 23-02 commits

**2. [Rule 2 - Missing Critical] Tests expected INVEST after CLOSING**
- **Found during:** Final verification
- **Issue:** 49 tests were written before INCOME phase, expected CLOSING->INVEST
- **Fix:** Updated tests to expect INCOME (phase 5) after CLOSING transitions
- **Files modified:** test_closing.py, test_wrap_up.py, test_integration.py
- **Commit:** d4409f0

## Verification Results

- Build: `python3 setup.py build_ext --inplace` - SUCCESS
- Tests: 340/340 PASSED
- Phase chain verified: INVEST -> WRAP_UP -> ACQUISITION -> CLOSING -> INCOME -> TEMP_END_TURN -> INVEST
- Turn increment confirmed in TEMP_END_TURN phase
- Roundtrip clearing confirmed in INVEST phase (before WRAP_UP)

## Technical Notes

### Non-Player Phase Pattern
TEMP_END_TURN follows the same pattern as WRAP_UP and INCOME:
- 0 valid actions (deterministic execution)
- Auto-applied by driver when reached
- Uses sentinel value (-104) in history tracking

### Temporary Nature
TEMP_END_TURN is a temporary consolidation phase. Once DIVIDENDS, END_CARD, ISSUE_SHARES, and IPO phases are implemented, the turn increment logic should move to the appropriate phase per game rules.

## Files Modified

1. **phases/closing.pyx** - Updated _transition_to_income to transition to PHASE_INCOME
2. **phases/temp_end_turn.pyx** - New TEMP_END_TURN phase handler
3. **phases/temp_end_turn.pxd** - cdef declaration for apply_temp_end_turn
4. **phases/invest.pyx** - Added roundtrip clearing before WRAP_UP transition
5. **phases/__init__.pyx** - Added temp_end_turn import
6. **phases/__init__.pxd** - Added temp_end_turn cimport
7. **core/driver.pyx** - Added TEMP_END_TURN phase dispatch
8. **tests/** - Updated 3 test files for new phase flow

## Next Phase Readiness

Ready for Phase 23 completion. All phases in the integration wave are complete:
- 23-01: Bankruptcy refactor (complete)
- 23-02: INCOME phase (complete)
- 23-03: TEMP_END_TURN phase (complete)
