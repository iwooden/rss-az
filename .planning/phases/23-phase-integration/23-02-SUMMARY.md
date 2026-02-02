---
# Execution metadata
phase: 23-phase-integration
plan: 02
completed: 2026-02-02
duration: ~9 minutes

# Dependency graph
requires:
  - "23-01 (Corp.go_bankrupt method)"
provides:
  - "INCOME phase handler"
  - "PHASE_TEMP_END_TURN constant"
  - "Full phase chain: CLOSING -> INCOME -> TEMP_END_TURN"
affects:
  - "23-03 (TEMP_END_TURN implementation - executed in parallel)"
  - "Tests expecting old phase flow"

# Tech tracking
tech-stack:
  patterns:
    - "Non-player phase pattern (0 actions, auto-executes)"
    - "Bankruptcy delegation pattern (phases call Corp.go_bankrupt)"
    - "Income calculation reuse (existing entity methods from Phase 22)"

# File tracking
key-files:
  created:
    - phases/income.pyx
    - phases/income.pxd
  modified:
    - core/data.pxd (added PHASE_TEMP_END_TURN=11)
    - phases/__init__.pyx
    - phases/__init__.pxd
    - core/driver.pyx

# Decisions
decisions:
  - id: INCOME-01
    choice: "Apply income per-entity with immediate bankruptcy check"
    reason: "Per CONTEXT.md: bankruptcy check immediately after each corp's income"
  - id: INCOME-02
    choice: "Player cash assertion (not exception)"
    reason: "Should never occur if CLOSING phase works correctly - assertion catches bugs"

# Metrics
metrics:
  tasks: 4/4
  tests-affected: 18 (expected - tests written before INCOME phase existed)
  commits: 4
---

# Phase 23 Plan 02: INCOME Phase Handler Summary

**One-liner:** INCOME phase applies entity income with per-corp bankruptcy, transitioning to TEMP_END_TURN

## What Was Built

Created the INCOME phase handler that applies income to all game entities:

### Core Implementation

**phases/income.pyx:**
- `_apply_income_to_corps()`: Iterates corps 0-7, calls calculate_income/apply_income, checks bankruptcy per-corp
- `_apply_income_to_fi()`: Applies FI income (+5 base bonus)
- `_apply_income_to_players()`: Applies player income with non-negative cash assertion
- `apply_income()`: Main handler - calls all helpers, transitions to PHASE_TEMP_END_TURN

### Integration Points

**core/data.pxd:**
- Added `PHASE_TEMP_END_TURN = 11` to GamePhases enum
- NUM_PHASES updated to 12 (blocking fix for phase transition)

**core/driver.pyx:**
- Added PHASE_INCOME to non-player phase check (returns True)
- Added ACTION_INCOME_SENTINEL = -103
- Added apply_income call in execute_non_player_phase

**phases/__init__:**
- Exports income module via cimport and import

## Key Design Decisions

1. **Immediate bankruptcy per-corp**: After each corp's income application, check if cash < 0 and call go_bankrupt(). Multiple corps can go bankrupt in same INCOME phase.

2. **Player cash assertion**: Added assertion that player cash >= 0 after income. This should never fail if CLOSING phase correctly handled mandatory closes.

3. **Reuse existing methods**: Used calculate_income() and apply_income() from Phase 22 entities - no duplication of income calculation logic.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] NUM_PHASES update**

- **Found during:** Task 2 verification (tests failing with infinite loop)
- **Issue:** NUM_PHASES = 11 but PHASE_TEMP_END_TURN = 11, so set_phase rejected value >= NUM_PHASES
- **Fix:** NUM_PHASES updated to 12 (externally, verified present)
- **Files modified:** core/data.pxd
- **Commit:** (included in parallel 23-03 execution)

## Test Impact

18 tests fail with expected behavioral changes:
- FI cash tests: FI now gets +5 income bonus in INCOME phase
- Turn increment tests: Turn now increments in TEMP_END_TURN, not CLOSING
- Phase transition tests: Full chain now includes INCOME and TEMP_END_TURN

These are expected changes - tests were written before INCOME phase existed.

## Phase Chain Verification

Manual verification confirmed correct flow:
```
WRAP_UP (phase 2) -> ACQUISITION (phase 3) -> CLOSING (phase 4)
-> INCOME (phase 5) -> TEMP_END_TURN (phase 11) -> INVEST (phase 0)
```

Turn number increments in TEMP_END_TURN as designed.

## Commits

| Hash | Message |
|------|---------|
| 4633ae5 | feat(23-02): add PHASE_TEMP_END_TURN to GamePhases enum |
| 02cfbc8 | feat(23-02): create INCOME phase handler |
| 63387fb | feat(23-02): export income module from phases package |
| 78306db | feat(23-02): add INCOME phase dispatch to driver |

## Next Steps

Plan 23-03 (executed in parallel):
- Creates TEMP_END_TURN phase handler
- Moves turn increment from CLOSING to TEMP_END_TURN
- Updates tests for new phase flow
