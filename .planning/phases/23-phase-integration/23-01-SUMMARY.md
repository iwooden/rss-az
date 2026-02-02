---
phase: 23-phase-integration
plan: 01
subsystem: entities
tags: [cython, bankruptcy, refactoring, corporation]

# Dependency graph
requires:
  - phase: 22-income-application
    provides: Corp.calculate_income() and Corp.apply_income() methods
provides:
  - Corp.go_bankrupt() method for centralized bankruptcy handling
  - Single source of truth for bankruptcy logic
  - Reusable bankruptcy from any phase (INVEST, INCOME)
affects: [23-02-INCOME-phase, 23-03-transitions]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Bankruptcy delegation pattern - phases call Corp.go_bankrupt() instead of inline logic"

key-files:
  created: []
  modified:
    - entities/corp.pyx
    - entities/corp.pxd
    - phases/invest.pyx
    - core/state.pyx
    - core/state.pxd

key-decisions:
  - "Bankruptcy logic moved to Corp entity as go_bankrupt() method"
  - "Method is cpdef for cross-module accessibility"
  - "Incomplete bankrupt_corp() stub deleted from state.pyx"

patterns-established:
  - "Bankruptcy delegation pattern: phases call Corp.go_bankrupt(state) instead of inline bankruptcy logic"

# Metrics
duration: 5min
completed: 2026-02-02
---

# Phase 23 Plan 01: Bankruptcy Refactor Summary

**Bankruptcy logic consolidated to Corp.go_bankrupt() method for reuse by INVEST and INCOME phases**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-02-02T19:05:00Z
- **Completed:** 2026-02-02T19:10:24Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- Added Corp.go_bankrupt() method with full bankruptcy handling logic
- Removed 57 lines of inline _execute_bankruptcy() from invest.pyx
- Deleted incomplete bankrupt_corp() stub from state.pyx
- All 340 tests pass (bankruptcy behavior unchanged)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Corp.go_bankrupt() method** - `0d92658` (feat)
2. **Task 2: Update invest.pyx to call Corp.go_bankrupt()** - `090c9f6` (refactor)
3. **Task 3: Delete bankrupt_corp() stub from state.pyx** - `4f2c513` (chore)

## Files Created/Modified
- `entities/corp.pyx` - Added go_bankrupt() method with full bankruptcy logic
- `entities/corp.pxd` - Added go_bankrupt() declaration
- `phases/invest.pyx` - Replaced _execute_bankruptcy() with corp.go_bankrupt(state)
- `core/state.pyx` - Removed incomplete bankrupt_corp() stub
- `core/state.pxd` - Removed bankrupt_corp() declaration

## Decisions Made
- Used cpdef (not cdef) for go_bankrupt() to allow cross-module calls from phases
- Added player_module import to corp.pyx for clearing player shares during bankruptcy
- Removed unused get_corp_share_count import from invest.pyx after refactoring

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Initial build failed due to missing go_bankrupt() declaration in corp.pxd (fixed by adding declaration)
- Missing get_corp_share_count import in corp.pyx (fixed by adding to cimport)
- Task 3 build failed due to bankrupt_corp declaration in state.pxd (fixed by removing from both files)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Corp.go_bankrupt() is ready for INCOME phase to call when corp has negative income
- Phase 23-02 (INCOME phase integration) can now implement bankruptcy-on-negative-income
- All existing bankruptcy tests pass, confirming behavior unchanged

---
*Phase: 23-phase-integration*
*Completed: 2026-02-02*
