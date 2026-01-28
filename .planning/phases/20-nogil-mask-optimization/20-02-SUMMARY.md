---
phase: 20-nogil-mask-optimization
plan: 02
subsystem: game-state
tags: [cython, nogil, optimization, low-level, refactoring]

# Dependency graph
requires:
  - phase: 20-01
    provides: Low-level nogil accessors (CorpOffsets, TurnOffsets) for corp and turn state
provides:
  - All 7 mask functions refactored to use low-level nogil accessors
  - Removed cpdef method calls from mask functions
  - Fixed off-by-one bug in 20-01 turn offset calculation
affects: [20-03-mark-nogil, mask-generation, performance-optimization]

# Tech tracking
tech-stack:
  added: []
  patterns: [mask refactoring pattern using low-level accessors]

key-files:
  created: []
  modified: [core/actions.pyx, entities/turn.pyx]

key-decisions:
  - "Keep state.get_player_cash and state.is_market_space_available in mask functions (don't prevent nogil)"
  - "Keep _fill_bid_mask unchanged (uses hidden compact storage, already efficient)"
  - "Fixed dividend_impact offset bug from 20-01 (26 -> 25) discovered during testing"

patterns-established:
  - "Mask function refactoring pattern: compute offsets once, use raw pointer accessors"

# Metrics
duration: 8min
completed: 2026-01-28
---

# Phase 20 Plan 02: nogil Mask Optimization Summary

**All 7 mask functions refactored to use low-level nogil accessors, removing cpdef dependencies and preparing for nogil marking**

## Performance

- **Duration:** 8 min
- **Started:** 2026-01-28T22:30:51Z
- **Completed:** 2026-01-28T22:38:47Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- Refactored 6 of 7 mask functions to use low-level nogil accessors (_fill_bid_mask already efficient)
- Removed all cpdef method calls from mask functions (state.get_corp_*, state.get_acq_*, etc.)
- Discovered and fixed off-by-one bug in 20-01 turn offset calculation (dividend_impact: 26 -> 25)
- All 312 existing tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Add imports for new low-level accessors** - `4048ec8` (refactor)
2. **Task 2: Refactor mask functions to use low-level accessors** - `094f32f` (refactor)
3. **Task 3: Fix 20-01 offset bug** - `48fbe36` (fix)

_Note: Task 3 was an auto-fix for a bug discovered during testing (deviation Rule 1)_

## Files Created/Modified

- `core/actions.pyx` - Refactored 6 mask functions to use low-level accessors (CorpOffsets/TurnOffsets)
- `entities/turn.pyx` - Fixed dividend_impact offset bug (26 -> 25)

## Decisions Made

1. **Keep efficient methods unchanged** - `_fill_bid_mask` already uses hidden compact storage (efficient), no changes needed
2. **Selective refactoring** - Kept `state.get_player_cash` and `state.is_market_space_available` calls as they don't prevent nogil (called before critical loops)
3. **Fixed offset calculation** - Discovered hardcoded 26 for dividend_impact should be 25 (MAX_DIVIDEND constant) in 20-01's `get_turn_offsets` function

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed dividend_impact offset calculation in get_turn_offsets**
- **Found during:** Task 2 execution - tests failed with acquisition mask returning wrong corp_id
- **Issue:** Plan 20-01's `get_turn_offsets` hardcoded dividend_impact size as 26, but MAX_DIVIDEND constant is 25. This caused all subsequent offsets to be off by one, making nogil accessors read from wrong memory locations.
- **Root cause:** Acquisition mask returned corp_id=7 instead of corp_id=0 because it scanned offset 179 (wrong) instead of 178 (correct)
- **Fix:** Changed `offset += 26` to `offset += 25` in `get_turn_offsets` function and updated comment
- **Files modified:** entities/turn.pyx
- **Verification:** All 312 tests pass, acquisition mask now returns correct corp_id
- **Committed in:** `48fbe36` (separate fix commit for 20-01 bug)

---

**Total deviations:** 1 auto-fixed (1 bug from prior plan)
**Impact on plan:** Bug fix was essential for correctness. Discovered through testing, not a plan deviation but a prior plan bug exposed by this plan's refactoring.

## Issues Encountered

- **Off-by-one offset bug:** Initial refactoring caused all 4 acquisition tests to fail. Root cause was Plan 20-01's `get_turn_offsets` using hardcoded 26 instead of 25 for dividend_impact size.
- **Debug process:** Added temporary Python wrapper to compare old cpdef accessors vs new nogil accessors, discovered offset mismatch (179 vs 178), traced back to dividend_impact size error
- **Resolution:** Fixed in separate commit `48fbe36` to clearly attribute bug to 20-01, not 20-02

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for Phase 20-03: Mark mask functions as nogil.

All mask functions now use only nogil-compatible accessors:
- `_fill_invest_mask`: CorpOffsets accessors
- `_fill_acquisition_mask`: TurnOffsets + CorpOffsets accessors
- `_fill_closing_mask`: TurnOffsets accessors
- `_fill_dividends_mask`: TurnOffsets + CorpOffsets accessors
- `_fill_issue_mask`: TurnOffsets + CorpOffsets accessors
- `_fill_ipo_mask`: TurnOffsets + CorpOffsets accessors
- `_fill_bid_mask`: Already efficient (no changes)

Next phase will add `nogil` keyword to function signatures and verify GIL-free execution.

---
*Phase: 20-nogil-mask-optimization*
*Completed: 2026-01-28*
