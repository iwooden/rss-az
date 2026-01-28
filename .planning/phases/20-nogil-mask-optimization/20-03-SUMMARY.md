---
phase: 20-nogil-mask-optimization
plan: 03
subsystem: game-state
tags: [cython, nogil, optimization, parallelization, performance]

# Dependency graph
requires:
  - phase: 20-02
    provides: Mask functions refactored to use low-level nogil accessors
provides:
  - All 7 mask functions marked nogil (GIL-free execution ready)
  - All 8 functions (7 mask + 1 dispatch) have noexcept nogil signature
  - 5 inline nogil accessor helpers for state access
  - Performance baseline: 2.7M masks/sec
affects: [future-parallelization, alphazero-self-play, thread-safety]

# Tech tracking
tech-stack:
  added: []
  patterns: [inline nogil accessor pattern for state methods]

key-files:
  created: []
  modified: [core/actions.pyx]

key-decisions:
  - "Added 5 inline nogil accessor helpers to unblock GIL-requiring state methods"
  - "Imported TurnStateOffsets from core.state (not entities.turn.TurnOffsets)"
  - "Established 2.7M masks/sec performance baseline for future regression testing"

patterns-established:
  - "Inline nogil accessor pattern: cdef inline return_type func_nogil(...) noexcept nogil wraps state access"

# Metrics
duration: 5min
completed: 2026-01-28
---

# Phase 20 Plan 03: nogil Mask Optimization Summary

**All 8 mask functions marked nogil for GIL-free execution, enabling future thread-level parallelization for AlphaZero self-play**

## Performance

- **Duration:** 5 min
- **Started:** 2026-01-28T22:41:11Z
- **Completed:** 2026-01-28T22:46:11Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments

- All 7 mask generation functions marked `noexcept nogil`
- Dispatch function `_fill_mask_for_phase` marked `noexcept nogil`
- Created 5 inline nogil accessor helpers to enable GIL-free state access
- All 312 tests pass with nogil implementation
- Established performance baseline: 2.7M masks/sec (0.37µs per mask)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add nogil to all mask function signatures** - `ff85ce9` (feat)
2. **Task 2: Run full test suite** - (verification only, no commit)
3. **Task 3: Run benchmark and verify no regression** - (verification only, no commit)

## Files Created/Modified

- `core/actions.pyx` - Added nogil signatures to 8 functions, created 5 inline nogil accessors

## Decisions Made

1. **Added inline nogil accessor helpers** - Plan 20-02 incorrectly assumed `state.get_player_cash` and `state.is_market_space_available` wouldn't prevent nogil, but compiler proved they require GIL. Created 5 inline helpers: `get_player_cash_nogil`, `get_corp_price_index_nogil`, `get_auction_company_nogil`, `get_auction_price_nogil`, `is_market_space_available_nogil`

2. **Imported TurnStateOffsets from core.state** - Used `TurnStateOffsets` (from core.state) instead of `TurnOffsets` (from entities.turn) because auction_price field only exists in the former

3. **Established performance baseline** - Measured 2.7M masks/sec (0.37µs per mask) as baseline for future regression testing since no prior baseline existed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added 5 inline nogil accessor helpers**
- **Found during:** Task 1 execution - compilation failed when marking functions nogil
- **Issue:** Plan 20-02 summary stated "Keep state.get_player_cash and state.is_market_space_available in mask functions (don't prevent nogil)" but compiler proved this was incorrect - these cpdef methods DO require GIL
- **Root cause:** cpdef methods always acquire GIL for Python interoperability, even if called from Cython code
- **Fix:** Created 5 inline nogil helpers that directly access state data: `get_player_cash_nogil`, `get_corp_price_index_nogil`, `get_auction_company_nogil`, `get_auction_price_nogil`, `is_market_space_available_nogil`
- **Implementation:** Each helper accesses state._data directly or uses existing nogil accessors (PlayerOffsets, TurnStateOffsets)
- **Files modified:** core/actions.pyx (added helpers, updated all calls)
- **Verification:** Build succeeds, all 312 tests pass
- **Committed in:** `ff85ce9` (part of Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking issue)
**Impact on plan:** Auto-fix was essential to unblock nogil marking. Plan 20-02's assumption was incorrect; compiler verification exposed the issue. The helpers are minimal inline functions with zero overhead.

## Issues Encountered

- **Compilation errors on nogil marking:** Initial attempt to mark functions nogil failed because Plan 20-02 left some cpdef method calls in place. Solution: added inline nogil accessor helpers wrapping the state access.

- **TurnOffsets vs TurnStateOffsets confusion:** entities/turn defines `TurnOffsets` struct without `auction_price` field, but core/state defines `TurnStateOffsets` struct with the field. Resolved by importing from core.state.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 20 complete! nogil optimization complete:
- Phase 20-01: Low-level nogil accessors (CorpOffsets, TurnOffsets)
- Phase 20-02: Refactored mask functions to use nogil accessors
- Phase 20-03: Marked all mask functions nogil

**Impact:**
- Mask generation now runs without holding the GIL
- Future work can parallelize self-play across threads (each thread generates masks independently)
- Performance baseline established: 2.7M masks/sec

**Next milestone:** v6.0 - Remaining game phases (INCOME, DIVIDENDS, ISSUE_SHARES, IPO, END_GAME)

---
*Phase: 20-nogil-mask-optimization*
*Completed: 2026-01-28*
