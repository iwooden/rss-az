---
phase: 20-nogil-mask-optimization
plan: 01
subsystem: game-state
tags: [cython, nogil, optimization, low-level]

# Dependency graph
requires:
  - phase: 15.1-encoding-helpers
    provides: established pattern for nogil accessors with PlayerOffsets struct
provides:
  - CorpOffsets struct with 10 fields for low-level nogil corp state access
  - TurnOffsets struct with 7 fields for low-level nogil turn state access
  - 6 nogil corp accessor functions (is_corp_active, get_corp_cash, get_corp_bank_shares, get_corp_unissued_shares, get_corp_issued_shares, is_corp_in_receivership)
  - 7 nogil turn accessor functions with _nogil suffix (acquisition, dividend, issue, IPO, closing state)
affects: [20-02-refactor-mask-functions, mask-optimization, nogil-performance]

# Tech tracking
tech-stack:
  added: []
  patterns: [low-level nogil accessor pattern extended to corp and turn entities]

key-files:
  created: []
  modified: [entities/corp.pyx, entities/corp.pxd, entities/turn.pyx, entities/turn.pxd]

key-decisions:
  - "Use inline one-hot scanning loops in turn accessors (not encoding helpers) since they operate on turn pointer not data pointer"
  - "Omit price_index, owned_companies, acquisition_companies from CorpOffsets struct as not needed by mask functions"
  - "Use _nogil suffix for turn accessor functions to distinguish from cpdef class methods"

patterns-established:
  - "TurnOffsets computation depends on num_players for dynamic sizing"
  - "Inline one-hot scanning for turn state fields"
  - "DEF constants (NUM_CORPS=8, NUM_COMPANIES=36) for array bounds"

# Metrics
duration: 2min
completed: 2026-01-28
---

# Phase 20 Plan 01: nogil Mask Optimization Summary

**Low-level nogil accessors for corp and turn state following PlayerOffsets pattern from Phase 15.1**

## Performance

- **Duration:** 2 min
- **Started:** 2026-01-28T22:25:18Z
- **Completed:** 2026-01-28T22:27:36Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added CorpOffsets struct with 10 fields and 6 nogil accessor functions for corporation state
- Added TurnOffsets struct with 7 fields and 7 nogil accessor functions with _nogil suffix for turn state
- All 312 existing tests pass - new accessors are additive with no production usage yet

## Task Commits

Each task was committed atomically:

1. **Task 1: Add low-level nogil accessors to corp.pyx** - `75a0d6b` (feat)
2. **Task 2: Add low-level nogil accessors to turn.pyx** - `1c3f2e0` (feat)
3. **Task 3: Verify tests pass** - `0f723f4` (test)

_Note: All commits include both .pyx implementation and .pxd declarations_

## Files Created/Modified

- `entities/corp.pyx` - Added CorpOffsets struct and 6 inline nogil accessor functions
- `entities/corp.pxd` - Declared CorpOffsets struct and accessor function signatures
- `entities/turn.pyx` - Added TurnOffsets struct, get_turn_offsets(num_players) function, and 7 inline nogil accessor functions with _nogil suffix
- `entities/turn.pxd` - Declared TurnOffsets struct and accessor function signatures

## Decisions Made

1. **Inline one-hot scanning for turn accessors** - Turn accessors use inline loops rather than encoding.pyx helpers because they operate on turn pointer (not data pointer) and the encoding helpers expect data pointer
2. **Omit unused fields from CorpOffsets** - Excluded price_index (uses hidden compact storage), owned_companies, and acquisition_companies (not accessed by mask functions)
3. **_nogil suffix for turn accessors** - Used _nogil suffix (e.g., get_acq_active_corp_nogil) to distinguish low-level nogil functions from existing cpdef class methods

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - build succeeded on first attempt, all tests passed, pattern from Phase 15.1 transferred cleanly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for Phase 20-02: Refactor mask functions to use low-level nogil accessors.

The low-level accessors are in place and tested. Next phase will refactor action mask functions to:
1. Compute offsets once (not per-access)
2. Use raw float* pointers with nogil accessors
3. Remove GIL dependencies from mask functions
4. Improve mask generation performance

---
*Phase: 20-nogil-mask-optimization*
*Completed: 2026-01-28*
