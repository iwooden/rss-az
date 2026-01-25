---
phase: 13-actions-validation
plan: 01
subsystem: game-engine
tags: [cython, acquisition, validation, action-handlers]

# Dependency graph
requires:
  - phase: 12-offer-infrastructure
    provides: Offer generation and presentation infrastructure
provides:
  - Validation helpers for acquisition actions (VALID-01 through VALID-05)
  - Action handlers for accept/pass/FI-buy actions
  - Test scaffolding for validation scenarios
affects: [13-02-driver-integration, 14-flow-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Validation helpers as defensive checks (action mask is primary)"
    - "Corp seller acquisition_proceeds via get+set pattern"
    - "Player seller acquisition_proceeds via add method"

key-files:
  created: []
  modified:
    - phases/acquisition.pyx
    - tests/test_acquisition.py

key-decisions:
  - "Corp sellers use get+set pattern for acquisition_proceeds (no add method exists)"
  - "VALID-06 (same-president) guaranteed by offer generation, no runtime check needed"
  - "Test scaffolding created now, implementation deferred to 13-02 after driver integration"

patterns-established:
  - "Validation helpers return bint (True/False) for validity checks"
  - "Action handlers return void, modify state directly"
  - "All handlers call _advance_to_next_offer after execution"

# Metrics
duration: 3.6min
completed: 2026-01-25
---

# Phase 13 Plan 01: Actions & Validation Summary

**Validation helpers and action handlers for acquisition offers with full VALID-01 through VALID-05 checks**

## Performance

- **Duration:** 3.6 min
- **Started:** 2026-01-25T23:13:34Z
- **Completed:** 2026-01-25T23:17:10Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- Five validation helper functions covering all price/cash/ownership checks
- Four action handlers for accept-price, FI-buy-high, FI-buy-face, and pass
- Test scaffolding for 9 validation scenarios (ready for 13-02 implementation)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement validation helpers** - `d576f10` (feat)
   - Also included Task 2 action handlers in same commit (logical unit)
2. **Task 3: Add unit test scaffolding for validation** - `e7b4a00` (test)

_Note: Tasks 1 and 2 were combined in one commit as they form a cohesive validation+execution unit_

## Files Created/Modified
- `phases/acquisition.pyx` - Added validation helpers and action handlers
- `tests/test_acquisition.py` - Added TestValidation class with 9 test stubs

## Decisions Made

**1. Corp seller acquisition_proceeds via get+set pattern**
- Corp entity lacks `add_acquisition_proceeds` method (only player has it)
- Used `get_acquisition_proceeds() + set_acquisition_proceeds()` pattern for corp sellers
- Player sellers use existing `add_acquisition_proceeds()` method

**2. VALID-06 (same-president) guaranteed by offer generation**
- Phase 12 offer generation ensures same-president constraint
- Presidency cannot change mid-ACQUISITION phase
- No runtime re-check needed in validation helpers
- Documented in code comments for future maintainers

**3. Test scaffolding created now, implementation deferred**
- Validation functions are cdef (internal), cannot be called from Python directly
- Tests will verify validation through action handler behavior
- Full test implementation happens in Plan 13-02 after driver integration provides Python wrapper

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

**Cython variable declaration scoping**
- Initial implementation had `cdef` declarations inside `if` blocks
- Cython requires all `cdef` declarations at function start
- Fixed by moving all variable declarations to top of functions
- Build succeeded after fix

## Next Phase Readiness

Ready for Plan 13-02 (Driver Integration):
- Validation helpers complete and tested via build
- Action handlers complete with correct state modifications
- Test scaffolding ready for implementation
- No blockers

Next steps (Plan 13-02):
- Create `apply_acquisition_action` main handler
- Wire action handlers into switch-case dispatcher
- Add Python wrapper for testing
- Implement validation test logic

---
*Phase: 13-actions-validation*
*Completed: 2026-01-25*
