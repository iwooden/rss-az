---
phase: 18-mandatory-close-and-transition
plan: 02
subsystem: testing
tags: [cython, pytest, closing-phase, mandatory-close, player-income]

# Dependency graph
requires:
  - phase: 18-01
    provides: Player.get_income() method, mandatory close implementation
provides:
  - Test coverage for Player.get_income() calculation (CLO-14)
  - Test coverage for mandatory close triggering and behavior (CLO-14, CLO-15)
  - Test coverage for phase transition after mandatory close (CLO-16)
affects: [19-integration-and-bugfixes]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Mandatory close test pattern - set high CoO level, give negative-income companies, verify close order"
    - "Phase transition test pattern - call apply_closing_auto_py and verify phase change"

key-files:
  created:
    - tests/test_mandatory_close.py
  modified: []

key-decisions:
  - "Renamed test_closing.py to test_mandatory_close.py to avoid conflict with tests/phases/test_closing.py"
  - "Used CoO level 7 (max valid level) instead of 8 for negative income tests"
  - "Simplified phase transition tests to call apply_closing_auto_py directly rather than using driver loop"

patterns-established:
  - "Income test pattern: Set CoO level, give companies, verify sum of adjusted income"
  - "Mandatory close test pattern: Set negative total (income + cash < 0), verify cheapest closed first"
  - "Integration test pattern: Use apply_closing_auto_py and apply_closing_action_py for phase flow"

# Metrics
duration: 4m 52s
completed: 2026-01-27
---

# Phase 18 Plan 02: Mandatory Close Tests Summary

**Comprehensive test coverage for player income calculation, mandatory close triggering, close ordering, Junkyard Scrappers bonus, and phase transition to INVEST**

## Performance

- **Duration:** 4m 52s
- **Started:** 2026-01-27T21:45:39Z
- **Completed:** 2026-01-27T21:50:31Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments
- 5 tests covering Player.get_income() method with various scenarios
- 7 tests covering mandatory close logic (triggering, cheapest-first order, JS bonus)
- 2 tests covering phase transition to INVEST after mandatory close
- All 296 tests pass with no regressions
- Requirements CLO-14, CLO-15, CLO-16 fully tested

## Task Commits

Each task was committed atomically:

1. **Task 1: Add tests for Player.get_income() method** - `e6b83c8` (test)
2. **Task 2: Add tests for mandatory close logic** - `63a57fa` (test)
3. **Task 3: Add test for phase transition (CLO-16)** - `9a37671` (test)

**Refactor:** `e1a9e6f` (renamed test file to avoid conflict)

## Files Created/Modified
- `tests/test_mandatory_close.py` - Test coverage for CLO-14, CLO-15, CLO-16 requirements

## Decisions Made

**1. CoO level 7 instead of 8**
- Discovered COST_OF_OWNERSHIP array only has 7 levels (indices 0-6)
- Level 8 is invalid and returns 0 for all star tiers
- Updated tests to use level 7 (max valid level) for high-CoO scenarios

**2. Simplified phase transition tests**
- Initially attempted driver.step() loop pattern
- Discovered GameDriver doesn't have step() method
- Simplified to direct apply_closing_auto_py() calls matching existing test patterns

**3. Renamed test file**
- Initial name test_closing.py conflicted with tests/phases/test_closing.py
- Renamed to test_mandatory_close.py to avoid pytest collection error

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test expectations for CoO level 1**
- **Found during:** Task 1 (test_get_income_single_company)
- **Issue:** Tests assumed CoO at level 1 would be non-zero, but COST_OF_OWNERSHIP shows level 1 has $0 for all tiers
- **Fix:** Updated test expectations - company 0 at CoO 1 has income $1 - $0 = $1 (not $0)
- **Files modified:** tests/test_mandatory_close.py
- **Verification:** All 5 income tests pass
- **Committed in:** e6b83c8 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed invalid company ownership setup**
- **Found during:** Task 1 test execution
- **Issue:** Tests called COMPANIES[x].set_location() which doesn't exist
- **Fix:** Removed set_location calls - only PLAYERS[x].set_owns_company() needed per existing test patterns
- **Files modified:** tests/test_mandatory_close.py
- **Verification:** Tests pass without location setting
- **Committed in:** e6b83c8 (Task 1 commit)

**3. [Rule 1 - Bug] Fixed corp activation API**
- **Found during:** Task 1 test execution
- **Issue:** Tests called CORPS[x].activate() which doesn't exist
- **Fix:** Used set_active(state, True) and set_price_index() per existing test patterns
- **Files modified:** tests/test_mandatory_close.py
- **Verification:** test_get_income_excludes_corp_subsidiaries passes
- **Committed in:** e6b83c8 (Task 1 commit)

**4. [Rule 1 - Bug] Fixed CoO level 8 expectation**
- **Found during:** Task 2 test planning
- **Issue:** Plan specified CoO level 8, but max valid level is 7
- **Fix:** Updated all tests to use level 7 with corrected income calculations
- **Files modified:** tests/test_mandatory_close.py
- **Verification:** All 7 mandatory close tests pass
- **Committed in:** 63a57fa (Task 2 commit)

**5. [Rule 1 - Bug] Fixed phase transition test API**
- **Found during:** Task 3 test execution
- **Issue:** Tests tried to use driver.step() and ACTION_PASS from core.actions
- **Fix:** Used apply_closing_auto_py() and ACTION_PASS_PY from core.actions per existing patterns
- **Files modified:** tests/test_mandatory_close.py
- **Verification:** Both phase transition tests pass
- **Committed in:** 9a37671 (Task 3 commit)

**6. [Rule 3 - Blocking] Fixed test file name conflict**
- **Found during:** Full test suite execution
- **Issue:** test_closing.py conflicted with tests/phases/test_closing.py causing pytest collection error
- **Fix:** Renamed to test_mandatory_close.py
- **Files modified:** tests/test_closing.py → tests/test_mandatory_close.py
- **Verification:** Full test suite passes (296 tests)
- **Committed in:** e1a9e6f (refactor commit)

---

**Total deviations:** 6 auto-fixed (5 bugs, 1 blocking)
**Impact on plan:** All auto-fixes necessary for test correctness. No scope creep - all tests cover planned requirements.

## Issues Encountered

None - all issues resolved via deviation rules auto-fix.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 19 (Integration and Bugfixes):**
- CLOSING phase fully tested (income calculation, mandatory close, phase transition)
- All requirements CLO-14, CLO-15, CLO-16 have test coverage
- Full test suite passes with 296 tests
- No regressions introduced

**Testing patterns established:**
- Income calculation tests verify sum of adjusted income from owned privates
- Mandatory close tests verify cheapest-first ordering and bankruptcy prevention
- Phase transition tests verify CLOSING → INVEST flow after offers and mandatory close

**Notes for future phases:**
- CoO max level is 7 (not 8) - array has indices 0-6
- Company ownership setup only needs set_owns_company() - no set_location() needed
- Corp activation requires set_active() + set_price_index()
- Phase transition tests should use apply_*_auto_py() directly, not driver loop

---
*Phase: 18-mandatory-close-and-transition*
*Completed: 2026-01-27*
