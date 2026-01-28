---
phase: quick
plan: 005
subsystem: game-logic
tags: [cython, closing-phase, junkyard-scrappers, game-rules]

# Dependency graph
requires:
  - phase: 19-testing-and-integration
    provides: CLOSING phase implementation and test suite
provides:
  - Correct Junkyard Scrappers bonus logic (only when JS closes its own companies)
  - Updated test suite validating correct JS bonus behavior
affects: [closing-phase, game-rules, testing]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - phases/closing.pyx
    - tests/phases/test_closing.py

key-decisions:
  - "JS bonus only applies when owner_type == LOC_CORP (5) AND owner_id == 0 (JS)"
  - "Removed incorrect JS bonus from player-owned company closures"
  - "Removed incorrect JS bonus from mandatory close path"

patterns-established: []

# Metrics
duration: 3min
completed: 2026-01-28
---

# Quick Task 005: Fix JS Scrapping Bonus Bug Summary

**Corrected Junkyard Scrappers bonus to only apply when JS closes its own companies, not for any company closure**

## Performance

- **Duration:** 3 minutes (176 seconds)
- **Started:** 2026-01-28T17:32:00Z
- **Completed:** 2026-01-28T17:34:56Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- Fixed JS bonus logic in `_close_company()` to check owner_type and owner_id
- Removed incorrect JS bonus from `_close_player_company()` (player mandatory closes)
- Removed incorrect JS bonus from `_handle_close_accept()` OWNER_PLAYER block
- Fixed 8 existing tests to verify correct behavior (no bonus for FI, receivership, player closes)
- Added 2 new tests verifying JS gets bonus only when JS closes its own companies
- All 51 CLOSING phase tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix JS bonus logic in phases/closing.pyx** - `61ad246` (fix)
2. **Task 2: Fix and add JS bonus tests in test_closing.py** - `2b35acf` (test)
3. **Task 3: Verify all closing tests pass** - (verification only, no commit)

## Files Created/Modified
- `phases/closing.pyx` - Fixed JS bonus logic in three locations (_close_company, _close_player_company, _handle_close_accept)
- `tests/phases/test_closing.py` - Fixed 8 tests, added 2 new tests for correct JS bonus behavior

## Decisions Made

**JS bonus condition:** Changed from "always apply if JS is active" to "only apply when owner_type == 5 (LOC_CORP) AND owner_id == 0 (JS)". This ensures bonus only triggers when JS itself closes a company it owns.

**Removed from player paths:** Player-owned companies (private companies) can never be owned by JS (a corporation), so JS bonus logic was removed from player close paths entirely.

**Test strategy:** Rather than just fixing assertions, renamed tests to clearly document expected behavior (e.g., `test_js_no_bonus_on_fi_close`).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - straightforward bug fix with clear root cause and solution.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

CLOSING phase JS bonus logic now matches RULES.md specification. All 51 CLOSING tests pass. No blockers for future phases.

---
*Quick task: 005*
*Completed: 2026-01-28*
