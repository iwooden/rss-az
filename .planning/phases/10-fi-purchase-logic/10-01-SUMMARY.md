---
phase: 10-fi-purchase-logic
plan: 01
subsystem: game-logic
tags: [cython, foreign-investor, company-availability, wrap-up-phase]

# Dependency graph
requires:
  - phase: 09-wrap-up-core-logic
    provides: WRAP_UP phase handler with player reordering and phase transitions
provides:
  - FI purchase loop with cheapest-first selection in face value order
  - Availability transition converting revealed companies to auction
  - Complete WRAP_UP phase integration with FI purchases and company state management
affects: [11-test-updates, future-game-loop-phases]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "While-loop with re-query pattern for FI purchases (no snapshotting)"
    - "Company iteration in ascending face value order (company_id 0-35)"
    - "Card draw and availability state management after each purchase"

key-files:
  created: []
  modified: [phases/wrap_up.pyx]

key-decisions:
  - "FI purchase loop uses while-loop with re-query pattern (no snapshotting) to handle dynamic availability changes"
  - "Purchase iteration in ascending company_id order (0-35) guarantees cheapest-first selection due to face value ordering"
  - "Availability transition occurs after all FI purchases complete (not incrementally)"

patterns-established:
  - "Entity interface usage: company.is_for_auction, transfer_to_fi, set_revealed, move_to_auction"
  - "Entity interface usage: fi.get_cash, add_cash"
  - "Entity interface usage: deck.draw"
  - "Deterministic phase handler pattern: helper functions + main handler integration"

# Metrics
duration: 2min
completed: 2026-01-23
---

# Phase 10 Plan 01: FI Purchase Logic Summary

**FI automatically purchases cheapest available companies at face value in WRAP_UP, then all revealed companies become available for next INVEST round**

## Performance

- **Duration:** 2 min
- **Started:** 2026-01-23T23:05:58Z
- **Completed:** 2026-01-23T23:07:39Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- FI purchase loop implemented with while-loop re-query pattern (no snapshotting)
- Purchase order guaranteed cheapest-first via ascending company_id iteration (0-35)
- Card draw after each purchase marks replacement as revealed (unavailable)
- Availability transition converts all revealed companies to auction after FI purchases
- Complete WRAP_UP phase integration with FI purchases and availability management

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement FI purchase loop** - `9c1f5af` (feat)
2. **Task 2: Add availability transition and integrate into apply_wrap_up** - `cd2f5bf` (feat)

## Files Created/Modified
- `phases/wrap_up.pyx` - Added FI purchase loop, availability transition, and integration into apply_wrap_up

## Decisions Made

**1. While-loop with re-query pattern for FI purchases**
- Rationale: Handles dynamic availability changes during purchases without snapshotting state
- Pattern: `while True: company_id = find(...); if company_id < 0: break; purchase(...)`

**2. Ascending company_id iteration (0-35) for cheapest-first selection**
- Rationale: Companies are ordered by ascending face value (0-35), so first affordable = cheapest
- Implementation: `for company_id in range(GameConstants.NUM_COMPANIES)`

**3. Availability transition after all FI purchases complete**
- Rationale: All revealed companies become available at once for next INVEST round
- Implementation: `_process_fi_purchases(state)` then `_make_all_revealed_available(state)`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

**Ready for Phase 11 (Test Updates):**
- WRAP_UP phase fully implemented with FI purchases and availability transition
- 9 test failures in test_invest.py are expected (covered in Phase 11)
- Tests need updates to verify WRAP_UP → ACQUISITION → INVEST flow
- Tests need verification of sentinel action history entries for non-player phases

**No blockers.**

---
*Phase: 10-fi-purchase-logic*
*Completed: 2026-01-23*
