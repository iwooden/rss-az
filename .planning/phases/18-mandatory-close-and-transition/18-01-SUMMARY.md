---
phase: 18-mandatory-close-and-transition
plan: 01
subsystem: game-rules
tags: [cython, closing-phase, mandatory-close, player-income, bankruptcy-prevention]

# Dependency graph
requires:
  - phase: 17-offer-based-close-flow
    provides: _close_company helper, offer generation and presentation pattern
  - phase: 16-fi-receivership-auto-close
    provides: Auto-close pattern, two-pass close safety
provides:
  - Player.get_income() method for income calculation
  - _process_mandatory_close() for player bankruptcy prevention
  - _close_player_company() helper for player-owned company closes
  - Complete CLOSING phase implementation ready for INCOME phase
affects: [19-integration-and-bugfixes, income-phase, dividends-phase]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Player income calculation pattern - sum of adjusted income from owned privates"
    - "Mandatory close pattern - iterative cheapest-first until income + cash >= 0"
    - "Phase-end protection pattern - prevent bankruptcy before phase transition"

key-files:
  created: []
  modified:
    - entities/player.pyx
    - entities/player.pxd
    - phases/closing.pyx

key-decisions:
  - "CoO fixed at phase start - no re-evaluation during mandatory close loop"
  - "Cheapest (lowest face value) negative-income company closed first"
  - "Players can end with zero companies - no minimum retention rule"
  - "Junkyard Scrappers bonus applies to mandatory closes"

patterns-established:
  - "Income calculation pattern: Player method returns sum of adjusted income (base - CoO) from owned privates only"
  - "Mandatory close pattern: Iterate players by ID, close cheapest negative-income company until income + cash >= 0"
  - "Phase-end protection: Mandatory close executes before _transition_to_income() to prevent player bankruptcy in next phase"

# Metrics
duration: 2m 39s
completed: 2026-01-27
---

# Phase 18 Plan 01: Mandatory Close and Transition Summary

**Player income calculation and mandatory close system to prevent bankruptcy in INCOME phase by auto-closing cheapest negative-income private companies**

## Performance

- **Duration:** 2m 39s
- **Started:** 2026-01-27T21:39:25Z
- **Completed:** 2026-01-27T21:42:04Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Player.get_income() method calculates total adjusted income from owned private companies
- Mandatory close system closes cheapest negative-income companies when player income + cash < 0
- Integration into phase transition ensures bankruptcy prevention before INCOME phase
- Junkyard Scrappers bonus applied to mandatory closes
- All existing tests pass with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Add get_income() method to Player entity** - `0299cd3` (feat)
2. **Task 2: Add mandatory close logic to closing.pyx** - `72f851c` (feat)

## Files Created/Modified
- `entities/player.pyx` - Added get_income() method to calculate sum of adjusted income from owned private companies
- `entities/player.pxd` - Added get_income() method declaration
- `phases/closing.pyx` - Added _close_player_company() and _process_mandatory_close() functions, integrated before transition

## Decisions Made

**1. CoO fixed at phase start**
- Per CONTEXT.md: CoO level captured once at start of mandatory close, not re-evaluated after each company close
- Simplifies logic and matches game phase boundary semantics

**2. Cheapest negative-income company closed first**
- Per CLO-15 requirement: Close lowest face value company first
- Minimizes asset loss for players while preventing bankruptcy

**3. Players can end with zero companies**
- Per CONTEXT.md: No minimum retention rule for players
- Unlike corps (must keep 1 company), players can lose all companies to mandatory close

**4. Junkyard Scrappers bonus applies**
- Per CONTEXT.md: All closes (voluntary and mandatory) trigger JS bonus
- Maintains consistent close mechanics across phase

**5. Player income = only private companies**
- Excludes corp subsidiaries from player income calculation
- Corp income handled separately by corp entities

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed as specified.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 19 (Integration and Bugfixes):**
- CLOSING phase fully implemented (auto-close + offer-based + mandatory close)
- Player bankruptcy prevention complete
- All existing tests passing
- Ready for integration testing and bugfixes

**Blockers:**
- INCOME phase not yet implemented (temporary transition to INVEST documented in code)

**Notes for future phases:**
- INCOME phase will use Player.get_income() to calculate income payments
- Mandatory close ensures players never enter INCOME phase with negative total income
- _transition_to_income() currently goes to INVEST, will need update when INCOME implemented

---
*Phase: 18-mandatory-close-and-transition*
*Completed: 2026-01-27*
