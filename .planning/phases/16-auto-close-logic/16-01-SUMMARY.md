---
phase: 16-auto-close-logic
plan: 01
subsystem: game-logic
tags: [cython, closing-phase, auto-close, receivership, fi, junkyard-scrappers, vintage-machinery]

# Dependency graph
requires:
  - phase: 15-auto-apply-acq
    provides: Acquisition phase completion, company transfer patterns
  - phase: 11-setup-acquisition
    provides: Corp ownership tracking patterns
  - phase: 10-fi-purchases
    provides: FI ownership patterns
provides:
  - Auto-close logic for FI and receivership corporations
  - Junkyard Scrappers bonus application on company closure
  - Vintage Machinery CoO reduction in closing threshold checks
  - CLOSING phase handler with apply_closing_auto entry point
affects: [17-closing-offers, 18-close-transitions]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Non-player deterministic phase pattern for CLOSING auto-close"
    - "Two-pass closing: identify then close to avoid iterator invalidation"
    - "Protected company pattern: highest face value immune to auto-close"
    - "Corp-specific special effect handling (JS bonus, VM reduction)"

key-files:
  created:
    - phases/closing.pxd
    - phases/closing.pyx
  modified: []

key-decisions:
  - "Use two-pass closing (identify, then close) to avoid state mutation during iteration"
  - "FI closes only negative income companies (< 0, not <= 0)"
  - "Receivership uses CoO value from get_cost_of_ownership, not CoO level"
  - "Highest face value company protected per receivership corp"
  - "Vintage Machinery $10 CoO reduction applied before threshold check"
  - "Junkyard Scrappers receives 2x printed income for all closures"

patterns-established:
  - "Phase handler pattern: cdef int apply_X(GameState) noexcept returning 0 for deterministic phases"
  - "Helper function pattern: _close_company handles shared cleanup logic"
  - "Corp special effect pattern: check corp_id for special behaviors (JS, VM)"

# Metrics
duration: 2min
completed: 2026-01-27
---

# Phase 16 Plan 01: Auto-Close Logic Summary

**CLOSING phase auto-close handler with FI negative income closure, receivership CoO thresholds, and Junkyard Scrappers/Vintage Machinery special effects**

## Performance

- **Duration:** 2 min
- **Started:** 2026-01-27T01:47:12Z
- **Completed:** 2026-01-27T01:49:34Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created CLOSING phase module with auto-close logic for FI and receivership corporations
- Implemented Junkyard Scrappers bonus (2x printed income) on every company closure
- Implemented Vintage Machinery CoO reduction ($10) in receivership auto-close checks
- Protected highest face value company from auto-close in each receivership corp
- Built on existing company.remove_from_game() pattern for clean closure handling

## Task Commits

Each task was committed atomically:

1. **Task 1: Create closing.pxd header** - `bde6ea5` (feat)
2. **Task 2: Create closing.pyx implementation** - `f6958f5` (feat)

## Files Created/Modified
- `phases/closing.pxd` - Declares apply_closing_auto entry point
- `phases/closing.pyx` - CLOSING phase handler with _close_company, _process_fi_auto_close, _process_receivership_auto_close helpers

## Decisions Made

**Two-pass closing pattern:** Identify all companies to close in first pass, close them in second pass. Prevents state mutation during iteration which could invalidate ownership checks.

**FI negative income threshold:** Close only when `income - CoO < 0` (strictly negative), not `<= 0`. Zero-income companies remain in FI ownership.

**CoO value vs level:** Use `get_cost_of_ownership(coo_level, stars)` to get actual dollar value, not CoO level integer. Red threshold is $4, orange is $7.

**Protected company selection:** For each receivership corp, find highest face value company and exclude it from auto-close. Uses single iteration to find max, then second iteration to close eligible companies.

**Special effect handling:** Check corp_id to apply special behaviors:
- Junkyard Scrappers (corp_id 0): Add 2x printed income on all closures
- Vintage Machinery (corp_id 6): Subtract $10 from CoO value before threshold check

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for Phase 17 (Closing Offers). The auto-close logic is complete and builds successfully.

**Blocker noted in STATE.md:** `_transition_to_closing` in acquisition.pyx (line ~970) currently transitions to INVEST as workaround. Will need update to transition to CLOSING once Phase 16-17 integration completes.

**Phase integration:** This is a non-player deterministic phase. Phase 17 will handle:
- Player/corp offer-based closing (voluntary closing)
- Phase transition from ACQUISITION → CLOSING
- Phase transition from CLOSING → next phase (likely INCOME or back to INVEST)

---
*Phase: 16-auto-close-logic*
*Completed: 2026-01-27*
