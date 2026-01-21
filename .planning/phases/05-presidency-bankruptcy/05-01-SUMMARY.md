---
phase: 05-presidency-bankruptcy
plan: 01
subsystem: game-engine
tags: [cython, bankruptcy, corporation, share-price, game-state]

# Dependency graph
requires:
  - phase: 04-share-trading
    provides: Share buy/sell handlers with price movement
provides:
  - Corporation bankruptcy procedure triggered at price index 0
  - Complete corp reset: companies removed, shares returned, cash cleared, state reset
affects: [05-02-presidency-transfer, 05-03-receivership]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Bankruptcy inline execution pattern - handle immediately during sell, no deferral"
    - "Early return pattern - skip remaining sell steps after bankruptcy"

key-files:
  created: []
  modified:
    - phases/invest.pyx

key-decisions:
  - "Execute bankruptcy inline during sell handler (not deferred)"
  - "Clear president status for all players (only for bankrupt corp)"
  - "Early return after bankruptcy to skip presidency/receivership checks"

patterns-established:
  - "Bankruptcy cleanup pattern: companies → shares → cash → market → corp state"
  - "set_president_of(state, corp_id, False) only affects specified corp_id parameter"

# Metrics
duration: 2min 7sec
completed: 2026-01-21
---

# Phase 05 Plan 01: Corporation Bankruptcy Summary

**Bankruptcy procedure triggered when share price drops to index 0, executing complete corp reset inline during sell action**

## Performance

- **Duration:** 2min 7sec
- **Started:** 2026-01-21T19:40:47Z
- **Completed:** 2026-01-21T19:42:54Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Implemented complete bankruptcy procedure covering all INV-22 through INV-27 requirements
- Integrated bankruptcy check into sell handler with early return optimization
- All existing share trading tests pass without modification

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement bankruptcy procedure helper** - `8d455ff` (feat)
2. **Task 2: Integrate bankruptcy check into sell handler** - `dc170e8` (feat)

## Files Created/Modified
- `phases/invest.pyx` - Added `_execute_bankruptcy()` helper and integrated into `_handle_sell_share()`

## Decisions Made

**1. Execute bankruptcy inline during sell handler**
- Rationale: CONTEXT.md specifies "Execute bankruptcy immediately inline during sell handler (no deferral)"
- Implementation: Check `new_index == 0` after price movement, call bankruptcy, early return

**2. Clear president status for all players**
- Rationale: When corp goes bankrupt, no one should be president anymore
- Implementation: Loop through all players calling `set_president_of(state, corp_id, False)` for the bankrupt corp
- Note: This only affects presidency for the specific `corp_id` parameter, not other corps

**3. Early return after bankruptcy**
- Rationale: After bankruptcy, corp is gone - no need to check receivership or update round-trip tracking
- Implementation: Return immediately after bankruptcy cleanup, player advancement, and net worth update
- Benefit: Avoids unnecessary state checks on non-existent corp

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 05-02 (Presidency Transfer):**
- Bankruptcy clears all president status correctly
- Share transfer handlers in place for presidency checks
- Market price movement integrated

**Ready for Phase 05-03 (Receivership):**
- Bankruptcy clears receivership flag
- Corp state fully reset for future IPO

**Blockers/Concerns:** None

---
*Phase: 05-presidency-bankruptcy*
*Completed: 2026-01-21*
