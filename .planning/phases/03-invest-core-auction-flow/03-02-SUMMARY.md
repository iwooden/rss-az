---
phase: 03-invest-core-auction-flow
plan: 02
subsystem: game-logic
tags: [cython, game-state, bid-phase, auction-resolution, turn-order]

# Dependency graph
requires:
  - phase: 03-01
    provides: INVEST phase auction start with state initialization
  - phase: 02-infrastructure-setup
    provides: GameDriver dispatch, entity handles, test infrastructure
provides:
  - BID_IN_AUCTION phase leave auction action with passed flag tracking
  - BID_IN_AUCTION phase raise bid action with price and high bidder updates
  - Auction resolution sequence (payment, transfer, deck draw, state cleanup)
  - Turn order advancement skipping passed bidders
  - Phase transition back to INVEST after auction completes
affects: [04-buy-sell-shares, wrap-up-phase, future-auction-mechanics]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Auction resolution pattern (payment → transfer → update → draw → cleanup → transition)
    - Active bidder counting pattern (iterate players checking passed flags)
    - Turn order advancement with skip logic (wrap around, check flags)

key-files:
  created: []
  modified:
    - phases/bid.pyx
    - core/state.pyx (infrastructure fix)

key-decisions:
  - "Count active bidders by iterating all players and checking has_player_passed_auction flag"
  - "Resolve auction when active bidder count reaches 1 (not 0)"
  - "Update winner net worth immediately after transfer (maintains invariant)"
  - "Draw new company and move to auction during resolution (maintains auction row size)"
  - "Fixed missing Company entity initialization in GameState.initialize_game (infrastructure bug)"

patterns-established:
  - "Auction resolution sequence: pay → transfer → update net worth → draw replacement → clear state → transition phase → set next player"
  - "Helper function pattern for turn order navigation: _find_player_at_position, _advance_to_next_bidder, _set_active_player_after"
  - "Active player tracking: always based on turn_order field, not player_id"

# Metrics
duration: 4min 57sec
completed: 2026-01-21
---

# Phase 03 Plan 02: BID_IN_AUCTION Phase Summary

**Full auction bidding with leave, raise bid, and resolution transferring company to winner and returning to INVEST phase**

## Performance

- **Duration:** 4min 57sec
- **Started:** 2026-01-21T01:14:18Z
- **Completed:** 2026-01-21T01:19:10Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- Leave auction action sets passed flag and resolves when one bidder remains
- Raise bid action updates auction price and high bidder, advances to next non-passed bidder
- Auction resolution: winner pays, receives company, net worth updated, new company drawn, state cleared
- Phase transitions back to INVEST with active player set to player after auction starter (not winner)

## Task Commits

Each task was committed atomically:

1. **Infrastructure Fix: Initialize COMPANIES entities** - `aeaeca4` (fix)
2. **Task 1: Implement BID_IN_AUCTION phase handler** - `b98b4a4` (feat)

## Files Created/Modified
- `phases/bid.pyx` - Full BID phase implementation with leave, raise bid, and resolution logic
- `core/state.pyx` - Added Company entity initialization in initialize_game (infrastructure fix)

## Decisions Made

**1. Active bidder counting approach**
- Count active bidders by iterating all players and checking has_player_passed_auction flag
- Rationale: Simple, correct, and matches the existing pattern for auction passed tracking
- Pattern: `for player_id in range(num_players): if not has_player_passed_auction: count += 1`

**2. Auction resolution trigger**
- Resolve when active bidder count == 1 (not 0)
- Rationale: When second-to-last player leaves, one bidder remains and wins automatically
- Pattern: `if _count_active_bidders(state) == 1: _resolve_auction(state)`

**3. Net worth update timing**
- Update winner's net worth immediately after company transfer
- Rationale: Maintains invariant that net_worth reflects current holdings + cash
- Pattern: `transfer → update_net_worth` sequence

**4. Replacement company drawing**
- Draw new company and move to auction during resolution (not deferred)
- Rationale: Maintains constant auction row size, matches game rules
- Pattern: `draw → if valid: move_to_auction`

**5. Infrastructure bug fix**
- Fixed missing Company entity initialization in GameState.initialize_game
- Rationale: Company._location cache must be initialized via _scan_location for transfers to work
- Without this, clear_location fails silently (checks stale cache, doesn't clear auction flag)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Company entity initialization in GameState.initialize_game**
- **Found during:** Task 1 verification (company transfer test failing)
- **Issue:** Company entities were created at module load but never initialized via .initialize(state)
  - Company._location cache was never set via _scan_location
  - clear_location uses cached _location to know which flag to clear
  - Without initialization, transfers silently failed (auction flag never cleared)
- **Fix:**
  - Added `from entities import company as company_module` import
  - Added `for company in company_module.COMPANIES: company.initialize(self)` loop in initialize_game
  - Placed after player/fi/corp initialization, before deck setup
- **Files modified:** core/state.pyx
- **Verification:** Company transfer test passes, auction resolution correctly transfers ownership
- **Committed in:** aeaeca4 (separate fix commit before task commit)

---

**Total deviations:** 1 auto-fixed (missing critical functionality)
**Impact on plan:** Essential infrastructure fix - Company transfers required for BID-06 and BID-07. No scope creep, strictly necessary for correct operation.

## Issues Encountered

None - after infrastructure fix, implementation proceeded smoothly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- BID_IN_AUCTION phase fully functional and tested
- Auction flow complete: INVEST → start auction → BID → resolution → back to INVEST
- Ready for Phase 4: Share trading (buy/sell shares in INVEST phase)
- Turn order navigation utilities proven and reusable

**Ready for:** Phase 4 (Buy/Sell Shares in INVEST phase)

**No blockers:** All requirements (BID-01 through BID-12) verified and passing

---
*Phase: 03-invest-core-auction-flow*
*Completed: 2026-01-21*
