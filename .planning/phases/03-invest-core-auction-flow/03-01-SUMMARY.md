---
phase: 03-invest-core-auction-flow
plan: 01
subsystem: game-logic
tags: [cython, game-state, invest-phase, auction, turn-order]

# Dependency graph
requires:
  - phase: 02-infrastructure-setup
    provides: GameDriver dispatch, entity handles, test infrastructure
provides:
  - INVEST phase pass action with consecutive pass tracking
  - INVEST phase start auction action with auction state initialization
  - Turn order navigation utilities for player advancement
  - Phase transition logic (INVEST -> WRAP_UP, INVEST -> BID_IN_AUCTION)
affects: [03-02-bid-phase, 04-buy-sell-shares, wrap-up-phase]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Turn order navigation pattern (find player at position, advance in turn order)
    - Auction state initialization pattern (set all fields atomically)
    - Module import pattern for entities (avoid circular imports)

key-files:
  created: []
  modified:
    - phases/invest.pyx

key-decisions:
  - "Use module import pattern (from entities import turn as turn_module) to avoid Cython circular imports"
  - "Declare all cdef variables at function start to satisfy Cython syntax requirements"
  - "Advance to next bidder after auction start (skipping passed players) even though no one has passed yet"

patterns-established:
  - "Turn order navigation: find player at position, calculate next position with wraparound"
  - "Helper function pattern: _find_player_at_position, _advance_active_player, _advance_to_next_bidder"

# Metrics
duration: 2min 37sec
completed: 2026-01-21
---

# Phase 03 Plan 01: INVEST Core & Auction Flow Summary

**Pass action with consecutive pass tracking and start auction with full state initialization for BID_IN_AUCTION phase**

## Performance

- **Duration:** 2min 37sec
- **Started:** 2026-01-21T00:42:08Z
- **Completed:** 2026-01-21T00:44:45Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Pass action increments consecutive_passes, advances turn order, triggers WRAP_UP when all players pass
- Start auction initializes all auction state (company, price, high_bidder, starter, passed flags)
- Start auction transitions to BID_IN_AUCTION phase and advances to next bidder
- Auction clears consecutive_passes counter (INV-02 compliance)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement INVEST phase pass and start auction** - `1d88cbb` (feat)

## Files Created/Modified
- `phases/invest.pyx` - Full implementation of pass and auction actions with helper functions for turn order navigation

## Decisions Made

**1. Module import pattern for entities**
- Used `from entities import turn as turn_module` instead of `from entities.turn cimport TURN`
- Rationale: Avoids Cython circular import issues while maintaining clean API access
- Pattern: Import module, access singleton handle via `module.HANDLE`

**2. Cdef variable declaration placement**
- Declared all cdef variables at function start: `cdef int company_id, face_value, bid_price, player_id`
- Rationale: Cython requires all cdef declarations before any code in if/elif blocks
- Pattern: Declare all potential variables upfront, assign in branches

**3. Advance to next bidder after auction start**
- Call `_advance_to_next_bidder(state)` even though no players have passed yet
- Rationale: Sets active player to next player in turn order (player after auction starter)
- Pattern: Consistent with turn order advancement in pass action

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

**1. Cython syntax error: cdef statement not allowed after elif**
- Problem: Initial implementation declared cdef variables inside elif blocks
- Solution: Moved all cdef declarations to function start
- Resolution time: ~30 seconds (quick compile-fix cycle)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- INVEST phase ready for integration with BID_IN_AUCTION phase (Plan 03-02)
- Turn order navigation utilities ready for reuse in other phases
- Consecutive pass tracking fully functional for WRAP_UP transition
- Auction state initialization complete, ready for bidding logic

**Ready for:** Phase 3 Plan 2 (BID_IN_AUCTION implementation)

**No blockers:** All requirements (INV-01 through INV-06) verified and passing

---
*Phase: 03-invest-core-auction-flow*
*Completed: 2026-01-21*
