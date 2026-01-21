---
phase: 04-share-trading
plan: 01
subsystem: game-engine
tags: [cython, share-trading, invest-phase, market, price-movement]

# Dependency graph
requires:
  - phase: 03-invest-and-bid
    provides: INVEST phase handler structure, player/corp entity modules
provides:
  - Buy share handler with price-before-payment mechanics
  - Sell share handler with current-price-then-move mechanics
  - Price movement helpers (find_next_higher_space, find_next_lower_space)
affects: [04-share-trading plan 02 (action masks), 05-presidency-bankruptcy]

# Tech tracking
tech-stack:
  added: []
  patterns: [price-movement-skip-occupied, index-26-always-available]

key-files:
  created: []
  modified:
    - entities/market.pyx
    - entities/market.pxd
    - phases/invest.pyx

key-decisions:
  - "Price movement uses cpdef helpers in Market class for Python accessibility"
  - "Index 26 (price $75) never marked occupied - multiple corps can share"
  - "Bankruptcy (index 0) deferred to Phase 5"

patterns-established:
  - "Price movement pattern: find_next_higher/lower_space skips occupied"
  - "Buy/sell handler pattern: cdef void noexcept with clear sequence"

# Metrics
duration: 2min 22sec
completed: 2026-01-21
---

# Phase 4 Plan 01: Buy/Sell Share Handlers Summary

**Implemented share buy and sell action handlers with proper price movement mechanics - buy pays new price after movement, sell receives current price before movement**

## Performance

- **Duration:** 2min 22sec
- **Started:** 2026-01-21T06:32:42Z
- **Completed:** 2026-01-21T06:35:04Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Market entity has find_next_higher_space() and find_next_lower_space() methods
- Buy share handler: price moves first, player pays new price to corp, share transfers from bank
- Sell share handler: player receives current price, then price moves down, share transfers to bank
- Price movement correctly skips occupied spaces and treats index 26 as always available
- Round-trip tracking incremented on buy/sell
- Player net worth updated after each transaction

## Task Commits

Each task was committed atomically:

1. **Task 1: Add price movement helper functions** - `9910f03` (feat)
2. **Task 2: Implement buy share handler** - `d1a9131` (feat)
3. **Task 3: Implement sell share handler** - `75db874` (feat)

## Files Created/Modified
- `entities/market.pyx` - Added find_next_higher_space() and find_next_lower_space() methods
- `entities/market.pxd` - Added cpdef declarations for new methods
- `phases/invest.pyx` - Added _handle_buy_share() and _handle_sell_share() handlers

## Decisions Made
- Used cpdef for market helper methods to allow Python test access
- CORP_NAMES imported via regular Python import (not cimport) since it's a Python list
- Bankruptcy handling (price index 0) deferred to Phase 5 - just set the index for now

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Initial build failed because cpdef methods need declaration in .pxd file - added declarations
- CORP_NAMES couldn't be cimported (it's a Python list) - changed to regular import

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Buy/sell handlers implemented, ready for action mask generation (plan 02)
- Action masks need to check round-trip limits and price affordability
- Presidency changes and bankruptcy procedures for Phase 5

---
*Phase: 04-share-trading*
*Completed: 2026-01-21*
