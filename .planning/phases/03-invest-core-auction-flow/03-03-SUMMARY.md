---
phase: 03-invest-core-auction-flow
plan: 03
subsystem: testing
tags: [pytest, test-coverage, invest-phase, bid-phase, auction-flow]

# Dependency graph
requires:
  - phase: 03-01
    provides: INVEST phase pass and start auction implementation
  - phase: 03-02
    provides: BID_IN_AUCTION phase implementation with resolution
  - phase: 02-02
    provides: Test patterns and fixtures
affects: [future-test-expansion, regression-prevention]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Test fixture pattern for phase-specific states (game_state, auction_state)
    - Parametrized player count testing (3-6 players)
    - Helper function pattern for common test operations
    - Public accessor pattern for internal state (get_active_player, get_num_players)

key-files:
  created:
    - tests/test_invest.py
    - tests/test_bid.py
  modified:
    - core/state.pyx (added get_active_player, get_num_players)
    - core/state.pxd (added public method declarations)
    - entities/company.pyx (fixed clear_location stale cache bug)

key-decisions:
  - "Added public accessors get_active_player() and get_num_players() to GameState for test access"
  - "Fixed Company._location cache staleness by rescanning in clear_location before clearing"
  - "Used get_owner_id() >= 0 pattern to check company player ownership in tests"

patterns-established:
  - "Test fixture pattern: Phase-specific fixtures return state in the target phase"
  - "Helper function pattern: Extract common operations like get_first_valid_auction_action"
  - "Parametrized testing: Use @pytest.mark.parametrize for player count variations"

# Metrics
duration: 10min 21sec
completed: 2026-01-21
---

# Phase 03 Plan 03: INVEST & BID Test Coverage Summary

**Comprehensive test suite verifying all 19 requirements (INV-01 through INV-06, BID-01 through BID-12) with infrastructure fixes for testability and correctness**

## Performance

- **Duration:** 10min 21sec
- **Started:** 2026-01-21T01:22:29Z
- **Completed:** 2026-01-21T01:32:50Z
- **Tasks:** 2
- **Files modified:** 4
- **Files created:** 2

## Accomplishments

- 25 INVEST phase tests covering all pass and auction start behaviors
- 26 BID phase tests covering leave, raise bid, and auction resolution
- Fixed critical bug in Company.clear_location (stale location cache)
- Added public accessors for active_player and num_players to enable testing
- All 103 tests passing (no regressions)

## Task Commits

Each task and fix was committed atomically:

1. **Infrastructure Fix: Add Python-accessible getters** - `c7704d4` (fix)
2. **Task 1: INVEST phase test coverage** - `ceeccdb` (test)
3. **Infrastructure Fix: Company clear_location bug** - `29c8079` (fix)
4. **Task 2: BID phase test coverage** - `ba97351` (test)

## Files Created/Modified

- **tests/test_invest.py** - 25 tests for INVEST phase (324 lines)
- **tests/test_bid.py** - 26 tests for BID phase (512 lines)
- **core/state.pyx** - Added cpdef get_active_player() and get_num_players()
- **core/state.pxd** - Added public method declarations
- **entities/company.pyx** - Fixed clear_location to rescan before clearing

## Decisions Made

**1. Public accessors for test infrastructure**
- Added `cpdef get_active_player()` and `cpdef get_num_players()` to GameState
- Rationale: Tests need to verify turn order advancement, player rotation
- Pattern: Wrap cdef methods with cpdef for Python accessibility
- Rule: Missing critical functionality (Rule 2)

**2. Company.clear_location bug fix**
- Fixed stale `_location` cache by rescanning at start of clear_location
- Rationale: Company.initialize() called before companies were drawn/placed, cache became stale
- Impact: Auction company transfers now work correctly (BID-07)
- Rule: Bug fix (Rule 1)

**3. Test API patterns**
- Used `get_owner_id() >= 0` to check player ownership (not `is_owned_by_player`)
- Created phase-specific fixtures (auction_state returns state in BID phase)
- Helper functions extract common operations (get_first_valid_auction_action)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Public accessors for active_player and num_players**
- **Found during:** Task 1 test implementation
- **Issue:** GameState._get_active_player() and ._num_players are cdef (not accessible from Python tests)
  - Tests need to verify turn order advancement after pass action
  - Tests need to verify player rotation in auctions
  - No public API existed to access these values
- **Fix:**
  - Added `cpdef int get_active_player(self)` wrapping `_get_active_player()`
  - Added `cpdef int get_num_players(self)` returning `_num_players`
  - Updated core/state.pxd with public method declarations
- **Files modified:** core/state.pyx, core/state.pxd
- **Verification:** INVEST tests successfully verify turn order advancement
- **Committed in:** c7704d4 (separate fix commit before Task 1)

**2. [Rule 1 - Bug] Company._location cache stale after initialization**
- **Found during:** Task 2 test verification (BID-07 test failing)
- **Issue:** Company.clear_location() relied on cached `_location` to know which flag to clear
  - Company.initialize() called `_scan_location()` during GameState.initialize_game
  - At that point, no companies had been drawn or placed yet (all in deck)
  - `_scan_location` set `_location = LOC_DECK` for all companies
  - After initialization, companies were drawn and marked for auction
  - But `_location` cache was never updated
  - When transfer_to_player called clear_location, it used stale cache
  - Result: Auction flag never cleared, company ownership broken
- **Fix:**
  - Added `self._scan_location(state)` at start of clear_location
  - Now always rescans to get current location before clearing
  - Slight performance cost but ensures correctness
- **Files modified:** entities/company.pyx
- **Verification:** BID-07 test passes, company properly transferred with auction flag cleared
- **Committed in:** 29c8079 (separate fix commit before Task 2)

---

**Total deviations:** 2 auto-fixed (1 missing critical functionality, 1 bug)
**Impact on plan:** Essential fixes - tests could not be written without get_active_player, and auction resolution was broken without the cache fix. Both are correctness issues, not scope creep.

## Issues Encountered

None after infrastructure fixes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Full test coverage for INVEST and BID phases (51 new tests)
- All 19 requirements verified with regression tests
- Infrastructure improvements enable future test development
- Bug fixes improve system correctness

**Ready for:** Phase 4 (Buy/Sell Shares in INVEST phase) or Phase 5 (Advanced INVEST features)

**No blockers:** All requirements tested and passing

## Test Coverage Summary

### INVEST Phase (tests/test_invest.py - 25 tests)

**TestPassAction (5 tests):**
- INV-01: Pass increments consecutive_passes ✓
- INV-02: Non-pass resets consecutive_passes ✓
- INV-03: All players pass → WRAP_UP transition ✓
- INV-04: Pass advances active player in turn order ✓
- INV-04a: Turn order navigation uses one-hot vectors ✓

**TestStartAuction (8 tests):**
- INV-05: Auction state initialization (company, price, high_bidder, starter, cleared flags) ✓
- INV-06: Phase transition to BID_IN_AUCTION ✓
- INV-02: Start auction resets consecutive_passes ✓
- Turn order advancement verified ✓

**TestMultiplePlayerCounts (12 tests):**
- Pass action works for 3-6 players ✓
- Auction action works for 3-6 players ✓
- WRAP_UP triggers at correct pass count for each player count ✓

### BID Phase (tests/test_bid.py - 26 tests)

**TestLeaveAuction (4 tests):**
- BID-01: Leave sets passed flag ✓
- BID-02: Active bidder rotation skips passed players ✓
- BID-05: Last leaver triggers resolution ✓

**TestRaiseBid (4 tests):**
- BID-03: Raise updates price and high bidder ✓
- BID-04: Bidder advancement verified ✓
- Bid value calculation verified (face + amount + 1) ✓

**TestAuctionResolution (7 tests):**
- BID-06: Winner pays bid price ✓
- BID-07: Winner receives company ✓
- BID-08: Auction state cleared ✓
- BID-09: New company drawn ✓
- BID-10: Returns to INVEST phase ✓
- BID-11: Turn to player after starter ✓
- BID-12: Net worth updated ✓

**TestFullAuctionCycle (3 tests):**
- Complete auction workflow ✓
- Immediate resolution (all leave) ✓
- Multiple raises before resolution ✓

**TestMultiplePlayerCounts (8 tests):**
- Auction flow works for 3-6 players ✓
- Bidder rotation correct for all counts ✓

### Overall Coverage

- **Total tests:** 103 (51 new, 52 existing)
- **All tests passing:** ✓
- **Requirements covered:** INV-01 through INV-06, BID-01 through BID-12
- **No regressions:** All Phase 1 and Phase 2 tests still passing

---
*Phase: 03-invest-core-auction-flow*
*Completed: 2026-01-21*
