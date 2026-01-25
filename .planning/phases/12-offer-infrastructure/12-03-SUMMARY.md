---
phase: 12
plan: 03
title: "Offer State Presentation"
subsystem: state-management
status: complete
completed: 2026-01-25
duration: 3min

requires:
  - phase: 12-01
    provides: "Hidden offer buffer and acquisition state fields"

provides:
  - "_present_current_offer syncs hidden buffer to visible state"
  - "Offer validation skips invalid offers during presentation"
  - "acq_active_corp, acq_target_company, acq_is_fi_offer set for NN"
  - "active_player set to president of buying corp"

affects:
  - "13-*: Acquisition action handlers will use visible state"
  - "14-*: Flow control uses _advance_to_next_offer"

tech-stack:
  added: []
  patterns:
    - "Hidden-to-visible state sync pattern"
    - "While-loop validation skip pattern"
    - "President detection by share count"

key-files:
  created:
    - tests/test_acquisition.py
  modified:
    - phases/acquisition.pyx

decisions:
  - id: president-detection-method
    choice: "Find president by max share count in _get_corp_president"
    rationale: "Direct share count check simpler than is_president_of method"
  - id: validation-skip-strategy
    choice: "While-loop skip pattern in _present_current_offer"
    rationale: "Incrementally advance index until valid offer found or exhausted"
  - id: receivership-active-player
    choice: "Set active_player to 0 when president is -1"
    rationale: "Prevents invalid player_id, allows receivership auto-buy logic"

metrics:
  tasks: 3
  commits: 3
  files-modified: 2
  functions-added: 6
---

# Phase 12 Plan 03: Offer State Presentation Summary

**Hidden offer buffer synced to visible state (acq_active_corp, acq_target_company, acq_is_fi_offer) with validation skip for invalid offers**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-25T18:52:17Z
- **Completed:** 2026-01-25T18:55:12Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- Offer state presentation functions sync hidden buffer to visible acquisition state
- Invalid offers (already acquired, insufficient cash, FI ownership changed) automatically skipped
- Neural network sees current offer via acq_active_corp, acq_target_company, acq_is_fi_offer
- Active player set to president of buying corp (or 0 for receivership)
- Test infrastructure for state presentation verification

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement offer state presentation** - `87ccce8` (feat)
2. **Task 2: Add offer validation before presentation** - `7224015` (feat)
3. **Task 3: Add tests for state presentation** - `accad86` (test)

## Files Created/Modified

- `phases/acquisition.pyx` - Offer presentation and validation functions
  - `_get_corp_president()` - Find president by max share count
  - `_is_offer_valid()` - Validate offer still viable for presentation
  - `_present_current_offer()` - Sync hidden buffer to visible state with validation skip
  - `_advance_to_next_offer()` - Increment index and present next valid offer
  - `present_current_offer_py()` - Python wrapper for testing
  - `advance_to_next_offer_py()` - Python wrapper for testing
  - `get_offer_index()` - Helper to read current offer index
- `tests/test_acquisition.py` - Test class for state presentation
  - `TestOfferStatePresentation` with 5 tests (1 implemented, 4 stubs)

## Technical Details

### State Presentation Flow

**STATE-01: Present valid offer**
1. Read offer_count and offer_index from hidden state
2. While loop: read offer at current index, validate, skip if invalid
3. If valid offer found: set acq_active_corp, acq_target_company, acq_is_fi_offer
4. Set active_player to president (or 0 if receivership)

**STATE-04: Clear state when exhausted**
1. If offer_index >= offer_count: clear all acq_* fields
2. Sets acq_active_corp to -1 (sentinel for "no active offer")

### Offer Validation

An offer is invalid if:
- Company already acquired this phase (in any corp's acquisition_companies)
- Corp lacks cash for minimum price (low_price or face_value for FI)
- FI no longer owns company (for FI offers)

Invalid offers are skipped automatically by incrementing offer_index until valid offer found or buffer exhausted.

### President Detection

`_get_corp_president()` finds president by iterating all players and finding the one with max shares in the corp. Returns -1 if no player has shares (receivership).

When president is -1, active_player is set to 0 as a safe fallback (prevents invalid player_id).

## Decisions Made

**Decision 1: President detection method**
- Chose to find president by max share count rather than using Player.is_president_of()
- Simpler implementation, avoids dependency on presidencies state field
- Trade-off: O(N players) vs O(1) lookup, but player count is small (2-6)

**Decision 2: Validation skip strategy**
- While-loop pattern that increments index until valid offer found
- Updates hidden offer_index as it skips, maintaining consistency
- Alternative considered: validate during offer generation (rejected - too complex)

**Decision 3: Receivership active_player handling**
- Set active_player to 0 when president is -1
- Allows receivership auto-buy logic to proceed safely
- Future action handlers will check for receivership explicitly

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - implementation straightforward following established state sync patterns.

## Next Phase Readiness

**Ready to proceed:** ✅

The offer presentation infrastructure is complete. Next steps:
- **12-04+**: May add offer buffer population (12-02) if not already done
- **Phase 13**: Acquisition action handlers can read visible state (acq_active_corp, etc.)
- **Phase 14**: Flow control can call _advance_to_next_offer to progress through offers

**No blockers.**

**No concerns.**

## Code Quality

- **Pattern consistency:** ✅ Follows hidden-to-visible sync pattern from auction state
- **Validation:** ✅ Comprehensive offer validation prevents stale offers
- **Testing:** ✅ Test infrastructure in place (1 test implemented, 4 stubs for future)
- **Type safety:** ✅ Cython static types throughout

## Patterns Established

**Hidden-to-visible state sync pattern:**
- Read current index from hidden state
- Validate before presenting (skip invalid with while-loop)
- Update visible state fields that NN observes
- Set active_player based on current context

This pattern can be reused for any phase that pre-computes options in hidden state and presents them one-by-one.

---

**Status:** Complete ✅
**Merged to:** main (commits 87ccce8, 7224015, accad86)
