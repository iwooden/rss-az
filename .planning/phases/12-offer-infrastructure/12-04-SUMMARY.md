---
phase: 12
plan: 04
title: "Phase Entry Integration"
subsystem: phase-control
status: complete
completed: 2026-01-25
duration: 5min

requires:
  - phase: 12-02
    provides: "Offer generation (_generate_offers)"
  - phase: 12-03
    provides: "Offer presentation (_present_current_offer)"

provides:
  - "setup_acquisition_phase wires generation and presentation at phase entry"
  - "WRAP_UP calls setup_acquisition_phase before ACQUISITION transition"
  - "Offers pre-generated at phase entry, not lazily"
  - "Phase flow: WRAP_UP -> setup offers -> ACQUISITION"

affects:
  - "13-*: Actions can assume offers are already generated when ACQUISITION starts"
  - "14-*: Flow control starts from already-presented offer"

tech-stack:
  added: []
  patterns:
    - "Phase entry setup pattern: generation + presentation before transition"
    - "Cross-module phase coordination (wrap_up -> acquisition)"

key-files:
  created: []
  modified:
    - phases/acquisition.pyx
    - phases/wrap_up.pyx
    - tests/test_acquisition.py

decisions:
  - id: restore-missing-functions
    choice: "Restored _is_offer_valid, _present_current_offer, _advance_to_next_offer from commits 87ccce8, 7224015"
    rationale: "Functions from 12-03 were lost when 12-02 rewrote acquisition.pyx (bdba0d1). Required for setup_acquisition_phase."
  - id: python-wrapper-for-testing
    choice: "Added apply_wrap_up_py wrapper to wrap_up.pyx"
    rationale: "apply_wrap_up is cdef (not exported), need wrapper for integration tests"

metrics:
  tasks: 3
  commits: 3
  files-modified: 3
  functions-added: 2
---

# Phase 12 Plan 04: Phase Entry Integration Summary

**WRAP_UP calls setup_acquisition_phase to generate and present offers before transitioning to ACQUISITION**

## Performance

- **Duration:** 5 min
- **Started:** 2026-01-25T20:04:59Z
- **Completed:** 2026-01-25T20:09:40Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- setup_acquisition_phase function orchestrates offer generation and presentation
- WRAP_UP phase calls setup before transitioning to ACQUISITION
- Offer buffer populated at phase entry (not lazily) per CONTEXT.md requirement
- Integration tests verify WRAP_UP -> ACQUISITION flow with offer setup
- Restored missing presentation functions lost in commit history

## Task Commits

Each task was committed atomically:

1. **Task 1: Create acquisition phase setup function** - `9fc9f47` (feat)
2. **Task 2: Integrate with WRAP_UP phase** - `81dd535` (feat)
3. **Task 3: Integration test for phase flow** - `c73a52a` (test)

## Files Created/Modified

- `phases/acquisition.pyx` - Phase entry setup
  - Restored `_is_offer_valid()` - Validate offer still viable (lost in bdba0d1)
  - Restored `_present_current_offer()` - Sync hidden buffer to visible state (lost in bdba0d1)
  - Restored `_advance_to_next_offer()` - Increment and present next (lost in bdba0d1)
  - Added `setup_acquisition_phase()` - Main entry point: generate + present
  - Added `setup_acquisition_phase_py()` - Python wrapper for testing
  - Added `get_offer_index()` - Helper to read current index
- `phases/wrap_up.pyx` - WRAP_UP phase handler
  - Import acquisition module
  - Call setup_acquisition_phase before phase transition
  - Added `apply_wrap_up_py()` - Python wrapper for testing
- `tests/test_acquisition.py` - Integration tests
  - TestPhaseFlow class with 3 tests
  - test_wrap_up_sets_up_acquisition: verifies full flow
  - test_empty_offers_detected: verifies buffer exhaustion
  - test_acquisition_with_fi_company: stub for future

## Technical Details

### Phase Entry Flow

**WRAP_UP execution:**
1. Reorder players by cash
2. Clear consecutive passes
3. FI purchases (if any)
4. Make revealed companies available
5. **NEW: setup_acquisition_phase()** ← added in this plan
6. Transition to ACQUISITION

**setup_acquisition_phase steps:**
1. Reset offer_index and offer_count to 0
2. Call _generate_offers() (from 12-02)
3. Call _present_current_offer() (from 12-03)

**Result:**
- Offer buffer populated with sorted offers
- First valid offer presented to visible state (acq_active_corp, acq_target_company, acq_is_fi_offer)
- Active player set to president of buying corp
- Or all fields cleared if no offers (fresh game)

### Cross-Module Integration

**Import pattern:**
```cython
from phases import acquisition as acquisition_module
```

**Call site in wrap_up.pyx:**
```cython
acquisition_module.setup_acquisition_phase(state)
```

**Function signature:**
```cython
cpdef void setup_acquisition_phase(GameState state)
```

cpdef allows calling from both Cython (via cimport) and Python (via import).

## Decisions Made

**Decision 1: Restore missing functions from 12-03**
- Found that _is_offer_valid, _present_current_offer, _advance_to_next_offer were missing
- These were added in commits 87ccce8 and 7224015 (12-03)
- Lost when bdba0d1 (12-02) rewrote acquisition.pyx
- Restored from git history to unblock setup_acquisition_phase
- Trade-off: Could have re-implemented, but restoring preserves original design

**Decision 2: Add Python wrapper for testing**
- apply_wrap_up is cdef (not exported to Python)
- Added apply_wrap_up_py wrapper to enable integration tests
- Alternative: Use cimport in test file (rejected - tests are .py not .pyx)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Restored missing presentation functions**
- **Found during:** Task 1 (setup_acquisition_phase implementation)
- **Issue:** _present_current_offer and _is_offer_valid didn't exist in current acquisition.pyx, but were required by setup_acquisition_phase and referenced in 12-03 SUMMARY
- **Root cause:** Commits 87ccce8 and 7224015 (12-03) added these functions, but bdba0d1 (12-02) came after and completely rewrote the file without including them
- **Fix:** Extracted functions from git history (87ccce8, 7224015) and re-added to current file
- **Files modified:** phases/acquisition.pyx
- **Verification:** Build succeeds, setup_acquisition_phase_py works, all tests pass
- **Committed in:** 9fc9f47 (Task 1 commit)

**2. [Rule 1 - Bug] Added missing get_company_low_price import**
- **Found during:** Task 1 (adding _is_offer_valid)
- **Issue:** _is_offer_valid uses get_company_low_price but import was missing
- **Fix:** Added get_company_low_price to imports from core.data
- **Files modified:** phases/acquisition.pyx
- **Verification:** Build succeeds without undefined symbol errors
- **Committed in:** 9fc9f47 (Task 1 commit)

**3. [Rule 2 - Missing Critical] Added Python wrapper for apply_wrap_up**
- **Found during:** Task 2 (integration test)
- **Issue:** apply_wrap_up is cdef (not exported), tests couldn't import it
- **Fix:** Added apply_wrap_up_py Python wrapper
- **Files modified:** phases/wrap_up.pyx
- **Verification:** Tests can import and call wrapper, all integration tests pass
- **Committed in:** 81dd535 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (1 blocking, 1 bug, 1 missing critical)
**Impact on plan:** Deviations 1-2 were essential to unblock implementation (missing code from previous plan). Deviation 3 enables testing. No scope creep.

## Issues Encountered

**Issue: Missing presentation functions**
- Problem: 12-03 SUMMARY claims presentation functions were added, but they're missing from current codebase
- Investigation: Git history shows commits 87ccce8 and 7224015 added them, but bdba0d1 (12-02) came after and rewrote the file
- Resolution: Restored functions from git history (commits 87ccce8, 7224015)
- Prevention: This suggests 12-02 and 12-03 were executed out of order or there was a merge conflict

**No other issues** - plan executed smoothly after restoring missing functions.

## Next Phase Readiness

**Ready to proceed:** ✅

Phase entry infrastructure complete:
- WRAP_UP properly sets up ACQUISITION phase
- Offers generated and presented before phase transition
- Fresh game correctly handles no offers (acq_active_corp == -1)
- Integration tests verify flow

**Next steps:**
- **Phase 13**: Acquisition action handlers (ACCEPT, PASS, PRICE)
- **Phase 14**: Flow control (advance after accept/pass, transition when exhausted)

**No blockers.**

**No concerns.**

## Code Quality

- **Pattern consistency:** ✅ Follows phase entry setup pattern
- **Cross-module integration:** ✅ Clean import and call from wrap_up to acquisition
- **Testing:** ✅ Integration tests verify full WRAP_UP -> ACQUISITION flow
- **Type safety:** ✅ Cython static types throughout

## Patterns Established

**Phase entry setup pattern:**
1. Phase handler (e.g., WRAP_UP) calls setup function before transition
2. Setup function orchestrates multiple operations (generate, present)
3. Setup completes before phase transition
4. Next phase starts with state already prepared

This pattern ensures phase invariants are established before the phase begins, rather than lazily during the phase.

**Cross-module phase coordination:**
- Import pattern: `from phases import module as module_name`
- Function signature: `cpdef` for cross-module callable functions
- Clean separation: each phase owns its setup logic

---

**Status:** Complete ✅
**Merged to:** main (commits 9fc9f47, 81dd535, c73a52a)
