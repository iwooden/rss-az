# Phase 11: Test Updates - Context

**Gathered:** 2026-01-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix existing tests that fail due to WRAP_UP auto-continuation and add verification tests for player reordering, FI purchases, and phase transitions. No new game logic — only test maintenance and verification.

</domain>

<decisions>
## Implementation Decisions

### Test fix approach
- Update assertions to expect actual target phase (not GAME_OVER as placeholder)
- Production code transitions to correct next phase with stub handlers for unimplemented phases
- Clean up any temporary workarounds written when phases were missing
- Keep existing test file structure (test_invest.py, test_bid_auction.py, etc.)

### Phase transition strategy
- Each phase transitions to the correct next phase enum from the start
- Unimplemented phases have proper stubs that don't throw errors
- Tests assert the real target phase — no GAME_OVER placeholders
- This prevents revisiting tests each time a new phase is implemented

### Verification coverage
- Comprehensive coverage: player reordering, FI purchases, availability transitions, phase flow, edge cases
- Player reordering: explicit tie scenarios where 2+ players have equal cash, verify old order preserved
- FI purchases: all edge cases — FI 0 cash, empty deck, no available companies, multiple purchases, partial affordability
- Per-phase test files: each file tests transitions originating from its phase (test_invest.py tests INVEST → WRAP_UP, test_wrap_up.py tests WRAP_UP → ACQUISITION)

### Integration tests
- Scan existing tests for any integration-style tests
- Move integration tests to dedicated test_integration.py
- This becomes the single place to extend as phases are added
- Avoids scattered partial-cycle tests that only cover some phases

### Test utilities
- Existing set_phase() in turn.pyx (line 103) is sufficient
- No additional wrappers or conftest fixtures needed

### Claude's Discretion
- History assertion style: explicit counts vs pattern-based, based on test purpose
- Assertion helpers: create vs inline, based on repetition in tests

</decisions>

<specifics>
## Specific Ideas

- Player reordering verification: assert final positions only (not cash values in assertions)
- FI purchase verification: assert FI cash decreases and company ownership transfers to FI

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 11-test-updates*
*Context gathered: 2026-01-23*
