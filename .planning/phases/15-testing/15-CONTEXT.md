# Phase 15: Testing - Context

**Gathered:** 2026-01-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Comprehensive test coverage validating ACQUISITION phase correctness. Covers offer generation, action execution, validation rules, receivership auto-buy, zone merging, and phase transitions. Does not add new game functionality.

</domain>

<decisions>
## Implementation Decisions

### Test organization
- Single file: `tests/phases/test_acquisition.py` (move from `tests/test_acquisition.py`)
- Move `tests/phases/test_integration.py` to `tests/test_integration.py` (cross-phase tests belong at root)
- Class-per-feature structure: TestOfferGeneration, TestAcceptAction, TestPassAction, TestValidation, TestReceivershipAutoBuy, TestZoneMerging, TestPhaseFlow
- Each test method references requirement ID in docstring (e.g., `"""VALID-03: Validates buyer has sufficient cash."""`)

### Coverage priorities
- Offer generation: Focus on priority order verification (OS-first, price sorting, same-president constraints)
- Validation rules: One test per rule PLUS boundary conditions (exactly enough cash, exactly 1 company left, etc.)
- Action execution: Full state verification — cash transferred, company moved, proceeds tracked, zones updated
- Zone merge: Critical dedicated tests — verify companies and cash move to correct locations, old locations zeroed/marked inactive

### Edge case selection
- Empty states: Comprehensive coverage — no valid offers, empty FI, all receivership, no player companies, no corp companies
- Buffer limits: Trust implementation, no explicit max capacity tests
- Receivership: Test auto-buy success, auto-buy skip (insufficient cash), and verify receivership corps never offered as sellers
- Same-president constraint: Explicit test verifying no offers generated when buyer/seller have different presidents

### Test data patterns
- Direct entity manipulation: Use entity APIs (CORPS.set_cash(), COMPANIES.set_owner(), etc.)
- Extend entities if needed: Add helper methods to entities for reuse rather than test-only helpers
- Test scope: Unit tests = minimal focused setup, integration tests = realistic mid-game scenarios
- Presidency: Direct share assignment via PLAYERS.set_shares_in_corp() — fix entity bugs if presidency not set correctly
- Fixtures: Use best judgment — check existing conftest.py, extend if reusable, factor out repeated inline code

### Claude's Discretion
- Exact number of tests per class
- Whether to use parametrized tests for similar scenarios
- Order of test classes in file
- Specific assertion messages

</decisions>

<specifics>
## Specific Ideas

- Zone merge tests should verify "default" values by referencing game state initialization logic
- If entity methods have bugs (e.g., presidency not correctly set when shares assigned), fix them during testing
- Receivership seller exclusion "should be handled by same-president logic" — worth one explicit verification test

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 15-testing*
*Context gathered: 2026-01-26*
