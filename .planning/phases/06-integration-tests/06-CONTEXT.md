# Phase 6: Integration & Tests - Context

**Gathered:** 2026-01-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Comprehensive test coverage validates all INVEST and BID_IN_AUCTION phase logic and edge cases. Tests verify action masks, state transitions, and game invariants. Creating new game features is out of scope.

</domain>

<decisions>
## Implementation Decisions

### Test Organization
- Tests live in `tests/phases/` directory (new subdirectory)
- Organize by game phase: `test_invest.py`, `test_bid_in_auction.py`
- Move existing phase tests from `tests/` root to `tests/phases/`
- Shared `conftest.py` with common fixtures (game_state, invest_state, bid_state, etc.)

### Scenario Coverage
- Rules coverage priority — test every rule from RULES.md systematically
- Descriptive test names and docstrings (no rule number citations since RULES.md is a summary)
- Player counts: test boundary only (3 players min, 6 players max)
- Both unit tests for individual actions AND integration tests for key sequences (e.g., full auction flow)

### Edge Case Selection
- **Bankruptcy:** Single bankruptcy via sell-to-zero only (no cascade scenarios — not possible per rules)
- **Presidency:** All transitions — buy-in, sell-out, tie-break (incumbent keeps), receivership entry/exit
- **Price movement:** All boundaries — skip occupied spaces, $75 shared space, boundary limits
- **Round-trips:** Multi-corp tracking (each corp tracked separately) + verify reset before WRAP_UP transition

### Test Verification
- **Action mask:** Exhaustive validation — assert mask before AND after each action
- **Shared helper:** `assert_valid_mask(state, expected_actions)` in conftest
- **State consistency:** Snapshot comparison (verify only expected fields changed) AND invariant assertions
- **Invariant timing:** After every action AND explicit check at test end

### Auction Flow Testing
- Full auction flow tests: slot mapping → bid calculation → auction state setup → winner resolution
- Unit tests for: slot index maps to correct company by face value order, starting bid = face value + price offset
- Example: companies [4, 6, 7] available, slot 2 + offset 3 → auction for face value 6 company, starting bid 9

### Claude's Discretion
- Exact test helper implementation details
- Test execution order within files
- Which specific state fields to snapshot vs invariant-check

</decisions>

<specifics>
## Specific Ideas

- Invariant helpers should verify:
  - Total shares per corp constant (unissued + bank + all players = game data total)
  - Cash balances never negative (player and corp)
  - Net worths never negative
  - Available auction companies never increase (new companies drawn but not available until next turn)
- Tests should follow existing patterns: parametrized player counts, phase-specific fixtures

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-integration-tests*
*Context gathered: 2026-01-21*
