# Phase 11: Test Updates - Research

**Researched:** 2026-01-23
**Domain:** Pytest test maintenance for Cython game engine with auto-continuation and phase transitions
**Confidence:** HIGH

## Summary

This phase focuses on test maintenance after implementing auto-continuation through WRAP_UP and ACQUISITION phases. The codebase already has a mature pytest testing infrastructure with shared fixtures (conftest.py), assertion helpers, and parametrization patterns. The key challenge is updating existing tests that now fail due to changed phase transition behavior, and adding comprehensive verification for new WRAP_UP/ACQUISITION logic.

The standard approach is to fix assertions to match actual target phases (not GAME_OVER placeholders), add per-phase test files that verify transitions originating from their phase, and consolidate integration-style tests into a dedicated file. The existing `set_phase()` method in turn.pyx (line 103) is sufficient for test utilities — no new wrappers needed.

**Primary recommendation:** Organize tests by phase with each file testing transitions originating from its phase, use existing fixture patterns (apply_action_and_verify, apply_and_track), add comprehensive edge case coverage for player reordering and FI purchases using @pytest.mark.parametrize for systematic scenario coverage.

## Standard Stack

The codebase already uses the established Python testing stack. No new libraries needed.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | ≥7.x | Test framework | Industry standard for Python, excellent parametrization and fixtures |
| numpy | (existing) | Array assertions | Already in project, testing requires numpy.testing.assert_array_equal |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest-cov | (optional) | Coverage reporting | If coverage tracking desired (Cython coverage requires linetrace directive) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pytest | unittest | pytest has better parametrization, fixtures, and assertion introspection |
| Custom helpers | pytest-cython plugin | Plugin is for doctesting C extensions; custom helpers are more flexible for game logic |

**Installation:**
```bash
# Already installed in project
pytest tests/
```

## Architecture Patterns

### Recommended Test Organization Structure

Based on pytest best practices and the existing codebase:

```
tests/
├── conftest.py                  # Shared fixtures (already exists)
├── test_init.py                 # GameState initialization (already exists)
├── test_driver.py               # Driver dispatch logic (already exists)
└── phases/
    ├── conftest.py              # Phase-specific fixtures and helpers (already exists)
    ├── test_invest.py           # INVEST phase tests (already exists)
    ├── test_bid_in_auction.py   # BID phase tests (already exists)
    ├── test_wrap_up.py          # NEW: WRAP_UP phase tests
    ├── test_acquisition.py      # NEW: ACQUISITION phase tests (stub verification)
    └── test_integration.py      # NEW: Multi-phase integration tests
```

### Pattern 1: Per-Phase Test Files

**What:** Each phase has a dedicated test file that verifies behavior originating from that phase.
**When to use:** Always — this is the organizational principle for phase-based testing.

**Responsibility mapping:**
- `test_invest.py`: Tests INVEST actions and INVEST → BID/WRAP_UP transitions
- `test_bid_in_auction.py`: Tests BID actions and BID → INVEST transitions
- `test_wrap_up.py`: Tests WRAP_UP → ACQUISITION transitions and player reordering
- `test_acquisition.py`: Tests ACQUISITION → INVEST transitions

**Example structure:**
```python
# test_wrap_up.py
class TestWrapUpTransition:
    """WRAP_UP phase execution and transitions."""

    def test_transitions_to_acquisition(self, wrap_up_state):
        """WRAP_UP phase transitions to ACQUISITION."""
        # Setup: state in WRAP_UP
        # Execute: auto-continuation triggers WRAP_UP
        # Verify: state.get_phase() == PHASE_ACQUISITION

class TestPlayerReordering:
    """Player reordering by cash with tie-breaking."""

    @pytest.mark.parametrize("cash_values,expected_order", [
        ([30, 25, 20], [0, 1, 2]),  # No ties
        ([30, 30, 20], [0, 1, 2]),  # Tie: old order preserved
    ])
    def test_reorder_scenarios(self, cash_values, expected_order):
        # Setup state with specific cash values
        # Execute WRAP_UP
        # Verify player turn order matches expected
```

### Pattern 2: Shared Assertion Helpers

**What:** Reusable validation functions in conftest.py with pytest.register_assert_rewrite.
**When to use:** When the same validation logic is needed across multiple tests.

**Existing helpers (from conftest.py):**
```python
def assert_valid_mask(state, expected_actions=None, msg="")
    """Verify action mask validity."""

def assert_invariants(state, msg="")
    """Check game state invariants (shares, cash, net worth)."""

def apply_action_and_verify(state, action_idx, msg="")
    """Apply action with invariant checks."""
```

**Pattern for new helpers:**
```python
# In conftest.py
def assert_player_order(state, expected_positions, msg=""):
    """Verify player turn order matches expected."""
    __tracebackhide__ = True  # Hide helper from traceback
    for player_id, expected_pos in enumerate(expected_positions):
        actual = PLAYERS[player_id].get_turn_order(state)
        assert actual == expected_pos, \
            f"{msg}\nPlayer {player_id} order: {actual} != {expected_pos}"
```

**Registration (if helper in separate module):**
```python
# At top of conftest.py or test file
pytest.register_assert_rewrite('tests.helpers')  # If using separate helpers module
```

### Pattern 3: History Verification with apply_and_track

**What:** Use the existing `apply_and_track` fixture for verifying action sequences and intermediate states.
**When to use:** When testing auto-continuation behavior or verifying specific action sequences.

**Example:**
```python
def test_invest_to_wrap_up_auto_continues(self, invest_state, apply_and_track):
    """INVEST → WRAP_UP transition auto-continues through WRAP_UP."""
    # Apply pass action that triggers WRAP_UP
    result = apply_and_track(invest_state, pass_action_idx)

    # Verify history includes sentinel for WRAP_UP
    assert len(result.history) >= 2  # User action + WRAP_UP sentinel
    assert result.get_action_at(-1) == ACTION_WRAP_UP_SENTINEL

    # Verify final phase after auto-continuation
    assert result.state.get_phase() == GamePhases.PHASE_ACQUISITION
```

### Pattern 4: Parametrized Edge Case Coverage

**What:** Use @pytest.mark.parametrize to systematically test edge cases and boundary conditions.
**When to use:** For comprehensive coverage of scenarios like tie-breaking, FI purchases with various conditions.

**Example:**
```python
class TestFIPurchases:
    """FI purchase logic edge cases."""

    @pytest.mark.parametrize("fi_cash,available_companies,expected_purchases", [
        (4, [1, 2, 5], 2),      # FI can afford 1+2, stops at 5
        (0, [1, 2, 5], 0),      # FI broke, no purchases
        (10, [], 0),            # No available companies
        (10, [1, 2, 5, 6], 3),  # Multiple purchases until can't afford
    ])
    def test_fi_purchase_scenarios(self, fi_cash, available_companies, expected_purchases):
        # Setup state with specific FI cash and available companies
        # Execute WRAP_UP
        # Verify number of FI purchases and final FI cash
```

### Pattern 5: Integration Tests in Dedicated File

**What:** Multi-phase workflow tests in test_integration.py, separate from unit tests.
**When to use:** Testing complete turn cycles or phase sequences.

**Example:**
```python
# test_integration.py
class TestFullTurnCycle:
    """Complete turn cycle: INVEST → WRAP_UP → ACQUISITION → INVEST."""

    def test_pass_until_wrap_up_completes_turn(self):
        """All players pass → WRAP_UP → ACQUISITION → new INVEST round."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Players pass until WRAP_UP triggers
        # Verify player reordering, FI purchases, phase transitions
        # Verify return to INVEST with incremented turn number
```

### Anti-Patterns to Avoid

- **Testing GAME_OVER placeholders:** Don't assert `state.get_phase() == GAME_OVER` when the real target phase is ACQUISITION or WRAP_UP. This creates technical debt.
- **Scattered integration tests:** Don't put multi-phase tests in test_invest.py or test_bid_in_auction.py. Use test_integration.py for clarity.
- **Implicit phase expectations:** Always assert expected phase explicitly. Don't assume phase based on other state changes.
- **Skipping tie scenarios:** Player reordering with equal cash is a critical edge case. Must test tie-breaking explicitly.

## Don't Hand-Roll

Problems that look simple but have existing solutions in the codebase or pytest ecosystem:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Test state snapshots | Custom copy logic | `state._array.copy()` (existing) | GameState already uses numpy arrays; copying is built-in |
| Phase manipulation | Custom test helpers | `TURN.set_phase(state, phase)` | Already exists in turn.pyx line 103 |
| Action validation | Manual mask checks | `apply_action_and_verify` fixture | Already handles mask validation, invariants, status checks |
| History tracking | Custom action log | `apply_and_track` fixture | Already collects (state.copy(), action) tuples |
| Parametrized scenarios | Manual test duplication | `@pytest.mark.parametrize` | Pytest's built-in parametrization is more maintainable |
| Assertion introspection | String formatting in asserts | Pytest's native assert rewriting | Pytest provides detailed failure messages automatically |

**Key insight:** The existing conftest.py fixtures (apply_action_and_verify, apply_and_track, ApplyTrackResult) provide all needed test utilities. Don't create new wrappers — use what exists.

## Common Pitfalls

### Pitfall 1: Phase Assertion After Auto-Continuation

**What goes wrong:** Tests assert phase based on old behavior (before auto-continuation was implemented), expecting state to stop in WRAP_UP when it actually continues to ACQUISITION.

**Why it happens:** Auto-continuation through non-player phases (WRAP_UP, ACQUISITION) was added after initial tests were written.

**How to avoid:**
- Always assert the actual target phase from requirements (not intermediate phases)
- For INVEST tests ending a round, assert `PHASE_ACQUISITION` (after WRAP_UP auto-continues)
- Use `apply_and_track` to inspect intermediate phases if needed

**Warning signs:**
- Test failures with "expected GAME_OVER, got ACQUISITION"
- Assertions checking for WRAP_UP when WRAP_UP is deterministic

### Pitfall 2: Missing Tie-Breaking Coverage

**What goes wrong:** Tests only verify player reordering with unique cash values, missing the critical edge case where 2+ players have equal cash.

**Why it happens:** Tie scenarios are less obvious and require intentional setup.

**How to avoid:**
- Always parametrize reordering tests with tie scenarios
- Explicitly test preservation of old position for ties (REORDER-02 requirement)
- Test at maximum player count (6 players) where ties are more likely

**Warning signs:**
- Only testing `[30, 25, 20]` scenarios (no ties)
- No explicit assertion on old position preservation

### Pitfall 3: FI Purchase Edge Cases Not Covered

**What goes wrong:** Tests verify happy path (FI buys companies) but miss edge cases like FI with 0 cash, empty deck, no available companies.

**Why it happens:** Edge cases require intentional setup and are easy to overlook.

**How to avoid:**
- Parametrize FI purchase tests with edge cases:
  - FI cash = 0
  - No available companies (all owned or revealed)
  - Empty deck (no replacement cards)
  - Partial affordability (can afford some but not all)
- Always verify both FI cash decrease AND company ownership transfer

**Warning signs:**
- Only testing scenarios where FI successfully purchases
- No tests for empty deck or 0 cash scenarios

### Pitfall 4: History Sentinel Values Not Verified

**What goes wrong:** Tests verify final state but don't check that non-player phases recorded sentinel actions (-100, -101) in history.

**Why it happens:** History is optional parameter, easy to skip verification.

**How to avoid:**
- Use `apply_and_track` for tests involving non-player phases
- Assert expected action sequence including sentinels
- Verify `result.get_action_at(i)` for each expected history entry

**Warning signs:**
- Never using `apply_and_track` in WRAP_UP/ACQUISITION tests
- No assertions on `result.history` or sentinel values

### Pitfall 5: Integration Tests Mixed with Unit Tests

**What goes wrong:** Multi-phase workflow tests scattered across test_invest.py and test_bid_in_auction.py, making it hard to find integration tests or understand scope.

**Why it happens:** Natural to add tests near related code, but integration tests have different purpose.

**How to avoid:**
- Dedicated test_integration.py for multi-phase workflows
- Each phase test file only tests transitions originating from that phase
- Move existing integration-style tests during this phase

**Warning signs:**
- Tests in test_invest.py that go INVEST → BID → INVEST → WRAP_UP
- Tests named "test_full_turn_cycle" in phase-specific files

## Code Examples

Verified patterns from the existing codebase and pytest documentation.

### Parametrized Player Reordering Tests

```python
# Source: pytest parametrize pattern + project requirements
class TestPlayerReordering:
    """Player reordering by cash with tie-breaking."""

    @pytest.mark.parametrize("num_players,cash_values,old_positions,expected_order", [
        # No ties - descending cash
        (3, [30, 25, 20], [0, 1, 2], [0, 1, 2]),
        (3, [20, 30, 25], [0, 1, 2], [1, 2, 0]),

        # Two-way tie - old position preserved
        (3, [30, 30, 20], [0, 1, 2], [0, 1, 2]),  # Players 0,1 tied → 0 first
        (3, [30, 30, 20], [1, 0, 2], [1, 0, 2]),  # Players 0,1 tied → 1 first

        # Three-way tie
        (3, [30, 30, 30], [0, 1, 2], [0, 1, 2]),
        (3, [30, 30, 30], [2, 1, 0], [2, 1, 0]),

        # Six players with complex ties
        (6, [30, 30, 25, 25, 25, 20], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]),
    ])
    def test_reorder_by_cash_with_ties(
        self, num_players, cash_values, old_positions, expected_order
    ):
        """Verify player reordering respects cash and tie-breaking."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Setup: Set cash and old positions
        for i in range(num_players):
            PLAYERS[i].set_cash(state, cash_values[i])
            PLAYERS[i].set_turn_order(state, old_positions[i])

        # Execute: Trigger WRAP_UP (would normally be via auto-continuation)
        apply_wrap_up(state)

        # Verify: Check final positions
        for player_id in range(num_players):
            actual = PLAYERS[player_id].get_turn_order(state)
            assert actual == expected_order[player_id], \
                f"Player {player_id} order mismatch"
```

### FI Purchase Edge Cases

```python
# Source: WRAP_UP logic requirements + pytest parametrize
class TestFIPurchases:
    """FI purchase loop edge cases."""

    @pytest.mark.parametrize("fi_cash,setup_companies,expected_fi_cash,expected_purchases", [
        # Happy path: FI buys affordable companies
        (10, [(1, True), (2, True), (5, True), (6, True)], 2, 2),  # Buys 1+2+5, stops at 6

        # Edge: FI has 0 cash
        (0, [(1, True), (2, True)], 0, 0),

        # Edge: No available companies
        (10, [], 10, 0),

        # Edge: All companies too expensive
        (4, [(5, True), (6, True), (7, True)], 4, 0),

        # Edge: Partial affordability
        (8, [(1, True), (2, True), (5, True), (10, True)], 0, 3),
    ])
    def test_fi_purchase_scenarios(
        self, fi_cash, setup_companies, expected_fi_cash, expected_purchases
    ):
        """FI purchase loop handles edge cases correctly."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Setup: Configure FI cash and available companies
        FI.set_cash(state, fi_cash)
        for company_id in range(36):
            COMPANIES[company_id].move_to_deck(state)  # Clear all

        for face_value, available in setup_companies:
            company_id = COMPANY_NAME_TO_ID[f"company_with_face_{face_value}"]
            COMPANIES[company_id].move_to_auction(state)

        # Track FI-owned companies before
        fi_owned_before = sum(
            1 for cid in range(36) if FI.owns_company(state, cid)
        )

        # Execute
        _process_fi_purchases(state)

        # Verify cash
        assert FI.get_cash(state) == expected_fi_cash

        # Verify purchases
        fi_owned_after = sum(
            1 for cid in range(36) if FI.owns_company(state, cid)
        )
        assert fi_owned_after - fi_owned_before == expected_purchases
```

### History Verification with Sentinels

```python
# Source: conftest.py ApplyTrackResult pattern + driver.pyx sentinels
class TestWrapUpHistory:
    """WRAP_UP phase history recording."""

    def test_wrap_up_records_sentinel(self, invest_state, apply_and_track):
        """WRAP_UP execution records sentinel action in history."""
        # Setup: All players pass to trigger WRAP_UP
        layout = get_action_layout(3)
        pass_idx = layout['pass_invest']

        # Apply passes until WRAP_UP triggers
        for _ in range(3):
            result = apply_and_track(invest_state, pass_idx)

        # Final pass triggers WRAP_UP auto-continuation
        result = apply_and_track(invest_state, pass_idx)

        # Verify history contains:
        # 1. User's pass action
        # 2. WRAP_UP sentinel (-100)
        # 3. ACQUISITION sentinel (-101)
        assert result.applied_count >= 3

        # Find sentinel in history
        sentinels = [
            action for _, action in result.history
            if action < 0
        ]
        assert -100 in sentinels  # ACTION_WRAP_UP_SENTINEL
        assert -101 in sentinels  # ACTION_ACQUISITION_SENTINEL
```

### Assertion Helper Pattern

```python
# Source: conftest.py existing helpers + pytest best practices
def assert_player_order(state, expected_positions, msg=""):
    """
    Verify player turn order matches expected positions.

    Args:
        state: GameState to check
        expected_positions: List[int] where index=player_id, value=expected position
        msg: Context for assertion failure
    """
    __tracebackhide__ = True  # Hide from pytest traceback

    for player_id, expected_pos in enumerate(expected_positions):
        actual = PLAYERS[player_id].get_turn_order(state)
        assert actual == expected_pos, \
            f"{msg}\nPlayer {player_id} turn order: expected {expected_pos}, got {actual}"

def assert_fi_owns_company(state, company_id, msg=""):
    """Verify FI owns specific company."""
    __tracebackhide__ = True
    assert FI.owns_company(state, company_id), \
        f"{msg}\nFI does not own company {company_id}"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual test isolation | Pytest fixtures with parametrize | Pytest 3.0+ (2016) | More maintainable, less duplication |
| Custom assertion libs | Pytest's assert rewriting | Pytest 2.1+ (2011) | Better introspection, no special syntax |
| unittest.TestCase | Pytest test classes/functions | Pytest adoption | Simpler syntax, better fixtures |
| Hardcoded test data | @pytest.mark.parametrize | Pytest 2.2+ (2012) | Systematic edge case coverage |
| Integration tests scattered | Dedicated test_integration.py | Best practice pattern | Clearer organization |

**Deprecated/outdated:**
- pytest.config (deprecated in pytest 5.0): Use `pytest` object directly or fixtures
- `pytest.register_assert_rewrite()` after import: Must call before importing module

## Open Questions

Things that couldn't be fully resolved:

1. **Empty deck during FI purchases**
   - What we know: `DECK.draw(state)` returns -1 when empty (based on deck implementation pattern)
   - What's unclear: Should this be explicitly tested, or is it edge case unlikely enough to defer?
   - Recommendation: Include in parametrized tests since deck depletion is possible in long games

2. **History size limits for long integration tests**
   - What we know: History is list of (state_array, action) tuples, state arrays are ~2000 floats
   - What's unclear: Memory implications for integration tests with 100+ actions
   - Recommendation: Integration tests should focus on critical paths (10-20 actions max), not exhaustive gameplay

## Sources

### Primary (HIGH confidence)
- pytest parametrize documentation: https://docs.pytest.org/en/stable/how-to/parametrize.html
- pytest good practices: https://docs.pytest.org/en/stable/explanation/goodpractices.html
- Existing conftest.py fixtures and helpers (tests/phases/conftest.py)
- Existing test patterns (test_invest.py, test_bid_in_auction.py parametrize usage)
- turn.pyx set_phase() method (line 103)
- driver.pyx history tracking (lines 139, 168, sentinel pattern lines 46-62)
- WRAP_UP implementation (phases/wrap_up.pyx)

### Secondary (MEDIUM confidence)
- [Writing pytest assertion helpers](https://lorepirri.com/pytest-register-assert-rewrite.html) - register_assert_rewrite pattern
- [pytest organize tests best practices](https://pytest-with-eric.com/pytest-best-practices/pytest-organize-tests/) - directory structure guidance
- [pytest integration tests organization](https://damianpiatkowski.com/blog/organizing-automated-tests-python) - separating unit vs integration
- [State-transition testing (Ubuntu Ops)](https://documentation.ubuntu.com/ops/latest/explanation/state-transition-testing/) - atomic state transition testing pattern

### Tertiary (LOW confidence)
- pytest-cython plugin: Not needed for this project (testing via Python interface, not Cython internals)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - pytest is established in codebase, all fixtures already exist
- Architecture: HIGH - patterns verified in existing tests (conftest.py, parametrize usage)
- Pitfalls: HIGH - based on requirements and common testing mistakes in phase-based systems

**Research date:** 2026-01-23
**Valid until:** 60 days (pytest stable, patterns mature, codebase-specific)
