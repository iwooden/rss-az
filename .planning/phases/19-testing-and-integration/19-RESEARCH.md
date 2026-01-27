# Phase 19: Testing and Integration - Research

**Researched:** 2026-01-27
**Domain:** pytest testing patterns for Cython game engine, test organization and coverage
**Confidence:** HIGH

## Summary

Phase 19 is a testing-only phase to consolidate and verify comprehensive test coverage for the CLOSING phase (Phases 16-18, CLO-01 through CLO-16). All 16 CLO requirements are already implemented and marked complete. This phase creates validation tests without implementing new functionality.

The codebase has established testing patterns:
- `tests/phases/test_closing.py` exists with ~90 tests covering CLO-01 through CLO-16
- `tests/test_integration.py` has cross-phase tests through ACQUISITION
- `tests/phases/conftest.py` provides fixtures and assertion helpers (`apply_action_and_verify`, `apply_and_track`, `assert_invariants`)
- `@pytest.mark.parametrize` used consistently for player count variations
- Docstrings follow `"""CLO-XX: Brief description."""` pattern for requirement traceability

**Primary recommendation:** Consolidate existing CLOSING tests into `tests/phases/test_closing.py`, extend `tests/test_integration.py` with ACQUISITION -> CLOSING -> INCOME flow tests, and add comprehensive edge case coverage using established patterns.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | 7.4.4 | Test framework | Already in use across all test files |
| numpy | - | State array operations, array assertions | Used for mask comparisons and state verification |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tests/phases/conftest.py | N/A | Shared fixtures, assertion helpers | All phase tests use these |
| pytest.mark.parametrize | N/A | Test parameterization | Player count variations, boundary values |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Flat test functions | Test classes | Classes group related tests (existing pattern uses both) |
| Inline setup | conftest fixtures | Fixtures preferred for reuse (per CONTEXT.md) |
| Random data | Scripted sequences | Scripted sequences for determinism (per CONTEXT.md) |

## Architecture Patterns

### Recommended Test File Structure
```
tests/
    conftest.py                   # Root-level re-exports from phases/conftest.py
    test_integration.py           # Cross-phase integration tests (extend here)
    phases/
        conftest.py               # Shared fixtures and helpers
        test_closing.py           # CLOSING phase tests (consolidate here)
```

### Pattern 1: Requirement Traceability via Docstrings
**What:** Document which CLO requirement each test validates
**When to use:** Every test function for CLOSING phase
**Example:**
```python
# Source: tests/phases/test_closing.py existing pattern
def test_fi_closes_negative_income_company(self):
    """CLO-01: FI closes company with negative adjusted income."""
    # test implementation
```

### Pattern 2: Test Class Organization
**What:** Group related tests in classes by requirement or feature area
**When to use:** When multiple tests target same functionality
**Example:**
```python
# Source: tests/phases/test_closing.py lines 25-114
class TestFIAutoClose:
    """CLO-01: FI closes companies where income - CoO < 0."""

    def test_fi_closes_negative_income_company(self):
        """FI closes company with negative adjusted income."""

    def test_fi_keeps_zero_income_company(self):
        """FI does NOT close company with exactly zero adjusted income."""

    def test_fi_keeps_positive_income_company(self):
        """FI keeps company with positive adjusted income."""
```

### Pattern 3: Parameterized Player Count Tests
**What:** Test behavior across all supported player counts
**When to use:** When behavior varies by player count or for robustness
**Example:**
```python
# Source: tests/test_integration.py lines 99-117
@pytest.mark.parametrize("num_players", [3, 6])
def test_wrap_up_transition_maintains_invariants(self, num_players):
    """Phase transition through WRAP_UP maintains invariants."""
    state = GameState(num_players=num_players)
    state.initialize_game(seed=42)
    # ...
```

### Pattern 4: Integration Test with Scripted Actions
**What:** Deterministic action sequences testing full phase flow
**When to use:** Cross-phase integration tests (ACQUISITION -> CLOSING -> INCOME)
**Example:**
```python
# Source: tests/test_integration.py lines 350-394
def test_acquisition_accept_maintains_invariants(self):
    """Accept action in ACQUISITION maintains invariants."""
    from tests.phases.conftest import apply_action_and_verify, assert_invariants
    # Setup state with valid acquisition offer
    # Apply action and verify invariants
    # Check phase transition
```

### Pattern 5: Edge Case Tests with Explicit Assertions
**What:** Boundary conditions tested with specific expected outcomes
**When to use:** Empty offers, all-pass scenarios, multi-close cascades
**Example:**
```python
# Source: tests/phases/test_closing.py lines 375-390
def test_zero_income_not_offered(self, closing_offer_state):
    """CLO-05: Companies with exactly zero adjusted income are NOT offered."""
    gs = closing_offer_state
    TURN.set_coo_level(gs, 4)  # Company 2 has income $2, CoO $2 -> adjusted = $0
    PLAYERS[0].set_owns_company(gs, 2, True)
    generate_close_offers_py(gs)
    assert get_close_offer_count_py(gs) == 0  # Zero income = NOT offered
```

### Anti-Patterns to Avoid
- **Random action selection in integration tests:** Use scripted sequences for determinism (per CONTEXT.md)
- **Creating new test files per requirement:** Consolidate in test_closing.py (per CONTEXT.md)
- **Duplicate setup code across tests:** Use fixtures from conftest.py
- **Missing negative tests:** Always include invalid action attempts, wrong phase errors

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| State invariant checking | Manual assertions | `assert_invariants(state)` from conftest | Comprehensive checks (share conservation, cash >= 0, etc.) |
| Action application with verification | `DRIVER.apply_action()` directly | `apply_action_and_verify(state, action)` | Validates mask, checks result, verifies invariants |
| Tracking action history | Manual list building | `apply_and_track(state, action)` | Returns `ApplyTrackResult` with history access |
| Creating test game states | Inline initialization | `game_state`, `closing_offer_state` fixtures | Consistent setup, reusable |

**Key insight:** The test infrastructure in conftest.py is comprehensive. Use existing fixtures and helpers for consistency and reliability.

## Common Pitfalls

### Pitfall 1: Testing Implementation Instead of Behavior
**What goes wrong:** Tests break when internal implementation changes
**Why it happens:** Assertions check intermediate state rather than final outcomes
**How to avoid:** Test observable behavior: state changes, phase transitions, action results
**Warning signs:** Tests fail after refactoring that preserves behavior

### Pitfall 2: Missing Phase Transition Tests
**What goes wrong:** Phase flows work in isolation but fail in sequence
**Why it happens:** Unit tests pass but integration tests not written
**How to avoid:** Include full ACQUISITION -> CLOSING -> INCOME flow tests
**Warning signs:** Bugs discovered in multi-phase gameplay, not in unit tests

### Pitfall 3: Insufficient Edge Case Coverage
**What goes wrong:** Empty offers, all-pass, multi-close scenarios untested
**Why it happens:** Tests focus on happy path
**How to avoid:** Explicitly test: no offers, single offer, max offers, all pass, corp last-company
**Warning signs:** Production bugs in rare but valid game states

### Pitfall 4: Player Count Blindness
**What goes wrong:** Tests pass for 3 players, fail for 6
**Why it happens:** Hard-coded player count assumptions
**How to avoid:** Use `@pytest.mark.parametrize("num_players", [3, 6])` for critical paths
**Warning signs:** Tests only use 3-player games

### Pitfall 5: State Pollution Between Tests
**What goes wrong:** Test passes in isolation, fails when run with others
**Why it happens:** Shared mutable state not reset
**How to avoid:** Use fixtures that create fresh state per test (`game_state` fixture)
**Warning signs:** Inconsistent test results based on execution order

### Pitfall 6: Ignoring Existing Test Coverage
**What goes wrong:** Duplicate tests, wasted effort
**Why it happens:** Not checking what Phase 16-18 already added to test_closing.py
**How to avoid:** Audit existing test_closing.py before adding new tests
**Warning signs:** Multiple tests for same requirement with same assertions

## Code Examples

Verified patterns from official sources:

### Using Existing Fixtures
```python
# Source: tests/phases/conftest.py lines 167-181
@pytest.fixture
def game_state():
    """Base initialized game state in INVEST phase."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)
    assert state.get_phase() == GamePhases.PHASE_INVEST
    return state

@pytest.fixture
def closing_offer_state():
    """Create game state with companies ready for close offers."""
    gs = GameState(num_players=3)
    gs.initialize_game(seed=42)
    TURN.set_coo_level(gs, 6)  # High CoO for negative income
    return gs
```

### Integration Test Pattern
```python
# Source: tests/test_integration.py lines 291-320
def test_acquisition_to_invest_new_turn(self):
    """ACQUISITION phase completes and transitions to CLOSING then INVEST with new turn."""
    from tests.phases.conftest import assert_invariants
    from phases.acquisition import transition_to_closing_py
    from phases.closing import apply_closing_auto_py

    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
    initial_turn = TURN.get_turn_number(state)

    assert_invariants(state, "Before transition")

    transition_to_closing_py(state)
    assert state.get_phase() == GamePhases.PHASE_CLOSING

    apply_closing_auto_py(state)
    assert state.get_phase() == GamePhases.PHASE_INVEST
    assert TURN.get_turn_number(state) == initial_turn + 1
    assert_invariants(state, "After CLOSING to INVEST")
```

### Negative Test Pattern
```python
# Source: tests/phases/test_closing.py lines 486-509
class TestOfferValidation:
    """Tests for offer validation (CLO-09, CLO-10)."""

    def test_corp_last_company_rule(self, closing_offer_state):
        """CLO-09: Corp closing offer invalid if corp would have 0 companies."""
        gs = closing_offer_state
        CORPS[1].set_active(gs, True)
        CORPS[1].set_in_receivership(gs, False)
        PLAYERS[0].set_president_of(gs, 1, True)
        # Corp owns ONLY company 3 (last company)
        CORPS[1].set_owns_company(gs, 3, True)
        # Validation happens at presentation time
```

### Parametrized Edge Case Tests
```python
# Source: tests/phases/test_invest.py lines 1183-1194
@pytest.mark.parametrize("num_players,seed", [
    (3, 42),
    (6, 123),
])
def test_full_turn_with_trading(self, num_players, seed):
    """Full turn with trading maintains invariants."""
    state = GameState(num_players=num_players)
    state.initialize_game(seed=seed)
    # test implementation
```

### Assertion Helper Usage
```python
# Source: tests/phases/conftest.py lines 68-113
def assert_invariants(state, msg=""):
    """
    Assert game state invariants are maintained.

    Checks:
    - Total shares per corp = unissued + bank + all players
    - Player cash >= 0
    - Corp cash >= 0
    - Net worths >= 0
    - Auction row size <= num_players
    """
    # Implementation checks share conservation, cash non-negative, etc.
```

## Existing Test Coverage Audit

**tests/phases/test_closing.py current coverage:**

| Requirement | Test Class | Test Count | Status |
|-------------|------------|------------|--------|
| CLO-01 | TestFIAutoClose | 4 | Covered |
| CLO-02 | TestReceivershipAutoClose | 4 | Covered |
| CLO-03 | TestReceivershipAutoClose | (included above) | Covered |
| CLO-04 | TestHighestFaceValueProtection | 2 | Covered |
| CLO-05 | TestOfferGeneration | 2 | Covered |
| CLO-06 | TestOfferGeneration | 1 | Covered |
| CLO-07 | TestOfferGeneration | 1 | Covered |
| CLO-08 | TestOfferGeneration | 2 | Covered |
| CLO-09 | TestOfferValidation | 2 | Covered |
| CLO-10 | TestOfferValidation | 1 | Covered |
| CLO-11 | TestCloseActions | 1 | Covered |
| CLO-12 | TestCloseActions | 1 | Covered |
| CLO-13 | TestCloseActions | 2 | Covered |
| CLO-14 | TestMandatoryClose | 5 | Covered |
| CLO-15 | TestMandatoryClose | 2 | Covered |
| CLO-16 | TestClosingPhaseTransition | 2 | Covered |

**Total existing tests:** ~32 tests in test_closing.py

**Gaps to fill:**
1. Integration tests for full ACQUISITION -> CLOSING -> INCOME flow
2. Edge cases: empty offers scenario, all-pass scenario, multi-close cascade
3. Player count parameterization for critical paths
4. Regression tests ensuring existing tests still pass

## Test Recommendations by Category

### Unit Tests (Already Covered)
- All 16 CLO requirements have unit test coverage
- Focus on edge cases and negative tests if gaps found

### Integration Tests (To Add)
- Full driver loop: ACQUISITION -> CLOSING -> INCOME
- Multi-turn cycles with CLOSING companies
- State transitions with offers accepted/passed

### Edge Case Tests (To Add)
| Scenario | What to Test | Expected Behavior |
|----------|--------------|-------------------|
| No close offers | All companies have positive income | Direct transition to INCOME |
| All pass | All offers declined | Mandatory close triggered, transition to INCOME |
| Multi-close cascade | Player closes multiple companies | Each close triggers JS bonus, eventual transition |
| Corp last-company dynamic | First close makes second invalid | Second offer skipped automatically |

### Negative Tests (Verify Coverage)
| Test | What to Verify | Existing? |
|------|----------------|-----------|
| Invalid close in wrong phase | Action rejected | Needs verification |
| Close on non-existent offer | Action rejected | Needs verification |
| Close already-removed company | Offer skipped | Covered in validation tests |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-phase test files | Consolidated test_closing.py | Phase 16-18 | Single source for CLOSING tests |
| Manual invariant checks | assert_invariants() helper | Phase 3 | Consistent, comprehensive validation |
| Inline state setup | conftest.py fixtures | Phase 1 | Reusable, DRY test setup |

**Deprecated/outdated:**
- Tests in separate files per requirement - consolidate in test_closing.py

## Open Questions

Things that couldn't be fully resolved:

1. **Test Completeness Threshold**
   - What we know: All 16 CLO requirements have tests
   - What's unclear: How many edge cases are "enough"
   - Recommendation: At minimum: empty offers, all-pass, multi-close, player counts 3 and 6

2. **Integration Test Depth**
   - What we know: test_integration.py has tests through ACQUISITION
   - What's unclear: How many CLOSING integration scenarios needed
   - Recommendation: At least 3 scenarios: no offers, offers with accepts, offers with passes + mandatory close

3. **Existing Test Duplication**
   - What we know: Phases 16-18 each added tests
   - What's unclear: Whether any tests overlap or contradict
   - Recommendation: Audit existing tests before adding new ones

## Sources

### Primary (HIGH confidence)
- `tests/phases/test_closing.py` - Existing CLOSING phase tests (907 lines)
- `tests/test_integration.py` - Existing integration tests (531 lines)
- `tests/phases/conftest.py` - Fixtures and helpers (272 lines)
- `tests/conftest.py` - Root-level fixture re-exports (34 lines)
- `19-CONTEXT.md` - User decisions on test organization

### Secondary (MEDIUM confidence)
- pytest 7.4.4 documentation - Current version patterns
- Existing test files (test_invest.py, test_acquisition.py) - Codebase testing conventions

### Tertiary (LOW confidence)
- None - all patterns verified in codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - pytest 7.4.4 verified, patterns from existing tests
- Architecture: HIGH - Direct application of existing test patterns
- Pitfalls: HIGH - Based on codebase patterns and testing best practices

**Research date:** 2026-01-27
**Valid until:** N/A (project-specific patterns don't expire)
