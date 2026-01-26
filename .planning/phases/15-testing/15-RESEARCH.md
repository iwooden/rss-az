# Phase 15: Testing - Research

**Researched:** 2026-01-26
**Domain:** pytest testing for Cython game engine ACQUISITION phase
**Confidence:** HIGH

## Summary

This phase focuses on comprehensive test coverage for the ACQUISITION phase of a Rolling Stock Stars game engine. The domain is well-understood: we're testing an existing Cython implementation with established patterns from prior phases (INVEST, BID_IN_AUCTION, WRAP_UP).

The research examined the existing test infrastructure (conftest.py fixtures, assertion helpers, test organization), the ACQUISITION phase implementation (phases/acquisition.pyx), and prior phase tests to identify patterns. The codebase already has mature testing conventions that should be followed consistently.

Key findings: The existing test infrastructure provides all necessary fixtures and helpers. Test patterns are consistent across phases (class-per-feature, requirement IDs in docstrings). The acquisition.pyx implementation exposes Python wrappers (*_py functions) specifically for testing. No additional libraries or infrastructure needed.

**Primary recommendation:** Follow existing test patterns exactly. Consolidate tests into `tests/phases/test_acquisition.py` using class-per-feature structure and leverage existing conftest.py fixtures/helpers.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | (installed) | Test framework | Already used by all existing tests |
| numpy | (installed) | Array assertions | Required for GameState operations |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest.mark.parametrize | (built-in) | Data-driven tests | Multiple player counts, boundary conditions |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| unittest | pytest | pytest already standard in codebase, no reason to change |
| hypothesis | pytest.parametrize | hypothesis overkill for deterministic game state tests |

**Installation:**
```bash
# No additional installation needed - all dependencies already present
```

## Architecture Patterns

### Recommended Project Structure
```
tests/
├── conftest.py                    # Root fixtures (path setup only)
├── test_integration.py            # Cross-phase integration tests (moved from phases/)
├── test_acquisition.py            # ACQUISITION tests (moved from root)
├── phases/
│   ├── conftest.py                # Shared phase fixtures and helpers
│   ├── test_invest.py             # INVEST phase tests
│   ├── test_bid_in_auction.py     # BID phase tests
│   └── test_wrap_up.py            # WRAP_UP phase tests
```

### Pattern 1: Class-Per-Feature Organization
**What:** Group related tests into classes named after the feature under test
**When to use:** All phase test files
**Example:**
```python
# Source: tests/test_acquisition.py (existing codebase pattern)
class TestOfferGeneration:
    """OFFER-01 through OFFER-05: Offer generation and priority."""

    def test_no_offers_fresh_game(self):
        """No offers when no corps active and FI has no companies."""
        ...

class TestValidation:
    """Validation tests - verify through action handler behavior."""
    ...
```

### Pattern 2: Requirement ID Docstrings
**What:** Include requirement IDs in test method docstrings
**When to use:** Every test method
**Example:**
```python
# Source: tests/phases/test_invest.py
def test_pass_increments_consecutive_passes(self, game_state):
    """INV-01: Pass action increments consecutive_passes counter."""
    ...
```

### Pattern 3: Helper Setup Methods in Test Classes
**What:** Private setup methods for complex state configuration
**When to use:** When multiple tests need similar game state setup
**Example:**
```python
# Source: tests/test_acquisition.py
class TestValidation:
    def _setup_player_private_offer(self, gs, player_id, company_id, corp_id, corp_cash):
        """Setup player private -> corp offer."""
        COMPANIES[company_id].transfer_to_player(gs, player_id)
        CORPS[CORP_NAMES[corp_id]].set_active(gs, True)
        CORPS[CORP_NAMES[corp_id]].set_cash(gs, corp_cash)
        PLAYERS[player_id].set_president_of(gs, corp_id, True)
        setup_acquisition_phase_py(gs)
```

### Pattern 4: Direct Entity Manipulation
**What:** Use entity APIs directly to set up test state
**When to use:** All test setup requiring specific game state
**Example:**
```python
# Source: tests/phases/test_invest.py
corp.set_active(state, True)
corp.set_price_index(state, 10)
corp.set_bank_shares(state, 2)
PLAYERS[0].set_shares(state, 0, 2)
PLAYERS[0].set_president_of(state, 0, True)
```

### Anti-Patterns to Avoid
- **Playing through game actions for setup:** Don't replay many game actions to reach a test state. Directly manipulate entities to create the exact state needed.
- **Testing multiple concerns in one test:** Keep tests focused on one assertion per test when possible.
- **Hardcoding magic numbers without context:** Use constants or comments explaining what values represent.
- **Skipping invariant checks:** Always use `assert_invariants()` after state-modifying operations in integration tests.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Invariant checking | Manual assert statements | `assert_invariants()` from conftest.py | Comprehensive checks for share conservation, cash non-negative, etc. |
| Action application with verification | DRIVER.apply_action + manual checks | `apply_action_and_verify()` from conftest.py | Handles mask validation, status check, and invariant check |
| History tracking | Manual state snapshots | `apply_and_track()` fixture | Returns ApplyTrackResult with history access |
| Finding valid actions | Manual mask iteration | `get_valid_action_mask()` + action layout | Standard pattern in all existing tests |
| Setting up ACQUISITION state | Manual buffer manipulation | `setup_acquisition_phase_py()` | Proper initialization of offer buffer |

**Key insight:** The conftest.py fixtures handle the complex verification logic. Tests should leverage these rather than reimplementing validation.

## Common Pitfalls

### Pitfall 1: Presidency Not Set When Shares Assigned
**What goes wrong:** Setting shares via `set_shares()` doesn't automatically set presidency
**Why it happens:** Share count and presidency are separate state
**How to avoid:** Always call `set_president_of()` after setting shares for president
**Warning signs:** Tests fail with "no offers generated" when offers expected

### Pitfall 2: Offer Generation Filters Invalid Offers
**What goes wrong:** Expecting offers to appear in buffer but they're filtered at generation time
**Why it happens:** `_collect_*_offers` functions filter by cash availability, active status
**How to avoid:** Ensure corp is active, has sufficient cash before calling setup_acquisition_phase_py
**Warning signs:** get_offer_count() returns 0 when offers expected

### Pitfall 3: Company Location State
**What goes wrong:** Company appears owned by wrong entity after transfer
**Why it happens:** transfer_* methods update multiple state fields
**How to avoid:** Use transfer_* methods rather than manually setting location/owner
**Warning signs:** Validation fails on target company location checks

### Pitfall 4: Receivership Corp Has President = -1
**What goes wrong:** Tests expect player action but receivership auto-executes
**Why it happens:** Receivership corps have no president, auto-buy from FI
**How to avoid:** Understand that receivership auto-buy happens in _present_current_offer loop
**Warning signs:** Offer count decreases unexpectedly, no player action presented

### Pitfall 5: Acquisition Zone vs Owned Zone
**What goes wrong:** Checking wrong zone for company after acquisition
**Why it happens:** Companies go to acquisition_companies during phase, merge to owned_companies at phase end
**How to avoid:** Use `has_acquisition_company()` during phase, `owns_company()` after merge
**Warning signs:** Company not found where expected

### Pitfall 6: Insufficient Cash Test Incorrectly Structured
**What goes wrong:** Test expects rejection but offer was never generated
**Why it happens:** Offers filtered at generation time, not at action validation time
**How to avoid:** Verify offer generation behavior, not action rejection for cash-based filtering
**Warning signs:** Test passes vacuously because no offer was presented

## Code Examples

Verified patterns from existing codebase:

### Fresh Game State Setup
```python
# Source: tests/test_acquisition.py
gs = GameState(3)
gs.initialize_game()
```

### Setting Up Offer Scenario
```python
# Source: tests/test_acquisition.py
def _setup_player_private_offer(self, gs, player_id, company_id, corp_id, corp_cash):
    """Setup player private -> corp offer."""
    COMPANIES[company_id].transfer_to_player(gs, player_id)
    CORPS[CORP_NAMES[corp_id]].set_active(gs, True)
    CORPS[CORP_NAMES[corp_id]].set_cash(gs, corp_cash)
    PLAYERS[player_id].set_president_of(gs, corp_id, True)
    setup_acquisition_phase_py(gs)
```

### Verifying Acquisition Action Execution
```python
# Source: tests/test_acquisition.py
result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0)
assert result == 0  # Success

# Verify company in acquisition zone
assert CORPS[CORP_NAMES[0]].has_acquisition_company(gs, 0)
```

### Testing Zone Merging
```python
# Source: tests/test_acquisition.py
# Put company in acquisition zone
company.transfer_to_corp_acquisition(gs, 0)
assert corp.has_acquisition_company(gs, 0)
assert not corp.owns_company(gs, 0)

# Trigger merge
merge_acquisition_zones_py(gs)

# Company moved from acquisition to owned
assert not corp.has_acquisition_company(gs, 0)
assert corp.owns_company(gs, 0)
```

### Receivership Auto-Buy Test
```python
# Source: tests/test_acquisition.py
COMPANIES[0].transfer_to_fi(gs)
corp.set_active(gs, True)
corp.set_cash(gs, 50000)
corp.set_in_receivership(gs, True)

setup_acquisition_phase_py(gs)

# Verify auto-buy executed
assert corp.has_acquisition_company(gs, 0)
```

### Using apply_and_track for History Verification
```python
# Source: tests/phases/test_invest.py
result = apply_and_track(state, pass_idx)
assert len(result.history) >= 3
action_values = [entry[1] for entry in result.history]
assert -100 in action_values  # WRAP_UP sentinel
```

### Parametrized Tests for Multiple Player Counts
```python
# Source: tests/phases/test_invest.py
@pytest.mark.parametrize("num_players", [3, 4, 5, 6])
def test_pass_works_all_player_counts(self, num_players):
    state = GameState(num_players=num_players)
    state.initialize_game(seed=42)
    ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual state inspection | Entity methods | Project design | All state access through entities |
| Direct buffer manipulation | Python wrappers (*_py) | Phase 11+ | Tests use Python-exposed functions |
| Test-specific fixtures | Shared conftest.py helpers | Project maturation | Consistent invariant checking |

**Deprecated/outdated:**
- None identified - testing patterns are stable

## Open Questions

Things that couldn't be fully resolved:

1. **Exact test class order in file**
   - What we know: CONTEXT.md leaves this to Claude's discretion
   - What's unclear: Whether to order by requirements or by logical flow
   - Recommendation: Order by logical flow (setup -> actions -> validation -> edge cases)

2. **Number of tests per validation rule**
   - What we know: CONTEXT.md says "one test per rule PLUS boundary conditions"
   - What's unclear: Exact boundary conditions for each rule
   - Recommendation: Include at minimum: valid boundary, just-below-valid, just-above-valid

## Sources

### Primary (HIGH confidence)
- tests/test_acquisition.py - Existing ACQUISITION tests (template)
- tests/phases/conftest.py - Shared fixtures and helpers
- tests/phases/test_invest.py - INVEST phase test patterns
- tests/phases/test_wrap_up.py - WRAP_UP phase test patterns
- phases/acquisition.pyx - Implementation under test

### Secondary (MEDIUM confidence)
- tests/phases/test_integration.py - Integration test patterns
- tests/phases/test_bid_in_auction.py - BID phase test patterns

### Tertiary (LOW confidence)
- None - all sources are primary codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using existing codebase dependencies only
- Architecture: HIGH - Patterns directly from existing test files
- Pitfalls: HIGH - Derived from implementation analysis and existing test patterns

**Research date:** 2026-01-26
**Valid until:** Indefinite (testing patterns are project-internal standards)
