# Phase 6: Integration & Tests - Research

**Researched:** 2026-01-21
**Domain:** pytest test organization, game state verification, Cython module testing
**Confidence:** HIGH

## Summary

This phase focuses on reorganizing and expanding test coverage for the INVEST and BID_IN_AUCTION phases. The existing test patterns established in phases 02-05 provide a solid foundation with parametrized fixtures, phase-specific state setup, and action mask verification. The decisions from CONTEXT.md are well-defined: tests go in `tests/phases/` with shared helpers in conftest.py, exhaustive action mask validation before and after every action, and specific edge case coverage for bankruptcy, presidency changes, and receivership.

The existing codebase already has substantial test coverage in `tests/test_invest.py`, `tests/test_bid.py`, and `tests/test_share_trading.py`. These need to be migrated to `tests/phases/` and supplemented with integration tests for full auction flows, state consistency invariants, and the specific edge cases identified in the discussion phase.

**Primary recommendation:** Migrate existing tests to `tests/phases/` directory structure, add shared invariant/mask assertion helpers to conftest.py, and systematically add tests for each rule from RULES.md that touches INVEST/BID phases.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | 7.x+ | Test framework | Already in use, parametrization support, fixtures |
| numpy | 1.x | Array comparison | Required for action mask testing |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest.mark.parametrize | built-in | Multi-player testing | Test 3 and 6 player counts (boundaries) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pytest | unittest | pytest has better fixtures and parametrization |
| Manual assertions | hypothesis | Hypothesis adds complexity for deterministic game rules |

**Installation:**
Already installed in project via existing test infrastructure.

## Architecture Patterns

### Recommended Project Structure
```
tests/
  conftest.py              # sys.path setup (existing)
  phases/
    __init__.py            # Empty (make importable)
    conftest.py            # Shared fixtures + assertion helpers
    test_invest.py         # INVEST phase unit + integration tests
    test_bid_in_auction.py # BID phase unit + integration tests
  test_init.py             # Existing game init tests (keep)
  test_driver.py           # Existing driver tests (keep)
```

### Pattern 1: Shared Assertion Helpers in conftest.py
**What:** Centralized helper functions for action mask validation and invariant checking
**When to use:** Every test that modifies game state
**Example:**
```python
# tests/phases/conftest.py

def assert_valid_mask(state, expected_actions, msg=""):
    """
    Assert that the action mask matches expected valid actions.

    Args:
        state: GameState to check
        expected_actions: Set of action indices that should be valid, or None for any
        msg: Additional context for assertion failure
    """
    mask = get_valid_action_mask(state)

    if expected_actions is not None:
        actual_valid = set(i for i in range(len(mask)) if mask[i] == 1.0)
        assert actual_valid == expected_actions, f"{msg}\nExpected: {expected_actions}\nActual: {actual_valid}"
    else:
        # Just verify at least one valid action exists
        assert np.sum(mask) > 0, f"{msg}\nNo valid actions in mask"


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
    num_players = state.get_num_players()

    # Share conservation
    for corp_id in range(8):
        if state.is_corp_active(corp_id):
            total = state.get_corp_unissued_shares(corp_id) + state.get_corp_bank_shares(corp_id)
            for p in range(num_players):
                total += PLAYERS[p].get_shares(state, corp_id)
            expected = get_corp_share_count(corp_id)
            assert total == expected, f"{msg}\nCorp {corp_id} share count: {total} != {expected}"

    # Cash non-negative
    for p in range(num_players):
        cash = PLAYERS[p].get_cash(state)
        assert cash >= 0, f"{msg}\nPlayer {p} cash negative: {cash}"

    for corp_id in range(8):
        if state.is_corp_active(corp_id):
            cash = CORPS[CORP_NAMES[corp_id]].get_cash(state)
            assert cash >= 0, f"{msg}\nCorp {corp_id} cash negative: {cash}"


def apply_action_and_verify(state, action_idx):
    """
    Apply action and verify invariants + mask validity.

    Returns the result status from DRIVER.apply_action.
    """
    result = DRIVER.apply_action(state, action_idx)
    assert result == STATUS_OK, f"Action {action_idx} failed with status {result}"
    assert_invariants(state, f"After action {action_idx}")
    assert np.sum(get_valid_action_mask(state)) > 0, f"No valid actions after {action_idx}"
    return result
```

### Pattern 2: Phase-Specific Fixtures
**What:** Fixtures that return state in a specific phase with controlled setup
**When to use:** Testing phase-specific behavior
**Example:**
```python
# tests/phases/conftest.py

@pytest.fixture
def game_state():
    """Base initialized game state in INVEST phase."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)
    assert state.get_phase() == GamePhases.PHASE_INVEST
    return state


@pytest.fixture
def invest_state(game_state):
    """Alias for clarity - game starts in INVEST."""
    return game_state


@pytest.fixture
def bid_state(game_state):
    """State with active auction in BID_IN_AUCTION phase."""
    # Find and apply first valid auction action
    mask = get_valid_action_mask(game_state)
    layout = get_action_layout(3)
    for i in range(layout['auction_base'], layout['buy_share_base']):
        if mask[i] == 1.0:
            DRIVER.apply_action(game_state, i)
            break
    assert game_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION
    return game_state


@pytest.fixture
def trade_state():
    """State with active corp for buy/sell testing."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # Manually configure corp for trading (as in existing tests)
    corp = CORPS[CORP_NAMES[0]]
    corp.set_active(state, True)
    corp.set_price_index(state, 10)
    corp.set_bank_shares(state, 3)
    corp.set_issued_shares(state, 4)

    PLAYERS[0].set_shares(state, 0, 2)
    PLAYERS[0].set_cash(state, 100)
    PLAYERS[0].set_president_of(state, 0, True)

    MARKET.set_space_available(state, 10, False)

    return state


@pytest.fixture
def bankruptcy_state():
    """State where one sell triggers bankruptcy."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    corp = CORPS[CORP_NAMES[0]]
    corp.set_active(state, True)
    corp.set_price_index(state, 1)  # One sell -> index 0 -> bankruptcy
    corp.set_bank_shares(state, 2)
    corp.set_issued_shares(state, 4)  # bank(2) + player(2) = 4

    COMPANIES[0].transfer_to_corp(state, 0)
    corp.set_owns_company(state, 0, True)

    PLAYERS[0].set_shares(state, 0, 2)
    PLAYERS[0].set_president_of(state, 0, True)
    PLAYERS[0].set_cash(state, 100)

    MARKET.set_space_available(state, 1, False)

    return state
```

### Pattern 3: Parametrized Player Count Tests
**What:** Test boundary player counts (3 and 6) using parametrize
**When to use:** Any test that could differ by player count
**Example:**
```python
@pytest.mark.parametrize("num_players", [3, 6])
def test_wrap_up_triggers_at_correct_pass_count(num_players):
    """WRAP_UP triggers after exactly num_players passes."""
    state = GameState(num_players=num_players)
    state.initialize_game(seed=42)
    layout = get_action_layout(num_players)

    for _ in range(num_players):
        result = DRIVER.apply_action(state, layout['pass_invest'])
        assert result == STATUS_OK

    assert state.get_phase() == GamePhases.PHASE_WRAP_UP
```

### Anti-Patterns to Avoid
- **Testing implementation details:** Test behavior, not internal state layout
- **Hardcoded action indices:** Always use `get_action_layout()` to compute indices
- **Missing mask verification:** Every test should verify mask before AND after actions
- **Skipping invariant checks:** State consistency should be verified after every action

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Action mask comparison | Custom loops | numpy.testing.assert_array_equal | Handles edge cases |
| Finding valid actions | Manual iteration | `[i for i in range(len(mask)) if mask[i] == 1.0]` | Idiomatic |
| Player iteration | Hardcoded range(3) | `range(state.get_num_players())` | Dynamic player counts |
| Corp lookup | Index magic numbers | `CORPS[CORP_NAMES[corp_id]]` | Existing pattern |

**Key insight:** The existing codebase has well-established patterns for entity access and action layout. Follow them.

## Common Pitfalls

### Pitfall 1: Fixture State Contamination
**What goes wrong:** Test modifies fixture state, affecting subsequent tests
**Why it happens:** pytest fixtures are function-scoped by default but state objects are mutable
**How to avoid:** Each test should work with a fresh state from fixture (already correct pattern)
**Warning signs:** Tests pass individually but fail when run together

### Pitfall 2: Action Index Hardcoding
**What goes wrong:** Test uses magic numbers like `DRIVER.apply_action(state, 137)`
**Why it happens:** Developer copies index from debug output
**How to avoid:** Always compute indices via `get_action_layout(num_players)`
**Warning signs:** Tests fail when player count changes or action layout is modified

### Pitfall 3: Missing Pre-Action Mask Check
**What goes wrong:** Test applies action without verifying it's valid first
**Why it happens:** Developer assumes action is valid based on fixture setup
**How to avoid:** `assert mask[action_idx] == 1.0` before every `apply_action`
**Warning signs:** Test fails with STATUS_INVALID but assertion doesn't catch why

### Pitfall 4: Incomplete Bankruptcy Testing
**What goes wrong:** Bankruptcy test doesn't verify all cleanup steps
**Why it happens:** Complex multi-step procedure, easy to miss one check
**How to avoid:** Use the bankruptcy fixture pattern (issued_shares = bank_shares + player_shares)
**Warning signs:** Subsequent IPO tests fail because state wasn't fully reset

### Pitfall 5: Round-Trip Counter Not Reset
**What goes wrong:** Test assumes round-trip counters start at 0 but fixture has prior state
**Why it happens:** Trade state fixture doesn't explicitly clear counters
**How to avoid:** Explicitly clear or verify round-trip state in tests that depend on it
**Warning signs:** Buy/sell actions mysteriously blocked

## Code Examples

Verified patterns from existing codebase:

### Full Auction Flow Test
```python
# Source: tests/test_bid.py TestFullAuctionCycle
def test_complete_auction_cycle(self):
    """Test full flow: INVEST -> auction -> BID -> resolution -> INVEST."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # Start in INVEST
    assert state.get_phase() == GamePhases.PHASE_INVEST

    # Start auction
    mask = get_valid_action_mask(state)
    layout = get_action_layout(3)
    for i in range(layout['auction_base'], layout['buy_share_base']):
        if mask[i] == 1.0:
            DRIVER.apply_action(state, i)
            break

    # Now in BID phase
    assert state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION

    # Two players leave
    for _ in range(2):
        if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            DRIVER.apply_action(state, layout['leave_auction'])

    # Back to INVEST
    assert state.get_phase() == GamePhases.PHASE_INVEST
```

### Presidency Transfer Test
```python
# Source: tests/test_share_trading.py TestPresidency
def test_presidency_transfers_to_most_shares(self, trade_state):
    """Player with most shares becomes president."""
    # Player 0 has 2 shares, is president
    # Give player 1 more shares
    PLAYERS[1].set_shares(trade_state, 0, 3)

    # Sell a share (triggers presidency check)
    layout = get_action_layout(3)
    sell_idx = layout['sell_share_base'] + 0
    DRIVER.apply_action(trade_state, sell_idx)

    # Player 1 should now be president (3 shares > 1 share)
    assert PLAYERS[1].is_president_of(trade_state, 0)
    assert not PLAYERS[0].is_president_of(trade_state, 0)
```

### Invariant Assertion Example
```python
# Recommended pattern for integration tests
def test_auction_maintains_invariants(game_state):
    """Full auction flow maintains all game invariants."""
    layout = get_action_layout(3)

    # Initial state
    assert_invariants(game_state, "Initial state")

    # Start auction
    mask = get_valid_action_mask(game_state)
    for i in range(layout['auction_base'], layout['buy_share_base']):
        if mask[i] == 1.0:
            apply_action_and_verify(game_state, i)
            break

    # Complete auction via leaves
    for _ in range(2):
        if game_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            apply_action_and_verify(game_state, layout['leave_auction'])

    # Final check
    assert_invariants(game_state, "After auction complete")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Tests scattered in tests/ root | Organized by phase in tests/phases/ | Phase 6 | Better maintainability |
| Manual state verification | Helper functions in conftest | Phase 6 | Consistent checking |

**Deprecated/outdated:**
- None - existing test patterns are current and should be extended

## Open Questions

Things that couldn't be fully resolved:

1. **Exact invariant set for different phases**
   - What we know: Share conservation, cash non-negative, net worth non-negative
   - What's unclear: Are there phase-specific invariants (e.g., auction state cleanup)?
   - Recommendation: Start with the four invariants from CONTEXT.md, add more as edge cases reveal them

2. **Test file migration vs. duplication**
   - What we know: Decision is to move existing tests to tests/phases/
   - What's unclear: Should existing test imports remain for backward compatibility?
   - Recommendation: Move files cleanly, update any imports, don't leave stubs

## Sources

### Primary (HIGH confidence)
- Existing test files in repository (test_invest.py, test_bid.py, test_share_trading.py)
- Phase implementation files (phases/invest.pyx, phases/bid.pyx)
- CONTEXT.md decisions from discuss-phase session
- RULES.md game rules specification

### Secondary (MEDIUM confidence)
- STATE.md accumulated patterns from prior phases

### Tertiary (LOW confidence)
- None - all findings based on existing codebase and documented decisions

## Metadata

**Confidence breakdown:**
- Test organization: HIGH - Directly specified in CONTEXT.md decisions
- Architecture patterns: HIGH - Derived from existing test patterns in codebase
- Invariant helpers: HIGH - Specified in CONTEXT.md with specific requirements
- Pitfalls: MEDIUM - Based on common testing issues and codebase analysis

**Research date:** 2026-01-21
**Valid until:** No expiration - patterns are codebase-specific, not library-version-dependent
