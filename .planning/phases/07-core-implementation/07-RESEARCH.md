# Phase 7: Core Implementation - Research

**Researched:** 2026-01-21
**Domain:** Cython game driver auto-application of forced actions
**Confidence:** HIGH

## Summary

This phase implements automatic application of forced actions in the GameDriver. When exactly one legal action exists, the driver auto-applies it iteratively until 2+ choices are available or the game ends. This ensures the neural network only sees states with real decisions to make.

The implementation builds on existing infrastructure: the `get_forced_action()` function in `core/actions.pyx` (lines 520-567) already detects forced actions, and the `GameDriver.apply_action()` method already validates and dispatches actions. The core work is: (1) creating helper infrastructure (`ForcedActionResult` struct, `_check_forced_action()`, `_apply_single_action()`), (2) implementing the iterative auto-apply loop with proper termination guards, and (3) adding optional history tracking for test observability.

**Primary recommendation:** Implement an iterative while-loop inside `apply_action()` that calls `_check_forced_action()` after each action. Exit when 0 actions (error), 2+ actions (choice needed), or GAME_OVER. Use iteration limit of 100 as safety guard.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Cython | 3.x | C-extension compiler | Already used throughout codebase |
| NumPy | 2.x | Array operations | Required for float32 action mask |
| Python | 3.12 | Runtime | Project standard |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | 8.x | Testing | All test files |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| while loop | recursion | Recursion risks stack overflow on long forced chains; iterative is safer |
| struct return | tuple return | Struct allows `count` field to distinguish 0 from 2+ actions; existing tuple API cannot |
| Exception | Status code | Status codes (existing pattern) have zero overhead vs exception handling |

**No new dependencies required** - existing Cython/NumPy stack is sufficient.

## Architecture Patterns

### Recommended Project Structure
```
core/
  driver.pyx          # Modified: add helper cdef functions, modify apply_action()
  driver.pxd          # Modified: add ForcedActionResult struct, declare helpers
src/
  exceptions.py       # NEW: custom exceptions module (per CONTEXT.md decision)
```

### Pattern 1: ForcedActionResult Struct
**What:** C struct with `count` (int) and `action_idx` (int) fields for returning forced action check results.
**When to use:** Internal helper return value from `_check_forced_action()`.
**Example:**
```cython
# Source: core/driver.pxd (new)
cdef struct ForcedActionResult:
    int count       # 0, 1, or 2 (stop counting at 2 for early exit)
    int action_idx  # -1 if count != 1, otherwise the single valid action index
```

**Rationale:** The existing `get_forced_action()` returns `(-1, False)` for both 0 and 2+ actions. We need to distinguish these cases to properly handle the zero-action error condition vs the multi-action choice condition.

### Pattern 2: Early-Exit Counting Loop
**What:** Count valid actions in mask, stop immediately when count reaches 2.
**When to use:** In `_check_forced_action()` to detect forced vs choice states.
**Example:**
```cython
# Source: Existing pattern from core/actions.pyx lines 554-563
cdef ForcedActionResult _check_forced_action(GameState state) noexcept:
    """Check legal action count and find single action if forced."""
    cdef ForcedActionResult result
    cdef object mask = get_valid_action_mask(state)
    cdef int total = mask.shape[0]
    cdef float* mask_ptr = <float*>cnp.PyArray_DATA(mask)
    cdef int i

    result.action_idx = -1
    result.count = 0

    for i in range(total):
        if mask_ptr[i] == 1.0:
            result.count += 1
            if result.count == 1:
                result.action_idx = i
            elif result.count == 2:
                result.action_idx = -1  # Not forced
                return result  # Early exit: no need to count higher

    return result
```

### Pattern 3: Iterative Auto-Apply Loop with Guards
**What:** While loop that applies forced actions until choice needed or game ends.
**When to use:** Main `apply_action()` method after applying the user's action.
**Example:**
```cython
# Source: Based on existing apply_action() structure in core/driver.pyx
DEF MAX_FORCED_ITERATIONS = 100

cpdef int apply_action(self, GameState state, int action_idx, object history=None):
    cdef int result, iterations
    cdef ForcedActionResult forced

    # Apply the user's action
    result = self._apply_single_action(state, action_idx, history)
    if result != STATUS_OK:
        return result
    if state.get_phase() == PHASE_GAME_OVER:
        return STATUS_GAME_OVER

    # Auto-apply forced actions
    iterations = 0
    while iterations < MAX_FORCED_ITERATIONS:
        forced = _check_forced_action(state)

        if forced.count == 0:
            # Error: no legal actions outside GAME_OVER
            raise ZeroLegalActionsError("Zero legal actions in non-terminal state")

        if forced.count >= 2:
            # Choice needed - return to caller
            return STATUS_OK

        # Exactly 1 action - auto-apply it
        result = self._apply_single_action(state, forced.action_idx, history)
        if result != STATUS_OK:
            return result
        if state.get_phase() == PHASE_GAME_OVER:
            return STATUS_GAME_OVER

        iterations += 1

    # Exceeded iteration limit
    raise ForcedActionLoopError(f"Forced action loop exceeded {MAX_FORCED_ITERATIONS} iterations")
```

### Pattern 4: Optional History Tracking
**What:** Optional list parameter that collects (state.copy(), action) tuples.
**When to use:** Test observability to see all actions including auto-applied ones.
**Example:**
```cython
# Source: Based on CONTEXT.md history API decision
cdef int _apply_single_action(GameDriver self, GameState state, int action_idx, object history):
    """Apply one action without auto-continuation."""
    # Append to history if provided
    if history is not None:
        history.append((state._array.copy(), action_idx))

    # ... existing apply logic ...
```

**Key points:**
- `state._array.copy()` creates independent numpy array snapshot
- Include ALL actions: user's initial + all auto-applied
- When `history is None`, no overhead (no copy, no list append)

### Pattern 5: Custom Exceptions in Pure Python Module
**What:** Define exceptions in separate `src/exceptions.py` module.
**When to use:** Error signaling from Cython to Python.
**Example:**
```python
# Source: src/exceptions.py (new file per CONTEXT.md decision)

class ForcedActionLoopError(RuntimeError):
    """Raised when forced action loop exceeds iteration limit."""
    pass

class ZeroLegalActionsError(RuntimeError):
    """Raised when zero legal actions exist outside GAME_OVER phase."""
    pass
```

**Rationale:** Pure Python exceptions can be raised from Cython code. Keeping them in a separate module avoids circular imports and follows existing codebase conventions (exceptions separate from core logic).

### Anti-Patterns to Avoid
- **Recursive auto-apply:** Phase handlers NEVER call `apply_action()` - would cause recursion. Use iterative loop only at driver level.
- **State hash cycle detection:** Over-engineering - iteration limit is sufficient for bug detection.
- **Mask caching:** Tempting but unnecessary - mask must be regenerated after each action since state changes.
- **nogil in auto-apply loop:** Not possible - `get_valid_action_mask()` uses Python objects (NumPy arrays).

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Forced action detection | Custom bit manipulation | Existing `get_forced_action()` pattern | Already optimal early-exit loop |
| Action mask format | Packed bit array | Float32 array | Neural network requires float32 |
| State copying | Manual memcpy | `state._array.copy()` | NumPy handles memory correctly |
| Exception definitions | Inline strings | Custom exception classes | Better error handling, testability |

**Key insight:** The existing `get_forced_action()` implementation (lines 520-567 in actions.pyx) is already optimal. Don't reinvent it - adapt its pattern for the struct-based return.

## Common Pitfalls

### Pitfall 1: Infinite Loop from Cyclic Forced Actions
**What goes wrong:** Auto-apply creates cycle where action A leads to forced action B which leads back to forced action A.
**Why it happens:** Bug in phase transition logic or mask generation.
**How to avoid:** Implement iteration limit (100) that raises exception if exceeded.
**Warning signs:** Tests timeout, CPU spikes to 100%, 0 games/minute in benchmark.

### Pitfall 2: Zero Actions Treated as Non-Forced
**What goes wrong:** Current `get_forced_action()` returns `(-1, False)` for BOTH 0 and 2+ actions. If you use this API, zero-action states exit the loop silently instead of raising an error.
**Why it happens:** API limitation - tuple cannot distinguish the cases.
**How to avoid:** Use new `ForcedActionResult` struct with explicit `count` field.
**Warning signs:** Game stuck in unexpected phase, no valid actions but not GAME_OVER.

### Pitfall 3: Test Brittleness from State Advancement
**What goes wrong:** Tests that assert intermediate state fail because auto-apply advanced past the expected state.
**Why it happens:** Test expects "after pass, player 1 is active" but auto-apply continued to player 2.
**How to avoid:** Tests should assert on behaviors/outcomes, not intermediate state. Or use history parameter to inspect all actions.
**Warning signs:** Many tests fail with "unexpected active player" or "unexpected phase".

### Pitfall 4: History Overhead in Production
**What goes wrong:** Always allocating history list even when not needed.
**Why it happens:** Not checking `if history is not None` before append.
**How to avoid:** Guard all history operations with `if history is not None:`.
**Warning signs:** Memory allocation in hot path, reduced games/minute.

### Pitfall 5: Exception in Cython Hot Path
**What goes wrong:** Raising exceptions adds try/except overhead even when no exception occurs.
**Why it happens:** Cython needs to check for pending exceptions after each potential raise point.
**How to avoid:** Use `noexcept` on helper functions that cannot raise. Only raise at top-level boundaries (apply_action).
**Warning signs:** Profiler shows exception handling overhead.

## Code Examples

Verified patterns from codebase and research:

### State Copy for History
```python
# Source: NumPy documentation, state._array is numpy ndarray
# state._array.copy() creates independent snapshot
history.append((state._array.copy(), action_idx))
```

### Existing apply_action Structure to Extend
```cython
# Source: core/driver.pyx lines 32-77
cpdef int apply_action(self, GameState state, int action_idx):
    cdef int num_players = state._num_players
    cdef ActionLayout layout = compute_action_layout(num_players)
    cdef ActionInfo info
    cdef int result

    # Validate action index is in bounds
    if action_idx < 0 or action_idx >= layout.total_size:
        return STATUS_INVALID

    # Get valid action mask and check if this action is legal
    cdef object mask = get_valid_action_mask(state)
    if mask[action_idx] != 1.0:
        return STATUS_INVALID

    # Decode action
    info = decode_action(&layout, action_idx)

    # Dispatch based on current phase
    cdef int phase = state.get_phase()

    if phase == PHASE_INVEST:
        result = apply_invest_action(state, &info)
    elif phase == PHASE_BID_IN_AUCTION:
        result = apply_bid_action(state, &info)
    else:
        return STATUS_INVALID

    # Check if game ended after action
    if state.get_phase() == PHASE_GAME_OVER:
        return STATUS_GAME_OVER

    return result
```

### Existing get_forced_action Pattern
```cython
# Source: core/actions.pyx lines 520-567
cpdef tuple get_forced_action(GameState state):
    # ... mask generation ...

    # Count valid actions
    count = 0
    single_action = -1
    for i in range(total_actions):
        if mask_ptr[i] == 1.0:
            count += 1
            if count == 1:
                single_action = i
            elif count > 1:
                return (-1, False)  # Multiple valid actions

    if count == 1:
        return (single_action, True)
    return (-1, False)
```

### Test Fixture Pattern for History Verification
```python
# Source: Based on tests/phases/conftest.py patterns
def test_auto_apply_records_history():
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    history = []
    result = DRIVER.apply_action(state, action_idx, history=history)

    # History contains all actions (initial + auto-applied)
    assert len(history) >= 1
    for state_snapshot, action in history:
        assert isinstance(state_snapshot, np.ndarray)
        assert isinstance(action, int)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No auto-apply | Manual forced action handling by caller | v2.0 | Caller must loop |
| Tuple return | Struct return for count distinction | v2.1 (this phase) | Can detect zero-action errors |

**Current codebase state:**
- `get_forced_action()` exists but is NOT integrated into `apply_action()`
- `apply_action()` applies exactly one action and returns
- Callers must check for forced actions themselves (or not)

## Open Questions

Things that couldn't be fully resolved:

1. **WRAP_UP Phase Behavior**
   - What we know: WRAP_UP phase exists but handlers not implemented
   - What's unclear: Should auto-apply continue into WRAP_UP or treat as terminal?
   - Recommendation: Per CONTEXT.md, no special handling - normal game flow continues

2. **Performance Baseline**
   - What we know: Current benchmark ~X games/minute
   - What's unclear: Exact overhead of auto-apply loop
   - Recommendation: Measure before/after; expect negligible impact (<5%)

## Sources

### Primary (HIGH confidence)
- `/home/icebreaker/rss-az-cython2/core/driver.pyx` - Current GameDriver implementation (93 lines)
- `/home/icebreaker/rss-az-cython2/core/driver.pxd` - Current declarations (15 lines)
- `/home/icebreaker/rss-az-cython2/core/actions.pyx` - get_forced_action() lines 520-567
- `/home/icebreaker/rss-az-cython2/.planning/research/ARCHITECTURE.md` - Integration approach
- `/home/icebreaker/rss-az-cython2/.planning/research/FORCED_ACTION_STACK.md` - Stack recommendations
- `/home/icebreaker/rss-az-cython2/.planning/research/PITFALLS.md` - Common pitfalls
- `/home/icebreaker/rss-az-cython2/.planning/phases/07-core-implementation/07-CONTEXT.md` - User decisions

### Secondary (MEDIUM confidence)
- Cython 3.x documentation - noexcept, struct declarations
- NumPy documentation - ndarray.copy() behavior

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies, existing patterns
- Architecture: HIGH - direct codebase analysis, clear modification points
- Pitfalls: HIGH - verified against codebase structure and prior research

**Research date:** 2026-01-21
**Valid until:** 60 days (stable domain, no external API dependencies)
