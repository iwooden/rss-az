# Phase 2: Infrastructure Setup - Research

**Researched:** 2026-01-20
**Domain:** Cython game driver architecture, action dispatch, phase handler pattern
**Confidence:** HIGH

## Summary

Phase 2 establishes the GameDriver infrastructure for routing actions to phase handlers and generating legal move masks. The existing codebase already has the foundation in place: `core/actions.pyx` contains `decode_action()` which converts action indices to `ActionInfo` structs, and `get_valid_action_mask()` which generates phase-specific legal move masks. The GameDriver needs to be a thin orchestration layer that:

1. Uses existing `decode_action()` to interpret action indices (DRV-04)
2. Dispatches to phase-specific handler functions based on `state.get_phase()` (DRV-01)
3. Returns status codes from `apply_action()` (DRV-02)
4. Wraps existing `get_valid_action_mask()` for public API (DRV-03)

The key architectural decision is whether GameDriver is a class or a collection of module-level functions. Given the stateless entity pattern already established (entities take GameState as parameter), a stateless GameDriver class with `cpdef` methods is the natural fit.

**Primary recommendation:** Implement GameDriver as a `cdef class` in `core/driver.pyx` with `cpdef int apply_action(self, GameState state, int action_idx)` and `cpdef object get_legal_moves(self, GameState state)`. Phase handlers should be `cdef` nogil functions in a new `phases/` directory structure, with one module per game phase.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Cython | 3.x | Performance-critical dispatch | Already in use, zero-overhead extensions |
| NumPy | Latest | Action mask arrays (float32) | Already used for mask generation in actions.pyx |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| libc.stdint | Standard | Integer types for status codes | If explicit int8/int16 status codes needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Class with methods | Module functions | Class keeps related functions together, matches entity pattern |
| Function pointers | If/else dispatch | Function pointers add complexity, if/else is clearer and optimizes well |
| Return objects | Integer status codes | Status codes are simpler, sufficient for ok/error distinction |

**Installation:**
```bash
# Already installed - no new dependencies needed
```

## Architecture Patterns

### Recommended Project Structure
```
core/
    driver.pyx          # GameDriver class with apply_action, get_legal_moves
    driver.pxd          # Driver declarations for cimport
    actions.pyx         # EXISTING - decode_action(), get_valid_action_mask()
    actions.pxd         # EXISTING - ActionInfo, ActionLayout structs
    state.pyx           # EXISTING - GameState class
    data.pyx            # EXISTING - GamePhases enum, GameConstants

phases/
    __init__.pyx        # Barrel export for phase handlers
    __init__.pxd        # Phase function declarations
    invest.pyx          # INVEST phase handler (Phase 3 implementation)
    invest.pxd          # INVEST declarations
    bid.pyx             # BID_IN_AUCTION phase handler (Phase 3 implementation)
    bid.pxd             # BID declarations
```

### Pattern 1: GameDriver Class Structure
**What:** Stateless driver class following entity handle pattern
**When to use:** Main entry point for game actions
**Example:**
```cython
# Source: Follows entities/turn.pyx pattern (lines 15-26)
cdef class GameDriver:
    """
    Game driver for action dispatch and legal move generation.

    Stateless design: all methods take GameState as parameter.
    Instantiate once at module load, reuse for all games.
    """

    cpdef int apply_action(self, GameState state, int action_idx):
        """
        Apply action to game state.

        Args:
            state: GameState to mutate
            action_idx: Action index from NN output

        Returns:
            Status code: 0=success, 1=invalid_action, 2=game_over
        """
        cdef ActionLayout layout = compute_action_layout(state._num_players)
        cdef ActionInfo info = decode_action(&layout, action_idx)

        # Dispatch based on phase
        cdef int phase = state.get_phase()
        if phase == PHASE_INVEST:
            return _apply_invest_action(state, &info)
        elif phase == PHASE_BID_IN_AUCTION:
            return _apply_bid_action(state, &info)
        # ... other phases

        return 1  # Invalid action (unknown phase)

    cpdef object get_legal_moves(self, GameState state):
        """
        Get valid action mask for current state.

        Returns:
            float32 numpy array where 1.0=valid, 0.0=invalid
        """
        return get_valid_action_mask(state)

# Global singleton instance
DRIVER = GameDriver()
```

### Pattern 2: Phase Handler Functions
**What:** Cdef nogil functions for phase-specific action application
**When to use:** Called by GameDriver.apply_action() dispatch
**Example:**
```cython
# Source: Pattern from actions.pyx _fill_invest_mask (lines 251-299)
# phases/invest.pyx

cdef int _apply_invest_action(GameState state, ActionInfo* info) noexcept:
    """
    Apply INVEST phase action to state.

    Returns: 0=success, 1=invalid
    """
    if info.action_type == ACTION_PASS:
        return _handle_pass(state)
    elif info.action_type == ACTION_AUCTION:
        return _handle_start_auction(state, info.slot, info.amount)
    elif info.action_type == ACTION_BUY_SHARE:
        return _handle_buy_share(state, info.corp_id)
    elif info.action_type == ACTION_SELL_SHARE:
        return _handle_sell_share(state, info.corp_id)

    return 1  # Invalid action type for this phase

cdef int _handle_pass(GameState state) noexcept:
    """Handle PASS action in INVEST phase."""
    # Implementation in Phase 3
    return 0

# Note: Cannot use nogil with cpdef methods on GameState
# Use cdef helper functions that take float* pointers for nogil paths
```

### Pattern 3: Status Code Return Convention
**What:** Integer return codes for action success/failure
**When to use:** All apply_action style functions
**Example:**
```cython
# Follows Cython convention of integer status codes
# (Similar to libc return conventions)

# Define status codes as enum for clarity
cdef enum ActionStatus:
    STATUS_OK = 0           # Action applied successfully
    STATUS_INVALID = 1      # Invalid action for current state
    STATUS_GAME_OVER = 2    # Game ended after this action
    STATUS_PHASE_CHANGE = 3 # Phase transitioned (informational)
```

### Pattern 4: Action Validation via Mask
**What:** Use existing mask generation for validation
**When to use:** Before applying action in apply_action()
**Example:**
```cython
# Source: actions.pyx get_valid_action_mask (lines 471-504)
cpdef int apply_action(self, GameState state, int action_idx):
    """Apply action with mask validation."""
    cdef cnp.ndarray mask = get_valid_action_mask(state)
    cdef float* mask_ptr = <float*>cnp.PyArray_DATA(mask)

    # Validate action is legal
    if action_idx < 0 or action_idx >= mask.shape[0]:
        return STATUS_INVALID
    if mask_ptr[action_idx] != 1.0:
        return STATUS_INVALID

    # Action is valid - decode and dispatch
    # ...
```

### Anti-Patterns to Avoid
- **Storing state in driver:** GameDriver must be stateless, pass GameState to all methods
- **Modifying actions.pyx:** DRV-04 requires using existing decode_action() without modification
- **Python exception raising:** Use integer status codes, not exceptions (noexcept requirement)
- **Complex return types:** Return int status, not tuples or objects (performance)
- **Reimplementing mask generation:** Wrap existing get_valid_action_mask(), don't reimplement

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Action decoding | Custom index parsing | `decode_action()` from actions.pyx | Already handles all action types, tested |
| Legal move mask | Custom validation logic | `get_valid_action_mask()` from actions.pyx | Phase-aware, handles all edge cases |
| Action layout computation | Hardcoded offsets | `compute_action_layout()` | Player-count dependent, already computed |
| Phase identification | Custom state parsing | `state.get_phase()` | Uses hidden state for O(1) access |
| Active player tracking | Manual turn tracking | `state._get_active_player()` | Built into GameState |

**Key insight:** The actions.pyx module already provides the complete action infrastructure. GameDriver is an orchestration layer, not a reimplementation.

## Common Pitfalls

### Pitfall 1: Breaking noexcept nogil Contract
**What goes wrong:** Phase handlers that hold GIL or raise exceptions block parallelism
**Why it happens:** Calling cpdef methods on GameState acquires GIL
**How to avoid:** Use cdef helper functions with float* pointers for performance-critical paths; cpdef wrapper for Python interface
**Warning signs:** Slow benchmark times, poor multi-threading performance

### Pitfall 2: Modifying actions.pyx
**What goes wrong:** Requirements violation (DRV-04 says use existing decode_action())
**Why it happens:** Temptation to add driver logic to actions module
**How to avoid:** Keep actions.pyx unchanged; put dispatch logic in new driver.pyx
**Warning signs:** Git diff shows changes to actions.pyx

### Pitfall 3: Incorrect Phase Dispatch
**What goes wrong:** Actions routed to wrong phase handler
**Why it happens:** Phase enum values confused, or phase not updated after transitions
**How to avoid:** Always use GamePhases enum constants (PHASE_INVEST, PHASE_BID_IN_AUCTION), verify phase matches action via mask first
**Warning signs:** Actions fail with "invalid" when they should be valid

### Pitfall 4: Missing Phase Handler Stubs
**What goes wrong:** apply_action crashes on valid phases
**Why it happens:** GameDriver dispatches to phase handler that doesn't exist yet
**How to avoid:** Create stub handlers for all phases that return STATUS_OK or STATUS_INVALID
**Warning signs:** Segfaults or unhandled phase values

### Pitfall 5: Forgetting Action Mask Validation
**What goes wrong:** Invalid actions applied, corrupting state
**Why it happens:** Assuming caller validated; double-validation seems wasteful
**How to avoid:** Always validate action against mask in apply_action() before dispatch
**Warning signs:** State corruption, inconsistent game outcomes

### Pitfall 6: Status Code Inconsistency
**What goes wrong:** Different handlers return different meanings for same code
**Why it happens:** No central status code definition
**How to avoid:** Define ActionStatus enum in driver.pxd, use consistently everywhere
**Warning signs:** Caller confusion about what return value means

## Code Examples

Verified patterns from the existing codebase:

### Existing decode_action() Usage
```cython
# Source: core/actions.pyx lines 150-244
cdef ActionInfo decode_action(ActionLayout* layout, int action_idx) noexcept nogil:
    """Decode an action index into an ActionInfo struct."""
    cdef ActionInfo info
    # ... decoding logic
    return info

# Usage in driver:
cdef ActionLayout layout = compute_action_layout(state._num_players)
cdef ActionInfo info = decode_action(&layout, action_idx)
# info now contains: phase, action_type, slot, corp_id, amount
```

### Existing Mask Generation
```cython
# Source: core/actions.pyx lines 471-504
cpdef object get_valid_action_mask(GameState state):
    """Generate valid action mask for current game state."""
    cdef int num_players = state._num_players
    cdef ActionLayout layout = compute_action_layout(num_players)
    cdef int total_actions = layout.total_size
    cdef cnp.ndarray mask = np.zeros(total_actions, dtype=np.float32)
    cdef float* mask_ptr = <float*>cnp.PyArray_DATA(mask)

    cdef int phase = state.get_phase()

    if phase == PHASE_INVEST:
        _fill_invest_mask(state, &layout, mask_ptr)
    elif phase == PHASE_BID_IN_AUCTION:
        _fill_bid_mask(state, &layout, mask_ptr)
    # ... other phases

    return mask
```

### Phase Enum Usage
```cython
# Source: core/data.pxd lines 24-35
cpdef enum GamePhases:
    PHASE_INVEST = 0
    PHASE_BID_IN_AUCTION = 1
    PHASE_WRAP_UP = 2
    PHASE_ACQUISITION = 3
    PHASE_CLOSING = 4
    PHASE_INCOME = 5
    PHASE_DIVIDENDS = 6
    PHASE_END_CARD = 7
    PHASE_ISSUE_SHARES = 8
    PHASE_IPO = 9
    PHASE_GAME_OVER = 10

# Usage in driver:
from core.data cimport GamePhases, PHASE_INVEST, PHASE_BID_IN_AUCTION

cdef int phase = state.get_phase()
if phase == PHASE_INVEST:
    # handle invest actions
```

### ActionInfo Struct Fields
```cython
# Source: core/actions.pxd lines 90-96
cdef struct ActionInfo:
    int phase           # PHASE_* constant
    int action_type     # ActionType enum (ACTION_PASS, ACTION_AUCTION, etc.)
    int slot            # auction_slot, par_slot
    int corp_id         # -1 if not applicable
    int amount          # price_offset, bid_offset, dividend amount

# Usage:
cdef ActionInfo info = decode_action(&layout, action_idx)
if info.action_type == ACTION_PASS:
    # handle pass
elif info.action_type == ACTION_AUCTION:
    # info.slot = auction slot (0 to num_players-1)
    # info.amount = bid offset (0-19)
```

### Entity Handle Pattern (to follow)
```cython
# Source: entities/turn.pyx lines 15-35
cdef class TurnState:
    """Entity handle for turn state. Stateless design."""

    def __cinit__(self):
        # Initialize offset caches to 0
        self._num_players = 0
        # ...

    cpdef void initialize(self, GameState state):
        """Initialize offsets from state layout."""
        cdef StateLayout layout = state._layout
        self._num_players = state._num_players
        # Cache offsets...

# Global singleton
TURN = TurnState()

# GameDriver should follow same pattern:
# - __cinit__ for minimal setup
# - No initialize() needed (doesn't cache offsets)
# - All methods take GameState parameter
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Monolithic game loop | Driver + phase handlers | v2 architecture | Cleaner separation of concerns |
| Direct state manipulation | Through entity handles | v1 architecture | Encapsulation, normalization |
| Python-level dispatch | Cython cdef dispatch | New in v2 | 10-100x faster dispatch |

**Deprecated/outdated:**
- None - this is new infrastructure for v2

## Open Questions

Things that couldn't be fully resolved:

1. **Status code granularity**
   - What we know: Need at least OK/INVALID distinction
   - What's unclear: Should GAME_OVER be separate status or detected via phase check?
   - Recommendation: Use STATUS_OK, STATUS_INVALID, STATUS_GAME_OVER for clarity; caller can check `state.get_phase() == PHASE_GAME_OVER` alternatively

2. **Phase handler location**
   - What we know: setup.py already includes `phases/` in package list
   - What's unclear: Whether to put handlers in phases/ or keep in core/
   - Recommendation: Use phases/ directory - matches existing setup.py expectation, cleaner organization

3. **Forced action handling**
   - What we know: `get_forced_action()` exists in actions.pyx
   - What's unclear: Should GameDriver auto-apply forced actions?
   - Recommendation: Let caller decide; provide `get_forced_action()` wrapper but don't auto-apply (keeps driver predictable)

4. **Batch action application**
   - What we know: AlphaZero training may want batch operations
   - What's unclear: Whether to add batch_apply_action() now or later
   - Recommendation: Defer to later; single-action API sufficient for v2

## Sources

### Primary (HIGH confidence)
- `core/actions.pyx` - decode_action(), get_valid_action_mask(), action layout (lines 55-505)
- `core/actions.pxd` - ActionInfo struct, ActionLayout struct, ActionType enum (lines 1-113)
- `core/state.pyx` - GameState class, get_phase() (lines 393-404)
- `core/state.pxd` - GameState declaration (lines 110-192)
- `core/data.pxd` - GamePhases enum (lines 24-35)
- `entities/turn.pyx` - Entity handle pattern example (lines 15-93)
- `setup.py` - Package structure including `phases/` (line 71, 90)

### Secondary (MEDIUM confidence)
- `.planning/codebase/ARCHITECTURE.md` - Layer descriptions, data flow
- `.planning/codebase/CONVENTIONS.md` - Naming, cdef/cpdef patterns

### Tertiary (LOW confidence)
- None - all findings verified with primary sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No new dependencies, all patterns from existing codebase
- Architecture: HIGH - Follows established entity handle pattern, wraps existing infrastructure
- Pitfalls: HIGH - Derived from actual code constraints and Cython requirements

**Research date:** 2026-01-20
**Valid until:** 60 days (stable architecture, internal infrastructure only)
