# Technology Stack: Action Dispatch & Phase State Machines

**Project:** Rolling Stock Stars Cython Engine - v2 Milestone
**Researched:** 2026-01-20
**Focus:** INVEST/BID_IN_AUCTION phase implementation, Game driver, State machines

## Executive Summary

The existing codebase already contains the foundation for action dispatch (ActionLayout, ActionInfo structs in `core/actions.pyx`). The recommended approach is to **extend existing patterns** rather than introduce new dependencies. The key addition is a `GameDriver` class that dispatches decoded actions to phase-specific handlers.

**Key recommendation:** Use the established patterns (cdef function pointers for dispatch, cpdef class methods for phase handlers) rather than adding new abstractions. The codebase is mature and performant; maintain consistency.

## Recommended Stack

### No New Dependencies Required

The existing stack is sufficient:

| Component | Current Version | Purpose | Status |
|-----------|-----------------|---------|--------|
| Cython | 3.2.4 | Game logic compilation | Already in use |
| NumPy | 2.4.0 | State array operations | Already in use |
| Python | 3.12.3 | Runtime | Already in use |

**Rationale:** Adding new libraries for a game engine's dispatch logic would introduce unnecessary complexity. The existing Cython patterns (enum-based dispatch, typed structs, nogil functions) are optimal for this use case.

### What NOT to Add

| Library | Why NOT |
|---------|---------|
| `transitions` | Pure Python state machine library - GIL overhead, not Cython-compatible |
| `python-statemachine` | Same issue - Python overhead defeats performance goals |
| `enum.auto()` | Python enum overhead; existing `cdef enum` in data.pxd is correct approach |
| External dispatch libraries | Would break nogil and add function call overhead |

**Rationale:** The codebase achieves 10,000+ games/minute benchmarks. Any Python-level abstraction in the hot path would degrade this significantly. The game loop must remain pure Cython.

## Action Dispatch Pattern

### Recommended: Enum-Switch Dispatch

Use the existing `ActionType` enum from `core/actions.pxd` with a switch-style dispatcher:

```cython
# In phases/invest.pyx (new file)
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from core.actions cimport ActionInfo, ActionType, ACTION_PASS, ACTION_AUCTION, ACTION_BUY_SHARE, ACTION_SELL_SHARE
from core.state cimport GameState

cdef int apply_invest_action(GameState state, ActionInfo* action) noexcept nogil:
    """
    Apply an INVEST phase action.
    Returns 0 on success, -1 on error.
    """
    if action.action_type == ACTION_PASS:
        return _handle_pass(state)
    elif action.action_type == ACTION_AUCTION:
        return _handle_start_auction(state, action.slot, action.amount)
    elif action.action_type == ACTION_BUY_SHARE:
        return _handle_buy_share(state, action.corp_id)
    elif action.action_type == ACTION_SELL_SHARE:
        return _handle_sell_share(state, action.corp_id)
    return -1  # Invalid action type for phase
```

**Why this pattern:**
1. Compiles to C switch statement - O(1) dispatch
2. Works with `noexcept nogil` - no GIL acquisition
3. Matches existing codebase style (`decode_action()` in actions.pyx)
4. No function pointer indirection overhead

### Alternative Considered: Function Pointer Table

```cython
# NOT RECOMMENDED - more complexity, marginal benefit
ctypedef int (*action_handler)(GameState, ActionInfo*) noexcept nogil

cdef action_handler INVEST_HANDLERS[4] = [
    &_handle_pass,
    &_handle_auction,
    &_handle_buy_share,
    &_handle_sell_share,
]

cdef int apply_invest_action(GameState state, ActionInfo* action) noexcept nogil:
    return INVEST_HANDLERS[action.action_type](state, action)
```

**Why rejected:**
- Requires maintaining parallel arrays
- Risk of index misalignment with ActionType enum
- Function pointer call has slight overhead vs direct call
- No benefit over switch-style for ~4-10 action types per phase

## Phase State Machine Pattern

### Recommended: Phase Enum + Transition Functions

The existing `GamePhases` enum in `core/data.pxd` defines all phases. Extend with explicit transition logic:

```cython
# In core/driver.pyx (new file)
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from core.data cimport GamePhases, GameConstants
from core.state cimport GameState
from core.actions cimport ActionLayout, ActionInfo, decode_action, compute_action_layout

# Import phase handlers
from phases.invest cimport apply_invest_action
from phases.bid cimport apply_bid_action

cdef class GameDriver:
    """
    Game driver for action dispatch and phase transitions.

    Stateless design: all state lives in GameState array.
    This class provides dispatch logic only.
    """
    cdef ActionLayout _layout
    cdef int _num_players

    def __cinit__(self, int num_players):
        self._num_players = num_players
        self._layout = compute_action_layout(num_players)

    cpdef int apply_action(self, GameState state, int action_idx):
        """
        Apply an action to the game state.

        Returns:
            0 on success
            -1 on invalid action
            1 if game is over
        """
        cdef ActionInfo action = decode_action(&self._layout, action_idx)
        cdef int phase = state.get_phase()
        cdef int result

        # Dispatch to phase handler
        if phase == GamePhases.PHASE_INVEST:
            result = apply_invest_action(state, &action)
        elif phase == GamePhases.PHASE_BID_IN_AUCTION:
            result = apply_bid_action(state, &action)
        # ... other phases
        else:
            return -1  # Unknown phase

        if result < 0:
            return result  # Error

        # Check for phase transitions
        return self._check_transitions(state)

    cdef int _check_transitions(self, GameState state) noexcept:
        """Check and apply phase transitions after action."""
        cdef int phase = state.get_phase()

        if phase == GamePhases.PHASE_INVEST:
            return self._check_invest_end(state)
        elif phase == GamePhases.PHASE_BID_IN_AUCTION:
            return self._check_auction_end(state)
        # ... other phases
        return 0
```

### State Machine Encoding

The state machine is **implicitly encoded** in the phase + turn state. No separate FSM abstraction needed because:

1. **Current state** = `state.get_phase()` (stored in hidden state)
2. **Transition triggers** = action completion + game conditions
3. **Next state** = computed from current phase + conditions

```
INVEST -> BID_IN_AUCTION (when player starts auction)
BID_IN_AUCTION -> INVEST (when auction resolves, continue turn)
BID_IN_AUCTION -> WRAP_UP (when auction resolves, turn passes)
INVEST -> WRAP_UP (when all players pass consecutively)
WRAP_UP -> ACQUISITION (automatic)
...
```

**Why no explicit FSM library:**
- State is already tracked in the float array
- Transitions are deterministic from game rules
- Adding FSM abstraction adds overhead without benefit
- Phase handlers naturally encode their own exit conditions

## File Organization

### New Files to Create

```
phases/
    __init__.pyx         # Re-exports phase handlers
    __init__.pxd         # Declaration for cimports
    invest.pyx           # INVEST phase action handlers
    invest.pxd           # INVEST phase declarations
    bid.pyx              # BID_IN_AUCTION phase handlers
    bid.pxd              # BID_IN_AUCTION phase declarations

core/
    driver.pyx           # GameDriver class
    driver.pxd           # GameDriver declarations
```

### Integration with Existing Code

| Existing Module | Integration Point |
|-----------------|-------------------|
| `core/state.pyx` | GameDriver takes GameState, modifies via existing methods |
| `core/actions.pyx` | GameDriver uses decode_action(), ActionLayout, ActionInfo |
| `core/data.pyx` | GameDriver uses GamePhases enum for dispatch |
| `entities/*.pyx` | Phase handlers use entity accessors (PLAYERS, CORPS, TURN) |

### Update setup.py

Add `phases/` to the extension discovery:

```python
# In setup.py, line 71
pyx_files = find_pyx_files('phases') + find_pyx_files('core') + find_pyx_files('entities')
```

**Note:** This is already configured in the existing setup.py. No changes needed.

## Phase Handler Structure

### Recommended Template

Each phase module should follow this structure:

```cython
# phases/invest.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
INVEST phase action handlers.

Actions:
- PASS: Pass turn (increment consecutive passes)
- AUCTION: Start auction for company at bid price
- BUY_SHARE: Buy share from bank pool
- SELL_SHARE: Sell share to bank pool
"""

from core.state cimport GameState
from core.actions cimport ActionInfo, ActionType
from core.data cimport GamePhases, GameConstants
from entities.player cimport PLAYERS
from entities.turn cimport TURN
from entities.corp cimport CORPS

# ============================================================================
# ACTION HANDLERS (internal, cdef)
# ============================================================================

cdef int _handle_pass(GameState state) noexcept nogil:
    """Handle PASS action in INVEST phase."""
    TURN.increment_consecutive_passes(state)
    # Advance to next player
    # ... implementation
    return 0

cdef int _handle_start_auction(GameState state, int slot, int bid_offset) noexcept nogil:
    """Handle AUCTION action - start auction for company."""
    # Set auction state
    # Transition to BID_IN_AUCTION phase
    # ... implementation
    return 0

cdef int _handle_buy_share(GameState state, int corp_id) noexcept nogil:
    """Handle BUY_SHARE action."""
    # Transfer share from bank to player
    # Adjust share price
    # Clear consecutive passes
    # ... implementation
    return 0

cdef int _handle_sell_share(GameState state, int corp_id) noexcept nogil:
    """Handle SELL_SHARE action."""
    # Transfer share from player to bank
    # Adjust share price
    # Clear consecutive passes
    # ... implementation
    return 0

# ============================================================================
# PHASE DISPATCHER (exported, cdef for C-level access)
# ============================================================================

cdef int apply_invest_action(GameState state, ActionInfo* action) noexcept nogil:
    """
    Apply an INVEST phase action.

    Args:
        state: Current game state
        action: Decoded action info

    Returns:
        0 on success, -1 on invalid action
    """
    if action.action_type == ActionType.ACTION_PASS:
        return _handle_pass(state)
    elif action.action_type == ActionType.ACTION_AUCTION:
        return _handle_start_auction(state, action.slot, action.amount)
    elif action.action_type == ActionType.ACTION_BUY_SHARE:
        return _handle_buy_share(state, action.corp_id)
    elif action.action_type == ActionType.ACTION_SELL_SHARE:
        return _handle_sell_share(state, action.corp_id)
    return -1

# ============================================================================
# TRANSITION CHECKS (for GameDriver)
# ============================================================================

cdef bint should_end_invest_phase(GameState state) noexcept nogil:
    """Check if INVEST phase should end (all players passed)."""
    return TURN.get_consecutive_passes(state) >= state._num_players
```

## Performance Considerations

### Must Maintain

| Concern | Current Approach | Keep It |
|---------|------------------|---------|
| nogil | All action handlers must be `noexcept nogil` | Yes |
| boundscheck | Disabled globally in compiler directives | Yes |
| Zero allocation | No Python object creation in hot path | Yes |
| O(1) dispatch | Enum-based switch, not dynamic dispatch | Yes |

### Benchmarking Integration

The existing benchmark command should work unchanged:

```bash
python setup.py benchmark --num-games=1000 --num-players=3
```

**Target:** Maintain >10,000 games/minute with full action dispatch.

## Transition Logic Patterns

### Active Player Advancement

```cython
cdef void advance_active_player(GameState state) noexcept nogil:
    """Advance to next player in turn order."""
    cdef int current = state._get_active_player()
    cdef int next_player = (current + 1) % state._num_players
    state._set_active_player(next_player)
```

### Phase Transition

```cython
cdef void transition_to_phase(GameState state, int new_phase) noexcept:
    """Transition to a new game phase."""
    # Use existing set_phase which updates both hidden and one-hot
    TURN.set_phase(state, new_phase)
```

### Auction Resolution

```cython
cdef int resolve_auction(GameState state) noexcept:
    """Resolve completed auction, assign company to winner."""
    cdef int winner = TURN.get_auction_high_bidder(state)
    cdef int company_id = TURN.get_auction_company(state)
    cdef int price = TURN.get_auction_price(state)

    if winner < 0:
        # No winner - return company to auction pool
        return 0

    # Transfer company to winner
    PLAYERS[winner].set_owns_company(state, company_id, True)
    PLAYERS[winner].add_cash(state, -price)

    # Remove from auction pool
    state.set_company_for_auction(company_id, False)

    # Clear auction state
    TURN.clear_auction_company(state)
    TURN.clear_auction_high_bidder(state)
    TURN.clear_auction_passed(state)

    return 0
```

## Summary

### Stack Changes

| Category | Change | Rationale |
|----------|--------|-----------|
| Dependencies | None | Existing stack sufficient |
| New modules | `phases/*.pyx`, `core/driver.pyx` | Organize phase logic |
| Patterns | Enum-switch dispatch | Matches codebase, optimal performance |
| State machine | Implicit in phase + conditions | No external FSM needed |

### Key Principles

1. **Extend, don't replace** - Build on existing ActionLayout, ActionInfo, GamePhases
2. **nogil everywhere** - All dispatch and handlers must release GIL
3. **Stateless handlers** - All state in GameState array, handlers are pure functions
4. **Enum dispatch** - Use existing enums, compile to C switch
5. **No new dependencies** - The stack is complete

### Integration Checklist

- [ ] Create `phases/` module structure with .pyx and .pxd files
- [ ] Create `core/driver.pyx` GameDriver class
- [ ] Phase handlers use existing entity accessors
- [ ] All handlers are `cdef ... noexcept nogil`
- [ ] Update imports in `core/__init__.py`
- [ ] Benchmark confirms no performance regression

## Sources

- Cython documentation on typed memoryviews and nogil: https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html (HIGH confidence - official docs)
- Existing codebase patterns in `core/actions.pyx`, `entities/*.pyx` (HIGH confidence - established patterns)
- Performance characteristics from `setup.py` benchmark command (HIGH confidence - validated)

---

*Stack research: 2026-01-20*
*Confidence: HIGH - recommendations based on existing codebase patterns*
