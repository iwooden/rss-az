# Phase 9: WRAP_UP Core Logic - Research

**Researched:** 2026-01-23
**Domain:** Deterministic phase handler implementation with player reordering
**Confidence:** HIGH

## Summary

Phase 9 implements the WRAP_UP phase, which executes deterministically when all players pass consecutively in INVEST. The phase reorders players by descending cash (tie-breaking by old turn order), then transitions to ACQUISITION (stub). This requires: (1) creating a new `phases/wrap_up.pyx` handler, (2) modifying INVEST to transition to WRAP_UP instead of GAME_OVER, (3) updating GameDriver to handle 0 legal actions for non-player phases, and (4) adding an ACQUISITION stub that immediately transitions back to INVEST.

The codebase already has all necessary entity interfaces (Player.get_cash, Player.set_turn_order, TURN.set_phase). The primary work is implementing the reordering algorithm and integrating WRAP_UP into the auto-apply loop. Per CONTEXT.md decisions, WRAP_UP creates a discrete history entry with a sentinel action value.

**Primary recommendation:** Create `apply_wrap_up` handler following the existing phase handler pattern in `phases/bid.pyx`. Use a sentinel action constant (e.g., `ACTION_WRAP_UP = -1`) for history recording. Implement stable sort by (descending cash, ascending old position) for deterministic reordering.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Cython | 3.x | Phase handler implementation | Already used for all handlers |
| NumPy | 2.x | State array manipulation | Required for history entries |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | 9.x | Test framework | Verification tests |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| New wrap_up.pyx | Inline in invest.pyx | Separation keeps INVEST clean per CONTEXT.md |
| Sentinel action constant | New action enum value | Sentinel simpler, avoids action layout changes |
| ACQUISITION stub transition | Loop back to INVEST | Stub future-proofs for ACQUISITION implementation |

**No new dependencies required** - existing Cython stack is sufficient.

## Architecture Patterns

### Recommended Project Structure
```
phases/
  __init__.pyx           # Unchanged
  __init__.pxd           # Add apply_wrap_up declaration
  invest.pyx             # MODIFIED: transition to WRAP_UP not GAME_OVER
  invest.pxd             # Unchanged
  bid.pyx                # Unchanged (reference pattern)
  wrap_up.pyx            # NEW: WRAP_UP phase handler
  wrap_up.pxd            # NEW: cdef declarations
  acquisition.pyx        # NEW: stub that transitions to INVEST
  acquisition.pxd        # NEW: cdef declaration
```

### Pattern 1: Deterministic Phase Handler (No Actions)
**What:** A phase handler that executes deterministic logic with 0 legal actions.
**When to use:** Non-player phases like WRAP_UP, INCOME.
**Example:**
```cython
# Source: new phases/wrap_up.pyx following bid.pyx pattern
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""WRAP_UP phase handler implementation."""

from core.state cimport GameState
from core.data cimport GamePhases
from entities import turn as turn_module
from entities import player as player_module


cdef int apply_wrap_up(GameState state) noexcept:
    """
    Execute WRAP_UP phase logic.

    This is a deterministic non-player phase with 0 actions.
    Steps:
    1. Reorder players by descending cash (tie-break by old position)
    2. Set active player to new position 0
    3. Clear consecutive passes for next INVEST round
    4. Transition to ACQUISITION (which stubs to INVEST)

    Returns: 0 always (deterministic, no failure modes)
    """
    _reorder_players_by_cash(state)
    turn_module.TURN.clear_consecutive_passes(state)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
    return 0
```

### Pattern 2: Stable Sort with Tie-Breaking (Player Reordering)
**What:** Sort players by descending cash, using old turn order as tie-breaker.
**When to use:** WRAP_UP reordering per REORDER-01, REORDER-02.
**Example:**
```cython
# Source: new phases/wrap_up.pyx
cdef void _reorder_players_by_cash(GameState state) noexcept:
    """
    Reorder players by descending cash with old position tie-breaking.

    Algorithm:
    1. Collect (cash, old_position, player_id) tuples
    2. Sort by (-cash, old_position) for descending cash, ascending old position
    3. Assign new turn order positions
    4. Set active player to new position 0
    """
    cdef int num_players = state._num_players
    cdef int[6] cash_values       # Max 6 players
    cdef int[6] old_positions
    cdef int[6] player_ids
    cdef int[6] new_order         # player_id at each position
    cdef int i, j, temp_id, best_idx
    cdef int best_cash, best_pos

    # Gather current state
    for i in range(num_players):
        player_ids[i] = i
        cash_values[i] = player_module.PLAYERS[i].get_cash(state)
        old_positions[i] = player_module.PLAYERS[i].get_turn_order(state)

    # Selection sort by (-cash, old_position) - stable for ties
    for i in range(num_players):
        best_idx = i
        best_cash = cash_values[player_ids[i]]
        best_pos = old_positions[player_ids[i]]

        for j in range(i + 1, num_players):
            curr_cash = cash_values[player_ids[j]]
            curr_pos = old_positions[player_ids[j]]

            # Higher cash wins, or if equal, lower old position wins
            if (curr_cash > best_cash or
                (curr_cash == best_cash and curr_pos < best_pos)):
                best_idx = j
                best_cash = curr_cash
                best_pos = curr_pos

        # Swap to front
        if best_idx != i:
            temp_id = player_ids[i]
            player_ids[i] = player_ids[best_idx]
            player_ids[best_idx] = temp_id

    # Apply new turn order
    for i in range(num_players):
        player_module.PLAYERS[player_ids[i]].set_turn_order(state, i)

    # Set active player to new position 0 (REORDER-03)
    state._set_active_player(player_ids[0])
```

### Pattern 3: Zero-Action Phase Detection in GameDriver
**What:** Allow 0 legal actions for non-player phases like WRAP_UP.
**When to use:** Phases where the system acts, not players.
**Example:**
```cython
# Source: core/driver.pyx modification
# In _check_forced_action or apply_action:

cdef bint _is_non_player_phase(int phase) noexcept nogil:
    """Check if phase has no player actions (deterministic execution)."""
    return (phase == PHASE_WRAP_UP or
            phase == PHASE_INCOME or  # Future
            phase == PHASE_END_CARD)  # Future

# In apply_action auto-apply loop:
if forced.count == 0:
    # Check if this is a non-player phase (0 actions is OK)
    phase = state.get_phase()
    if _is_non_player_phase(phase):
        # Execute phase logic directly, record to history
        if phase == PHASE_WRAP_UP:
            _execute_wrap_up_with_history(state, history)
        # Continue auto-apply loop
        continue
    else:
        raise ZeroLegalActionsError("Zero legal actions in non-terminal state")
```

### Pattern 4: Discrete History Entry for Non-Player Phases
**What:** Record WRAP_UP execution to history like any other action.
**When to use:** Non-player phases that need state history tracking.
**Example:**
```cython
# Source: core/driver.pyx or phases/wrap_up.pyx

# Sentinel action constant for WRAP_UP (negative to distinguish from real actions)
DEF ACTION_WRAP_UP_SENTINEL = -100

cdef void _execute_wrap_up_with_history(GameState state, object history) noexcept:
    """Execute WRAP_UP and record to history."""
    # Record state BEFORE wrap_up executes
    if history is not None:
        history.append((state._array.copy(), ACTION_WRAP_UP_SENTINEL))

    # Execute deterministic logic
    apply_wrap_up(state)
```

### Pattern 5: ACQUISITION Stub (Minimal Transition)
**What:** Stub phase that immediately transitions to next phase.
**When to use:** Future phases not yet implemented.
**Example:**
```cython
# Source: new phases/acquisition.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""ACQUISITION phase stub - transitions immediately to INVEST."""

from core.state cimport GameState
from core.data cimport GamePhases
from entities import turn as turn_module


cdef int apply_acquisition_stub(GameState state) noexcept:
    """
    Stub: ACQUISITION immediately transitions to new INVEST turn.

    When ACQUISITION is fully implemented, this will be replaced with:
    - FI purchase logic (Phase 10)
    - Corp acquisition offers
    - Company availability updates

    For now, just increment turn number and start new INVEST.
    """
    cdef int current_turn = turn_module.TURN.get_turn_number(state)
    turn_module.TURN.set_turn_number(state, current_turn + 1)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)

    # Clear per-turn tracking for all players
    cdef int i
    for i in range(state._num_players):
        player_module.PLAYERS[i].clear_roundtrip_tracking(state)

    return 0
```

### Anti-Patterns to Avoid
- **Modifying action mask for WRAP_UP:** WRAP_UP has 0 actions by design; don't add fake actions.
- **Raising ZeroLegalActionsError for WRAP_UP:** This is valid for non-player phases; add phase check.
- **History entry with action_idx from mask:** WRAP_UP has no mask actions; use sentinel constant.
- **Looping INVEST back to INVEST:** Per CONTEXT.md, go through WRAP_UP -> ACQUISITION -> INVEST.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Player cash comparison | Manual array iteration | `PLAYERS[i].get_cash(state)` | Entity method handles normalization |
| Turn order update | Direct array write | `PLAYERS[i].set_turn_order(state, pos)` | Handles one-hot encoding |
| Phase transition | Manual phase value set | `TURN.set_phase(state, PHASE_X)` | Updates both hidden and one-hot |
| Active player update | Manual hidden state write | `state._set_active_player(id)` | Existing cdef method |
| Consecutive passes reset | Manual write | `TURN.clear_consecutive_passes(state)` | Already implemented |

**Key insight:** All entity interfaces needed for WRAP_UP already exist. No new entity methods required.

## Common Pitfalls

### Pitfall 1: Unstable Sort for Tie-Breaking
**What goes wrong:** Players with equal cash get inconsistent ordering between runs.
**Why it happens:** Python's sort is stable but C qsort is not; naive implementations break ties randomly.
**How to avoid:** Always include old_position as secondary sort key, use selection sort (stable) or explicit tie-breaker.
**Warning signs:** Non-deterministic player order in tests with equal cash scenarios.

### Pitfall 2: Forgetting to Update Active Player
**What goes wrong:** Active player points to wrong player after reordering.
**Why it happens:** Reordering changes positions but active player ID stays at old value.
**How to avoid:** After reordering, explicitly set active player to whoever is now at position 0.
**Warning signs:** Wrong player takes first action in new INVEST round.

### Pitfall 3: ZeroLegalActionsError on WRAP_UP
**What goes wrong:** GameDriver raises error when entering WRAP_UP phase.
**Why it happens:** Auto-apply loop expects 1+ actions; WRAP_UP has 0.
**How to avoid:** Add phase check before raising error; execute non-player phases directly.
**Warning signs:** Exception when all players pass in tests.

### Pitfall 4: History Entry Missing for WRAP_UP
**What goes wrong:** WRAP_UP state transition not visible in history.
**Why it happens:** Skipping history append because there's no action_idx from mask.
**How to avoid:** Use sentinel constant for action value; history is (state, action) tuples.
**Warning signs:** Unexpected history length in tests, missing state between INVEST and new INVEST.

### Pitfall 5: INVEST Still Transitions to GAME_OVER
**What goes wrong:** All-pass scenario ends game instead of triggering WRAP_UP.
**Why it happens:** Forgot to update invest.pyx transition logic.
**How to avoid:** Change `PHASE_GAME_OVER` to `PHASE_WRAP_UP` in pass handler.
**Warning signs:** Tests expecting WRAP_UP get GAME_OVER instead.

## Code Examples

Verified patterns from codebase:

### Existing Phase Handler Pattern (bid.pyx)
```cython
# Source: phases/bid.pyx lines 70-117
cdef int apply_bid_action(GameState state, ActionInfo* info) noexcept:
    """Apply BID_IN_AUCTION phase action to state."""
    # ... action-specific logic ...
    return 0  # success
```

### Existing Turn Order Navigation (turn.pyx)
```cython
# Source: entities/turn.pyx lines 487-502
cpdef int find_player_at_position(self, GameState state, int position):
    """Find player_id with given turn order position."""
    cdef int player_id
    for player_id in range(state._num_players):
        if player_module.PLAYERS[player_id].get_turn_order(state) == position:
            return player_id
    return -1
```

### Existing Active Player Update (invest.pyx)
```cython
# Source: phases/invest.pyx lines 169-175
cdef void _advance_active_player(GameState state) noexcept:
    """Advance to next player in turn order."""
    cdef int current_player = state._get_active_player()
    cdef int current_position = player_module.PLAYERS[current_player].get_turn_order(state)
    cdef int next_position = (current_position + 1) % state._num_players
    cdef int next_player = turn_module.TURN.find_player_at_position(state, next_position)
    state._set_active_player(next_player)
```

### Current INVEST All-Pass Transition (invest.pyx)
```cython
# Source: phases/invest.pyx lines 341-346 (TO BE MODIFIED)
# Check if all players have passed
if turn_module.TURN.get_consecutive_passes(state) >= state._num_players:
    # TODO(v3+): Replace with PHASE_WRAP_UP when implemented
    turn_module.TURN.set_phase(state, GamePhases.PHASE_GAME_OVER)  # <-- Change to WRAP_UP
```

### History Recording Pattern (driver.pyx)
```cython
# Source: core/driver.pyx lines 103-105
# Append to history if provided (before applying action)
if history is not None:
    history.append((state._array.copy(), action_idx))
```

### GameDriver Auto-Apply Loop (driver.pyx)
```cython
# Source: core/driver.pyx lines 156-175
# Auto-apply forced actions
iterations = 0
while iterations < MAX_FORCED_ITERATIONS:
    forced = _check_forced_action(state)

    if forced.count == 0:
        raise ZeroLegalActionsError(...)  # <-- Add phase check here

    if forced.count >= 2:
        return STATUS_OK

    # Exactly 1 action - auto-apply it
    result = self._apply_single_action(state, forced.action_idx, history)
    # ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| All-pass -> GAME_OVER | All-pass -> WRAP_UP -> ACQUISITION -> INVEST | v3.0 Phase 9 | Proper game flow |
| 0 actions = error | 0 actions OK for non-player phases | v3.0 Phase 9 | Enables WRAP_UP, INCOME |
| No turn order reordering | Descending cash reordering | v3.0 Phase 9 | Correct game rules |

**Phase constants in data.pxd:**
- `PHASE_WRAP_UP = 2` - Already defined, ready to use
- `PHASE_ACQUISITION = 3` - Already defined, ready for stub
- `PHASE_INVEST = 0` - Transition target after ACQUISITION

## Implementation Sequence

Recommended order for Phase 9 implementation:

1. **Create wrap_up.pyx/pxd** - Player reordering algorithm
2. **Create acquisition.pyx/pxd** - Stub that transitions to INVEST
3. **Modify invest.pyx** - Change all-pass to PHASE_WRAP_UP
4. **Modify driver.pyx** - Add non-player phase handling
5. **Update setup.py** - Add new .pyx files to build
6. **Add tests** - Reordering correctness, phase transitions

## Open Questions

Things that couldn't be fully resolved:

1. **Sentinel Action Value**
   - What we know: Need a value for history entry action field
   - What's unclear: Best practice for sentinel values (-100, -1, MAX_INT?)
   - Recommendation: Use -100 for WRAP_UP, -101 for ACQUISITION stub, etc. (negative range reserved for system actions)

2. **Turn Number Increment Location**
   - What we know: Turn number should increment when starting new INVEST round
   - What's unclear: Should it increment in WRAP_UP, ACQUISITION, or at INVEST start?
   - Recommendation: Increment in ACQUISITION stub (the final phase before INVEST), as this matches game flow

3. **Round-Trip Tracking Clear Timing**
   - What we know: Per-turn tracking should reset for new turn
   - What's unclear: Clear in WRAP_UP or ACQUISITION?
   - Recommendation: Clear in ACQUISITION stub (same reasoning as turn number)

## Sources

### Primary (HIGH confidence)
- `/home/icebreaker/rss-az-cython2/phases/invest.pyx` - Current INVEST handler, transition logic (392 lines)
- `/home/icebreaker/rss-az-cython2/phases/bid.pyx` - Phase handler pattern reference (118 lines)
- `/home/icebreaker/rss-az-cython2/core/driver.pyx` - Auto-apply loop, history tracking (191 lines)
- `/home/icebreaker/rss-az-cython2/entities/player.pyx` - Cash/turn order methods (427 lines)
- `/home/icebreaker/rss-az-cython2/entities/turn.pyx` - Phase transitions, consecutive passes (548 lines)
- `/home/icebreaker/rss-az-cython2/core/data.pxd` - Phase constants (PHASE_WRAP_UP = 2)
- `/home/icebreaker/rss-az-cython2/.planning/phases/09-wrap-up-core-logic/09-CONTEXT.md` - User decisions

### Secondary (MEDIUM confidence)
- `/home/icebreaker/rss-az-cython2/.planning/REQUIREMENTS.md` - Phase 9 requirements (REORDER-01 to PHASE-04)
- `/home/icebreaker/rss-az-cython2/.planning/ROADMAP.md` - Phase dependencies and success criteria
- `/home/icebreaker/rss-az-cython2/tests/phases/conftest.py` - Test fixtures and assertion helpers

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies, established Cython patterns
- Architecture: HIGH - follows existing phase handler pattern exactly
- Pitfalls: HIGH - direct observation of codebase + auto-apply loop analysis
- Player reordering algorithm: HIGH - straightforward sort with documented tie-breaking

**Research date:** 2026-01-23
**Valid until:** 60 days (stable Cython patterns, no external dependencies)
