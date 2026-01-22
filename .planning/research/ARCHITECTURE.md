# Architecture Research: Forced Action Auto-Application (v2.1)

**Researched:** 2026-01-21
**Confidence:** HIGH (based on direct codebase analysis)

## Current GameDriver Structure

### Overview

The `GameDriver` class in `core/driver.pyx` follows a **stateless singleton pattern** - all game state lives in the `GameState` object, and `GameDriver` is a dispatcher.

```
core/driver.pyx (93 lines)
  - class GameDriver (cdef class)
    - cpdef apply_action(state, action_idx) -> int (STATUS_OK/INVALID/GAME_OVER)
    - cpdef get_legal_moves(state) -> numpy array

core/driver.pxd
  - ActionStatus enum: STATUS_OK=0, STATUS_INVALID=1, STATUS_GAME_OVER=2
  - Declares cpdef methods for Python accessibility
```

### Current apply_action Flow

```python
cpdef int apply_action(self, GameState state, int action_idx):
    1. Validate action_idx bounds
    2. Get legal action mask via get_valid_action_mask(state)
    3. Check mask[action_idx] == 1.0, else return STATUS_INVALID
    4. Decode action via decode_action()
    5. Dispatch to phase handler based on state.get_phase()
    6. Check if game ended (PHASE_GAME_OVER), return STATUS_GAME_OVER
    7. Return result from phase handler
```

### Key Dependencies

| Module | What GameDriver Uses |
|--------|---------------------|
| `core/actions.pyx` | `get_valid_action_mask()`, `compute_action_layout()`, `decode_action()` |
| `core/actions.pxd` | `ActionLayout`, `ActionInfo` structs |
| `phases/invest.pyx` | `apply_invest_action()` |
| `phases/bid.pyx` | `apply_bid_action()` |

### Existing Forced Action Detection

`core/actions.pyx` already has `get_forced_action()` (lines 520-567):

```python
cpdef tuple get_forced_action(GameState state):
    """
    Returns:
        (action_idx, True) if exactly one valid action (forced)
        (-1, False) if zero or multiple valid actions
    """
```

This function:
1. Generates the action mask for current phase
2. Counts valid actions
3. Returns early if count > 1 (not forced)
4. Returns (single_action, True) if count == 1
5. Returns (-1, False) if count == 0 or > 1

**Key insight:** The detection logic exists but is not integrated into `apply_action()`.

## Integration Approach

### Recommendation: Loop Inside apply_action()

**Where:** Inside `GameDriver.apply_action()` method, wrapping the existing logic.

**Why inside (not wrapper):**
1. **Single entry point** - All callers automatically get auto-application
2. **Return value semantics** - STATUS_GAME_OVER still works correctly
3. **Existing pattern** - Phase handlers return success/failure, no change needed
4. **Performance** - No extra function call overhead per action

**Why iterative (not recursive):**
1. **Stack safety** - Long forced action chains possible (e.g., auction resolution -> everyone passes -> wrap-up)
2. **Cython optimization** - Iterative loops optimize better
3. **Predictable memory** - No risk of stack overflow

### Proposed apply_action Structure

```python
cpdef int apply_action(self, GameState state, int action_idx):
    cdef int result
    cdef int forced_action
    cdef bint is_forced
    cdef int legal_count

    # Apply the requested action (existing code)
    result = self._apply_single_action(state, action_idx)
    if result != STATUS_OK:
        return result

    # Check for game over after initial action
    if state.get_phase() == PHASE_GAME_OVER:
        return STATUS_GAME_OVER

    # Auto-apply forced actions iteratively
    while True:
        legal_count = _count_legal_actions(state)

        if legal_count == 0:
            # Error: no legal actions but not game over
            # This indicates a bug or unhandled game state
            return STATUS_INVALID  # Or could be a new STATUS_NO_MOVES

        if legal_count == 1:
            forced_action = _find_single_legal_action(state)
            result = self._apply_single_action(state, forced_action)
            if result != STATUS_OK:
                return result
            if state.get_phase() == PHASE_GAME_OVER:
                return STATUS_GAME_OVER
            continue  # Check again

        # legal_count >= 2: real choice exists, return to caller
        return STATUS_OK
```

### File Modifications

| File | Change | Scope |
|------|--------|-------|
| `core/driver.pyx` | Add helper methods, modify `apply_action()` | Main implementation |
| `core/driver.pxd` | Declare new cdef helpers | Declaration only |
| `core/actions.pyx` | Add `cdef` helpers if needed (or use existing) | Optional |
| `core/actions.pxd` | Declare any new cdef functions | Declaration only |

## Helper Method Design

### Option A: Helpers in driver.pyx (Recommended)

Keep helpers close to usage. Reuse existing `get_forced_action()` which already counts and finds.

```cython
# core/driver.pyx

cdef int _count_legal_actions(GameState state) noexcept:
    """Count valid actions in mask. Returns count."""
    cdef object mask = get_valid_action_mask(state)
    cdef int total = mask.shape[0]
    cdef float* mask_ptr = <float*>cnp.PyArray_DATA(mask)
    cdef int count = 0
    cdef int i

    for i in range(total):
        if mask_ptr[i] == 1.0:
            count += 1
    return count


cdef int _find_single_legal_action(GameState state) noexcept:
    """Find the single legal action. Assumes exactly 1 exists."""
    cdef object mask = get_valid_action_mask(state)
    cdef int total = mask.shape[0]
    cdef float* mask_ptr = <float*>cnp.PyArray_DATA(mask)
    cdef int i

    for i in range(total):
        if mask_ptr[i] == 1.0:
            return i
    return -1  # Should never reach if called correctly


cdef int _apply_single_action(GameDriver self, GameState state, int action_idx) noexcept:
    """Apply one action without auto-continuation. Internal use only."""
    # Current apply_action body moves here
    ...
```

### Option B: Leverage get_forced_action()

The existing `get_forced_action()` already does count + find in one pass:

```cython
cpdef int apply_action(self, GameState state, int action_idx):
    ...
    # After initial action applied
    while True:
        forced = get_forced_action(state)  # Returns (action_idx, is_forced)
        if forced[1]:  # is_forced == True
            result = self._apply_single_action(state, forced[0])
            ...
        else:
            if forced[0] == -1:
                # Either 0 or 2+ legal actions
                # Need separate check for 0 vs multiple
                ...
```

**Problem with Option B:** `get_forced_action()` returns (-1, False) for both 0 actions and 2+ actions. We need to distinguish these cases to catch the error condition.

### Option C: Modify get_forced_action() Return (Not Recommended)

Could change return to `(action_idx, count)` but this changes existing API.

### Recommendation: Option A with Combined Helper

Create one optimized helper that returns both count and action:

```cython
cdef struct ForcedActionResult:
    int action_idx   # -1 if not forced
    int legal_count  # 0, 1, or 2+ (stop counting at 2)

cdef ForcedActionResult _check_forced_action(GameState state) noexcept:
    """Check legal action count and find single action if forced."""
    cdef ForcedActionResult result
    cdef object mask = get_valid_action_mask(state)
    cdef int total = mask.shape[0]
    cdef float* mask_ptr = <float*>cnp.PyArray_DATA(mask)
    cdef int i

    result.action_idx = -1
    result.legal_count = 0

    for i in range(total):
        if mask_ptr[i] == 1.0:
            result.legal_count += 1
            if result.legal_count == 1:
                result.action_idx = i
            elif result.legal_count == 2:
                result.action_idx = -1  # Not forced
                return result  # Early exit: no need to count higher

    return result
```

## cdef vs cpdef Decisions

| Function | Visibility | Rationale |
|----------|-----------|-----------|
| `apply_action()` | `cpdef` | Must remain callable from Python (existing API) |
| `_apply_single_action()` | `cdef` | Internal only, max performance |
| `_check_forced_action()` | `cdef` | Internal only, returns C struct |
| `get_legal_moves()` | `cpdef` | Must remain callable from Python (existing API) |

## nogil Compatibility

**Current state:** The mask generation functions use Python objects (NumPy arrays) and cannot be `nogil`. This is acceptable because:

1. The GIL release would only matter for multi-threaded usage
2. Self-play training typically parallelizes at the process level (separate game instances)
3. The NumPy array pattern is established throughout the codebase

**Recommendation:** Do not attempt `nogil` for the auto-application loop. The mask generation requires GIL, and the performance benefit is minimal for the use case.

## Return Value Semantics

| Condition | Return | Meaning |
|-----------|--------|---------|
| Action applied, 2+ legal next | STATUS_OK (0) | Normal case |
| Action applied, chain leads to game over | STATUS_GAME_OVER (2) | Game ended |
| Invalid action_idx | STATUS_INVALID (1) | Caller error |
| Action not in mask | STATUS_INVALID (1) | Caller error |
| 0 legal actions (bug) | STATUS_INVALID (1) | Implementation error |

**Open question:** Should 0 legal actions be a distinct status code?

Options:
1. Keep STATUS_INVALID (simple, existing)
2. Add STATUS_NO_MOVES (explicit, easier debugging)
3. Raise exception (breaks Cython performance pattern)

**Recommendation:** Keep STATUS_INVALID for now. The only way to reach 0 legal actions is a bug in phase transition logic or mask generation. Add an assertion/warning in debug builds if needed.

## Implementation Order

### Phase 1: Helper Methods (Low Risk)

1. Add `ForcedActionResult` struct to `core/driver.pxd`
2. Add `_check_forced_action()` cdef function to `core/driver.pyx`
3. Add `_apply_single_action()` cdef function - extract current `apply_action()` body
4. Add declarations to `core/driver.pxd`

### Phase 2: apply_action Modification (Medium Risk)

5. Modify `apply_action()` to use iterative loop pattern
6. Build and test manually

### Phase 3: Test Updates (Required)

7. Update existing tests that check state after `apply_action()` - state may have advanced further
8. Add new tests for:
   - Forced action sequences (chain of single-action phases)
   - Zero legal actions detection (if reachable)
   - Long forced chains (stress test)

## Edge Cases to Handle

### Forced Chain Termination

The loop must terminate. Termination conditions:
- `legal_count == 0` - Error, return invalid
- `legal_count >= 2` - Normal exit, return OK
- `PHASE_GAME_OVER` - Game ended, return GAME_OVER

**Question:** Can infinite loops occur?

Analysis: Each action changes game state. The game has finite resources (cash, shares, companies) and all phases eventually advance. However, a buggy phase handler could theoretically oscillate between two states.

**Mitigation:** Add a max iteration guard (e.g., 1000 iterations). If exceeded, return STATUS_INVALID or raise. This catches bugs without affecting normal gameplay.

```cython
DEF MAX_FORCED_ITERATIONS = 1000

cpdef int apply_action(...):
    cdef int iterations = 0
    ...
    while iterations < MAX_FORCED_ITERATIONS:
        iterations += 1
        ...
    # If we exit loop via iterations, it's a bug
    return STATUS_INVALID
```

### Bankruptcy During Forced Chain

If a forced action causes bankruptcy, that bankruptcy is processed inline (existing pattern in `_handle_sell_share()`). The loop continues checking for more forced actions after bankruptcy completes.

### Phase Transitions

Phase handlers set the next phase. Some transitions might immediately have only one legal action:
- INVEST -> BID_IN_AUCTION (but multiple bid options usually exist)
- BID_IN_AUCTION -> INVEST (after auction resolution)
- INVEST -> WRAP_UP (when all pass - WRAP_UP not yet implemented)

The loop handles these naturally by checking after each action.

## Sources

All analysis from direct codebase reading:
- `/home/icebreaker/rss-az-cython2/core/driver.pyx` (lines 1-93)
- `/home/icebreaker/rss-az-cython2/core/driver.pxd` (lines 1-15)
- `/home/icebreaker/rss-az-cython2/core/actions.pyx` (lines 1-621, especially 520-567 for `get_forced_action`)
- `/home/icebreaker/rss-az-cython2/core/actions.pxd` (lines 1-113)
- `/home/icebreaker/rss-az-cython2/phases/invest.pyx` (lines 1-391)
- `/home/icebreaker/rss-az-cython2/phases/bid.pyx` (lines 1-118)
- `/home/icebreaker/rss-az-cython2/core/state.pxd` (lines 1-196)
- `/home/icebreaker/rss-az-cython2/.planning/PROJECT.md` (milestone context)

---

# Architecture Patterns: INVEST/BID_IN_AUCTION Phase Implementation (v2 Reference)

**Domain:** Cython game engine phase implementation
**Researched:** 2026-01-20

## Recommended Architecture

Phase implementations integrate with the existing Single-Vector State Machine architecture as new modules that mutate state through established entity accessors.

```
                                +-----------------+
                                |   Game Driver   |
                                | core/driver.pyx |
                                +-----------------+
                                        |
                    action_idx + state  |
                                        v
                        +-------------------------------+
                        |     Action Dispatch Layer     |
                        |  (decode action, route by     |
                        |   phase, call phase handler)  |
                        +-------------------------------+
                                        |
                                        v
        +-------------------+-------------------+-------------------+
        |                   |                   |                   |
        v                   v                   v                   v
+---------------+   +---------------+   +---------------+   +---------------+
| INVEST Phase  |   | BID_IN_AUCTION|   | ACQUISITION   |   | Other Phases  |
| phases/invest |   | Phase         |   | Phase         |   |     ...       |
|    .pyx       |   | (in invest.pyx|   | (future)      |   |               |
+---------------+   +---------------+   +---------------+   +---------------+
        |                   |                   |                   |
        v                   v                   v                   v
        +-------------------+-------------------+-------------------+
                                        |
                        +-------------------------------+
                        |     Entity Accessor Layer     |
                        |  PLAYERS, CORPS, TURN, FI,    |
                        |  DECK, MARKET, COMPANIES      |
                        +-------------------------------+
                                        |
                                        v
                        +-------------------------------+
                        |      GameState._data[]        |
                        |   (contiguous float32 array)  |
                        +-------------------------------+
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `core/driver.pyx` | Main game loop, action dispatch, forced action handling | actions.pyx, phases/*.pyx, state.pyx |
| `phases/invest.pyx` | INVEST phase logic (pass, start auction, buy/sell shares) + BID_IN_AUCTION phase logic (raise bid, leave auction) | Entity accessors, turn.pyx, data.pyx |
| `core/actions.pyx` | Action decoding, mask generation (already exists) | state.pyx, data.pyx |
| Entity accessors | State field access (already exist) | state.pyx |

### Data Flow

**Action Dispatch Flow:**

```
1. Driver receives (state, action_idx)
2. Driver reads phase from state: TURN.get_phase(state)
3. Driver decodes action: decode_action(&layout, action_idx) -> ActionInfo
4. Driver dispatches to phase handler based on phase:
   - PHASE_INVEST -> invest_phase_apply_action(state, &action_info)
   - PHASE_BID_IN_AUCTION -> bid_phase_apply_action(state, &action_info)
   - etc.
5. Phase handler mutates state via entity accessors
6. Phase handler may transition phase via TURN.set_phase()
7. Driver returns updated state
```

**INVEST Phase Action Flow:**

```
ACTION_PASS:
1. Increment consecutive_passes counter
2. Advance active player
3. If consecutive_passes == num_players: transition to PHASE_WRAP_UP

ACTION_AUCTION (slot, bid_offset):
1. Decode slot to company_id via get_auction_company_for_slot()
2. Compute bid = face_value + bid_offset
3. Deduct bid from player cash
4. Set auction state: company, price, high_bidder, starter
5. Remove company from auction pool
6. Transition to PHASE_BID_IN_AUCTION
7. Advance to next player for bidding

ACTION_BUY_SHARE (corp_id):
1. Find next higher available market space
2. Deduct buy_price from player cash
3. Add buy_price to corp cash
4. Transfer share: bank_shares--, player_shares++
5. Move corp price index up
6. Clear consecutive_passes (action taken)
7. Update round-trip tracking

ACTION_SELL_SHARE (corp_id):
1. Find next lower available market space
2. Add sell_price to player cash
3. Transfer share: player_shares--, bank_shares++
4. Move corp price index down
5. Handle president change if needed
6. Clear consecutive_passes
7. Update round-trip tracking
```

**BID_IN_AUCTION Phase Action Flow:**

```
ACTION_LEAVE_AUCTION:
1. Mark player as passed in auction
2. Find next non-passed player
3. If only high_bidder remains:
   - Award company to high_bidder
   - Transition back to PHASE_INVEST
   - Set active player to auction_starter for next action
4. Else: advance to next non-passed player

ACTION_RAISE_BID (bid_offset):
1. Compute new_bid = face_value + bid_offset + 1
2. Refund previous high_bidder (add old price to their cash)
3. Deduct new_bid from current player cash
4. Update auction state: price, high_bidder
5. Advance to next non-passed player
```

## Integration Points with Existing Components

### Existing Components (No Modifications Needed)

| Component | Integration Point | Notes |
|-----------|------------------|-------|
| `core/state.pyx` | GameState class, all getters/setters | Phases use via cpdef methods |
| `core/data.pyx` | GameConstants, GamePhases, company data | Import constants and lookup functions |
| `core/actions.pyx` | Action decoding, mask generation | Masks already implemented; decode_action() used by driver |
| `entities/player.pyx` | Player cash, shares, president status | Use PLAYERS[i].method(state) pattern |
| `entities/turn.pyx` | Phase, auction state, consecutive passes | Use TURN.method(state) pattern |
| `entities/corp.pyx` | Corp cash, bank shares, price index | Use CORPS['XX'].method(state) pattern |
| `entities/company.pyx` | Company location, transfer operations | Use COMPANIES[id].method(state) pattern |
| `entities/market.pyx` | Market space availability | Use MARKET.method(state) pattern |
| `entities/fi.pyx` | FI ownership (for auction winners) | Use FI.method(state) pattern |

### New Components Required

| Component | Purpose | Depends On |
|-----------|---------|------------|
| `phases/invest.pyx` | INVEST + BID_IN_AUCTION phase logic | state, data, actions, all entities |
| `phases/invest.pxd` | C-level declarations for invest.pyx | state.pxd, data.pxd, actions.pxd |
| `phases/__init__.pyx` | Package init, re-exports | invest.pyx |
| `phases/__init__.pxd` | Package declarations | invest.pxd |
| `core/driver.pyx` | Game loop, action dispatch | state, actions, phases |
| `core/driver.pxd` | C-level declarations for driver | state.pxd, actions.pxd |

### setup.py Modifications

The existing setup.py already scans `phases/` directory for .pyx files, so new phase modules will be automatically discovered and compiled. No changes needed to setup.py.

## Patterns to Follow

### Pattern 1: Phase Handler Function Signature

**What:** Phase handlers are cpdef functions that take state and action info, return nothing (mutate in place).

**When:** All phase implementations.

**Example:**

```cython
# phases/invest.pyx

from core.state cimport GameState
from core.actions cimport ActionInfo

cpdef void invest_phase_apply_action(GameState state, int action_type, int slot, int corp_id, int amount):
    """Apply an action during INVEST phase."""
    if action_type == ACTION_PASS:
        _handle_invest_pass(state)
    elif action_type == ACTION_AUCTION:
        _handle_start_auction(state, slot, amount)
    elif action_type == ACTION_BUY_SHARE:
        _handle_buy_share(state, corp_id)
    elif action_type == ACTION_SELL_SHARE:
        _handle_sell_share(state, corp_id)

cdef void _handle_invest_pass(GameState state) noexcept:
    """Handle pass action in INVEST phase."""
    from entities.turn import TURN

    TURN.increment_consecutive_passes(state)

    if TURN.get_consecutive_passes(state) >= state._num_players:
        # All players passed - end invest round
        TURN.set_phase(state, GamePhases.PHASE_WRAP_UP)
    else:
        _advance_to_next_player(state)
```

### Pattern 2: Entity Access Through Global Instances

**What:** Use pre-initialized entity singletons from entity modules.

**When:** All state access in phase handlers.

**Example:**

```cython
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.turn import TURN
from entities.company import COMPANIES

cdef void _handle_buy_share(GameState state, int corp_id) noexcept:
    cdef int player_id = state._get_active_player()
    cdef int buy_price
    cdef int current_index = CORPS_BY_ID[corp_id].get_price_index(state)

    # Find next higher available market space
    buy_price = _find_buy_price(state, current_index)

    # Mutate state through entity accessors
    PLAYERS[player_id].add_cash(state, -buy_price)
    CORPS_BY_ID[corp_id].add_cash(state, buy_price)
    CORPS_BY_ID[corp_id].set_bank_shares(state,
        CORPS_BY_ID[corp_id].get_bank_shares(state) - 1)
    PLAYERS[player_id].set_shares(state, corp_id,
        PLAYERS[player_id].get_shares(state, corp_id) + 1)
```

### Pattern 3: Phase Transition

**What:** Phase transitions via TURN.set_phase() with proper state cleanup.

**When:** End of phase, sub-phase transitions.

**Example:**

```cython
cdef void _complete_auction(GameState state) noexcept:
    """Award company to high bidder and return to INVEST."""
    from entities.turn import TURN
    from entities.company import COMPANIES

    cdef int company_id = TURN.get_auction_company(state)
    cdef int winner_id = TURN.get_auction_high_bidder(state)
    cdef int starter_id = TURN.get_auction_starter(state)

    # Transfer company to winner
    COMPANIES[company_id].transfer_to_player(state, winner_id)

    # Clear auction state
    TURN.clear_auction_company(state)
    TURN.clear_auction_high_bidder(state)
    TURN.clear_auction_passed(state)

    # Clear consecutive passes (auction counts as action)
    TURN.clear_consecutive_passes(state)

    # Return to INVEST phase
    TURN.set_phase(state, GamePhases.PHASE_INVEST)

    # Next action goes to player after auction starter
    _set_active_player_after(state, starter_id)
```

### Pattern 4: Driver Dispatch Pattern

**What:** Central dispatcher routes actions to phase handlers.

**When:** Main game loop.

**Example:**

```cython
# core/driver.pyx

from core.state cimport GameState
from core.actions cimport ActionLayout, ActionInfo, decode_action, compute_action_layout
from core.data cimport GamePhases

cpdef void apply_action(GameState state, int action_idx):
    """Apply an action to game state."""
    cdef ActionLayout layout = compute_action_layout(state._num_players)
    cdef ActionInfo info = decode_action(&layout, action_idx)
    cdef int phase = state.get_phase()

    if phase == GamePhases.PHASE_INVEST:
        from phases.invest import invest_phase_apply_action
        invest_phase_apply_action(state, info.action_type, info.slot, info.corp_id, info.amount)
    elif phase == GamePhases.PHASE_BID_IN_AUCTION:
        from phases.invest import bid_phase_apply_action
        bid_phase_apply_action(state, info.action_type, info.slot, info.corp_id, info.amount)
    # ... other phases
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Storing State in Phase Modules

**What:** Storing game state or mutable data at module level in phase files.

**Why bad:** Breaks thread-safety, causes cross-game contamination, violates stateless accessor pattern.

**Instead:** All state lives in GameState._data[]. Phase functions take state as parameter, never cache it.

### Anti-Pattern 2: Direct Array Access in Phase Logic

**What:** Accessing state._data[offset] directly in phase handlers.

**Why bad:** Fragile, layout-dependent, bypasses normalization, hard to maintain.

**Instead:** Use entity accessor methods: `PLAYERS[i].get_cash(state)`, `TURN.set_phase(state, phase)`.

### Anti-Pattern 3: Modifying actions.pyx Masks in Phase Modules

**What:** Having phase logic influence mask generation.

**Why bad:** Creates circular dependency, masks should be pure state queries.

**Instead:** Masks in actions.pyx query current state. Phase handlers only mutate state. Clear separation.

### Anti-Pattern 4: Python-Level Loops in Performance Paths

**What:** Using Python `for` loops or list comprehensions in hot paths.

**Why bad:** GIL overhead, slow iteration, prevents vectorization.

**Instead:** Use cdef functions with C-level loops and `noexcept nogil` where possible.

## Suggested Build Order

Based on dependencies, implement in this order:

### Phase 1: Core Infrastructure (Must be first)

1. **`phases/__init__.pxd`** - Empty package declaration file
2. **`phases/__init__.pyx`** - Empty package init (will grow with re-exports)
3. **`phases/invest.pxd`** - Function declarations for invest phase

### Phase 2: INVEST Phase Actions

4. **`phases/invest.pyx`** - INVEST phase implementation
   - `invest_phase_apply_action()` main handler
   - `_handle_invest_pass()` - Pass action
   - `_advance_to_next_player()` - Player rotation helper
   - Start with pass action only; test thoroughly

5. Add auction start handling:
   - `_handle_start_auction()` - Start auction action

6. Add share trading:
   - `_handle_buy_share()` - Buy share action
   - `_handle_sell_share()` - Sell share action
   - `_update_president()` - President change logic

### Phase 3: BID_IN_AUCTION Phase

7. Add bid phase handlers to invest.pyx:
   - `bid_phase_apply_action()` - Main bid phase handler
   - `_handle_leave_auction()` - Leave auction
   - `_handle_raise_bid()` - Raise bid
   - `_complete_auction()` - Award company, transition back

### Phase 4: Driver Integration

8. **`core/driver.pxd`** - Driver declarations
9. **`core/driver.pyx`** - Game driver implementation
   - `apply_action()` - Main entry point
   - `step()` - Single step including forced actions
   - `play_game()` - Full game loop (optional)

### Phase 5: Modify core/__init__.py

10. Export driver in core package (one-line addition)

### Dependency Graph

```
phases/__init__.pxd (new)
        |
        v
phases/invest.pxd (new)
        |
        +-- core/state.pxd (existing)
        +-- core/actions.pxd (existing)
        +-- core/data.pxd (existing)
        |
        v
phases/invest.pyx (new)
        |
        +-- entities/player.pyx (existing)
        +-- entities/turn.pyx (existing)
        +-- entities/corp.pyx (existing)
        +-- entities/company.pyx (existing)
        +-- entities/market.pyx (existing)
        +-- entities/fi.pyx (existing)
        |
        v
core/driver.pxd (new)
        |
        +-- core/state.pxd (existing)
        +-- core/actions.pxd (existing)
        |
        v
core/driver.pyx (new)
        |
        +-- phases/invest.pyx (new)
        +-- core/actions.pyx (existing)
```

## Testing Strategy

### Unit Tests for Phase Handlers

```
tests/
  test_invest_pass.py      - Pass action, consecutive passes, phase transition
  test_invest_auction.py   - Start auction, state setup
  test_invest_shares.py    - Buy/sell share, price movement, president
  test_bid_actions.py      - Raise bid, leave auction, award company
  test_driver.py           - Action dispatch, forced actions
```

### Integration Tests

```
tests/
  test_invest_round.py     - Full invest round: multiple actions, transitions
  test_auction_cycle.py    - Complete auction from start to award
  test_action_mask.py      - Verify masks match valid actions after state changes
```

## Sources

- `/home/icebreaker/rss-az-cython2/core/state.pyx` - GameState implementation (lines 1-802)
- `/home/icebreaker/rss-az-cython2/core/actions.pyx` - Action layout and mask generation (lines 1-608)
- `/home/icebreaker/rss-az-cython2/entities/turn.pyx` - TurnState entity (lines 1-489)
- `/home/icebreaker/rss-az-cython2/entities/player.pyx` - Player entity (lines 1-422)
- `/home/icebreaker/rss-az-cython2/.planning/codebase/ARCHITECTURE.md` - Existing architecture documentation
- `/home/icebreaker/rss-az-cython2/.planning/codebase/CONVENTIONS.md` - Coding conventions

---

*Architecture analysis: 2026-01-20, Updated: 2026-01-21 for v2.1 forced action auto-application*
