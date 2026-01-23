# Architecture Patterns: WRAP_UP Phase Integration

**Domain:** Rolling Stock Stars Cython Game Engine
**Researched:** 2026-01-22
**Focus:** WRAP_UP phase integration with existing architecture

## Executive Summary

The WRAP_UP phase integrates cleanly with the existing Cython game engine architecture. It follows established patterns from INVEST and BID_IN_AUCTION phases: a stateless phase handler function (`apply_wrap_up_action`) dispatched by GameDriver, with ForeignInvestor entity handle managing company purchases. The phase transitions from INVEST (all players pass) and proceeds to either ACQUISITION (if companies exist) or back to INVEST (new turn).

**Key Integration Points:**
1. Phase handler: `phases/wrap_up.pyx` following `cdef int apply_wrap_up_action(GameState state, ActionInfo* info) noexcept` pattern
2. GameDriver dispatch: Add WRAP_UP case to driver.pyx routing logic
3. Action encoding: ACTION_FI_BUY with company_id parameter in actions.pyx
4. Entity integration: ForeignInvestor entity (entities/fi.pyx) already supports company ownership
5. Phase transitions: INVEST -> WRAP_UP (trigger: consecutive_passes >= num_players), WRAP_UP -> next phase

## Recommended Architecture

### Component Structure

```
GameDriver (core/driver.pyx)
    ├─> Phase Dispatch Logic
    │   ├─> PHASE_INVEST -> apply_invest_action()
    │   ├─> PHASE_BID_IN_AUCTION -> apply_bid_action()
    │   └─> PHASE_WRAP_UP -> apply_wrap_up_action()  [NEW]
    │
    └─> Auto-Apply Loop
        └─> Continues through WRAP_UP forced actions

Phase Handler (phases/wrap_up.pyx) [NEW FILE]
    ├─> apply_wrap_up_action(state, info)
    │   ├─> Decode action (ACTION_FI_BUY or ACTION_PASS)
    │   ├─> Execute FI purchase logic
    │   └─> Transition to next phase
    │
    └─> Helper Functions
        ├─> _fi_buy_company(state, company_id)
        ├─> _get_next_fi_target(state) -> company_id
        └─> _transition_to_next_phase(state)

Entity Integration
    ├─> ForeignInvestor (entities/fi.pyx)
    │   ├─> get_cash() / set_cash()
    │   ├─> owns_company() / set_owns_company()
    │   └─> add_cash()  [existing methods]
    │
    ├─> Company (entities/company.pyx)
    │   ├─> transfer_to_fi(state)
    │   ├─> get_high_price()
    │   └─> is_for_auction()  [existing methods]
    │
    └─> TurnState (entities/turn.pyx)
        ├─> set_phase(state, phase)
        ├─> set_turn_number(state, turn)
        └─> clear_consecutive_passes(state)  [existing methods]

Action System (core/actions.pyx)
    ├─> ACTION_FI_BUY = 4  [NEW]
    │   └─> Encodes: company_id (0-35)
    │
    └─> Legal Action Generation
        ├─> In WRAP_UP: if FI has cash >= high_price
        ├─> ACTION_FI_BUY for each affordable auction company
        └─> ACTION_PASS always legal (FI declines to buy)
```

### Data Flow: WRAP_UP Phase Execution

```
1. Phase Entry (from INVEST)
   INVEST: all players pass
   └─> consecutive_passes >= num_players
       └─> TURN.set_phase(state, PHASE_WRAP_UP)
           └─> active_player unchanged (last player who passed)

2. FI Buying Loop
   GameDriver.apply_action(state, action_idx)
   └─> decode_action() -> ActionInfo
       └─> if ACTION_FI_BUY:
           ├─> _fi_buy_company(state, company_id)
           │   ├─> FI.add_cash(state, -high_price)
           │   ├─> COMPANIES[id].transfer_to_fi(state)
           │   └─> COMPANIES[id].clear_from_auction(state)
           │
           └─> Check if more companies affordable
               ├─> YES: Return (stay in WRAP_UP, auto-apply if forced)
               └─> NO: Fall through to phase transition

       └─> if ACTION_PASS:
           └─> FI declines to buy, proceed to phase transition

3. Phase Transition (exit WRAP_UP)
   _transition_to_next_phase(state)
   ├─> if any companies in FI ownership:
   │   └─> TURN.set_phase(state, PHASE_ACQUISITION)
   │       └─> Begin corporation acquisition offers
   │
   └─> else:
       └─> Start new turn
           ├─> TURN.increment_turn_number(state)
           ├─> TURN.clear_consecutive_passes(state)
           ├─> Rotate player order (future: WRAP_UP determines new first player)
           └─> TURN.set_phase(state, PHASE_INVEST)
```

## Integration Points with Existing Components

### 1. GameDriver Dispatch (core/driver.pyx)

**File:** `core/driver.pyx`
**Function:** `GameDriver._apply_single_action()`
**Lines:** ~110-120

**Current Pattern:**
```cython
if phase == PHASE_INVEST:
    result = apply_invest_action(state, &info)
elif phase == PHASE_BID_IN_AUCTION:
    result = apply_bid_action(state, &info)
else:
    return STATUS_INVALID
```

**Integration:**
```cython
# Add import at top
from phases.wrap_up cimport apply_wrap_up_action

# Add to dispatch logic
if phase == PHASE_INVEST:
    result = apply_invest_action(state, &info)
elif phase == PHASE_BID_IN_AUCTION:
    result = apply_bid_action(state, &info)
elif phase == PHASE_WRAP_UP:
    result = apply_wrap_up_action(state, &info)  # NEW
else:
    return STATUS_INVALID
```

**Why:** Follows established pattern. Phase handlers are stateless functions dispatched by phase ID.

---

### 2. Phase Transition from INVEST (phases/invest.pyx)

**File:** `phases/invest.pyx`
**Function:** `apply_invest_action()`
**Lines:** 337-350

**Current Implementation:**
```cython
if info.action_type == ACTION_PASS:
    turn_module.TURN.increment_consecutive_passes(state)

    if turn_module.TURN.get_consecutive_passes(state) >= state._num_players:
        # All players passed - end game
        # TODO(v3+): Replace with PHASE_WRAP_UP when implemented
        turn_module.TURN.set_phase(state, GamePhases.PHASE_GAME_OVER)
    else:
        _advance_active_player(state)

    return 0
```

**Integration Change:**
```cython
if turn_module.TURN.get_consecutive_passes(state) >= state._num_players:
    # All players passed - transition to WRAP_UP
    turn_module.TURN.set_phase(state, GamePhases.PHASE_WRAP_UP)
    # Active player remains last player who passed (for future turn order rotation)
else:
    _advance_active_player(state)
```

**Why:** WRAP_UP is triggered by all players passing consecutively. Remove GAME_OVER stub transition.

---

### 3. Action Encoding (core/actions.pyx)

**File:** `core/actions.pyx`
**Additions Needed:**

**Action Type Enum:**
```cython
cpdef enum ActionType:
    ACTION_PASS = 0
    ACTION_AUCTION = 1
    ACTION_BUY_SHARE = 2
    ACTION_SELL_SHARE = 3
    ACTION_FI_BUY = 4  # NEW - FI buys company in WRAP_UP
    ACTION_LEAVE_AUCTION = 5
    ACTION_RAISE_BID = 6
```

**Action Layout Update:**
```cython
cdef ActionLayout compute_action_layout(int num_players) noexcept nogil:
    cdef ActionLayout layout
    cdef int offset = 0

    layout.pass_action = offset
    offset += 1

    layout.auction_offset = offset
    layout.auction_count = MAX_AUCTION_SLOTS * (MAX_BID_AMOUNT + 1)
    offset += layout.auction_count

    # ... existing offsets ...

    # FI Buy actions (WRAP_UP phase only)
    layout.fi_buy_offset = offset
    layout.fi_buy_count = NUM_COMPANIES  # One action per company
    offset += layout.fi_buy_count

    layout.total_size = offset
    return layout
```

**Legal Action Generation (WRAP_UP):**
```python
def get_valid_action_mask(state: GameState) -> np.ndarray:
    # ... existing phase checks ...

    elif phase == GamePhases.PHASE_WRAP_UP:
        # FI can buy any auction company if affordable
        fi_cash = fi_module.FI.get_cash(state)

        for company_id in range(GameConstants.NUM_COMPANIES):
            company = company_module.COMPANIES[company_id]
            if company.is_for_auction(state):
                high_price = company.get_high_price()
                if fi_cash >= high_price:
                    # ACTION_FI_BUY for this company
                    action_idx = layout.fi_buy_offset + company_id
                    mask[action_idx] = 1.0

        # PASS always available (FI declines to buy)
        mask[layout.pass_action] = 1.0
```

**Why:** Follows established action encoding pattern. Each company gets a dedicated action slot.

---

### 4. ForeignInvestor Entity (entities/fi.pyx)

**File:** `entities/fi.pyx`
**Status:** Already complete - no changes needed

**Existing Interface:**
```cython
cpdef int get_cash(self, GameState state)
cpdef void set_cash(self, GameState state, int cash)
cpdef void add_cash(self, GameState state, int amount)
cpdef bint owns_company(self, GameState state, int company_id)
cpdef void set_owns_company(self, GameState state, int company_id, bint owns)
```

**Usage in WRAP_UP:**
```cython
# Purchase flow
fi_cash = fi_module.FI.get_cash(state)
high_price = company.get_high_price()
if fi_cash >= high_price:
    fi_module.FI.add_cash(state, -high_price)
    company_module.COMPANIES[company_id].transfer_to_fi(state)
```

**Why:** Existing entity handle pattern works perfectly. No new methods needed.

---

### 5. Company Entity (entities/company.pyx)

**File:** `entities/company.pyx`
**Status:** Already complete - no changes needed

**Existing Interface Used:**
```cython
cpdef bint is_for_auction(self, GameState state)
cpdef int get_high_price(self)
cpdef void transfer_to_fi(self, GameState state)
cpdef void clear_location(self, GameState state)
```

**Usage in WRAP_UP:**
```cython
# Check if company available for FI purchase
if company.is_for_auction(state):
    high_price = company.get_high_price()
    # ... affordability check ...
    company.transfer_to_fi(state)  # Atomic transfer
```

**Why:** Company entity handles all location transfers atomically. Existing methods sufficient.

---

### 6. TurnState Entity (entities/turn.pyx)

**File:** `entities/turn.pyx`
**Status:** Already complete - no changes needed for WRAP_UP core

**Existing Interface Used:**
```cython
cpdef void set_phase(self, GameState state, int phase)
cpdef void set_turn_number(self, GameState state, int turn)
cpdef void clear_consecutive_passes(self, GameState state)
cpdef int get_turn_number(self, GameState state)
```

**Usage in WRAP_UP:**
```cython
# Phase transitions
turn_module.TURN.set_phase(state, GamePhases.PHASE_WRAP_UP)
turn_module.TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)

# New turn setup (if skipping ACQUISITION)
turn_module.TURN.increment_turn_number(state)
turn_module.TURN.clear_consecutive_passes(state)
```

**Why:** TurnState already manages phase transitions and turn tracking. No new methods needed.

## New File Structure

### phases/wrap_up.pyx

**Purpose:** WRAP_UP phase handler - FI company purchasing logic
**Pattern:** Matches phases/invest.pyx and phases/bid.pyx structure

**Function Signature:**
```cython
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""WRAP_UP phase handler implementation."""

from core.state cimport GameState
from core.actions cimport ActionInfo, ActionType, ACTION_PASS, ACTION_FI_BUY
from entities import turn as turn_module
from entities import fi as fi_module
from entities import company as company_module
from core.data cimport GamePhases, GameConstants

cdef int apply_wrap_up_action(GameState state, ActionInfo* info) noexcept:
    """
    Apply WRAP_UP phase action to state.

    In WRAP_UP phase, the Foreign Investor (FI) has the opportunity to buy
    companies from the auction row at high price. FI buys companies sequentially
    until it either runs out of cash or chooses to pass.

    After FI buying completes:
    - If FI owns companies: transition to ACQUISITION phase
    - Otherwise: start new turn in INVEST phase

    Args:
        state: GameState to modify
        info: Decoded action (ACTION_FI_BUY or ACTION_PASS)

    Returns:
        0 = success, 1 = invalid
    """
    cdef int company_id, high_price, fi_cash

    if info.action_type == ACTION_FI_BUY:
        # FI buys company at high price
        company_id = info.company_id  # Decoded from action

        # Get company high price
        high_price = company_module.COMPANIES[company_id].get_high_price()

        # Deduct cash from FI
        fi_module.FI.add_cash(state, -high_price)

        # Transfer company to FI ownership
        company_module.COMPANIES[company_id].transfer_to_fi(state)

        # Stay in WRAP_UP phase - FI may buy more companies
        # Auto-apply will continue if only one affordable company remains
        return 0

    elif info.action_type == ACTION_PASS:
        # FI declines to buy (or no more affordable companies)
        # Transition to next phase
        _transition_to_next_phase(state)
        return 0

    return 1  # Invalid action type

cdef void _transition_to_next_phase(GameState state) noexcept:
    """
    Determine next phase after WRAP_UP completes.

    - If FI owns companies: ACQUISITION phase (corps make offers)
    - Otherwise: New turn in INVEST phase
    """
    cdef int company_id
    cdef bint fi_has_companies = False

    # Check if FI owns any companies
    for company_id in range(GameConstants.NUM_COMPANIES):
        if fi_module.FI.owns_company(state, company_id):
            fi_has_companies = True
            break

    if fi_has_companies:
        # FI has companies - begin ACQUISITION phase
        turn_module.TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
    else:
        # FI has no companies - start new turn
        _start_new_turn(state)

cdef void _start_new_turn(GameState state) noexcept:
    """
    Start a new turn after WRAP_UP with no FI companies.

    Future enhancement: Rotate player order based on WRAP_UP phase results.
    For now: Keep existing turn order, increment turn number.
    """
    # Increment turn number
    cdef int current_turn = turn_module.TURN.get_turn_number(state)
    turn_module.TURN.set_turn_number(state, current_turn + 1)

    # Clear consecutive passes for new turn
    turn_module.TURN.clear_consecutive_passes(state)

    # TODO(future): Rotate player order based on WRAP_UP results
    # For now, player order remains unchanged from initialization

    # Transition to INVEST phase
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)
```

**Why This Structure:**
- Matches existing phase handler pattern (cdef noexcept for performance)
- Delegates entity manipulation to entity handles (fi_module, company_module)
- Phase transition logic centralized in helper function
- Stateless - all state in GameState object

---

### phases/wrap_up.pxd

**Purpose:** Declaration file for wrap_up phase handler
**Pattern:** Matches phases/invest.pxd and phases/bid.pxd

```cython
# cython: language_level=3
"""WRAP_UP phase handler declarations."""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef int apply_wrap_up_action(GameState state, ActionInfo* info) noexcept
```

**Why:** Allows GameDriver to cimport the phase handler function.

## Phase Transition Flow

### Complete Phase Sequence

```
INVEST (players take actions)
    └─> All players pass
        └─> consecutive_passes >= num_players
            └─> WRAP_UP

WRAP_UP (FI buys companies)
    ├─> FI buys company A (ACTION_FI_BUY)
    │   └─> More affordable companies?
    │       ├─> YES: Stay in WRAP_UP (auto-apply if forced)
    │       └─> NO: Proceed to transition
    │
    ├─> FI passes (ACTION_PASS)
    │   └─> Proceed to transition
    │
    └─> Phase Transition Decision:
        ├─> FI owns companies?
        │   └─> YES: ACQUISITION (corps make offers for FI companies)
        │
        └─> NO: New turn
            ├─> Increment turn number
            ├─> Clear consecutive passes
            ├─> Reset player state for new turn
            └─> INVEST (first player acts)

ACQUISITION (if FI has companies)
    └─> [Future implementation]
        └─> After all acquisitions: New turn -> INVEST
```

### State Modifications

**INVEST -> WRAP_UP:**
```cython
# Trigger condition
if consecutive_passes >= num_players:
    # State changes
    TURN.set_phase(state, PHASE_WRAP_UP)
    # Note: active_player unchanged (last player who passed)
    # Note: consecutive_passes NOT cleared (preserved for debugging)
```

**WRAP_UP -> ACQUISITION:**
```cython
# Trigger condition
if FI owns at least one company:
    # State changes
    TURN.set_phase(state, PHASE_ACQUISITION)
    # Note: FI ownership flags remain set
    # Note: Companies removed from auction row (transferred to FI)
```

**WRAP_UP -> INVEST (new turn):**
```cython
# Trigger condition
if FI owns zero companies:
    # State changes
    TURN.increment_turn_number(state)
    TURN.clear_consecutive_passes(state)
    TURN.set_phase(state, PHASE_INVEST)
    # TODO(future): Rotate player order
    # Note: active_player reset to position 0 (or rotated first player)
```

## Player Order State Management

### Current Implementation

**State Storage:**
```cython
# Player turn_order field (one-hot encoding)
# In state vector: players_offset + player_id * player_stride + turn_order_field
# Size: num_players floats per player
# player_0: [1.0, 0.0, 0.0, 0.0] = position 0
# player_1: [0.0, 1.0, 0.0, 0.0] = position 1
# player_2: [0.0, 0.0, 1.0, 0.0] = position 2
# player_3: [0.0, 0.0, 0.0, 1.0] = position 3
```

**Access Pattern:**
```cython
# entities/player.pyx
cpdef int get_turn_order(self, GameState state):
    """Get player's position in turn order (0 to num_players-1)."""
    cdef int i
    for i in range(self._num_players):
        if state._data[self._turn_order_offset + i] == 1.0:
            return i
    return -1  # Should never happen

cpdef void set_turn_order(self, GameState state, int position):
    """Set player's position in turn order."""
    cdef int i
    for i in range(self._num_players):
        state._data[self._turn_order_offset + i] = 1.0 if i == position else 0.0
```

**Navigation:**
```cython
# entities/turn.pyx
cpdef int find_player_at_position(self, GameState state, int position):
    """Find which player_id is at given turn order position."""
    cdef int player_id
    for player_id in range(state._num_players):
        if player_module.PLAYERS[player_id].get_turn_order(state) == position:
            return player_id
    return -1
```

### WRAP_UP Player Order Changes

**Current Behavior:**
- Player order is set once at initialization (player i -> position i)
- Never rotated during game
- Active player advances through positions 0 -> 1 -> 2 -> ... -> (n-1) -> 0

**Future Enhancement (deferred):**
Player order rotation based on WRAP_UP results:
1. Last player to take action in INVEST phase before all-pass
2. Becomes first player for next turn
3. Turn order rotates accordingly

**Implementation (deferred to future milestone):**
```cython
cdef void _rotate_player_order(GameState state) noexcept:
    """
    Rotate player order so active_player becomes position 0.

    Example:
        Before: P0=pos0, P1=pos1, P2=pos2, P3=pos3, active=P2
        After:  P2=pos0, P3=pos1, P0=pos2, P1=pos3, active=P2
    """
    cdef int current_active = state._get_active_player()
    cdef int current_position = player_module.PLAYERS[current_active].get_turn_order(state)
    cdef int player_id, old_position, new_position

    # Rotate all players
    for player_id in range(state._num_players):
        old_position = player_module.PLAYERS[player_id].get_turn_order(state)
        new_position = (old_position - current_position + state._num_players) % state._num_players
        player_module.PLAYERS[player_id].set_turn_order(state, new_position)

    # Active player is now at position 0
    state._set_active_player(current_active)
```

**Why Deferred:**
- Not required for WRAP_UP core functionality
- Can add in future milestone without breaking changes
- Current behavior (fixed turn order) is simpler and sufficient for v3

## Auto-Apply Integration

### FI Buying Decision: Deterministic vs. Policy-Driven

**CRITICAL DESIGN QUESTION:** Should FI buying be deterministic (always buy cheapest) or policy-driven (NN chooses)?

**Option A: Deterministic FI Buying (RECOMMENDED)**
```cython
# FI always buys cheapest affordable company
# Repeat until out of cash or no affordable companies
# No NN decisions - fully automated
# Legal actions: [ACTION_PASS] only
# Auto-apply: Immediately triggers, entire WRAP_UP phase automated
```

**Option B: Policy-Driven FI Buying**
```cython
# FI buying is an NN decision
# Legal actions: [ACTION_FI_BUY(company_0), ..., ACTION_FI_BUY(company_n), ACTION_PASS]
# Auto-apply: Only when exactly one affordable company
# NN sees WRAP_UP states and makes purchase decisions
```

**Recommendation: Option A (Deterministic)**
- Matches typical board game NPC behavior
- Simpler implementation
- Reduces action space for NN
- Faster training (no NN evaluation during WRAP_UP)
- Can switch to Option B later if needed

### Auto-Apply Behavior (Deterministic Mode)

**Legal Action Mask:**
```python
elif phase == GamePhases.PHASE_WRAP_UP:
    # Only PASS action is exposed to NN
    # FI buying is deterministic and handled internally
    mask[layout.pass_action] = 1.0
```

**Phase Handler (Deterministic):**
```cython
cdef int apply_wrap_up_action(GameState state, ActionInfo* info) noexcept:
    """WRAP_UP phase is fully deterministic - no player/NN decisions."""
    if info.action_type == ACTION_PASS:
        # Execute full FI buying sequence deterministically
        _execute_fi_buying_sequence(state)
        _transition_to_next_phase(state)
        return 0
    return 1  # Only PASS is valid

cdef void _execute_fi_buying_sequence(GameState state) noexcept:
    """Buy companies from cheapest to most expensive until out of cash."""
    cdef int company_id, high_price, cheapest_id, cheapest_price, fi_cash

    while True:
        fi_cash = fi_module.FI.get_cash(state)
        cheapest_id = -1
        cheapest_price = 999999

        # Find cheapest affordable company
        for company_id in range(GameConstants.NUM_COMPANIES):
            if company_module.COMPANIES[company_id].is_for_auction(state):
                high_price = company_module.COMPANIES[company_id].get_high_price()
                if fi_cash >= high_price and high_price < cheapest_price:
                    cheapest_id = company_id
                    cheapest_price = high_price

        if cheapest_id < 0:
            break  # No affordable companies

        # Buy cheapest company
        fi_module.FI.add_cash(state, -cheapest_price)
        company_module.COMPANIES[cheapest_id].transfer_to_fi(state)
```

**Auto-Apply Flow:**
```
INVEST: All players pass
└─> Phase transitions to WRAP_UP
    └─> Auto-apply checks legal actions
        └─> Exactly 1 action: ACTION_PASS
            └─> Auto-apply triggers
                └─> apply_wrap_up_action(ACTION_PASS)
                    └─> _execute_fi_buying_sequence()
                    └─> _transition_to_next_phase()
                        ├─> -> ACQUISITION (if FI owns companies)
                        └─> -> INVEST (new turn, if FI owns nothing)
```

**Result:** NN never sees WRAP_UP phase. It's completely transparent to training loop.

## Patterns to Follow

### Pattern 1: Stateless Phase Handler

**What:** Phase handlers are cdef functions, not classes. All state in GameState.
**When:** Always for phase handlers
**Example:**
```cython
# Good - stateless function
cdef int apply_wrap_up_action(GameState state, ActionInfo* info) noexcept:
    # All state accessed via state parameter
    fi_cash = fi_module.FI.get_cash(state)
    return 0

# Bad - stateful class
cdef class WrapUpPhase:
    cdef int fi_cash  # Don't store state here!
```

### Pattern 2: Entity Handle Delegation

**What:** Delegate all state manipulation to entity handles (FI, Company, TurnState)
**When:** Always for state modifications
**Example:**
```cython
# Good - delegate to entity handle
fi_module.FI.add_cash(state, -high_price)
company_module.COMPANIES[company_id].transfer_to_fi(state)

# Bad - direct state manipulation
state._data[fi_cash_offset] -= high_price / CASH_DIVISOR
state._data[company_owner_offset + company_id] = FI_OWNER_ID
```

### Pattern 3: Phase Transition Centralization

**What:** Centralize phase transition logic in helper function
**When:** When multiple exit paths from phase
**Example:**
```cython
# Good - centralized transition
cdef void _transition_to_next_phase(GameState state) noexcept:
    if fi_has_companies:
        turn_module.TURN.set_phase(state, PHASE_ACQUISITION)
    else:
        _start_new_turn(state)

# Bad - scattered transitions
if fi_has_companies:
    turn_module.TURN.set_phase(state, PHASE_ACQUISITION)
else:
    turn_module.TURN.set_turn_number(state, turn + 1)
    # ... duplicate logic in multiple places
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Stateful Phase Handlers

**What:** Storing state in phase handler class instead of GameState
**Why Bad:** Breaks stateless singleton pattern, prevents state serialization
**Instead:** All state in GameState, phase handlers are pure functions

### Anti-Pattern 2: Direct State Array Manipulation

**What:** Accessing state._data offsets directly in phase handler
**Why Bad:** Breaks encapsulation, hard to maintain, error-prone
**Instead:** Use entity handle methods (FI.get_cash(), Company.transfer_to_fi())

### Anti-Pattern 3: Python-Level Logic in Phase Handler

**What:** Implementing phase logic in Python instead of Cython
**Why Bad:** Performance penalty, defeats purpose of Cython optimization
**Instead:** Keep phase handlers in .pyx files with cdef functions

### Anti-Pattern 4: Incomplete Phase Transitions

**What:** Changing phase without cleaning up related state
**Why Bad:** Leaves stale state, causes bugs in next phase
**Instead:** Clear all phase-specific state (e.g., consecutive_passes) during transition

## Scalability Considerations

### At 100 Games/Minute
**Approach:** Current architecture sufficient
- WRAP_UP is deterministic (no NN evaluation needed)
- Entity handle operations are O(1)
- Phase transition overhead negligible

### At 10K Games/Minute
**Approach:** Current architecture sufficient
- Cython phase handlers already optimized (noexcept, nogil where possible)
- FI buying loop is O(n) where n = auction row size (max 6)
- No allocations in hot path

### At 1M Games/Minute
**Approach:** Profile and optimize if needed
**Potential Optimization:**
- Pre-sort auction companies by price (avoid linear search)
- Batch FI purchases (single loop instead of one-at-a-time)
- Consider parallel game execution (multi-threading)

**Note:** WRAP_UP phase is not a bottleneck. Most time spent in NN evaluation during INVEST/BID phases.

## Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| GameDriver | Route actions to phase handlers, auto-apply loop | Phase handlers (invest, bid, wrap_up) |
| phases/wrap_up.pyx | WRAP_UP phase logic, FI buying, phase transition | Entities (FI, Company, Turn) |
| entities/fi.pyx | FI cash and company ownership state | GameState (via offsets) |
| entities/company.pyx | Company location and transfer operations | GameState, FI |
| entities/turn.pyx | Phase tracking, turn number, consecutive passes | GameState |
| core/actions.pyx | Action encoding/decoding, legal action mask | Phase handlers (via ActionInfo) |

**Interface Contract:**
- Phase handlers receive `ActionInfo*` pointer (decoded action)
- Phase handlers return `int` (0=success, 1=invalid)
- Phase handlers modify `GameState` via entity handles only
- GameDriver handles STATUS_OK/STATUS_INVALID/STATUS_GAME_OVER logic

## Testing Strategy

### Unit Tests (tests/phases/test_wrap_up.py)

**Coverage:**
1. FI buys single company (basic flow)
2. FI buys multiple companies (iteration)
3. FI insufficient cash (no purchases)
4. FI buys cheapest companies first (ordering)
5. Phase transition to ACQUISITION (FI owns companies)
6. Phase transition to INVEST (FI owns no companies)
7. Turn number increment on new turn
8. Consecutive passes cleared on new turn
9. Action mask correctness (PASS only in deterministic mode)
10. Auto-apply behavior (WRAP_UP completely transparent)

**Pattern:**
```python
def test_fi_buys_single_company(game_state):
    """FI buys one company and transitions to ACQUISITION."""
    # Setup: FI has $50, auction has company worth $30
    fi_module.FI.set_cash(game_state, 50)
    company_module.COMPANIES[0].move_to_auction(game_state)
    # Company 0 has high_price = 30 (from game data)

    # Execute WRAP_UP (deterministic - auto-applies)
    turn_module.TURN.set_phase(game_state, GamePhases.PHASE_INVEST)
    # ... simulate all players passing ...
    # Phase auto-transitions to WRAP_UP and completes

    # Verify
    assert fi_module.FI.get_cash(game_state) == 20  # 50 - 30
    assert fi_module.FI.owns_company(game_state, 0)
    assert game_state.get_phase() == GamePhases.PHASE_ACQUISITION
```

### Integration Tests

**Coverage:**
1. Full turn: INVEST (all pass) -> WRAP_UP -> INVEST (new turn)
2. Full turn: INVEST -> WRAP_UP -> ACQUISITION
3. Auto-apply behavior through WRAP_UP
4. Invariants maintained (cash conservation, company location uniqueness)
5. Multiple turns with alternating WRAP_UP outcomes

## Sources

**Codebase Analysis (HIGH confidence):**
- /home/icebreaker/rss-az-cython2/core/driver.pyx (lines 1-191) - GameDriver dispatch, auto-apply loop
- /home/icebreaker/rss-az-cython2/phases/invest.pyx (lines 1-392) - Phase handler pattern, INVEST->WRAP_UP transition stub
- /home/icebreaker/rss-az-cython2/phases/bid.pyx (lines 1-118) - Phase handler reference implementation
- /home/icebreaker/rss-az-cython2/entities/fi.pyx (lines 1-75) - ForeignInvestor entity complete interface
- /home/icebreaker/rss-az-cython2/entities/company.pyx (lines 1-384) - Company entity transfer operations
- /home/icebreaker/rss-az-cython2/entities/turn.pyx (lines 1-150+) - TurnState entity phase management
- /home/icebreaker/rss-az-cython2/entities/player.pyx (lines 1-200) - Player turn_order implementation
- /home/icebreaker/rss-az-cython2/core/state.pyx (lines 1-828) - State layout, player turn_order storage
- /home/icebreaker/rss-az-cython2/core/data.pxd (lines 1-96) - GamePhases enum (PHASE_WRAP_UP = 2)

**Documentation (HIGH confidence):**
- /home/icebreaker/rss-az-cython2/.planning/PROJECT.md - Architecture patterns, entity handle pattern
- /home/icebreaker/rss-az-cython2/.planning/research/STACK.md - Technology stack, phase transition flow
- /home/icebreaker/rss-az-cython2/.planning/research/PITFALLS.md - Phase transition edge cases, auto-apply warnings
- /home/icebreaker/rss-az-cython2/.planning/research/FORCED_ACTION_FEATURES.md - Auto-apply behavior specification
- /home/icebreaker/rss-az-cython2/.planning/phases/03-invest-core-auction-flow/03-RESEARCH.md - INVEST phase design
- /home/icebreaker/rss-az-cython2/.planning/phases/07-core-implementation/07-RESEARCH.md - Auto-apply loop implementation

**Overall Confidence: HIGH**
- All integration points verified in existing code
- Entity interfaces confirmed complete (no new methods needed)
- Phase handler pattern established (2 working implementations)
- Auto-apply behavior specified and implemented
- WRAP_UP stub already exists in codebase (PHASE_WRAP_UP = 2 in enum)
- No unknowns or external dependencies
