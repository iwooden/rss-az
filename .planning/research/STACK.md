# Technology Stack: WRAP_UP Phase Implementation

**Project:** Rolling Stock Stars Cython Engine - WRAP_UP Phase
**Researched:** 2026-01-22
**Focus:** Player reordering, FI purchasing logic, company state transitions, phase transitions

## Executive Summary

The WRAP_UP phase requires NO new dependencies. All algorithms can be implemented using existing Cython patterns and NumPy utilities already in the codebase. The phase involves deterministic algorithms (player sorting, FI purchasing iteration) that map cleanly to the established `cdef noexcept` handler pattern.

**Key recommendation:** Use in-place sorting with temporary C arrays for player reordering, and sequential iteration over face values for FI purchases. Both fit naturally into the existing phase handler architecture.

## Recommended Stack

### No New Dependencies Required

The existing stack is sufficient:

| Component | Current Version | Purpose | WRAP_UP Usage |
|-----------|-----------------|---------|---------------|
| Cython | 3.0+ | Game logic compilation | Phase handler, sorting algorithms |
| NumPy | 2.0+ | State array operations | Array views for efficient data access |
| libc.stdlib | Standard C | qsort for player reordering | Fast in-place sorting |

**Rationale:** The WRAP_UP phase is purely algorithmic (sorting, iteration, state updates). No domain-specific libraries exist for board game logic, and the existing Cython patterns are optimal for performance-critical sorting and iteration.

### What NOT to Add

| Library/Approach | Why NOT |
|------------------|---------|
| Python `sorted()` | GIL acquisition overhead; incompatible with `noexcept nogil` |
| NumPy `argsort()` | Returns new array allocation; we need in-place reordering |
| Custom sorting algorithms | libc.stdlib.qsort is optimal for small N (≤6 players) |
| Python heapq | Pure Python; GIL overhead defeats performance goals |

**Rationale:** The codebase must maintain `noexcept nogil` in hot paths. WRAP_UP runs once per turn (low frequency compared to action application), but consistency with the existing architecture is critical.

## Player Reordering Algorithm

### Recommended: C qsort with Tie-Breaking

WRAP_UP Phase Rule: "Determine new Player Order by descending remaining money (ties broken by old player order)"

```cython
# In phases/wrap.pyx (new file)
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from libc.stdlib cimport qsort
from core.state cimport GameState
from entities import player as player_module

# Struct for sorting
cdef struct PlayerSortKey:
    int player_id
    int cash
    int old_order

cdef int _compare_players(const void* a, const void* b) noexcept nogil:
    """
    Compare function for qsort: descending cash, ascending old_order for ties.

    Returns:
        <0 if a should come before b
        >0 if a should come after b
        0 if equal (should not happen with tie-breaking)
    """
    cdef PlayerSortKey* pa = <PlayerSortKey*>a
    cdef PlayerSortKey* pb = <PlayerSortKey*>b

    # Primary: descending cash (higher cash = lower order position)
    if pa.cash != pb.cash:
        return pb.cash - pa.cash  # Reversed for descending

    # Tie-breaker: ascending old_order (lower old_order wins ties)
    return pa.old_order - pb.old_order

cdef void _reorder_players(GameState state) noexcept:
    """
    Update player turn order based on cash and old order.

    Algorithm:
    1. Collect (player_id, cash, old_order) for all players
    2. Sort by descending cash, ascending old_order for ties
    3. Assign new turn order positions
    """
    cdef int num_players = state._num_players
    cdef PlayerSortKey[6] sort_keys  # Max 6 players
    cdef int player_id, i

    # Gather sort keys
    for player_id in range(num_players):
        sort_keys[player_id].player_id = player_id
        sort_keys[player_id].cash = player_module.PLAYERS[player_id].get_cash(state)
        sort_keys[player_id].old_order = player_module.PLAYERS[player_id].get_turn_order(state)

    # Sort in-place
    qsort(&sort_keys[0], num_players, sizeof(PlayerSortKey), &_compare_players)

    # Apply new turn order
    for i in range(num_players):
        player_id = sort_keys[i].player_id
        player_module.PLAYERS[player_id].set_turn_order(state, i)
```

**Why this pattern:**
1. **Performance:** O(N log N) with highly optimized C library implementation
2. **Simplicity:** Single comparison function encapsulates all ordering logic
3. **Correctness:** Stable tie-breaking via old_order comparison
4. **No allocations:** Stack-allocated array (max 6 players = 72 bytes)
5. **Consistency:** Matches existing codebase's nogil philosophy

**Alternatives considered:**
- **Insertion sort:** Simpler but O(N²); no benefit for N≤6
- **NumPy argsort:** Requires GIL and array allocation
- **Manual bubble sort:** Error-prone; qsort is battle-tested

## Foreign Investor Purchasing Logic

### Recommended: Sequential Iteration by Face Value

WRAP_UP Phase Rule: "In ascending Face Value order, Foreign Investor buys as many available companies as possible at Face Value"

```cython
# In phases/wrap.pyx

from core.data cimport get_company_face_value, COMPANY_NAMES
from entities import fi as fi_module
from entities import company as company_module

cdef void _fi_purchase_companies(GameState state) noexcept:
    """
    Foreign Investor buys available companies in ascending face value order.

    For each available company (sorted by face value):
    1. If FI can afford face value, purchase it
    2. Draw and reveal new company, mark as unavailable (vertical)
    3. Continue until no affordable companies remain
    """
    cdef int company_id, face_value, fi_cash
    cdef object fi = fi_module.FI
    cdef object company

    # Companies are already sorted by face value (see core/data.pyx, COMPANY_NAMES)
    # Iterate in order: this gives us ascending face value automatically
    for company_id in range(36):  # GameConstants.NUM_COMPANIES
        company = company_module.COMPANIES[COMPANY_NAMES[company_id]]

        # Check if available for FI purchase
        if not company.is_for_auction(state):
            continue

        face_value = get_company_face_value(company_id)
        fi_cash = fi.get_cash(state)

        # Can FI afford it?
        if fi_cash < face_value:
            continue  # Skip this company, check next (might be cheaper)

        # Purchase: FI pays face value to bank
        fi.add_cash(state, -face_value)

        # Transfer company from auction to FI
        company.transfer_to_fi(state)

        # Draw new company and mark unavailable
        # (This will be implemented in deck entity)
        _draw_and_mark_unavailable(state)
```

**Why this pattern:**
1. **Simplicity:** Sequential iteration exploits pre-sorted company data
2. **No sorting needed:** Companies are already sorted by face value in `COMPANY_NAMES`
3. **Greedy algorithm:** FI buys all affordable companies (rules-compliant)
4. **No allocations:** Stateless iteration over existing data structures
5. **Early termination:** Can optimize by tracking if FI cash < min available face value

**Key insight:** The codebase already stores companies in face value order (see `core/data.pyx:33-39`). This means ascending face value iteration is simply `for company_id in range(36)`.

**Alternatives considered:**
- **Build sorted list of available companies:** Unnecessary allocation; we can filter during iteration
- **Priority queue:** Overkill for deterministic rules-based purchasing
- **Reverse iteration for descending:** Wrong - rules specify ascending face value

## Company Availability State Transitions

### Recommended: Batch Flag Updates

WRAP_UP Phase Rule: "After Foreign Investor done, all unavailable companies become available (turn horizontal)"

```cython
# In entities/company.pyx (extend existing)

cpdef void mark_unavailable(self, GameState state):
    """Mark company as unavailable (vertical) - drawn this turn."""
    # Set a flag in state array (exact location TBD based on state layout)
    # This may be implicit in auction vs revealed state
    state._data[self._revealed_offset] = 1.0
    state._data[self._auction_offset] = 0.0

cpdef void mark_available(self, GameState state):
    """Mark company as available (horizontal) - ready for auction."""
    state._data[self._auction_offset] = 1.0
    state._data[self._revealed_offset] = 0.0

# In phases/wrap.pyx

cdef void _make_all_companies_available(GameState state) noexcept:
    """
    Convert all unavailable (revealed) companies to available (auction).

    This is the final step of WRAP_UP: newly drawn companies become
    available for next turn's auctions.
    """
    cdef int company_id
    cdef object company

    for company_id in range(36):
        company = company_module.COMPANIES[COMPANY_NAMES[company_id]]
        if company.is_revealed(state):
            company.mark_available(state)
```

**Why this pattern:**
1. **Batch operation:** Single pass over all companies
2. **State locality:** Touches company state sequentially (cache-friendly)
3. **Explicit semantics:** Clear distinction between revealed (unavailable) and auction (available)
4. **Existing pattern:** Matches how company locations are tracked in current codebase

**State representation note:** The existing codebase tracks company locations with flags:
- `auction_companies_offset`: Available for auction
- `revealed_companies_offset`: Revealed but not yet available

The WRAP_UP phase simply flips these flags for companies drawn this turn.

## Phase Transition Mechanics

### Recommended: Existing Pattern Extension

The WRAP_UP phase handler follows the established pattern from INVEST/BID_IN_AUCTION:

```cython
# In phases/wrap.pyx

cdef int apply_wrap_action(GameState state, ActionInfo* action) noexcept:
    """
    Apply WRAP_UP phase action.

    WRAP_UP is fully deterministic (no player choices), so this is always
    a forced action that executes the full phase sequence:
    1. Reorder players by cash (tie-break by old order)
    2. FI purchases available companies (ascending face value)
    3. Make all unavailable companies available
    4. Transition to ACQUISITION phase

    Returns:
        STATUS_OK (0) on success
    """
    # Step 1: Reorder players
    _reorder_players(state)

    # Step 2: FI purchases
    _fi_purchase_companies(state)

    # Step 3: Make companies available
    _make_all_companies_available(state)

    # Step 4: Transition to next phase
    turn_module.TURN.set_phase(state, PHASE_ACQUISITION)

    return STATUS_OK
```

**Integration with GameDriver:**

```cython
# In core/driver.pyx (extend existing dispatch)

cdef int _apply_single_action(self, GameState state, int action_idx, object history):
    # ... existing code ...

    cdef int phase = state.get_phase()

    if phase == PHASE_INVEST:
        result = apply_invest_action(state, &info)
    elif phase == PHASE_BID_IN_AUCTION:
        result = apply_bid_action(state, &info)
    elif phase == PHASE_WRAP_UP:
        result = apply_wrap_action(state, &info)  # NEW
    else:
        return STATUS_INVALID
```

**Why this pattern:**
1. **Consistency:** Matches INVEST/BID_IN_AUCTION handler signatures
2. **Forced action:** WRAP_UP has no choices, so always auto-applied by GameDriver loop
3. **Atomic execution:** All four steps execute as single transaction
4. **Clear responsibilities:** Phase handler owns phase logic; GameDriver owns dispatch

## State Array Modifications

### Recommended: NO Changes to State Array Layout

The existing state array layout (see `core/state.pyx:53-200`) contains all necessary fields:

| Field | Current Location | WRAP_UP Usage |
|-------|------------------|---------------|
| Player turn_order | `players_offset + player_stride * player_id + turn_order` | Updated by `_reorder_players()` |
| Player cash | `players_offset + player_stride * player_id + cash` | Read for sorting key |
| FI cash | `fi_offset` | Decremented during purchases |
| FI owned_companies | `fi_offset + 1 + company_id` | Set when FI purchases |
| Company auction flags | `auction_companies_offset + company_id` | Cleared when FI purchases, set at end |
| Company revealed flags | `revealed_companies_offset + company_id` | Set when drawn, cleared at end |

**Key insight:** WRAP_UP reuses existing state fields. No new state is needed.

**Validation:** The existing `auction_companies` and `revealed_companies` arrays already track availability. WRAP_UP simply moves companies between these states.

## Cython-Specific Recommendations

### Use Established Patterns

1. **Phase handler signature:** `cdef int apply_wrap_action(GameState state, ActionInfo* action) noexcept`
2. **Entity module imports:** Use the pattern `from entities import player as player_module`
3. **Helper functions:** Prefix with `_` for internal use (e.g., `_reorder_players()`)
4. **Memory views:** Not needed - libc.stdlib.qsort operates on C arrays directly

### Performance Considerations

| Operation | Complexity | Frequency | Optimization |
|-----------|-----------|-----------|--------------|
| Player reordering | O(N log N) | Once per turn | Use qsort (optimal for N≤6) |
| FI purchasing | O(36) | Once per turn | Sequential iteration (companies pre-sorted) |
| Company state flip | O(36) | Once per turn | Batch update (single pass) |

**Expected performance:** Negligible impact on overall game throughput. WRAP_UP runs once per turn (~10-30 actions per turn), so even an O(36) operation is <1% of total runtime.

### Error Handling

WRAP_UP is deterministic and always succeeds (no invalid states possible):

```cython
# No error conditions to check:
# - Player sorting always succeeds (valid comparison function)
# - FI purchases are conditional (skip if unaffordable)
# - Company state transitions are simple flag flips

# Return STATUS_OK unconditionally
return STATUS_OK
```

**Rationale:** Unlike INVEST (can have invalid actions) or BID_IN_AUCTION (can have invalid bids), WRAP_UP has no failure modes. The phase executes deterministically based on current state.

## Module Structure

### Recommended File Organization

```
phases/
  __init__.pyx       (existing - add WRAP_UP import)
  invest.pyx         (existing)
  bid.pyx            (existing)
  wrap.pyx           (NEW - WRAP_UP phase handler)
```

**Contents of `phases/wrap.pyx`:**

1. **Imports:** Core entities (player, fi, company, turn, deck)
2. **Helper functions:** `_reorder_players()`, `_fi_purchase_companies()`, `_make_all_companies_available()`
3. **Main handler:** `apply_wrap_action()`
4. **Module initialization:** None needed (stateless)

**Integration points:**
- `core/driver.pyx`: Add dispatch case for `PHASE_WRAP_UP`
- `core/data.pyx`: Add `PHASE_WRAP_UP = 2` to `GamePhases` enum
- `phases/__init__.pyx`: Export `apply_wrap_action`

## Confidence Assessment

| Decision | Confidence | Rationale |
|----------|------------|-----------|
| No new dependencies | **HIGH** | All algorithms implementable with existing stack |
| qsort for player reordering | **HIGH** | Standard C library, optimal for small N |
| Sequential FI iteration | **HIGH** | Companies pre-sorted, rules are deterministic |
| No state array changes | **HIGH** | Existing fields sufficient (verified against layout) |
| Phase handler pattern | **HIGH** | Established pattern from INVEST/BID_IN_AUCTION |

## Open Questions

**None.** The WRAP_UP phase maps cleanly to existing patterns. All algorithms are deterministic and well-defined by the rules.

## Sources

- **Existing codebase patterns:** Analyzed `phases/invest.pyx`, `phases/bid.pyx`, `core/driver.pyx`
- **State layout:** Verified against `core/state.pyx:53-200`
- **Entity patterns:** Examined `entities/player.pyx`, `entities/fi.pyx`, `entities/company.pyx`
- **Game rules:** `RULES.md` lines 132-139 (Phase 2: Wrap-up specification)
- **C stdlib reference:** libc.stdlib.qsort (standard sorting algorithm)
