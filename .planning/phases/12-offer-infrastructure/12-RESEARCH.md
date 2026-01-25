# Phase 12: Offer Infrastructure - Research

**Researched:** 2026-01-25
**Domain:** Cython game state management, offer generation and sorting algorithms
**Confidence:** HIGH

## Summary

This phase implements offer infrastructure for the ACQUISITION phase of a board game engine. The research focuses on understanding existing codebase patterns for state management, entity handling, and phase transitions - all specific to this Cython game engine project.

The primary challenge is generating valid acquisition offers in sorted priority order with proper state tracking. The codebase already has established patterns for state layout, entity handles, and phase handlers that must be followed. Key decisions from CONTEXT.md are locked: fixed-size offer buffer (~250 slots), (corp_id, company_id) tuple storage, pre-computed sorting at phase entry.

**Primary recommendation:** Follow existing codebase patterns for entity handles and phase handlers. Implement offer generation in acquisition.pyx using cdef noexcept functions, store offers in hidden state as a fixed-size buffer, and use selection sort for deterministic ordering.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Cython | 3.x | Performance-critical game logic | Zero-overhead phase handlers via `cdef noexcept nogil` |
| NumPy | Current | State array storage | Float32 array for zero-copy PyTorch integration |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | Current | Testing | Verify offer generation, sorting, state tracking |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Fixed buffer | Dynamic list | Fixed buffer matches existing pattern, avoids Python object overhead |
| Selection sort | Quick sort | Selection sort is stable and explicit for small N (~250), matches existing player reordering pattern |

**Installation:** No new dependencies - uses existing project stack.

## Architecture Patterns

### Recommended Project Structure
```
phases/
    acquisition.pyx     # Offer generation, sorting, state management
core/
    state.pyx           # Add hidden state for offer buffer
    state.pxd           # Declare offer buffer structs
entities/
    player.pyx          # Add acquisition_proceeds field
    corp.pyx            # Already has acquisition_proceeds, acquisition_companies
    turn.pyx            # Existing acq_* state methods
```

### Pattern 1: Entity Handle Pattern
**What:** Global singleton instances initialized once, all methods take GameState as first argument
**When to use:** Any entity with per-game state stored in the state array
**Example:**
```cython
# Source: entities/corp.pyx
cdef class Corporation:
    cpdef void initialize(self, GameState state):
        """Initialize offsets from state layout. Call once when starting a new game."""
        cdef StateLayout layout = state._layout
        self._base_offset = layout.corps_offset + (self.corp_id * layout.corp_stride)
        # ... cache absolute offsets for each field

# Global instances
CORPS = {name: Corporation(i, name) for i, name in enumerate(CORP_NAMES)}
```

### Pattern 2: Phase Handler Pattern
**What:** `cdef noexcept` functions for zero-overhead phase execution
**When to use:** Any phase logic that must run at maximum speed
**Example:**
```cython
# Source: phases/wrap_up.pyx
cdef int apply_wrap_up(GameState state) noexcept:
    """Execute WRAP_UP phase logic. Returns: 0 always."""
    _reorder_players_by_cash(state)
    turn_module.TURN.clear_consecutive_passes(state)
    _process_fi_purchases(state)
    _make_all_revealed_available(state)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
    return 0
```

### Pattern 3: Selection Sort for Small Collections
**What:** O(n^2) selection sort with explicit tie-breaking
**When to use:** Sorting small collections (players n<=6, offers n<=250) where stability and explicit tie-breaking matter
**Example:**
```cython
# Source: phases/wrap_up.pyx lines 99-150
cdef void _reorder_players_by_cash(GameState state) noexcept:
    """Selection sort by (-cash, old_position)."""
    for i in range(num_players):
        best_idx = i
        for j in range(i + 1, num_players):
            # Higher cash wins, or if equal, lower old position wins
            if (curr_cash > best_cash or
                (curr_cash == best_cash and curr_pos < best_pos)):
                best_idx = j
        # Swap to front
        if best_idx != i:
            temp_id = player_ids[i]
            player_ids[i] = player_ids[best_idx]
            player_ids[best_idx] = temp_id
```

### Pattern 4: Hidden State for Internal Bookkeeping
**What:** State beyond visible_size offset is hidden from neural network
**When to use:** Internal tracking (deck order, offer buffer, active player)
**Example:**
```cython
# Source: core/state.pyx hidden state layout
# Hidden state layout:
# [0] active_player
# [1] num_players
# [2] deck_top
# [3..38] deck_order (36 slots)
# [39] phase (compact)
# ... etc
layout.hidden_active_player_offset = offset
```

### Pattern 5: While-Loop Re-Query for Dynamic State
**What:** Re-query state each iteration instead of snapshotting
**When to use:** When state changes during iteration (e.g., FI purchases)
**Example:**
```cython
# Source: phases/wrap_up.pyx
cdef void _process_fi_purchases(GameState state) noexcept:
    while True:
        company_id = _find_cheapest_affordable_available(state)
        if company_id < 0:
            break
        _fi_purchase_company(state, company_id)
```

### Anti-Patterns to Avoid
- **Python objects in hot paths:** Use cdef structs and C arrays, not Python lists
- **Snapshotting collections:** Re-query state each iteration for dynamic data
- **Circular imports:** Use `from entities import X as X_module` pattern
- **Modifying state before handle initialization:** Initialize all handles first

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Corporation lookup | Dict by name each time | `CORPS[corp_name]` global | Pre-instantiated, cached offsets |
| Company location query | Manual offset calculation | `company.get_location(state)` | Handles all location types |
| President lookup | Iterate players | `state.is_player_president(player_id, corp_id)` | One-hot encoded in state |
| Share price lookup | Raw state array access | `corp.get_share_price(state)` | Handles denormalization |
| Phase transition | Manual one-hot + compact | `turn_module.TURN.set_phase(state, GamePhases.PHASE_X)` | Updates both |

**Key insight:** The codebase has complete entity handles for all game objects. Always use the entity API, not raw state array access.

## Common Pitfalls

### Pitfall 1: Forgetting to Initialize Entity Handles
**What goes wrong:** Accessing entity methods before `initialize(state)` gives garbage offsets
**Why it happens:** Entity handles cache offsets from the state layout
**How to avoid:** Follow `initialize_game()` pattern - initialize ALL handles before setting ANY state
**Warning signs:** Tests fail with wrong values despite correct logic

### Pitfall 2: Modifying State Array Directly
**What goes wrong:** One-hot and hidden compact values get out of sync
**Why it happens:** State uses dual encoding for many fields (one-hot for NN, compact for logic)
**How to avoid:** Always use entity setters which update both encodings
**Warning signs:** Phase reads wrong, corp price index wrong, etc.

### Pitfall 3: Using Python Objects in Cdef Functions
**What goes wrong:** Cannot use `noexcept nogil` annotation, performance penalty
**Why it happens:** Natural to reach for Python lists/dicts
**How to avoid:** Use C arrays with fixed sizes, cdef structs
**Warning signs:** Cannot add `nogil`, compilation warnings

### Pitfall 4: Sorting Without Explicit Tie-Breaking
**What goes wrong:** Non-deterministic offer order breaks reproducibility
**Why it happens:** Share prices can be equal (bankrupt=0, game end=75)
**How to avoid:** Per CONTEXT.md: share prices are unique except 0/75. Face values are unique per company. Add secondary sort keys.
**Warning signs:** Different offer orders between runs

### Pitfall 5: Not Re-Validating Offers Before Presentation
**What goes wrong:** Present invalid offers (company already acquired, corp out of cash)
**Why it happens:** State changes between buffer population and offer presentation
**How to avoid:** Per CONTEXT.md: "Re-validate offers before presenting"
**Warning signs:** Invalid action masks, assertion failures

## Code Examples

Verified patterns from existing codebase:

### Iterating Active Corporations by Share Price
```cython
# Pattern for finding corporations sorted by descending share price
# Source: Derived from existing corp iteration patterns

cdef void _collect_corps_by_price(GameState state, int* corp_ids, int* prices, int* count) noexcept:
    """Collect active corps with their share prices for sorting."""
    cdef int corp_id, share_price
    count[0] = 0
    for corp_id in range(GameConstants.NUM_CORPS):
        if corp_module.CORPS[CORP_NAMES[corp_id]].is_active(state):
            corp_ids[count[0]] = corp_id
            prices[count[0]] = corp_module.CORPS[CORP_NAMES[corp_id]].get_share_price(state)
            count[0] += 1
```

### Checking FI Company Ownership
```cython
# Source: entities/fi.pyx
cdef bint _fi_owns_company(GameState state, int company_id) noexcept:
    return fi_module.FI.owns_company(state, company_id)
```

### Getting President for Corporation
```cython
# Pattern for finding president of a corp
cdef int _get_corp_president(GameState state, int corp_id) noexcept:
    """Get player_id of corp president, or -1 if in receivership."""
    cdef int player_id
    for player_id in range(state._num_players):
        if player_module.PLAYERS[player_id].is_president_of(state, corp_id):
            return player_id
    return -1  # Receivership
```

### Getting Companies Owned by Player
```cython
# Pattern for iterating player's private companies
cdef void _get_player_companies(GameState state, int player_id, int* companies, int* count) noexcept:
    """Collect company_ids owned by player."""
    count[0] = 0
    for company_id in range(GameConstants.NUM_COMPANIES):
        if player_module.PLAYERS[player_id].owns_company(state, company_id):
            companies[count[0]] = company_id
            count[0] += 1
```

### Hidden State Buffer Pattern
```cython
# Pattern for fixed-size buffer in hidden state
# Derived from hidden_deck_order pattern in state.pyx

# In StateLayout struct:
cdef struct StateLayout:
    # ... existing fields ...
    int hidden_offer_buffer_offset    # Start of offer buffer
    int hidden_offer_count_offset     # Current number of offers
    int hidden_offer_index_offset     # Current offer being processed

# Each offer is 2 floats: (corp_id, company_id)
DEF OFFER_BUFFER_SIZE = 250
DEF OFFER_STRIDE = 2
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Dynamic Python lists | Fixed C arrays in hidden state | v1 | Zero GC pressure, nogil compatible |
| Dict lookups in hot paths | Global entity handles | v1 | Cached offsets, O(1) access |
| Manual dual encoding | Entity setters | v1 | Consistent one-hot + compact |

**Deprecated/outdated:**
- None specific to this phase - following established v1-v3 patterns

## Open Questions

Things that couldn't be fully resolved:

1. **Exact hidden state offset calculation**
   - What we know: Hidden state follows visible state, uses float array
   - What's unclear: Exact offset position relative to existing hidden fields
   - Recommendation: Add after existing hidden fields, update `compute_layout()`

2. **Receivership corp active_player handling**
   - What we know: CONTEXT.md says "Claude's discretion"
   - What's unclear: Best practice for AlphaZero training
   - Recommendation: Use -1 or keep previous player; active_player is hidden state, not visible to NN

3. **Player acquisition_proceeds field location**
   - What we know: CONTEXT.md says "requires changes to state.pyx, VECTORS.md, entities/player.pyx"
   - What's unclear: Whether to add at end of player stride or insert between fields
   - Recommendation: Add at end of player stride to minimize offset disruption

## Sources

### Primary (HIGH confidence)
- `/home/icebreaker/rss-az-cython2/core/state.pyx` - State layout, hidden state pattern
- `/home/icebreaker/rss-az-cython2/phases/wrap_up.pyx` - Selection sort, while-loop re-query
- `/home/icebreaker/rss-az-cython2/entities/corp.pyx` - Entity handle pattern, acquisition fields
- `/home/icebreaker/rss-az-cython2/entities/player.pyx` - Player entity pattern
- `/home/icebreaker/rss-az-cython2/core/actions.pyx` - Action mask generation pattern
- `/home/icebreaker/rss-az-cython2/entities/turn.pyx` - Turn state accessors, acq_* methods
- `/home/icebreaker/rss-az-cython2/.planning/phases/12-offer-infrastructure/12-CONTEXT.md` - Locked decisions

### Secondary (MEDIUM confidence)
- `/home/icebreaker/rss-az-cython2/.planning/STATE.md` - Accumulated patterns
- `/home/icebreaker/rss-az-cython2/.planning/PROJECT.md` - Project constraints

### Tertiary (LOW confidence)
- None - all research based on actual codebase inspection

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - This is internal codebase, no external research needed
- Architecture: HIGH - Patterns derived directly from existing code
- Pitfalls: HIGH - Based on documented patterns and code inspection

**Research date:** 2026-01-25
**Valid until:** Indefinite - internal codebase patterns, not external dependencies
