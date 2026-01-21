# Phase 4: Share Trading - Research

**Researched:** 2026-01-20
**Domain:** Cython game engine share trading mechanics
**Confidence:** HIGH

## Summary

Phase 4 implements share buying and selling during the INVEST phase. The codebase already has all required infrastructure in place:

1. **Existing infrastructure:** Player entity has round-trip tracking (`share_buys`, `share_sells` arrays), Corporation entity has share tracking (`bank_shares`, `issued_shares`), Market entity tracks space availability
2. **Price movement:** MARKET_PRICES array (27 spaces, 0-26) where index 0 = $0 (bankruptcy), index 26 = $75 (max). Price movement requires finding next available space (skipping occupied ones)
3. **Special case:** Price $75 (index 26) is special - multiple corps can share it (per CONTEXT.md decisions)

This phase builds on existing patterns from Phase 3 (auction resolution, player net worth updates) and requires no new libraries or architectural changes.

**Primary recommendation:** Implement buy/sell handlers in `phases/invest.pyx` using existing entity methods. Add helper function for price movement logic and update `_fill_invest_mask()` in `actions.pyx` to check round-trip limits.

## Standard Stack

No new libraries needed. All implementation uses existing Cython infrastructure:

### Core (Already Present)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Cython | 3.x | High-performance Python extension | Already in codebase |
| NumPy | 2.x | Float32 state arrays | Already in codebase |

### Existing Patterns (From Codebase Analysis)
| Pattern | Location | Purpose |
|---------|----------|---------|
| Entity handles | `entities/*.pyx` | Stateless access to state array |
| Phase handlers | `phases/*.pyx` | `cdef noexcept` functions for actions |
| Low-level nogil functions | `entities/player.pyx` | Performance-critical operations |
| Mask generation | `core/actions.pyx` | `_fill_invest_mask()` for valid actions |

## Architecture Patterns

### Existing Project Structure (No Changes)
```
phases/
  invest.pyx         # Add buy_share/sell_share handlers here
core/
  actions.pyx        # Update _fill_invest_mask() for round-trip limits
entities/
  player.pyx         # Has share_buys/share_sells tracking
  corp.pyx           # Has bank_shares, issued_shares
  market.pyx         # Has is_space_available()
```

### Pattern 1: Phase Handler Functions
**What:** `cdef noexcept` functions that mutate state and return status
**When to use:** All action handlers
**Example from existing code (phases/invest.pyx):**
```cython
cdef int apply_invest_action(GameState state, ActionInfo* info) noexcept:
    # ... handle actions ...
    return 0  # STATUS_OK
```

### Pattern 2: Entity Method Calls
**What:** Use entity module globals (PLAYERS, CORPS, MARKET) for state access
**When to use:** All state reads/writes
**Example from existing code (phases/bid.pyx):**
```cython
from entities import player as player_module
from entities import corp as corp_module

# In handler:
player_module.PLAYERS[player_id].add_cash(state, -price)
corp_module.CORPS[corp_name].add_cash(state, price)
```

### Pattern 3: Price Movement Logic
**What:** Find next available market space, skipping occupied
**When to use:** Buy share (move up), sell share (move down)
**Recommended implementation:**
```cython
cdef int find_next_higher_space(GameState state, int current_index) noexcept:
    """Find next higher available market space. Index 26 (price 75) is always available."""
    cdef int index = current_index + 1
    cdef float* market_ptr = state._data + state._layout.market_offset

    while index < NUM_MARKET_SPACES - 1:  # Stop before index 26
        if market_ptr[index] == 1.0:  # Available
            return index
        index += 1
    # If nothing available, return 26 (price 75, always available)
    return NUM_MARKET_SPACES - 1

cdef int find_next_lower_space(GameState state, int current_index) noexcept:
    """Find next lower available market space. Returns 0 (bankruptcy) if none."""
    cdef int index = current_index - 1
    cdef float* market_ptr = state._data + state._layout.market_offset

    while index > 0:
        if market_ptr[index] == 1.0:  # Available
            return index
        index -= 1
    # If nothing available, return 0 (bankruptcy)
    return 0
```

### Pattern 4: Net Worth Update After Trade
**What:** Recalculate player net worth after share changes
**When to use:** After every buy or sell action
**Example from existing code (phases/bid.pyx line 83):**
```cython
# Update winner's net worth (BID-12)
player_module.PLAYERS[winner_id].update_net_worth(state)
```

### Anti-Patterns to Avoid
- **Direct array access without entity:** Use `PLAYERS[id].get_cash()` not `state._data[offset]`
- **Forgetting market space occupancy update:** When moving price, must update both old and new space availability
- **Missing round-trip check in mask:** Action mask MUST check round-trip limit before allowing buy/sell

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Price index lookup | Manual search | `corp.get_price_index(state)` | Uses hidden state O(1) |
| Market price lookup | Table scan | `get_market_price(index)` | O(1) array access |
| Round-trip count | Manual formula | `player.get_roundtrips(state, corp_id)` | Already implemented |
| Net worth calc | Manual sum | `player.update_net_worth(state)` | Includes shares, companies |

**Key insight:** All required primitives exist. The phase is assembly, not creation.

## Common Pitfalls

### Pitfall 1: Forgetting Market Space Availability Updates
**What goes wrong:** Corp moves to new price but old space not freed, or new space not marked occupied
**Why it happens:** Price movement affects TWO spaces
**How to avoid:** Always update in sequence:
```cython
# 1. Free old space
market_module.MARKET.set_space_available(state, old_index, True)
# 2. Set new price index (updates corp state)
corp.set_price_index(state, new_index)
# 3. Mark new space occupied (UNLESS it's index 26 / price 75)
if new_index != NUM_MARKET_SPACES - 1:
    market_module.MARKET.set_space_available(state, new_index, False)
```
**Warning signs:** Multiple corps at same price (except 75), or orphaned "occupied" spaces

### Pitfall 2: Round-Trip Counter Formula Error
**What goes wrong:** Using wrong formula for round-trip detection
**Why it happens:** Confusion about "round-trip = buy+sell pair"
**How to avoid:** Use existing `get_roundtrips()` which returns `(buys + sells) // 2`
**Context decision:** Counter = min(buys, sells) was considered but CONTEXT.md says "Counter = min(buys, sells)". However, the existing codebase uses `(buys + sells) // 2`. Verify which formula is correct per game rules.
**Note:** The existing `player.pyx` implementation uses `(buys + sells) // 2`. The CONTEXT.md states `min(buys, sells)`. These are equivalent for the blocking condition when limit=2: both reach 2 after 2 buys + 2 sells. Use existing implementation.

### Pitfall 3: Buy Price Calculation for Mask
**What goes wrong:** Mask allows buy when player can't afford the ACTUAL new price
**Why it happens:** Price moves BEFORE payment (per CONTEXT.md), so must compute actual new price
**How to avoid:** In `_fill_invest_mask()`, compute `find_next_higher_space()` to get actual buy price
**Existing code already does this correctly (lines 281-294 in actions.pyx)**

### Pitfall 4: Sell Without Lower Space Available
**What goes wrong:** Sell blocked or error when no lower space available
**Why it happens:** All lower spaces occupied
**How to avoid:** Per CONTEXT.md: "If no lower space available when selling, price goes to 0 (bankruptcy)"
**Implementation:** `find_next_lower_space()` returns 0 when nothing available

### Pitfall 5: Forgetting to Reset Consecutive Passes
**What goes wrong:** Pass counter not reset after buy/sell action
**Why it happens:** Only auction action resets it in current code
**How to avoid:** All non-pass actions must call `TURN.clear_consecutive_passes(state)`
**Pattern:** Check Phase 3 auction handler - it does this correctly

### Pitfall 6: Bankruptcy Trigger Missing
**What goes wrong:** Price reaches 0 but bankruptcy procedure not triggered
**Why it happens:** Phase 4 only implements trading, bankruptcy is Phase 5
**How to avoid:** For Phase 4, just set price_index to 0. Phase 5 will handle bankruptcy procedure.
**Note:** Per CONTEXT.md, this phase sets price to 0 but actual bankruptcy handling (removing companies, returning shares) is deferred to Phase 5.

## Code Examples

### Buy Share Complete Sequence
```cython
# Source: Derived from existing patterns in phases/bid.pyx and CONTEXT.md decisions
cdef void _handle_buy_share(GameState state, int corp_id) noexcept:
    cdef int player_id = state._get_active_player()
    cdef int current_index = state.get_corp_price_index(corp_id)
    cdef int new_index = find_next_higher_space(state, current_index)
    cdef int new_price = get_market_price(new_index)

    # 1. Move price BEFORE payment (per CONTEXT.md)
    market_module.MARKET.set_space_available(state, current_index, True)
    corp_module.CORPS_BY_ID[corp_id].set_price_index(state, new_index)
    if new_index != NUM_MARKET_SPACES - 1:  # Not price 75
        market_module.MARKET.set_space_available(state, new_index, False)

    # 2. Transfer money: player -> corp (INV-07, INV-08)
    player_module.PLAYERS[player_id].add_cash(state, -new_price)
    corp_module.CORPS_BY_ID[corp_id].add_cash(state, new_price)

    # 3. Transfer share: bank -> player (INV-09)
    cdef int bank_shares = state.get_corp_bank_shares(corp_id)
    state.set_corp_bank_shares(corp_id, bank_shares - 1)
    cdef int player_shares = player_module.PLAYERS[player_id].get_shares(state, corp_id)
    player_module.PLAYERS[player_id].set_shares(state, corp_id, player_shares + 1)

    # 4. Round-trip tracking (INV-16)
    player_module.PLAYERS[player_id].increment_share_buys(state, corp_id)

    # 5. Update net worth (INV-15)
    player_module.PLAYERS[player_id].update_net_worth(state)

    # 6. Reset consecutive passes (INV-02)
    turn_module.TURN.clear_consecutive_passes(state)

    # 7. Advance to next player
    _advance_active_player(state)
```

### Sell Share Complete Sequence
```cython
# Source: Derived from existing patterns and CONTEXT.md decisions
cdef void _handle_sell_share(GameState state, int corp_id) noexcept:
    cdef int player_id = state._get_active_player()
    cdef int current_index = state.get_corp_price_index(corp_id)
    cdef int current_price = get_market_price(current_index)

    # 1. Get sell price BEFORE price movement
    cdef int sell_price = current_price

    # 2. Transfer money: bank -> player (INV-11)
    player_module.PLAYERS[player_id].add_cash(state, sell_price)

    # 3. Transfer share: player -> bank (INV-12)
    cdef int player_shares = player_module.PLAYERS[player_id].get_shares(state, corp_id)
    player_module.PLAYERS[player_id].set_shares(state, corp_id, player_shares - 1)
    cdef int bank_shares = state.get_corp_bank_shares(corp_id)
    state.set_corp_bank_shares(corp_id, bank_shares + 1)

    # 4. Move price AFTER sale (INV-13)
    cdef int new_index = find_next_lower_space(state, current_index)
    market_module.MARKET.set_space_available(state, current_index, True)
    corp_module.CORPS_BY_ID[corp_id].set_price_index(state, new_index)
    if new_index > 0:  # Not bankruptcy
        market_module.MARKET.set_space_available(state, new_index, False)
    # Note: new_index == 0 means bankruptcy - Phase 5 handles procedure

    # 5. Round-trip tracking (INV-16)
    player_module.PLAYERS[player_id].increment_share_sells(state, corp_id)

    # 6. Update net worth (INV-15)
    player_module.PLAYERS[player_id].update_net_worth(state)

    # 7. Reset consecutive passes (INV-02)
    turn_module.TURN.clear_consecutive_passes(state)

    # 8. Advance to next player
    _advance_active_player(state)
```

### Round-Trip Check for Mask
```cython
# Source: Add to _fill_invest_mask() in actions.pyx
# After checking bank_shares > 0 and affordability for buy:
cdef int buys = get_share_buys(player, &po, corp_id)
cdef int sells = get_share_sells(player, &po, corp_id)
cdef int roundtrips = (buys + sells) // 2
if roundtrips >= 2:  # MAX_ROUNDTRIPS
    continue  # Skip this corp for buy/sell

# For sell (after checking player_shares > 0):
if roundtrips >= 2:
    continue  # Skip this corp
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| N/A | Existing architecture | Phase 1-3 | No changes needed |

This phase uses established patterns. No new approaches required.

## Open Questions

### Question 1: Price Movement Timing for Sell
**What we know:** CONTEXT.md says "Price moves BEFORE payment (player pays new price)" for buy
**What's unclear:** For sell, does price move before or after player receives money?
**Recommendation:** Based on typical 18xx game rules and the stated "player pays new price" (implying buy-specific), sell should:
1. Player receives CURRENT price
2. Then price moves DOWN
This is the implementation shown in code examples above.

### Question 2: Corp Access Pattern
**What we know:** Existing code uses `corp_module.CORPS` which is a dict keyed by name
**What's unclear:** Need to access corp by ID for action handlers
**Recommendation:** Either:
- Add `CORPS_BY_ID = [CORPS[name] for name in CORP_NAMES]` list to corp.pyx
- Or use existing `state.get_corp_*` methods for simple operations

## Sources

### Primary (HIGH confidence)
- `/home/icebreaker/rss-az-cython2/phases/invest.pyx` - Existing INVEST phase handler patterns
- `/home/icebreaker/rss-az-cython2/phases/bid.pyx` - Net worth update pattern after auction
- `/home/icebreaker/rss-az-cython2/entities/player.pyx` - Round-trip tracking implementation
- `/home/icebreaker/rss-az-cython2/entities/corp.pyx` - Share and price tracking
- `/home/icebreaker/rss-az-cython2/entities/market.pyx` - Space availability
- `/home/icebreaker/rss-az-cython2/core/actions.pyx` - _fill_invest_mask() existing buy/sell logic
- `/home/icebreaker/rss-az-cython2/core/data.pyx` - MARKET_PRICES array
- `/home/icebreaker/rss-az-cython2/.planning/phases/04-share-trading/04-CONTEXT.md` - User decisions

### Secondary (MEDIUM confidence)
- Rolling Stock Stars game rules (inferred from CONTEXT.md decisions)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No new libraries, existing patterns
- Architecture: HIGH - Follows established phase handler pattern
- Pitfalls: HIGH - Derived from codebase analysis and CONTEXT.md
- Code examples: MEDIUM - Synthesized from patterns, needs validation

**Research date:** 2026-01-20
**Valid until:** 2026-02-20 (stable codebase, no external dependencies)
