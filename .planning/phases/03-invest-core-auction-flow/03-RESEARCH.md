# Phase 3: INVEST Core & Auction Flow - Research

**Researched:** 2026-01-20
**Domain:** Cython game state mutation, auction mechanics, turn order management
**Confidence:** HIGH

## Summary

Phase 3 implements the core INVEST phase actions (pass, start auction) and full BID_IN_AUCTION flow. The codebase already has:

1. **Complete infrastructure** - GameDriver dispatches to phase handlers, legal move masks work
2. **Action encoding** - `decode_action()` returns `ActionInfo` with `slot`, `amount` fields for auctions
3. **State fields** - All auction state fields exist (company, price, high_bidder, starter, passed flags)
4. **Entity handles** - TurnState, Player, Company handles provide clean access patterns
5. **Stub handlers** - `phases/invest.pyx` and `phases/bid.pyx` contain minimal stubs ready for implementation

The implementation requires filling in the stub handlers using existing entity APIs. No new state fields or action encoding changes needed.

**Primary recommendation:** Implement the phase handlers by following the existing entity handle patterns, using TurnState for auction state and Player handles for cash/ownership operations.

## Standard Stack

### Core (Existing - No Changes)

| Module | Purpose | Status |
|--------|---------|--------|
| `phases/invest.pyx` | INVEST phase handler stub | Exists, needs implementation |
| `phases/bid.pyx` | BID_IN_AUCTION handler stub | Exists, needs implementation |
| `entities/turn.pyx` | Auction state access (TURN handle) | Complete API |
| `entities/player.pyx` | Player cash, company ownership | Complete API |
| `entities/company.pyx` | Company transfer operations | Complete API |
| `entities/deck.pyx` | Draw new company on auction win | Complete API |
| `core/actions.pyx` | Action decoding, mask generation | Complete |
| `core/driver.pyx` | Action dispatch | Complete |

### Supporting (No Dependencies to Add)

The codebase is self-contained Cython. No new imports needed.

## Architecture Patterns

### Recommended Implementation Structure

```
phases/
  invest.pyx      # Apply INVEST actions: pass, auction, (buy/sell in Phase 4)
  bid.pyx         # Apply BID actions: leave, raise, resolve auction
```

### Pattern 1: Phase Handler Function

**What:** `cdef int apply_X_action(GameState state, ActionInfo* info) noexcept`
**When to use:** All phase handlers follow this signature
**Example:**
```cython
# Source: phases/invest.pyx (existing stub)
cdef int apply_invest_action(GameState state, ActionInfo* info) noexcept:
    if info.action_type == ACTION_PASS:
        # 1. Increment consecutive passes
        turn_module.TURN.increment_consecutive_passes(state)
        # 2. Check for phase transition
        if turn_module.TURN.get_consecutive_passes(state) >= state._num_players:
            turn_module.TURN.set_phase(state, GamePhases.PHASE_WRAP_UP)
        else:
            # 3. Advance to next player in turn order
            _advance_active_player(state)
        return 0
    # ... other action types
```

### Pattern 2: Entity Handle Access

**What:** Import entity modules and use global handles (TURN, PLAYERS, COMPANIES)
**When to use:** Any state mutation
**Example:**
```cython
# Import pattern (avoid circular imports by importing module, not class)
from entities import turn as turn_module
from entities import player as player_module
from entities import company as company_module

# Usage
turn_module.TURN.set_auction_company(state, company_id)
player_module.PLAYERS[player_id].add_cash(state, -amount)
company_module.COMPANIES[company_id].transfer_to_player(state, player_id)
```

### Pattern 3: Turn Order Navigation

**What:** Find player with turn_order position N
**When to use:** Advancing active player, finding next after auction starter
**Example:**
```cython
cdef int _find_player_at_turn_position(GameState state, int position) noexcept:
    """Find player_id with turn_order == position."""
    cdef int player_id
    for player_id in range(state._num_players):
        if player_module.PLAYERS[player_id].get_turn_order(state) == position:
            return player_id
    return -1  # Should never happen

cdef int _get_next_turn_position(GameState state, int current_position) noexcept:
    """Get next turn order position (wraps around)."""
    return (current_position + 1) % state._num_players
```

### Pattern 4: Auction State Management

**What:** Use TurnState methods for all auction fields
**When to use:** Start auction, bid, resolve auction
**Example:**
```cython
# Start auction
turn_module.TURN.set_auction_company(state, company_id)
turn_module.TURN.set_auction_price(state, bid_price)
turn_module.TURN.set_auction_high_bidder(state, player_id)
turn_module.TURN.set_auction_starter(state, player_id)
turn_module.TURN.clear_auction_passed(state)

# Leave auction
turn_module.TURN.set_player_passed_auction(state, player_id, True)

# Clear auction (after resolution)
turn_module.TURN.clear_auction_company(state)
turn_module.TURN.clear_auction_high_bidder(state)
turn_module.TURN.clear_auction_starter(state)
turn_module.TURN.clear_auction_passed(state)
turn_module.TURN.set_auction_price(state, -1)
```

### Anti-Patterns to Avoid

- **Direct state array access in phase handlers:** Use entity handles, not `state._data[offset]`
- **Modifying player turn order during INVEST:** Turn order only changes in WRAP_UP phase
- **Forgetting to advance active player after pass:** Every action must update who acts next
- **Not clearing consecutive_passes on non-pass actions:** INV-02 requires this

## Don't Hand-Roll

Problems that have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Auction state access | Direct offset math | `turn_module.TURN.*` methods | Handles one-hot encoding |
| Company transfer | Manual flag clearing | `company.transfer_to_player()` | Clears old location atomically |
| Player cash change | Direct `_data` access | `player.add_cash(state, amount)` | Handles normalization |
| Drawing new company | Manual deck pointer | `deck_module.DECK.draw(state)` | Returns company_id, updates state |
| Finding player by turn order | Linear scan each time | Cache or utility function | Will be called frequently |

**Key insight:** The entity handles encapsulate all the normalization divisors (CASH_DIVISOR, etc.) and one-hot encoding. Using them prevents subtle bugs from incorrect scaling.

## Common Pitfalls

### Pitfall 1: Active Player vs Turn Order Position

**What goes wrong:** Confusing player_id (0-5) with turn_order position (0-5)
**Why it happens:** Both are small integers, easy to mix up
**How to avoid:**
- `state._get_active_player()` returns player_id
- `player.get_turn_order(state)` returns position
- Always think: "Is this a WHO or a WHEN?"
**Warning signs:** Auction resolution giving turn to wrong player

### Pitfall 2: Bidder Rotation in Auction

**What goes wrong:** Rotating through player_ids instead of turn_order positions
**Why it happens:** Natural to iterate `for player_id in range(num_players)`
**How to avoid:**
1. Get current active player's turn_order position
2. Increment position (mod num_players)
3. Find player at that position
4. If passed, continue to next position
**Warning signs:** Players bidding out of order

### Pitfall 3: Auction Slot to Company ID

**What goes wrong:** Using slot directly as company_id
**Why it happens:** Slot is just an index (0, 1, 2, ...)
**How to avoid:** Use `get_auction_company_for_slot(state, slot)` from `entities/company.pyx`
**Warning signs:** Auctioning wrong company

### Pitfall 4: Consecutive Passes Normalization

**What goes wrong:** Treating consecutive_passes as raw integer
**Why it happens:** Stored normalized by num_players
**How to avoid:** Always use `TURN.get_consecutive_passes(state)` / `TURN.set_consecutive_passes(state, n)`
**Warning signs:** Phase never transitions to WRAP_UP, or transitions too early

### Pitfall 5: Auction Winner Calculation

**What goes wrong:** Checking "one bidder remains" by counting active players
**Why it happens:** Confusing active player count with non-passed bidder count
**How to avoid:**
```cython
cdef int _count_active_bidders(GameState state) noexcept:
    cdef int count = 0
    for player_id in range(state._num_players):
        if not turn_module.TURN.has_player_passed_auction(state, player_id):
            count += 1
    return count
```
**Warning signs:** Auction resolves prematurely or never resolves

## Code Examples

### Starting an Auction (INV-05, INV-06)

```cython
# Source: Derived from existing API patterns

cdef int _start_auction(GameState state, int slot, int bid_offset) noexcept:
    """Start auction for company at slot with given bid offset over face value."""
    cdef int company_id = get_auction_company_for_slot(state, slot)
    if company_id < 0:
        return 1  # Invalid slot

    cdef int face_value = get_company_face_value(company_id)
    cdef int bid_price = face_value + bid_offset
    cdef int player_id = state._get_active_player()

    # Initialize auction state
    turn_module.TURN.set_auction_company(state, company_id)
    turn_module.TURN.set_auction_price(state, bid_price)
    turn_module.TURN.set_auction_high_bidder(state, player_id)
    turn_module.TURN.set_auction_starter(state, player_id)
    turn_module.TURN.clear_auction_passed(state)  # Nobody has left yet
    turn_module.TURN.clear_consecutive_passes(state)  # Reset passes (INV-02)

    # Transition to BID phase
    turn_module.TURN.set_phase(state, GamePhases.PHASE_BID_IN_AUCTION)

    # Set next bidder (player after auction starter in turn order)
    _advance_to_next_bidder(state)

    return 0
```

### Resolving an Auction (BID-05 through BID-12)

```cython
# Source: Derived from existing API patterns

cdef int _resolve_auction(GameState state) noexcept:
    """Resolve auction - winner pays, gets company, draws new company."""
    cdef int winner_id = turn_module.TURN.get_auction_high_bidder(state)
    cdef int starter_id = turn_module.TURN.get_auction_starter(state)
    cdef int company_id = turn_module.TURN.get_auction_company(state)
    cdef int price = turn_module.TURN.get_auction_price(state)

    # Winner pays (BID-06)
    player_module.PLAYERS[winner_id].add_cash(state, -price)

    # Winner receives company (BID-07)
    company_module.COMPANIES[company_id].transfer_to_player(state, winner_id)

    # Update winner's net worth (BID-12)
    player_module.PLAYERS[winner_id].update_net_worth(state)

    # Draw new company to auction row (BID-09)
    cdef int new_company = deck_module.DECK.draw(state)
    if new_company >= 0:
        state.set_company_for_auction(new_company, True)

    # Clear auction state (BID-08)
    turn_module.TURN.clear_auction_company(state)
    turn_module.TURN.clear_auction_high_bidder(state)
    turn_module.TURN.clear_auction_starter(state)
    turn_module.TURN.clear_auction_passed(state)
    turn_module.TURN.set_auction_price(state, -1)

    # Transition back to INVEST (BID-10)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)

    # Next action goes to player after starter (BID-11) - NOT the winner!
    _set_active_player_after(state, starter_id)

    return 0
```

### Finding Next Player in Turn Order

```cython
# Source: Utility needed for INV-04, INV-04a, BID-02, BID-11

cdef int _get_player_turn_position(GameState state, int player_id) noexcept:
    """Get turn order position (0 to num_players-1) for a player."""
    return player_module.PLAYERS[player_id].get_turn_order(state)

cdef int _find_player_at_position(GameState state, int position) noexcept:
    """Find player_id with given turn order position."""
    cdef int player_id
    for player_id in range(state._num_players):
        if player_module.PLAYERS[player_id].get_turn_order(state) == position:
            return player_id
    return -1

cdef void _set_active_player_after(GameState state, int reference_player_id) noexcept:
    """Set active player to the one after reference_player_id in turn order."""
    cdef int ref_position = _get_player_turn_position(state, reference_player_id)
    cdef int next_position = (ref_position + 1) % state._num_players
    cdef int next_player = _find_player_at_position(state, next_position)
    state._set_active_player(next_player)
```

### Advancing Through Bidders (BID-02)

```cython
# Source: Utility for auction bidder rotation

cdef void _advance_to_next_bidder(GameState state) noexcept:
    """Advance active player to next non-passed bidder in turn order."""
    cdef int current_player = state._get_active_player()
    cdef int current_position = _get_player_turn_position(state, current_player)
    cdef int next_position, candidate
    cdef int checked = 0

    while checked < state._num_players:
        next_position = (current_position + 1) % state._num_players
        candidate = _find_player_at_position(state, next_position)

        if not turn_module.TURN.has_player_passed_auction(state, candidate):
            state._set_active_player(candidate)
            return

        current_position = next_position
        checked += 1

    # Should never reach here - means all players passed (auction should resolve first)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct array access | Entity handle methods | Phase 1 | Clean API, normalization handled |
| Python dict for state | Contiguous float32 array | Initial design | Zero-copy to PyTorch |
| Separate phase classes | `cdef noexcept` functions | Phase 2 | Minimal overhead |

**Deprecated/outdated:**
- N/A - This is a new implementation

## Open Questions

### Question 1: Auction Winner When All Others Pass at Start

**What we know:** If player A starts auction at $X and all others immediately leave, A wins
**What's unclear:** Does this count as A "bidding" or just "starting"?
**Recommendation:** Treat starter as initial bidder; if all others pass, starter wins at their opening price. This is standard auction semantics.

### Question 2: Empty Deck During Auction Resolution

**What we know:** BID-09 says draw new company
**What's unclear:** What if deck is empty?
**Recommendation:** Skip the draw (no new company added). The `DECK.draw()` returns -1 for empty deck, already handled in example code.

### Question 3: Active Player State During Phase Transitions

**What we know:** INVEST -> BID_IN_AUCTION requires setting who bids next
**What's unclear:** Should we use `state._set_active_player()` directly or go through an entity?
**Recommendation:** Use `state._set_active_player()` directly since there's no entity method for this. It's a low-level operation already used in `initialize_game()`.

## Sources

### Primary (HIGH confidence)

- `/home/icebreaker/rss-az-cython2/core/actions.pyx` - Action encoding, decode_action()
- `/home/icebreaker/rss-az-cython2/core/state.pyx` - GameState class, auction state methods
- `/home/icebreaker/rss-az-cython2/entities/turn.pyx` - TurnState handle (TURN singleton)
- `/home/icebreaker/rss-az-cython2/entities/player.pyx` - Player handle (PLAYERS list)
- `/home/icebreaker/rss-az-cython2/entities/company.pyx` - Company handle, transfer methods
- `/home/icebreaker/rss-az-cython2/phases/invest.pyx` - Existing stub
- `/home/icebreaker/rss-az-cython2/phases/bid.pyx` - Existing stub

### Secondary (MEDIUM confidence)

- `/home/icebreaker/rss-az-cython2/.planning/REQUIREMENTS.md` - INV-*, BID-* requirements
- `/home/icebreaker/rss-az-cython2/.planning/ROADMAP.md` - Phase dependencies and scope

### Tertiary (LOW confidence)

- N/A - All research based on existing codebase inspection

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All modules examined, APIs verified
- Architecture: HIGH - Patterns extracted from existing Phase 1/2 code
- Pitfalls: HIGH - Derived from actual state layout and normalization

**Research date:** 2026-01-20
**Valid until:** Indefinitely (researching existing codebase, not external dependencies)
