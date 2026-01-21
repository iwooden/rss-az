# Phase 5: Presidency & Bankruptcy - Research

**Researched:** 2026-01-21
**Domain:** Game state management, multi-location state updates, ownership transfers
**Confidence:** HIGH

## Summary

This phase extends the existing share trading handlers in `phases/invest.pyx` to include presidency transfer, receivership detection, and bankruptcy procedures. The codebase already has all required state fields and entity methods; the implementation involves adding check/update logic after share transfers.

The standard approach follows the established codebase patterns:
1. Use existing entity methods for state queries and updates
2. Follow the order-of-operations defined in CONTEXT.md
3. Integrate checks inline in `_handle_sell_share` and add new helper functions
4. No new data structures or files needed

**Primary recommendation:** Extend `_handle_sell_share` with bankruptcy/receivership/presidency checks in the exact order specified, using helper functions for each check to keep the code modular.

## Standard Stack

The implementation uses only existing codebase infrastructure:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Cython 3.x | existing | Performance-critical game logic | Already in use for all phase handlers |
| NumPy | existing | State vector storage | Float32 array for NN compatibility |

### Supporting
No new libraries needed. All functionality comes from existing entity modules:
- `entities/corp.pyx` - Corporation state (receivership, active, shares, companies)
- `entities/player.pyx` - Player state (shares, president status, cash, net worth)
- `entities/company.pyx` - Company removal via `remove_from_game()`
- `entities/market.pyx` - Price space availability
- `core/state.pyx` - Direct state accessors, `bankrupt_corp()` method exists

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Helper functions | Inline all logic | Inline is faster but harder to test/maintain - helpers preferred for clarity |
| Direct state array access | Entity methods | Direct access slightly faster but entity methods ensure correct normalization |

## Architecture Patterns

### Recommended Code Organization

No new files needed. Extend existing `phases/invest.pyx`:

```
phases/
  invest.pyx  # Add: _check_bankruptcy, _check_receivership, _check_presidency helpers
              # Modify: _handle_sell_share to call new helpers
              # Modify: _handle_buy_share to call presidency/receivership helpers
```

### Pattern 1: Order-of-Operations in Sell Handler

**What:** Specific sequence of state checks/updates after selling a share
**When to use:** Every sell action during INVEST phase
**Example:**
```python
# Source: 05-CONTEXT.md order of operations
cdef void _handle_sell_share(GameState state, int corp_id) noexcept:
    # 1. Transfer share from player to bank (existing)
    # 2. Move price down (existing)
    # 3. Pay player the old price (existing)
    # 4. Check bankruptcy - NEW
    if corp.get_price_index(state) == 0:
        _execute_bankruptcy(state, corp_id)
        # Skip receivership/presidency checks - corp is gone
        return
    # 5. Check receivership - NEW
    _check_receivership(state, corp_id)
    # 6. Check presidency - NEW (only if not in receivership)
    if not corp.is_in_receivership(state):
        _check_presidency(state, corp_id)
    # 7. Update net worth (existing)
```

### Pattern 2: Bankruptcy Procedure as Atomic Reset

**What:** Complete corp reset to pre-IPO state
**When to use:** When price drops to index 0
**Example:**
```python
# Source: existing bankrupt_corp() in core/state.pyx
cdef void _execute_bankruptcy(GameState state, int corp_id) noexcept:
    cdef int company_id, player_id
    cdef int current_index

    # Get corp entity
    corp = corp_module.CORPS[CORP_NAMES[corp_id]]

    # 1. Remove all owned companies (just mark removed, no closing procedure)
    for company_id in range(GameConstants.NUM_COMPANIES):
        if corp.owns_company(state, company_id):
            company_module.COMPANIES[company_id].remove_from_game(state)

    # 2. Return shares to unissued
    # Clear all player shares
    for player_id in range(state._num_players):
        player_module.PLAYERS[player_id].set_shares(state, corp_id, 0)
        player_module.PLAYERS[player_id].set_president_of(state, corp_id, False)
    # Reset corp share counts
    corp.set_unissued_shares(state, get_corp_share_count(corp_id))
    corp.set_issued_shares(state, 0)
    corp.set_bank_shares(state, 0)

    # 3. Return money to bank (just clear corp cash)
    corp.set_cash(state, 0)

    # 4. Return price card to row - current space already marked available by sell
    # Corp's old space was freed during price movement

    # 5. Clear corp state
    current_index = corp.get_price_index(state)
    if current_index > 0:
        market_module.MARKET.set_space_available(state, current_index, True)
    corp.set_active(state, False)
    corp.set_price_index(state, 0)  # Also sets share_price to 0
    corp.set_in_receivership(state, False)
    corp.set_income(state, 0)
    corp.set_stars(state, 0)
    corp.set_acquisition_proceeds(state, 0)

    # Clear owned/acquisition company flags
    for company_id in range(GameConstants.NUM_COMPANIES):
        corp.set_owns_company(state, company_id, False)
        corp.set_acquisition_company(state, company_id, False)
```

### Pattern 3: Presidency Check with Tie-Breaking

**What:** Find player with most shares; incumbent keeps on tie
**When to use:** After any share transfer (buy or sell)
**Example:**
```python
cdef void _check_presidency(GameState state, int corp_id) noexcept:
    cdef int player_id, shares, max_shares, president_id, current_president
    cdef int i

    # Find current president
    current_president = -1
    for i in range(state._num_players):
        if player_module.PLAYERS[i].is_president_of(state, corp_id):
            current_president = i
            break

    # Find player with most shares
    max_shares = 0
    president_id = -1
    for player_id in range(state._num_players):
        shares = player_module.PLAYERS[player_id].get_shares(state, corp_id)
        if shares > max_shares:
            max_shares = shares
            president_id = player_id
        elif shares == max_shares and player_id == current_president:
            # Tie - incumbent keeps it
            president_id = current_president

    # Update if changed
    if president_id != current_president and president_id >= 0:
        if current_president >= 0:
            player_module.PLAYERS[current_president].set_president_of(state, corp_id, False)
        player_module.PLAYERS[president_id].set_president_of(state, corp_id, True)
```

### Pattern 4: Receivership Detection

**What:** Check if total player shares = 0
**When to use:** After sell, before presidency check
**Example:**
```python
cdef void _check_receivership(GameState state, int corp_id) noexcept:
    cdef int player_id, total_player_shares

    total_player_shares = 0
    for player_id in range(state._num_players):
        total_player_shares += player_module.PLAYERS[player_id].get_shares(state, corp_id)

    corp = corp_module.CORPS[CORP_NAMES[corp_id]]
    if total_player_shares == 0:
        corp.set_in_receivership(state, True)
        # Clear all president flags (no president in receivership)
        for player_id in range(state._num_players):
            player_module.PLAYERS[player_id].set_president_of(state, corp_id, False)
    else:
        corp.set_in_receivership(state, False)
```

### Anti-Patterns to Avoid

- **Batching state updates:** CONTEXT.md requires immediate updates after each action for model training accuracy
- **Separate bankruptcy phase:** Execute inline during sell, not deferred
- **Checking presidency for receivership corps:** Skip presidency check if corp enters receivership
- **Turn-order tie-breaking:** CONTEXT.md specifies incumbent keeps on tie, NOT turn order

## Don't Hand-Roll

Problems with existing solutions in the codebase:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Resetting corp to initial state | Custom reset logic | Pattern from `initialize_game()` in state.pyx | Already tested initialization pattern |
| Company removal | Custom company tracking | `company.remove_from_game(state)` | Handles location caching correctly |
| Share count queries | Direct array access | `player.get_shares(state, corp_id)` | Handles denormalization |
| President flag updates | Multiple direct writes | `player.set_president_of(state, corp_id, bool)` | Single source of truth |
| Price index updates | Manual one-hot + compact | `corp.set_price_index(state, index)` | Updates both representations |

**Key insight:** The entity methods handle normalization/denormalization correctly. The existing `bankrupt_corp()` method in `core/state.pyx` only sets active=False and price_index=0; a complete bankruptcy procedure needs to also clear shares, companies, cash, and president flags.

## Common Pitfalls

### Pitfall 1: Wrong Tie-Breaking Logic

**What goes wrong:** Using turn order for presidency ties instead of incumbent advantage
**Why it happens:** Turn order is used elsewhere (auction high bidder); easy to mix up
**How to avoid:** CONTEXT.md explicitly states "current president keeps it when shares are equal"
**Warning signs:** Tests fail where multiple players have same share count

### Pitfall 2: Forgetting to Clear President Flags on Bankruptcy

**What goes wrong:** Player still marked as president of bankrupt corp
**Why it happens:** Focus on corp state reset, forget player state
**How to avoid:** Iterate all players and clear `is_president_of(corp_id)` during bankruptcy
**Warning signs:** Player president array inconsistent with corp active status

### Pitfall 3: Not Freeing Market Space on Bankruptcy

**What goes wrong:** Market space stays occupied by bankrupt corp
**Why it happens:** The sell action frees old space, but if corp goes to price 0 it needs cleanup
**How to avoid:** Check if price_index > 0 and free that space during bankruptcy
**Warning signs:** Future IPOs can't use that market space

### Pitfall 4: Presidency Check After Receivership Entry

**What goes wrong:** Trying to find president for corp with 0 player shares
**Why it happens:** Running checks in wrong order
**How to avoid:** Exit early from presidency check if corp is in receivership
**Warning signs:** Spurious presidency changes or crashes

### Pitfall 5: Forgetting Buy Action Also Needs Checks

**What goes wrong:** Buying from receivership doesn't set new president
**Why it happens:** Focus on sell handler for bankruptcy, forget buy needs presidency/receivership checks too
**How to avoid:** Add same checks to `_handle_buy_share`: receivership exit, then presidency
**Warning signs:** Player buys share from receivership corp but no president set

## Code Examples

### Existing Methods to Use

```python
# Source: entities/corp.pyx - Already implemented
corp.is_in_receivership(state)  # Check receivership flag
corp.set_in_receivership(state, bool)  # Set receivership flag
corp.get_price_index(state)  # Get market index (0 = bankruptcy)
corp.set_active(state, bool)  # Deactivate corp
corp.owns_company(state, company_id)  # Check if corp owns company
corp.set_owns_company(state, company_id, bool)  # Clear ownership

# Source: entities/player.pyx - Already implemented
player.get_shares(state, corp_id)  # Get share count
player.set_shares(state, corp_id, 0)  # Clear shares
player.is_president_of(state, corp_id)  # Check president status
player.set_president_of(state, corp_id, bool)  # Set/clear president

# Source: entities/company.pyx - Already implemented
company.remove_from_game(state)  # Mark company as removed

# Source: core/state.pyx - Partial implementation exists
state.bankrupt_corp(corp_id)  # Sets active=False, price_index=0 only
# Need to extend with full bankruptcy procedure
```

### Market Price Array Reference

```python
# Source: core/data.pyx MARKET_PRICES array
# Index 0 = $0 (bankruptcy)
# Index 1 = $5
# ...
# Index 26 = $75 (max, always available)

# When selling, price moves to find_next_lower_space()
# If no lower space available, returns 0 (bankruptcy)
```

### Test Fixture Pattern

```python
# Source: tests/test_share_trading.py
@pytest.fixture
def bankruptcy_state():
    """State with corp at low price index where one sell triggers bankruptcy."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # Set up corp at price index 1 (price $5, one sell away from 0)
    corp = CORPS[CORP_NAMES[0]]
    corp.set_active(state, True)
    corp.set_price_index(state, 1)
    corp.set_bank_shares(state, 2)
    corp.set_issued_shares(state, 3)
    corp.set_cash(state, 50)

    # Give corp a company to verify it gets removed
    company_module.COMPANIES[0].transfer_to_corp(state, 0)
    corp.set_owns_company(state, 0, True)

    # Give player shares
    PLAYERS[0].set_shares(state, 0, 2)
    PLAYERS[0].set_president_of(state, 0, True)
    PLAYERS[0].set_cash(state, 100)

    MARKET.set_space_available(state, 1, False)

    return state
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Deferred bankruptcy | Inline bankruptcy | This phase | State accurate after every action |
| Turn-order tie-break | Incumbent advantage | This phase | Simpler tie-breaking logic |

**Deprecated/outdated:**
- The existing `state.bankrupt_corp()` method is incomplete; it only sets active=False and price_index=0. Phase 5 implements the full procedure.

## Open Questions

All questions resolved by CONTEXT.md:

1. **Resolved: When does presidency transfer happen?**
   - Answer: After each share trade (not batched)

2. **Resolved: Tie-breaking for presidency?**
   - Answer: Current president keeps it when shares equal

3. **Resolved: What happens to companies on bankruptcy?**
   - Answer: Just removed from game, no closing procedure

## Sources

### Primary (HIGH confidence)
- `phases/invest.pyx` - Current sell/buy handlers to extend
- `entities/corp.pyx` - Corporation entity with receivership flag (offset 9)
- `entities/player.pyx` - Player entity with president status per corp
- `entities/company.pyx` - Company removal via `remove_from_game()`
- `entities/market.pyx` - Market space availability
- `core/state.pyx` - State layout, `bankrupt_corp()` partial implementation
- `05-CONTEXT.md` - User decisions on order of operations

### Secondary (MEDIUM confidence)
- `tests/test_share_trading.py` - Test patterns for share trading

### Tertiary (LOW confidence)
- None - all patterns verified in existing codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using only existing codebase infrastructure
- Architecture: HIGH - Extending existing handlers with same patterns
- Pitfalls: HIGH - Derived from CONTEXT.md decisions and codebase analysis

**Research date:** 2026-01-21
**Valid until:** Indefinite - internal implementation details, no external dependencies
