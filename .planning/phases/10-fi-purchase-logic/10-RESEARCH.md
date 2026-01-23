# Phase 10: Foreign Investor Purchase Logic - Research

**Researched:** 2026-01-23
**Domain:** Deterministic purchase loop with re-checking pattern in Cython
**Confidence:** HIGH

## Summary

Phase 10 implements the Foreign Investor purchase logic within the existing WRAP_UP phase handler. The FI purchases companies in ascending face value order at face value, drawing new cards after each purchase until it cannot afford any remaining available company. This is a deterministic, non-player operation that integrates into the existing WRAP_UP flow after player reordering.

**Key findings:**
- All required entity interfaces already exist (transfer_to_fi, move_to_auction, deck.draw)
- Company iteration by company_id naturally yields ascending face value order (companies indexed 0-35 are pre-sorted)
- Re-check pattern is simple: iterate all 36 companies to find cheapest affordable available company, purchase, repeat
- Availability transition uses simple for-loop over all companies checking LOC_REVEALED flag

**Primary recommendation:** Implement FI purchases as while-loop that re-checks all companies on each iteration to find cheapest affordable available company. No snapshotting needed - direct state queries ensure correctness.

## Standard Stack

All required infrastructure already exists in the codebase. No new libraries or dependencies needed.

### Core Entities (Already Implemented)

| Entity | Methods Used | Purpose | Location |
|--------|--------------|---------|----------|
| Company | transfer_to_fi(), move_to_auction(), is_for_auction(), is_revealed(), get_face_value() | FI purchases and availability transition | entities/company.pyx |
| ForeignInvestor | get_cash(), add_cash() | Track FI cash and deduct purchases | entities/fi.pyx |
| Deck | draw(), is_empty() | Draw new cards after purchase | entities/deck.pyx |

### Supporting Infrastructure

| Component | Purpose | Location |
|-----------|---------|----------|
| GameConstants.NUM_COMPANIES | Company count constant (36) | core/data.pyx |
| COMPANY_FACE_VALUE[36] | Face values by company_id (pre-sorted ascending) | core/data.pyx |
| CompanyLocation.LOC_REVEALED | Flag for revealed-but-unavailable companies | entities/company.pxd |
| CompanyLocation.LOC_FI | Flag for FI-owned companies | entities/company.pxd |

**Installation:** N/A - all code already exists

## Architecture Patterns

### Pattern 1: Company Iteration Order is Face Value Order

**What:** Companies are indexed 0-35 in ascending face value order. Iterating `for company_id in range(NUM_COMPANIES)` naturally processes companies from cheapest to most expensive.

**Why it works:**
```python
# From core/data.pyx lines 44-51
COMPANY_FACE_VALUE = [
    1, 2, 5, 6, 7, 8,           # Reds (company_id 0-5)
    11, 12, 13, 14, 15, 16, 17, 19,  # Oranges (6-13)
    20, 21, 22, 23, 24, 25, 26, 29,  # Yellows (14-21)
    30, 31, 32, 33, 34, 36, 43,      # Greens (22-28)
    45, 46, 47, 50, 56, 58, 60,      # Blues (29-35)
]
```

**When to use:** Whenever ascending face value order is needed - just iterate company_id.

**Example:**
```cython
# Find cheapest available company FI can afford
for company_id in range(GameConstants.NUM_COMPANIES):
    if company_module.COMPANIES[company_id].is_for_auction(state):
        face_value = get_company_face_value(company_id)
        if face_value <= fi_cash:
            # This is the cheapest affordable company
            return company_id
return -1  # None found
```

### Pattern 2: Re-Check Loop Without Snapshotting

**What:** Deterministic purchase loop that re-queries state on each iteration instead of snapshotting available companies.

**When to use:** When each iteration modifies state (FI cash, company locations, deck) and subsequent iterations depend on fresh state.

**Why this pattern:**
- Prevents stale data bugs (trying to buy company that's no longer available)
- Matches auction resolution pattern (bid.pyx lines 47-50: draw card, move to auction)
- Simpler than snapshot+validation approach

**Example:**
```cython
# Source: Derived from bid.pyx _resolve_auction pattern
while True:
    cheapest_id = _find_cheapest_affordable_available(state)
    if cheapest_id < 0:
        break  # No more affordable companies

    # Purchase logic here
    # - FI pays face value
    # - Company transfers to FI
    # - Draw new card and mark unavailable
    # Loop continues with fresh state
```

### Pattern 3: Availability Transition with Simple For-Loop

**What:** After FI purchases complete, convert all LOC_REVEALED companies to LOC_AUCTION (available).

**Why simple loop works:**
- All companies are already initialized at game start
- is_revealed() checks single flag in O(1)
- move_to_auction() handles all state transitions atomically
- No special ordering needed (company_id order is fine)

**Example:**
```cython
# Source: Adapted from company.pyx transfer patterns
cdef void _make_all_revealed_available(GameState state) noexcept:
    """Convert all revealed companies to available for auction."""
    cdef int company_id
    for company_id in range(GameConstants.NUM_COMPANIES):
        if company_module.COMPANIES[company_id].is_revealed(state):
            company_module.COMPANIES[company_id].move_to_auction(state)
```

### Pattern 4: Sentinel History Recording

**What:** Non-player phases record single history entry with sentinel action value before phase execution.

**Implementation:**
```cython
# Source: core/driver.pyx lines 46-59
if phase == PHASE_WRAP_UP:
    sentinel = ACTION_WRAP_UP_SENTINEL  # -100

# Record state BEFORE execution
if history is not None:
    history.append((state._array.copy(), sentinel))

# Execute all WRAP_UP logic (player reorder + FI purchases + availability)
apply_wrap_up(state)
```

**Key points:**
- State snapshot taken BEFORE any WRAP_UP operations
- All FI purchases are batched into single history entry
- Individual purchases do NOT get separate history entries

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Finding available companies | Custom snapshot/filter logic | Direct is_for_auction() queries in iteration | Entity interface handles all location flags correctly |
| Drawing and marking unavailable | Custom deck/location logic | deck.draw() + company.set_revealed(True) | Matches auction resolution pattern (bid.pyx lines 48-50) |
| Availability transition | Manual flag manipulation | company.move_to_auction() | Handles cache invalidation and atomic state updates |
| FI cash tracking | Direct state array access | fi.get_cash(), fi.add_cash() | Entity interface with proper cash normalization (CASH_DIVISOR) |

**Key insight:** The entity interface layer exists precisely to prevent direct state manipulation bugs. Using company.transfer_to_fi() instead of manual flag-setting prevents location cache desync issues.

## Common Pitfalls

### Pitfall 1: Snapshotting Available Companies

**What goes wrong:** Capturing list of available companies before loop, then iterating snapshot.

```cython
# WRONG - snapshot can become stale
cdef list available = []
for company_id in range(NUM_COMPANIES):
    if COMPANIES[company_id].is_for_auction(state):
        available.append(company_id)

for company_id in available:  # BUG: may try to buy unavailable company
    # ... purchase logic
```

**Why it happens:** Intuition from functional programming or avoiding repeated queries.

**How to avoid:** Re-check availability on each iteration. Performance cost is negligible (36 companies, O(1) flag checks).

**Warning signs:** Comments like "cache available companies" or variables named `snapshot`, `initial_list`.

### Pitfall 2: Forgetting to Draw Card After Purchase

**What goes wrong:** FI purchases company, but no new card is drawn. Auction row doesn't replenish.

**Why it happens:** Purchase logic focuses on transfer and cash, forgets the "draw replacement card" step from rules.

**How to avoid:** Follow auction resolution pattern exactly (bid.pyx lines 47-50):
```cython
# Winner receives company
company_module.COMPANIES[company_id].transfer_to_player(state, winner_id)

# Draw new company to auction row
new_company = deck_module.DECK.draw(state)
if new_company >= 0:
    company_module.COMPANIES[new_company].move_to_auction(state)
```

**Warning signs:** Tests failing with "expected N auction companies, got N-1" after FI purchases.

### Pitfall 3: Drawing Card When Deck is Empty

**What goes wrong:** Calling deck.draw() when empty, not checking return value (-1), trying to call move_to_auction(-1).

**Why it happens:** Forgetting edge case where FI purchases exhaust the deck.

**How to avoid:** Always check deck.draw() return value:
```cython
new_company = deck_module.DECK.draw(state)
if new_company >= 0:  # Only if deck wasn't empty
    company_module.COMPANIES[new_company].move_to_auction(state)
```

**Warning signs:** Segfault or assertion failure when deck runs out during FI purchases.

### Pitfall 4: Using move_to_auction() for Newly Drawn Cards

**What goes wrong:** Drawing a card then calling move_to_auction() on it - makes it immediately available for FI to purchase again in same WRAP_UP.

**Why it happens:** Misunderstanding the unavailable state. Rules say newly drawn companies are "revealed" (unavailable) until end of WRAP_UP.

**How to avoid:** Use set_revealed(True) for newly drawn cards during FI purchase loop:
```cython
new_company = deck_module.DECK.draw(state)
if new_company >= 0:
    # Mark as revealed (unavailable) - will become available in transition
    company_module.COMPANIES[new_company].set_revealed(state, True)
```

**Warning signs:** Infinite loop where FI keeps buying the same newly-drawn cheap card.

### Pitfall 5: Not Handling FI with 0 Cash

**What goes wrong:** Special-casing FI == 0 cash with early return, skipping availability transition.

**Why it happens:** Optimization instinct - "if FI has no money, skip the whole thing."

**How to avoid:** Purchase loop naturally terminates when no affordable companies exist. Availability transition ALWAYS runs regardless:
```cython
# Purchase loop (naturally terminates if FI cash == 0)
while True:
    cheapest_id = _find_cheapest_affordable_available(state)
    if cheapest_id < 0:
        break
    # ... purchase

# Availability transition ALWAYS runs
_make_all_revealed_available(state)
```

**Warning signs:** Tests failing with "companies still marked revealed" when FI starts with 0 cash.

## Code Examples

Verified patterns from existing codebase and user context:

### Finding Cheapest Affordable Available Company

```cython
# Source: Derived from get_auction_company_for_slot pattern (company.pyx lines 27-48)
cdef int _find_cheapest_affordable_available(GameState state) noexcept nogil:
    """
    Find cheapest available company FI can afford, or -1 if none.

    Iterates company_id 0-35 which is ascending face value order.
    Returns first company that is available and affordable.
    """
    cdef int fi_cash = fi_module.FI.get_cash(state)
    cdef int company_id, face_value

    for company_id in range(GameConstants.NUM_COMPANIES):
        if company_module.COMPANIES[company_id].is_for_auction(state):
            face_value = get_company_face_value(company_id)
            if face_value <= fi_cash:
                return company_id  # Found cheapest affordable

    return -1  # No affordable companies
```

### FI Purchase Operation

```cython
# Source: Adapted from bid.pyx auction resolution (lines 38-50)
cdef void _fi_purchase_company(GameState state, int company_id) noexcept:
    """
    FI purchases one company: pay face value, transfer, draw replacement.
    """
    cdef int face_value = get_company_face_value(company_id)
    cdef int new_company

    # FI pays face value
    fi_module.FI.add_cash(state, -face_value)

    # Transfer company to FI
    company_module.COMPANIES[company_id].transfer_to_fi(state)

    # Draw replacement card and mark as revealed (unavailable)
    new_company = deck_module.DECK.draw(state)
    if new_company >= 0:
        company_module.COMPANIES[new_company].set_revealed(state, True)
```

### Complete FI Purchase Loop

```cython
# Source: Pattern derived from user context and existing codebase patterns
cdef void _process_fi_purchases(GameState state) noexcept:
    """
    FI buys companies in ascending face value order until cannot afford more.

    Re-checks available companies after each purchase (no snapshotting).
    """
    cdef int company_id

    while True:
        # Find cheapest affordable available company (fresh query each time)
        company_id = _find_cheapest_affordable_available(state)
        if company_id < 0:
            break  # No more affordable companies

        # Purchase this company
        _fi_purchase_company(state, company_id)

        # Loop continues - will re-check all companies for new cheapest
```

### Availability Transition

```cython
# Source: Adapted from company.pyx transfer patterns
cdef void _make_all_revealed_available(GameState state) noexcept:
    """
    After FI purchases, all revealed companies become available for auction.

    This always runs, even if FI made zero purchases.
    """
    cdef int company_id

    for company_id in range(GameConstants.NUM_COMPANIES):
        if company_module.COMPANIES[company_id].is_revealed(state):
            company_module.COMPANIES[company_id].move_to_auction(state)
```

### Updated apply_wrap_up() Integration

```cython
# Source: phases/wrap_up.pyx (lines 72-88) + Phase 10 additions
cdef int apply_wrap_up(GameState state) noexcept:
    """
    Execute WRAP_UP phase logic.

    Steps:
    1. Reorder players by descending cash
    2. Clear consecutive passes for next INVEST round
    3. FI purchases companies in ascending face value order
    4. All revealed companies become available
    5. Transition to ACQUISITION

    Returns: 0 always (deterministic, no failure modes)
    """
    _reorder_players_by_cash(state)
    turn_module.TURN.clear_consecutive_passes(state)

    # Phase 10: FI purchase logic
    _process_fi_purchases(state)
    _make_all_revealed_available(state)

    turn_module.TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
    return 0
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| N/A - new feature | Re-check loop with fresh queries | Phase 10 (2026-01-23) | First deterministic purchase loop in codebase |
| Manual state flags | Entity transfer methods | Phase 1 foundation | Prevents cache desync bugs |
| Python-style iteration | C-level for loops with range | Cython best practice | Required for noexcept nogil functions |

**No deprecated patterns:** This phase implements new functionality using established patterns.

## Open Questions

### Question 1: Short-Circuit When FI Cash is 0

**What we know:** Loop naturally terminates when FI has 0 cash (no affordable companies on first iteration).

**What's unclear:** Whether to add explicit early-return check `if fi_cash == 0: return` for micro-optimization.

**Recommendation:** Skip the optimization. Loop body never executes if cash is 0 (first iteration finds nothing), and explicit check adds branch complexity for zero performance gain. Availability transition must run regardless.

### Question 2: nogil Qualification for Helper Functions

**What we know:** apply_wrap_up() is `noexcept` but not `nogil`. Helper functions could be `noexcept nogil` for future flexibility.

**What's unclear:** Whether GIL is required for entity module access (company_module.COMPANIES, fi_module.FI).

**Recommendation:** Start with `noexcept` only (match apply_wrap_up signature). If performance profiling later shows GIL contention, investigate nogil compatibility with entity interfaces.

## Sources

### Primary (HIGH confidence)

- entities/company.pyx - transfer_to_fi(), move_to_auction(), set_revealed() implementations
- entities/fi.pyx - get_cash(), add_cash() implementations
- entities/deck.pyx - draw() implementation
- core/data.pyx - COMPANY_FACE_VALUE array (lines 44-51), NUM_COMPANIES constant
- phases/bid.pyx - auction resolution pattern (lines 30-64) for drawing replacement cards
- phases/wrap_up.pyx - existing WRAP_UP structure (lines 72-88)
- core/driver.pyx - sentinel history recording (lines 32-62)
- entities/company.pxd - CompanyLocation enum (lines 20-30)

### Secondary (MEDIUM confidence)

- User context from .planning/phases/10-fi-purchase-logic/10-CONTEXT.md - implementation decisions
- RULES.md lines 3-4 of WRAP_UP section - "ascending Face Value order" rule confirmation

### Tertiary (LOW confidence)

None - all findings verified with codebase inspection.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All entity interfaces exist and are tested in phases 1-9
- Architecture: HIGH - Patterns directly verified in existing code (bid.pyx, company.pyx)
- Pitfalls: HIGH - Derived from common Cython mistakes and entity interface contracts

**Research date:** 2026-01-23
**Valid until:** 2026-02-22 (30 days - stable domain, no external dependencies)
