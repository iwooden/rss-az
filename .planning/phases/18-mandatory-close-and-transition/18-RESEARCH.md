# Phase 18: Mandatory Close and Transition - Research

**Researched:** 2026-01-27
**Domain:** Player bankruptcy prevention, phase transition logic
**Confidence:** HIGH

## Summary

Phase 18 implements the final portion of the CLOSING phase: mandatory close and transition. After all player-driven close offers are processed (Phase 17), players with negative total income must have their private companies auto-closed until their income becomes non-negative (preventing bankruptcy in INCOME phase). Then the game transitions to INCOME (currently stubbed as INVEST until INCOME phase is implemented).

The codebase already has all required infrastructure:
- `_close_company()` helper in `closing.pyx` handles company removal and Junkyard Scrappers bonus
- `get_adjusted_company_income()` in `core/data.pyx` calculates income after CoO
- `_transition_to_income()` in `closing.pyx` handles the phase transition (currently goes to INVEST)
- Player entity methods for ownership and cash in `entities/player.pyx`
- Face value sorting pattern from offer generation in Phase 17

**Primary recommendation:** Add mandatory close logic at the end of `_present_next_close_offer()` (when no more offers exist) before calling `_transition_to_income()`. This keeps the logic cleanly isolated and follows the established pattern.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Cython | 3.0+ | Performance-critical phase handler | Existing pattern from all phase handlers |
| core.data | N/A | `get_adjusted_company_income()`, `get_company_face_value()` | Authoritative source for income calculation |
| entities.player | N/A | `owns_company()`, `get_cash()` access | Established player entity pattern |
| phases.closing | N/A | `_close_company()` helper, `_transition_to_income()` | Built in Phases 16-17 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| turn_module | N/A | `get_coo_level()` for income calculation | CoO level needed for adjusted income |
| company_module | N/A | Company iteration, removal status | Finding eligible companies |
| corp_module | N/A | Junkyard Scrappers bonus | Already handled by `_close_company()` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Inline in `_present_next_close_offer()` | Separate handler function | Separate function cleaner, better testability |
| Player entity method | Static function | Entity method matches CONTEXT.md decision (`get_income()`) |
| Re-evaluate CoO during loop | Fixed CoO at phase start | CONTEXT.md specifies fixed CoO |

## Architecture Patterns

### Recommended Project Structure
```
phases/
    closing.pyx           # Extend with mandatory close logic
    closing.pxd           # Add declarations if needed
entities/
    player.pyx            # Add get_income() method
```

**Note:** All changes go to existing files. No new files needed.

### Pattern 1: Player Income Calculation Method
**What:** Method on Player entity to calculate total income from private companies
**When to use:** Mandatory close eligibility check, future INCOME phase
**Example:**
```cython
# Source: entities/player.pyx (new method, based on CONTEXT.md decision)
cpdef int get_income(self, GameState state):
    """
    Calculate total income from player's private companies.

    Income = sum of (base_income - CoO) for each owned private company.
    Note: Only player-owned privates, NOT corp subsidiaries.
    """
    cdef int total = 0
    cdef int company_id, base_income, stars, coo_value
    cdef int coo_level = turn_module.TURN.get_coo_level(state)

    for company_id in range(GameConstants.NUM_COMPANIES):
        if self.owns_company(state, company_id):
            base_income = get_company_income(company_id)
            stars = get_company_stars(company_id)
            coo_value = get_cost_of_ownership(coo_level, stars)
            total += base_income - coo_value

    return total
```

### Pattern 2: Mandatory Close Loop
**What:** Iteratively close cheapest negative-income company until player income + cash >= 0
**When to use:** After offer-based closing exhausted, before transition
**Example:**
```cython
# Source: phases/closing.pyx (new function)
cdef void _process_mandatory_close(GameState state) noexcept:
    """
    Auto-close player private companies to prevent negative cash in INCOME.

    For each player with income + cash < 0:
    1. Find cheapest (lowest face value) negative-income private company
    2. Close it
    3. Recheck income + cash
    4. Repeat until income + cash >= 0

    Per CONTEXT.md: CoO is fixed at phase start, no re-evaluation during loop.
    """
    cdef int player_id, company_id, income, cash
    cdef int cheapest_company, cheapest_fv, fv
    cdef int coo_level = turn_module.TURN.get_coo_level(state)

    # Iterate players by player ID order
    for player_id in range(state._num_players):
        # While player has negative total (income + cash)
        while True:
            income = player_module.PLAYERS[player_id].get_income(state)
            cash = player_module.PLAYERS[player_id].get_cash(state)

            if income + cash >= 0:
                break  # Player is safe

            # Find cheapest negative-income company
            cheapest_company = -1
            cheapest_fv = 999999

            for company_id in range(GameConstants.NUM_COMPANIES):
                if not player_module.PLAYERS[player_id].owns_company(state, company_id):
                    continue

                # Check if negative income
                if get_adjusted_company_income(company_id, coo_level) >= 0:
                    continue

                fv = get_company_face_value(company_id)
                if fv < cheapest_fv:
                    cheapest_fv = fv
                    cheapest_company = company_id

            if cheapest_company < 0:
                # No more negative-income companies (should never happen per CONTEXT.md)
                break

            # Close the company
            _close_player_company(state, cheapest_company, player_id)
```

### Pattern 3: Player Company Close Helper
**What:** Close player-owned company with proper cleanup
**When to use:** Mandatory close needs to close player companies (not FI/corp)
**Example:**
```cython
# Source: phases/closing.pyx (new helper)
cdef void _close_player_company(GameState state, int company_id, int player_id) noexcept:
    """
    Close a player-owned private company.

    Steps:
    1. Clear player ownership
    2. Apply Junkyard Scrappers bonus (2x printed income)
    3. Remove company from game

    Similar to _close_company but for player-owned privates.
    """
    cdef int printed_income = get_company_income(company_id)

    # Clear ownership
    player_module.PLAYERS[player_id].set_owns_company(state, company_id, False)

    # Junkyard Scrappers bonus
    if corp_module.CORPS[0].is_active(state):
        corp_module.CORPS[0].add_cash(state, printed_income * 2)

    # Remove from game
    company_module.COMPANIES[company_id].remove_from_game(state)
```

### Pattern 4: Integration Point
**What:** Call mandatory close before transition when offers exhausted
**When to use:** In `_present_next_close_offer()` when no more valid offers
**Example:**
```cython
# Source: phases/closing.pyx (modify existing _present_next_close_offer)
cdef void _present_next_close_offer(GameState state) noexcept:
    """..."""
    # ... existing offer presentation loop ...

    # No more valid offers - process mandatory close then transition
    turn_module.TURN.clear_closing_company(state)
    _process_mandatory_close(state)  # NEW: mandatory close before transition
    _transition_to_income(state)
```

### Anti-Patterns to Avoid
- **Re-evaluating CoO during mandatory close loop:** Per CONTEXT.md, CoO is fixed at phase start. Don't recalculate.
- **Including corp subsidiaries in mandatory close:** Only player PRIVATE companies are mandatory-closed. Corp companies are managed by corps, not forced closure.
- **Closing positive-income companies:** Only negative-income companies are candidates for mandatory close.
- **Complex tie-breaking:** Face values are unique per CONTEXT.md. No tie-breaking logic needed.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Adjusted income calculation | Manual income - CoO | `get_adjusted_company_income()` | Already exists in core.data |
| Company removal | Manual flag manipulation | `company.remove_from_game()` | Handles location clearing atomically |
| JS bonus logic | Duplicate bonus code | Use pattern from `_close_company()` | Already handles bonus correctly |
| Face value lookup | Calculate from company data | `get_company_face_value()` | Direct lookup in data.pyx |

**Key insight:** The codebase already has all primitives. Phase 18 is orchestration of existing helpers with one new method (`get_income()` on Player).

## Common Pitfalls

### Pitfall 1: Re-evaluating CoO During Loop
**What goes wrong:** Recalculating CoO after each close changes eligibility mid-loop
**Why it happens:** Intuition that "current state" should be used
**How to avoid:** Per CONTEXT.md: "CoO is fixed at phase start; no re-evaluation during mandatory close loop"
**Warning signs:** Different companies closed than expected based on initial state

### Pitfall 2: Including Corp Subsidiaries
**What goes wrong:** Mandatory close tries to close corp-owned companies
**Why it happens:** Confusing player-owned privates with all player-controlled assets
**How to avoid:** Per CONTEXT.md: "Mandatory close applies only to player privates, not corporation subsidiaries"
**Warning signs:** Corps losing companies during mandatory close

### Pitfall 3: Closing Positive/Zero Income Companies
**What goes wrong:** Closing companies that aren't causing the negative income
**Why it happens:** Not checking adjusted income before selecting close candidate
**How to avoid:** Filter to only companies where `get_adjusted_company_income() < 0`
**Warning signs:** Player still has negative income after closing a company

### Pitfall 4: Wrong Close Order
**What goes wrong:** Closing expensive company before cheap company
**Why it happens:** Not sorting by face value ascending
**How to avoid:** Find CHEAPEST (lowest face value) negative-income company each iteration
**Warning signs:** Higher-value companies closed first

### Pitfall 5: Infinite Loop on No Negative Companies
**What goes wrong:** Loop never terminates if player has no negative-income companies left
**Why it happens:** Not handling edge case where no close candidate found
**How to avoid:** Per CONTEXT.md: "Impossible for player to still be negative after closing ALL negative-income privates"
**Warning signs:** Infinite loop in mandatory close

### Pitfall 6: Missing Junkyard Scrappers Bonus
**What goes wrong:** Mandatory closes don't trigger JS bonus
**Why it happens:** Forgetting that ALL closes trigger JS bonus, not just voluntary ones
**How to avoid:** Per CONTEXT.md: "Junkyard Scrappers bonus applies to mandatory closes"
**Warning signs:** JS cash doesn't increase after mandatory close

## Code Examples

Verified patterns from official sources:

### Adjusted Income Check
```cython
# Source: core/data.pyx lines 234-239 (existing function)
cpdef inline int get_adjusted_company_income(int company_id, int coo_level) noexcept nogil:
    """Get company income after cost of ownership."""
    cdef int base_income = COMPANY_INCOME[company_id]
    cdef int stars = COMPANY_STARS[company_id]
    cdef int cost = get_cost_of_ownership(coo_level, stars)
    return base_income - cost
```

### Player Company Ownership Check
```cython
# Source: entities/player.pyx lines 346-348 (existing method)
cpdef bint owns_company(self, GameState state, int company_id):
    """Check if player owns a private company."""
    return state._data[self._owned_companies_offset + company_id] == 1.0
```

### Existing Close Company Pattern (for reference)
```cython
# Source: phases/closing.pyx lines 56-84 (existing helper)
cdef void _close_company(GameState state, int company_id, int owner_type, int owner_id) noexcept:
    """Close a company and handle cleanup."""
    cdef int printed_income = get_company_income(company_id)

    # Clear ownership before removal
    if owner_type == 4:  # LOC_FI
        fi_module.FI.set_owns_company(state, company_id, False)
    elif owner_type == 5:  # LOC_CORP
        corp_module.CORPS[owner_id].set_owns_company(state, company_id, False)

    # Junkyard Scrappers bonus
    if corp_module.CORPS[0].is_active(state):
        corp_module.CORPS[0].add_cash(state, printed_income * 2)

    # Remove company from game
    company_module.COMPANIES[company_id].remove_from_game(state)
```

### Existing Transition Function (for reference)
```cython
# Source: phases/closing.pyx lines 309-333 (existing function)
cdef void _transition_to_income(GameState state) noexcept:
    """Complete CLOSING phase and transition to INCOME."""
    cdef int current_turn = turn_module.TURN.get_turn_number(state)

    # Check for terminal state
    if _is_game_terminal(state):
        turn_module.TURN.set_phase(state, PHASE_GAME_OVER)
        return

    # Increment turn number
    turn_module.TURN.set_turn_number(state, current_turn + 1)

    # Clear per-turn tracking for all players
    for i in range(state._num_players):
        player_module.PLAYERS[i].clear_roundtrip_tracking(state)

    # TEMPORARY: Transition to INVEST (INCOME phase not implemented yet)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct transition after offers | Mandatory close then transition | This phase | Player bankruptcy prevention |
| No player income method | `get_income()` on Player entity | This phase | Preparation for INCOME phase |

**Deprecated/outdated:**
- `_transition_to_income()` transitions to INVEST (temporary, documented in code)
- Will be updated when INCOME phase is implemented (future milestone)

## Open Questions

Things that couldn't be fully resolved:

1. **Player order vs arbitrary order for mandatory close**
   - What we know: CONTEXT.md says "Iterate players by player ID order"
   - What's unclear: Whether game rules specify a particular order
   - Recommendation: Use player ID order (0, 1, 2, ...) as specified in CONTEXT.md

2. **Edge case: player can't pay even after mandatory close**
   - What we know: CONTEXT.md says "Impossible for player to still be negative after closing ALL negative-income privates"
   - What's unclear: What if player has ONLY positive-income companies but huge negative cash?
   - Recommendation: This shouldn't happen in valid game states. If income is from companies, and all negative-income companies are closed, remaining income >= 0. If cash is what makes total negative, that's a different problem (INCOME phase handles this via corp bankruptcy, not player bankruptcy).

## Sources

### Primary (HIGH confidence)
- `phases/closing.pyx` - Existing close helpers, transition function
- `entities/player.pyx` - Player entity pattern, ownership methods
- `core/data.pyx` - `get_adjusted_company_income()`, face value lookup
- `18-CONTEXT.md` - User decisions on income calculation, close logic, transition

### Secondary (MEDIUM confidence)
- `RULES.md` Phase 4: Closing - "If a player will have negative total income in Phase 5 and cannot pay, they must close enough negative-income companies"
- `16-RESEARCH.md`, `17-RESEARCH.md` - Prior phase patterns

### Tertiary (LOW confidence)
- None - all patterns verified in codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All components exist or follow established patterns
- Architecture: HIGH - Simple orchestration of existing helpers
- Pitfalls: HIGH - Based on detailed CONTEXT.md analysis and existing patterns

**Research date:** 2026-01-27
**Valid until:** N/A (project-specific patterns don't expire)
