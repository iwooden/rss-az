# Phase 16: Auto-Close Logic - Research

**Researched:** 2026-01-26
**Domain:** Cython non-player phase implementation, company closing logic
**Confidence:** HIGH

## Summary

Phase 16 implements the auto-close portion of the CLOSING phase. This is a deterministic non-player phase where FI and receivership corporations automatically close unprofitable companies before player-driven closing offers begin (Phase 17).

The codebase already has all required infrastructure:
- `Company.remove_from_game()` for closing companies (sets `companies_removed` flag)
- `turn_module.TURN.get_coo_level()` for current Cost of Ownership level
- `get_cost_of_ownership(coo_level, star_tier)` for CoO values by company color
- `get_company_income(company_id)` for base income
- Non-player phase pattern from `wrap_up.pyx` (0 valid actions, deterministic execution)
- `_transition_to_closing()` stub in `acquisition.pyx` ready to be updated

**Primary recommendation:** Implement as a deterministic non-player phase handler that processes FI and receivership corps in a single pass before transitioning to Phase 17 (offer-based closing).

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Cython | 3.0+ | Performance-critical phase handler | Existing pattern from all phase handlers |
| core.data | N/A | Static game data (CoO table, company data) | Authoritative source for game constants |
| entities | N/A | Entity handles for state access | Established pattern in codebase |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| turn_module | N/A | Phase transitions, CoO level | All phase logic |
| company_module | N/A | Company removal | Close operations |
| corp_module | N/A | Corp ownership, receivership | Identifying closing targets |
| fi_module | N/A | FI company ownership | FI close eligibility |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Single pass (FI + receivership) | Separate passes | Single pass is simpler, order doesn't matter |
| Inline close logic | Helper function | Helper improves readability, reuse for Phase 17 |

## Architecture Patterns

### Recommended Project Structure
```
phases/
    closing.pyx           # New file for CLOSING phase (16 + 17 combined)
    closing.pxd           # Header for cdef function declarations
```

**Note:** Could also add to existing `acquisition.pyx` since CLOSING follows ACQUISITION, but a new file provides cleaner separation.

### Pattern 1: Non-Player Phase Handler
**What:** Phase with 0 valid actions that executes deterministically
**When to use:** Auto-close has no player decisions - all closes are rule-mandated
**Example:**
```cython
# Source: phases/wrap_up.pyx lines 158-186
cdef int apply_closing_auto(GameState state) noexcept:
    """
    Execute auto-close logic for FI and receivership corps.

    This is a deterministic non-player phase with 0 actions.
    Returns: 0 always (deterministic, no failure modes)
    """
    _process_fi_auto_close(state)
    _process_receivership_auto_close(state)

    # Transition to offer-based closing (Phase 17)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_CLOSING)
    return 0
```

### Pattern 2: Company Close Helper
**What:** Reusable function to close a company and apply bonuses
**When to use:** Both auto-close (Phase 16) and offer-close (Phase 17) need same logic
**Example:**
```cython
# Source: Based on invest.pyx lines 110-167 (bankruptcy pattern)
cdef void _close_company(GameState state, int company_id, int owner_type, int owner_id) noexcept:
    """
    Close a company and handle all side effects.

    Args:
        state: Game state
        company_id: Company to close
        owner_type: LOC_FI, LOC_CORP, or LOC_PLAYER
        owner_id: Corp/player ID if applicable
    """
    cdef int base_income

    # Clear ownership from previous owner
    if owner_type == LOC_FI:
        fi_module.FI.set_owns_company(state, company_id, False)
    elif owner_type == LOC_CORP:
        corp_module.CORPS[owner_id].set_owns_company(state, company_id, False)

    # Junkyard Scrappers bonus (JS is corp_id 0)
    if corp_module.CORPS[0].is_active(state):
        base_income = get_company_income(company_id)
        corp_module.CORPS[0].add_cash(state, base_income * 2)

    # Remove company from game
    company_module.COMPANIES[company_id].remove_from_game(state)
```

### Pattern 3: While-Loop Re-Query Pattern
**What:** Re-evaluate eligibility after each close (dynamic state)
**When to use:** When closes can affect other close eligibility (they don't here, but pattern is useful)
**Example:**
```cython
# Source: phases/wrap_up.pyx lines 66-84
# NOTE: For Phase 16, a simple for-loop is sufficient since:
# - FI closes don't affect receivership closes
# - Each receivership corp is processed independently
# - Highest face value protection is per-corp, not affected by other closes
```

### Anti-Patterns to Avoid
- **Snapshotting company list:** Don't pre-calculate close candidates. Iterate companies directly.
- **Multiple owner queries:** Use `company.get_location()` and `company.get_owner_id()` once, not per-company queries.
- **Modifying company ownership without clearing old location:** Always use `remove_from_game()` which calls `clear_location()`.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Getting CoO for company | Calculate from level + stars | `get_cost_of_ownership(coo_level, star_tier)` | Static table in data.pyx |
| Removing company from game | Manual flag setting | `company.remove_from_game(state)` | Handles location clearing atomically |
| Checking if corp is in receivership | Check bank_shares vs issued_shares | `corp.is_in_receivership(state)` | Single flag check |
| Finding company star tier | Map face value to tier | `company.get_stars()` or `get_company_stars(id)` | Direct lookup |

**Key insight:** The codebase already has all primitives. Phase 16 is pure orchestration of existing methods.

## Common Pitfalls

### Pitfall 1: Closing Protected Company
**What goes wrong:** Closing the highest face value company from receivership corp
**Why it happens:** Forgetting to check protection rule before close
**How to avoid:** Track highest face value per corp BEFORE processing closes
**Warning signs:** Test failure when receivership corp has only one company

### Pitfall 2: Missing Junkyard Scrappers Bonus
**What goes wrong:** Auto-closed companies don't trigger JS bonus
**Why it happens:** Bonus logic only in player-close path
**How to avoid:** Use shared helper function for all closes
**Warning signs:** JS cash doesn't increase after auto-close

### Pitfall 3: FI Zero Income vs Negative Income
**What goes wrong:** FI closes company with income = CoO (zero adjusted income)
**Why it happens:** Misreading rule as "CoO >= income" instead of "CoO > income"
**How to avoid:** Per CONTEXT.md: FI closes only when `adjusted_income < 0`
**Warning signs:** FI companies with zero net income getting closed

### Pitfall 4: Receivership Using Wrong CoO Value
**What goes wrong:** Using CoO level (1-7) instead of actual dollar value
**Why it happens:** Confusing `coo_level` with `get_cost_of_ownership()` return value
**How to avoid:** Always call `get_cost_of_ownership(coo_level, star_tier)` for threshold comparison
**Warning signs:** Red companies closing at CoO level 4 instead of CoO value $4

### Pitfall 5: Forgetting Vintage Machinery CoO Reduction
**What goes wrong:** VM receivership corp closes companies using full CoO
**Why it happens:** VM special ability reduces CoO by up to $10
**How to avoid:** Check if owner is VM (corp_id 6), apply reduction before threshold check
**Warning signs:** VM receivership closing companies that should survive

## Code Examples

Verified patterns from official sources:

### CoO Lookup
```cython
# Source: core/data.pyx lines 228-232
cpdef inline int get_cost_of_ownership(int coo_level, int star_tier) noexcept nogil:
    """Get cost of ownership for a company with given stars at given CoO level."""
    if coo_level < 1 or coo_level > 7 or star_tier < 1 or star_tier > 5:
        return 0
    return COST_OF_OWNERSHIP[coo_level - 1][star_tier - 1]
```

### Company Removal
```cython
# Source: entities/company.pyx lines 320-328
cpdef void remove_from_game(self, GameState state):
    """Remove company from the game (closed)."""
    # Clear old location
    self.clear_location(state)

    # Set removed flag
    state._data[self._removed_offset] = 1.0
    self._location = LOC_REMOVED
    self._owner_id = -1
```

### Corp Ownership Check
```cython
# Source: entities/corp.pyx lines 192-198
cpdef bint owns_company(self, GameState state, int company_id):
    """Check if corporation owns a company."""
    return state._data[self._owned_companies_offset + company_id] == 1.0
```

### Receivership Check
```cython
# Source: entities/corp.pyx lines 180-182
cpdef bint is_in_receivership(self, GameState state):
    """Check if corporation is in receivership (no president)."""
    return state._data[self._in_receivership_offset] == 1.0
```

### FI Close Eligibility Pattern
```cython
# New code pattern based on existing patterns
cdef void _process_fi_auto_close(GameState state) noexcept:
    """Close all FI companies with negative adjusted income."""
    cdef int company_id, base_income, coo, stars, coo_level

    coo_level = turn_module.TURN.get_coo_level(state)

    for company_id in range(GameConstants.NUM_COMPANIES):
        if fi_module.FI.owns_company(state, company_id):
            base_income = get_company_income(company_id)
            stars = get_company_stars(company_id)
            coo = get_cost_of_ownership(coo_level, stars)

            # FI closes when adjusted income is NEGATIVE (not zero)
            if base_income - coo < 0:
                _close_company(state, company_id, LOC_FI, -1)
```

### Receivership Close Eligibility Pattern
```cython
# New code pattern based on existing patterns
cdef void _process_receivership_auto_close(GameState state) noexcept:
    """Close eligible companies from receivership corps."""
    cdef int corp_id, company_id, stars, coo, coo_level
    cdef int highest_fv_company, highest_fv
    cdef int vm_reduction, total_coo
    cdef bint is_vm

    coo_level = turn_module.TURN.get_coo_level(state)

    for corp_id in range(GameConstants.NUM_CORPS):
        if not corp_module.CORPS[corp_id].is_active(state):
            continue
        if not corp_module.CORPS[corp_id].is_in_receivership(state):
            continue

        is_vm = (corp_id == 6)  # Vintage Machinery

        # Find highest face value company (protected)
        highest_fv = -1
        highest_fv_company = -1
        for company_id in range(GameConstants.NUM_COMPANIES):
            if corp_module.CORPS[corp_id].owns_company(state, company_id):
                fv = get_company_face_value(company_id)
                if fv > highest_fv:
                    highest_fv = fv
                    highest_fv_company = company_id

        # Close eligible companies (excluding protected)
        for company_id in range(GameConstants.NUM_COMPANIES):
            if company_id == highest_fv_company:
                continue  # Protected
            if not corp_module.CORPS[corp_id].owns_company(state, company_id):
                continue

            stars = get_company_stars(company_id)
            coo = get_cost_of_ownership(coo_level, stars)

            # VM reduces CoO by up to 10
            if is_vm:
                coo = max(0, coo - 10)

            # Red (stars=1): close when CoO >= $4
            # Orange (stars=2): close when CoO >= $7
            # Yellow/Green/Blue: never auto-close
            if stars == 1 and coo >= 4:
                _close_company(state, company_id, LOC_CORP, corp_id)
            elif stars == 2 and coo >= 7:
                _close_company(state, company_id, LOC_CORP, corp_id)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Stub transition to INVEST | Proper CLOSING phase handler | This phase | Correct game flow |

**Deprecated/outdated:**
- `_transition_to_closing()` in acquisition.pyx currently goes to INVEST as workaround

## Open Questions

Things that couldn't be fully resolved:

1. **Phase 17 integration timing**
   - What we know: Auto-close happens first, then offer-based closing
   - What's unclear: Whether they share one `PHASE_CLOSING` or have sub-states
   - Recommendation: Single phase with auto-close at entry before offers

2. **Driver integration for hybrid phase**
   - What we know: CLOSING is hybrid (non-player auto-close, then player offers)
   - What's unclear: How driver handles transition from 0-action to n-action within same phase
   - Recommendation: Model after ACQUISITION pattern where `acq_active_corp == -1` means non-player

## Sources

### Primary (HIGH confidence)
- `phases/wrap_up.pyx` - Non-player phase pattern
- `phases/invest.pyx` - Company close pattern from bankruptcy
- `core/data.pyx` - CoO table and accessor functions
- `entities/company.pyx` - `remove_from_game()` implementation
- `entities/corp.pyx` - Receivership check, ownership methods
- `16-CONTEXT.md` - User decisions on close rules

### Secondary (MEDIUM confidence)
- `RULES.md` Section "Phase 4: Closing" and "Receivership Automatic Actions"
- `VECTORS.md` - State vector layout for `companies_removed`

### Tertiary (LOW confidence)
- None - all patterns verified in codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All components exist in codebase
- Architecture: HIGH - Follows established phase handler patterns
- Pitfalls: HIGH - Based on detailed rule analysis and codebase patterns

**Research date:** 2026-01-26
**Valid until:** N/A (project-specific patterns don't expire)
