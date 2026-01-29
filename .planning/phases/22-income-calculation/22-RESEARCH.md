# Phase 22: Income Calculation - Research

**Researched:** 2026-01-28
**Domain:** Cython entity income computation with modifiers and special abilities
**Confidence:** HIGH

## Summary

This phase implements income calculation for all entities (players, corporations, Foreign Investor) including Cost of Ownership deductions, synergy bonuses, and corporation special abilities. The foundation is already established: Phase 21 provides `compute_synergy_bonuses()` in `core/data.pyx`, player income calculation exists in `entities/player.pyx`, and the entity class patterns are well-established.

Key implementation insight: Entity income calculation methods should follow the existing `get_income()` pattern in `entities/player.pyx` (lines 444-463), which already correctly computes base income minus CoO for player-owned companies. Corporation income is fundamentally more complex (synergies, special abilities) so a new `calculate_income()` method in `entities/corp.pyx` is needed. Foreign Investor income follows a simpler pattern (base income - CoO + 5 fixed bonus).

**Primary recommendation:** Add `calculate_income()` methods to Corporation and ForeignInvestor classes following the existing Player.get_income() pattern. Corporation implementation requires calling `compute_synergy_bonuses()` and applying special ability logic based on `corp_id` enum values.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Cython | 3.0+ | Performance compilation | Project uses boundscheck=False, wraparound=False, cdivision=True throughout |
| NumPy C API | 1.7+ | State vector access | All game state stored as float32 contiguous arrays |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | Latest | Unit testing | All tests use pytest with fixtures from conftest.py |

### Cython Compiler Directives (from setup.py)
```python
compiler_directives = {
    'language_level': '3',
    'boundscheck': False,
    'wraparound': False,
    'cdivision': True,
    'initializedcheck': False,
    'nonecheck': False,
    'overflowcheck': False,
}
```

**Installation:**
Already configured in project's setup.py

## Architecture Patterns

### Recommended Function Locations

```
entities/
  corp.pyx         # Add calculate_income() method to Corporation class
  fi.pyx           # Add calculate_income() method to ForeignInvestor class
  player.pyx       # Already has get_income() - reuse pattern

core/
  data.pyx         # compute_synergy_bonuses() already implemented (Phase 21)
  data.pxd         # Update with CorpIndices enum if needed
```

### Pattern 1: Entity Method for Income Calculation

**What:** Instance method on entity class that takes GameState parameter
**When to use:** Income calculation that accesses entity-specific state
**Example:**
```cython
# Source: entities/player.pyx:444-463 (existing pattern)
cpdef int calculate_income(self, GameState state):
    """
    Calculate total income for corporation.

    Formula: printed_income - CoO + synergy + special_abilities

    Returns:
        Total income (can be negative)
    """
    cdef int company_id, base_income, stars, coo_value
    cdef int coo_level = turn_module.TURN.get_coo_level(state)
    cdef int gross_income = 0
    cdef int total_coo = 0
    cdef int synergy_income = 0
    cdef int synergy_markers = 0
    cdef int special_ability_bonus = 0

    # ... implementation
    return gross_income - total_coo + synergy_income + special_ability_bonus
```

### Pattern 2: Company Iteration for Owned Companies

**What:** Loop over company IDs, check ownership via entity method
**When to use:** Summing values across owned companies
**Example:**
```cython
# Source: entities/player.pyx:456-461
for company_id in range(GameConstants.NUM_COMPANIES):
    if self.owns_company(state, company_id):
        base_income = get_company_income(company_id)
        stars = get_company_stars(company_id)
        coo_value = get_cost_of_ownership(coo_level, stars)
        total += base_income - coo_value
```

### Pattern 3: Special Ability Switch via corp_id

**What:** Use enum CorpIndices from `core/data.pxd` to identify special abilities
**When to use:** Applying corporation-specific modifiers
**Example:**
```cython
# Source: core/data.pxd:38-47 (CorpIndices enum)
from core.data cimport CorpIndices

# In calculate_income:
if self.corp_id == CorpIndices.CORP_PR:  # Prussian Railway
    special_ability_bonus += company_count
elif self.corp_id == CorpIndices.CORP_DA:  # Doppler AG
    # Double printed income of highest face value company
    special_ability_bonus += highest_fv_income
elif self.corp_id == CorpIndices.CORP_S:  # Synergistic
    special_ability_bonus += synergy_markers // 2
elif self.corp_id == CorpIndices.CORP_VM:  # Vintage Machinery
    # VM reduces CoO, not adds bonus - handled earlier
    pass
```

### Pattern 4: Using compute_synergy_bonuses from core/data.pyx

**What:** Call existing Phase 21 function with company ID array
**When to use:** Corporation synergy calculation
**Example:**
```cython
# Source: core/data.pyx:270-310 (Phase 21 implementation)
from core.data cimport compute_synergy_bonuses

cdef int company_ids[36]
cdef int count = 0
cdef int synergy_income, synergy_markers

# Collect owned company IDs
for company_id in range(GameConstants.NUM_COMPANIES):
    if self.owns_company(state, company_id):
        company_ids[count] = company_id
        count += 1

# Compute synergies
(synergy_income, synergy_markers) = compute_synergy_bonuses(company_ids, count)
```

### Anti-Patterns to Avoid

- **Computing CoO before iterating companies:** CoO depends on each company's star tier - must be per-company
- **Ignoring VM special ability timing:** VM reduces TOTAL CoO, so need gross income and total CoO first, then apply reduction
- **Double-counting synergy markers:** Use marker_count from compute_synergy_bonuses(), don't count pairs again
- **Applying DA to adjusted income:** DA doubles PRINTED income (before CoO), not adjusted income

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Synergy pair counting | Custom loop over owned companies | `compute_synergy_bonuses()` | Triangular iteration already correct, tested |
| CoO lookup | Manual table indexing | `get_cost_of_ownership(coo_level, stars)` | Handles bounds checking |
| Company base income | Parse from company data | `get_company_income(company_id)` | Already nogil-compatible |
| Face value lookup | Parse from company data | `get_company_face_value(company_id)` | Already nogil-compatible |

**Key insight:** All static company data accessors exist in `core/data.pyx`. Use them; don't duplicate.

## Common Pitfalls

### Pitfall 1: DA Doubles Wrong Value
**What goes wrong:** Doubling income after CoO deduction instead of printed income
**Why it happens:** Misreading "highest Face Value company" vs "highest income company"
**How to avoid:** DA bonus = printed income of highest FV company (second copy). Track highest FV during iteration, add its printed income to bonus.
**Warning signs:** Test with negative-income highest-FV company fails

### Pitfall 2: VM Applied Per-Company Instead of Total
**What goes wrong:** Reducing CoO of each company by up to 10
**Why it happens:** Applying reduction during per-company iteration
**How to avoid:** Calculate total_coo first, then apply `total_coo = max(0, total_coo - 10)` for VM only
**Warning signs:** VM with 3 companies each at CoO=4 should total CoO=2, not 0

### Pitfall 3: Synergistic Uses Wrong Synergy Count
**What goes wrong:** Using synergy_income instead of synergy_markers for S ability
**Why it happens:** Confusion between income (total bonus amount) and markers (pair count)
**How to avoid:** `compute_synergy_bonuses()` returns tuple (income, markers). S uses markers // 2.
**Warning signs:** S ability gives fractional or overly large bonus

### Pitfall 4: Missing FI +5 Bonus
**What goes wrong:** FI income equals player income for same companies
**Why it happens:** Forgetting Foreign Investor's fixed +5 bonus per RULES.md line 354
**How to avoid:** Add +5 at end of FI income calculation
**Warning signs:** FI test cases are $5 short

### Pitfall 5: Player Income Already Implemented
**What goes wrong:** Re-implementing player income calculation
**Why it happens:** Not checking existing codebase
**How to avoid:** Player.get_income() already exists at entities/player.pyx:444-463. Verify it handles CoO correctly and reuse.
**Warning signs:** Duplicate code, inconsistent behavior

### Pitfall 6: Applying Income During Calculation
**What goes wrong:** Adding/subtracting from entity cash during calculate_income()
**Why it happens:** Conflating calculation with application
**How to avoid:** calculate_income() returns int. Separate apply_income() adds to cash.
**Warning signs:** Double-application, race conditions, test isolation issues

## Code Examples

Verified patterns from codebase analysis:

### Example 1: Player Income Pattern (Existing)
```cython
# Source: entities/player.pyx:444-463
cpdef int get_income(self, GameState state):
    """
    Calculate total income from player's private companies.

    Income = sum of (base_income - CoO) for each owned private company.
    Note: Only player-owned privates, NOT corp subsidiaries.
    Used by mandatory close to check if player income + cash < 0.
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

### Example 2: Corporation Income (New Implementation Pattern)
```cython
# Pattern derived from Player.get_income() and CONTEXT.md decisions
cpdef int calculate_income(self, GameState state):
    """
    Calculate total income for corporation with all modifiers.

    Formula: (sum_printed_income - total_coo) + synergy + special_abilities

    Special abilities:
    - PR: +1 per company owned
    - DA: double printed income of highest FV company
    - S: +1 per 2 synergy markers
    - VM: reduce total CoO by up to 10
    """
    cdef int company_id, base_income, stars, coo_value, face_value
    cdef int coo_level = turn_module.TURN.get_coo_level(state)

    # Accumulators
    cdef int gross_printed_income = 0
    cdef int total_coo = 0
    cdef int company_count = 0
    cdef int highest_fv = -1
    cdef int highest_fv_income = 0

    # Company ID collection for synergy calculation
    cdef int company_ids[36]

    # First pass: collect companies, sum printed income, sum CoO, track highest FV
    for company_id in range(GameConstants.NUM_COMPANIES):
        if self.owns_company(state, company_id):
            company_ids[company_count] = company_id
            company_count += 1

            base_income = get_company_income(company_id)
            gross_printed_income += base_income

            stars = get_company_stars(company_id)
            coo_value = get_cost_of_ownership(coo_level, stars)
            total_coo += coo_value

            face_value = get_company_face_value(company_id)
            if face_value > highest_fv:
                highest_fv = face_value
                highest_fv_income = base_income

    # Apply VM special ability (reduces total CoO)
    if self.corp_id == CorpIndices.CORP_VM:
        total_coo = max(0, total_coo - 10)

    # Compute synergy bonuses
    cdef int synergy_income = 0
    cdef int synergy_markers = 0
    if company_count > 1:
        (synergy_income, synergy_markers) = compute_synergy_bonuses(company_ids, company_count)

    # Compute special ability bonuses
    cdef int special_bonus = 0
    if self.corp_id == CorpIndices.CORP_PR:  # Prussian Railway: +1 per company
        special_bonus = company_count
    elif self.corp_id == CorpIndices.CORP_DA:  # Doppler AG: double highest FV income
        special_bonus = highest_fv_income  # Adds printed income again (doubling)
    elif self.corp_id == CorpIndices.CORP_S:  # Synergistic: +1 per 2 markers
        special_bonus = synergy_markers // 2
    # VM handled above (CoO reduction, not bonus addition)

    # Final formula: printed - CoO + synergy + special
    return gross_printed_income - total_coo + synergy_income + special_bonus
```

### Example 3: FI Income Pattern
```cython
# Pattern for ForeignInvestor.calculate_income()
cpdef int calculate_income(self, GameState state):
    """
    Calculate total income for Foreign Investor.

    Formula: sum(printed_income - CoO) + 5
    FI always receives +5 base income bonus.
    """
    cdef int company_id, base_income, stars, coo_value
    cdef int coo_level = turn_module.TURN.get_coo_level(state)
    cdef int total = 0

    for company_id in range(GameConstants.NUM_COMPANIES):
        if self.owns_company(state, company_id):
            base_income = get_company_income(company_id)
            stars = get_company_stars(company_id)
            coo_value = get_cost_of_ownership(coo_level, stars)
            total += base_income - coo_value

    # FI always gets +5 bonus (RULES.md line 354)
    total += 5

    return total
```

### Example 4: Income Application (Phase 22-03)
```cython
# Pattern for applying calculated income to entity cash
def apply_income(self, GameState state, int income):
    """Apply calculated income to entity cash."""
    self.add_cash(state, income)  # Works for positive and negative
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Multiple passes over companies | Single pass with accumulators | Established pattern | Simpler, faster |
| Synergy recalculation | compute_synergy_bonuses() in data.pyx | Phase 21 | Reusable, tested |
| String-based corp identification | CorpIndices enum | Established | nogil-compatible |

**Deprecated/outdated:**
- None identified - this is new functionality building on established patterns

## Open Questions

1. **Should Player.get_income() be renamed to calculate_income() for consistency?**
   - What we know: Player already has get_income() at line 444-463
   - What's unclear: Whether renaming adds value vs consistency with Corp/FI
   - Recommendation: Leave as is, get_income() works. Corp/FI use calculate_income() to distinguish from stored income field.

2. **Should income be stored in entity state or calculated on demand?**
   - What we know: Corporation has `income` field (lines 221-227 in corp.pyx)
   - What's unclear: Whether to update stored income during calculation
   - Recommendation: Calculate on demand, update stored value only when applying. The stored `income` field is used for display/debugging, not as source of truth.

3. **Integration with INCOME phase handler (Phase 23)?**
   - What we know: Phase 23 will call these calculate_income() methods
   - What's unclear: Order of entity processing, bankruptcy detection
   - Recommendation: This phase focuses on calculation. Phase 23 handles orchestration.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: entities/player.pyx lines 444-463 (Player.get_income pattern)
- Codebase analysis: core/data.pyx lines 270-310 (compute_synergy_bonuses)
- Codebase analysis: core/data.pxd lines 38-47 (CorpIndices enum)
- Codebase analysis: entities/corp.pyx (Corporation class structure)
- RULES.md lines 350-362 (Collect Income procedure)
- RULES.md lines 415-422 (Corporation Special Abilities table)
- CONTEXT.md (locked decisions from /gsd:discuss-phase)

### Secondary (MEDIUM confidence)
- Codebase analysis: phases/closing.pyx (phase handler pattern)
- Codebase analysis: tests/phases/conftest.py (test fixture patterns)
- Codebase analysis: tests/phases/test_income.py (existing synergy tests)

### Tertiary (LOW confidence)
- None - all findings verified against codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - using existing project patterns
- Architecture: HIGH - extending established entity patterns
- Pitfalls: HIGH - derived from CONTEXT.md decisions and game rules
- Code examples: HIGH - adapted from existing codebase patterns

**Research date:** 2026-01-28
**Valid until:** ~90 days (stable domain - internal project patterns)
