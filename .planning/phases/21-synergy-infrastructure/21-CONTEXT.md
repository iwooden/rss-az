# Phase 21: Synergy Infrastructure - Context

**Gathered:** 2026-01-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Data structures and functions for identifying synergy pairs between companies owned by the same corporation, returning both the total synergy income and the count of synergy markers.

</domain>

<decisions>
## Implementation Decisions

### Data Foundation
- Synergy matrix already exists: `COMPANY_SYNERGY[36][36]` in `core/data.pyx`
- Accessor exists: `get_company_synergy(company_id, target_id)` returns bonus (nogil)
- Pattern established: `cpdef inline int ... noexcept nogil` with C array backing

### Return Values Required
- **Synergy income**: Total bonus income from all synergy pairs (for corp income calculation)
- **Synergy marker count**: Number of synergy pairs found (for Synergistic corp ability: +1 per 2 markers)
- Both values needed because:
  - SYN-03: Add synergy income to corporation total
  - CSA-03: Synergistic (S) ability uses marker count, not income

### Pair Counting Rule
- Count each pair ONCE: if A synergizes with B, that's ONE marker regardless of whether B also synergizes with A
- Per RULES.md line 569: "Count each pair **once only**"
- Implementation: only check (i, j) where i < j, OR use a visited set, OR sum lower triangle

### API Design
- Follow existing patterns in `core/data.pyx`
- Function takes list of company IDs (owned by corp) or GameState + corp_id
- Returns struct or tuple with (income, marker_count)

### Claude's Discretion
- Exact function signature (standalone in data.pyx vs method on corp entity)
- Whether to use struct return or two separate functions
- Optimization approach (iteration order, early exit conditions)

</decisions>

<specifics>
## Specific Ideas

- Check existing accessor patterns in `core/data.pyx` for consistency
- The synergy matrix is already populated via `_populate_synergies()` at module init
- Synergy bonuses range from 1 to 16 (int8 storage)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 21-synergy-infrastructure*
*Context gathered: 2026-01-28*
