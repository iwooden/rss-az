---
phase: 17
plan: 01
subsystem: closing
tags: [state-layout, offer-generation, hidden-buffer, closing-phase]
requires: [16-02]
provides:
  - close-offer-buffer
  - close-offer-generation-functions
affects: [17-02]
tech-stack:
  added: []
  patterns:
    - hidden-buffer-pattern
    - selection-sort-pattern
    - owner-type-tuple-pattern
key-files:
  created: []
  modified:
    - core/state.pxd
    - core/state.pyx
    - phases/closing.pyx
    - phases/closing.pxd
decisions:
  - id: separate-buffer
    title: Separate close offer buffer from ACQUISITION
    rationale: Cleaner phase isolation, negligible cost
  - id: face-value-ascending
    title: Sort offers by face value ascending (lowest first)
    rationale: Consistent with Phase 18 mandatory closing, simplifies bankruptcy prevention
  - id: three-field-tuples
    title: Store offers as (owner_type, owner_id, company_id)
    rationale: Distinguishes player vs corp ownership for action handling
metrics:
  duration: 15min
  completed: 2026-01-27
---

# Phase 17 Plan 01: Close Offer Buffer Infrastructure Summary

**One-liner:** Hidden state buffer for close offers with generation functions that filter by negative adjusted income and sort by face value ascending

## What Was Delivered

Phase 17-01 adds the state infrastructure and core logic for offer-based closing:

1. **State layout extended** with close offer buffer (302 floats)
   - `hidden_close_offer_count_offset` - number of offers
   - `hidden_close_offer_index_offset` - current offer position
   - `hidden_close_offer_buffer_offset` - start of 100-offer buffer
   - Each offer: 3 floats (owner_type, owner_id, company_id)

2. **Offer generation functions** in `phases/closing.pyx`
   - `_has_negative_adjusted_income()` - Filters eligible companies (income - CoO < 0)
   - `_get_corp_president()` - Identifies president for corp ownership
   - `_collect_player_close_offers()` - Gathers player private company offers
   - `_collect_corp_close_offers()` - Gathers corp-owned company offers (excludes receivership)
   - `_sort_close_offers_by_face_value()` - Selection sort ascending
   - `_generate_close_offers()` - Main orchestration function

3. **Header declarations** in `phases/closing.pxd` for cdef function access

**Scope:** Infrastructure only - no integration with `apply_closing_auto` yet (Plan 02)

## Technical Implementation

### Architecture Pattern: Hidden Buffer with Pre-Generation

Following ACQUISITION pattern exactly:
- Generate ALL offers upfront at phase entry
- Store in hidden state (not visible to NN)
- Sort once, present one at a time
- 100-slot buffer (3 floats per offer vs ACQUISITION's 2)

### Key Design Decisions

**Buffer sizing:** 100 offers (300 floats) sufficient for worst case
- Max ~50 companies × 2 owner types = 100 theoretical max
- Reality much lower (FI and receivership excluded)

**Owner type encoding:**
- `OWNER_PLAYER = 0` - Player-owned private companies
- `OWNER_CORP = 1` - Corp-owned companies (where player is president)
- Distinguishes who decides: player directly vs player as president

**Filtering logic:**
- ONLY negative adjusted income (`income - CoO < 0`)
- NOT zero (zero means profitable with no net cost)
- Excludes FI-owned companies (handled by Phase 16 auto-close)
- Excludes receivership corps (no president, handled by Phase 16 auto-close)

**Sort order:**
- Face value ascending (lowest first)
- Rationale: Consistent with Phase 18 mandatory closing
- Simplifies bankruptcy prevention logic (cheaper companies closed first)

### Code Organization

```
phases/closing.pyx
├── Constants (CLOSE_OFFER_BUFFER_SIZE, OWNER_*)
├── Existing Phase 16 logic (_close_company, auto-close functions)
├── NEW: Close offer eligibility (_has_negative_adjusted_income)
├── NEW: Close offer collection (_collect_player_*, _collect_corp_*)
├── NEW: Close offer sorting (_sort_close_offers_by_face_value)
└── NEW: Close offer generation (_generate_close_offers)
```

## Testing Results

**Build verification:** ✅ `python setup.py build_ext --inplace` succeeded
**Test suite:** ✅ All 268 tests passed (no regressions)

**Coverage:**
- State layout compiles correctly
- Offer generation functions compile without errors
- No runtime testing yet (functions not called - Plan 02 integration)

## Deviations from Plan

None - plan executed exactly as written.

## Commits

| Commit  | Type | Description |
|---------|------|-------------|
| ed337fc | feat | Add close offer buffer to state layout |
| 370afe7 | feat | Implement close offer generation and sorting |

**Files modified:**
- `core/state.pxd` - StateLayout struct with close offer buffer offsets
- `core/state.pyx` - Layout computation (+302 floats)
- `phases/closing.pyx` - Offer generation and sorting functions
- `phases/closing.pxd` - Function declarations

## Next Phase Readiness

**Blockers:** None

**Plan 17-02 ready:** YES
- Buffer infrastructure in place
- Generation functions complete
- Next: Integrate `_generate_close_offers()` call into `apply_closing_auto`
- Next: Implement offer presentation loop and action handlers

**Dependencies satisfied:**
- Phase 16 `_close_company()` helper available for reuse
- State layout supports hybrid phase detection pattern
- ACQUISITION pattern verified and ready to replicate

## Knowledge Capture

### Patterns Established

**Three-field offer tuples:** Unlike ACQUISITION's (corp_id, company_id), close offers need (owner_type, owner_id, company_id) because:
- Player can decide on their own private companies
- Player can decide on companies owned by corps they preside over
- Action handler needs to know which entity actually owns the company

**Separate buffers:** ACQUISITION buffer and close offer buffer are separate because:
- Phases don't overlap (ACQUISITION before CLOSING in turn cycle)
- Separate buffers = no phase interaction risk
- Cleaner code (no buffer slot reinterpretation)

### Gotchas Avoided

**Negative vs zero:** `adjusted_income < 0` NOT `<= 0`
- Zero means break-even (no profit, but no cost either)
- Only negative income companies are eligible for closing

**Receivership exclusion:** Check president exists before including corp offers
- Receivership corps have no president (`_get_corp_president()` returns -1)
- These are handled by Phase 16 auto-close only

**FI exclusion:** FI-owned companies never appear in player offers
- FI auto-close (Phase 16) handles all FI companies
- No player decision needed for FI

### Technical Debt

None introduced.

## Session Notes

**Duration:** ~15 minutes
**Approach:** Direct implementation following ACQUISITION pattern
**Confidence:** HIGH - established patterns, clear requirements

**Build notes:**
- All modules recompiled due to state.pxd changes (expected)
- Zero compilation errors
- Zero test failures

---

*Completed: 2026-01-27*
*Phase: 17-offer-based-close-flow*
*Plan: 01*
