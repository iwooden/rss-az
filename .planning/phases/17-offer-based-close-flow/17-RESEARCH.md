# Phase 17: Offer-Based Close Flow - Research

**Researched:** 2026-01-27
**Domain:** Player-driven company closing via offer system, hybrid phase pattern
**Confidence:** HIGH

## Summary

Phase 17 implements the player-decision portion of the CLOSING phase. After Phase 16 auto-close completes, players (and corporations they preside over) receive offers to voluntarily close companies with negative adjusted income. This follows the hybrid phase pattern established in ACQUISITION: deterministic auto-close (Phase 16) transitions to player-driven offers (Phase 17).

The codebase already has the complete pattern:
- Hybrid phase detection in `core/driver.pyx` (ACQUISITION checks `acq_active_corp == -1` for non-player mode)
- Offer buffer system in hidden state (250-slot buffer from ACQUISITION can be reused or separate buffer added)
- Offer presentation pattern: generate all upfront, present one at a time with re-validation
- `_close_company()` helper from Phase 16 handles actual closure including Junkyard Scrappers bonus
- State tracking via `TURN.closing_company` (one-hot encoded, -1 when no offer active)
- Action mask pattern: when `closing_company != -1`, enable `ACTION_CLOSE` and `ACTION_PASS`

**Primary recommendation:** Follow ACQUISITION's offer-queue pattern exactly. Generate all close offers at phase entry, store in hidden buffer sorted by face value ascending, present one at a time with dynamic re-validation for corp last-company rule.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Cython | 3.0+ | Performance-critical offer generation and validation | All phase handlers use Cython |
| Hidden state buffer | N/A | Offer queue storage outside NN-visible state | ACQUISITION uses identical pattern |
| entities.turn | N/A | `closing_company` state tracking | Established in Phase 1 |
| phases.closing | N/A | Shared close logic from Phase 16 | `_close_company()` already implemented |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| core.data | N/A | `get_company_face_value()`, `get_cost_of_ownership()` | Offer eligibility and sorting |
| entities.player | N/A | Player ownership queries | Identifying private companies for offers |
| entities.corp | N/A | Corp ownership, president checks | President control scope |
| core.actions | N/A | `ACTION_CLOSE`, `ACTION_PASS` | Player actions on offers |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Separate close offer buffer | Reuse ACQUISITION buffer | Separate buffer cleaner (phases don't share state), negligible cost |
| Generate offers on-demand | Pre-generate and sort | Pre-generation matches ACQUISITION, simpler |
| Snapshot validation | Re-validate each offer | Re-validation required for corp last-company rule |

## Architecture Patterns

### Recommended Project Structure
```
phases/
    closing.pyx           # Combined Phase 16 + 17 logic
    closing.pxd           # Header for cdef function declarations
```

**Note:** Phase 16 already created `closing.pyx`. Phase 17 extends it with offer-based logic.

### Pattern 1: Hybrid Phase Detection
**What:** Phase transitions from non-player to player mode based on state
**When to use:** When phase has both deterministic and decision-making portions
**Example:**
```cython
# Source: core/driver.pyx lines 48-56 (ACQUISITION pattern)
cdef bint _is_non_player_phase_check(GameState state, int phase) noexcept:
    if phase == PHASE_CLOSING:
        # CLOSING with no active offer = non-player auto-close phase
        # When closing_company == -1, still in auto-close (Phase 16)
        # When closing_company >= 0, player offers active (Phase 17)
        return turn_module.TURN.get_closing_company(state) == -1
    return False
```

### Pattern 2: Offer Generation and Buffering
**What:** Generate all offers upfront, store in hidden buffer, present one at a time
**When to use:** Offer-driven phases (ACQUISITION, CLOSING)
**Example:**
```cython
# Source: Based on phases/acquisition.pyx lines 307-357
cdef void _generate_close_offers(GameState state) noexcept:
    """
    Generate all close offers and store in hidden buffer.

    Offers sorted by face value ascending (cheapest first).
    Only companies with negative adjusted income.
    Only from players and player-presided corps (not FI, not receivership).

    Stores in hidden state buffer: [offer_count][offer_index][buffer...]
    Each offer: (owner_type, owner_id, company_id)
    """
    cdef int temp_owner_types[CLOSE_OFFER_BUFFER_SIZE]
    cdef int temp_owner_ids[CLOSE_OFFER_BUFFER_SIZE]
    cdef int temp_company_ids[CLOSE_OFFER_BUFFER_SIZE]
    cdef int temp_face_values[CLOSE_OFFER_BUFFER_SIZE]
    cdef int offer_count = 0

    # Initialize counters
    state._data[state._layout.hidden_close_offer_count_offset] = 0.0
    state._data[state._layout.hidden_close_offer_index_offset] = 0.0

    # Collect offers from players and player-presided corps
    offer_count = _collect_player_close_offers(state, temp_owner_types, temp_owner_ids,
                                                temp_company_ids, temp_face_values)
    offer_count += _collect_corp_close_offers(state, temp_owner_types, temp_owner_ids,
                                               temp_company_ids, temp_face_values, offer_count)

    # Sort by face value ascending (selection sort)
    _sort_close_offers_by_face_value(temp_owner_types, temp_owner_ids,
                                      temp_company_ids, temp_face_values, offer_count)

    # Write to buffer
    for i in range(offer_count):
        base = state._layout.hidden_close_offer_buffer_offset + (i * 3)
        state._data[base] = <float>temp_owner_types[i]
        state._data[base + 1] = <float>temp_owner_ids[i]
        state._data[base + 2] = <float>temp_company_ids[i]

    state._data[state._layout.hidden_close_offer_count_offset] = <float>offer_count
```

### Pattern 3: Dynamic Offer Validation
**What:** Re-validate each offer before presentation (prior accepts can invalidate later offers)
**When to use:** When offer validity depends on dynamic state changes
**Example:**
```cython
# Source: Based on phases/acquisition.pyx lines 382-423
cdef bint _is_close_offer_valid(GameState state, int owner_type, int owner_id,
                                 int company_id) noexcept:
    """
    Check if close offer is still valid for presentation.

    Invalid if:
    - Company already closed earlier in this phase
    - Owner no longer owns company
    - Corp owner would have 0 companies after close (last-company rule)

    Returns True if offer is valid.
    """
    # Check company still exists
    if company_module.COMPANIES[company_id].is_removed(state):
        return False

    # Check ownership
    if owner_type == LOC_PLAYER:
        if not player_module.PLAYERS[owner_id].owns_company(state, company_id):
            return False
    elif owner_type == LOC_CORP:
        if not corp_module.CORPS[owner_id].owns_company(state, company_id):
            return False

        # Corp last-company rule: count remaining companies
        if _count_corp_companies(state, owner_id, company_id) < 1:
            return False

    return True
```

### Pattern 4: Offer Presentation Loop
**What:** Loop through buffer, skip invalid offers, present first valid offer
**When to use:** Offer-based phases with dynamic validation
**Example:**
```cython
# Source: Based on phases/acquisition.pyx lines 425-489
cdef void _present_next_close_offer(GameState state) noexcept:
    """
    Advance to next valid offer and update visible state.

    Loops until valid offer found or offers exhausted.
    When no more offers, clears closing_company to signal phase end.
    """
    cdef int count = <int>state._data[state._layout.hidden_close_offer_count_offset]
    cdef int index = <int>state._data[state._layout.hidden_close_offer_index_offset]
    cdef int owner_type, owner_id, company_id, president, base

    while index < count:
        base = state._layout.hidden_close_offer_buffer_offset + (index * 3)
        owner_type = <int>state._data[base]
        owner_id = <int>state._data[base + 1]
        company_id = <int>state._data[base + 2]

        # Skip invalid offers
        if not _is_close_offer_valid(state, owner_type, owner_id, company_id):
            index += 1
            state._data[state._layout.hidden_close_offer_index_offset] = <float>index
            continue

        # Found valid offer - set visible state
        turn_module.TURN.set_closing_company(state, company_id)

        # Determine active player (owner or president)
        if owner_type == LOC_PLAYER:
            state._set_active_player(owner_id)
        elif owner_type == LOC_CORP:
            president = _get_corp_president(state, owner_id)
            state._set_active_player(president if president >= 0 else 0)
        return

    # No more valid offers - clear state
    turn_module.TURN.clear_closing_company(state)
```

### Pattern 5: Action Handlers
**What:** Accept closes company, Pass skips offer
**When to use:** Player actions on offers
**Example:**
```cython
# Source: New pattern based on ACQUISITION accept/pass
cdef void _handle_close_accept(GameState state) noexcept:
    """
    Accept current close offer: close the company and advance to next offer.
    """
    cdef int company_id = turn_module.TURN.get_closing_company(state)
    cdef int owner_type, owner_id

    # Determine current owner
    owner_type, owner_id = _get_company_owner(state, company_id)

    # Close company (reuses Phase 16 helper)
    _close_company(state, company_id, owner_type, owner_id)

    # Advance to next offer
    cdef int index = <int>state._data[state._layout.hidden_close_offer_index_offset]
    state._data[state._layout.hidden_close_offer_index_offset] = <float>(index + 1)
    _present_next_close_offer(state)

cdef void _handle_close_pass(GameState state) noexcept:
    """
    Pass on current close offer: keep company, advance to next offer.
    """
    cdef int index = <int>state._data[state._layout.hidden_close_offer_index_offset]
    state._data[state._layout.hidden_close_offer_index_offset] = <float>(index + 1)
    _present_next_close_offer(state)
```

### Anti-Patterns to Avoid
- **Generating offers during action handling:** Generate ALL offers at phase entry, not lazily
- **Assuming offer validity persists:** Always re-validate before presenting (corp last-company rule)
- **Including FI/receivership in offers:** These are handled by Phase 16 auto-close only
- **Forgetting to advance index on Pass:** Both Accept and Pass must advance the offer index

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Closing company logic | Duplicate close steps | `_close_company()` from Phase 16 | Already handles ownership clearing, JS bonus, removal |
| Hybrid phase detection | Custom state flags | `closing_company == -1` check | Matches ACQUISITION pattern exactly |
| Offer buffer layout | New buffer format | Copy ACQUISITION buffer pattern | Proven, tested, familiar |
| Sorting by face value | Custom sort | Selection sort from ACQUISITION | Simple, in-place, works for small N |
| President lookup | Scan shares | `_get_corp_president()` from ACQUISITION | Already implemented for corp-corp offers |

**Key insight:** ACQUISITION phase is the perfect template. Phase 17 is structurally identical: generate offers upfront, sort them, present one at a time, handle accept/pass actions, advance to next phase when exhausted.

## Common Pitfalls

### Pitfall 1: Including FI/Receivership in Offers
**What goes wrong:** Generating close offers for FI-owned or receivership-corp companies
**Why it happens:** Forgetting that FI and receivership are handled by Phase 16 auto-close
**How to avoid:** Per CONTEXT.md: "Receivership corps and FI excluded from offers (handled by auto-close only)"
**Warning signs:** Offers generated for companies that were already auto-closed

### Pitfall 2: Corp Last-Company Validation at Generation Time
**What goes wrong:** Checking corp last-company rule when generating offers, not when presenting
**Why it happens:** Misunderstanding dynamic validation requirement
**How to avoid:** Per CONTEXT.md: "Check at presentation time, not at generation time. If first of two corp offers accepted, second becomes invalid (skip it)."
**Warning signs:** Corp presented with offer to close its last remaining company

### Pitfall 3: Missing Adjusted Income Calculation
**What goes wrong:** Offering companies with positive or zero adjusted income
**Why it happens:** Forgetting to calculate `income - CoO` for eligibility
**How to avoid:** Per CLO-05: "Only offer companies with negative adjusted income"
**Warning signs:** Profitable companies appearing in close offers

### Pitfall 4: Forgetting Face Value Sort
**What goes wrong:** Presenting offers in arbitrary or generation order
**Why it happens:** Not implementing sort step after collection
**How to avoid:** Per CLO-06 and CONTEXT.md: "Sort by face value ascending (lowest first)"
**Warning signs:** High-value companies offered before low-value companies

### Pitfall 5: Not Advancing on Pass
**What goes wrong:** Pass action doesn't advance to next offer (infinite loop on same offer)
**Why it happens:** Forgetting that Pass permanently skips current offer
**How to avoid:** Both Accept and Pass must increment offer index and call `_present_next_close_offer()`
**Warning signs:** Player repeatedly sees same company in close offers

### Pitfall 6: President vs Owner Confusion
**What goes wrong:** Asking wrong player to decide on corp-owned company
**Why it happens:** Confusing company ownership with decision authority
**How to avoid:** Per CONTEXT.md: "President control scope: player's private companies + companies owned by corps player presides over"
**Warning signs:** Wrong player marked as active for corp company close decisions

## Code Examples

Verified patterns from official sources:

### Eligibility Check (Negative Adjusted Income)
```cython
# Source: Based on phases/closing.pyx lines 88-110 (FI auto-close pattern)
cdef bint _has_negative_adjusted_income(GameState state, int company_id) noexcept:
    """Check if company has negative adjusted income (eligible for close offer)."""
    cdef int coo_level = turn_module.TURN.get_coo_level(state)
    cdef int base_income = get_company_income(company_id)
    cdef int stars = get_company_stars(company_id)
    cdef int coo_value = get_cost_of_ownership(coo_level, stars)
    cdef int adjusted_income = base_income - coo_value

    return adjusted_income < 0  # NEGATIVE only, not zero
```

### Corp Company Count (Last-Company Rule)
```cython
# Source: Based on phases/acquisition.pyx lines 541-563 (_count_seller_companies)
cdef int _count_corp_companies(GameState state, int corp_id, int exclude_company_id) noexcept:
    """
    Count companies corp retains after excluding target.

    Used to enforce last-company rule: corp cannot close if it would have 0 companies left.
    Returns count of companies excluding the specified company_id.
    """
    cdef int count = 0
    cdef int company_id

    for company_id in range(GameConstants.NUM_COMPANIES):
        if company_id == exclude_company_id:
            continue
        if corp_module.CORPS[corp_id].owns_company(state, company_id):
            count += 1

    return count
```

### Face Value Sorting
```cython
# Source: Based on phases/acquisition.pyx lines 80-103 (selection sort pattern)
cdef void _sort_close_offers_by_face_value(
    int* owner_types, int* owner_ids, int* company_ids, int* face_values, int count
) noexcept:
    """Sort close offers by face value ascending (lowest first)."""
    cdef int i, j, best_idx, best_fv, curr_fv
    cdef int swap_ot, swap_oi, swap_cid, swap_fv

    for i in range(count):
        best_idx = i
        best_fv = face_values[i]

        for j in range(i + 1, count):
            curr_fv = face_values[j]
            if curr_fv < best_fv:  # Lower face value wins
                best_idx = j
                best_fv = curr_fv

        # Swap to front
        if best_idx != i:
            swap_ot = owner_types[i]
            owner_types[i] = owner_types[best_idx]
            owner_types[best_idx] = swap_ot

            swap_oi = owner_ids[i]
            owner_ids[i] = owner_ids[best_idx]
            owner_ids[best_idx] = swap_oi

            swap_cid = company_ids[i]
            company_ids[i] = company_ids[best_idx]
            company_ids[best_idx] = swap_cid

            swap_fv = face_values[i]
            face_values[i] = face_values[best_idx]
            face_values[best_idx] = swap_fv
```

### Phase Entry Setup
```cython
# Source: Based on phases/acquisition.pyx lines 813-835 (setup_acquisition_phase)
cpdef void setup_close_offers_phase(GameState state):
    """
    Set up offer-based closing at entry (called after auto-close completes).

    Steps:
    1. Clear offer buffer index to 0
    2. Generate all close offers into buffer (sorted by face value)
    3. Present first valid offer (or transition if none)

    This is called from apply_closing_auto after auto-close finishes.
    """
    # Reset offer tracking
    state._data[state._layout.hidden_close_offer_index_offset] = 0.0
    state._data[state._layout.hidden_close_offer_count_offset] = 0.0

    # Generate offers (populates buffer)
    _generate_close_offers(state)

    # Present first offer (or transition to INCOME if none)
    _present_next_close_offer(state)
```

### Phase Transition (No More Offers)
```cython
# Source: Based on phases/closing.pyx lines 207-222 (Phase 16 transition)
cdef void _transition_to_income(GameState state) noexcept:
    """
    Complete CLOSING phase and transition to INCOME.

    Called when no more close offers exist (closing_company == -1).

    Steps:
    1. Increment turn number
    2. Clear per-turn tracking
    3. Transition to INCOME phase
    """
    cdef int current_turn = turn_module.TURN.get_turn_number(state)
    cdef int i

    # Check for terminal state
    if _is_game_terminal(state):
        turn_module.TURN.set_phase(state, PHASE_GAME_OVER)
        return

    # Increment turn number
    turn_module.TURN.set_turn_number(state, current_turn + 1)

    # Clear per-turn tracking for all players
    for i in range(state._num_players):
        player_module.PLAYERS[i].clear_roundtrip_tracking(state)

    # Transition to INCOME phase
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INCOME)
```

### Main Action Handler
```cython
# Source: Based on phases/acquisition.pyx lines 986-1031
cdef int apply_closing_action(GameState state, ActionInfo* info) noexcept:
    """
    Apply CLOSING phase action to state.

    Action types:
    - ACTION_CLOSE: Close current company
    - ACTION_PASS: Keep company, skip to next offer

    Returns: 0=success, 1=invalid
    """
    cdef int company_id

    if info.action_type == ACTION_CLOSE:
        # Validate offer still active
        company_id = turn_module.TURN.get_closing_company(state)
        if company_id < 0:
            return 1  # No active offer

        # Close company
        _handle_close_accept(state)
        return 0

    elif info.action_type == ACTION_PASS:
        # Validate offer still active
        company_id = turn_module.TURN.get_closing_company(state)
        if company_id < 0:
            return 1  # No active offer

        # Pass on offer
        _handle_close_pass(state)
        return 0

    # Unknown action type
    return 1
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Phase 16 stub to INVEST | Phase 17 offer-based closing | This phase | Correct game flow, player agency |
| Manual offer generation | Upfront buffer pattern | ACQUISITION (Phase 12-13) | Established pattern to follow |
| Single company offers | Batch generation with sorting | ACQUISITION | Consistent with other offer phases |

**Deprecated/outdated:**
- Phase 16 temporary transition to INVEST will be replaced with offer generation call

## Open Questions

Things that couldn't be fully resolved:

1. **Buffer size for close offers**
   - What we know: ACQUISITION uses 250-slot buffer for acquisition offers
   - What's unclear: Whether close offers need separate buffer or can share
   - Recommendation: Use separate buffer (cleaner, no phase interaction risk). Size 100 should suffice (max ~50 companies * 2 owner types = ~100 theoretical max, reality much lower)

2. **Phase transition timing**
   - What we know: Phase 16 transitions to INVEST (temporary), should go to Phase 17 offers, then to Phase 18 mandatory close, then to INCOME
   - What's unclear: Whether Phase 17 transitions directly to INCOME or to Phase 18
   - Recommendation: Per requirements traceability, Phase 17 should transition to INCOME (Phase 18 is separate mandatory close). Update when Phase 18 is planned.

3. **Player vs Corp offer mixing**
   - What we know: "If player is president of multiple corps, all companies in single pool (mixed with privates)" - CONTEXT.md
   - What's unclear: Whether player private offers and corp offers are truly intermixed or presented in groups
   - Recommendation: Mix completely - sort ALL offers (player + corp) by face value ascending. Simpler and matches "single pool" description.

## Sources

### Primary (HIGH confidence)
- `phases/closing.pyx` - Phase 16 implementation with `_close_company()` helper
- `phases/acquisition.pyx` - Complete offer-queue pattern (lines 307-1031)
- `core/driver.pyx` - Hybrid phase detection pattern (lines 39-85)
- `entities/turn.pyx` - `closing_company` state tracking (lines 378-388)
- `core/actions.pyx` - Action mask for CLOSING (lines 378-385)
- `17-CONTEXT.md` - User decisions on offer ordering, validation, scope
- `REQUIREMENTS.md` - CLO-05 through CLO-13 requirements

### Secondary (MEDIUM confidence)
- `VECTORS.md` - State representation for `closing_company` (line 198)
- `.planning/phases/16-auto-close-logic/16-RESEARCH.md` - Phase 16 patterns

### Tertiary (LOW confidence)
- None - all patterns verified in codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All components exist, ACQUISITION is proven template
- Architecture: HIGH - Direct adaptation of ACQUISITION patterns
- Pitfalls: HIGH - Based on user decisions and ACQUISITION lessons learned

**Research date:** 2026-01-27
**Valid until:** N/A (project-specific patterns don't expire)
