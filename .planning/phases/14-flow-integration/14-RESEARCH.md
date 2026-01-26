# Phase 14: Flow & Integration - Research

**Researched:** 2026-01-26
**Domain:** Cython phase flow control, receivership auto-buy, zone merging
**Confidence:** HIGH

## Summary

This phase integrates the offer infrastructure (Phase 12) and action handlers (Phase 13) into a complete acquisition flow. The key additions are:

1. **Receivership auto-buy**: Corps in receivership automatically buy from FI when affordable, skipping player involvement
2. **Phase transitions**: ACQUISITION transitions to CLOSING when all offers exhausted
3. **Zone merging**: At phase exit, acquisition_companies merge into owned_companies and acquisition_proceeds merge into cash

The implementation follows established patterns in the codebase: the forced-action auto-apply loop from `driver.pyx` (lines 203-230), the deterministic non-player phase execution pattern from WRAP_UP, and entity handle operations for state manipulation.

**Primary recommendation:** Implement receivership auto-buy inside `_present_current_offer` using a while-loop pattern, transition to CLOSING when `acq_active_corp == -1`, and merge zones before transitioning.

## Standard Stack

This phase uses existing codebase infrastructure - no external libraries needed.

### Core
| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| `phases/acquisition.pyx` | Current | All flow logic implemented here | Contains offer infrastructure |
| `core/driver.pyx` | Current | Auto-apply loop, phase checks | Driver dispatches ACQUISITION actions |
| `entities/turn.pyx` | Current | Phase transitions, offer state | TURN.set_phase(), TURN.get_acq_active_corp() |
| `entities/corp.pyx` | Current | Corp receivership, cash, companies | is_in_receivership(), acquisition zones |
| `entities/player.pyx` | Current | Player acquisition_proceeds | Merge proceeds to cash |

### Supporting
| Component | Purpose | When to Use |
|-----------|---------|-------------|
| `entities/company.pyx` | Company location tracking | transfer_to_corp() for merge |
| `entities/fi.pyx` | FI cash for receivership buys | FI.add_cash() on auto-buy |
| `core/data.pxd` | PHASE_CLOSING constant | Phase transition target |

## Architecture Patterns

### Recommended Module Structure

```
phases/acquisition.pyx additions:
    # RECEIVERSHIP AUTO-BUY (Phase 14 - NEW)
    cdef bint _is_corp_in_receivership(GameState state, int corp_id) noexcept
    cdef void _execute_receivership_auto_buy(GameState state) noexcept

    # OFFER PRESENTATION (modify existing)
    cdef void _present_current_offer(GameState state) noexcept
    # -> Add receivership auto-buy loop before returning

    # ZONE MERGING (Phase 14 - NEW)
    cdef void _merge_acquisition_zones(GameState state) noexcept
    cdef void _merge_player_proceeds(GameState state) noexcept
    cdef void _merge_corp_proceeds(GameState state) noexcept
    cdef void _merge_corp_companies(GameState state) noexcept

    # PHASE TRANSITION (Phase 14 - NEW)
    cdef void _transition_to_closing(GameState state) noexcept
```

### Pattern 1: Receivership Auto-Buy in Presentation Loop

**What:** When presenting an offer to a receivership corp, auto-execute or auto-pass instead of returning to driver
**When to use:** Inside `_present_current_offer` before returning with a valid offer
**Source:** Follows forced-action pattern from `driver.pyx` lines 203-230

```cython
cdef void _present_current_offer(GameState state) noexcept:
    """
    Update visible state to reflect current offer in buffer.

    For receivership corps: auto-execute buy if affordable, else auto-pass.
    Loops until a player-president offer is found or offers exhausted.
    """
    cdef int count = <int>state._data[state._layout.hidden_offer_count_offset]
    cdef int index = <int>state._data[state._layout.hidden_offer_index_offset]
    cdef int corp_id, company_id, president, base
    cdef int face_value, corp_cash
    cdef bint is_fi_offer

    while index < count:
        base = state._layout.hidden_offer_buffer_offset + (index * 2)
        corp_id = <int>state._data[base]
        company_id = <int>state._data[base + 1]

        # Skip invalid offers
        if not _is_offer_valid(state, corp_id, company_id):
            index += 1
            state._data[state._layout.hidden_offer_index_offset] = <float>index
            continue

        # Check if buying corp is in receivership
        if corp_module.CORPS[CORP_NAMES[corp_id]].is_in_receivership(state):
            is_fi_offer = fi_module.FI.owns_company(state, company_id)

            # Receivership only buys from FI (per RULES.md)
            if is_fi_offer:
                face_value = get_company_face_value(company_id)
                corp_cash = corp_module.CORPS[CORP_NAMES[corp_id]].get_cash(state)

                if corp_cash >= face_value:
                    # Auto-execute: buy at face value (like OS)
                    _execute_receivership_fi_buy(state, corp_id, company_id)
                # else: auto-pass by falling through

            # Advance to next offer (auto-pass)
            index += 1
            state._data[state._layout.hidden_offer_index_offset] = <float>index
            continue

        # Found player-president offer - set visible state and return
        turn_module.TURN.set_acq_active_corp(state, corp_id)
        turn_module.TURN.set_acq_target_company(state, company_id)
        turn_module.TURN.set_acq_fi_offer(state, fi_module.FI.owns_company(state, company_id))

        president = _get_corp_president(state, corp_id)
        state._set_active_player(president if president >= 0 else 0)
        return

    # No more valid offers - clear acquisition state
    turn_module.TURN.clear_acq_active_corp(state)
    turn_module.TURN.clear_acq_target_company(state)
    turn_module.TURN.set_acq_fi_offer(state, False)
```

### Pattern 2: Zone Merging at Phase Exit

**What:** Transfer acquisition zone contents to permanent locations before CLOSING
**When to use:** When ACQUISITION detects no more offers (before transition)
**Source:** Entity handle patterns from `entities/corp.pyx`, `entities/player.pyx`

```cython
cdef void _merge_acquisition_zones(GameState state) noexcept:
    """
    Merge all acquisition zones into final state.

    Called before transitioning to CLOSING phase.
    - Player acquisition_proceeds -> player.cash
    - Corp acquisition_proceeds -> corp.cash
    - Corp acquisition_companies -> corp.owned_companies
    """
    _merge_player_proceeds(state)
    _merge_corp_proceeds(state)
    _merge_corp_companies(state)


cdef void _merge_player_proceeds(GameState state) noexcept:
    """Add player acquisition_proceeds to cash, then clear."""
    cdef int player_id, proceeds
    for player_id in range(state._num_players):
        proceeds = player_module.PLAYERS[player_id].get_acquisition_proceeds(state)
        if proceeds > 0:
            player_module.PLAYERS[player_id].add_cash(state, proceeds)
            player_module.PLAYERS[player_id].clear_acquisition_proceeds(state)


cdef void _merge_corp_proceeds(GameState state) noexcept:
    """Add corp acquisition_proceeds to cash, then clear."""
    cdef int corp_id, proceeds
    for corp_id in range(GameConstants.NUM_CORPS):
        if corp_module.CORPS[CORP_NAMES[corp_id]].is_active(state):
            proceeds = corp_module.CORPS[CORP_NAMES[corp_id]].get_acquisition_proceeds(state)
            if proceeds > 0:
                corp_module.CORPS[CORP_NAMES[corp_id]].add_cash(state, proceeds)
                corp_module.CORPS[CORP_NAMES[corp_id]].set_acquisition_proceeds(state, 0)


cdef void _merge_corp_companies(GameState state) noexcept:
    """Transfer acquisition_companies to owned_companies, then clear."""
    cdef int corp_id, company_id
    for corp_id in range(GameConstants.NUM_CORPS):
        if corp_module.CORPS[CORP_NAMES[corp_id]].is_active(state):
            for company_id in range(GameConstants.NUM_COMPANIES):
                if corp_module.CORPS[CORP_NAMES[corp_id]].has_acquisition_company(state, company_id):
                    # Transfer from acquisition to owned
                    company_module.COMPANIES[company_id].transfer_to_corp(state, corp_id)
                    # Note: transfer_to_corp clears acquisition_company flag via clear_location
```

### Pattern 3: Phase Transition to CLOSING

**What:** Detect when ACQUISITION should end and transition to CLOSING
**When to use:** When `acq_active_corp == -1` after presenting offers
**Source:** Driver hybrid phase detection in `_is_non_player_phase_check`

```cython
# In driver.pyx: ACQUISITION with no active corp transitions to CLOSING

cdef bint _is_non_player_phase_check(GameState state, int phase) noexcept:
    """Check if phase has no player actions."""
    if phase == PHASE_WRAP_UP:
        return True

    if phase == PHASE_ACQUISITION:
        # No active corp = offers exhausted = transition needed
        if turn_module.TURN.get_acq_active_corp(state) == -1:
            return True

    return False

# The existing _execute_non_player_phase handles ACQUISITION stub
# Modify to: merge zones, then transition to CLOSING
```

### Anti-Patterns to Avoid

- **Checking receivership after driver involvement:** Receivership auto-buy must happen BEFORE driver sees the state
- **Transitioning without zone merge:** Always merge acquisition zones before leaving ACQUISITION
- **Modifying owned_companies while company is in acquisition:** Let Company.transfer_to_corp handle flag management
- **Forgetting to clear acquisition proceeds:** Zero out after merging, not just after transfer

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Corp receivership check | Manual bank_shares check | `corp.is_in_receivership(state)` | Entity tracks this flag |
| Company relocation | Manual flag toggling | `Company.transfer_to_corp()` | Handles all location bookkeeping |
| President lookup | Share iteration | `_get_corp_president()` exists | Phase 12 already implemented this |
| Offer validation | Re-implement checks | `_is_offer_valid()` exists | Consistency with presentation |

## Common Pitfalls

### Pitfall 1: Receivership Check After Player Action

**What goes wrong:** Receivership corp offer reaches driver, action mask generated
**Why it happens:** Receivership check not in presentation loop
**How to avoid:** Check receivership BEFORE setting visible state in `_present_current_offer`
**Warning signs:** Action mask generated for receivership offers

### Pitfall 2: Zone Merge Missing Acquisition Companies

**What goes wrong:** Companies stay in acquisition_companies after phase ends
**Why it happens:** Only merged proceeds, forgot companies
**How to avoid:** `_merge_acquisition_zones` calls both proceeds and companies merge
**Warning signs:** Company not in owned_companies after CLOSING starts

### Pitfall 3: Receivership Corp Tries Non-FI Buy

**What goes wrong:** Receivership corp buys from player/corp instead of FI only
**Why it happens:** Auto-buy executed for non-FI offers
**How to avoid:** Check `is_fi_offer` before receivership auto-execute
**Warning signs:** Corp acquires company from non-FI source during receivership

### Pitfall 4: Double Transition to CLOSING

**What goes wrong:** ACQUISITION tries to transition twice or loops
**Why it happens:** Transition happens both in action handler and stub
**How to avoid:** Transition only when `acq_active_corp == -1` detected, stub checks state
**Warning signs:** Phase changes unexpectedly or infinite loop

### Pitfall 5: Proceeds Not Cleared After Merge

**What goes wrong:** Proceeds added multiple times or persist to next phase
**Why it happens:** Forgot to zero out acquisition_proceeds after adding to cash
**How to avoid:** Always `set_acquisition_proceeds(state, 0)` or `clear_acquisition_proceeds(state)` after merge
**Warning signs:** Player/corp cash is inflated

### Pitfall 6: Infinite Loop on Invalid Offers

**What goes wrong:** Presentation loop never exits
**Why it happens:** Index not incremented when offer is invalid
**How to avoid:** Always increment index in all loop branches
**Warning signs:** Endless while loop

## Code Examples

### Receivership FI Auto-Buy

```cython
cdef void _execute_receivership_fi_buy(GameState state, int corp_id, int company_id) noexcept:
    """
    Execute FI purchase for receivership corp at face value.

    Receivership corps always buy from FI at face value (like OS special ability).
    This mirrors _handle_fi_buy_face but without advancing offer index
    (caller handles that).
    """
    cdef int face_value = get_company_face_value(company_id)

    # Transfer money: corp -> FI
    corp_module.CORPS[CORP_NAMES[corp_id]].add_cash(state, -face_value)
    fi_module.FI.add_cash(state, face_value)

    # Transfer company to corp's acquisition zone
    company_module.COMPANIES[company_id].transfer_to_corp_acquisition(state, corp_id)
```

### Transition to CLOSING

```cython
cdef void _transition_to_closing(GameState state) noexcept:
    """
    Complete ACQUISITION phase and transition to CLOSING.

    Steps:
    1. Merge all acquisition zones
    2. Set phase to CLOSING
    """
    # Merge before leaving
    _merge_acquisition_zones(state)

    # Transition
    turn_module.TURN.set_phase(state, GamePhases.PHASE_CLOSING)
```

### Driver Integration Update

```cython
# In _execute_non_player_phase (driver.pyx):

cdef void _execute_non_player_phase(GameState state, object history):
    """Execute deterministic non-player phase and record to history."""
    cdef int phase = state.get_phase()
    cdef int sentinel

    if phase == PHASE_WRAP_UP:
        sentinel = ACTION_WRAP_UP_SENTINEL
    elif phase == PHASE_ACQUISITION:
        sentinel = ACTION_ACQUISITION_SENTINEL
    else:
        return

    # Record state BEFORE execution
    if history is not None:
        history.append((state._array.copy(), sentinel))

    # Execute phase logic
    if phase == PHASE_WRAP_UP:
        apply_wrap_up(state)
    elif phase == PHASE_ACQUISITION:
        # ACQUISITION with no offers transitions to CLOSING
        _transition_to_closing(state)  # New function
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `apply_acquisition_stub` transitions to INVEST | Real flow to CLOSING | Phase 14 | Complete acquisition flow |
| No receivership handling | Auto-buy in presentation loop | Phase 14 | Receivership corps work |
| No zone merging | Merge before CLOSING | Phase 14 | Proper state finalization |

## Driver Integration Changes

### Current State (Phase 13)

```python
# driver.pyx line 46-48
if phase == PHASE_ACQUISITION:
    # ACQUISITION with no active corp = no offers = non-player phase
    return turn_module.TURN.get_acq_active_corp(state) == -1
```

### Phase 14 Changes

1. **`_is_non_player_phase_check`**: No changes needed - already checks `acq_active_corp == -1`
2. **`_execute_non_player_phase`**: Replace `apply_acquisition_stub` call with `_transition_to_closing`
3. **Action dispatch**: No changes - `apply_acquisition_action` handles player offers

### Flow Summary

```
WRAP_UP
  -> generate_offers + present_first
  -> set PHASE_ACQUISITION

ACQUISITION (offers exist, acq_active_corp >= 0)
  -> driver returns to caller for action
  -> apply_acquisition_action (accept/pass)
  -> _advance_to_next_offer
  -> _present_current_offer (handles receivership loop)

ACQUISITION (no offers, acq_active_corp == -1)
  -> _is_non_player_phase_check returns True
  -> _execute_non_player_phase
  -> _transition_to_closing (merge + set PHASE_CLOSING)
```

## Open Questions

None - CONTEXT.md provides complete specification.

## Sources

### Primary (HIGH confidence)
- `/home/icebreaker/rss-az-cython2/core/driver.pyx` - Forced action loop, non-player phase pattern
- `/home/icebreaker/rss-az-cython2/phases/acquisition.pyx` - Offer infrastructure, action handlers
- `/home/icebreaker/rss-az-cython2/phases/wrap_up.pyx` - Non-player phase pattern
- `/home/icebreaker/rss-az-cython2/entities/corp.pyx` - is_in_receivership, acquisition zones
- `/home/icebreaker/rss-az-cython2/entities/player.pyx` - acquisition_proceeds
- `/home/icebreaker/rss-az-cython2/RULES.md` - Receivership corp behavior specification

### Secondary (MEDIUM confidence)
- `/home/icebreaker/rss-az-cython2/.planning/phases/13-actions-validation/13-RESEARCH.md` - Phase 13 patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All components exist in codebase
- Architecture: HIGH - Following established driver/phase patterns
- Pitfalls: HIGH - Based on careful analysis of existing code paths

**Research date:** 2026-01-26
**Valid until:** Until Phase 14 completion (internal codebase, stable patterns)
