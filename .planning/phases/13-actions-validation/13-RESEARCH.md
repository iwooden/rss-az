# Phase 13: Actions & Validation - Research

**Researched:** 2026-01-25
**Domain:** Cython action handlers with validation for acquisition offers
**Confidence:** HIGH

## Summary

This research documents the patterns and approaches needed to implement accept/pass actions for acquisition offers. The phase builds on Phase 12's offer infrastructure and follows established patterns from the INVEST phase action handling.

The key challenge is implementing robust validation that covers all edge cases (cash affordability, seller minimum ownership, same-president constraints) while maintaining the existing action mask pattern. The validation must happen both in the action mask (pre-emptive) and in the action handler (defensive).

**Primary recommendation:** Implement `apply_acquisition_action` following the INVEST phase pattern with a switch-case on action type, validating each action before execution and returning error codes for invalid actions.

## Standard Stack

This phase uses existing codebase infrastructure - no external libraries needed.

### Core
| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| Cython 3.0+ | Current | Low-level action handlers | Already in use, nogil compatibility |
| entities module | Current | Entity handles (corp, player, company, fi) | Established pattern |
| core/actions.pyx | Current | Action decoding, mask generation | Contains action constants |
| core/data.pyx | Current | Company price lookups | `get_company_low_price`, `get_company_high_price`, `get_company_face_value` |

### Supporting
| Component | Purpose | When to Use |
|-----------|---------|-------------|
| `phases/acquisition.pyx` | Phase handler location | All action implementation goes here |
| `entities/turn.pyx` | TURN.get_acq_active_corp(), TURN.get_acq_target_company() | Get current offer state |
| `entities/company.pyx` | Company.transfer_to_corp_acquisition() | Execute company transfers |

## Architecture Patterns

### Recommended Module Structure

```
phases/acquisition.pyx
    # OFFER GENERATION HELPERS (existing from Phase 12)
    cdef _collect_fi_offers(...)
    cdef _collect_corp_corp_offers(...)
    cdef _collect_player_private_offers(...)

    # MAIN OFFER GENERATION (existing from Phase 12)
    cdef _generate_offers(...)
    cdef _present_current_offer(...)
    cdef _advance_to_next_offer(...)

    # ACTION HANDLERS (Phase 13 - NEW)
    cdef void _handle_accept_price(GameState state, int price) noexcept
    cdef void _handle_fi_buy_high(GameState state) noexcept
    cdef void _handle_fi_buy_face(GameState state) noexcept
    cdef void _handle_pass(GameState state) noexcept

    # VALIDATION HELPERS (Phase 13 - NEW)
    cdef bint _validate_price_action(GameState state, int price) noexcept
    cdef bint _validate_fi_buy_high(GameState state) noexcept
    cdef bint _validate_fi_buy_face(GameState state) noexcept
    cdef int _count_seller_companies(GameState state, int seller_type, int seller_id) noexcept

    # MAIN PHASE HANDLER (Phase 13 - NEW)
    cdef int apply_acquisition_action(GameState state, ActionInfo* info) noexcept
```

### Pattern 1: Action Handler with Validation

**What:** Phase handler that validates and executes actions following INVEST pattern
**When to use:** For all action execution in ACQUISITION phase
**Source:** `phases/invest.pyx` lines 329-391

```cython
cdef int apply_acquisition_action(GameState state, ActionInfo* info) noexcept:
    """
    Apply ACQUISITION phase action to state.

    Returns: 0=success, 1=invalid
    """
    if info.action_type == ACTION_ACQ_PRICE:
        # Calculate actual price from offset
        cdef int company_id = TURN.get_acq_target_company(state)
        cdef int low_price = get_company_low_price(company_id)
        cdef int price = low_price + info.amount

        # Validate and execute
        if not _validate_price_action(state, price):
            return 1
        _handle_accept_price(state, price)
        return 0

    elif info.action_type == ACTION_ACQ_FI_HIGH:
        if not _validate_fi_buy_high(state):
            return 1
        _handle_fi_buy_high(state)
        return 0

    elif info.action_type == ACTION_ACQ_FI_FACE:
        if not _validate_fi_buy_face(state):
            return 1
        _handle_fi_buy_face(state)
        return 0

    elif info.action_type == ACTION_PASS:
        _handle_pass(state)
        return 0

    return 1  # Invalid action type
```

### Pattern 2: Seller Company Count Check

**What:** Count owned + acquisition_companies to validate minimum ownership
**When to use:** Before accepting any non-FI acquisition

```cython
cdef int _count_seller_companies(GameState state, int seller_type, int seller_id) noexcept:
    """
    Count companies seller retains after selling current target.

    seller_type: 0=FI, 1=Player, 2=Corp
    Returns: total owned_companies + acquisition_companies (excluding target)
    """
    cdef int count = 0
    cdef int company_id
    cdef int target = TURN.get_acq_target_company(state)

    if seller_type == 2:  # Corp
        for company_id in range(GameConstants.NUM_COMPANIES):
            if company_id == target:
                continue
            if corp_module.CORPS[CORP_NAMES[seller_id]].owns_company(state, company_id):
                count += 1
            if corp_module.CORPS[CORP_NAMES[seller_id]].has_acquisition_company(state, company_id):
                count += 1
    # FI has no minimum (seller_type=0)
    # Players have no minimum (seller_type=1)

    return count
```

### Pattern 3: Internal Pass Loop

**What:** Pass advances offer index internally, loops to present next
**When to use:** When player declines current offer

```cython
cdef void _handle_pass(GameState state) noexcept:
    """
    Pass on current offer, advance to next.

    Pass permanently skips - offer_index advances, _present_current_offer
    handles finding/presenting the next valid offer or clearing state.
    """
    _advance_to_next_offer(state)
    # _advance_to_next_offer calls _present_current_offer internally
```

### Anti-Patterns to Avoid

- **Re-checking validation in handler:** Action mask already validates; handler validation is defensive, not primary
- **Modifying seller state before buyer:** Always transfer money first, then transfer company
- **Forgetting acquisition_companies in seller count:** Corp seller retains ownership if owned + acquisition >= 2

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Price calculation | Custom formula | `low_price + info.amount` | VECTORS.md defines this mapping |
| Company transfer | Manual flag setting | `Company.transfer_to_corp_acquisition()` | Entity handles location tracking |
| President lookup | Share count iteration | `_get_corp_president()` from Phase 12 | Already implemented |
| Offer validation | Inline checks | `_is_offer_valid()` from Phase 12 | Consistency with presentation logic |

## Common Pitfalls

### Pitfall 1: Negative Cash After Action

**What goes wrong:** Corp cash goes negative because validation missed edge case
**Why it happens:** FI Buy Face checked `cash >= face` but action was FI Buy High
**How to avoid:** Each action type has separate validation function matching its price logic
**Warning signs:** Tests pass individually but fail in integration

### Pitfall 2: Seller Retains Zero Companies (Corp Seller)

**What goes wrong:** Corp sells last company, invalid game state
**Why it happens:** Only checked `owned_companies`, forgot `acquisition_companies`
**How to avoid:** Count both arrays, exclude target company explicitly
**Warning signs:** Seller ends up with 0 companies but test expected 1

### Pitfall 3: FI Action on Non-FI Offer

**What goes wrong:** FI Buy High/Face accepted for corp-to-corp offer
**Why it happens:** Action mask allowed it, handler didn't re-check
**How to avoid:** Defensive check: `if not TURN.is_acq_fi_offer(state): return 1`
**Warning signs:** Company ends up in wrong location

### Pitfall 4: OS Using FI Buy High

**What goes wrong:** OS pays high price instead of face value
**Why it happens:** Action mask allowed both, handler didn't check corp_id
**How to avoid:** FI Buy High validates `corp_id != OS_CORP_ID`
**Warning signs:** OS overpays, FI gets extra cash

### Pitfall 5: Pass Not Advancing Offer

**What goes wrong:** Same offer presented after pass
**Why it happens:** Forgot to increment offer_index
**How to avoid:** `_advance_to_next_offer` increments then presents
**Warning signs:** Infinite loop on same offer

## Code Examples

### Accept Price Action (Non-FI)

```cython
cdef void _handle_accept_price(GameState state, int price) noexcept:
    """Execute price-based acquisition (non-FI offers)."""
    cdef int corp_id = TURN.get_acq_active_corp(state)
    cdef int company_id = TURN.get_acq_target_company(state)
    cdef int seller_corp, seller_player

    # Determine seller from company location
    cdef object company = company_module.COMPANIES[company_id]
    cdef int loc = company.get_location(state)

    # Transfer money buyer -> seller
    corp_module.CORPS[CORP_NAMES[corp_id]].add_cash(state, -price)

    if loc == LOC_CORP:
        seller_corp = company.get_owner_id(state)
        corp_module.CORPS[CORP_NAMES[seller_corp]].add_acquisition_proceeds(state, price)
    elif loc == LOC_PLAYER:
        seller_player = company.get_owner_id(state)
        player_module.PLAYERS[seller_player].add_acquisition_proceeds(state, price)

    # Transfer company to buyer's acquisition zone
    company.transfer_to_corp_acquisition(state, corp_id)

    # Advance to next offer
    _advance_to_next_offer(state)
```

### FI Buy Face Action (OS Only)

```cython
cdef void _handle_fi_buy_face(GameState state) noexcept:
    """Execute FI purchase at face value (OS only)."""
    cdef int corp_id = TURN.get_acq_active_corp(state)
    cdef int company_id = TURN.get_acq_target_company(state)
    cdef int face_value = get_company_face_value(company_id)

    # Transfer money: OS -> FI
    corp_module.CORPS[CORP_NAMES[corp_id]].add_cash(state, -face_value)
    fi_module.FI.add_cash(state, face_value)

    # Transfer company to OS acquisition zone
    company_module.COMPANIES[company_id].transfer_to_corp_acquisition(state, corp_id)

    # Advance to next offer
    _advance_to_next_offer(state)
```

### Validation Example

```cython
cdef bint _validate_price_action(GameState state, int price) noexcept:
    """Validate price-based acquisition is legal."""
    cdef int corp_id = TURN.get_acq_active_corp(state)
    cdef int company_id = TURN.get_acq_target_company(state)
    cdef int low_price = get_company_low_price(company_id)
    cdef int high_price = get_company_high_price(company_id)
    cdef int corp_cash = corp_module.CORPS[CORP_NAMES[corp_id]].get_cash(state)

    # VALID-01: Price in range
    if price < low_price or price > high_price:
        return False

    # VALID-02: Buyer can afford
    if corp_cash < price:
        return False

    # VALID-03: Seller retains >=1 company (corp sellers only)
    # FI (is_acq_fi_offer) has no minimum
    # Players have no minimum
    # Corps need minimum
    if not TURN.is_acq_fi_offer(state):
        cdef object company = company_module.COMPANIES[company_id]
        cdef int loc = company.get_location(state)
        if loc == LOC_CORP:
            seller_corp = company.get_owner_id(state)
            if _count_seller_companies(state, 2, seller_corp) < 1:
                return False

    return True
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Stub handler | Full action handler | Phase 13 | Replace `apply_acquisition_stub` with `apply_acquisition_action` |

## Action Mask Updates

The existing `_fill_acquisition_mask` in `core/actions.pyx` handles:
- Price range (+0 to +50): Valid if `corp_cash >= low_price + offset`
- FI Buy High (+51): Valid if non-OS corp, `corp_cash >= high_price`
- FI Buy Face (+52): Valid if OS corp, `corp_cash >= face_value`
- Pass (+53): Always valid

**Required additions:** None - mask generation is complete from Phase 12.

## Integration Points

| Component | Current State | Phase 13 Change |
|-----------|---------------|-----------------|
| `core/driver.pyx` | Calls `apply_acquisition_stub` | Update to call `apply_acquisition_action` |
| `phases/acquisition.pyx` | Has stub, offer generation | Add action handlers |
| `core/actions.pyx` | Mask generation complete | No changes needed |

## Open Questions

None - all requirements are well-specified in CONTEXT.md.

## Sources

### Primary (HIGH confidence)
- `/home/icebreaker/rss-az-cython2/phases/invest.pyx` - Reference pattern for action handling
- `/home/icebreaker/rss-az-cython2/phases/acquisition.pyx` - Phase 12 offer infrastructure
- `/home/icebreaker/rss-az-cython2/core/actions.pyx` - Action constants, mask generation
- `/home/icebreaker/rss-az-cython2/VECTORS.md` - Action encoding specification
- `/home/icebreaker/rss-az-cython2/.planning/phases/13-actions-validation/13-CONTEXT.md` - User decisions

### Secondary (MEDIUM confidence)
- `/home/icebreaker/rss-az-cython2/entities/*.pyx` - Entity handle APIs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All components exist in codebase
- Architecture: HIGH - Following established INVEST pattern
- Pitfalls: HIGH - Based on similar logic in existing code

**Research date:** 2026-01-25
**Valid until:** Until Phase 13 completion (internal codebase, stable patterns)
