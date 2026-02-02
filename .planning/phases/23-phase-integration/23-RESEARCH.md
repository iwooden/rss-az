# Phase 23: Phase Integration - Research

**Researched:** 2026-02-02
**Domain:** Cython phase integration, game loop transitions, bankruptcy refactoring
**Confidence:** HIGH

## Summary

Phase 23 integrates the INCOME phase (Phase 22) into the game loop with proper transitions and bankruptcy handling. This is primarily a **refactoring and integration task**, not new feature development. All required methods already exist in the entities folder.

The core work involves:
1. Creating a new INCOME phase handler following the non-player phase pattern (WRAP_UP is the reference)
2. Refactoring bankruptcy code from `invest.pyx` into `Corp.go_bankrupt()` method
3. Creating TEMP_END_TURN phase to consolidate end-of-turn bookkeeping
4. Updating phase transitions (CLOSING → INCOME → TEMP_END_TURN → INVEST)
5. Moving roundtrip clear from CLOSING to end of INVEST phase (before WRAP_UP)

**Primary recommendation:** Follow WRAP_UP phase as the architecture template. It's a proven non-player phase pattern with 0 valid actions, deterministic execution, and clear separation of concerns.

## Standard Stack

Phase integration uses the existing Cython architecture with no new dependencies.

### Core Components
| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| Phase handlers | `phases/*.pyx` | Phase execution logic | Existing pattern |
| Entity methods | `entities/*.pyx` | State manipulation | All methods exist |
| GamePhases enum | `core/data.pxd` | Phase constants | PHASE_INCOME=5 defined |
| Phase transitions | `turn_module.TURN.set_phase()` | State machine control | Existing API |

### Required Entity Methods (ALL EXIST)
| Method | Location | Purpose | Verified |
|--------|----------|---------|----------|
| `Corp.calculate_income(state)` | `entities/corp.pyx:314` | Pure income calculation | ✓ HIGH |
| `Corp.apply_income(state, income)` | `entities/corp.pyx:397` | Mutate corp cash | ✓ HIGH |
| `FI.calculate_income(state)` | `entities/fi.pyx:76` | Pure income calculation | ✓ HIGH |
| `FI.apply_income(state, income)` | `entities/fi.pyx:106` | Mutate FI cash | ✓ HIGH |
| `Player.get_income(state)` | `entities/player.pyx:444` | Pure income calculation | ✓ HIGH |
| `Player.add_cash(state, amount)` | `entities/player.pyx:287` | Mutate player cash | ✓ HIGH |
| `Player.clear_roundtrip_tracking(state)` | `entities/player.pyx:412` | Clear per-turn flags | ✓ HIGH |

### Bankruptcy Code Location (REFACTOR TARGET)
| Code | Location | Status | Action |
|------|----------|--------|--------|
| `_execute_bankruptcy()` | `phases/invest.pyx:110-167` | Complete implementation (57 lines) | Refactor to `Corp.go_bankrupt()` |
| `bankrupt_corp()` | `core/state.pyx:606-610` | Incomplete stub | DELETE (confirmed by user) |

**Installation:** None required - this is refactoring existing code.

## Architecture Patterns

### Pattern 1: Non-Player Phase Handler (WRAP_UP Reference)

**What:** Deterministic phase with 0 valid actions that auto-executes

**When to use:** Any phase with no player decisions (INCOME, future END_CARD, etc.)

**Example:**
```cython
# Source: phases/wrap_up.pyx:158-186
cdef int apply_wrap_up(GameState state) noexcept:
    """
    Execute WRAP_UP phase logic.

    This is a deterministic non-player phase with 0 actions.
    Steps:
    1. Reorder players by descending cash
    2. Clear consecutive passes
    3. FI purchases companies
    4. Make revealed companies available
    5. Set up acquisition phase
    6. Transition to ACQUISITION

    Returns: 0 always (deterministic, no failure modes)
    """
    _reorder_players_by_cash(state)
    turn_module.TURN.clear_consecutive_passes(state)

    _process_fi_purchases(state)
    _make_all_revealed_available(state)

    acquisition_module.setup_acquisition_phase(state)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
    return 0
```

**Key characteristics:**
- `cdef int apply_PHASE(GameState state) noexcept` signature
- Returns 0 always (no failure modes)
- No ActionInfo parameter (contrast with player phases)
- Calls helper functions (prefixed with `_`)
- Uses entity methods, not inline state manipulation
- Single entry point from game loop

### Pattern 2: Entity Method Delegation

**What:** Phase handlers call entity methods, don't manipulate state directly

**When to use:** Always - maintains separation of concerns

**Example:**
```cython
# GOOD: Use entity methods (existing pattern from all phases)
corp_module.CORPS[corp_id].calculate_income(state)
corp_module.CORPS[corp_id].apply_income(state, income)

# BAD: Direct state manipulation (anti-pattern)
state._data[corp_cash_offset] += income  # NEVER DO THIS
```

**Why:**
- Entity methods encapsulate normalization/denormalization
- Single source of truth for state access
- Easier to test and verify
- Consistent with all existing phases

### Pattern 3: Bankruptcy Refactoring (From Inline to Method)

**What:** Move inline helper to entity method for reuse

**Current state (invest.pyx:110-167):**
```cython
cdef void _execute_bankruptcy(GameState state, int corp_id) noexcept:
    """Execute bankruptcy procedure for a corporation (INV-22 through INV-27)."""
    # ... 57 lines of implementation
```

**Target state (entities/corp.pyx):**
```cython
cpdef void go_bankrupt(self, GameState state):
    """
    Execute bankruptcy procedure (RULES.md lines 378-385).

    Steps:
    1. Remove all owned companies from game
    2. Return all shares to unissued (clear player shares)
    3. Return money to bank (set cash to 0)
    4. Free market space
    5. Deactivate corp and reset state
    """
    # Move implementation from invest.pyx here
```

**Migration:**
```cython
# OLD (invest.pyx)
_execute_bankruptcy(state, corp_id)

# NEW (both invest.pyx and income.pyx)
corp_module.CORPS[corp_id].go_bankrupt(state)
```

### Pattern 4: Phase Transition Refactoring

**Current (temporary):**
```
CLOSING → INVEST (direct)
  - Increments turn in CLOSING
  - Clears roundtrip in CLOSING
```

**Target (Phase 23):**
```
CLOSING → INCOME → TEMP_END_TURN → INVEST
  - Roundtrip clear at end of INVEST (before WRAP_UP)
  - Turn increment in TEMP_END_TURN
```

**Rationale (from user decisions):**
- Roundtrip info only relevant in INVEST phase
- Clearing elsewhere pollutes state vector for model
- TEMP_END_TURN consolidates bookkeeping for future phases

### Recommended Project Structure
```
phases/
├── income.pyx          # NEW: INCOME phase handler
├── temp_end_turn.pyx   # NEW: Temporary end-of-turn bookkeeping
├── invest.pyx          # MODIFY: Add roundtrip clear before WRAP_UP transition
├── closing.pyx         # MODIFY: Remove turn increment/roundtrip clear
└── wrap_up.pyx         # UNCHANGED: Reference pattern

entities/
└── corp.pyx            # MODIFY: Add go_bankrupt() method

core/
└── state.pyx           # MODIFY: Delete bankrupt_corp() stub (line 606-610)
```

## Don't Hand-Roll

Problems that have existing solutions in this codebase:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Income calculation | Inline sum of company income - CoO | `Corp.calculate_income()`, `FI.calculate_income()` | Already handles CoO, synergies, special abilities (22-02) |
| Income application | Direct state manipulation | `Corp.apply_income()`, `FI.apply_income()`, `Player.add_cash()` | Encapsulates normalization, maintains invariants |
| Bankruptcy procedure | Copy-paste from invest.pyx | Refactor to `Corp.go_bankrupt()`, call from both places | Single source of truth prevents divergence |
| Phase transitions | Custom state machine | `turn_module.TURN.set_phase(state, GamePhases.PHASE_X)` | Used by all phases, handles state updates |
| Roundtrip clearing | Manual loop over arrays | `Player.clear_roundtrip_tracking(state)` | Existing method used by CLOSING, ACQUISITION |

**Key insight:** Phase 22 already implemented all income logic. Phase 23 is pure integration - no new calculations.

## Common Pitfalls

### Pitfall 1: Duplicate Income Calculation

**What goes wrong:** Writing new income calculation code instead of using Phase 22 methods

**Why it happens:** Unfamiliarity with existing entity methods, instinct to write inline code

**How to avoid:**
- Read entities/corp.pyx:314-391 (calculate_income implementation)
- Read entities/fi.pyx:76-100 (calculate_income implementation)
- Call methods, don't duplicate logic

**Warning signs:**
- Code iterating companies and computing income inline
- Direct state access to company income/stars offsets
- CoO calculation in phase handler

### Pitfall 2: Wrong Roundtrip Clear Location

**What goes wrong:** Clearing roundtrip at turn end (in TEMP_END_TURN or INCOME)

**Why it happens:** "End of turn" sounds like the right place

**How to avoid:**
- User explicitly stated: "Roundtrip clear should happen at end of INVEST phase (before WRAP_UP), NOT at turn end"
- Reason: Roundtrip info only relevant in INVEST phase
- Clearing elsewhere pollutes state vector for model

**Warning signs:**
- `clear_roundtrip_tracking()` called in TEMP_END_TURN
- `clear_roundtrip_tracking()` called in INCOME

**Correct location:**
```cython
# phases/invest.pyx - in apply_invest_action() at PHASE_WRAP_UP transition
if turn_module.TURN.get_consecutive_passes(state) >= state._num_players:
    # Clear roundtrip BEFORE transitioning to WRAP_UP
    for i in range(state._num_players):
        player_module.PLAYERS[i].clear_roundtrip_tracking(state)
    turn_module.TURN.set_phase(state, PHASE_WRAP_UP)
```

### Pitfall 3: Player Negative Cash After Income

**What goes wrong:** Player ends with negative cash balance after income application

**Why it happens:** Player has negative income but mandatory close doesn't catch them

**How to avoid:**
- Mandatory close runs in CLOSING phase (already implemented)
- Add assertion in INCOME phase: `assert player_cash >= 0` after income application
- Per user decision: "Balance of $0 is allowed, negative is not"

**Warning signs:**
- Missing assertion after `player.add_cash(state, income)`
- Test that applies large negative income without mandatory close

**Code pattern:**
```cython
# After applying income to player
income = player.get_income(state)
player.add_cash(state, income)
# CRITICAL: Verify no negative cash (mandatory close should have prevented this)
assert player.get_cash(state) >= 0, "Player has negative cash after income - mandatory close failed"
```

### Pitfall 4: Corporation Bankruptcy Order Matters

**What goes wrong:** Assuming all bankruptcies can be processed in a batch

**Why it happens:** Looks like independent operations

**How to avoid:**
- Per user decision: "Check bankruptcy immediately per-corp after income application (before next entity)"
- Multiple corps can go bankrupt in same INCOME phase
- Must handle each in sequence (check after each apply_income)

**Warning signs:**
- Collecting list of bankrupt corps, then processing batch
- Checking bankruptcy only at end of phase

**Correct pattern:**
```cython
# For each corporation
for corp_id in range(GameConstants.NUM_CORPS):
    if not corp.is_active(state):
        continue

    income = corp.calculate_income(state)
    corp.apply_income(state, income)

    # CRITICAL: Check bankruptcy IMMEDIATELY after income application
    if corp.get_cash(state) < 0:
        corp.go_bankrupt(state)
        # Corp is now inactive, won't process again
```

### Pitfall 5: Deleting Wrong Bankruptcy Code

**What goes wrong:** Deleting `_execute_bankruptcy()` from invest.pyx instead of moving it

**Why it happens:** Misunderstanding "refactor" vs "delete"

**How to avoid:**
- User decision: "Refactor TO entities/corp.pyx as Corporation.go_bankrupt(state) method"
- MOVE the implementation, don't delete
- Update invest.pyx to call new method
- DELETE only the stub in core/state.pyx (lines 606-610)

**Verification:**
- `grep -n "_execute_bankruptcy" phases/invest.pyx` should show NO definition (only calls)
- `grep -n "go_bankrupt" entities/corp.pyx` should show the method definition
- `grep -n "bankrupt_corp" core/state.pyx` should show NOTHING (stub deleted)

## Code Examples

Verified patterns from existing codebase:

### Non-Player Phase Entry Point
```cython
# Source: phases/wrap_up.pyx:158-186
cdef int apply_wrap_up(GameState state) noexcept:
    """
    Execute WRAP_UP phase logic.

    This is a deterministic non-player phase with 0 actions.
    Returns: 0 always (deterministic, no failure modes)
    """
    _reorder_players_by_cash(state)
    turn_module.TURN.clear_consecutive_passes(state)

    _process_fi_purchases(state)
    _make_all_revealed_available(state)

    acquisition_module.setup_acquisition_phase(state)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
    return 0

def apply_wrap_up_py(GameState state):
    """Python wrapper for testing."""
    return apply_wrap_up(state)
```

### Entity Iteration with Bankruptcy Check
```cython
# Pattern: Process entities in order, check bankruptcy per-entity
cdef void _apply_income_to_corps(GameState state) noexcept:
    """Apply income to all active corporations, handling bankruptcy."""
    cdef int corp_id, income

    for corp_id in range(GameConstants.NUM_CORPS):
        corp = corp_module.CORPS[corp_id]

        if not corp.is_active(state):
            continue

        # Calculate and apply income
        income = corp.calculate_income(state)
        corp.apply_income(state, income)

        # Check bankruptcy immediately after application
        if corp.get_cash(state) < 0:
            corp.go_bankrupt(state)
```

### Player Income Application with Assertion
```cython
# Pattern: Apply player income with negative cash assertion
cdef void _apply_income_to_players(GameState state) noexcept:
    """Apply income to all players."""
    cdef int player_id, income, final_cash

    for player_id in range(state._num_players):
        player = player_module.PLAYERS[player_id]

        income = player.get_income(state)
        player.add_cash(state, income)

        # Verify no negative cash (mandatory close should have prevented)
        final_cash = player.get_cash(state)
        assert final_cash >= 0, f"Player {player_id} has negative cash ${final_cash} after income"
```

### Bankruptcy Method (Refactored from invest.pyx)
```cython
# Source: phases/invest.pyx:110-167 (to be moved to entities/corp.pyx)
cpdef void go_bankrupt(self, GameState state):
    """
    Execute bankruptcy procedure for this corporation (RULES.md lines 378-385).

    Triggered when corporation cannot pay negative income. This is a complete reset:
    - All owned companies removed from game
    - All shares returned to unissued (cleared from players)
    - Corp cash returned to bank (set to 0)
    - Market space freed
    - Corp deactivated and available for future IPO
    """
    cdef int company_id, player_id, current_index

    # Step 1: Remove all owned companies from game
    for company_id in range(GameConstants.NUM_COMPANIES):
        if self.owns_company(state, company_id):
            company_module.COMPANIES[company_id].remove_from_game(state)
            self.set_owns_company(state, company_id, False)

    # Step 2: Return all shares to unissued - clear player shares first
    for player_id in range(state._num_players):
        player_module.PLAYERS[player_id].set_shares(state, self.corp_id, 0)
        player_module.PLAYERS[player_id].set_president_of(state, self.corp_id, False)

    # Step 3: Reset corp share counts
    self.set_unissued_shares(state, get_corp_share_count(self.corp_id))
    self.set_issued_shares(state, 0)
    self.set_bank_shares(state, 0)

    # Step 4: Return money to bank - clear corp cash
    self.set_cash(state, 0)

    # Step 5: Free market space if needed
    current_index = self.get_price_index(state)
    if current_index > 0:
        market_module.MARKET.set_space_available(state, current_index, True)

    # Step 6: Deactivate corp and clear remaining state
    self.set_active(state, False)
    self.set_price_index(state, 0)
    self.set_in_receivership(state, False)
    self.set_income(state, 0)
    self.set_stars(state, 0)
    self.set_acquisition_proceeds(state, 0)

    # Step 7: Clear acquisition company flags
    for company_id in range(GameConstants.NUM_COMPANIES):
        self.set_acquisition_company(state, company_id, False)
```

### Phase Transition Update
```cython
# phases/closing.pyx:382-407 (BEFORE Phase 23)
cdef void _transition_to_income(GameState state) noexcept:
    """Complete CLOSING phase and transition to INCOME."""
    # ... terminal state check ...

    # Increment turn number
    turn_module.TURN.set_turn_number(state, current_turn + 1)

    # Clear per-turn tracking for all players
    for i in range(state._num_players):
        player_module.PLAYERS[i].clear_roundtrip_tracking(state)

    # Transition to INCOME phase (temporary: using INVEST)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)

# phases/closing.pyx (AFTER Phase 23)
cdef void _transition_to_income(GameState state) noexcept:
    """Complete CLOSING phase and transition to INCOME."""
    # ... terminal state check ...

    # Transition to INCOME phase
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INCOME)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CLOSING transitions to INVEST | CLOSING → INCOME → TEMP_END_TURN → INVEST | Phase 23 | Proper phase separation |
| Turn increment in CLOSING | Turn increment in TEMP_END_TURN | Phase 23 | Consolidates end-of-turn logic |
| Roundtrip clear in CLOSING | Roundtrip clear at end of INVEST | Phase 23 | Reduces state vector pollution |
| Inline bankruptcy in invest.pyx | Corp.go_bankrupt() method | Phase 23 | Reusable by INCOME phase |
| bankrupt_corp() stub in state.pyx | Deleted (unused) | Phase 23 | Code cleanup |

**Deprecated/outdated:**
- **CLOSING temporary transition**: Lines 403-406 in closing.pyx have comment "Note: INCOME phase not implemented yet, using INVEST as temporary target" - this becomes obsolete in Phase 23
- **bankrupt_corp() stub**: core/state.pyx:606-610 is incomplete and never used - delete in Phase 23

## Open Questions

Things that couldn't be fully resolved:

1. **TEMP_END_TURN file naming convention**
   - What we know: Should create `phases/temp_end_turn.pyx` (user specified "phase file")
   - What's unclear: User left naming/structure to Claude's discretion
   - Recommendation: Use `temp_end_turn.pyx` (snake_case matches other phase files)

2. **Roundtrip clear exact location**
   - What we know: Must be at end of INVEST phase, before WRAP_UP transition
   - What's unclear: Should it be in the PASS handler (line 344) or in a separate helper?
   - Recommendation: In PASS handler right before `set_phase(PHASE_WRAP_UP)` (line 344) - keeps logic localized

3. **TEMP_END_TURN comment wording**
   - What we know: Should add comments explaining TEMP_END_TURN is temporary
   - What's unclear: Exact wording
   - Recommendation: "Temporary phase for end-of-turn bookkeeping. Will be integrated into final phase when all phases are implemented."

## Sources

### Primary (HIGH confidence)
- **Codebase inspection**: phases/wrap_up.pyx (non-player phase pattern)
- **Codebase inspection**: phases/invest.pyx:110-167 (bankruptcy implementation)
- **Codebase inspection**: entities/corp.pyx:314-400 (income calculation/application)
- **Codebase inspection**: entities/fi.pyx:76-108 (income calculation/application)
- **Codebase inspection**: entities/player.pyx:287-290, 444-463 (income methods)
- **Codebase inspection**: phases/closing.pyx:382-407 (current transition logic)
- **User decisions**: .planning/phases/23-phase-integration/23-CONTEXT.md (all decisions)
- **RULES.md**: Lines 378-385 (bankruptcy procedure)
- **STATE.md**: Lines 36-48 (established patterns)

### Secondary (MEDIUM confidence)
- **Requirements**: .planning/REQUIREMENTS.md (INC-06, TRN-01, TRN-02, TRN-03, TRN-04)

### Tertiary (LOW confidence)
- None - all findings verified against codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all methods exist, verified by file inspection
- Architecture: HIGH - WRAP_UP pattern is proven and complete
- Pitfalls: HIGH - based on user's explicit decisions and codebase patterns
- Bankruptcy refactoring: HIGH - existing code location and structure verified
- Phase transitions: HIGH - all phases and enums exist, pattern clear

**Research date:** 2026-02-02
**Valid until:** 2026-03-04 (30 days - stable refactoring task)
