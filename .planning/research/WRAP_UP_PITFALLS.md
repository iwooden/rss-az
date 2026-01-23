# Domain Pitfalls: WRAP_UP Phase Implementation

**Domain:** Turn-based board game engine with sequential phase execution
**Context:** Adding WRAP_UP phase to existing Cython game engine with float32 state array
**Researched:** 2026-01-22

---

## Critical Pitfalls

Mistakes that cause rewrites, data corruption, or game rule violations.

### Pitfall 1: Player Reordering Tie-Breaking Corruption

**What goes wrong:** When reordering players by descending cash with tie-breaking by old order, incorrect implementation creates non-deterministic behavior or violates tie-breaking rules.

**Why it happens:**
- Stable sort not used (relative order of equal elements changes)
- Sort comparison inverted (ascending vs descending)
- Tie-breaking logic uses wrong field (player_id instead of old turn_order position)
- Two-phase mutation (reading old positions while writing new ones overwrites data)

**Consequences:**
- Non-deterministic games (breaks reproducible training with seeds)
- Rule violation (wrong player goes first after WRAP_UP)
- Cascading errors in all subsequent phases (wrong active player)
- Impossible to debug (appears random, hard to reproduce)

**Prevention:**
```python
# WRONG: Tie-breaking by player_id (arbitrary)
sorted_players = sorted(players, key=lambda p: (-p.cash, p.id))

# WRONG: Unstable sort (Python's sort is stable, but explicit is better)
sorted_players = sorted(players, key=lambda p: -p.cash)
players = sorted(players, key=lambda p: p.old_order if p.cash == tie_cash else 0)

# RIGHT: Single-pass stable sort with correct tie-breaking
# Step 1: Capture old state BEFORE any mutations
old_positions = [(player_id, PLAYERS[player_id].get_cash(state),
                  PLAYERS[player_id].get_turn_order(state))
                 for player_id in range(num_players)]

# Step 2: Sort by (-cash, old_position) in one pass
old_positions.sort(key=lambda x: (-x[1], x[2]))

# Step 3: Apply new positions (separate phase)
for new_position, (player_id, _, _) in enumerate(old_positions):
    PLAYERS[player_id].set_turn_order(state, new_position)
```

**Detection:**
- Test with players having equal cash, verify tie broken by old order
- Test with all players having equal cash (should preserve exact order)
- Seed-based reproducibility test (run same game twice, verify identical)
- Property test: old_order[i] < old_order[j] AND cash[i] == cash[j] => new_order[i] < new_order[j]

**Phase assignment:** Phase 1 (WRAP_UP Core Logic)

---

### Pitfall 2: Foreign Investor Purchase Loop State Corruption

**What goes wrong:** During FI's company purchase loop, state modifications from one iteration corrupt the next iteration's queries.

**Why it happens:**
- Company availability state modified mid-loop (marking newly-drawn company unavailable)
- Iterating over dynamic collection that changes during iteration
- Face value order calculation includes companies not yet drawn
- Entity handle cache invalidation (company location changes, handle still points to old location)

**Consequences:**
- FI skips companies it should buy
- FI attempts to buy unavailable companies
- Crash from accessing invalid state
- Companies marked unavailable but not owned by anyone (orphaned state)

**Prevention:**
```python
# WRONG: Query state inside loop that modifies it
for company_id in get_available_companies(state):
    if FI.can_afford(company_id):
        FI.buy_company(state, company_id)  # Modifies availability!
        draw_new_company(state)             # Changes what's "available"!

# RIGHT: Re-query each iteration (always operates on current state)
while True:
    # Step 1: Query current available companies
    available = get_available_companies_sorted_by_face_value(state)
    if not available:
        break

    # Step 2: Find cheapest affordable
    company_id = None
    for cid in available:
        if FI.get_cash(state) >= get_company_face_value(cid):
            company_id = cid
            break

    if company_id is None:
        break  # Can't afford any

    # Step 3: Execute atomic purchase
    execute_fi_purchase(state, company_id)  # All modifications here

    # Step 4: Loop continues, re-queries on next iteration

# Atomic purchase function
def execute_fi_purchase(state, company_id):
    face_value = get_company_face_value(company_id)

    # Transfer money
    FI.add_cash(state, -face_value)

    # Transfer company
    COMPANIES[company_id].set_location(state, LOC_FI, -1)
    COMPANIES[company_id].set_for_auction(state, False)
    FI.set_owns_company(state, company_id, True)

    # Draw new company (if deck not empty)
    if not DECK.is_empty(state):
        new_company_id = DECK.draw_top(state)
        COMPANIES[new_company_id].set_revealed(state, True)
        COMPANIES[new_company_id].set_for_auction(state, False)  # Unavailable
```

**Detection:**
- Log each FI purchase: company_id, face_value, FI cash before/after
- Assert FI cash decreases monotonically
- Assert each purchased company owned by FI at end
- Count drawn companies == count purchased companies
- Invariant: No company is both for_auction=True and owned_by_fi=True

**Phase assignment:** Phase 2 (FI Purchase Logic)

---

### Pitfall 3: Company Availability State Confusion

**What goes wrong:** WRAP_UP phase flips company availability states (unavailable → available) but loses track of which companies should flip.

**Why it happens:**
- Confusing "revealed" vs "for_auction" vs "unavailable_this_phase" flags
- No clear state machine for company lifecycle
- Rules say "mark unavailable (turn vertical)" but implementation has no corresponding flag
- Attempting to infer state from multiple flags instead of explicit tracking

**Consequences:**
- Companies available for auction when they shouldn't be
- Newly-drawn companies immediately auctionable (violates rules)
- FI purchases companies that should be unavailable
- Phase transition leaves companies in limbo state

**Prevention:**
```python
# Existing state has: revealed, for_auction, removed
# Need to track: "drawn_this_turn" or "unavailable_until_wrap_up"

# Option 1: Add explicit unavailable flag (RECOMMENDED)
# In state layout, add:
#   unavailable_companies: [NUM_COMPANIES]  # Binary flags

def mark_company_unavailable_until_wrap_up(state, company_id):
    COMPANIES[company_id].set_unavailable(state, True)
    COMPANIES[company_id].set_for_auction(state, False)

def wrap_up_make_all_available(state):
    for company_id in range(NUM_COMPANIES):
        if COMPANIES[company_id].is_unavailable(state):
            COMPANIES[company_id].set_unavailable(state, False)
            COMPANIES[company_id].set_for_auction(state, True)

# Option 2: Use revealed + !for_auction as "unavailable" (FRAGILE)
# Only if state array size is constrained
# Must document clearly and enforce invariant

# WRONG: Trying to track in turn state
# turn.unavailable_companies_list = []  # Python list in Cython state!

# WRONG: Implicit tracking
# "If revealed and not for_auction and not owned, then unavailable"
# Too many conditions, easy to break
```

**Detection:**
- Invariant check: revealed AND !for_auction AND !owned => unavailable
- Test: Draw company, verify for_auction=False until WRAP_UP
- Test: After WRAP_UP, verify all unavailable companies become for_auction=True
- Count test: Track len(for_auction) before/after each phase step

**Phase assignment:** Phase 2 (FI Purchase Logic) — must resolve before implementing

---

### Pitfall 4: Phase Transition Timing with Auto-Apply

**What goes wrong:** WRAP_UP phase ends but auto-apply loop doesn't properly transition to next phase, causing stuck state or wrong action set.

**Why it happens:**
- Phase set to WRAP_UP but action handlers not implemented
- WRAP_UP phase has sub-states (reordering, FI buying, availability flip) but no clear "done" condition
- Auto-apply loop expects get_legal_moves() to work but WRAP_UP actions not in action enum
- Transition to Phase 3 (Acquisition) happens mid-WRAP_UP

**Consequences:**
- Game stuck: auto-apply finds 0 legal actions, throws ZeroLegalActionsError
- Wrong phase: advances to Phase 3 before FI purchases complete
- Action set mismatch: legal moves returned for wrong phase
- History contamination: auto-apply records sub-steps that aren't real actions

**Prevention:**
```python
# WRAP_UP is fully deterministic - no player actions
# Therefore: execute entire phase in one atomic operation

def apply_invest_action(state, action_info):
    # ... handle pass/auction/buy/sell ...

    # Check if INVEST phase should end
    if turn_module.TURN.get_consecutive_passes(state) >= state._num_players:
        # All players passed - enter WRAP_UP
        execute_wrap_up_phase(state)
        # WRAP_UP completes atomically, transitions to Phase 3
        turn_module.TURN.set_phase(state, PHASE_ACQUISITION)
        # Initialize Phase 3 state (any setup needed)
        initialize_acquisition_phase(state)

def execute_wrap_up_phase(state):
    """
    Execute entire WRAP_UP phase atomically.
    No actions, no forced-action iterations - just rule execution.
    """
    # Step 1: Reorder players
    reorder_players_by_cash_desc(state)

    # Step 2: FI purchases
    execute_all_fi_purchases(state)

    # Step 3: Flip availability
    make_all_unavailable_companies_available(state)

    # No phase change here - caller handles transition
```

**Detection:**
- Test: Trigger WRAP_UP, verify phase advances to PHASE_ACQUISITION
- Test: After last INVEST pass, verify no actions required (direct to PHASE_ACQUISITION)
- Test: During WRAP_UP, verify get_legal_moves() doesn't crash
- Integration test: Full turn through INVEST → WRAP_UP → ACQUISITION

**Phase assignment:** Phase 1 (WRAP_UP Core Logic) — architectural decision needed first

---

## Moderate Pitfalls

Mistakes that cause delays, technical debt, or hard-to-debug issues.

### Pitfall 5: Forced Action Interaction Misunderstanding

**What goes wrong:** Assuming WRAP_UP phase will auto-apply like other forced actions, but it's actually a rule-driven transition without actions.

**Why it happens:**
- Misreading rules: "Wrap-up" sounds like a phase with actions
- Existing pattern: other phases have actions, so WRAP_UP should too
- Confusing "Foreign Investor buys" with player action (it's deterministic rule execution)

**Consequences:**
- Over-engineering: implementing action encoding for WRAP_UP when none needed
- Testing complexity: trying to test forced actions that don't exist
- Performance overhead: unnecessary action mask computation

**Prevention:**
- Read rules carefully: "Determine new Player Order" is not a player action
- FI purchases are deterministic (ascending face value, buy all affordable) - no choices
- Compare to game implementation: physical game has no cards/actions for WRAP_UP
- Implementation pattern: Execute WRAP_UP as atomic procedure, not action loop

**Detection:**
- Code review: Look for ACTION_WRAP_UP_* enum values (shouldn't exist)
- Test coverage: No tests for "legal moves during WRAP_UP" (phase is atomic)
- Performance benchmark: WRAP_UP should be single operation, not iteration

**Phase assignment:** Phase 1 (WRAP_UP Core Logic) — clarify during planning

---

### Pitfall 6: Entity Handle Offset Invalidation

**What goes wrong:** Player reordering changes turn_order fields, but code caches old positions and reads stale data.

**Why it happens:**
- Entity handle pattern caches offsets, assumes state layout doesn't change
- Player reordering changes which player is at position 0, but code expects old player
- Active player pointer not updated after reordering
- Auction state (starter, high bidder) references old player positions

**Consequences:**
- Wrong active player in Phase 3 (Acquisition)
- Auction state orphaned (starter no longer at expected position)
- find_player_at_position() returns stale results

**Prevention:**
```python
# Player entity handles cache _base_offset (player's data block)
# They do NOT cache turn_order position (it's dynamic)

# After reordering:
# 1. Player entity handles still point to same player data - GOOD
# 2. Turn order query re-reads one-hot encoded position - GOOD
# 3. Active player must be updated to new position 0 - ACTION REQUIRED

def reorder_players_by_cash_desc(state):
    # ... reorder logic ...

    # Find player now at position 0
    new_first_player = turn_module.TURN.find_player_at_position(state, 0)
    state._set_active_player(new_first_player)

    # Clear any phase-specific state that references old positions
    # (e.g., auction state should be cleared by INVEST phase end already)
```

**Detection:**
- Test: After WRAP_UP, verify get_active_player() returns player at position 0
- Test: After reordering, verify get_turn_order(player_id) matches new cash rank
- Invariant: state._active_player == find_player_at_position(state, current_position)

**Phase assignment:** Phase 1 (WRAP_UP Core Logic)

---

### Pitfall 7: FI Cash Underflow

**What goes wrong:** FI purchases more companies than it can afford, ending with negative cash.

**Why it happens:**
- Off-by-one in affordability check (<= vs <)
- Forgetting FI has income from Phase 5 but purchases in Phase 2
- Incorrect Face Value lookup (using wrong company_id)

**Consequences:**
- Negative cash (invalid state)
- FI bankruptcy (not a game mechanic)
- Game rule violation (FI can't go into debt)

**Prevention:**
```python
def execute_all_fi_purchases(state):
    while True:
        available = get_available_companies_sorted_by_face_value(state)
        if not available:
            break

        fi_cash = FI.get_cash(state)
        company_id = None

        for cid in available:
            face_value = get_company_face_value(cid)
            if fi_cash >= face_value:  # >= not >, FI pays exact face value
                company_id = cid
                break

        if company_id is None:
            break  # No affordable companies

        execute_fi_purchase(state, company_id)

        # Post-condition check
        assert FI.get_cash(state) >= 0, "FI cash underflow!"
```

**Detection:**
- Invariant: FI.get_cash(state) >= 0 after every purchase
- Test: FI with 10 cash, companies with face values [3, 3, 3, 3] → buys 3, stops at 1 cash
- Test: FI with 5 cash, companies [6, 7] → buys 0 (can't afford any)

**Phase assignment:** Phase 2 (FI Purchase Logic)

---

### Pitfall 8: Determinism Violation from Sorting Instability

**What goes wrong:** Python's sort is stable, but Cython's C-level sort might not be, causing non-deterministic reordering.

**Why it happens:**
- Using C qsort() which is not guaranteed stable
- Relying on dict/set ordering (guaranteed in Python 3.7+, but not in C)
- Parallel access patterns if future optimization adds threading

**Consequences:**
- Seed-based reproducibility broken
- Training data non-deterministic
- Impossible to reproduce bugs

**Prevention:**
- Use Python's sorted() which guarantees stable sort
- If performance-critical, use NumPy's stable sort: np.argsort(kind='stable')
- Document sort stability requirement in comments
- Add reproducibility test (same seed → same outcome)

```python
# WRONG: C-level sort (not stable)
cdef void sort_players_by_cash(PlayerData* players, int n):
    qsort(players, n, sizeof(PlayerData), compare_by_cash)

# RIGHT: Python stable sort
def reorder_players_by_cash_desc(state):
    players = [(i, PLAYERS[i].get_cash(state), PLAYERS[i].get_turn_order(state))
               for i in range(state._num_players)]
    players.sort(key=lambda x: (-x[1], x[2]))  # Stable sort guaranteed
    for new_pos, (pid, _, _) in enumerate(players):
        PLAYERS[pid].set_turn_order(state, new_pos)
```

**Detection:**
- Run same game with same seed 100 times, verify identical outcomes
- Automated CI test: seed-based reproducibility for all tests
- Log player order after WRAP_UP, verify deterministic

**Phase assignment:** Phase 1 (WRAP_UP Core Logic)

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable.

### Pitfall 9: Off-By-One in Face Value Ordering

**What goes wrong:** FI purchase order is wrong because face values sorted descending instead of ascending.

**Why it happens:**
- Misreading rules: "ascending Face Value order"
- Copy-paste from player reordering (which is descending cash)
- Confusing "buy cheapest first" with "buy highest value first"

**Consequences:**
- FI buys expensive companies first, runs out of cash faster
- Game rule violation
- Tests fail with wrong purchase order

**Prevention:**
```python
# Rules: "In ascending Face Value order, Foreign Investor buys..."
# Ascending = lowest to highest = cheapest first

available.sort(key=lambda cid: get_company_face_value(cid))  # Ascending

# NOT:
# available.sort(key=lambda cid: -get_company_face_value(cid))  # Descending - WRONG!
```

**Detection:**
- Test: FI with 50 cash, companies [5, 10, 20, 30] → purchases in order [5, 10, 20]
- Visual test: Log purchase order, verify ascending face values

**Phase assignment:** Phase 2 (FI Purchase Logic)

---

### Pitfall 10: Forgetting to Clear Auction State

**What goes wrong:** After INVEST phase ends (all players passed), auction state (high bidder, starter, passed flags) not cleared, leaking into next turn.

**Why it happens:**
- Assuming auction state auto-clears
- Phase transition doesn't explicitly reset all phase-specific state
- Auction end handler clears some fields but not all

**Consequences:**
- Next turn's INVEST phase sees stale auction data
- Player marked as "passed in auction" when no auction active
- get_legal_moves() incorrectly filters actions

**Prevention:**
```python
def execute_wrap_up_phase(state):
    # Clear INVEST phase state
    turn_module.TURN.set_consecutive_passes(state, 0)

    # Clear auction state (should already be done, but defensive)
    turn_module.TURN.clear_auction_state(state)

    # ... rest of WRAP_UP ...
```

**Detection:**
- Invariant check at phase boundaries: assert auction state is clear when phase != BID_IN_AUCTION
- Test: End INVEST phase, verify all auction flags = 0

**Phase assignment:** Phase 1 (WRAP_UP Core Logic)

---

### Pitfall 11: Missing Net Worth Update After Reordering

**What goes wrong:** Player net worth not recalculated after WRAP_UP, so next turn's reordering uses stale values.

**Why it happens:**
- Assuming net worth only changes during buy/sell
- Forgetting company value changes affect net worth
- FI purchases change company availability, indirectly affecting player values

**Consequences:**
- Next WRAP_UP uses incorrect cash values for reordering
- Net worth display incorrect for UI/logging

**Prevention:**
```python
def execute_wrap_up_phase(state):
    # Step 1: Reorder players (uses current cash, not net worth)
    reorder_players_by_cash_desc(state)

    # Step 2: FI purchases
    execute_all_fi_purchases(state)

    # Step 3: Update net worth (defensive, may not be needed)
    player_module.update_all_net_worths(state)

    # Step 4: Flip availability
    make_all_unavailable_companies_available(state)
```

**Detection:**
- Test: Verify get_net_worth() accurate after WRAP_UP
- Invariant: Net worth includes cash + company values + share values

**Phase assignment:** Phase 1 (WRAP_UP Core Logic) — verify if needed

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|----------------|------------|
| Player reordering | Tie-breaking uses player_id instead of old turn_order | Test with equal cash, verify order preserved |
| FI purchase loop | State corruption from mid-loop modifications | Snapshot iteration list, validate each step |
| Company availability | Confusing revealed/for_auction/unavailable states | Add explicit unavailable flag or document invariant |
| Phase transition | WRAP_UP has no actions, must execute atomically | Execute entire phase in one function call |
| Forced action interaction | Assuming WRAP_UP auto-applies actions | Clarify WRAP_UP is procedural, not action-based |
| Entity handles | Cached positions after reordering | Update active_player after reordering |
| Determinism | Unstable sort breaks seed reproducibility | Use Python's stable sort |

---

## Testing Strategy

### Must-Have Tests

1. **Player Reordering:**
   - All equal cash → preserve exact order
   - Descending cash with unique values → correct order
   - Mixed (some equal, some unique) → tie-breaking correct
   - Seed reproducibility (same input → same output)

2. **FI Purchases:**
   - FI cash insufficient for any company → buys 0
   - FI cash sufficient for some companies → buys in ascending FV order
   - FI cash exactly equals face value → buys that company
   - FI cash underflow test → assert never negative

3. **Company Availability:**
   - Companies drawn during auction → unavailable until WRAP_UP
   - After WRAP_UP → all unavailable become available
   - Count test: len(available) before/after correct

4. **Phase Transition:**
   - INVEST ends (all pass) → WRAP_UP executes → ACQUISITION starts
   - Active player after WRAP_UP → player at position 0

5. **Integration:**
   - Full turn: INVEST → WRAP_UP → ACQUISITION → ... → next turn's INVEST
   - Multi-turn: Verify player order changes persist across turns

### Edge Cases

- 2-player game (minimal reordering)
- 6-player game (maximum reordering)
- All players have same cash for multiple turns
- FI has 0 cash (buys nothing)
- No companies available for FI (deck empty)

---

## Sources

**Primary sources:**
- RULES.md (official game rules, lines 132-138)
- PROJECT.md (v2.1 implementation patterns)
- core/driver.pyx (auto-apply loop pattern)
- phases/invest.pyx (existing phase handler patterns)
- entities/player.pyx (turn_order management)
- entities/turn.pyx (find_player_at_position pattern)

**Confidence:** HIGH
- Rules specification is authoritative
- Existing codebase patterns well-established
- Entity handle pattern documented in PROJECT.md
- Auto-apply behavior tested in v2.1

**Knowledge gaps:**
- Whether company availability needs new state flag or can reuse existing flags
- Whether WRAP_UP is atomic operation or has sub-actions (rules strongly suggest atomic)
- Performance requirements for player sorting (Python vs Cython implementation)

**Recommendations:**
1. Add explicit unavailable_companies flag to state layout (cleaner than inferring from multiple flags)
2. Implement WRAP_UP as atomic procedure, not action-based phase
3. Use Python's stable sort for determinism
4. Add seed-based reproducibility test to CI
