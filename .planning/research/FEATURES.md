# Feature Landscape: WRAP_UP Phase

**Domain:** Board game engine - end-of-turn phase (WRAP_UP)
**Researched:** 2026-01-22
**Confidence:** HIGH (based on official RULES.md)

## Executive Summary

The WRAP_UP phase is a deterministic end-of-turn phase with four sequential sub-features. It has no player choices—all actions are forced by game rules. The phase exists to reset state between INVEST rounds: reordering players by cash, redistributing turn order cards, and executing Foreign Investor purchases.

**Critical insight:** WRAP_UP is entirely deterministic. There are no player decisions, no branching paths. This is fundamentally different from INVEST/BID_IN_AUCTION which have rich decision spaces.

## Table Stakes

Features required for WRAP_UP phase to be minimally functional per official rules.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **TS-01: Player order recalculation** | RULES.md line 134: "Determine new Player Order by descending remaining money (ties broken by old player order)" | Medium | Must handle tie-breaking correctly. Descending order (richest first). Preserve old order for ties. Update each player's turn_order field (one-hot encoded). |
| **TS-02: Turn order card redistribution** | RULES.md line 135: "Redistribute player order cards" | Low | Conceptual operation (physical game). In our engine, this is already done by TS-01 when we update player turn_order state. No separate implementation needed. |
| **TS-03: Foreign Investor purchases** | RULES.md lines 136-137: "In ascending Face Value order, Foreign Investor buys as many available companies as possible at Face Value" | High | Multi-step: (1) Identify available companies, (2) Sort by face value ascending, (3) For each: check if FI can afford at face value, (4) Purchase: FI pays face value to bank, (5) Transfer company to FI ownership, (6) Draw new card and mark unavailable, (7) Repeat until FI cannot afford next or no companies remain. |
| **TS-04: Unavailable companies become available** | RULES.md line 138: "After Foreign Investor done, all unavailable companies become available (turn horizontal)" | Low | Flip all `revealed_companies` flags to `auction_companies` flags. Simple state transformation. |

### Dependencies on Existing Infrastructure

| Feature | Depends On | Status |
|---------|------------|--------|
| TS-01 | Player.get_cash(), Player.get_turn_order(), Player.set_turn_order() | ✓ EXISTS (entities/player.pyx) |
| TS-03 | FI.get_cash(), FI.add_cash(), FI.set_owns_company() | ✓ EXISTS (entities/fi.pyx) |
| TS-03 | Company.is_for_auction(), Company.get_face_value(), Company.transfer_to_fi() | ✓ EXISTS (entities/company.pyx) |
| TS-03 | DECK.draw(), Company.set_revealed() | ✓ EXISTS (entities/deck.pyx, entities/company.pyx) |
| TS-04 | Company.is_revealed(), Company.move_to_auction() | ✓ EXISTS (entities/company.pyx) |

## Edge Cases

Important but secondary features that handle boundary conditions.

| Feature | Scenario | Handling | Complexity |
|---------|----------|----------|------------|
| **EC-01: FI cannot afford any companies** | All available companies have face value > FI cash | Skip purchase loop, proceed to TS-04 | Low |
| **EC-02: No available companies** | All companies in deck/ownership, none for auction | Skip purchase loop, proceed to TS-04 | Low |
| **EC-03: Tie-breaking in player order** | Multiple players with same cash | Use old player order (lower old position wins tie) | Medium |
| **EC-04: All players tied at 0 cash** | Edge case in degenerate game states | Old order preserved (all have 0, so descending sort is stable) | Low |
| **EC-05: Deck exhausted during FI purchases** | FI buys last company in deck, draw returns -1 | No new company revealed, but purchase still completes | Medium |
| **EC-06: FI buys company, exhausts deck, then can't afford next** | Specific sequencing scenario | Normal termination of purchase loop | Low |

### Edge Case Priorities

**Must handle:** EC-01, EC-02, EC-03, EC-05
**Nice to have:** EC-04, EC-06 (natural consequences of table stakes implementation)

## Anti-Features

Features to explicitly NOT build. Common mistakes avoided.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **AF-01: Player choices during WRAP_UP** | WRAP_UP is fully deterministic per rules | Zero legal actions. Phase executes automatically until transition to next phase (ACQUISITION). |
| **AF-02: FI purchases at price spans** | RULES.md line 136 specifies "at Face Value" exactly | Always use face value, never use company price span (low_price to high_price). |
| **AF-03: FI purchases unavailable companies** | RULES.md line 136: "buys as many available companies as possible" | Only purchase from `auction_companies` (available), never from `revealed_companies` (unavailable). |
| **AF-04: Revealed companies stay revealed across turns** | RULES.md line 138 is explicit about "all unavailable companies become available" | Clear all revealed flags at end of WRAP_UP. No persistent revealed state. |
| **AF-05: FI purchases in descending order** | RULES.md line 136: "ascending Face Value order" | Sort by face value ascending (lowest first), NOT descending. |
| **AF-06: FI purchasing logic with interventions** | Acquisition phase rules (Phase 3) allow corp interventions when FI sells | WRAP_UP is FI buying, not selling. No interventions possible. Fully deterministic. |

## Feature Dependencies

```
TS-01: Player Order Recalculation
  ↓
TS-02: Turn Order Card Redistribution (conceptual, handled by TS-01)
  ↓
TS-03: Foreign Investor Purchases
  ↓ (after each purchase, but within same feature)
  Draw new company → mark unavailable
  ↓ (loop until termination condition)
TS-04: Unavailable → Available
  ↓
Transition to PHASE_ACQUISITION
```

**Critical sequencing:**
1. TS-01 must complete before TS-03 (new turn order determines active player for next phase)
2. TS-03 must complete before TS-04 (don't make newly-drawn companies available until FI is done)
3. TS-04 is final action before phase transition

**No parallelization possible:** All features are sequential and stateful.

## MVP Recommendation

For MVP WRAP_UP implementation, prioritize:

1. **TS-01: Player order recalculation** (CRITICAL)
   - Without this, turn order never updates based on cash
   - Foundation for all future turn sequencing

2. **TS-03: Foreign Investor purchases** (CRITICAL)
   - Core economic mechanic that drives company availability
   - Without this, FI never acquires companies, blocking Acquisition phase mechanics

3. **TS-04: Unavailable → Available** (CRITICAL)
   - Without this, newly drawn companies never enter auction pool
   - Would cause deck to drain without replenishing available companies

4. **TS-02: Turn order card redistribution** (AUTOMATIC)
   - Already handled by TS-01 state updates
   - No separate implementation needed

**All table stakes features are MVP.** WRAP_UP is a small, tightly-scoped phase. No features can be deferred.

Defer to post-MVP:
- **None.** Phase is atomic and small.

## Implementation Complexity Assessment

| Feature | Lines of Code (est.) | Risk Level | Rationale |
|---------|---------------------|------------|-----------|
| TS-01 | 40-60 | LOW | Sort players by cash descending, stable sort preserves ties. Set turn_order for each. Well-understood algorithm. |
| TS-02 | 0 | NONE | Conceptual only, handled by TS-01. |
| TS-03 | 80-120 | MEDIUM | Loop with multiple state updates per iteration. Needs careful sequencing: check affordability, transfer company, update cash, draw card, mark unavailable. Edge cases (deck exhaustion, no affordable companies). |
| TS-04 | 20-30 | LOW | Iterate all companies, flip revealed → auction flags. Simple state transformation. |

**Total estimated LOC:** 140-210 lines of Cython

**Risk factors:**
- TS-03 has most state mutations (company ownership, FI cash, deck, availability flags)
- TS-03 loop termination conditions must be correct (affordability, availability)
- EC-05 (deck exhaustion) requires careful handling in TS-03

**Mitigation:**
- Test TS-03 extensively with edge cases
- Separate affordability check from purchase execution
- Unit test deck exhaustion scenario explicitly

## Test Coverage Requirements

### Table Stakes Tests

| Feature | Test Case | Assertion |
|---------|-----------|-----------|
| TS-01 | Player order sorts by cash descending | Richest player has turn_order=0 |
| TS-01 | Ties broken by old order | If P0 and P1 have same cash, old P0 < old P1 → new P0 < new P1 |
| TS-01 | All players have 0 cash | Old order preserved |
| TS-03 | FI purchases single affordable company | FI cash decreases by face_value, FI owns company, new company drawn and unavailable |
| TS-03 | FI purchases multiple companies in face value order | First purchase is lowest face value, second is next lowest, etc. |
| TS-03 | FI stops when cannot afford next | FI owns N companies, N+1th company still available (not purchased) |
| TS-03 | FI skips unaffordable companies | If companies [10, 20, 15] available and FI has 12 cash, buys 10 only (not 15, even though could afford if sorted differently) |
| TS-04 | All revealed companies become available | After WRAP_UP, revealed_companies all 0.0, auction_companies updated |

### Edge Case Tests

| Feature | Test Case | Assertion |
|---------|-----------|-----------|
| EC-01 | FI cannot afford any companies | No purchases, phase completes normally |
| EC-02 | No available companies | No purchases, TS-04 has no effect, phase completes |
| EC-03 | 3-way tie in cash | Stable sort maintains relative old order |
| EC-05 | Deck exhausts during FI purchase | Purchase completes, no new company revealed, revealed_companies flag for that slot unchanged |

### Anti-Feature Tests

| Anti-Feature | Test Case | Assertion |
|--------------|-----------|-----------|
| AF-02 | FI pays face value exactly | FI cash reduction equals face_value, not low_price or high_price |
| AF-03 | FI only buys available companies | Revealed companies are not purchased |
| AF-05 | FI purchases in ascending order | Verify first purchase < second purchase < third purchase by face value |

## Relationship to Existing Features

### INVEST Phase
- **Transition:** INVEST → WRAP_UP when consecutive_passes >= num_players
- **State handoff:** INVEST clears round-trip tracking before WRAP_UP (already implemented)
- **Order dependency:** INVEST uses old turn order, WRAP_UP establishes new turn order

### BID_IN_AUCTION Phase
- **Transition:** BID_IN_AUCTION → WRAP_UP when auction resolves (winner determined)
- **State handoff:** Auction state cleared before WRAP_UP transition
- **Order dependency:** BID_IN_AUCTION uses old turn order, next INVEST uses new turn order from WRAP_UP

### Forced Action Auto-Application (v2.1)
- **Applicability:** WRAP_UP has ZERO legal actions (fully deterministic)
- **Expected behavior:** Auto-apply loop should execute entire WRAP_UP phase without stopping
- **Implementation note:** WRAP_UP phase handler must transition to PHASE_ACQUISITION when complete, triggering auto-apply to continue if ACQUISITION also has forced actions

### Future ACQUISITION Phase
- **Transition:** WRAP_UP → ACQUISITION (automatic)
- **State dependency:** ACQUISITION relies on updated turn order from WRAP_UP
- **Company availability:** ACQUISITION expects available companies to be refreshed by TS-04

## Sources

- **RULES.md** (lines 132-138) - Official game rules for Phase 2: Wrap-up
- **core/data.pxd** (line 27) - PHASE_WRAP_UP constant definition
- **entities/player.pyx** - Player entity infrastructure (cash, turn_order)
- **entities/fi.pyx** - Foreign Investor entity infrastructure
- **entities/company.pyx** - Company ownership and transfer infrastructure
- **entities/deck.pyx** - Deck draw infrastructure
- **.planning/milestones/v2-REQUIREMENTS.md** - WRAP_UP mentioned as out of scope for v2
- **.planning/research/FORCED_ACTION_FEATURES.md** - WRAP_UP noted as fully deterministic

**Confidence level: HIGH** - WRAP_UP phase rules are explicit in official rulebook with no ambiguity.
