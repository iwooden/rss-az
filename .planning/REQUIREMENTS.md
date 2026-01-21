# Requirements: Rolling Stock Stars - v2 INVEST & BID_IN_AUCTION

**Defined:** 2026-01-20
**Core Value:** Fast, reproducible game simulation for AI training with full rules compliance

## v2 Requirements

Requirements for INVEST and BID_IN_AUCTION phase implementation with game driver architecture.

### Game Driver

- [ ] **DRV-01**: GameDriver class dispatches actions to phase handlers based on game phase
- [ ] **DRV-02**: GameDriver.apply_action(state, action_idx) mutates state and returns status
- [ ] **DRV-03**: GameDriver.get_legal_moves(state) returns action mask for current state
- [ ] **DRV-04**: Action dispatch uses existing decode_action() from actions.pyx

### INVEST Phase - Core Actions

- [ ] **INV-01**: Pass action increments consecutive_passes counter
- [ ] **INV-02**: Non-pass actions reset consecutive_passes to 0
- [ ] **INV-03**: Phase transitions to WRAP_UP when consecutive_passes >= num_players
- [ ] **INV-04**: Active player advances to next player in turn order (not just next index)
- [ ] **INV-04a**: Turn order is read from player turn_order one-hot vectors in game state
- [ ] **INV-05**: Start auction action initializes auction state (company, price, high_bidder, starter)
- [ ] **INV-06**: Start auction action transitions phase to BID_IN_AUCTION

### INVEST Phase - Share Trading

- [ ] **INV-07**: Buy share deducts buy price from player cash
- [ ] **INV-08**: Buy share adds buy price to corporation cash
- [ ] **INV-09**: Buy share transfers 1 share from bank to player
- [ ] **INV-10**: Buy share moves corp price to next higher available market space
- [ ] **INV-11**: Sell share adds sell price to player cash
- [ ] **INV-12**: Sell share transfers 1 share from player to bank
- [ ] **INV-13**: Sell share moves corp price to next lower available market space
- [ ] **INV-14**: Price movement skips market spaces occupied by other corps
- [ ] **INV-15**: Player net worth updated after buy/sell share actions

### INVEST Phase - Round-Trip Limits

- [ ] **INV-16**: Round-trip tracking increments share_buys/share_sells counters per player per corp
- [ ] **INV-17**: Buy/sell blocked when round-trips >= MAX_ROUNDTRIPS (2) for that corp

### INVEST Phase - Presidency & Receivership

- [ ] **INV-18**: Change of presidency triggers when another player has more shares
- [ ] **INV-19**: Presidency tie-breaking uses turn order (earlier player wins)
- [ ] **INV-20**: Receivership flag set when all player-owned shares are sold
- [ ] **INV-21**: Buying share from receivership corp requires taking president share

### INVEST Phase - Bankruptcy

- [ ] **INV-22**: Corporation goes bankrupt when share price drops to 0
- [ ] **INV-23**: Bankruptcy removes all corporation's companies from game
- [ ] **INV-24**: Bankruptcy returns all issued shares to unissued stack
- [ ] **INV-25**: Bankruptcy returns corporation money to bank
- [ ] **INV-26**: Bankruptcy returns share price card to market row
- [ ] **INV-27**: Bankrupt corporation available for future IPO

### BID_IN_AUCTION Phase

- [ ] **BID-01**: Leave auction sets auction_passed flag for player
- [ ] **BID-02**: Active bidder rotation skips players who have left auction
- [ ] **BID-03**: Raise bid updates auction price and high bidder
- [ ] **BID-04**: Raise bid must exceed current auction price
- [ ] **BID-05**: Auction resolves when only one bidder remains
- [ ] **BID-06**: Auction winner pays bid price to bank
- [ ] **BID-07**: Auction winner receives company
- [ ] **BID-08**: Auction resolution clears all auction state
- [ ] **BID-09**: Auction resolution draws new company (marked unavailable)
- [ ] **BID-10**: Auction resolution transitions back to INVEST phase
- [ ] **BID-11**: Next action goes to player after auction starter in turn order (not winner)
- [ ] **BID-12**: Player net worth updated when winning auction

### Test Coverage

- [ ] **TST-01**: Comprehensive test suite in tests/phases/ directory
- [ ] **TST-02**: Tests cover common scenarios from game rules
- [ ] **TST-03**: Tests cover edge cases (bankruptcy, presidency change, receivership)
- [ ] **TST-04**: Tests verify action mask matches valid actions after state changes

## Future Requirements

Deferred to later milestones.

### Wrap-Up Phase

- **WRAP-01**: Player order recalculated by descending cash
- **WRAP-02**: Foreign investor buys available companies at face value

### Other Phases

- **ACQ-01**: Corporation acquisition phase
- **CLO-01**: Closing phase
- **INC-01**: Income phase
- **DIV-01**: Dividends phase
- **END-01**: End card phase
- **ISS-01**: Issue shares phase
- **IPO-01**: IPO phase

## Out of Scope

| Feature | Reason |
|---------|--------|
| FI auction fallback | Edge case, defer to v2+ |
| State cloning optimization | Basic NumPy copy sufficient |
| Undo/redo stack | Not needed for AI training |
| Action history/replay | Not needed for self-play |
| Save/load to disk | In-memory state sufficient |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DRV-01 | Phase 2 | Pending |
| DRV-02 | Phase 2 | Pending |
| DRV-03 | Phase 2 | Pending |
| DRV-04 | Phase 2 | Pending |
| INV-01 | Phase 3 | Pending |
| INV-02 | Phase 3 | Pending |
| INV-03 | Phase 3 | Pending |
| INV-04 | Phase 3 | Pending |
| INV-04a | Phase 3 | Pending |
| INV-05 | Phase 3 | Pending |
| INV-06 | Phase 3 | Pending |
| INV-07 | Phase 4 | Pending |
| INV-08 | Phase 4 | Pending |
| INV-09 | Phase 4 | Pending |
| INV-10 | Phase 4 | Pending |
| INV-11 | Phase 4 | Pending |
| INV-12 | Phase 4 | Pending |
| INV-13 | Phase 4 | Pending |
| INV-14 | Phase 4 | Pending |
| INV-15 | Phase 4 | Pending |
| INV-16 | Phase 4 | Pending |
| INV-17 | Phase 4 | Pending |
| INV-18 | Phase 5 | Pending |
| INV-19 | Phase 5 | Pending |
| INV-20 | Phase 5 | Pending |
| INV-21 | Phase 5 | Pending |
| INV-22 | Phase 5 | Pending |
| INV-23 | Phase 5 | Pending |
| INV-24 | Phase 5 | Pending |
| INV-25 | Phase 5 | Pending |
| INV-26 | Phase 5 | Pending |
| INV-27 | Phase 5 | Pending |
| BID-01 | Phase 3 | Pending |
| BID-02 | Phase 3 | Pending |
| BID-03 | Phase 3 | Pending |
| BID-04 | Phase 3 | Pending |
| BID-05 | Phase 3 | Pending |
| BID-06 | Phase 3 | Pending |
| BID-07 | Phase 3 | Pending |
| BID-08 | Phase 3 | Pending |
| BID-09 | Phase 3 | Pending |
| BID-10 | Phase 3 | Pending |
| BID-11 | Phase 3 | Pending |
| BID-12 | Phase 3 | Pending |
| TST-01 | All | Pending |
| TST-02 | All | Pending |
| TST-03 | All | Pending |
| TST-04 | All | Pending |

**Coverage:**
- v2 requirements: 45 total
- Mapped to phases: 45
- Unmapped: 0 ✓

---
*Requirements defined: 2026-01-20*
*Last updated: 2026-01-20 after initial definition*
