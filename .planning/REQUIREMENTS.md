# Requirements: Game State Initialization

**Defined:** 2026-01-20
**Core Value:** Given a player count and optional seed, produce a correctly initialized game state

## v1 Requirements

### Method Signature

- [ ] **INIT-01**: Method accepts optional seed parameter for reproducible deck shuffling
- [ ] **INIT-02**: Method can be called on existing GameState to reinitialize

### Player Setup

- [ ] **PLYR-01**: Each player receives 30● starting cash (25● for 6-player)
- [ ] **PLYR-02**: Players assigned linear turn order (player 0 = order 1, etc.)
- [ ] **PLYR-03**: All player-owned companies cleared
- [ ] **PLYR-04**: All player-owned shares cleared

### Foreign Investor Setup

- [ ] **FI-01**: Foreign Investor receives 4● starting cash
- [ ] **FI-02**: Foreign Investor owns no companies

### Corporation Setup

- [ ] **CORP-01**: All 8 corporations inactive
- [ ] **CORP-02**: Each corporation's shares reset (all unissued)
- [ ] **CORP-03**: No corporation owns any companies
- [ ] **CORP-04**: No corporation has a share price card

### Market Setup

- [ ] **MKT-01**: All 27 share price slots marked available

### Deck Building

- [ ] **DECK-01**: Game end card placed at deck bottom
- [ ] **DECK-02**: Highest face value company of each color set aside initially
- [ ] **DECK-03**: Remaining companies shuffled by color (using seed)
- [ ] **DECK-04**: Correct company count per color based on player count
- [ ] **DECK-05**: Colors stacked: blue, green, yellow, orange, red (red on top)

### Initial Draw

- [ ] **DRAW-01**: N companies drawn from deck (N = player count)
- [ ] **DRAW-02**: Drawn companies marked as available for auction

### Turn State

- [ ] **TURN-01**: Phase set to 1 (Investment)
- [ ] **TURN-02**: CoO level set to 1
- [ ] **TURN-03**: Turn number set to 1
- [ ] **TURN-04**: Active player set to player 0
- [ ] **TURN-05**: All auction/dividend/IPO state cleared

## v2 Requirements

(None — focused feature)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Phase transition logic | Separate feature, not initialization |
| Action masking | Already exists in core/actions.pyx |
| Game loop mechanics | Initialization only |
| Save/load game state | Different feature |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INIT-01 | Phase 1 | Pending |
| INIT-02 | Phase 1 | Pending |
| PLYR-01 | Phase 1 | Pending |
| PLYR-02 | Phase 1 | Pending |
| PLYR-03 | Phase 1 | Pending |
| PLYR-04 | Phase 1 | Pending |
| FI-01 | Phase 1 | Pending |
| FI-02 | Phase 1 | Pending |
| CORP-01 | Phase 1 | Pending |
| CORP-02 | Phase 1 | Pending |
| CORP-03 | Phase 1 | Pending |
| CORP-04 | Phase 1 | Pending |
| MKT-01 | Phase 1 | Pending |
| DECK-01 | Phase 1 | Pending |
| DECK-02 | Phase 1 | Pending |
| DECK-03 | Phase 1 | Pending |
| DECK-04 | Phase 1 | Pending |
| DECK-05 | Phase 1 | Pending |
| DRAW-01 | Phase 1 | Pending |
| DRAW-02 | Phase 1 | Pending |
| TURN-01 | Phase 1 | Pending |
| TURN-02 | Phase 1 | Pending |
| TURN-03 | Phase 1 | Pending |
| TURN-04 | Phase 1 | Pending |
| TURN-05 | Phase 1 | Pending |

**Coverage:**
- v1 requirements: 25 total
- Mapped to phases: 25
- Unmapped: 0 ✓

---
*Requirements defined: 2026-01-20*
*Last updated: 2026-01-20 after initial definition*
