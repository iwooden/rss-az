# Requirements Archive: v1 Game State Initialization

**Archived:** 2026-01-20
**Status:** ✅ SHIPPED

This is the archived requirements specification for v1.
For current requirements, see `.planning/REQUIREMENTS.md` (created for next milestone).

---

# Requirements: Game State Initialization

**Defined:** 2026-01-20
**Core Value:** Given a player count and optional seed, produce a correctly initialized game state

## v1 Requirements

### Method Signature

- [x] **INIT-01**: Method accepts optional seed parameter for reproducible deck shuffling
- [x] **INIT-02**: Method can be called on existing GameState to reinitialize

### Player Setup

- [x] **PLYR-01**: Each player receives 30● starting cash (25● for 6-player)
- [x] **PLYR-02**: Players assigned linear turn order (player 0 = order 1, etc.)
- [x] **PLYR-03**: All player-owned companies cleared
- [x] **PLYR-04**: All player-owned shares cleared

### Foreign Investor Setup

- [x] **FI-01**: Foreign Investor receives 4● starting cash
- [x] **FI-02**: Foreign Investor owns no companies

### Corporation Setup

- [x] **CORP-01**: All 8 corporations inactive
- [x] **CORP-02**: Each corporation's shares reset (all unissued)
- [x] **CORP-03**: No corporation owns any companies
- [x] **CORP-04**: No corporation has a share price card

### Market Setup

- [x] **MKT-01**: All 27 share price slots marked available

### Deck Building

- [x] **DECK-01**: Game end card placed at deck bottom
- [x] **DECK-02**: Highest face value company of each color set aside initially
- [x] **DECK-03**: Remaining companies shuffled by color (using seed)
- [x] **DECK-04**: Correct company count per color based on player count
- [x] **DECK-05**: Colors stacked: blue, green, yellow, orange, red (red on top)

### Initial Draw

- [x] **DRAW-01**: N companies drawn from deck (N = player count)
- [x] **DRAW-02**: Drawn companies marked as available for auction

### Turn State

- [x] **TURN-01**: Phase set to 1 (Investment)
- [x] **TURN-02**: CoO level set to 1
- [x] **TURN-03**: Turn number set to 1
- [x] **TURN-04**: Active player set to player 0
- [x] **TURN-05**: All auction/dividend/IPO state cleared

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
| INIT-01 | Phase 1 | Complete |
| INIT-02 | Phase 1 | Complete |
| PLYR-01 | Phase 1 | Complete |
| PLYR-02 | Phase 1 | Complete |
| PLYR-03 | Phase 1 | Complete |
| PLYR-04 | Phase 1 | Complete |
| FI-01 | Phase 1 | Complete |
| FI-02 | Phase 1 | Complete |
| CORP-01 | Phase 1 | Complete |
| CORP-02 | Phase 1 | Complete |
| CORP-03 | Phase 1 | Complete |
| CORP-04 | Phase 1 | Complete |
| MKT-01 | Phase 1 | Complete |
| DECK-01 | Phase 1 | Complete |
| DECK-02 | Phase 1 | Complete |
| DECK-03 | Phase 1 | Complete |
| DECK-04 | Phase 1 | Complete |
| DECK-05 | Phase 1 | Complete |
| DRAW-01 | Phase 1 | Complete |
| DRAW-02 | Phase 1 | Complete |
| TURN-01 | Phase 1 | Complete |
| TURN-02 | Phase 1 | Complete |
| TURN-03 | Phase 1 | Complete |
| TURN-04 | Phase 1 | Complete |
| TURN-05 | Phase 1 | Complete |

**Coverage:**
- v1 requirements: 25 total
- Mapped to phases: 25
- Complete: 25 ✓

---

## Milestone Summary

**Shipped:** 25 of 25 v1 requirements
**Adjusted:** None
**Dropped:** None

---
*Archived: 2026-01-20 as part of v1 milestone completion*
