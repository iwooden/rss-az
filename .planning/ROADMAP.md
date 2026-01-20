# Roadmap: Game State Initialization

## Overview

Single-phase implementation of GameState.initialize_new_game() method. All 25 requirements work together to deliver one atomic capability: producing a valid starting game state following official setup rules.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Game State Initialization** - Implement complete game setup method

## Phase Details

### Phase 1: Game State Initialization
**Goal**: GameState can initialize a valid starting game from scratch
**Depends on**: Nothing (first phase)
**Requirements**: INIT-01, INIT-02, PLYR-01, PLYR-02, PLYR-03, PLYR-04, FI-01, FI-02, CORP-01, CORP-02, CORP-03, CORP-04, MKT-01, DECK-01, DECK-02, DECK-03, DECK-04, DECK-05, DRAW-01, DRAW-02, TURN-01, TURN-02, TURN-03, TURN-04, TURN-05
**Success Criteria** (what must be TRUE):
  1. Developer can call GameState.initialize_new_game() with optional seed and receive a valid starting state
  2. Players receive correct starting cash (30● for 3-5p, 25● for 6p) and turn order
  3. All corporations start inactive with reset shares, Foreign Investor has 4●
  4. Deck is built correctly per RULES.md (game end card at bottom, colors stacked, correct counts by player count)
  5. N companies (N = player count) drawn from deck and marked available for auction
  6. Turn state reflects game start (phase 1, CoO level 1, turn 1, active player 0)
**Plans**: TBD

Plans:
- TBD (will be created during plan-phase)

## Progress

**Execution Order:**
Phases execute in numeric order.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Game State Initialization | 0/TBD | Not started | - |
