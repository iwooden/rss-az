# Rolling Stock Stars - Game State Initialization

## What This Is

A game state initialization method for the Rolling Stock Stars Cython game engine. The method sets up a valid starting game state according to the official rules, enabling reproducible games for AlphaZero-style self-play training.

## Core Value

Given a player count and optional seed, produce a correctly initialized game state that follows all setup rules from RULES.md.

## Requirements

### Validated

- ✓ GameState class allocates contiguous float32 array — existing
- ✓ State layout computed based on player count — existing
- ✓ Entity accessors (Player, Corp, Company, etc.) provide getters/setters — existing
- ✓ Deck entity tracks card order and draw state — existing
- ✓ TurnState entity tracks phase, CoO level, auction state — existing
- ✓ Static company data (face values, synergies, prices) in core/data — existing

### Active

- [ ] GameState.initialize_new_game(seed=None) method
- [ ] Player starting cash set correctly (30● or 25● for 6p)
- [ ] Random player turn order (1 to N, seeded)
- [ ] Foreign Investor starting cash (4●)
- [ ] All corporations inactive, shares reset
- [ ] All 27 market price slots marked available
- [ ] Deck built per rules (game end card at bottom, colors stacked correctly)
- [ ] Company counts per color based on player count
- [ ] Deck shuffled with provided seed
- [ ] N companies drawn and marked available for auction
- [ ] Turn state initialized (phase 1, CoO level 1, turn 1)
- [ ] Active player set to player 0

### Out of Scope

- Phase transition logic — separate feature, not part of initialization
- Action validation/masking — already exists separately
- Game loop/play mechanics — initialization only

## Context

**Existing codebase:** High-performance Cython game engine with single-vector state representation. State stored as float32 array passed directly to PyTorch. Entity accessor pattern provides clean API.

**Key files:**
- `core/state.pyx` — GameState class (add method here)
- `core/data.pyx` — Game constants, company data
- `entities/` — Entity accessors for state sections
- `RULES.md` — Official game rules, setup section (lines 90-113)

**Deck building rules (from RULES.md):**
1. Game end card at bottom
2. Highest face value company of each color set aside
3. Remaining companies shuffled by color
4. Add N companies per color (special cases: 4p gets 5 orange, 5p gets 7 orange, 6p uses all)
5. Stack: blue, green, yellow, orange, red (red on top)
6. Draw N cards as available for auction

## Constraints

- **Performance**: Method runs once per game, so performance is less critical than game loop. Still prefer Cython for consistency.
- **Reproducibility**: Must accept seed parameter for deterministic deck shuffling
- **Compatibility**: Must work with existing entity accessor pattern

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| GameState method (not standalone function) | Keeps initialization close to state management | — Pending |
| Seed parameter with None default | Enables reproducible training while allowing random games | — Pending |

---
*Last updated: 2026-01-20 after initialization*
