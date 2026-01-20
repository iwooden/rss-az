# Rolling Stock Stars - Cython Game Engine

## What This Is

A high-performance Cython game engine for "Rolling Stock Stars" board game with complete game state initialization. The engine stores state as a single contiguous float32 array for zero-copy passing to PyTorch, optimized for AlphaZero-style self-play training.

## Core Value

Fast, reproducible game simulation for AI training with full rules compliance.

## Requirements

### Validated

- ✓ GameState.initialize_game(seed) method produces valid starting state — v1
- ✓ Players receive correct starting cash (30 for 3-5p, 25 for 6p) — v1
- ✓ Foreign Investor starts with 4 cash, no companies — v1
- ✓ All 8 corporations start inactive with unissued shares — v1
- ✓ All 27 market price slots start available — v1
- ✓ Deck built per rules (game end at bottom, colors stacked, correct counts) — v1
- ✓ N companies drawn and marked available for auction — v1
- ✓ Turn state initialized (phase 1, CoO 1, turn 1, player 0) — v1
- ✓ Reproducible games via seed parameter — v1

### Active

(Defined in next milestone's REQUIREMENTS.md)

### Out of Scope

- Mobile/web client — CLI/API only
- Game replay viewer — training focus only
- Save/load to disk — in-memory state only for now

## Context

**Shipped v1:** Game State Initialization (2026-01-20)
- 1 phase, 1 plan, 25 requirements
- ~24,500 LOC Cython, ~350 LOC Python (tests)
- Comprehensive test suite: 28 tests

**Tech stack:** Cython, NumPy, PyTorch-compatible state format

**Patterns established:**
- Entity handle initialization (init all handles before state modification)
- Module import pattern for Cython globals
- Per-task atomic commits with feat/test prefixes

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| GameState method (not standalone) | Keeps initialization close to state management | ✓ Good |
| Seed parameter with -1 default | Enables reproducible training while allowing random games | ✓ Good |
| Entity initialization order | Initialize all handles before setting state to ensure offset caching | ✓ Good |
| Module import pattern for entities | Avoids Cython circular import issues | ✓ Good |
| Starting cash: 30 (3-5p), 25 (6p) | Official game rules | ✓ Good |
| Float32 state array | Zero-copy to PyTorch tensors | ✓ Good (existing) |

## Constraints

- **Performance:** Must support high-throughput self-play (thousands of games/minute)
- **Reproducibility:** Seed parameter must produce identical games
- **Compatibility:** State array must be directly usable by PyTorch

---
*Last updated: 2026-01-20 after v1 milestone*
