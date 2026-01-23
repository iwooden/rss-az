# Rolling Stock Stars - Cython Game Engine

## What This Is

A high-performance Cython game engine for "Rolling Stock Stars" board game with complete INVEST and BID_IN_AUCTION phases. The engine stores state as a single contiguous float32 array for zero-copy passing to PyTorch, optimized for AlphaZero-style self-play training.

## Core Value

Fast, reproducible game simulation for AI training with full rules compliance.

## Requirements

### Validated

**v1 - Game State Initialization:**
- ✓ GameState.initialize_game(seed) method produces valid starting state — v1
- ✓ Players receive correct starting cash (30 for 3-5p, 25 for 6p) — v1
- ✓ Foreign Investor starts with 4 cash, no companies — v1
- ✓ All 8 corporations start inactive with unissued shares — v1
- ✓ All 27 market price slots start available — v1
- ✓ Deck built per rules (game end at bottom, colors stacked, correct counts) — v1
- ✓ N companies drawn and marked available for auction — v1
- ✓ Turn state initialized (phase 1, CoO 1, turn 1, player 0) — v1
- ✓ Reproducible games via seed parameter — v1

**v2 - INVEST & BID_IN_AUCTION:**
- ✓ GameDriver class dispatches actions to phase handlers — v2
- ✓ INVEST phase: pass, start auction, buy/sell shares — v2
- ✓ BID_IN_AUCTION phase: leave auction, raise bid, resolution — v2
- ✓ Share price movement skips occupied spaces — v2
- ✓ Round-trip limit enforcement (2 per corp per turn) — v2
- ✓ Corporation bankruptcy at price 0 — v2
- ✓ Presidency transfer with incumbent advantage — v2
- ✓ Receivership when all player shares sold — v2
- ✓ 170 tests with invariant checking — v2

**v2.1 - Forced Action Auto-Application:**
- ✓ Auto-apply forced actions when only 1 legal action exists — v2.1
- ✓ Iterative loop until 2+ choices available or game over — v2.1
- ✓ Validate 0 legal actions is error (unless GAME_OVER phase) — v2.1
- ✓ Update tests to account for auto-advancement behavior — v2.1

### Active

(Pending next milestone definition)

### Out of Scope

- Mobile/web client — CLI/API only
- Game replay viewer — training focus only
- Save/load to disk — in-memory state only for now
- FI auction fallback — edge case, defer
- State cloning optimization — basic NumPy copy sufficient

## Context

**Shipped v2.1:** Forced Action Auto-Application (2026-01-23)
- 2 phases (7-8), 3 plans, 21 requirements
- ~25,100 LOC Cython total
- Test suite: 176 tests

**Shipped v2:** INVEST & BID_IN_AUCTION (2026-01-21)
- 5 phases (2-6), 12 plans, 48 requirements
- ~25,000 LOC Cython, ~2,850 LOC Python (tests)

**Tech stack:** Cython, NumPy, PyTorch-compatible state format

**Patterns established:**
- Entity handle initialization (init all handles before state modification)
- Module import pattern for Cython globals
- Per-task atomic commits with feat/test prefixes
- Stateless singleton pattern for GameDriver
- Phase handler pattern (cdef noexcept functions for zero overhead)
- Two-pass presidency algorithm for tie-breaking
- Bankruptcy inline execution during sell handler

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| GameState method (not standalone) | Keeps initialization close to state management | ✓ Good |
| Seed parameter with -1 default | Enables reproducible training while allowing random games | ✓ Good |
| Entity initialization order | Initialize all handles before setting state to ensure offset caching | ✓ Good |
| Module import pattern for entities | Avoids Cython circular import issues | ✓ Good |
| Starting cash: 30 (3-5p), 25 (6p) | Official game rules | ✓ Good |
| Float32 state array | Zero-copy to PyTorch tensors | ✓ Good (existing) |
| GameDriver stateless singleton | Following entity handle pattern, all state in GameState | ✓ Good |
| Phase handlers as cdef noexcept | Maximum performance, zero error-handling overhead | ✓ Good |
| Bankruptcy inline execution | Execute immediately during sell, no deferral | ✓ Good |
| Two-pass presidency algorithm | Correct incumbent tie-breaking | ✓ Good |
| Shared test fixtures (conftest.py) | Consistent invariant checking across all tests | ✓ Good |
| Auto-apply loop pattern | Iterative forced action until 2+ choices | ✓ Good |
| Early-exit counting | Stop at count=2 instead of counting all | ✓ Good |
| History tracking via optional param | Zero overhead in production, full observability in tests | ✓ Good |

## Constraints

- **Performance:** Must support high-throughput self-play (thousands of games/minute)
- **Reproducibility:** Seed parameter must produce identical games
- **Compatibility:** State array must be directly usable by PyTorch

---
*Last updated: 2026-01-23 after v2.1 milestone complete*
