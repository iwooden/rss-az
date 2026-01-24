# Project Milestones: Rolling Stock Stars

## v3.0 WRAP_UP Phase (Shipped: 2026-01-24)

**Delivered:** Deterministic WRAP_UP phase that reorders players by descending cash and executes Foreign Investor automatic company purchases at end of each INVEST round.

**Phases completed:** 9-11 + 10.1 inserted (6 plans total)

**Key accomplishments:**

- WRAP_UP phase handler with player reordering by descending cash (tie-break by old position)
- Foreign Investor purchases cheapest available companies at face value until unaffordable
- Availability transition makes all revealed companies available for next INVEST round
- GameDriver auto-executes non-player phases (WRAP_UP, ACQUISITION) with sentinel action history
- Terminal state detection prevents infinite phase loops
- Fixed player_stride calculation bug (phase 10.1) affecting player 1+ and FI data

**Stats:**

- 12 files created/modified
- ~25,419 lines Cython, ~3,384 lines Python tests
- 4 phases, 6 plans, 18 requirements
- 1 day from v2.1 to v3.0 ship

**Git range:** `feat(09-01)` → `docs(11): complete test-updates phase`

**What's next:** Remaining game phases (CLO, INC, DIV, END, ISS, IPO)

---

## v2.1 Forced Action Auto-Application (Shipped: 2026-01-23)

**Delivered:** GameDriver auto-applies forced actions iteratively until 2+ choices available, with optional history tracking for test observability and comprehensive error handling.

**Phases completed:** 7-8 (3 plans total)

**Key accomplishments:**

- GameDriver.apply_action() auto-applies forced actions iteratively until real choice exists
- Optional history parameter for full action chain observability in tests
- ForcedActionLoopError (100 iteration limit) and ZeroLegalActionsError exceptions
- GameState.from_array() staticmethod for state reconstruction from snapshots
- apply_and_track() fixture and ApplyTrackResult class for clean test assertions
- Test suite expanded to 176 tests with edge case coverage

**Stats:**

- 29 files created/modified
- ~25,100 lines Cython total
- 2 phases, 3 plans, 21 requirements
- 2 days from milestone start to ship

**Git range:** `feat(07-01)` → `docs(08): complete Test Updates phase`

**What's next:** WRAP_UP phase, remaining game phases (ACQ, CLO, INC, DIV, END, ISS, IPO)

---

## v2 INVEST & BID_IN_AUCTION (Shipped: 2026-01-21)

**Delivered:** Complete INVEST and BID_IN_AUCTION phase implementation with game driver architecture, full share trading mechanics, corporation lifecycle management (bankruptcy, presidency, receivership), and comprehensive test coverage.

**Phases completed:** 2-6 (12 plans total)

**Key accomplishments:**

- GameDriver class for action dispatch and legal move mask generation
- INVEST phase: pass, start auction, buy/sell shares with price movement and round-trip limits
- BID_IN_AUCTION phase: leave auction, raise bid, auction resolution with company transfer
- Corporation bankruptcy procedure (price 0 → company removal → share return → corp available for IPO)
- Presidency transfer with incumbent tie-breaking advantage
- Receivership detection and exit handling
- 170 tests with shared fixtures and invariant checking in tests/phases/

**Stats:**

- 160 files created/modified
- ~25,000 lines Cython, ~2,850 lines Python tests
- 5 phases, 12 plans, 48 requirements
- 17 days from v1 to v2 ship

**Git range:** `feat(02-01)` → `Cleanup for milestone 2`

**What's next:** WRAP_UP phase, remaining game phases (ACQ, CLO, INC, DIV, END, ISS, IPO)

---

## v1 Game State Initialization (Shipped: 2026-01-20)

**Delivered:** Complete game state initialization method that produces valid starting states for 3-6 players with reproducible seed support for AlphaZero training.

**Phases completed:** 1 (1 plan total)

**Key accomplishments:**

- `GameState.initialize_game()` method with optional seed parameter for reproducibility
- Proper entity initialization order pattern (handle init before state modification)
- Comprehensive test suite (28 tests) covering all 25 requirements
- Foundation for all game logic - games can now be created with valid starting state
- Support for 3-6 players with correct starting cash (30 for 3-5p, 25 for 6p)

**Stats:**

- 13 files created/modified
- ~1,800 lines added
- 1 phase, 1 plan, 2 tasks
- 16 days from project init to ship

**Git range:** `09cac9e` → `17e8413`

**What's next:** Game action implementation (auction, pass, phase transitions)

---
