# Project Milestones: Rolling Stock Stars

## v6.0 INCOME Phase (Shipped: 2026-02-02)

**Delivered:** Complete INCOME phase with synergy calculation, corporation special abilities (PR, DA, S, VM), FI income bonus, corporation bankruptcy on negative income, and proper phase transitions.

**Phases completed:** 21-23 (7 plans total)

**Key accomplishments:**

- Synergy pair identification with bidirectional bonus summing (i<j loop pattern)
- Corporation income calculation with CoO deduction, synergy bonuses, and 4 special abilities
- ForeignInvestor income with +5 base bonus
- Corporation bankruptcy when negative income (cannot pay)
- INCOME phase handler with per-entity income application and immediate bankruptcy check
- TEMP_END_TURN phase for end-of-turn bookkeeping (turn increment, roundtrip clear)
- Complete phase chain: CLOSING → INCOME → TEMP_END_TURN → INVEST

**Stats:**

- 55 files created/modified
- ~28,000 lines Cython
- 3 phases, 7 plans, 17 requirements
- 5 days from v5.1 to v6.0 ship (2026-01-28 → 2026-02-02)

**Git range:** `5dcb70d` → `439506e`

**What's next:** v7.0 - DIVIDENDS phase (dividend declaration and share price adjustment)

---

## v5.1 nogil Optimization (Shipped: 2026-01-28)

**Delivered:** GIL-free mask generation functions enabling future thread-level parallelization for AlphaZero self-play training.

**Phases completed:** 20 (3 plans total)

**Key accomplishments:**

- Low-level nogil accessors for corp state (CorpOffsets struct + 6 functions)
- Low-level nogil accessors for turn state (TurnOffsets struct + 7 functions with _nogil suffix)
- All 7 mask functions + dispatch refactored to use low-level accessors and marked `noexcept nogil`
- 5 inline nogil accessor helpers to wrap remaining state access
- Fixed off-by-one bug in turn offset calculation discovered during refactoring
- Established pattern: dual-layer architecture (low-level nogil + high-level cpdef)

**Stats:**

- 15 files created/modified
- ~27,655 lines Cython
- 1 phase, 3 plans, 9 tasks
- 1 day from v5.0 to v5.1 ship (same day as Phase 20 added)

**Git range:** `1c4799d` → `2b7211e`

**What's next:** v6.0 - Remaining game phases (INCOME, DIVIDENDS, ISSUE_SHARES, IPO, END_GAME)

---

## v5.0 CLOSING Phase (Shipped: 2026-01-27)

**Delivered:** Complete CLOSING phase with FI/receivership auto-close, offer-based closing flow, Junkyard Scrappers bonus, and mandatory close protection.

**Phases completed:** 15.1, 16-19 (14 plans total)

**Key accomplishments:**

- Code quality refactoring: encoding helpers, CORPS list pattern, buffer optimization, test consolidation
- Auto-close logic: FI closes unprofitable companies, receivership corps close per color/CoO rules
- Offer-based close flow: sorted by face value, accept/pass actions, dynamic re-validation
- Junkyard Scrappers 2x printed income scrapping bonus
- Mandatory close at phase end protects players from negative cash in INCOME
- 312 tests with comprehensive edge case and integration coverage

**Stats:**

- 45 files created/modified
- ~27,100 lines Cython, ~5,500 lines Python tests
- 5 phases, 14 plans, 16 requirements
- 1 day from v4.0 to v5.0 ship

**Git range:** `feat(15.1-01)` → `docs(19): complete testing and integration phase`

**What's next:** nogil optimization (v5.1), then remaining game phases

---

## v4.0 ACQUISITION Phase (Shipped: 2026-01-26)

**Delivered:** AlphaZero-optimized ACQUISITION phase with offer-based flow, same-president trade restrictions, receivership auto-buy integration, and acquisition zone management.

**Phases completed:** 12-15 (13 plans total)

**Key accomplishments:**

- Offer-based acquisition flow with priority-sorted offers (OS→FI first, then by share price, corp-to-corp, player privates)
- Full action support: Accept at price, FI Buy High (max price), FI Buy Face (OS only), Pass
- 6 validation rules: price range, sufficient cash, minimum companies, no re-acquire, not already owned, same-president
- Receivership auto-buy: Corps in receivership automatically execute affordable FI purchases
- Zone merging: Acquisition proceeds and companies merge into owned state at phase end
- 60 comprehensive tests covering unit, validation, integration, and edge cases

**Stats:**

- 58 files created/modified
- ~26,518 lines Cython, ~4,929 lines Python tests
- 4 phases, 13 plans, 26 requirements
- 2 days from v3.0 to v4.0 ship

**Git range:** `da6e12e` → `c0ffed5`

**What's next:** CLOSING phase, remaining game phases (INC, DIV, END, ISS, IPO)

---

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
