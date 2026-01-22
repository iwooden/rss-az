# Milestone v2: INVEST & BID_IN_AUCTION

**Status:** SHIPPED 2026-01-21
**Phases:** 2-6
**Total Plans:** 12

## Overview

Implement INVEST and BID_IN_AUCTION phase actions with game driver architecture, full share trading mechanics, and corporation lifecycle management.

## Phases

### Phase 2: Infrastructure Setup

**Goal**: Game driver can dispatch actions to phase handlers and generate legal move masks
**Depends on**: Phase 1 (game state initialization)
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md — Create GameDriver class and phase handler stubs
- [x] 02-02-PLAN.md — Test GameDriver dispatch and validation

**Details:**
- GameDriver class with apply_action() dispatching to phase handlers based on current phase
- get_legal_moves() wrapper providing action mask for neural network
- Phase handler stubs for INVEST (PASS/AUCTION/BUY_SHARE/SELL_SHARE) and BID (LEAVE/RAISE)
- Complete infrastructure ready for Phase 3 to fill in actual game logic

### Phase 3: INVEST Core & Auction Flow

**Goal**: Players can pass, start auctions, bid, and complete full auction cycles
**Depends on**: Phase 2
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md — Implement INVEST phase pass and start auction
- [x] 03-02-PLAN.md — Implement BID_IN_AUCTION phase handler
- [x] 03-03-PLAN.md — Test coverage for INVEST and BID phases

**Details:**
- Pass action increments consecutive_passes, advances turn order, triggers WRAP_UP when all players pass
- Start auction initializes all auction state (company, price, high_bidder, starter, passed flags)
- Leave auction action sets passed flag and resolves when one bidder remains
- Raise bid action updates auction price and high bidder, advances to next non-passed bidder
- Auction resolution: winner pays, receives company, net worth updated, new company drawn, state cleared
- Phase transitions back to INVEST with active player set to player after auction starter

### Phase 4: Share Trading

**Goal**: Players can buy and sell shares with proper price movement and trading limits
**Depends on**: Phase 3
**Plans**: 2 plans

Plans:
- [x] 04-01-PLAN.md — Implement buy/sell share handlers with price movement
- [x] 04-02-PLAN.md — Add round-trip limits to mask and test coverage

**Details:**
- Buy share handler: price moves first, player pays new price to corp, share transfers from bank
- Sell share handler: player receives current price, then price moves down, share transfers to bank
- Price movement correctly skips occupied spaces and treats index 26 as always available
- Round-trip tracking incremented on buy/sell
- Action mask checks round-trip limits before allowing buy/sell actions
- Player net worth updated after each transaction

### Phase 5: Presidency & Bankruptcy

**Goal**: Corporation ownership transfers correctly and bankruptcy procedure completes cleanly
**Depends on**: Phase 4
**Plans**: 2 plans

Plans:
- [x] 05-01-PLAN.md — Implement bankruptcy procedure and integrate into sell handler
- [x] 05-02-PLAN.md — Implement presidency/receivership checks and test coverage

**Details:**
- Bankruptcy procedure triggered at price index 0, executing complete corp reset
- Presidency transfer logic with correct incumbent advantage (two-pass algorithm)
- Receivership detection when all player shares = 0
- Receivership exit when player buys share (automatic presidency assignment)

### Phase 6: Integration & Tests

**Goal**: Comprehensive test coverage validates all phase logic and edge cases
**Depends on**: Phases 2-5
**Plans**: 3 plans

Plans:
- [x] 06-01-PLAN.md — Migrate tests to tests/phases/ and create shared assertion helpers
- [x] 06-02-PLAN.md — Add INVEST integration tests and edge case coverage
- [x] 06-03-PLAN.md — Add BID_IN_AUCTION integration tests and edge case coverage

**Details:**
- Created shared test infrastructure (conftest.py) with reusable fixtures and assertion helpers
- Migrated all phase tests to tests/phases/ directory
- 170 tests passing with comprehensive invariant checking

---

## Milestone Summary

**Key Decisions:**
- GameDriver uses stateless singleton pattern matching entity handles
- Phase handlers are cdef functions with noexcept for performance
- Action validation happens in driver before dispatching to phase handlers
- Two-pass presidency algorithm for correct incumbent tie-breaking
- Bankruptcy executed inline during sell handler (no deferral)

**Issues Resolved:**
- Company entity initialization (cache staleness on transfers)
- Tests directory naming conflict with Cython phases module
- Round-trip limit enforcement in action mask

**Issues Deferred:**
- WRAP_UP phase implementation
- Remaining game phases (ACQ, CLO, INC, DIV, END, ISS, IPO)
- FI auction fallback edge case

**Technical Debt Incurred:**
- None identified

---

*For current project status, see .planning/PROJECT.md*
*Archived: 2026-01-21*
