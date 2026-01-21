# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-20)

**Core value:** Fast, reproducible game simulation for AI training with full rules compliance
**Current focus:** Phase 4 - Share Trading

## Current Position

Phase: 4 of 6 (Share Trading)
Plan: 2 of 2 in current phase
Status: Phase complete
Last activity: 2026-01-21 - Completed 04-02-PLAN.md

Progress: v1 [completed] | v2 [██████░░░░] 60%

## Performance Metrics

**Velocity:**
- Total plans completed: 8 (1 v1, 7 v2)
- Average duration: 4min 5sec
- Total execution time: 0.55 hours

**By Milestone:**

| Milestone | Phases | Plans | Duration |
|-----------|--------|-------|----------|
| v1 Game State Init | 1 | 1 | 4min 25sec |
| v2 INVEST/BID | 5 | 5/5 | 22min 43sec (avg 4min 33sec) |
| v2 Share Trading | 4 | 2/2 | 4min 35sec (avg 2min 17sec) |

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Key patterns from v1 and v2:

- Entity initialization order pattern — Initialize all handles before setting state
- Module import pattern for entities — Avoid Cython circular imports (03-01)
- Per-task atomic commits — feat/test prefixes for git bisect
- Stateless singleton pattern — GameDriver follows entity handle design (02-01)
- Phase handler pattern — cdef noexcept functions for zero overhead (02-01)
- Validation at dispatch — Check action mask before routing to phase handlers (02-01)
- Test fixture pattern — game_state, invest_state, bid_state fixtures (02-02)
- Parametrized player count tests — Verify all 3-6 player configurations (02-02)
- pytest conftest pattern — Add project root to sys.path for Cython modules (02-02)
- Cdef variable declaration pattern — Declare all cdef vars at function start in Cython (03-01)
- Turn order navigation pattern — Find player at position, advance with wraparound (03-01)
- Auction resolution sequence pattern — pay -> transfer -> update -> draw -> cleanup -> transition (03-02)
- Active player counting pattern — Iterate all players checking flag status (03-02)
- Company entity initialization — Must call company.initialize(state) in GameState.initialize_game (03-02)
- Public accessor pattern — Add cpdef wrappers for test access to cdef methods (03-03)
- Test fixture pattern — Phase-specific fixtures return state in target phase (03-03)
- Location cache invalidation — Rescan location before clearing to avoid stale cache (03-03)
- Price movement pattern — find_next_higher/lower_space skips occupied spaces (04-01)
- Index 26 always available — Price $75 never marked occupied, multiple corps can share (04-01)
- Round-trip blocking pattern — Check round-trips before buy/sell mask generation (04-02)
- Trade state fixture pattern — Manually configure corp for testing without IPO (04-02)

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-01-21
Stopped at: Completed 04-02-PLAN.md (Phase 4 complete)
Resume file: None
Next action: Begin Phase 5 (Presidency & Bankruptcy) or discuss next phase
