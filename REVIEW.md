# REVIEW-2: Principal Engineer Review of Cython Engine (`core`, `entities`, `phases`)

## Scope
Reviewed non-test Cython implementation in:
- `core/`
- `entities/`
- `phases/`

Priority order used:
1. Rules compliance (`RULES.md`)
2. Code organization / architecture (Entities pattern, factoring, duplication)
3. Performance (self-play + MCTS throughput)

## Executive Summary
The engine is structurally complete (all phases exist and are wired). Current `main` still has a few high-impact correctness and contract risks, primarily around stale derived corporation state and maintainability hazards in acquisition internals.

## Intentional Training-Variant Deviations (Acknowledged)
These are intentionally retained and now documented in `RULES.md` under `Training-Variant Constraints (Intentional Engine Deviations)`:
- Acquisition offer scope constraint (same-president scope + one-by-one offer presentation)
- Closing offer scope constraint (negative adjusted income only + one-by-one presentation)
- INVEST share round-trip limiter (`min(share_buys, share_sells)` capped at 2)

They should be treated as design choices for training stability/action-space control, not defects.

## Findings (Current `main`, Ordered by Severity)

### P0-1: Corporation `stars` can drift from actual holdings
- File: `entities/corp.pyx:221`
- File: `entities/corp.pyx:297`
- File: `phases/dividends.pyx:101`
- `corp.stars` is consumed by dividend share-price adjustment, but appears set at IPO and reset at bankruptcy only; I did not find authoritative updates tied to company add/remove events.
- Impact: share-price movement can diverge from true owned-company star totals.

### P0-2: Corporation `income` state field appears stale vs vector contract
- File: `entities/corp.pyx:289`
- File: `entities/corp.pyx:293`
- File: `phases/income.pyx:34`
- File: `phases/income.pyx:35`
- `income` exists in corp state layout (`VECTORS.md`) but INCOME phase computes local values and applies cash without updating stored corp `income`.
- Impact: NN-visible state can carry incorrect corp income features.

### P1-1: Auction leave semantics may allow invalid high-bidder behavior
- File: `core/actions.pyx:347`
- File: `phases/bid.pyx:76`
- `LEAVE_AUCTION` is always legal for the active bidder; no explicit guard prevents the current high bidder from leaving.
- Resolution logic still uses stored high bidder (`phases/bid.pyx:32`).
- Impact: potential invalid auction resolution path depending on action order.

### P1-2: Stale acquisition stub artifacts remain in active codebase
- File: `phases/acquisition.pyx:1040`
- File: `phases/acquisition.pxd:2`
- File: `phases/acquisition.pxd:8`
- Acquisition stub path/declarations are dead relative to current flow, but still present/exported.
- Impact: maintenance ambiguity and avoidable risk of accidental reuse.

### P2-1: Missing in-repo source doc references in comments
- File: `phases/invest.pyx:40`
- File: `phases/income.pyx:23`
- File: `phases/closing.pyx:99`
- File: `phases/ipo.pyx:153`
- Multiple comments cite `CONTEXT.md`, which is not present in repo root.
- Impact: intent is not auditable from repository sources.

### P2-2: Hidden-buffer writes bypass entity-level invariants in many places
- File: `phases/acquisition.pyx:385`
- File: `phases/acquisition.pyx:503`
- File: `phases/closing.pyx:278`
- File: `phases/closing.pyx:359`
- Direct writes to hidden offer/count/index offsets are common. This is fast, but invariant safety depends on discipline across modules.
- Impact: higher risk of subtle state divergence during future edits.

### P2-3: Global action mask buffer is process-global and not thread-safe
- File: `core/actions.pyx:55`
- File: `core/actions.pyx:58`
- Explicitly documented as non-thread-safe.
- Impact: threaded rollout would require isolation/redesign.

## Architecture Review

### Strengths
- Entity-handle pattern is broadly established and consistent.
- Compact hidden mirrors for one-hot state enable O(1) access where needed.
- Offer-buffer design in acquisition/closing keeps action space bounded and model-friendly.
- Cython-first implementation style is generally strong for performance-sensitive paths.

### Key structural risks
- Derived corp state ownership is unclear (`stars`, `income`): fields exist and are consumed, but lifecycle updates are not clearly centralized.
- Comment intent depends on missing docs (`CONTEXT.md`), reducing maintainability.
- Hidden-buffer manipulation is performant but diffuse; invariant enforcement is fragile without tighter helper boundaries.

## Performance Review

### Good
- Tight loops and static arrays are used in key paths.
- Hidden-state compact fields and buffered offer presentation avoid combinatorial blowups.

### Remaining opportunities
- If threaded execution is needed, replace process-global mask state with thread-local/per-instance strategy.
- Wrap hidden-buffer mutation in small internal helpers to preserve speed while shrinking invariant risk.

## Completeness / Gaps

### Implemented phases
All phases are implemented and wired in current `main`:
- INVEST
- BID_IN_AUCTION
- WRAP_UP
- ACQUISITION
- CLOSING
- INCOME
- DIVIDENDS
- END_CARD
- ISSUE_SHARES
- IPO
- GAME_OVER transition handling

### Remaining completeness concerns
- Derived-state drift risk (`stars`, `income`) can materially affect gameplay correctness and model signals.
- Dead acquisition stub exports and missing-reference comments still create avoidable maintenance risk.

## Recommended Fix Order
1. Resolve derived-state lifecycle for corp `stars` and corp `income` (single authoritative update points).
2. Tighten auction leave/high-bidder invariants.
3. Remove dead acquisition stub exports and replace `CONTEXT.md` references with in-repo canonical docs.
4. Plan thread-safety strategy for action mask generation before any threaded rollout.
