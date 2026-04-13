# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project Overview

High-performance Cython game engine for "Rolling Stock Stars", optimized for AlphaZero-style self-play. Game state is a single contiguous int16 numpy array; per-token NN features are produced lazily by `get_token_data()`, separate from state storage.

**Important:** "Rolling Stock Stars" is NOT "Rolling Stock" — rules differ. Always consult `RULES.md` as the authoritative source.

- Engine supports 2–6 players; training/model/MCTS scoped to 3–5p
- Compact int16 state vector (sizes in `VECTORS.md`)
- Nogil hot paths, target thousands of games/min

## Transformer Refactor (current state)

We are on the `transformer-refactor` branch. Old MLP-era design (state rotation, dense global action vector, phase-specific MLP heads) is **torn out**. New design: each game entity is its own transformer token, state is compact int16 with no rotation/pre-normalization, `get_token_data()` is the sole engine→NN interface, action encodings are phase-local and sparse, and policy logits are read directly from the relevant entity tokens.

**Engine layer is complete:** `core/state`, `core/data`, `core/actions` (encode/decode + all 8 `_enumerate_*` helpers), `core/driver` (game-loop dispatch + forced-action auto-chain), `entities/*` (all 7 handles), and `phases/*` (all 11 phase handlers) are implemented and building. Remaining work: `get_token_data()`, MCTS rewrite, and train pipeline rewrite.

Backwards compatibility is **not** a goal. The `main` branch preserves pre-refactor code as a reference for **rule intent only** — do not copy state layout, field offsets, action indices, or function signatures from it.

### Transformer (summary of `nn/transformer.py` + `transformers.md`)

**Tokens** (3p: 57, 4p: 58, 5p: 59). Shared `token_dim=63` projected via type-specific linear layers + learned type embeddings. No positional encoding. Token order in eval buffer: `[players..., corps..., companies..., FI, market, global, invest, auction, dividend, issue, par, acq_offer, pass]`. Full per-token feature spec in `token-data.md`.

| Type | Count (3p) | Carries |
|------|-----------|---------|
| Player | 3–5 | identity, cash, net worth, liquidity, income, turn order, has passed, owned shares, share buys/sells, round trips, presidencies, owned companies |
| Corporation | 8 | identity, active, receivership, passed ACQ_OFFER, shares (unissued/issued/bank), price index+value, pending move, cash, acq proceeds, income breakdown (raw/synergy/CoO/ability), total stars, president, owned companies |
| Company | 36 | identity, active company flag, location flags, ownership (corp/player/FI), adjusted income, static data (prices, stars, base income), 36-dim synergy |
| FI | 1 | cash, income, owned companies |
| Market | 1 | 27 availability flags |
| Global | 1 | num players, phase one-hot, CoO one-hot, end-card flipped, cards remaining |
| Invest (phase ctx) | 1 | consecutive passes, buy/sell share price impacts per corp. Zeroed outside INVEST. |
| Auction (phase ctx) | 1 | price index+value, high bidder, starter. Zeroed outside BID. |
| Dividend (phase ctx) | 1 | 26 dividend-amount impacts, corp-remaining. Zeroed outside DIVIDENDS. |
| Issue (phase ctx) | 1 | price impact, corp-remaining. Zeroed outside ISSUE. |
| PAR (phase ctx) | 1 | per-par player cost, corp cash, issued shares. Zeroed outside IPO. |
| Acq Offer (phase ctx) | 1 | offer price index+value, offer corp, FI-company flag. Zeroed outside ACQ_OFFER. |
| Pass | 1 | no input — type embedding only. Emits pass logit. |

**Arch:** Pre-RMSNorm blocks, SwiGLU FFN (`ff_mult=3.0`), `d_model=128`, `num_layers=10`, `num_heads=2`, ~2.3M params.

**Policy heads** (entity-readout, weight-shared within type):

| Phase | Readout | Count |
|-------|---------|-------|
| INVEST | pass(1) + each company→15 auction offsets(540) + each corp→buy/sell(16) | **557** |
| BID | pass(1) + auction token→14 raise offsets | **15** |
| ACQUISITION | pass(1) + shared corp/company pair head→52 logits/pair (51 price offsets + FI buy) | **14977** |
| ACQ_OFFER | pass(1) + acq_offer token→buy(1) | **2** |
| CLOSING | pass(1) + each company→close logit | **37** |
| DIVIDENDS | dividend token→26 amounts | **26** |
| ISSUE | pass(1) + issue token→issue(1) | **2** |
| IPO | pass(1) + each corp→14 par-price logits (merged IPO+PAR) | **113** |

Action counts live in `core/data.pxd` as `ActionSize` `cpdef enum`; `core/actions.pxd` cimports them, `nn/transformer.py` imports `PHASE_ACTION_SIZES` + `MAX_ACTION_SIZE`. Single source of truth. An import-time roundtrip assert in `core/actions.pyx` catches encode-formula drift.

**Value head:** `(d_model → d_model/2 → 1 → tanh)` MLP shared across player tokens. No rotation means outputs are already in canonical player order.

### Engine implications

- **Compact state, `get_token_data()` is the only NN interface.** No visible/hidden split, no pre-normalization, no one-hot in state vector. `get_token_data(state, buffer, num_tokens, token_dim)` fills the eval buffer with raw/lightly-normalized features in nogil Cython. Feature spec: `token-data.md`. Implementation: `core/token_data.{pyx,pxd}`. **Still pending** — the missing link between Phase 1 and Phase 4.
- **Per-phase action indices.** Same integer means different things in different phases. Callers must carry `phase_id` with `action_id`. `MAX_LEGAL_ACTIONS = 256`; over-cap is a bug, not graceful degradation.
- **PAR folded into IPO.** Single `(corp, par_index)` action on merged `PHASE_IPO`.
- **`PHASE_ACQ_OFFER` is a first-class engine phase** for FI preemption / receivership offers. Both ACQUISITION and ACQ_OFFER are separate decision phases.
- **Offer-based phases are gone.** ACQUISITION exposes the full `(corp, company, offset)` space via masking; CLOSING exposes all eligible companies directly. Players pick in one shot. FI fixed-price buys fold into the pair space as the 52nd per-pair option (`FI_BUY`). Receivership auto-acquisitions and mandatory negative-income-and-cash closes remain as forced actions resolved by the driver.
- **FI-priority via `PHASE_ACQ_OFFER`.** When a player/receivership corp tries to acquire an FI-owned company and higher-priority corps exist (OS first at face value; rest by descending share price at high value), the engine pushes into `PHASE_ACQ_OFFER` and offers each in turn a 2-action `{pass, buy}`. Reuses existing `active_corp` (= preempting corp), `active_company` (= contested FI company), `active_player` (= that corp's president). No new state fields.
- **No per-ACQ active entity.** During ACQUISITION itself, `active_corp`/`active_company` sit at `-1`. The generic `active_corp`/`active_company`/`active_player` selectors plus per-phase remaining bitmasks cover every decision phase — no `closing_company`/`dividend_corp`/`issue_corp`/`ipo_company` fields needed.
- **Cross-president ACQ transfers:** supported via `state.acq_same_president` flag. When `False`, corps can acquire from entities presided by a different player; the owner gets an accept/decline choice via `PHASE_ACQ_OFFER`. Known 18xx.games divergence: cross-corp transfers between corps of the *same* president are always allowed (no offer needed).
- **3–5 player training.** Engine supports 2–6p for 18xx test compatibility, but all NN/MCTS/training code targets 3–5p only. No 2p or 6p training configs, token data, or model support.

### What currently builds

`setup.py` compiles all `.pyx` files in `core/`, `entities/`, and `phases/`.

| Area | Status |
|------|--------|
| `core/data.{pyx,pxd}` | ✅ Pure data + enums + norm divisors |
| `core/state.{pyx,pxd}` | ✅ Compact int16 layout with module-level `LAYOUT`/`*_FIELDS`/`*_OFFSETS` |
| `core/actions.{pyx,pxd}` | ✅ Phase-local encode/decode + all 8 `_enumerate_*` helpers implemented |
| `core/driver.{pyx,pxd}` | ✅ Game-loop dispatch, forced-action auto-chain, all phases wired |
| `entities/*.pyx` | ✅ All 7 handles rewritten, fully stateless |
| `phases/*.pyx` | ✅ All 11 phase handlers (8 decision + 3 automated) implemented |
| `nn/transformer.py` | ✅ Token-based transformer, imports cleanly |
| `mcts/` | ❌ Pre-refactor (`search.py`, `node.py`, `evaluator.py`, `mcts_core.pyx`) |
| `train/` | 🟡 Leaf modules import; orchestration (`eval_server`, `self_play`, `main`, `analyze_game`, `tournament`) needs rewrite for sparse policy + token buffers |
| `tests/` | 🟡 Only `tests/games_18xx/`; needs rewrite for new driver API |
| `interp/` | ❌ Removed |

Rule when touching anything on the broken list: **rewrite it against the new `core/state` + `core/actions` + `entities/` contract.**

## Devbox

- WSL2 on Windows, AMD Ryzen 9 9950X3D (32 cores), AMD Radeon RX 9070 XT (ROCm 7.2.0)

## Directory Structure

```
core/        # Engine: state.pyx, data.pyx, actions.pyx, driver.pyx
entities/    # Entity handles: player, corp, company, deck, turn, market, fi
phases/      # All 11 phase handlers (invest, bid, acquisition, acq_offer, closing, dividends, income, issue, ipo, wrap_up, end_card)
mcts/        # Stale pre-refactor MCTS
nn/          # transformer.py — token-based model
train/       # Self-play scaffolding; orchestration broken
  gpu/       # nvidia.py, amd.py (auto-detected)
tests/
  games_18xx/  # Replay tests (broken on core.driver)
scratchpad/  # Ad-hoc scripts (see Agent Instructions)
RULES.md, VECTORS.md          # Authoritative rules + state layout
transformers.md, sparse-refactor.md  # Design docs (deep reference only)
```

## Architecture

### Entity Handles

Global singletons provide access to state array regions:

```python
PLAYERS = [Player(i) for i in range(6)]
CORPS = [Corporation(i) for i in range(8)]
COMPANIES = [Company(i) for i in range(36)]
TURN = TurnState(); FI = ForeignInvestor(); MARKET = Market(); DECK = Deck()
```

Every handle is **fully stateless** — no per-instance offset cache, no `initialize()`. Singletons are reused with any `GameState` at any player count. Only per-instance fields are display identifiers (`player_id`, `corp_id`, `name`). Each handle exposes `cdef` nogil methods and `cpdef` Python wrappers.

### GameState (`core/state.pyx`)

Thin wrapper around a single int16 numpy array. Raw integers only — no normalization, no one-hot. See `VECTORS.md` for the authoritative layout.

- `GameState.from_array(array, num_players)` — copy-in path; accepts non-contiguous 1-D int16 views.
- `GameState.from_buffer(buffer, num_players)` / `state.rebind(buffer, num_players)` — zero-copy; require writable C-contiguous int16 buffers whose canonical `turn.num_players` slot already matches.

**Module-level layout constants** on `core.state` (computed once, shared across all `GameState`s): `LAYOUT`, `PLAYER_FIELDS`, `CORP_FIELDS`, `TURN_OFFSETS`, `COMPANY_OFFSETS`, `DECK_OFFSETS`, `FI_OFFSETS`. These are Cython `cdef` structs — NOT accessible from Python. Cython: `from core.state cimport LAYOUT, ...`. Python: `core.state.get_layout(num_players)`, `get_player_fields()`, etc. (namedtuple accessors).

`total_size` is not on `LAYOUT` (depends on num_players); compute inline as `LAYOUT.players_offset + PLAYER_FIELDS.size * num_players`. Section sizes live on their field structs (`PLAYER_FIELDS.size`, etc.).

All per-player tracking (cash, net_worth, liquidity, turn order, shares, income, share buys/sells, `has_passed`) lives inside one player block — single pointer hop per player. Presidency is tracked per-corp via `CORP_FIELDS.president_id` (`player_id` or `-1`), not in the player block. Round-trip counts are derived on demand from `min(share_buys, share_sells)` per corp — no dedicated slot.

Corps cache `company_stars`; `share_price` is derived from `price_index`. Derived values (revenue, synergy income, CoO, ability income) cached lazily behind per-corp dirty bits (`TURN_OFFSETS.corp_cache_dirty`). Players have an analogous `player_cache_dirty`.

`companies.locations` (`CompanyLocation` enum) + `companies.owner_ids` are the single source of truth for ownership — no per-player/per-corp ownership bitmaps. `LOC_DECK = 0` is zero-init default; `__cinit__` seeds `owner_ids` and `corps.president_id` to `-1`.

### Actions (`core/actions.{pyx,pxd}`)

Per-phase, phase-local encoding. Read the `.pxd` directly — it's the contract.

- **8 decision phases** (`DecisionPhase` in `core/data.pxd`): `DPHASE_INVEST`, `DPHASE_BID`, `DPHASE_ACQUISITION`, `DPHASE_ACQ_OFFER`, `DPHASE_CLOSING`, `DPHASE_DIVIDENDS`, `DPHASE_ISSUE`, `DPHASE_IPO`. Engine's 12 `GamePhases` fold to these via `get_decision_phase(state)`.
- **Action counts:** `[557, 15, 14977, 2, 37, 26, 2, 113]`. Canonical in `core/data.pxd::ActionSize`.
- **Encode/decode:** family of `encode_*` `cdef inline` helpers + single `decode_action(phase_id, action_id)` inverse. `encode_action(ActionInfo)` is the tested roundtrip.
- **`get_decision_phase(state)`** returns `DecisionPhase` or `-1` for automated/terminal phases (`WRAP_UP`, `INCOME`, `END_CARD`, `GAME_OVER`).
- **`enumerate_legal_actions(state, phase_id, uint16_t* ids)`** — public contract; returns count, writes phase-local ids in deterministic order. All 8 `_enumerate_*` helpers are implemented with real mask logic.
- Buffers pad to `MAX_LEGAL_ACTIONS = 256`. Over-cap is a bug; `enumerate_legal_actions` asserts on overflow.

### Data (`core/data.pyx`)

Pure data + constants: static tables (36 companies, 8 corporations, 27 market prices, par/CoO tables, synergy matrix), enums (`GameConstants`, `GamePhases`, `CorpIndices`), NN normalization divisors. **No accessor functions, no computational helpers.** Other modules `cimport` underlying arrays directly. Helpers (synergy aggregation, required-stars, CoO lookup, par validity) live as private `cdef` functions in the entity that uses them — e.g. `_aggregate_synergies` in `entities/corp.pyx`. Promote to `cimport`-able symbols in the owning entity if needed by multiple modules.

### NN Model (`nn/transformer.py`)

Expects `(batch, num_tokens, token_dim)` features and per-phase action indices. The engine is now fully wired (driver + phases + enumerators), but `get_token_data()` — the bridge from compact state to token features — doesn't exist yet. Don't change the action-space contract without updating `core/actions.pxd` (import-time assert catches drift).

## Game Flow & Phases

**12 engine phases** (`GamePhases`):

| # | Phase | Notes |
|---|-------|-------|
| 0 | INVEST | Buy/sell shares, start auctions |
| 1 | BID | Bidding |
| 2 | WRAP_UP | Automated: FI buys at face value |
| 3 | ACQUISITION | Corps acquiring companies |
| 4 | ACQ_OFFER | FI preemption sub-phase |
| 5 | CLOSING | Player-owned companies closing |
| 6 | INCOME | Automated: payouts |
| 7 | DIVIDENDS | Declaration |
| 8 | END_CARD | Automated: end trigger |
| 9 | ISSUE_SHARES | Corp issuing |
| 10 | IPO | `(corp, par_index)` in one action (merged IPO+PAR) |
| 11 | GAME_OVER | Terminal |

Automated (no input): WRAP_UP, INCOME, END_CARD.

PAR is gone — merged into IPO. ACQ_OFFER took PAR's slot in the enum.

**8 decision phases** (what the model sees): INVEST, BID, ACQUISITION, ACQ_OFFER, CLOSING, DIVIDENDS, ISSUE, IPO. `core.data.ENGINE_TO_DECISION_PHASE` is the 12-slot lookup table (-1 for automated/terminal).

## Code Conventions

- **Naming:** `corp_id`/`company_id`/`player_id` = indices; `CORPS[i]`/`PLAYERS[i]`/`COMPANIES[i]` = singletons; `PHASE_*` = `GamePhases`, `DPHASE_*` = `DecisionPhase`, `LOC_*` = `CompanyLocation`.
- **ALL state access goes through entity handles.** Non-negotiable. Use `PLAYERS[i].get_cash(state)`, `CORPS[c].get_share_price(state)`, `COMPANIES[i].get_location(state)`, etc. Do **not** import `LAYOUT`/`*_FIELDS`/`*_OFFSETS` outside entity modules. Do **not** index `state._data` directly. Do **not** write ad-hoc field accessors in phase code. Entity handles own dirty-mask invalidation, cached-star bookkeeping, and ownership-location sync — reaching around them corrupts state.
- **Entity method surfaces are mostly finalized — ask before extending.** If you want a method that doesn't exist (e.g. `Corporation.net_treasury()`, `Player.can_afford_share()`), **stop and ask the user**. Same for changing existing method semantics.
- **`core/data` is data-only.** No field-level helpers. Static data is `cimport`ed directly.
- **Pointer safety:** Entity handles read offsets from cimported layout constants, cache nothing per-instance. Player/Corp use an inline `_slot(field)` helper Cython optimizes away in nogil. This is the only code that should cimport those constants.
- **Guard with `assert`, not silent fallbacks.** In Cython, validate invariants with `assert` — not `if ...: return False/0`. Out-of-range IDs, inactive entities that should be active, malformed state: crash loudly. `assert` compiles out under `python -O`. Include an f-string message. Example: `assert 0 <= player_id < state._num_players, f"bad player_id {player_id}"`. **Exception:** genuine business-logic branches (e.g. "can't afford the share, so the action is illegal") still return cleanly. This rule is about defensive checks for things callers should guarantee.
- **No method-level imports.** All imports must be at file scope (top of the file). Do not use inline `import` inside functions, methods, or test bodies. This applies everywhere — production code, test code, and scripts.

## Build

**Python:** always `.venv/bin/python` (venv may not be activated).
**Pyright:** system `pyright` (at `/usr/bin/pyright`), NOT `.venv/bin/pyright`.
**Submodules:** absolute paths or single-line `cd /home/icebreaker/rss-az-cython2/submodules/18xx && ...`. Never `cd` and forget.

```bash
# Build (required before running Python)
.venv/bin/python setup.py build_ext --inplace 2>&1 | grep -E "(warning|error)" || true

# Clean
.venv/bin/python setup.py clean
```

> ⚠️ Refactor in progress. `core/*.pyx`, `entities/*.pyx`, and `phases/*.pyx` all build. `pytest tests/`, `python -m train`, benchmarks do not run yet (blocked on `get_token_data()` + MCTS/train rewrite).

- **Warning-free builds required.** File a beads issue if warnings appear.
- **Fix pyright errors before moving on**, even pre-existing ones. Run `pyright <file>` via Bash for definitive results (auto-injected diagnostics can be stale).

## Testing

Most of the old suite is deleted. Surviving code lives in `tests/games_18xx/` and currently doesn't import (depends on `core.driver`). Expect to rewrite piece by piece as refactor phases land — there is no "all tests pass" gate right now.

- Prefer invariant assertions (cash conservation, share counts, ownership consistency) applied at every state transition over narrow field-level unit tests. See old `tests/phases/conftest.py::assert_invariants` on `main` for the pattern.
- **When a test fails, assume the implementation is broken** until proven otherwise.

**18xx.games replay tests** (`tests/games_18xx/`): replay completed games and compare state at phase boundaries. Requires Ruby — `extract_states.rb` runs as subprocess. Broken on `core.driver`; comes back online after driver rewrite. Known divergences (verify still apply): (1) cross-president ACQ transfers, (2) directly offering positive-income company closes in CLO. Adding a game: export JSON → `tests/games_18xx/data/<id>.json` → add to `@pytest.mark.parametrize` in `test_replay.py`.

## Key Files by Task

| Task | Primary | Secondary |
|------|---------|-----------|
| Layout / field offsets | `core/state.{pyx,pxd}`, `VECTORS.md` | `entities/*.pxd` |
| Static game data / synergies / CoO | `core/data.{pyx,pxd}` | `RULES.md` |
| Action encoding | `core/actions.{pyx,pxd}` | `core/data.pxd::ActionSize` |
| Entity field access | `entities/<entity>.{pyx,pxd}` | — |
| Model | `nn/transformer.py` | Transformer Refactor section above |
| Driver | `core/driver.{pyx,pxd}` | `phases/*.pyx` |
| Phase implementation | `phases/<phase>.{pyx,pxd}` | entity handles, `RULES.md`; ask before adding handle methods |

---

# Agent Instructions

Project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

- Read `RULES.md` before touching game logic; `VECTORS.md` before touching state layout or entity handles.
- Don't auto-load `transformers.md` / `sparse-refactor.md` — the Transformer Refactor section above covers day-to-day needs.
- File beads issues for any out-of-scope discoveries. **No insights lost.** Always include `--description` — a title alone is not enough.
- Use `bd create --parent=<id>` subtasks for related work (dot-notation IDs), independent issues for unrelated bugs.

**Ad-hoc scripts:** Write to `scratchpad/`, not inline in Bash. Use `Edit` for iteration. Prepend `PYTHONPATH=/home/icebreaker/rss-az-cython2` when running. Saves tokens vs writing 100-line scripts inline.

**Referencing `main`:** OK for game rule intent, edge cases, and shape of old MCTS/train code. NOT OK for state layout, field offsets, action indices, or signatures — those have all changed.

**Session close:** Follow the beads session-close protocol from the session-start hook. Minimum gate is the build command above (clean + build_ext). This branch is ephemeral with no upstream — code is merged to `main` locally, not pushed.
