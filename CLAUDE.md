# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance Cython game engine for "Rolling Stock Stars" board game, optimized for AlphaZero-style self-play training. Game state is stored as a single contiguous int16 numpy array. Per-token features for the NN are produced lazily by `get_token_data()`, separate from state storage.

**Important:** "Rolling Stock Stars" is a different game than "Rolling Stock". They share similarities but many rules differ. Do NOT rely on knowledge of "Rolling Stock" — always consult `RULES.md` as the authoritative source for game rules.

**Key characteristics:**
- 2-6 player support with dynamic state sizing
- Single compact int16 state vector (sizes per player count are in `VECTORS.md`)
- No Python object overhead in hot paths (nogil execution)
- Benchmark target: thousands of games per minute

## Transformer Refactor (where we are right now)

We are mid-refactor on the `transformer-refactor` branch. The old MLP-era implementation (flat visible/hidden state, rotated per active player, dense global action vector, phase-specific MLP heads) has been **torn out**. The new design in one paragraph: each game entity is its own transformer token, the engine stores a compact int16 state with no rotation and no pre-normalization, `get_token_data()` is the sole engine→NN interface, action encodings are phase-local and sparse, and policy logits are read directly from the relevant entity tokens.

### Why we are doing this

- **No state rotation.** The MLP needed the active player at slot 0, which forced per-evaluation rotation of the state vector and per-player un-rotation of values. A transformer is permutation-equivariant over its input tokens, so the active player is just marked with an `is_active=1` flag on their player token. Rotation logic disappears from both the evaluator and the state builder.
- **Entity-readout policy heads.** Instead of phase-specific MLP heads that emit a dense action vector, each entity token emits only its own logits — corp tokens emit buy/sell and par logits, company tokens emit auction/close logits, dedicated phase-context tokens emit dividend/issue/acq_offer/bid-raise logits. Heads are weight-shared across tokens of the same type, so parameter counts stay low and the inductive bias matches the game structure.
- **Sparse legal-action interface.** The old engine/trainer bolted a single global dense action vector onto every surface (masks, replay targets, IPC, trainer softmax). ACQUISITION's full `(corp × company × offset)` space is 14,977 actions — feasible inside the model, painful as a dense runtime interface. The new stack is `(phase_id, num_legal, action_ids[:num_legal], policy_targets[:num_legal])`: the engine enumerates legal actions, the model scores just those, and MCTS/replay/trainer all operate over that legal set. Standard AlphaZero masking semantics — invalid actions are excluded from the softmax competition set, not trained with `-inf` logits.
- **Long-term: one model for 2-6p.** A transformer can handle a variable number of player tokens. The 3p prototype validates the architecture first; Phase 6 revisits multi-player-count batching.

### What the transformer looks like (summary of `nn/transformer.py` + `transformers.md`)

**Tokens** (3p: 56 total). All tokens are projected from a shared `token_dim` (default 63, zero-padded) through type-specific linear layers, then added to a learned type embedding. Token order is fixed for implementation convenience, but the model has no positional encoding — identity comes from type embeddings plus explicit ID features on the entity tokens.

| Type | Count (3p) | Carries |
|------|-----------|---------|
| Player | 3 | identity, cash, net worth, income, turn order, owned shares + presidencies, round trips, `is_active` flag |
| Corporation | 8 | identity, active flag, cash, shares, income, stars, price index, pending price move, owned companies, receivership flag, invest buy/sell impacts, `is_phase_active` flag |
| Company | 36 | identity, location flags, income, static features (face/low/high/stars), 36-dim synergy vector, `is_phase_active` flag |
| FI | 1 | cash, income, owned companies |
| Market | 1 | 27 availability flags |
| Global | 1 | phase one-hot, CoO one-hot, end-card flipped, cards remaining, consecutive passes |
| Auction (phase ctx) | 1 | price, starter, high bidder, passed flags. Zeroed outside BID. |
| Dividend (phase ctx) | 1 | dividend impact, corp-remaining flags. Zeroed outside DIVIDENDS. Policy logits read from here. |
| Issue (phase ctx) | 1 | corp-remaining flags, price impact, cash gain. Zeroed outside ISSUE. |
| PAR (phase ctx) | 1 | par treasury + share availability, shared across all corps. Zeroed outside IPO. |
| Acq Offer (phase ctx) | 1 | offer price, OS-flag, FI-flag. Zeroed outside ACQ_OFFER. |
| Pass | 1 | no input features — representation is the type embedding. Emits the shared pass logit. |

**Architecture.** Pre-RMSNorm transformer blocks with SwiGLU FFN (three weight matrices, `ff_mult=3.0`), `d_model=128`, `num_layers=10`, `num_heads=2`. No positional encoding. ~2.3M params at default config.

**Policy heads** (entity-readout, weight-shared within a type):

| Phase | Readout |
|-------|---------|
| INVEST | pass token → pass (1); each company → 15 auction offset logits (540); each corp → buy/sell (16). **557 total.** |
| BID | pass token → leave-auction (1); auction token → 14 raise offsets. **15.** |
| ACQUISITION | pass token → pass (1); shared corp/company pair head → 52 logits per pair (51 price offsets + FI buy). **14,977.** |
| ACQ_OFFER | pass token → pass (1); acq_offer token → buy (1). **2.** |
| CLOSING | pass token → pass (1); each company → close logit (36). **37.** |
| DIVIDENDS | dividend token → 26 amounts. **26.** |
| ISSUE | pass token → pass (1); issue token → issue (1). **2.** |
| IPO | pass token → pass (1); each corp → 14 par-price logits (112). **113.** (The merged IPO+PAR phase.) |

These `PHASE_ACTION_SIZES` must stay in lockstep with `core/actions.pxd`. Both files have import-time self-checks against this, so drift fails fast.

**Value head.** A small `(d_model → d_model/2 → 1 → tanh)` MLP shared across all player tokens. Each player token emits its own value in `[-1, 1]`. Because player tokens aren't slot-rotated, the output values are already in canonical order (player 0's token → v_0, etc.) — no un-rotation.

### What that implies for the engine

- **Compact state, `get_token_data()` is the only NN interface.** No visible/hidden split, no pre-normalization in setters, no one-hot encoding in the state vector, no duplicated "active entity" blobs. `get_token_data(state, buffer, num_tokens, token_dim)` fills the eval buffer with raw/lightly-normalized features from the compact state, in nogil Cython. This helper is still pending — it's the missing link between Phase 1 and Phase 4.
- **Per-phase action indices, phase-local.** Same integer means different things in different phases. Callers must always carry the `phase_id` alongside the `action_id`. Max legal-action set per phase is capped at `MAX_LEGAL_ACTIONS = 256` for the sparse buffer width; over-cap is a bug, not a graceful-degradation case.
- **PAR folded into IPO.** The old two-step IPO (pick corp) → PAR (pick price) collapses into one `(corp, par_index)` action on the merged IPO phase.
- **Engine phase graph gains `ACQ_OFFER`.** What used to be a hidden sub-state of ACQUISITION is now a first-class engine phase for FI preemption / receivership offers. Both `ACQUISITION` and `ACQ_OFFER` are separate decision phases from the model's perspective.
- **Action-space simplifications are finalized — offer-based phases are gone.** Under the transformer refactor, the engine indirections that existed to keep the MLP's action space small are removed:
  - **Auction slots.** Replaced by company-indexed auction actions (`company_id × 15 + offset`).
  - **ACQUISITION offer buffer — deleted.** The old pre-sorted one-by-one `(corp, company, price)` offer presentation is gone. ACQUISITION exposes the full `(corp, company, offset)` space via masking, and the acting player picks the tuple in one shot (or passes). FI fixed-price purchases fold into the same pair space as the 52nd per-pair option (`FI_BUY`); there is no separate "FI buy" sub-flow. Receivership auto-acquisitions remain as forced actions resolved by the driver.
  - **CLOSING offer buffer — deleted.** The old pre-sorted one-by-one close presentation is gone. CLOSING exposes all eligible companies directly: the player picks one company to close, or passes. Mandatory closes for negative-income-and-cash corps still happen at the end as forced actions.
  - **FI-priority resolution via `PHASE_ACQ_OFFER`.** The old engine achieved FI-priority preemption by carefully ordering offers in the acquisition buffer. With the buffer gone, priority is now handled explicitly: whenever a player or receivership corp attempts to acquire an FI-owned company and one or more *higher-priority* corps exist (OS first at face value; remaining corps ordered by descending share price at high value), the engine pushes into `PHASE_ACQ_OFFER` and offers each higher-priority corp in turn the chance to preempt. Each offered corp's president sees the 2-action `{pass, buy}` space. Once all preempting candidates decline, control returns to the original acquirer and ACQUISITION resumes.
  - **No per-ACQ "active entity" bookkeeping.** The old hidden "current offer pointer" state is gone. During ACQUISITION itself there is no active corp or active company — the player picks freely, so `turn.active_corp` and `turn.active_company` sit at `-1`. During `ACQ_OFFER`, those existing selectors are reused: `active_corp` = preempting corp being offered, `active_company` = contested FI company, `active_player` = that corp's president. No new state fields (`closing_company` / `dividend_corp` / `issue_corp` / `ipo_company`) are needed — the generic `active_corp` / `active_company` / `active_player` selectors plus the per-phase remaining bitmasks already in the turn block cover every decision phase.
  - **Cross-player/cross-corp acquisition transfers.** Still not supported by our engine, tracked as a known divergence from 18xx.games replays.
  The old phase handlers on `main` are a reference for *rule intent* only; do not port their state machines verbatim — the new flow is a single direct action for ACQ/CLOSING plus a dedicated `ACQ_OFFER` phase for FI priority.
- **3-player only.** The token shape and replay buffer are sized for 3p. Multi-player-count support is explicitly out of scope for the prototype.

### Reference docs

`transformers.md` (architecture design doc) and `sparse-refactor.md` (sparse policy/replay design doc) have the full rationale, alternatives considered, and implementation-phase breakdown. Most sessions should not need them — everything above is enough to work against. Dip into them only when you need the fuller "why" behind a specific decision.

Backwards compatibility with the old implementation is **not** a goal. Anything outside `core/state.*`, `core/data.*`, `core/actions.*`, and `entities/` is either waiting to be rewritten or already stale. The pre-refactor code is preserved on the `main` branch for reference, but do not treat it as a contract — many of its internal structures, action indices, and phase handlers are intentionally being replaced.

### What currently builds and imports

`setup.py` only compiles `core/data.pyx`, `core/state.pyx`, `core/actions.pyx`, and `entities/*.pyx`. Anything else listed below that depends on those modules either hasn't been touched yet or is a stub.

| Area | Status |
|------|--------|
| `core/data.{pyx,pxd}` | ✅ Pure data + enums + normalization divisors. No accessor helpers. |
| `core/state.{pyx,pxd}` | ✅ Compact int16 layout with module-level `LAYOUT`/`*_OFFSETS`/`*_FIELDS` constants. |
| `core/actions.{pyx,pxd}` | ✅ Phase-local encode/decode + engine→decision bridge. ⚠ Legal-action enumeration helpers are still **empty stubs** (rss-az-848a). |
| `core/driver.{pyx,pxd}` | ❌ Pre-refactor code. Imports from `phases/*` which no longer exist. Not in the build. Will be rewritten against the new action/phase graph. |
| `entities/*.pyx` | ✅ All 7 entity handles (`player`, `corp`, `company`, `turn`, `fi`, `market`, `deck`) rewritten to be fully stateless against the module-level LAYOUT constants. |
| `phases/` | ❌ Empty package (just a stub `__init__.pyx`). Every phase handler (`invest`, `bid`, `acquisition`, `acq_offer`, `closing`, `dividends`, `income`, `issue`, `ipo`, `wrap_up`, `end_card`) needs to be rewritten. |
| `nn/transformer.py` | ✅ Token-based transformer (~2.3M params). Already done and imports cleanly. `nn/model_3p.py` is gone. |
| `mcts/` | ❌ `search.py`, `node.py`, `evaluator.py`, and `mcts_core.pyx` are pre-refactor. `mcts_core.pyx` is not in `setup.py`'s build set. Will be re-plumbed once driver + eval buffers exist. |
| `train/` | 🟡 Leaf modules (`config.py`, `replay_buffer.py`, `trainer.py`, `checkpoint.py`, `logging.py`, `augment.py`, `tb_reader.py`, `profile_stats.py`) still import cleanly. `eval_server.py`, `self_play.py`, `main.py`, `analyze_game.py`, `tournament.py` all depend on `core.driver` or `mcts.mcts_core` and are currently broken. |
| `tests/` | 🟡 Only `tests/games_18xx/` remains. The replay harness currently imports `core.driver` and therefore doesn't run; it'll come back online after the driver is rebuilt. No `conftest.py`, no `tests/phases/`. |
| `interp/` | ❌ Removed. |

When you reach for something on the broken list, the rule is: **rewrite it against the new `core/state` + `core/actions` + `entities/` contract.** The old file on `main` is fine as a reference for rules/intent, but its function signatures, offsets, and action indices are out of date and should not be copied verbatim.

## Devbox Hardware

- **Platform:** WSL2 on Windows
- **CPU:** AMD Ryzen 9 9950X3D (32 usable cores)
- **GPU:** AMD Radeon RX 9070 XT (ROCm 7.2.0)

## Directory Structure

```
rss-az-cython2/
├── core/              # Low-level engine: state.pyx, data.pyx, actions.pyx (driver.pyx = stale)
├── entities/          # Entity handles: player, corp, company, deck, turn, market, fi
├── phases/            # Empty stub package — all phase handlers await rewrite
├── mcts/              # Stale pre-refactor MCTS (node.py, search.py, evaluator.py, mcts_core.pyx)
├── nn/                # nn/transformer.py — token-based transformer (rewritten)
├── train/             # Self-play training scaffolding; many files still reference the old driver/mcts_core
│   └── gpu/           # Vendor-specific GPU optimizations: nvidia.py, amd.py (auto-detected)
├── tests/
│   └── games_18xx/    # 18xx.games replay tests (currently broken — depends on core.driver)
├── scratchpad/        # Ad-hoc scripts (see Agent Instructions)
├── RULES.md           # Complete game rules (authoritative)
├── VECTORS.md         # State buffer layout (authoritative)
├── transformers.md    # Transformer refactor design doc (large; deep reference only)
├── sparse-refactor.md # Sparse policy / replay refactor design doc (large; deep reference only)
```

## Architecture Overview

### Entity Handles Pattern

Global singleton instances provide clean access to state array regions:

```python
PLAYERS = [Player(i) for i in range(6)]       # Player handles
CORPS = [Corporation(i) for i in range(8)]    # Corporation handles
COMPANIES = [Company(i) for i in range(36)]   # Company handles
TURN = TurnState()                            # Turn tracking
FI = ForeignInvestor()                        # Foreign investor
MARKET = Market()                             # Stock market availability
DECK = Deck()                                 # Draw deck
```

Each entity provides:
- **cdef methods** (nogil): Direct pointer arithmetic, max performance
- **cpdef methods**: Python-accessible wrappers for testing
- **Access pattern**: `entity.get_field(state)` indexes the state buffer at a constant offset derived from the module-level `LAYOUT` / `PLAYER_FIELDS` / `CORP_FIELDS` / `TURN_OFFSETS` / `COMPANY_OFFSETS` / `DECK_OFFSETS` structs on `core.state`. **Every** handle is fully stateless — no per-instance offset cache, no `initialize()` step. The singletons are reused with any `GameState` at any player count; the only per-instance fields are display identifiers (`player_id`, `corp_id`, `name`).

Low-level (`cdef` nogil) wraps high-level (cpdef/def) with no code duplication.

## Key Modules

### GameState (`core/state.pyx`)

Single contiguous int16 numpy array. Raw integers only — no normalization, no one-hot encoding, no visible/hidden split. See `VECTORS.md` for the full authoritative layout (section order, strides, field offsets, sizes per player count).

`GameState` is basically just a thin wrapper around the state vector. There are no per-instance layout fields — entity handles read offsets directly from the module-level constants below. All field-level reads and writes go through entity handles in `entities/`.

`GameState.from_array(array, num_players)` is the copy-in path and accepts non-contiguous 1-D `int16` views. `GameState.from_buffer(buffer, num_players)` and `state.rebind(buffer, num_players)` are the zero-copy paths and require writable C-contiguous `int16` buffers whose canonical `turn.num_players` slot already matches the claimed player count.

**Module-level layout constants** on `core.state` (computed once at import, shared by every `GameState`):

- `LAYOUT` — `cdef StateLayout`: top-level section offsets (`fi_offset`, `companies_offset`, `market_offset`, `corps_offset`, `turn_offset`, `deck_offset`, `players_offset`). `total_size` is *not* in this struct because it depends on `num_players`; compute it inline as `LAYOUT.players_offset + PLAYER_FIELDS.size * num_players` at the few sites that need it (allocation, length validation). Section sizes live inside their respective field structs (`PLAYER_FIELDS.size`, `CORP_FIELDS.size`, `TURN_OFFSETS.size`, `COMPANY_OFFSETS.size`, `DECK_OFFSETS.size`), not on the top-level layout.
- `PLAYER_FIELDS` — `cdef PlayerFieldOffsets`: relative offsets within a player block (`cash`, `owned_shares`, `has_passed`, `share_buys`, `share_sells`, …, plus `size`).
- `CORP_FIELDS` — `cdef CorpFieldOffsets`: relative offsets within a corp block (`active`, `cash`, `company_stars`, `price_index`, …, plus `size`).
- `TURN_OFFSETS` — `cdef TurnStateOffsets`: relative offsets within the (fixed-size) turn block, plus `size`.
- `COMPANY_OFFSETS` — `cdef CompanyOffsets`: sub-offsets for the three parallel 36-slot arrays (`incomes`, `locations`, `owner_ids`), plus `size`.
- `DECK_OFFSETS` — `cdef DeckOffsets`: sub-offsets for `top` (1) + `order` (36), plus `size`.

These are Cython `cdef` structs — **NOT accessible from Python directly**. Cython code uses `from core.state cimport LAYOUT, TURN_OFFSETS, PLAYER_FIELDS, CORP_FIELDS, COMPANY_OFFSETS, DECK_OFFSETS`. Python code uses the namedtuple accessors `core.state.get_layout(num_players)`, `get_player_fields()`, `get_corp_fields()`, `get_turn_fields()`, `get_company_fields()`, `get_deck_fields()`.

All per-player tracking — cash, net_worth, liquidity, turn order, owned shares, presidencies, round trips, income, per-turn share buys/sells, and the per-phase `has_passed` flag — lives inside one player block, so a single pointer hop reaches everything for player `i`.

The corp block carries a single cached `company_stars` slot; total stars and cash stars are derived on demand. `share_price` is not stored; it is derived from `price_index`. Corps cache derived values (revenue, synergy income, CoO cost, ability income) lazily behind a per-corp dirty bit; `LAYOUT.turn_offset + TURN_OFFSETS.corp_cache_dirty` holds the mask. Players have an analogous `player_cache_dirty` mask.

`companies.locations` (`CompanyLocation` enum) plus `companies.owner_ids` are the single source of truth for "who owns what" — there are no per-player or per-corp ownership bitmaps. `LOC_DECK = 0` is the zero-init default; `__cinit__` explicitly seeds `companies.owner_ids` to `-1`.

### Actions (`core/actions.{pyx,pxd}`)

**Per-phase, phase-local action encoding.** Replaces the old single global dense action vector. `core/actions.pxd` is the source of truth for the contract — read it directly, it's short.

- 8 decision phases: `DPHASE_INVEST`, `DPHASE_BID`, `DPHASE_ACQUISITION`, `DPHASE_ACQ_OFFER`, `DPHASE_CLOSING`, `DPHASE_DIVIDENDS`, `DPHASE_ISSUE`, `DPHASE_IPO`. These are what the model sees; the engine's 12 `GamePhases` fold down into them via `get_decision_phase`.
- **Action counts per phase:** `[557, 15, 14977, 2, 37, 26, 2, 113]`. These must stay in lockstep with `nn/transformer.py::PHASE_ACTION_SIZES`; an import-time assertion guards against drift.
- **Encode/decode:** each decision phase has a family of `encode_*` `cdef inline` helpers and a single `decode_action(phase_id, action_id)` inverse. `encode_action(ActionInfo)` is the tested roundtrip.
- **Engine → decision bridge:** `get_decision_phase(state)` reads the engine phase from the state buffer and maps it to a `DecisionPhase`, or returns `-1` for automated/terminal phases (`WRAP_UP`, `INCOME`, `END_CARD`, `GAME_OVER`).
- **Sparse legal-action enumeration:** `enumerate_legal_actions(state, phase_id, uint16_t* ids)` is the public contract. Every phase has a `_enumerate_*` helper that writes phase-local ids in a deterministic order and returns the count. **Right now these helpers are all empty stubs** (tracked as rss-az-848a). The mask-generation logic has to be ported out of the old `actions-old.pyx` on `main` and re-keyed to the new encoding.
- **Buffer width:** legal-action buffers pad to `MAX_LEGAL_ACTIONS = 256` (revisit after profiling). An over-cap enumeration is a bug, not a recoverable condition, and `enumerate_legal_actions` asserts on overflow.

### Driver (`core/driver.pyx`) — STALE

The current `core/driver.pyx` is the pre-refactor driver. It imports from `phases/*.pyx` (which don't exist anymore) and uses the old dense action layout, so it is **not in the build set** and does not compile. The rewritten driver will route decision-phase actions through `enumerate_legal_actions` + `decode_action`, auto-apply forced actions, and dispatch to new phase handlers once those land. Until that happens, there is no runnable game loop.

### Data (`core/data.pyx`)

Pure data + constants module. Holds the static game tables (36 companies, 8 corporations, 27 market prices, par/CoO tables, synergy matrix), the shared enums (`GameConstants`, `GamePhases`, `CorpIndices`), and the normalization divisors used by NN token extraction. **No accessor functions, no computational helpers** — other modules `cimport` the underlying arrays directly. All gamestate reads/writes go through entity handles, never through helpers in `core/data`. Computational helpers (synergy aggregation, required-stars formula, cost-of-ownership lookup, par-price validity, etc.) live as private cdef functions in the module that uses them — e.g. `_aggregate_synergies` and `_required_stars` are inlined in `entities/corp.pyx`. If a helper ends up needed by multiple modules, promote it to a `cimport`-able symbol in whichever entity owns the concept rather than recreating an accessor layer in `core/data`.

### NN Model (`nn/transformer.py`)

Token-based transformer (~2.3M params at default config). See the "What the transformer looks like" subsection above for the full token/head/architecture summary — that section replaces the need to consult `transformers.md` for day-to-day work. Token order in the eval buffer is `[players..., corps..., companies..., FI, market, global, auction, dividend, issue, par, acq_offer, pass]`.

The model is ahead of the engine: it already expects `(batch, num_tokens, token_dim)` token features and per-phase action indices, but nothing actually produces those yet — `get_token_data()` is the missing Phase-1 link. The transformer is the reason the rest of the refactor is happening; don't change its action-space contract without also updating `core/actions.pxd` (the import-time assert will catch drift).

## Game Flow & Phases

**12 engine phases** (`GamePhases`):

| Index | Phase | Description |
|-------|-------|-------------|
| 0 | INVEST | Buy/sell shares, start auctions |
| 1 | BID_IN_AUCTION | Bidding for a company |
| 2 | WRAP_UP | Automated: FI buys companies at face value |
| 3 | ACQUISITION | Corps acquiring companies |
| 4 | ACQ_OFFER | FI preemption sub-phase (higher-priority corps get first shot) |
| 5 | CLOSING | Player-owned companies closing |
| 6 | INCOME | Automated: income payouts |
| 7 | DIVIDENDS | Dividend declaration |
| 8 | END_CARD | Automated: game end card trigger |
| 9 | ISSUE_SHARES | Corp issuing a share |
| 10 | IPO | Select corp charter for company **and** par price in one action |
| 11 | GAME_OVER | Terminal state |

**Automated phases** (no player input): `WRAP_UP`, `INCOME`, `END_CARD`.

**PAR is gone.** The old IPO → PAR two-step is merged into a single `PHASE_IPO` that selects `(corp, par_index)` in one action. `ACQ_OFFER` has taken the slot PAR used to occupy in the engine phase enum.

**8 decision phases** (what the model sees, defined in `core/actions.pxd` as `DecisionPhase`): INVEST, BID, ACQUISITION, ACQ_OFFER, CLOSING, DIVIDENDS, ISSUE, IPO. `get_decision_phase(state)` maps from the engine phase to this compressed space.

**The old offer-buffer pattern is gone.** ACQUISITION and CLOSING no longer pre-generate sorted offers and walk through them one-by-one with CLOSE/BUY/PASS actions. ACQUISITION exposes the full `(corp, company, offset)` action space directly via masking; CLOSING exposes all eligible companies directly; the acting player picks in a single shot, or passes. The old phase code on `main` is *only* a rule-intent reference — do not port its state machine.

**`PHASE_ACQ_OFFER` is the FI-priority resolution phase.** When a player or receivership corp attempts to acquire an FI-owned company and one or more higher-priority corps exist (OS has absolute priority at face value; remaining corps ordered by descending share price at high value), the engine pushes into `PHASE_ACQ_OFFER` and offers each higher-priority corp in turn the 2-action `{pass, buy}` choice. The offered corp's president makes the decision. Once all preempting candidates decline, control returns to the original acquirer and ACQUISITION resumes. ACQ_OFFER reuses the turn block's existing `active_corp` slot (= preempting corp being offered), `active_company` slot (= contested FI company), and `active_player` slot (= that corp's president); no new state fields are introduced.

**No per-ACQ "active entity" bookkeeping.** During ACQUISITION itself there is no active corp or active company — the player picks freely from the full masked action space, so `active_corp` and `active_company` sit at `-1`. The hidden "current offer pointer" state from the old engine is gone, and with it the need for any per-phase flags like `closing_company` / `dividend_corp` / `issue_corp` / `ipo_company`. The generic `active_corp` / `active_company` / `active_player` selectors cover every phase that still needs active-entity context (BID, IPO, ISSUE, DIVIDENDS, ACQ_OFFER); ACQUISITION and CLOSING leave them at `-1`.

## Code Conventions

- **Naming:** `corp_id`/`company_id`/`player_id` = indices; `CORPS[i]`/`PLAYERS[i]`/`COMPANIES[i]` = singletons; `PHASE_*` from `GamePhases` (engine) or `DPHASE_*` from `DecisionPhase` (model); `LOC_*` from `CompanyLocation`.
- **ALL game state access goes through entity handles.** This is non-negotiable, especially for phase handlers. Use `PLAYERS[i].get_cash(state)` / `PLAYERS[i].set_cash(state, x)`, `CORPS[c].get_share_price(state)`, `COMPANIES[i].get_location(state)`, etc. Do **not** import `LAYOUT` / `PLAYER_FIELDS` / `CORP_FIELDS` / `TURN_OFFSETS` / `COMPANY_OFFSETS` / `DECK_OFFSETS` from `core.state` outside the entity modules themselves. Do **not** index into `state._data` directly. Do **not** write new ad-hoc field accessors in phase code. The entity handles are the single source of truth for layout knowledge, and their invariants (dirty-mask invalidation, cached-star bookkeeping, ownership-location synchronization) are only correct when callers go through their published methods. Reaching around them is how state corruption enters the codebase.
- **Entity handle method surfaces are mostly finalized — ask before extending.** If you are implementing a phase handler and find yourself wanting a method that doesn't exist on an entity handle (e.g. "I need a `Corporation.net_treasury()` helper" or "I need a `Player.can_afford_share(corp_id)` check"), **stop and ask the user first**. Do not silently bolt on new cpdef/cdef methods to entity modules. Most of the accessors you need already exist; the ones that don't are design decisions the user wants to make deliberately, not drive-by additions from phase code. Same rule applies to modifying existing methods' semantics — check in before changing behavior.
- **`GameState` exposes only structural primitives.** `core/data.{pxd,pyx}` is data-only (static arrays + enums + normalization constants); there are no field-level helpers there. Modules that need static game data (company face values, synergy matrix, par-price tables, etc.) `cimport` the underlying arrays from `core.data` directly.
- **Pointer safety:** `nogil` functions take pointers/offsets. Every entity handle reads offsets directly from `LAYOUT` / `PLAYER_FIELDS` / `CORP_FIELDS` / `TURN_OFFSETS` / `COMPANY_OFFSETS` / `DECK_OFFSETS` (cimported from `core.state`) and caches nothing per-instance. Player and Corp use a small inline `_slot(field)` helper that computes `<section_offset> + id * <section_stride> + field` — Cython inlines it away in nogil hot paths. **This is the only code that should be cimporting those layout constants.**
- **Guard with `assert`, not silent fallbacks:** In Cython code, validate parameters and invariants with `assert` rather than `if ...: return`/`return False`/`return 0` style guards. Out-of-range IDs, inactive entities that should be active, malformed state — all of these should crash loudly in development. Silently propagating a default value hides the real bug at the call site and lets corrupt state spread. `assert` statements compile out under `python -O`, so there is zero overhead in production. Examples: `assert 0 <= player_id < state._num_players`, `assert corp.is_active(state)`. Include a descriptive f-string message so failures are debuggable. This rule does *not* apply to genuine business-logic branches (e.g. "this player can't afford the share, so the action is illegal") — those still return cleanly. The rule is about *defensive* checks for things that should never happen if callers are correct.

## Build Commands

> ⚠️ **Refactor in progress.** `setup.py` currently only builds `core/data.pyx`, `core/state.pyx`, `core/actions.pyx`, and `entities/*.pyx`. `core/driver.pyx`, `phases/*.pyx`, `mcts/mcts_core.pyx`, and any other root-level `.pyx` files are intentionally excluded and will not compile against the new state layout. The build target list grows as each refactor phase lands. For this period, ignore anything outside the build set unless explicitly told otherwise; tests, training, and benchmarks are expected to be broken until later phases.

**Python binary:** Always use `.venv/bin/python` (not `python` or `python3`). The venv may not be activated in the shell.

**Pyright:** Use `pyright` (system-installed at `/usr/bin/pyright`), NOT `.venv/bin/pyright`.

**Submodules:** When running commands in `submodules/18xx/`, use absolute paths or `cd /home/icebreaker/rss-az-cython2/submodules/18xx && ...` in a single Bash call. Do NOT `cd` into a submodule and forget — subsequent commands will run in the wrong directory.

```bash
# Build Cython extensions (required before running any Python code)
.venv/bin/python setup.py build_ext --inplace 2>&1 | grep -E "(warning|error)" || true

# Clean build artifacts
.venv/bin/python setup.py clean
```

`pytest tests/`, `python -m train`, and `setup.py benchmark` all require pieces of the engine that aren't building yet — don't expect them to run until the driver + phases + MCTS are back online.

**Warning-free builds:** No compiler warnings expected. If warnings appear, create a beads issue.

**Pyright errors:** Fix before moving on. Run `pyright <file>` via Bash for definitive results (auto-injected diagnostics can be stale). Note: if you encounter a Pyright error, don't ignore it just because it wasn't caused by your change — fix pre-existing errors to keep the codebase clean.

## Testing Approach

Most of the old test suite (`tests/conftest.py`, `tests/phases/`) has been deleted as part of the refactor. The surviving test code lives in `tests/games_18xx/` and currently doesn't import because its harness depends on `core.driver`. Expect to rewrite the test suite piece by piece as each refactor phase lands — there is no "all tests pass" gate to run against right now.

When adding new tests during the refactor:

- Prefer invariant-style assertions (cash conservation, share counts, ownership consistency) applied at every state transition over narrowly scoped unit tests for individual fields. See the old `tests/phases/conftest.py::assert_invariants` on `main` for the pattern.
- **When a test fails, assume the implementation is broken** until proven otherwise. Only "fix" a test after confirming the implementation is correct and the test setup was invalid.

### 18xx.games Replay Tests

The replay harness (`tests/games_18xx/`) validates our engine against completed games from 18xx.games by replaying every action and comparing state at phase boundaries. **Requires Ruby** — `extract_states.rb` runs as a subprocess. Key files: `extract_states.rb`, `action_parser.py`, `replay_harness.py`, `test_replay.py`.

This is currently broken because the harness imports `core.driver`. Once the new driver is in place, it comes back online.

**Known action-space differences** (carried over from the old engine — verify these still apply after rewrite): (1) cross-president ACQ transfers, (2) directly offering positive-income company closes in CLO. Check for these before investigating engine bugs if replay fails.

**Adding a game:** Export JSON → save to `tests/games_18xx/data/<id>.json` → add to the `@pytest.mark.parametrize` in `test_replay.py`.

## MCTS and Self-Play (deferred)

The old MCTS (`mcts/search.py`, `mcts/node.py`, `mcts/evaluator.py`, `mcts/mcts_core.pyx`) and training orchestration (`train/eval_server.py`, `train/self_play.py`, `train/main.py`) are all pre-refactor. They describe the target architecture but do not run against the new state/action layout. Read them as reference for the intended shape of the pipeline:

- Pure-Python AlphaZero MCTS (PUCT selection, Dirichlet root noise, A0GB greedy-backup value targets, subtree reuse via `StatePool`).
- Tanh per-player value head in `[-1, 1]`.
- Self-play workers (CPU, per-process GIL) talking to eval-server processes via shared memory; per-server uint64 bitmaps for lockfree request submission, per-worker `mp.Event` for done signaling; gather/scatter via Cython `nogil` memcpy in `mcts/mcts_core.pyx`.
- Replay buffer stores `(state tokens, legal mask, policy target, value target)`; value targets are A0GB leaves, policy targets are MCTS visit distributions. Under the new sparse design this becomes `(phase_id, action_ids[:num_legal], policy_targets[:num_legal], value_target)` — the "Transformer Refactor" section above covers the contract.

When these pieces are rebuilt, they will have to talk to the new token eval buffers, per-phase action indices, and sparse legal-action enumeration. The *mechanics* (shared memory layout, `StatePool`, A0GB, subtree reuse, graceful `q+Enter` shutdown) stay the same; the *interfaces* all change. Look at the files on `main` for the current implementation when rewriting, but don't blindly copy — the buffer layouts and action indices all differ.

## Key Files by Task

| Task | Primary Files | Secondary Files |
|------|---------------|-----------------|
| Layout / field offsets | `core/state.{pyx,pxd}`, `VECTORS.md` | `entities/*.pxd` |
| Static game data / synergies / CoO | `core/data.{pyx,pxd}` | `RULES.md` |
| Action encoding / decoding | `core/actions.{pyx,pxd}` | `nn/transformer.py::PHASE_ACTION_SIZES` |
| Entity field access | `entities/<entity>.{pyx,pxd}` | — |
| Model architecture | `nn/transformer.py` | "Transformer Refactor" section above |
| Rewriting driver / phases | (currently TBD) | old files on `main` + `RULES.md` |
| Phase implementation | entity handle methods + `core/actions.pyx::_enumerate_*` | `RULES.md`; ask user before adding handle methods |

---

# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

**Before working on any game logic**, read `RULES.md` — authoritative source for rules.

**Before working on state layout or entity handles**, read `VECTORS.md` — authoritative source for the compact state layout.

The refactor design docs (`transformers.md`, `sparse-refactor.md`) are large. Don't auto-load them into context — the "Transformer Refactor" section above pulls out the parts you need day-to-day. Dip into the full docs only if you need the fuller "why" behind a specific decision that isn't covered here.

## Agent Work Standards

Create beads issues for anything discovered that needs follow-up but is out of scope. **No insights or work should be lost.**

**Always include `--description="..."` when creating a `bd` issue.** A title alone is not enough — descriptions provide the context needed to pick up work later.

## Using Subtasks

Use subtasks (`bd create --parent=<id> --title="..." --type=task`) for related work. Results in dot-notation IDs: `abc.1`, `abc.2`. Use independent issues for unrelated bugs or cross-cutting concerns.

## Ad-hoc Test Scripts

Write ad-hoc scripts to the project `scratchpad` directory, not inline in Bash. Use `Edit` for iteration. Always prepend `PYTHONPATH=/home/icebreaker/rss-az-cython2` when running. This will save you a ton of tokens by making sure you don't have to write 100-line scripts from scratch on each invocation!

## Referencing the old implementation

The pre-refactor implementation lives on the `main` branch. It is a useful reference for:

- Game rule intent and edge cases (when `RULES.md` is ambiguous)
- The shape of the old phase handlers, driver loop, and MCTS plumbing
- Legality logic that needs to be ported into the new `_enumerate_*` helpers in `core/actions.pyx`

It is **not** a reference for: state layout, field offsets, action indices, or any struct/function signature. Those have all changed. Do not copy code verbatim from `main` — translate it into the new contract.

## Verification Before Closing

Right now there is no full "build + test" gate to run:

```bash
.venv/bin/python setup.py clean && .venv/bin/python setup.py build_ext --inplace 2>&1 | grep -E "(warning|error)" || true
```

That's the minimum. `pytest tests/` and `python -m train` do not currently run; don't pretend they do. As the refactor progresses and pieces come back online, re-add them to the verification checklist.

## Landing the Plane (Session Completion)

This branch is ephemeral and has no upstream — code is merged to `main` locally, not pushed. Session close looks like:

1. File issues for remaining work (`bd create`)
2. Run the build gate above
3. Update issue status (`bd close`, `bd update --status=in_progress`, etc.)
4. `git status` → `git add <files>` → `bd sync --from-main` → `git commit`

Follow the beads session-close protocol at the top of every session.
