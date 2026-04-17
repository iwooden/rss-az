# Transformer Architecture for Rolling Stock Stars

> **Refactor status.** This work lives on the `transformer-refactor` branch and has touched nearly every layer of the codebase — state representation, action space, model architecture, evaluator, and training loop. Phases 1-5 (compact state, action space, driver+phase handlers, transformer model, evaluator integration, and self-play + replay buffer rewrite) are landed. Multi-player-count training (Phase 6) is the remaining scoped work. Backward compatibility with the MLP model, old state vectors, and old action indices is not maintained; the `main` branch is preserved as a reference for rule intent only.
>
> **Training scope:** 3–5 players. The engine supports 2–6p for test compatibility, but all NN/MCTS/training code targets 3–5p only.

## Motivation

Replace the residual MLP (~4.1M params, flat state vector) with a transformer that treats each game entity as a token. Three key advantages:

1. **No state rotation.** The MLP requires rotating player data so the active player is always at slot 0. A transformer is permutation-equivariant on its input tokens - just mark the active player with a flag. Eliminates rotation logic in the evaluator and state construction.

2. **One model for 3-5p.** The MLP requires separate models for 3p/4p/5p because the input dimension changes. A transformer supports a variable number of player tokens natively.

3. **Entity-readout policy heads.** Instead of phase-specific MLP heads that output the full action vector, each entity token produces only its own action logits. Corp tokens output buy/sell logits, company tokens output auction logits, etc. Structurally encodes the right inductive bias.

## Token Decomposition

Each token is projected to a common `d_model` dimension via type-specific linear layers. Type discrimination is handled by the per-type projection (including its bias) — no shared type-embedding table. The pass token, which has no input features, is instead represented by a single learned `nn.Parameter` anchor (BERT `[CLS]`-style).

The authoritative per-token feature spec is in `token-data.md`. The table below summarizes token types and counts.

### Entity Tokens

| Token type | Count | Notes |
|------------|-------|-------|
| Player | N (3-5) | identity, cash, net worth, liquidity, income, turn order, has passed, owned shares, share buys/sells, round trips, presidencies, owned companies |
| Corporation | 8 | identity, active, receivership, passed ACQ_OFFER, shares, price index+value, pending move, cash, acq proceeds, income breakdown, total stars, president, owned companies |
| Company | 36 | identity, active company flag, location flags, ownership (corp/player/FI), adjusted income, static data (prices, stars, base income), 36-dim synergy |
| FI | 1 | cash, income, owned companies (36) |
| Market | 1 | 27 availability flags |
| Global | 1 | num players (3-slot one-hot), phase (8-slot one-hot), CoO (7-slot one-hot), end-card flipped, cards remaining |
| Invest | 1 | consecutive passes, buy/sell share price impacts per corp (8+8). Zeroed outside INVEST. |
| Auction | 1 | price index+value, high bidder (5-slot), starter (5-slot). Zeroed outside BID. |
| Dividend | 1 | 26 dividend-amount price impacts, dividend remaining (8 corp flags). Zeroed outside DIVIDENDS. |
| Issue | 1 | price impact, issue remaining (8 corp flags). Zeroed outside ISSUE. |
| PAR | 1 | per-par player cost (14), corp cash (14), issued shares (14). Zeroed outside IPO. |
| Acq Offer | 1 | offer price index+value, offer corp (8-slot), FI-company flag. Zeroed outside ACQ_OFFER. |
| Pass | 1 | No input features — single learned `nn.Parameter` anchor. Emits pass logit. |

**Total: 54 + N tokens** (57 for 3p, 58 for 4p, 59 for 5p)

### What disappears from the state vector

The current turn state has ~225 floats of context-dependent fields. Many exist because the MLP can't selectively attend to entities:

- **Active entity duplication** (~100 floats): `active_company` (36 one-hot + 5 scalars), `active_corp` (8 one-hot + 14 scalars + 36 owned_companies). The transformer can attend directly to the relevant entity token. Replaced by a single `is_phase_active` flag on the relevant corp/company token — the transformer already has the entity's full features via its token.
- **Auction slot info** (5 * N floats): Per-slot company features (stars, prices, income) duplicated from company data. Unnecessary when company tokens carry their own features.
- **Invest impacts** (16 floats): Buy/sell price impact per corp. Moved onto the dedicated Invest phase-context token.

### Phase-specific context tokens

Seven dedicated tokens carry phase-specific information that the model needs for decision-making, zeroed outside their respective phases. Some double as policy readout points. Full feature lists in `token-data.md`.

- **Invest token**: consecutive passes, buy/sell share price impacts per corp. Context for INVEST.
- **Auction token**: price index+value, high bidder, starter. Raises are read from this token during BID.
- **Dividend token**: 26 dividend-amount price impacts, corp-remaining flags. Dividend amount logits are read directly from this token.
- **Issue token**: price impact, corp-remaining flags. Issue logit is read from this token.
- **PAR token**: per-par player cost, resulting corp cash, resulting issued shares. Context for IPO (IPO logits are read from corp tokens, PAR provides shared context through attention).
- **Acq Offer token**: offer price index+value, offer corp, FI-company flag. Buy logit is read from this token during ACQ_OFFER.
- **Pass token**: no input features — a single learned `nn.Parameter` vector (BERT `[CLS]`-style anchor) that rides the residual stream and picks up phase-dependent context through attention. Emits the pass logit shared across all phases that use it.

## Transformer Architecture

**Implementation: `nn/transformer.py`** (~2.3M params at default config)

```
Input tokens (57 for 3p)
    ↓
Type-specific linear projections → d_model      (pass token: learned anchor only)
    ↓
L transformer blocks (pre-RMSNorm, multi-head self-attention + SwiGLU FFN)
    ↓
Output token representations (57 × d_model)
    ↓
Entity-readout policy heads (+ full ACQUISITION pair head) + value head
```

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| d_model | 128 | Scale to 256 once architecture validated |
| num_heads | 2 | Default in `TransformerConfig` |
| num_layers | 10 | Enough depth for multi-hop reasoning |
| ff_mult | 3.0 | SwiGLU FFN inner dim = ceil(ff_mult * d_model) |
| d_bilinear | 64 | Hidden width for the ACQUISITION pair-feature policy head |
| token_dim | 97 | Fixed width, zero-padded to uniform size across all token types. Sourced from `core.token_data.TokenDataSize.TOKEN_DIM` so the model and the Cython extractor can't drift out of sync. |

**No positional encoding over token order.** Permutation equivariance is a feature, not a bug. Token order is fixed for implementation convenience, but entity identity should come from explicit player/corp/company ID features (and the per-type projection, which already puts each token type in its own subspace of `d_model`), not from sequence position.

**Normalization:** RMSNorm (not LayerNorm). Drops the mean-centering step — only divides by root mean square. Simpler, slightly faster, no downside.

**FFN:** SwiGLU (`W_down(SiLU(W_gate(x)) * W_up(x))`) instead of standard GELU FFN. Learned gating gives finer control over information flow. Three weight matrices (all bias=False) instead of two; the current config uses `ff_mult=3.0`.

### Type-specific input projections

All projections take the full zero-padded `token_dim` input (no per-type feature slicing). Weights for always-zero feature positions are inert; the simplicity is worth the ~2% extra params.

```python
# All projections: nn.Linear(token_dim, d_model) — one per entity type
# Each projection's bias already provides a per-type shift in d_model space,
# so no shared type-embedding table is needed for discrimination.
self.player_proj  = nn.Linear(token_dim, d_model)
self.corp_proj    = nn.Linear(token_dim, d_model)
# ... (12 total, one per non-pass token type)

# Pass token has no input features. Its representation is a single learned
# anchor vector (BERT [CLS]-style) initialized with trunc_normal_(std=0.02).
self.pass_embed = nn.Parameter(torch.empty(d_model))
```

## Policy Head: Entity-Readout

Each entity token produces the action logits for actions related to that entity. All logits are concatenated into a flat action vector and masked.

### INVEST phase example

```
Pass token    →  small head  →  [pass]                          (1 logit)
Company 0     →  small head  →  [auction_offset_0 ... _14]      (15 logits)
Company 1     →  small head  →  [auction_offset_0 ... _14]      (15 logits)
  ...
Company 35    →  small head  →  [auction_offset_0 ... _14]      (15 logits)
Corp 0        →  small head  →  [buy, sell]                     (2 logits)
Corp 1        →  small head  →  [buy, sell]                     (2 logits)
  ...
Corp 7        →  small head  →  [buy, sell]                     (2 logits)
```

The legal action mask zeros out non-auctionable companies, corps you can't buy/sell, etc.

**Key change from current design:** Auction actions are indexed by `company_id × 15 + offset` instead of `slot × 15 + offset`. This eliminates the slot indirection (`get_auction_company_for_slot`) entirely. The action semantics become "auction company X at price offset Y" rather than "auction slot N at price offset Y".

### Other phases

| Phase | Entity → logits |
|-------|----------------|
| BID | **Pass token** → [pass] (1, = leave auction); **Auction token** → [raise_0 ... raise_13] (14) |
| ACQUISITION | **Pass token** → [pass] (1); corp×company pair head → [corp_0_company_0_offset_0 ...] (14976) |
| ACQ_OFFER | **Pass token** → [pass] (1); **Acq Offer token** → [buy] (1) |
| CLOSING | **Pass token** → [pass] (1); Company 0..35 → [close] (36, most masked) |
| DIVIDENDS | **Dividend token** → [div_0 ... div_25] (26) |
| ISSUE | **Pass token** → [pass] (1); **Issue token** → [issue] (1) |
| IPO | **Pass token** → [pass] (1); Corp 0..7 → [par_0 ... par_13] (112, most masked) |

INVEST, CLOSING, and IPO use multi-entity readout (company/corp tokens + pass). ACQUISITION builds a shared representation for each `(corp, company)` pair, then reads 52 logits per pair (51 price offsets + FI buy). ACQ_OFFER uses Pass (pass) + Acq Offer token (buy). DIVIDENDS reads solely from its dedicated phase token. BID splits between Pass (leave the auction) and Auction (raises).

### Action space size

The new action layout (for any player count):

```
INVEST:      1 (pass) + 36*15 (auction) + 8*2 (buy/sell) = 557
BID:         15 (pass + 14 raises; pass = leave the auction)
ACQUISITION: 14977 (pass + 8*36*52 corp×company×offset/action)
ACQ_OFFER:   2 (pass + buy)
CLOSE:       37 (36 company close + pass)
DIV:         26 (dividend amounts)
ISSUE:       2 (issue + pass)
IPO:         113 (8*14 corp par prices + pass)
```

Per-phase action indices (no global action vector). The largest phase is ACQUISITION at 14,977 actions. All sizes are **fixed across player counts**.

ACQUISITION and ACQ_OFFER are distinct phases from the model's perspective (separate one-hot IDs in the Global token's phase encoding), even though the game engine treats ACQ_OFFER as a sub-phase of ACQUISITION. This is the same pattern as INV/BID today — BID is a "sub-phase" of INVEST in the game rules, but the model sees them as distinct phases.

**ACQ flow:**
- **ACQUISITION** → the acting player picks `(corp, company, offset/action)` in one shot, or passes. No offer-buffer indirection: the engine exposes the full `(corp, company, offset)` space via masking and the player chooses directly from it. FI fixed-price purchases fold into the same pair space as the 52nd per-pair option (`FI_BUY`). Receivership auto-acquisitions are forced actions — the driver resolves them without a decision point. During ACQUISITION there is no "active" corp or company; `turn.active_corp` and `turn.active_company` stay at `-1`.
- **ACQ_OFFER** (conditional) → entered whenever a player or receivership corp attempts to acquire an FI-owned company AND one or more higher-priority corps exist. Priority order: OS first at face value, remaining corps ordered by descending share price at high value. The engine offers each higher-priority corp in turn the 2-action `{pass, buy}` choice, with the offered corp's president as the active player. Once all preempting candidates decline, control returns to the original acquirer and ACQUISITION resumes. ACQ_OFFER reuses the turn block's existing `active_corp` / `active_company` / `active_player` selectors (= preempting corp being offered, contested FI company, and that corp's president); no new state fields are introduced.

The current transformer implementation uses a fully joint ACQUISITION head: `1 + 8 × 36 × 52 = 14,977` actions. `ACQ_OFFER` remains a separate phase because FI preemption is a distinct decision point, often for a different player (the preempting corp's president) than the original acquirer.

**All 8 decision phases:** INV, BID, ACQUISITION, ACQ_OFFER, CLO, DIV, ISS, IPO.

### Head architecture

Entity-readout heads are small linear layers (1-2 hidden layers) applied per-token:

```python
# Shared across all company tokens (weight sharing = parameter efficiency)
self.company_auction_head = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, 15),  # 15 auction offsets
)
self.company_close_head = nn.Linear(d_model, 1)  # per-company close logit

# Shared across all corp tokens
self.corp_trade_head = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, 2),  # buy, sell
)

# Pass token → single pass logit (shared across all phases that use it; BID's pass = leave the auction)
self.pass_head = nn.Linear(d_model, 1)

# Phase-specific context tokens → policy logits (read directly from token)
self.auction_raise_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 14))  # raise amounts
self.dividend_head  = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 26))
self.issue_head     = nn.Linear(d_model, 1)  # issue (pass comes from Pass token)
self.acq_offer_head = nn.Linear(d_model, 1)  # buy logit (pass from pass_head)

# ACQUISITION: shared corp/company pair features → 52 logits per pair
self.acquisition_corp_proj    = nn.Linear(d_model, dk)
self.acquisition_company_proj = nn.Linear(d_model, dk)
self.acquisition_pair_head    = nn.Sequential(
    nn.Linear(3 * dk, dk),
    nn.GELU(),
    nn.Linear(dk, 52),  # 51 price offsets + FI buy
)

# IPO: per-corp readout directly from corp tokens
self.corp_ipo_head = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, 14),
)
```

Heads shared across tokens of the same type means very few parameters. Phase-specific tokens carry both the context and the decision — clean separation of concerns.

## Value Head

Player tokens produce per-player values directly:

```python
self.value_head = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, 1),
    nn.Tanh(),
)

# Applied to each player token → v_i ∈ [-1, 1]
# Output: [v_player0, v_player1, ..., v_playerN]
```

Since player tokens aren't ordered by slot, the output values are already in canonical order (player 0's token → v_0, etc.). **No un-rotation needed.**

## State Simplification

The transformer refactor rebuilds `GameState` around a single compact `int16` array used only by the game engine, with `get_token_data()` as the sole engine→NN interface. The full state layout (section offsets, player/corp/turn blocks, company tracking, action-space encoding) is authoritative in [`VECTORS.md`](VECTORS.md); the pipeline it participates in is:

```
GameState (int16, 469 slots at 3p)  ── engine-only, raw values ──
                     ↓
           get_token_data(state, buffer)   ← Cython, nogil
                     ↓
           eval_buffer: (batch, num_players + 54, 97)  float32
                     ↓
           Type-specific projections → d_model → transformer trunk
```

**What was deleted compared to the pre-refactor MLP layout:**
- **Visible/hidden split**: No more `get_visible_size()`, `get_hidden_size()`, separate offset tracking. One compact array.
- **One-hot encodings in state**: Phase, CoO, turn order, price indices stored as plain integers. One-hot fan-out happens inside `get_token_data()`, not in the backing buffer.
- **Normalization in setters**: No more `cash / CASH_DIVISOR` on write + `cash * CASH_DIVISOR + 0.5` on read. Raw values stored directly; token extraction applies fixed divisors from `core/data.pxd`.
- **Duplicated entity features**: `active_company` scalar block, `active_corp` scalar block, and auction-slot feature duplication are gone. Instead, `active_player` / `active_corp` / `active_company` are single int16 selector slots in the turn block, and the transformer attends to the relevant entity's own token for its features.
- **State rotation**: Gone entirely — the compact state is never rotated; the value head reads from player tokens in canonical order.

### `get_token_data()` interface

```cython
cpdef void get_token_data(GameState state, float[:, ::1] buffer)

cpdef void get_token_data_batch(
    list state_arrays, int num_players, float[:, :, ::1] buffer,
)
```

Implementation lives in `core/token_data.{pyx,pxd}`. Feature spec: `token-data.md`. Each `_fill_*_token()` helper reads from the compact state and writes raw/lightly-normalized features into the buffer inside a single `nogil` block. Token order is fixed: `[players..., corps..., companies..., FI, market, global, invest, auction, dividend, issue, par, acq_offer, pass]`.

Both entries accept C-contiguous `float32` memoryviews sized to at least `(num_players + 54, TOKEN_DIM)` (single) or `(n, num_players + 54, TOKEN_DIM)` (batch). `get_token_data_batch` reuses a single scratch `GameState` across rows via `rebind`, amortizing Python dispatch over the whole batch — this is the path the evaluator and eval server use on their hot paths. A small GIL-held prologue runs `refresh_player_cache_if_dirty` for each player before the nogil fill body, so the fill itself can read cached net-worth / liquidity / income slots directly.

Static features (synergies, face values, etc.) are written every call — it's a straight memcpy and the simplicity is worth more than the minor bandwidth savings. Phase-specific tokens are left at the zeroed default when the current engine phase does not match.

### Eval buffer sizing

The eval buffer is `(batch_size, num_tokens, token_dim)`:
- `num_tokens = num_players + 54` — 57 (3p), 58 (4p), 59 (5p)
- `token_dim = 97` (all token types zero-padded to the same width; `TokenDataSize.TOKEN_DIM`)
- Total per sample: 57 × 97 = 5,529 floats (~21.6 KiB) for 3p

The rectangular layout enables clean GPU operations. All type-specific projections take the same `token_dim` input (no per-type slicing needed).

### Replay buffer storage

The replay buffer stores training examples. With per-phase action indices, each example contains:
- **Token data**: the eval buffer contents (or the compact state + `get_token_data()` at training time)
- **Phase ID**: which phase this decision was in (determines action space)
- **Legal action mask**: per-phase (max 14,977 for ACQUISITION)
- **Policy target**: MCTS visit distribution (per-phase, same size as mask)
- **Value target**: A0GB values (N floats)

Option A: store the compact state and re-extract tokens at training time. Smallest storage.
Option B: store the eval buffer directly. Faster training (no re-extraction), larger storage.

Option B is simpler for a very small-scale prototype, but expensive at current training defaults: a `(57, 97)` float32 token buffer is ~22 KiB per sample, so a 500k-sample replay buffer costs ~11 GiB for tokens alone — before per-sample phase ids, legal-action masks, sparse policy targets, and value targets. Option A is the more realistic long-term default.

## Impact on Training & MCTS Pipeline

### What changes

- **GameState**: Compact state array, no visible/hidden split. New `get_token_data()` method fills eval buffers.
- **Eval buffers**: `(batch, num_tokens, token_dim)` rank-3 tensor in shared memory. All token types use the same padded width.
- **Action layout**: Per-phase action indices (max 14,977 for ACQUISITION). Update `actions.pyx` layout and mask generation.
- **Evaluator**: Remove state rotation. Remove un-rotation of values. Simpler.
- **Model interface**: `forward(x, phase_ids) → (policy_logits, values)` where `x` is `(batch, num_tokens, token_dim)` and `phase_ids` is `(batch,)`. Policy logits padded to `MAX_ACTION_SIZE` (14,977), -1e9 beyond phase action range.

### Game engine simplifications

Four major pieces of action-space indirection in the MLP-era engine are **removed** under the refactor. They were engine/training choices to keep the MLP's action space small; the transformer's entity-readout eliminates the need for them, so the new engine drops the indirection outright:

- **Auction slots** (deleted). The old engine mapped auctionable companies to positional slots (0, 1, ..., N-1) so the action space scaled with player count. Auction actions are now company-indexed: `company_id × 15 + offset`. The slot machinery (`get_auction_company_for_slot()` and friends) is gone; the mask just enables the 15 logits for each auctionable company directly.

- **Closing offer buffer** (deleted). The old engine pre-generated close offers into a hidden buffer, sorted by priority, and presented them one-by-one with CLOSE/PASS actions. The new engine exposes all eligible companies directly: in player order, mask all valid closes, let the model pick one company to close or pass, repeat until pass, then move to the next player. Mandatory closes for negative-income-and-cash corps still happen at the end as forced actions.

- **Acquisition offer buffer** (deleted). The old engine pre-generated `(corp, company)` acquisition offers into a hidden buffer, sorted by priority (OS→FI, Corp→FI, Corp→Corp, Corp→Player), and presented them one-by-one. The new engine scores the full `(corp, company, offset/action)` space directly in `ACQUISITION` — the acting player picks the tuple in one shot, or passes. FI fixed-price purchases fold into the same pair space as the 52nd per-pair option (`FI_BUY`); there is no separate "FI buy" sub-flow. Receivership auto-acquisitions remain as forced actions resolved by the driver. FI *priority* preemption is no longer achieved by careful offer ordering; it is handled explicitly by a dedicated `PHASE_ACQ_OFFER` decision phase (see the *ACQ flow* subsection earlier in this document). The hidden "current offer pointer" state is gone entirely, and with it the need for any per-phase active-entity flags (`closing_company` / `dividend_corp` / `issue_corp` / `ipo_company`) — the generic `active_corp` / `active_company` / `active_player` turn slots cover every phase that still needs active-entity context (BID, IPO, ISSUE, DIVIDENDS, ACQ_OFFER). ACQUISITION and CLOSING sit at `-1` for both active slots.

- **IPO/PAR phase merge** (done). The old engine split IPO into two phases — IPO (select corp) then PAR (select par price). Each corp token now produces 14 par-price logits directly; the player picks "corp X at par price Y" in one action. The PAR phase is eliminated entirely, simplifying the phase graph and removing a state transition.

All four simplifications follow the same pattern: the MLP needed a small, fixed action space, so the engine compressed entity-level decisions into slot/offer indirection or multi-step sequences. The transformer's entity-readout makes the direct formulations cheap, so the engine drops the indirection.

### What stays the same (conceptually)

- MCTS search logic (PUCT, A0GB, subtree reuse) — code needs rewrite for sparse policy
- Self-play worker architecture (shared memory, eval servers — buffer layout changes but architecture doesn't)
- Training loop (policy CE + value MSE)
- Game rules — driver and phase handlers are rewritten but encode the same game logic

## Open Questions

1. ~~**Compact state layout**~~: **Done.** Single int16 array, raw values, entity handles with module-level layout constants. See `VECTORS.md` for the authoritative layout.

2. ~~**`get_token_data()` feature lists / implementation**~~: **Done.** See `token-data.md` for the per-token feature spec and `core/token_data.{pyx,pxd}` for the implementation (nogil fill + batched `get_token_data_batch`).

3. ~~**Head dispatch efficiency**~~: **Resolved.** The model emits per-row gathered logits of shape `(B, K_MAX)` — never the dense `(B, 14977)` ACQUISITION pad — by gathering against each row's legal-action list inside `_policy_forward`. Callers pass precomputed per-phase row indices (`phase_indices`) so the dispatch uses `index_select` / `index_copy_` with no H↔D sync.

4. **Batching with variable token counts** (Phase 6): For 3-5p training, batches contain games with different numbers of player tokens (57-59). Options: pad to 59 and use attention mask, or batch by player count. Padding is simpler; batching by count is more efficient.

5. **Replay storage strategy**: Currently Option A — the replay buffer stores compact state + phase id + enumerated legal ids + sparse policy/value targets, and re-runs `get_token_data_batch` at training time. Option B (pre-extracted token buffers) is not planned at current buffer sizes.

6. **Inference speed**: Validated via self-play throughput on CPU + ROCm; per-layer `F.scaled_dot_product_attention` fuses cleanly under `torch.compile` with `mark_unbacked` applied to each per-phase row-index tensor so recompile counts stay bounded.

## Implementation Phases

### Phase 1: Compact state + token extraction (core/) ✅
- Redesign `GameState` as a single compact `int16` array (no visible/hidden split) — **done.** `core/state.{pxd,pyx}` reduced to structural primitives + entity-handle delegation; player block now contains all per-player tracking incl. share buys/sells/has_passed.
- Delete one-hot encoding, normalization-on-write, entity duplication — **done.**
- Reduce `core/data.{pxd,pyx}` to pure data + constants — **done.** All field-level accessors removed; the file exposes only the static arrays (company/corp/market/CoO/par tables, synergy matrix), the shared enums (`GameConstants`, `GamePhases`, `DecisionPhase`, `CorpIndices`, `ActionSize`), and the normalization divisors used by token extraction. Computational helpers that used to live here (synergy aggregation, required stars, cost-of-ownership lookup, par-price validity) now live as private cdef functions in the entity that uses them.
- Update entity handles (Player, Corp, Company, FI, Market, Deck, TurnState) for the new offset layout — **done.** Every handle is fully stateless: no per-instance offset cache, no `initialize()` step, no normalization-on-read. Each accessor reads its slot inline from the module-level `LAYOUT` / `PLAYER_FIELDS` / `CORP_FIELDS` / `TURN_OFFSETS` / `COMPANY_OFFSETS` / `DECK_OFFSETS` / `FI_OFFSETS` constants, so the singletons in `PLAYERS` / `CORPS` / `COMPANIES` are reused with any `GameState` at any player count. Company ownership is read from the shared `company_locations` / `company_owner_ids` arrays — the per-player and per-corp `owned_companies` bitmaps are gone.
- Implement `get_token_data()` in Cython — **done.** `core/token_data.{pyx,pxd}` fills a `(num_players + 54, 97)` float32 memoryview inside a single nogil block, with a batched `get_token_data_batch` variant that amortizes per-state Python dispatch via `GameState.rebind`. `TokenDataSize.TOKEN_DIM` is the single source of truth for the padded width shared with `nn/transformer.py`.

### Phase 2: Action space refactor (core/actions.pyx) ✅
- **Done.** Per-phase action indices, company-indexed auctions/closes, full ACQUISITION head, merged IPO.
- 8 decision phases, all `_enumerate_*` helpers implemented with real mask logic.
- Deterministic enumeration order, `MAX_LEGAL_ACTIONS=256` overflow asserts.
- Import-time roundtrip assert in `core/actions.pyx` catches drift between `encode_*` arithmetic and the `ActionSize` enum in `core/data.pxd`.

### Phase 2b: Driver + phase handlers (core/driver.pyx, phases/*.pyx) ✅
- **Done.** `core/driver.{pyx,pxd}` rewritten: game-loop dispatch, forced-action auto-chain (single legal action ⇒ auto-dispatch), all 8 decision phases + 3 automated phases wired. `step_mode` flag pauses after each decision point for tests / replay tooling.
- All 11 phase handlers in `phases/` implemented: invest, bid, acquisition, acq_offer, closing, dividends, income, issue, ipo, wrap_up, end_card.
- Cross-president acquisition offers supported via `acq_same_president` flag + `PHASE_ACQ_OFFER`.
- Optional `history` list records `(state._array.copy(), phase_id, action_id)` tuples before each mutation — used by replay / test scaffolding.

### Phase 3: Transformer model (nn/) ✅
- **Done.** See `nn/transformer.py` (~2.37M params at `TransformerConfig()` defaults).
- Pre-RMSNorm transformer blocks with SwiGLU FFN; attention via packed QKV + `F.scaled_dot_product_attention` (traces cleanly under `torch.compile`, unlike `nn.MultiheadAttention`).
- Type-specific projections (uniform `token_dim=97` input), entity-readout heads for every decision phase.
- Full ACQUISITION pair head (corp × company × 52) and direct per-corp IPO head (merged IPO+PAR).
- Per-phase head dispatch uses precomputed per-phase row indices (`phase_indices`) with `index_select` / `index_copy_`, so the policy gather has no H↔D sync. All callers (trainer, evaluator, eval server) build them on the host before the H→D copy.
- `forward(x, action_ids, n_legals, phase_indices)` returns `(policy_logits, values)` where `policy_logits` has shape `(B, K_MAX)` gathered against each row's legal-action slice — the trainer, evaluator, and eval server never allocate anything as wide as the full 14,977-action ACQUISITION output.
- Smoke test covers all 8 phases: shapes, finite logits inside the legal slice, `-1e9` tail, values ∈ [-1, 1].

### Phase 4: Evaluator integration (mcts/evaluator.py) ✅
- **Done.** `BaseEvaluator` + `NNEvaluator` (in-process) + `RemoteEvaluator` (shared-mem IPC, in `train/eval_server.py`) speak token buffers end-to-end.
- Preallocated pinned-host + device scratch tensors (tokens / phase_ids / action_ids / n_legals), grown in powers of two, aliased on CPU devices so no H→D copy runs locally.
- `fill_token_buffer_batch` routes to `get_token_data_batch` for a single Cython entry per batch.
- No state rotation, no value un-rotation — the model emits values in canonical player order directly from player tokens.
- GPU-side gather + softmax live inside the model itself; callers receive sparse `(n_legal,)` priors over the enumerated legal list plus canonical-order values.
- MCTS search (`mcts/search.py`), persistent-tree state, and subtree reuse rebuilt against the sparse token/phase contract; replay buffer (`train/replay_buffer.py`) stores per-sample compact state + phase id + legal-id list + sparse policy target + canonical value target.

### Phase 5: Training loop updates (train/) ✅
- **Done.** `train/main.py`, `train/self_play.py`, `train/eval_server.py`, `train/trainer.py`, `train/replay_buffer.py`, `train/analyze_game.py`, `train/tournament.py` all ported to sparse / token / per-phase-id contract.
- Shared-memory eval server carries token buffers, phase ids, sparse legal ids, and n_legals per leaf; GPU gathers logits against the sparse list and returns `(n_legal, num_players)` priors+values slices to each worker.
- Self-play + training end-to-end runs: eval-server workers, mixed-phase batches, replay buffer ingest, and policy-CE + value-MSE training all validated on 3p.
- State rotation, dense legal masks, and the old `(B, dense_action_size)` logit tensors are fully removed from the pipeline.

### Phase 6: Multi-player-count training (3-5p) — pending
- Variable player token count (57–59 tokens); choose between pad-to-max+attention-mask and bucket-by-player-count.
- Single model for 3-5p; requires generalizing the `TransformerConfig.num_players` check in `NNEvaluator.__init__` and the per-player masking in the value head consumer.
- Self-play orchestration needs per-worker player-count selection.
