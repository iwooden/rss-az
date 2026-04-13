# Transformer Architecture for Rolling Stock Stars

> **WARNING: Breaking refactor.** This work lives on the `transformer-refactor` branch and touches nearly every layer of the codebase — state representation, action space, model architecture, evaluator, and training loop. Backward compatibility with the MLP model, old state vectors, and old action indices is not maintained. Most existing tests will fail until the refactor is complete and tests are updated. The `main` branch is untouched and safe.
>
> **Training scope:** 3–5 players. The engine supports 2–6p for test compatibility, but all NN/MCTS/training code targets 3–5p only.

## Motivation

Replace the residual MLP (~4.1M params, flat state vector) with a transformer that treats each game entity as a token. Three key advantages:

1. **No state rotation.** The MLP requires rotating player data so the active player is always at slot 0. A transformer is permutation-equivariant on its input tokens - just mark the active player with a flag. Eliminates rotation logic in the evaluator and state construction.

2. **One model for 3-5p.** The MLP requires separate models for 3p/4p/5p because the input dimension changes. A transformer supports a variable number of player tokens natively.

3. **Entity-readout policy heads.** Instead of phase-specific MLP heads that output the full action vector, each entity token produces only its own action logits. Corp tokens output buy/sell logits, company tokens output auction logits, etc. Structurally encodes the right inductive bias.

## Token Decomposition

Each token is projected to a common `d_model` dimension via type-specific linear layers, then augmented with a learned type embedding.

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
| Pass | 1 | No input features — type embedding only. Emits pass logit. |

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
- **Pass token**: no input features — purely learned type embedding. Emits the pass logit shared across all phases that use it.

## Transformer Architecture

**Implementation: `nn/transformer.py`** (~2.3M params at default config)

```
Input tokens (57 for 3p)
    ↓
Type-specific linear projections → d_model
    + learned type embeddings
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
| token_dim | 63 | Fixed width, zero-padded to uniform size across all token types. |

**No positional encoding over token order.** Permutation equivariance is a feature, not a bug. Token order is fixed for implementation convenience, but entity identity should come from explicit player/corp/company ID features or ID embeddings plus the type embeddings, not from sequence position.

**Normalization:** RMSNorm (not LayerNorm). Drops the mean-centering step — only divides by root mean square. Simpler, slightly faster, no downside.

**FFN:** SwiGLU (`W_down(SiLU(W_gate(x)) * W_up(x))`) instead of standard GELU FFN. Learned gating gives finer control over information flow. Three weight matrices (all bias=False) instead of two; the current config uses `ff_mult=3.0`.

### Type-specific input projections

All projections take the full zero-padded `token_dim` input (no per-type feature slicing). Weights for always-zero feature positions are inert; the simplicity is worth the ~2% extra params.

```python
# All projections: nn.Linear(token_dim, d_model) — one per entity type
self.player_proj  = nn.Linear(token_dim, d_model)
self.corp_proj    = nn.Linear(token_dim, d_model)
# ... (12 total, one per non-pass token type)
# Pass token has no input features — its initial representation is just its type embedding

# Learned type embeddings (added after projection)
self.type_embed = nn.Embedding(13, d_model)  # player/corp/company/fi/market/global/invest/auction/dividend/issue/par/acq_offer/pass
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

The transformer refactor enables a fundamental simplification of `GameState`: eliminate the visible/hidden split entirely. The state becomes a single compact array used only by the game engine. A new `get_token_data()` method is the sole interface between the engine and the NN.

### Current design (MLP)

```
GameState: [visible state (1109 floats, 3p)] [hidden state (1271 floats)]
                     ↓
           State rotation (for active player)
                     ↓
           Flat float32 vector → MLP input
```

The visible state contains one-hot encodings, normalized values, duplicated entity features, and player-rotated data — all to make it consumable by the MLP directly.

### New design (transformer)

```
GameState: [compact state (~500-800 floats, estimated)]
                     ↓
           get_token_data(eval_buffer_slice)   ← Cython, fast
                     ↓
           eval_buffer: (batch, num_tokens, token_dim)
                     ↓
           Eval server routes token data → type-specific projections → d_model
```

**What gets deleted from the state array:**
- **Visible/hidden split**: No more `get_visible_size()`, `get_hidden_size()`, separate offset tracking. One compact array.
- **One-hot encodings**: Phase, CoO, turn order, price indices stored as plain integers. The projection layers (or small embeddings) handle encoding.
- **Normalization in setters**: No more `cash / CASH_DIVISOR` on write + `cash * CASH_DIVISOR + 0.5` on read. Raw values stored directly. Projection layers learn appropriate scaling (or apply fixed divisors).
- **Duplicated entity features**: active_company (41 floats), active_corp (58 floats), auction slot info (5*N floats) — all gone. `is_phase_active` flags on entity tokens replace them.
- **State rotation**: Gone entirely — no visible state to rotate.

### `get_token_data()` interface

```cython
cdef void get_token_data(GameState state, float* buffer, int num_tokens, int token_dim) nogil:
    """Fill eval buffer with per-token features for NN input.
    
    buffer shape: (num_tokens, token_dim), zero-initialized.
    
    Token order is fixed: [players..., corps..., companies..., FI, market,
                           global, invest, auction, dividend, issue, par,
                           acq_offer, pass]
    """
```

Implementation lives in `core/token_data.{pyx,pxd}`. Feature spec: `token-data.md`. Each `_fill_*_token()` helper reads from the compact state and writes raw/lightly-processed features into the buffer. Static features (synergies, face values) are written every time — it's just memcpy and the simplicity is worth more than the minor bandwidth savings.

### Eval buffer sizing

The eval buffer is `(batch_size, num_tokens, token_dim)`:
- `num_tokens`: 57 for the 3p prototype
- `token_dim`: 63 for the 3p prototype (all token types zero-padded to the same width)
- Total per sample: 57 × 63 = 3,591 floats (~14.0KB)
- Compare to current visible state: 1,109 floats (~4.4KB)

~3x larger per sample, but still small in absolute terms. The rectangular layout enables clean GPU operations. All type-specific projections take the same `token_dim` input (no per-type slicing needed).

### Replay buffer storage

The replay buffer stores training examples. With per-phase action indices, each example contains:
- **Token data**: the eval buffer contents (or the compact state + `get_token_data()` at training time)
- **Phase ID**: which phase this decision was in (determines action space)
- **Legal action mask**: per-phase (max 14,977 for ACQUISITION)
- **Policy target**: MCTS visit distribution (per-phase, same size as mask)
- **Value target**: A0GB values (N floats)

Option A: store the compact state and re-extract tokens at training time. Smallest storage.
Option B: store the eval buffer directly. Faster training (no re-extraction), larger storage.

Option B is simpler for a very small-scale prototype, but expensive at current training defaults: with `57 × 63` float token buffers and `buffer_capacity=500_000`, token storage alone is ~6.6 GiB, and total replay storage is materially higher once masks, policy targets, and value targets are included, especially with a 14,977-action ACQUISITION head. If we start with Option B, we should also reduce buffer capacity. Option A is the more realistic long-term default.

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

2. ~~**`get_token_data()` feature lists**~~: **Spec'd.** See `token-data.md` for the authoritative per-token feature spec. Implementation in `core/token_data.{pyx,pxd}` — still pending.

3. **Batching with variable token counts**: For 3-5p training, batches contain games with different numbers of player tokens (57-59). Options: pad to 59 and use attention mask, or batch by player count. Padding is simpler; batching by count is more efficient.

4. **Replay storage strategy**: Option B (store token buffers directly) is convenient for prototyping but probably requires a much smaller replay buffer. Option A (store compact state and re-extract tokens) is likely the long-term path.

5. **Inference speed**: Need to benchmark transformer vs MLP for the sequence lengths we're dealing with (~57-59 tokens for 3-5p). Should be fine, but verify with `torch.compile`.

6. **Head dispatch**: The forward pass has 8 phase-specific dispatch paths. The current implementation uses a full ACQUISITION pair head plus direct per-corp IPO readout. Need to keep mixed-phase batches efficient despite the padded `(B, 14977)` output interface.

## Implementation Phases

### Phase 1: Compact state + token extraction (core/)
- Redesign `GameState` as a single compact array (no visible/hidden split) — **done in `09e5048`** (`core/state.{pxd,pyx}` reduced to structural primitives + entity-handle delegation; player block now contains all per-player tracking incl. share buys/sells).
- Delete one-hot encoding, normalization-on-write, entity duplication — **done in `09e5048`**.
- Reduce `core/data.{pxd,pyx}` to pure data + constants — **done.** All field-level accessors removed; the file exposes only the static arrays (company/corp/market/CoO/par tables, synergy matrix), the shared enums (`GameConstants`, `GamePhases`, `CorpIndices`), and the normalization divisors used by token extraction. Computational helpers that used to live here (synergy aggregation, required stars, cost-of-ownership lookup, par-price validity) now live as private cdef functions in the module that uses them.
- Update entity handles (Player, Corp, Company, FI, Market, Deck, TurnState) for the new offset layout — **done.** Every handle is fully stateless: no per-instance offset cache, no `initialize()` step, no normalization-on-read. Each accessor reads its slot inline from the module-level `LAYOUT` / `PLAYER_FIELDS` / `CORP_FIELDS` / `TURN_OFFSETS` constants, so the singletons in `PLAYERS` / `CORPS` / `COMPANIES` are reused with any `GameState` at any player count. Company ownership is read from the shared `company_locations` / `company_owner_ids` arrays — the per-player and per-corp `owned_companies` bitmaps are gone.
- Implement `get_token_data()` in Cython — fills eval buffer from compact state. **Pending** — this is the next chunk of Phase 1 work.
- Target: 3p only

### Phase 2: Action space refactor (core/actions.pyx) ✅
- **Done.** Per-phase action indices, company-indexed auctions/closes, full ACQUISITION head, merged IPO.
- 8 decision phases, all `_enumerate_*` helpers implemented with real mask logic.
- Deterministic enumeration order, `MAX_LEGAL_ACTIONS=256` overflow asserts.

### Phase 2b: Driver + phase handlers (core/driver.pyx, phases/*.pyx) ✅
- **Done.** `core/driver.{pyx,pxd}` rewritten: game-loop dispatch, forced-action auto-chain, all 8 decision phases + 3 automated phases wired.
- All 11 phase handlers in `phases/` implemented: invest, bid, acquisition, acq_offer, closing, dividends, income, issue, ipo, wrap_up, end_card.
- Cross-president acquisition offers supported via `acq_same_president` flag + `PHASE_ACQ_OFFER`.

### Phase 3: Transformer model (nn/) ✅
- **Done.** See `nn/transformer.py` (~2.3M params, `TransformerConfig` defaults)
- Pre-RMSNorm transformer blocks with SwiGLU FFN
- Type-specific projections (uniform `token_dim` input), entity-readout heads
- Full ACQUISITION pair head and direct per-corp IPO head
- Smoke test passes: all 8 phases produce correct shapes, values in [-1,1]
- Target: 3p only, all dimensions parameterized via `TransformerConfig`

### Phase 4: Evaluator integration (mcts/evaluator.py)
- Eval buffers: `(batch, num_tokens, token_dim)` in shared memory
- Remove state rotation, remove value un-rotation
- Verify MCTS works end-to-end
- **Blocked on:** `get_token_data()` (Phase 1 pending item) + MCTS rewrite

### Phase 5: Training loop updates (train/)
- Update replay buffer for per-phase action indices + token data storage
- Update shared memory buffer layout
- Verify self-play + training loop works
- **Blocked on:** Phase 4

### Phase 6: Multi-player-count training (3-5p)
- Variable player token count (57-59 tokens) + attention masking or pad-to-max
- Single model for 3-5p
- Batching strategy for mixed player counts
