# Transformer Architecture for Rolling Stock Stars

> **WARNING: Breaking refactor.** This work lives on the `transformer-refactor` branch and touches nearly every layer of the codebase — state representation, action space, model architecture, evaluator, and training loop. Backward compatibility with the MLP model, old state vectors, and old action indices is not maintained. Most existing tests will fail until the refactor is complete and tests are updated. The `main` branch is untouched and safe.
>
> **Initial scope:** validate the architecture in **3-player only**. The longer-term goal is a single model across 2-6 players, but multi-player-count support is explicitly deferred until the 3p transformer trains competitively with the current MLP.

## Motivation

Replace the residual MLP (~4.1M params, flat state vector) with a transformer that treats each game entity as a token. Three key advantages:

1. **No state rotation.** The MLP requires rotating player data so the active player is always at slot 0. A transformer is permutation-equivariant on its input tokens - just mark the active player with a flag. Eliminates rotation logic in the evaluator and state construction.

2. **Long-term: one model for all player counts.** The MLP requires separate models for 3p/4p/5p because the input dimension changes. A transformer can support a variable number of player tokens, but the initial prototype stays 3p-only to validate the architecture first. If the 3p transformer matches the MLP well enough, Phase 6 extends the design to 2-6p.

3. **Entity-readout policy heads.** Instead of phase-specific MLP heads that output the full action vector, each entity token produces only its own action logits. Corp tokens output buy/sell logits, company tokens output auction logits, etc. Structurally encodes the right inductive bias.

## Token Decomposition

Each token is projected to a common `d_model` dimension via type-specific linear layers, then augmented with a learned type embedding.

The feature lists below are planning sketches, not frozen interfaces. The exact `get_token_data()` contract will be finalized during core/evaluator integration. In particular, player/corp/company identity will be provided explicitly (likely via learned ID embeddings; one-hot ID fields are acceptable for the first prototype). Type embeddings alone are not enough because corp abilities are asymmetric.

### Entity Tokens

| Token type | Count | Raw features | Notes |
|------------|-------|-------------|-------|
| Player | N (2-6) | ~63 (3p prototype) | explicit player identity, cash, net_worth, income, turn_order one-hot, owned_companies (36), shares (8), is_president (8), round_trips. Active player gets `is_active=1` flag. |
| Corporation | 8 | ~55 | explicit corp identity / ability signal, active, cash, shares, income, stars, share_price, price_index, pending_price_move, revenue breakdown, owned_companies (36), in_receivership, invest buy/sell price impacts (2), `is_phase_active` flag (1) |
| Company | 36 | ~48 | explicit company identity, location flags (for_auction, revealed, removed, acquired), income, static features (face_value, low_price, high_price, stars, base_income), synergy vector (36 — bidirectional synergy $ with each other company), `is_phase_active` flag (1 — this company is the current subject of ACQ/CLOSING/BID/etc.) |
| FI | 1 | 38 | cash, income, owned_companies (36) |
| Market | 1 | 27 | Availability flags for 27 price points |
| Global | 1 | ~19 | Phase (9 one-hot: INV/BID/ACQ_SELECT/ACQ_OFFER/ACQ_PRICE/CLO/DIV/ISS/IPO), CoO (7 one-hot), end_card_flipped, cards_remaining, consecutive_passes |
| Auction | 1 | 2 + 3N | auction_price, auction_price_offset, auction_high_bidder (N one-hot), auction_starter (N one-hot), auction_passed (N flags). Zeroed when not in BID phase. |
| Dividend | 1 | ~35 | dividend_impact (26), dividend_remaining (8), active corp flag. Zeroed when not in DIV phase. |
| Issue | 1 | ~11 | issue_remaining (8), issue_price_impact (1), issue_cash_gain (1), active corp flag. Zeroed when not in ISSUE phase. |
| PAR | 1 | 28 | par_corp_treasury (14), par_shares (14). Zeroed when not in IPO phase. These are global (not per-corp) — they depend on market availability, company face value, and player cash. |
| Acq Offer | 1 | 3 | offer_price (1, normalized — face value for OS, high value for others), is_os_offer (1), acq_is_fi_offer (1). Zeroed when not in ACQ_OFFER sub-phase. Produces buy/pass logits for FI preemption offers. |
| Acq Price | 1 | 1 | acq_is_fi_offer (1 — whether the target company is owned by FI). Zeroed when not in ACQ_PRICE sub-phase. Shaped by attention to is_phase_active-flagged corp and company tokens. |
| Pass | 1 | 0 | No input features — representation is purely the learned type embedding, shaped by attention. Produces the pass/leave logit for INVEST, BID (leave), ACQ_SELECT, CLOSING, ISSUE, and IPO phases. |

**Total: 54 + N tokens** (57 for 3p, 60 for 6p)

### What disappears from the state vector

The current turn state has ~225 floats of context-dependent fields. Many exist because the MLP can't selectively attend to entities:

- **Active entity duplication** (~100 floats): `active_company` (36 one-hot + 5 scalars), `active_corp` (8 one-hot + 14 scalars + 36 owned_companies). The transformer can attend directly to the relevant entity token. Replaced by a single `is_phase_active` flag on the relevant corp/company token — the transformer already has the entity's full features via its token.
- **Auction slot info** (5 * N floats): Per-slot company features (stars, prices, income) duplicated from company data. Unnecessary when company tokens carry their own features.
- **Invest impacts** (16 floats): Buy/sell price impact per corp. Moved onto corp tokens (2 extra floats each) where they belong.

### Phase-specific context tokens

Two dedicated tokens carry phase-specific information that the model needs for decision-making, and double as policy readout points for their respective phases:

- **Dividend token**: dividend_impact (26 levels), dividend_remaining (8 corp flags), active corp flag. Zeroed outside DIV phase. Policy logits for DIVIDENDS (26 actions) are read directly from this token.
- **Issue token**: issue_remaining (8 corp flags), issue_price_impact, issue_cash_gain, active corp flag. Zeroed outside ISSUE phase. Policy logits for ISSUE (2 actions) are read directly from this token.

Interpretability analysis on the MLP model showed these phase-specific features significantly improve model quality. Giving them dedicated tokens preserves that signal and provides a natural readout point — the token that carries the decision context also produces the decision.

**PAR context token**: The PAR phase is eliminated as a separate game phase, but the PAR token remains as context for the merged IPO phase. It carries par_corp_treasury and par_shares — these are global (not per-corp) since they depend on market availability, company face value, and player cash. The PAR token projects into 14 price-feature vectors; corp tokens project into queries; the bilinear dot product produces (8, 14) IPO logits. This avoids redundant par price evaluation across corps — the shared price computation happens once in the PAR token, and each corp just determines its affinity with each price.

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
Entity-readout policy heads (+ bilinear for ACQ_SELECT and IPO) + value head
```

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| d_model | 128 | Scale to 256 once architecture validated |
| num_heads | 4 | Standard ratio with d_model |
| num_layers | 10 | Enough depth for multi-hop reasoning |
| ff_mult | 2.67 | SwiGLU FFN inner dim = ceil(ff_mult * d_model); 2.67x gives parameter parity with standard 4x GELU FFN (3 matrices vs 2) |
| d_bilinear | 64 | Key dimension for bilinear policy heads (ACQ_SELECT, IPO) |
| token_dim | 63 | Fixed width for the 3p prototype. Phase 6 revisits padding/embeddings for shared 2-6p support. |

**No positional encoding over token order.** Permutation equivariance is a feature, not a bug. Token order is fixed for implementation convenience, but entity identity should come from explicit player/corp/company ID features or ID embeddings plus the type embeddings, not from sequence position.

**Normalization:** RMSNorm (not LayerNorm). Drops the mean-centering step — only divides by root mean square. Simpler, slightly faster, no downside.

**FFN:** SwiGLU (`W_down(SiLU(W_gate(x)) * W_up(x))`) instead of standard GELU FFN. Learned gating gives finer control over information flow. Three weight matrices (all bias=False) instead of two; the 2.67x multiplier compensates to keep parameter count matched.

### Type-specific input projections

All projections take the full zero-padded `token_dim` input (no per-type feature slicing). Weights for always-zero feature positions are inert; the simplicity is worth the ~2% extra params.

```python
# All projections: nn.Linear(token_dim, d_model) — one per entity type
self.player_proj  = nn.Linear(token_dim, d_model)
self.corp_proj    = nn.Linear(token_dim, d_model)
# ... (12 total, one per non-pass token type)
# Pass token has no input features — its initial representation is just its type embedding

# Learned type embeddings (added after projection)
self.type_embed = nn.Embedding(13, d_model)  # player/corp/company/fi/market/global/auction/dividend/issue/par/acq_offer/acq_price/pass
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
| BID | **Pass token** → [leave] (1); **Auction token** → [raise_0 ... raise_13] (14) |
| ACQ_SELECT | **Pass token** → [pass] (1); bilinear corp×company dot product → [select_0 ... select_287] (288) |
| ACQ_OFFER | **Pass token** → [pass] (1); **Acq Offer token** → [buy] (1) |
| ACQ_PRICE | **Acq Price token** → [price_offset_0 ... _50, fi_buy] (52) |
| CLOSING | **Pass token** → [pass] (1); Company 0..35 → [close] (36, most masked) |
| DIVIDENDS | **Dividend token** → [div_0 ... div_25] (26) |
| ISSUE | **Pass token** → [pass] (1); **Issue token** → [issue] (1) |
| IPO | **Pass token** → [pass] (1); bilinear corp×PAR dot product → [corp_0_par_0 ... corp_7_par_13] (112, most masked) |

INVEST, CLOSING, and IPO use multi-entity readout (company/corp tokens + pass). ACQ_SELECT uses bilinear dot product between corp and company tokens to select a (corp, company) pair. ACQ_OFFER uses Pass (pass) + Acq Offer token (buy). ACQ_PRICE reads from its dedicated token. DIVIDENDS reads solely from its dedicated phase token. BID splits between Pass (leave) and Auction (raises).

### Action space size

The new action layout (for any player count):

```
INVEST:      1 (pass) + 36*15 (auction) + 8*2 (buy/sell) = 557
BID:         15 (leave + 14 raises)
ACQ_SELECT:  289 (8*36 corp×company pairs + pass)
ACQ_OFFER:   2 (pass + buy)
ACQ_PRICE:   52 (51 price offsets + FI buy)
CLOSE:       37 (36 company close + pass)
DIV:         26 (dividend amounts)
ISSUE:       2 (issue + pass)
IPO:         113 (8*14 corp par prices + pass)
```

Per-phase action indices (no global action vector). The largest phase is INVEST at 557 actions. All sizes are **fixed across player counts**.

ACQ_SELECT, ACQ_OFFER, and ACQ_PRICE are distinct phases from the model's perspective (separate one-hot IDs in the Global token's phase encoding), even though the game engine treats them as sub-phases of ACQUISITION. This is the same pattern as INV/BID today — BID is a "sub-phase" of INVEST in the game rules, but the model sees them as distinct phases.

**ACQ flow:**
- **ACQ_SELECT** → player picks (corp, company) or passes
- **ACQ_OFFER** (conditional) → if target is FI-owned, higher-priority corps are offered the chance to preempt in priority order (OS first at face value, then by share price descending at high value). Also triggered for receivership auto-buy attempts at the start of ACQ.
- **ACQ_PRICE** → current prototype compromise: either choose a negotiated price offset for a non-FI purchase, or execute the fixed-price `FI buy` action after the FI preemption path resolves.

This phase split is deliberate. A fully joint ACQ head over `(corp, company, price)` would be `8 × 36 × 51 = 14,688` actions before pass/FI special cases, which is too large for the current eval-server IPC path and replay-buffer storage model.

**All 9 decision phases:** INV, BID, ACQ_SELECT, ACQ_OFFER, ACQ_PRICE, CLO, DIV, ISS, IPO. (Down from 8 in the old model: -PAR from IPO/PAR merge, +ACQ_OFFER and +ACQ_PRICE from ACQ split.)

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

# Pass token → single pass/leave logit (shared across all phases that use it)
self.pass_head = nn.Linear(d_model, 1)

# Phase-specific context tokens → policy logits (read directly from token)
self.auction_head   = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 14))  # raise amounts
self.dividend_head  = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 26))
self.issue_head     = nn.Linear(d_model, 1)  # issue (pass comes from Pass token)
self.acq_offer_head = nn.Linear(d_model, 1)  # buy logit (pass from pass_head)
self.acq_price_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 52))  # price offsets + FI buy

# ACQ_SELECT: bilinear dot product between corp and company tokens
self.acq_select_corp    = nn.Linear(d_model, d_k)  # corp → query
self.acq_select_company = nn.Linear(d_model, d_k)  # company → key
# selection_logits = (corp_q @ company_k.T) / sqrt(d_k)  →  (batch, 8, 36) → flatten to 288

# IPO: bilinear dot product between corp tokens and PAR token price projections
self.corp_ipo_proj  = nn.Linear(d_model, d_k)       # corp → query (batch, 8, d_k)
self.par_price_proj = nn.Linear(d_model, 14 * d_k)  # PAR → 14 price features, reshaped to (batch, 14, d_k)
# ipo_logits = (corp_q @ par_prices.T) / sqrt(d_k)  →  (batch, 8, 14) → flatten to 112
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
                           global, auction, dividend, issue, par, acq_offer,
                           acq_price, pass]
    """
    # Player tokens (N tokens, ~20-30 dynamic features each)
    for i in range(num_players):
        _fill_player_token(state, buffer + i * token_dim, i)
    
    # Corp tokens (8 tokens, ~20 dynamic features each)
    for i in range(8):
        _fill_corp_token(state, buffer + (num_players + i) * token_dim, i)
    
    # Company tokens (36 tokens, ~12 dynamic + ~36 static synergy features each)
    for i in range(36):
        _fill_company_token(state, buffer + (num_players + 8 + i) * token_dim, i)
    
    # ... FI, market, global, phase-specific tokens
```

Each `_fill_*_token()` helper reads from the compact state and writes raw/lightly-processed features into the buffer. Static features (synergies, face values) are written every time — it's just memcpy and the simplicity is worth more than the minor bandwidth savings.

### Eval buffer sizing

The eval buffer is `(batch_size, num_tokens, token_dim)`:
- `num_tokens`: 57 for the 3p prototype
- `token_dim`: 63 for the 3p prototype (all token types zero-padded to the same width)
- Total per sample: 57 × 63 = 3,591 floats (~14KB)
- Compare to current visible state: 1,109 floats (~4.4KB)

~3x larger per sample, but still small in absolute terms for a 3p prototype. The rectangular layout enables clean GPU operations. All type-specific projections take the same `token_dim` input (no per-type slicing needed). Phase 6 can either pad to 6p maxima or move more categorical information into embeddings.

### Replay buffer storage

The replay buffer stores training examples. With per-phase action indices, each example contains:
- **Token data**: the eval buffer contents (or the compact state + `get_token_data()` at training time)
- **Phase ID**: which phase this decision was in (determines action space)
- **Legal action mask**: per-phase (max 557 for INVEST)
- **Policy target**: MCTS visit distribution (per-phase, same size as mask)
- **Value target**: A0GB values (N floats)

Option A: store the compact state and re-extract tokens at training time. Smallest storage.
Option B: store the eval buffer directly. Faster training (no re-extraction), larger storage.

Option B is simpler for a very small-scale prototype, but expensive at current training defaults: with `57 × 63` float token buffers and `buffer_capacity=500_000`, token storage alone is ~6.7 GiB, and total replay storage is roughly ~8.8 GiB once masks, policy targets, and value targets are included. If we start with Option B, we should also reduce buffer capacity. Option A is the more realistic long-term default.

## Impact on Training & MCTS Pipeline

### What changes

- **GameState**: Compact state array, no visible/hidden split. New `get_token_data()` method fills eval buffers.
- **Eval buffers**: `(batch, num_tokens, token_dim)` rank-3 tensor in shared memory. All token types use the same padded width.
- **Action layout**: Per-phase action indices (max 557 for INVEST). Update `actions.pyx` layout and mask generation.
- **Evaluator**: Remove state rotation. Remove un-rotation of values. Simpler.
- **Model interface**: `forward(x, phase_ids) → (policy_logits, values)` where `x` is `(batch, num_tokens, token_dim)` and `phase_ids` is `(batch,)`. Policy logits padded to MAX_ACTIONS (557), -1e9 beyond phase action range.

### Game engine simplifications

Four major pieces of action-space indirection in the current engine become candidates for removal. These are not direct requirements of the rules; they are engine/training choices that helped the MLP focus on smaller action spaces. The transformer refactor is a chance to test whether relaxing some of them simplifies the engine without unacceptable search/training regressions:

- **Auction slots** (`entities/company.pyx`): The current engine maps auctionable companies to positional slots (0, 1, ..., N-1) so the action space scales with player count. With company-indexed auction actions, this indirection is removed — `get_auction_company_for_slot()` and the slot machinery are deleted. The mask just enables the 15 logits for each auctionable company directly.

- **Closing offer buffer** (`phases/closing.pyx`): The current engine pre-generates close offers into a hidden buffer, sorts by priority, and presents them one-by-one with CLOSE/PASS actions. The transformer experiment is to relax that training constraint and instead present all eligible companies directly: go in player order, mask all valid closes, let the model pick one company or pass, repeat until pass, then move to the next player. Mandatory closes for negative income+cash still happen at the end.

- **Acquisition offer buffer** (`phases/acquisition.pyx`): The current engine pre-generates (corp, company) acquisition offers into a hidden buffer, sorted by priority (OS→FI, Corp→FI, Corp→Corp, Corp→Player), and presents them one-by-one with price offset/pass actions. We do **not** expose the fully joint `(corp, company, price)` action because it would create a ~14.7k-action head and corresponding IPC/replay costs. Instead, the proposal is to remove the hidden offer buffer while still factorizing ACQ into `ACQ_SELECT`, `ACQ_OFFER`, and `ACQ_PRICE`. FI preemption (higher-priority corps getting the option to buy first) is handled explicitly in `ACQ_OFFER` rather than being baked into offer ordering. Receivership auto-acquisitions remain as forced actions in the engine. This is the highest-risk simplification in the design.

- **IPO/PAR phase merge** (`phases/ipo.pyx`, `phases/wrap_up.pyx`): The current engine splits IPO into two phases — IPO (select corp or pass) then PAR (select par price). With entity-readout, each corp token produces 14 par price logits directly. The player picks "corp X at par price Y" in a single action. The PAR phase is eliminated entirely, simplifying the phase graph and removing a state transition.

All four simplifications follow the same pattern: the MLP needed a small, fixed action space, so the engine compressed entity-level decisions into slot/offer indirection or multi-step sequences. The transformer's entity-readout makes it plausible to relax some of those constraints and test more direct action formulations.

### What stays the same

- MCTS search logic (PUCT, A0GB, subtree reuse)
- Self-play worker architecture (shared memory, eval servers — buffer layout changes but architecture doesn't)
- Training loop (policy CE + value MSE)
- Game engine core logic (driver.pyx, phases/*.pyx — core rules unchanged, though some current action-space-focusing constraints may be relaxed experimentally)

## Open Questions

1. **Compact state layout**: Need to design the new compact state array. All the game data currently split across visible+hidden goes into one array, stored as raw values (integers, enums, counts). Estimated ~500-800 floats for 3p. The entity handle pattern (Player, Corp, Company, etc.) stays but their offset calculations simplify.

2. **`get_token_data()` feature lists**: Exactly which features go into each token type's buffer slot. Some features are dynamic (cash, income, ownership), some are static (synergies, face values), some are phase-conditional (zeroed outside relevant phase). This is the new single source of truth for the state→NN mapping.

3. **Batching with variable token counts**: Deferred until after the 3p prototype is validated. For multi-player-count training, batches would contain games with different numbers of player tokens (3p: 57 tokens, 6p: 60). Options: pad to max and use attention mask, or batch by player count. Padding is simpler; batching by count is more efficient.

4. **Replay storage strategy**: Option B (store token buffers directly) is convenient for prototyping but probably requires a much smaller replay buffer. Option A (store compact state and re-extract tokens) is likely the long-term path.

5. **Inference speed**: Need to benchmark transformer vs MLP for the sequence lengths we're dealing with (~57 tokens). Should be fine, but verify with `torch.compile`.

6. **Graduating from 3p prototype**: If 3p works, what's needed to support 2-6p? Mainly: variable player token count, re-derive action masks, adjust eval buffer dimensions, and lock down a shared fixed-width token contract.

7. **Head dispatch**: The forward pass has 9 phase-specific dispatch paths plus 2 bilinear operations (ACQ_SELECT, IPO). Need a clean implementation that handles mixed-phase batches efficiently.

## Implementation Phases

### Phase 1: Compact state + token extraction (core/)
- Redesign `GameState` as a single compact array (no visible/hidden split) — **done in `09e5048`** (`core/state.{pxd,pyx}` reduced to structural primitives + entity-handle delegation; player block now contains all per-player tracking incl. share buys/sells).
- Delete one-hot encoding, normalization-on-write, entity duplication — **done in `09e5048`**.
- Reduce `core/data.{pxd,pyx}` to pure data + constants — **done.** All field-level accessors removed; the file exposes only the static arrays (company/corp/market/CoO/par tables, synergy matrix), the shared enums (`GameConstants`, `GamePhases`, `CorpIndices`), and the normalization divisors used by token extraction. Computational helpers that used to live here (synergy aggregation, required stars, cost-of-ownership lookup, par-price validity) now live as private cdef functions in the module that uses them.
- Update entity handles (Player, Corp, Company, FI, Market, Deck, TurnState) for the new offset layout — **done.** Every handle is fully stateless: no per-instance offset cache, no `initialize()` step, no normalization-on-read. Each accessor reads its slot inline from the module-level `LAYOUT` / `PLAYER_FIELDS` / `CORP_FIELDS` / `TURN_OFFSETS` constants, so the singletons in `PLAYERS` / `CORPS` / `COMPANIES` are reused with any `GameState` at any player count. Company ownership is read from the shared `company_locations` / `company_owner_ids` arrays — the per-player and per-corp `owned_companies` bitmaps are gone.
- Implement `get_token_data()` in Cython — fills eval buffer from compact state. **Pending** — this is the next chunk of Phase 1 work.
- Target: 3p only

### Phase 2: Action space refactor (core/actions.pyx)
- Per-phase action indices (no global action vector)
- New action layout: company-indexed auctions, company-indexed closes, bilinear ACQ, merged IPO
- ACQ sub-phases (SELECT/OFFER/PRICE), 9 decision phases total
- Update mask generation, action decoding, phase handlers

### Phase 3: Transformer model (nn/) ✅
- **Done.** See `nn/transformer.py` (~2.3M params, `TransformerConfig` defaults)
- Pre-RMSNorm transformer blocks with SwiGLU FFN
- Type-specific projections (uniform `token_dim` input), entity-readout heads
- Bilinear heads for ACQ_SELECT and IPO
- Smoke test passes: all 9 phases produce correct shapes, values in [-1,1]
- Target: 3p only, all dimensions parameterized via `TransformerConfig`

### Phase 4: Evaluator integration (mcts/evaluator.py)
- Eval buffers: `(batch, num_tokens, token_dim)` in shared memory
- Remove state rotation, remove value un-rotation
- Verify MCTS works end-to-end

### Phase 5: Training loop updates (train/)
- Update replay buffer for per-phase action indices + token data storage
- Update shared memory buffer layout
- Verify self-play + training loop works

### Phase 6: Multi-player-count support (only if the 3p prototype is competitive)
- Variable player token count + attention masking for padding
- Single model for 2-6p
- Batching strategy for mixed player counts
