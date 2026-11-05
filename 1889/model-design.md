# Shikoku 1889 Model Design

Status: design for step 8 of [`pipeline-refactor.md`](pipeline-refactor.md),
dated 2026-11-03; the user reviewed it and settled Section 10 on 2026-11-04.
Sub-steps 8.1-8.4 are built: the model, the transport, the stats hook, the
configs, and a local smoke run with a resume (Section 7.2). The cloud run (8.5)
and the capacity probe (8.6) are not. Step 8 builds the first 1889 model,
`nn/transformer-1889.py` v0, on step 7's inputs ([`token-data-1889.md`](../token-data-1889.md), rationale
in [`extraction-design.md`](extraction-design.md)), adds what the pipeline
needs to train it, and runs it once.

The RSS template is `nn/transformer-v3.py` at
`train_configs/v3-10-01-26.json`. Paths are relative to the repo, and
`path:line` cites the code at 65c62b7; a bare `:line` cites
`nn/transformer-v3.py`. **Measured:** marks a number from the scripts in
`scratchpad/g1889_model_design/` (Section 2); **Test:** marks a claim for
implementation to check.

## 1. Scope

**In scope:**

- `nn/transformer-1889.py` v0: `INPUT_LAYOUT_VERSION = 1`,
  `build_model(config, spec)`, and the model of Sections 3-4;
- the pipeline changes it needs, chiefly flat relation records in replay,
  the trainer, and the eval server for 1889 (Section 6);
- a 1889 self-play stats hook (Section 6.6);
- `train_configs/1889-v0.json`, a local smoke config, and the first cloud run
  (Section 7);
- the capacity probe that settles the model's size once the run has data
  (Section 5.7).

**Out of scope:** build distance (7b), the v0-v1 comparison itself
(milestone 6; Section 9 lists the hooks v0 leaves it), 1889 support in
`train/analyze_game.py` and `train/tournament.py`, and live play. The v2 and
v3 model files, RSS's relation transport, and RSS training stay as they are,
and `tests/test_v2_parity.py` passes on its fixture.

## 2. Measurements

Scripts in `scratchpad/g1889_model_design/`, outputs beside them as
`*_65c62b7.txt`. CPU only, since the local GPU serves the live bot: the
Ryzen 9 9950X3D, 16 threads, torch 2.9.1, fp32.

| Script | Measures |
|---|---|
| `params.py` | v3 at v3-10-01-26 and v2 at `bigger-multi.json`: parameters by group |
| `cost_model.py` | FLOPs per state and CPU forward time, v3 against a cost proxy of this design at six sizes |
| `bias_bench.py` | the relation bias path alone, padded against flat records, on synthetic records (Section 5.3) |
| `size_grid.py` | the analytic FLOP and parameter model across widths and depths |
| `game_length.py` | returned decisions per uniform random playout, RSS and 1889, 300 games per player count |
| `fixture_length.py` | the decisions self-play would search in the four 18xx.games fixtures, replayed through the engine |
| `branching.py` | policy actions per decision, and the replay row's size |
| `map_hops.py` | hop distances on the hex graph |
| `tb_ranges.py`, `tb_tags.py` | RSS self-play telemetry from TensorBoard: epochs 19-36 of `runs/02-10-26-muon-v3`, the v3-10-01-26 run (its simulation counts match that config's ramp), and epoch times of the earlier `runs/26-09-29-muon-v3` |

The cost proxy has the design's shapes on random inputs: 95 rows of 108, the
per-type projections with a shared tile block, the identity embeddings, v3's
`TransformerBlock`, the sparse relation bias over E = 2,048 padded records with 25
dynamic and 10 static relations, and the heads of Section 4 at their input
widths and entity counts. It is a measuring aid, not the model.

## 3. Architecture

### 3.1 Overview

The forward pass is v3's (`nn/transformer-v3.py:1595-1744`) on 1889's rows:

1. type projections of each row, plus identity and type embeddings
   (3.2-3.3);
2. per layer and head, an attention bias from 25 dynamic and 10 static
   relations (3.4);
3. L pre-RMSNorm blocks with SwiGLU, then a final RMSNorm (3.5);
4. the selected rows: the actor, the decision corp, and the other
   `is_selected` slots (3.6);
5. heads that fill the 19 phase blocks of the unified 228 slots, with
   illegal slots at −1e9 (Section 4);
6. a value per player row (3.7).

v0 departs from v3 in four places, each argued below: a projection shared by
two row types, identity embeddings for four entity types, no relation input
mixing, and heads that point at entity rows on a larger scale (hexes,
candidates).

### 3.2 Projections

| Rows | Type | Projected slots | Projections | Added |
|---|---|---|---|---|
| 0 | GLOBAL | 1-43 | `global_proj` | type |
| 1 | PRICE | 1-23 | `price_proj` | type |
| 2 | DIVIDEND | 1-7 | `dividend_proj` | type |
| 3-9 | CORP | 1-50 | `corp_proj` | type, corp identity |
| 10-16 | PRIVATE | 1-17 | `private_proj` | type, private identity |
| 17-22 | TRAIN | 1-12 | `train_proj` | type, train identity |
| 23-74 | HEX | 2-77; 1 and 78-104 | `tile_proj` + `hex_proj` | type, hex identity |
| 75-89 | CANDIDATE | 2-77; 1 and 78-107 | `tile_proj` + `candidate_proj` | type, the identity of its hex |
| 90-94 | PLAYER | 1-37 | `player_proj` | type |

- Slot 0, the key mask, is not projected, as in v3 (`:157-171`). The rows
  carry no relation tails, so each projection reads its whole row, where
  v3's drop theirs (`:115-127`).
- **The tile block** (slots 2-77, token-data-1889.md) is one biasless
  `tile_proj` over rows 23-89, which are contiguous. A HEX row holds its
  current tile and a CANDIDATE row the tile as it would be laid, so after a
  lay the hex row's block equals the candidate's (extraction-design §10,
  item 2). Candidate rows are visible only at TILE and ER_BONUS, about 5% of
  decisions, while 52 hex rows carry a block in every state; sharing gives
  the candidates the tile semantics the hexes learn: tile ids (the one-hot
  makes `tile_proj` a tile embedding), path masks, colors, and centers.
- Hidden rows (slot 0 = 0) project to their biases and embeddings; they are
  masked as keys, and no legal slot reads them.

### 3.3 Embeddings

- **Identity:** `hex_id_embed` (52), `corp_id_embed` (7),
  `private_id_embed` (7), and `train_id_embed` (6), added at projection as
  v3 adds its company and corp identities (`:832-854`). They stand for what
  the rows leave to the row order: a hex's place on the map, a corp's home
  and charter, a private's ability, a train's rust schedule. RSS's corp
  identity is load-bearing for the corps' powers, and 1889's privates differ
  the same way.
- **Candidates** get the identity of the hex they would be laid on, the row
  HEX 1 selects (the pending hex at TILE, C4 at ER_BONUS), so a candidate's
  absolute-side path mask meets its place on the map before the first layer
  rather than through attention along CANDIDATE_AT.
- **Players** get none: their rows are in seat order, and PLAYER 2-6 give
  the seat relative to the actor, as in v3.
- **`type_embeds`** (9 types), added to every row (`:626-645`, `:1708`).
- All are `nn.Embedding`, which the trainer keeps on AdamW without decay
  (`train/trainer.py:129-142`, `train/trainer.py:176-179`).

### 3.4 Relations

The mechanism is v3's (`:886-1016`): per layer, each head's additive bias is
the relations weighted by `relation_bias_mult[layer, head]`, plus the key
mask from slot 0.

- **35 channels:** the 25 dynamic relations (token-data-1889.md,
  "Relations") and the 10 static planes. `relation_bias_mult` is
  (L, H, 35), zero at initialization (`:1849`), so training starts from
  plain attention.
- **Scales:** a buffer of 25, shares (2, 3) × 1/10 and trains (12, 13) × 1/4,
  the rest 1, applied to record values or dense planes once per forward
  (`:899-906`, `:952-955`).
- **Flat records** (every path but tests, Section 6.1): `(N, 5)` int16 rows
  `(batch, relation, query, key, value)`, the batch's real records with no
  per-state padding, with N from the input's shape. A bucketed eval launch
  rounds N up with value-0 rows (Section 6.5). Workers still write `(E, 4)`
  uint8 records with a zero tail, and the caller flattens them before the
  forward, since `nonzero` would break the compiled graph. v3 pads every state
  to E and scatters all of it (`:984-1016`), which here would be about 90%
  padding (Section 5.3). A record counts when its value is above 0 and its id
  below 25; any other gets weight 0 and id 0, as v2 filters
  (`nn/transformer-v2.py:832`).
- **One dense tensor per layer.** SDPA needs a dense `(B, H, T, T)` mask, which
  v3 builds in three tensors a layer: zeros, plus static, plus visibility
  (`:1012-1016`, `:311-314`). v0 starts from a copy of the layer's static bias,
  scatters the records into it, and hides keys in place, with the key mask
  computed once per forward.
- **Dense planes** (tests): `(B, R, T, T)` with R ≥ 25, sliced to the first 25.
- **Static planes:** `static_relation_planes(cfg.num_players)` at
  construction, a non-persistent `(10, T, T)` buffer (Section 6.3), mixed per
  layer by its 10 columns of `relation_bias_mult` and broadcast over the
  batch (`:930-936`).
- Dense and flat inputs give the same outputs (**Test:** 8.1), as
  PIPELINE.md requires of v3.

What a layer does with the map: through HEX_SIDE_s a hex attends to the hex
across side s, through TRACK_LINK to neighbors joined by track, through
STATION and REACHES_* to the corps stationed at or reaching it, and through
CANDIDATE_AT and CANDIDATE_EXIT to a candidate and the hexes across its
exits. Each relation-biased layer moves information one hop on the hex
graph, which sets how much depth map reasoning needs (Section 5.5).

### 3.5 Trunk

- **Blocks:** L copies of v3's `TransformerBlock` (`:263-322`): pre-RMSNorm,
  biasless packed QKV, SDPA with 3.4's finished mask, SwiGLU of hidden width
  ⌈ff_mult · d⌉ rounded up to 64, and residuals; then a final RMSNorm. The
  module copies the block rather than sharing v3's (pipeline-refactor §5),
  and initializes as v3's `_init_weights` (`:1828-1850`).
- **No relation input mixing.** v3-10-01-26 runs with it off; v3 still builds
  its weights, 1.05M of its 17.16M (Section 5.1). Over 35 relations it would
  cost 37 d² per token, about 2.8 blocks' worth of token-wise FLOPs (a block
  is 13 d² at ff_mult 3), for one more round of the hops the bias already
  offers every layer.
- **Phase conditioning is in the input,** as in v3: GLOBAL 1-19 carry the
  decision phase, the phase rows (PRICE, DIVIDEND, CANDIDATE) are visible
  only at their decisions, and every head reads GLOBAL. No adaLN: v2's
  `phase_conditioning` (`nn/transformer-v2.py:163`, `nn/transformer-v2.py:275-300`) is off in
  every config, and v3 dropped it.
- **Sizes** come from the config's `d_model`, `num_heads`, `num_layers`, and
  `ff_mult`: 256, 4, 15, and 3.0 (Section 5.7).

### 3.6 Selections

v3's `_active_token` (`:1167-1178`) with the slot as a parameter: a selected
row is the `is_selected` slot's weighted sum over its type's rows, zero when
no row sets it. Raw slots are selected the same way, in fp32, as v3 selects
its raw features (`:1323-1337`).

| Selection | Row, slot | Read by |
|---|---|---|
| actor `a` | PLAYER 1 | every head |
| decision corp `c` | CORP 1 | every head but `auction_private`, `bid`, and `stock_corp`, whose phases have none (token-data-1889.md, "Conventions"); zero in shared heads at SELL_AMOUNT, an SR's PORT, STOCK's DR and MF slots, and AUCTION's and STOCK's passes |
| the train's seller | CORP 2 | SELECT_TRAIN, TRAIN_PRICE |
| the sold corp | CORP 3 | SELL_AMOUNT, EMERGENCY_SELL_AMOUNT |
| the private | PRIVATE 1 | BID, PRIVATE_PRICE |
| the topmost private | PRIVATE 2 | AUCTION's buy |
| the train type | TRAIN 1 | TRAIN_PRICE |
| the hex | HEX 1 | TILE and ER_BONUS: the candidate scorer and the candidates' identity |

### 3.7 Value head

As v3 (`:817-822`, `:1737-1743`): `value_head` is Linear(d, d), GELU,
Linear(d, 1), and tanh on each player row, in canonical order, padded to
`cfg.num_players`. With `zero_sum_values` the mean over real players (PLAYER
slot 0) is subtracted and padded slots are zeroed. Mixed 3-5 player batches
work as in RSS: padded rows are zero and hidden as keys, and the trainer
masks their value slots (`train/trainer.py:486`). Terminal rewards come from
`final_scores` through `compute_terminal_values`, which is game-agnostic. The
trainer's raw-value diagnostics hook the module named `value_head`
(`train/trainer.py:83-85`).

## 4. Policy heads

### 4.1 Pattern

Every head is v3's two-layer MLP, Linear(n_in, d), GELU (tanh), Linear(d,
n_out), run on every row of the batch, with the legal mask selecting
(`:1539-1589`). A head reads:

- **the context:** `[g, a]`, the final GLOBAL row and the actor, plus the
  decision corp `c` except in `auction_private`, `bid`, and `stock_corp`. Their
  phases have no decision corp, so its d² weights in each would never train:
  the input is zero at their own phases, and other phases' rows are masked;
- **its rows:** the entity row the choice names (a pointer head, whose
  weights are shared across the rows of the type) or the phase row and
  selections (a joint head, which emits a fixed block);
- **raw per-choice slots** from token-data-1889.md's "Per-choice readouts",
  unprojected, as v3 feeds its outcome features (`:1315-1350`,
  `:1352-1404`), so exact amounts and deltas reach the logit without passing
  through 15 blocks.

### 4.2 Readouts by phase

| Phase (size) | Ids | Head | Reads, beyond the context | Raw slots |
|---|---|---|---|---|
| AUCTION (8) | 0 pass | pass | | |
| | 1 buy the topmost, 1+p bid on p, p ≥ 1 | `auction_private`: pointer over PRIVATE rows but private 0's, a bid logit per row, and one buy logit from the row PRIVATE 2 selects | PRIVATE p | PRIVATE 7-13 |
| BID (17) | 0 drop, 1+k point k | `bid`: joint, 17 | PRICE; the private | PRICE 5-23 |
| STOCK (31) | 0 pass | pass | | |
| | 1+c IPO, 8+c pool, 15+c par, 22+c sell | `stock_corp`: pointer over CORP rows, 4 each, laid out by kind | CORP c | CORP 7-25, 44-48; the actor's PLAYER 12+c and 19+c |
| | 29 DR, 30 MF | `private` (4.3) | | |
| PAR (6) | par index | `par`: joint, 6 | PRICE | PRICE 5-23 |
| SELL_AMOUNT (5) | n − 1 | `sale`: joint, 5 | the sold corp | its CORP 15-19, 48, 49 |
| BUY_PRIVATE_PRE (10) | 0 done | pass | | |
| | 1+p, 8 DR, 9 MF | `private` | | |
| TRACK (46) | 0 pass | pass | | |
| | 1+i | `track_hex`: pointer over the HEX rows `TRACK_HEXES` names | HEX `TRACK_HEXES[i]` | HEX 92-102 |
| TILE (15) | candidate k | `candidate`: pointer over CANDIDATE rows | CANDIDATE k; the hex | CANDIDATE 1, 99-107 |
| TOKEN (16) | 0 pass | pass | | |
| | 1+i | `token_hex`: pointer over the HEX rows `TOKEN_HEXES` names | HEX `TOKEN_HEXES[i]` | HEX 103, 104 |
| DIVIDEND (2) | 0 withhold, 1 payout | `dividend`: joint, 2 | DIVIDEND | DIVIDEND 1-7 |
| DISCARD (3) | i | `discard`: pointer over the TRAIN rows `DISCARD_TYPES` names | TRAIN `DISCARD_TYPES[i]` | TRAIN 12 |
| BUY_TRAIN (18) | 0 pass, 1 depot, 2 pool, 17 bankrupt | `buy_train`: joint, 4 | the six TRAIN rows | TRAIN 2-9 of each |
| | 3+c c's trains, 10+c emergency sale of c | `buy_train_corp`: pointer over CORP rows, 2 each | CORP c | CORP 15-19, 46, 49 |
| SELECT_TRAIN (9) | type t; 6+i a D for `TRADE_IN_TYPE[i]` | `select_train`: pointer over TRAIN rows, an add logit per row and a trade-in logit from the rows `TRADE_IN_TYPE` names | TRAIN t; the seller | TRAIN 10, 11 |
| TRAIN_PRICE (5) | point k | `train_price`: joint, 5 | PRICE; the train type; the seller | PRICE 5-23 |
| EMERGENCY_SELL_AMOUNT (5) | n − 1 | `sale`, shared with SELL_AMOUNT | the sold corp | its CORP 15-19, 48, 49 |
| BUY_PRIVATE_POST (9) | 0 done | pass | | |
| | 1+p, 8 DR | `private` | | |
| PRIVATE_PRICE (4) | point k | `private_price`: joint, 4 | PRICE; the private | PRICE 5-23 |
| ER_BONUS (9) | 0 pass | pass | | |
| | 1+k | `candidate`, shared with TILE | CANDIDATE k; the hex | CANDIDATE 1, 99-107 |
| PORT (10) | i | `port`: pointer over the 10 (port hex, rotation) pairs | HEX `PORT_LAY_HEX[i]` | the port tile's path mask at `PORT_LAY_ROT[i]` |

- **STOCK's four kinds** for corp c are four outputs of one scorer, so what
  it learns about a corp (price, zone, the actor's stake) serves all four, as
  v3's INVEST trade scorer emits buy and sell together (`:1230-1241`). The
  actor's shares and SR sales of c join the table's slots, as v3 feeds the
  actor's share count (`:1198-1207`).
- **TILE and ER_BONUS share the candidate scorer,** extraction-design §7.1's
  sketch: CANDIDATE 1 marks an ER_BONUS candidate, and the context says
  whose lay it is.
- **PORT** is extraction-design §13 item 9, as accepted: the port hexes and
  rotations are static, so a rotation's geometry is the 21-slot mask of
  `PORT_TILE` at r from `TILE_MASK` (`g1889/data`), a buffer built at
  construction.
- **The slot tables.** Slots hold only what can ever be legal (engine-design
  §12 item 8), so the pointers above score a subset of their rows, named by
  `TRACK_HEXES`, `TOKEN_HEXES`, `DISCARD_TYPES`, and `PORT_LAY_HEX`, index
  buffers built at construction from `g1889/data`. Every row stays in the
  trunk, and the tables do not change the input layout.

### 4.3 Interjection and pass slots

**`private`** is one pointer over the PRIVATE rows: a buy logit per row, an
ability logit from DR's and MF's rows, and raw PRIVATE 16-17
(`can_buy_private`, `can_exchange_dr`). The phases read it:

- buy private p: BUY_PRIVATE_PRE 1+p and BUY_PRIVATE_POST 1+p;
- DR's ability, the exchange: STOCK 29, BUY_PRIVATE_PRE 8, and
  BUY_PRIVATE_POST 8;
- MF's ability, the port tile: STOCK 30 and BUY_PRIVATE_PRE 9.

These are the only decisions that offer them (engine-design §5.1); a human's
move at any other reaches the engine through `DRIVER.override_private_slot`,
never through a model. One head serves them because a corp's purchase of p is
one decision at either end of its turn, and GLOBAL in the context says which
end. AUCTION's buy is a player's purchase at auction, so it keeps its own
head.

**Passes:** one MLP on the context with a column for each of AUCTION, STOCK,
BUY_PRIVATE_PRE, TRACK, TOKEN, BUY_PRIVATE_POST, and ER_BONUS. A pass reads
only the context, so the seven share a hidden layer and keep their own output
weights, where v3 builds an MLP per pass (`:742-745`, `:752-755`, `:769-772`,
`:791-794`). BID's drop and BUY_TRAIN's pass are outputs of their joint heads.
The logit-scale diagnostic, `pass_action_logit_abs`, is v3's (`:1750-1792`)
over 1889's layout and all nine `PHASES_WITH_PASS_SLOT`.

### 4.4 Fixed heads against entity readouts

Pointer heads score every choice that names a row: privates, corps, hexes,
candidates, train types, and port hexes. Joint heads take the rest:

- choices with no row: the passes, withhold and payout, and BUY_TRAIN's
  depot, pool, and bankruptcy;
- the price grids (BID, PAR, TRAIN_PRICE, PRIVATE_PRICE) and the sale
  blocks: a few ids whose meaning is fixed within the phase (PAR's index is
  a price; TRAIN_PRICE's points are ¥1, list, ⌈cash/3⌉, ⌈2·cash/3⌉, and
  cash), with the yen per id in PRICE 5 + id or CORP 15-19. v3 scores BID,
  PAR, DIVIDENDS, and ACQ_SELECT_PRICE the same way, jointly with per-option
  raw features (`:1339-1533`). A shared per-point scorer would add structure
  where there is no row to point at.

### 4.5 The unified block

The heads fill the 19 blocks in `DecisionPhase` order and concatenate them to
228 (`games/g1889.py:50-61`). At construction the module checks each block
against `PHASE_ACTION_SIZES` and the total against the layout's
`unified_dim`, as v3's `_validate_policy_layout` does (`:1018-1120`), and the
row widths against `get_token_widths(max_players, 1)`, as `_validate_layout`
does (`:226-256`). Logits become fp32 before the −1e9 mask (`:1727-1735`).

The heads hold 5.26M parameters at d = 256 against v3's 3.17M: 19
phases, and the context in every input. They cost 2.7% of a forward's FLOPs.

## 5. Model size

### 5.1 v3's size

**Measured** (`params.py`): v3 at v3-10-01-26 has **17,159,889
parameters**, so the user's "about 17M" is right as built. 1,048,846 of them
are `RelationInputMixing`'s, which the config turns off, leaving **16.11M
active**:

| Group | Parameters | Share |
|---|---|---|
| trunk: SwiGLU | 8.85M | 51.6% |
| trunk: attention | 3.93M | 22.9% |
| policy heads | 3.17M | 18.5% |
| relation input mixing, unused | 1.05M | 6.1% |
| projections, embeddings, value head, norms | 0.16M | 1.0% |

v2 at `bigger-multi.json` (d 320, 16 layers, 5 heads, ff_mult 2.67),
PIPELINE.md's active v2 run, has 21.41M.

### 5.2 FLOPs per evaluation

Per state, a block's token-wise matmuls (QKV, output, SwiGLU) cost
2T(4d² + 3d·d_ff) FLOPs and scale with T; attention (QKᵀ and AV) costs 4T²d
and scales with T². At ff_mult 3 their ratio is 2T/13d: 3.5% for v3
(T = 59, d = 256) and 5.7% for 1889 (T = 95). The T² term is small, so at
equal size 1889 costs close to 95/59 = 1.61 times v3 per state.

**Measured** (`cost_model.py`: matmuls by `FlopCounterMode` on one state,
attention by formula because the counter misses CPU SDPA; `size_grid.py`
extends the formula, which matches the counts within 0.5%):

| Model | Trunk | All parameters | GFLOP per state | of which attention | × v3 |
|---|---|---|---|---|---|
| v3, 256 × 15, 4 heads, T = 59 | 12.8M | 16.1M active | 1.62 | 0.05 | 1.00 |
| 1889, 192 × 12 | 5.8M | 8.9M | 1.24 | 0.08 | 0.77 |
| 1889, 256 × 15, 4 heads | 12.8M | 18.4M | 2.69 | 0.14 | 1.66 |
| 1889, 320 × 16, 5 heads | 21.3M | 30.0M | 4.42 | 0.19 | 2.73 |
| 1889, 384 × 18, 6 heads | 34.5M | 47.0M | 7.07 | 0.25 | 4.38 |
| 1889, 384 × 20, 8 heads | 38.4M | 50.9M | 7.83 | 0.28 | 4.85 |
| 1889, 512 × 20, 8 heads | 68.2M | 90.3M | 13.78 | 0.37 | 8.53 |

The proxy fed `c` to every head, so its counts run 3d² high (0.2M at d = 256,
which makes v0 18.2M), and its pointers ran over every hex, so its FLOPs run
1.4% high: v0 as built has 18,218,872 parameters and 2.65 GFLOP per state, 1.64
times v3. At v3's shape, 1889's
model has 2.1M more parameters, all in the heads, and does 1.66 times the work per evaluation. The input projections are 0.15% of
it. Across widths and depths (`size_grid.py`; FLOPs per state against v3's,
with the trunk's parameters):

| | L = 12 | L = 15 | L = 18 | L = 24 |
|---|---|---|---|---|
| d = 192 | 0.77 (5.8M) | 0.95 (7.2M) | 1.13 (8.6M) | 1.49 (11.5M) |
| d = 256 | 1.34 (10.2M) | 1.66 (12.8M) | 1.98 (15.3M) | 2.61 (20.5M) |
| d = 320 | 2.07 (16.0M) | 2.56 (20.0M) | 3.05 (24.0M) | 4.04 (32.0M) |
| d = 384 | 2.96 (23.0M) | 3.66 (28.8M) | 4.36 (34.5M) | 5.77 (46.0M) |
| d = 512 | 5.21 (40.9M) | 6.45 (51.1M) | 7.69 (61.4M) | 10.16 (81.8M) |

The size at v3's cost per evaluation is about 192 × 12 or 192 × 15.

### 5.3 CPU forward

**Measured** (`cost_model.py`; fp32, inference mode, padded records; median ms
per batch, and the ratio to v3 at the same batch):

| Model | B = 64 | B = 128 | B = 256 | FLOPs |
|---|---|---|---|---|
| v3 | 225 | 425 | 858 | 1.00 |
| 1889, 256 × 15, 4 heads | 419 (1.86) | 830 (1.95) | 1,746 (2.04) | 1.66 |
| the same, 8 heads | 499 (2.22) | 1,031 (2.43) | 1,978 (2.31) | 1.67 |
| 1889, 320 × 16, 5 heads | 581 (2.58) | 1,167 (2.75) | 2,382 (2.78) | 2.73 |
| 1889, 384 × 18, 6 heads | 843 (3.74) | 1,695 (3.99) | 3,922 (4.57) | 4.38 |
| 1889, 384 × 20, 8 heads | 980 (4.35) | 1,969 (4.63) | 4,506 (5.25) | 4.85 |
| 1889, 512 × 20, 8 heads | 1,415 (6.29) | 2,899 (6.82) | 6,419 (7.49) | 8.53 |
| 1889, 256 × 15, 4 heads, no relation bias | 336 (1.49) | 675 (1.59) | 1,350 (1.57) | 1.66 |
| the same, 8 heads, no relation bias | 356 (1.58) | 687 (1.62) | 1,395 (1.63) | 1.67 |

- **Without the relation bias, time tracks FLOPs:** 1.5-1.6 times v3, under
  the FLOP ratio of 1.66.
- **The bias is the T² cost that matters.** It adds 0.3-0.45 of v3's time at
  4 heads and 0.6-0.8 at 8: a `(B, H, 95, 95)` tensor per layer, with
  H × 2,048 records scattered into it, about 90% of them padding (the median
  state has about 180, extraction-design §6.3). On the CPU that is 20-25% of
  the 1889 forward at 4 heads.
- **Larger models tilt toward the FLOP ratio,** as matmuls grow and the bias
  does not.

**v0's bias path** drops the padding and builds one dense tensor a layer (3.4).
**Measured** (`bias_bench.py`; the bias path alone, on synthetic records fitted
to extraction-design §6.3):

| Bias path per forward, 4 heads | B = 64 | B = 256 |
|---|---|---|
| padded, three dense tensors a layer (v3's) | 80 ms | 411 ms |
| padded, one dense tensor | 68 ms | 325 ms |
| flat, three dense tensors | 21 ms | 166 ms |
| **flat, one dense tensor (v0)** | **13 ms** | **106 ms** |

The first row reproduces the bias's cost in the table above, 83 and 396 ms.
At B = 256 flat records are 55K values against 524K, and doubling the median
to 360 records gives 15 and 114 ms. So a forward at B = 256 should take about
1,450 ms against 1,746, with the bias at 7% of it.

**On the GPU,** v3's forward is matmul-bound: 1.62 GFLOP × 256 states in
4.35 ms model-only (`v3-cloud-profile.json`, `v3_no_mixing` at batch 256),
about 95 TFLOP/s by this count. So FLOPs should predict GPU time better than
they predict CPU time, plus the bias's memory traffic, about 0.07 MB per state
and layer in bf16 at 4 heads on v0's path. **Test:** model-only time on the
5090 at batch 64 and 256, 1889 at 256 × 15 against v3, measured as
`v3-cloud-profile.json` measured v3; expect 1.7-2.0 times. Time the padded path
too, to confirm the gain: every padded record in a state and head lands on one
cell, which may contend on a GPU.

### 5.4 Game length and self-play cost

**Measured** (`game_length.py`, `fixture_length.py`, `tb_tags.py`), searched
decisions per game:

| | 3 players | 4 players | 5 players |
|---|---|---|---|
| RSS, random playouts (mean) | 164 | 199 | 232 |
| RSS v3 self-play, epochs 19-36 of the v3-10-01-26 run: mean (range) | 336 (319-357) | 458 (446-478) | 584 (554-611) |
| 1889, random playouts (mean) | 653 | 701 | 794 |
| 1889, 18xx.games fixtures through the engine | 500 (314, ended by a bankruptcy) | 439 (962, ended by the bank) | |

- The fixture counts are the decisions in the driver's history with two or
  more policy actions under the self-play flags: what self-play would have
  searched. The two `hs_*` fixtures end by hand after 61-66 and are left out.
- Random 1889 play churns its stock rounds: STOCK is 61% of its decisions,
  against 53% and 46% in the fixtures. RSS self-play games run 2.0-2.6 times
  its random ones, but 1889's random games already outrun its human ones, so
  that ratio does not carry over.
- **Estimate:** 1889 self-play lands between the human games and random
  play, 440-800 decisions, 1.0-1.7 times RSS's ~460. The planning number is
  600, 1.3 times. **Test:** `self_play_*/game_length_mean` over the first 20
  epochs (Section 7.1 rescales from it).
- **Branching is alike** (`branching.py`): 4.1 policy actions per decision in
  1889 random play (median 3, p90 8, max 26) against RSS's 3.6 (median 2, p90
  7, max 15), so v3's simulation counts buy a similar search per decision.

**RSS self-play is GPU-bound.** In the same epochs (866-1,007 simulations), a
500-game epoch took 1,890-2,080 s (mean 1,990); a game made 459 searches and
149K network evaluations on average, about 325 per search, the rest of each
search's simulations replayed through subtree reuse (about 605 virtual
backups per search). The eval server ran 38.3-38.8K evaluations a second at
a mean batch of 177-187 and 9.0-9.4 ms a batch, never idle, with the workers
waiting on it 99% of the time. So self-play makes about **415K samples and
900 games an hour.**

For 1889 at the same simulations, a game costs the decision ratio times the
per-evaluation ratio, if subtree reuse holds evaluations per search near
RSS's:

| v0 size | Per evaluation | Per game (1.0-1.7 × decisions) | Samples an hour | Games an hour at 600 |
|---|---|---|---|---|
| 192 × 12 | 0.77 | 0.8-1.3 | ~540K | ~900 |
| 256 × 15 | 1.66-2.0 | 1.7-3.4 | 205-250K | 340-415 |
| 384 × 18 | 4.4 | 4.4-7.5 | ~95K | ~155 |

- The per-evaluation ratio is the model's. The server's 9.2 ms per batch is
  about twice v3's model-only time, so part of each evaluation is outside the
  model, at a ratio no CPU run measures. **Test:** the server's evaluations
  a second in the first 1889 epochs, against v3's 38.7K.
- **Wall clock:** RSS's 300 epochs take 8-9 days on the 5090 with the ramp
  to 3,200 simulations (epochs of 1,950-2,860 s across the two muon v3 runs).
  At 256 × 15, 1889's take 2-4 weeks; at 384 × 18, 5-9 weeks.

### 5.5 What the network computes

Extraction already does the graph work, so the network learns the rest:

| Question | Source |
|---|---|
| Where can a corp lay track, and which centers does it reach? | GRAPH, every corp: relations 16-19, CORP 40-43, HEX 94-101 |
| A corp's best route now | the solver: CORP 34-35 |
| What a lay does for the decision corp | the candidates: CANDIDATE 101-105 |
| Whose reach a lay grows | CANDIDATE 106-107 |
| What a token, a purchase, or a discard does | HEX 104, TRAIN 10-12 |
| Which hexes track joins | TRACK_LINK, and the static side planes for direction |
| Where two or three lays from the frontier lead | **learned**, hop by hop over the side planes; build distance (7b) supplies it |
| What a lay gives the other corps' routes | **learned** (extraction-design §13 item 5) |
| Stock play, the train rush, privates, blocking | **learned**, the kind of reasoning RSS v3 learns at its size |

**Depth.** **Measured** (`map_hops.py`): the map's 52 hexes span 11 hops at
most (radius 6, mean pairwise 4.3); a revenue center lies 1.1 hops from its
nearest and 2.0 from its third nearest, and homes average 5.2 hops apart.
Within 3 hops of a hex lies 40% of the map. One layer is one hop, so 15
blocks cover the diameter and leave ten or more for composing a horizon of
two or three lays with everything else. Depth does not bind v0 on the map,
and build distance will turn multi-lay distance into one hop.

**Parameters are not tokens.** A transformer applies its weights to every
row, so a bigger board costs computation, not parameters, as AlphaZero used
one 20-block, 256-filter network for 8 × 8 chess and 19 × 19 Go. At v3's
parameter count, 1889's 95 rows already get 1.66 times v3's computation. The
question for parameters is whether 1889's concepts need more of them than
RSS's, and extraction took the costliest concepts out of the network:
routes, reach, lay legality, and lay outcomes.

### 5.6 Outside evidence

- **KataGo** (Wu 2019, "Accelerating Self-Play Learning in Go") began its run
  at 6 blocks × 96 channels and moved to 10 × 128, 15 × 192, and 20 × 256,
  each larger net trained on the self-play data so far and switched in once
  stronger. Early on, the data's quality limits strength, and small nets
  make data faster.
- **Jones** (2021, "Scaling Scaling Laws with Board Games") found for
  AlphaZero on Hex that the compute-optimal model grows with the compute
  budget, so smaller models win at smaller budgets.
- **RSS** grew its transformer from 192 × 10 to 256 × 15 (`f4efd24`) and v2
  to 320 × 16 at 21.4M (`bigger-multi.json`). v3 went back to 256 × 15 at
  16.1M active and plays v2 about even: in 24 four-player games, v3 at epoch
  254 finished ahead of v2 at epoch 324 in 11 (`v3-vs-v2.md`), with fewer
  epochs and different inputs. A model a third larger, its trunk 1.6 times
  v3's, bought no clear strength at this scale.
- **Data per hour:** at 256 × 15, v0 makes about 250K samples an hour, 6M a
  day; 384 × 18 makes about 95K.

### 5.7 Recommendation and the capacity probe

**v0 is v3's shape: d = 256, 4 heads, 15 layers, ff_mult 3.0.** 12.8M in the
trunk and 18.2M in all, at 1.66 times v3's FLOPs per evaluation.

The user's intuition that 1889 needs more than RSS is right about
computation and, on present evidence, wrong about parameters:

- 1889 gets more computation without more parameters: 1.66-2.0 times v3's
  per evaluation and 1.7-3.4 times per game.
- Nothing measured here says it needs more parameters. Extraction does the
  graph search, 15 layers span the map, and RSS saw no gain from a third
  more. Self-play is GPU-bound, so parameters cost data: 384 × 18 cuts
  samples an hour 2.6 times against v0 and stretches the run to 5-9 weeks.
- What would prove it right is capacity binding on v0's own data, which the
  probe measures.

Not 192 × 12, v3's cost per evaluation: RSS grew past 192 × 10 in April
(`f4efd24`), its trunk is 45% of v3's, and v0 is the baseline for the
build-distance comparison, where a capacity-starved baseline would inflate
the features' gain.

**The capacity probe** (8.6), at epochs 50 and 150 of the run: train on v0's
saved replay buffer, offline, with the last two epochs' rows held out.

- Models: two seeds of 256 × 15 for the noise floor, 192 × 12, 384 × 18
  (6 heads), and 256 × 15 with 8 heads.
- The same steps, batch, optimizer, and LR schedule for each (for example
  40K steps at 256).
- Measures on the held-out rows: value MSE and policy KL (cross-entropy minus
  target entropy), overall, by player count, and for TRACK, TILE, STOCK, and
  BUY_TRAIN.

**Criterion:**

- **Grow** to 384 × 18 if it beats 256 × 15 on both measures by more than
  twice the gap between the two seeds and by at least 3%: capacity binds.
  Switch self-play to it, warm-started from its probe weights as KataGo
  switched nets, and confirm by strength per GPU-hour once 1889 tournaments
  exist (milestone 6).
- **Shrink** to 192 × 12 if, at epoch 150, it is within the seed gap of
  256 × 15 on both: v0 is larger than its data needs, and the smaller net
  self-plays 2.2 times faster.
- **8 heads** if they beat 4 by the same margin; they cost about 10% on the
  CPU on v0's bias path, and 13-24% on padded records (**Test:** on the GPU).
- Otherwise keep v0.

The probe costs only training, hours of GPU time against weeks of self-play
(**Test:** the probe's steps a second on the 5090).

## 6. Pipeline changes

### 6.1 Relation transport per game

Replay and the trainer build dense planes for every sample
(`train/replay_buffer.py:356-369`, `train/replay_buffer.py:419-424`;
`train/trainer.py:252-255`, `train/trainer.py:400`), as do `NNEvaluator`
(`mcts/evaluator.py:276-298`, `mcts/evaluator.py:376-382`,
`mcts/evaluator.py:421-430`, `mcts/evaluator.py:500-509`) and the
single-process compile warmup (`train/main.py:1114-1117`). For 1889 a
sample's planes are 225 KB, 115 MB per trainer batch of 512 and 462 MB as
fp32, against 8 KB of records (extraction-design §6.4, §9).

Extraction-design §13 item 11 chose records for 1889, per game. The change:

- **`Game.relation_format`:** `"dense"` for RSS, `"flat"` for 1889. A
  per-game constant, not a config field: it follows from the game's sizes,
  and RSS must not move.
- **`flatten_relation_records`:** one helper from `(B, E, 4)` uint8 records to
  `(N, 5)` int16 (3.4), used by every path below.
- **Replay:** `sample` and `sample_into` fill `(B, R, T, T)` with
  `fill_relations_dense_batch` or `(B, E, 4)` with
  `fill_relations_sparse_batch`, which the caller flattens.
- **Trainer:** `_ensure_scratch` sizes `_rel_h` and `_rel_d` by the format, up
  to B × E rows, and the forward call gets the first N.
- **`NNEvaluator`** follows the field too. In single-process mode one compiled
  model serves the trainer and `NNEvaluator` (`train/main.py:1087-1124`), so
  both must send one format, and the warmup builds its dummy relations in it.
- **The eval server** follows the field too: it flattens on the CPU as it packs
  a batch, so the GPU receives no padded records, only real ones, rounded up to
  a launch shape in bucketed mode (6.5).

RSS keeps its dense planes and padded records, so its code paths and numerics
are as today. **Test:** for 1889, `sample_into`'s flat records equal the
flattened `fill_relations_sparse_batch` on the sampled rows, and a trainer step
on them gives the loss of a step on dense planes, to fp32 tolerance.

### 6.2 Layout id, model path, and the module contract

- **Already enforced:** the loader requires `model_path` and
  `INPUT_LAYOUT_VERSION` (`nn/__init__.py:155-158`, `nn/__init__.py:197`) and
  sends a game other than RSS to `build_model` (`nn/__init__.py:161-172`,
  `nn/__init__.py:265-268`); the adapter takes layout 1 only
  (`tests/test_model_factory_checkpoint.py:288-297`). The module declares
  `INPUT_LAYOUT_VERSION = 1` and `build_model(config, spec)`.
- **`build_model`** asserts the spec: game "1889", layout 1, `token_dim` 108,
  `num_tokens` 90 + `num_players`, `policy_dim` 228, `value_dim` =
  `num_players`. It reads `d_model`, `num_heads`, `num_layers`, `ff_mult`, and
  `zero_sum_values`. Those fields have `TrainingConfig` defaults
  (`train/config.py:163-171`), which v3's config relies on, so 1889's config
  sets them (Section 7.1). The v2- and v3-only fields (`d_proj`,
  `phase_conditioning`, `price_slot_fourier_bands`, `relation_input_mixing`)
  are ignored, as PIPELINE.md says of v2's and v3's.
- **Duck-typed members** (pipeline-refactor §5): `cfg.layout_version` and
  `cfg.num_players`, from which `NNEvaluator` infers its spec
  (`mcts/evaluator.py:42-65`); `value_head`; `input_output_layers()`, the
  projections with `tile_proj` and each head's last Linear (as `:1798-1822`);
  and `pass_action_logit_abs`.
- Checkpoints store the config, game "1889" included, so
  `load_model_from_checkpoint` rebuilds through `build_model`.

### 6.3 Static planes

`static_relation_planes(cfg.num_players)` is called at construction and
registered with `persistent=False`, as v3 registers its synergy planes
(`nn/transformer-v3.py:703-710`), as are the relation scales and the port
masks. They are not in checkpoints, so a change to the planes changes a
loaded model, as with v3's; `tests/g1889` pins them against the tables.

### 6.4 What extraction asked of step 8

Extraction-design §13 item 10, adopted:

- N from the input's shape; a dense input's planes past 25 are sliced off;
- relation ids past 24 carry no weight;
- the static planes from `static_relation_planes`;
- `is_selected` and the decision corp as token-data-1889.md's "Conventions"
  give them: at most one row per slot, zero when none.

**Test** (8.1): appending records with ids 25 and up, or value-0 records,
leaves the outputs unchanged.

### 6.5 Eval server and IPC

Sizes come from the spec and the `Game` (`train/eval_server.py:254-292`,
`train/eval_server.py:646-684`); the code changes are flattening (6.1) and
rounding N up in bucketed mode.

- **Per slot:** tokens 20.5 KB as fp16, records 8 KB of capacity, the mask
  228 B, and priors 0.9 KB, about 30 KB against RSS's 14 KB.
- **Launch shapes:** bucketed mode keeps CUDA graphs on, on the condition
  `train/gpu/nvidia.py` states, a small repeated set of launch shapes, since
  Inductor records a graph for each distinct set of shape inputs. So a bucketed
  launch ships B × K flat rows, with B its bucket and K the smallest of 256, 512,
  1,024, and 2,048 that holds its real records. This always fits: a state has at
  most 2,048. Dynamic mode runs without CUDA graphs and ships the real N.
- **Filler:** the rows past the real records have value 0, which the model gives
  no weight (3.4), so outputs are bit for bit those without filler. They come
  from a table built once, 2,048 rows a state, each on a cell of its own, and a
  launch copies a prefix, whose batch ids stay below B. **Measured**
  (`scratchpad/g1889_relation_transport/`): the copy takes 2.5-17 µs at batch 256,
  and the design's forward at batch 128, uncompiled on ROCm, takes 106.8 ms on
  real N, 107.0 ms at K = 256, and 108.4 ms at K = 1,024.
- **N stays a dynamic dimension in both modes:** the warmup
  (`train/eval_server.py:606-625`) marks it with `mark_unbacked`, as `dynamic`
  mode marks the batch, so one compiled graph takes every K. Its records have a
  launch's shape, (B, 256). Inductor needs nothing more from it: it runs a
  shape's first launch without a graph, records its second, and replays from its
  third, in the serve loop, as with RSS's buckets.
- **Shapes a run can reach:** at the cap of 256, 9 buckets × 4 Ks, or 36 pairs.
  Dynamo compiles batch 1 apart from the rest, so the most one compiled graph
  records is 32, under Inductor's warning at 50. **Measured**
  (`launch_distribution.py`; 188K launches of epoch-0 self-play, up to 32 workers,
  on a stand-in model): 18 pairs, with K = 256 on 65-100% of a bucket's launches
  and K = 1,024 only at B ≤ 32, and 143 real records a leaf.
- **In all:** 128 workers × 8 leaves come to about 31 MB of shared memory
  (RSS 14 MB), the pinned buffers at batch 256, two deep, to about 22 MB, with
  room for a full batch's flat rows (20.5 KB a state), and a batch of 256 sends
  5.97 MB to the GPU at K = 256, 0.66 MB of it flat rows, or 6.62 MB at K = 512.
- **Workers:** extraction costs 4.9 µs median a state (extraction-design §9),
  small beside a GPU evaluation every ~45 µs.
- **Test:** on the 5090 in bucketed mode under self-play, with no graph breaks
  and no recompiles as N varies: launch latency by (B, K) settles once a shape
  replays; GPU memory stops growing once the shapes in use are recorded; the
  count of recorded graphs stays at the (B, K) pairs reached; and compiled
  outputs match eager on the same launches, as `v3-cloud-profile.json` compares
  v3's.

### 6.6 Tooling

- **A 1889 `SelfPlayStats`** (`games/g1889_selfplay.py`), logged for each
  reporting group as `self_play_<group>/<name>`: the end reason (`end_bank`,
  `end_bankruptcy`), the last game phase (`last_phase_<phase>`), ORs started
  (`ors_played`), the share of TRACK decisions that pass (`track_pass_share`),
  corps floated (`corps_floated`), and new trains bought by type
  (`trains_bought_<type>`). It is small, and a first run is hard to read without it.
  - The driver plays forced decisions without a search, so `observe` never sees
    them. ORs and trains therefore come from the state: ORs from each set's
    `or_count` and the last `or_num`, and trains from the depot, or for the
    unlimited D from the Ds in play. `track_pass_share` is over the TRACK
    decisions with a choice.
  - Trains bought from the pool or another corp move a train already counted.
- **Deferred,** as pipeline-refactor §1 plans: `train/analyze_game.py`,
  `train/tournament.py`, `train/debug_trace.py`, and `train/strategy_data.py`
  import the RSS engine directly (`train/analyze_game.py:31-36`,
  `train/tournament.py:46-57`). TensorBoard's per-phase policy loss and KL,
  value loss by player count, and game length already work for 1889. The
  comparison needs 1889 tournaments (Section 9); v0 needs none.

### 6.7 RSS guard

- `nn/transformer-v2.py`, `nn/transformer-v3.py`, `core/`, `entities/`, and
  `phases/` do not change.
- RSS's `relation_format` is dense, so replay, the trainer, `NNEvaluator`,
  and the warmup run RSS through today's code.
- `tests/test_v2_parity.py` passes on its fixture, and the RSS suite stays
  green.

## 7. The first run

### 7.1 Config

`train_configs/1889-v0.json` is v3-10-01-26 with these changes:

| Field | v3-10-01-26 | 1889-v0 | Why |
|---|---|---|---|
| `game` | (RSS) | `"1889"` | |
| engine options | `v3_behavior`, `acq_same_president`, `max_acq_price_actions` | `private_sale_same_president` true, `train_sale_same_president` true, `auction_pass_payout_cap` 1 | 1889's self-play flags, stated |
| `model_path` | `nn/transformer-v3.py` | `nn/transformer-1889.py` | |
| `d_model`, `num_heads`, `num_layers`, `ff_mult`, `batch_size`, `learning_rate` | defaults: 256, 4, 15, 3.0, 256, 0.001 | the same values | stated, not defaulted |
| `relation_input_mixing` | false | absent | v3 only |
| `buffer_capacity` | 750,000 | 1,000,000 | 3.3 epochs of samples, as v3 |
| `training_steps_per_epoch` | 1,000 | 1,300 | v3's replay ratio, 1.1 |
| `temp_anneal_starts`, `_ends` | [60, 80, 100], [120, 160, 200] | [100, 100, 115], [200, 200, 230] | 17.5% and 35% of the game, as v3 |
| `policy_target_temp_anneal_starts`, `_ends` | as `temp_` | as `temp_` | inert: the target temperature is 1.0 throughout |
| `seed` | (42) | 42 | stated: the comparison reruns it |
| `checkpoint_dir`, `tensorboard_dir` | `*/v3-10-01-26` | `*/1889-v0` | |

The rest stays at v3-10-01-26's: 500 games and 500 epochs, simulations
ramping from 800 to 3,200 over epochs 10-300, search batches of 8, 128
workers and one eval server, bucketed batches of 8-256 in bf16, Muon with
the rate decaying to 1e-4 by epoch 300, weight decay 0.01, clipping at 5, prioritized replay 0.5 and 0.5, dynamic Dirichlet noise
(numerator 5, ε 0.15), forced playouts at k = 2 with raw-prior pruning,
c_puct 1.7, the value blend from epoch 10 to 300 up to 0.7, terminal blend
0.75, and zero-sum values.

- **Temperature** is keyed to move numbers. v3's windows sit at 17-18% and
  34-36% of its self-play length at each player count (60-120 of 336,
  80-160 of 458, 100-200 of 584). 1889's planning lengths are 575 and 570
  at 3 and 4 players, midway between the fixtures and random play (Section
  5.4), and 650 at 5, where there is no fixture, from random play's 13% rise
  from 4 players to 5; 17.5% and 35% of them give the values above.
- **Samples per epoch:** at ~600 decisions, 500 games give ~300K samples
  against v3's ~230K. Buffer and steps scale to keep v3's per-sample
  dynamics: a row lives about 3.3 epochs, and each epoch trains 1.1 times its
  samples (v3: 1,000 × 256 / 230K). Games per epoch stay at 500, so an epoch
  keeps v3's count of game outcomes.
- **Simulations** stay: branching is alike (Section 5.4), and the price is
  wall clock.
- **Memory:** a replay row is about 2.9 KB (an 850-slot int16 state, the
  mask, and the target), 2.9 GB at a million.

**Test**, at epoch 20, on resume: reset the windows to 17.5% and 35% of
`self_play_<n>p/game_length_mean`, and the buffer and steps to the measured
samples per epoch, by the same rules. A new `buffer_capacity` makes the resume
skip the saved buffer with only a warning (`train/PIPELINE.md`), so that epoch
starts its buffer empty unless the rows are carried over by hand.

### 7.2 Where it runs

- **The run:** the cloud RTX 5090, compiled, evaluating in bf16, as v3's.
- **Smoke tests:** the local AMD ROCm GPU, with
  `train_configs/1889-v0-smoke.json`: `eval_dtype` null (fp32), 8 workers,
  `batch_size` and `eval_max_batch_size` 64, 12 games an epoch, 64 simulations
  without the ramp, a buffer of 20,000 with `min_buffer_size` 512, and 2 epochs,
  run with `--no-compile`.
  - The validator holds `eval_max_batch_size` to workers × leaves, 8 × 8 here.
  - The GPU may be in use by the live bot, which runs from another worktree and
    leaves 3-5 GB of its 17 GB free. A fp32 training step holds 4.7 GB at batch 128.
  - The local box's trouble with fp32 dense relation batches above ~150 does
    not arise: 1889 trains on records.
- **Measured** (`scratchpad/g1889_smoke/`, 2026-11-04): 2 epochs and a resume into
  a third, 36 games beside the live bot, near-uniform priors throughout. It tests
  the pipeline; 64 simulations and 3 epochs say nothing of learning.

  | | 3 players | 4 players | 5 players |
  |---|---|---|---|
  | Decisions a game, by epoch | 616, 938, 847 | 795, 1,022, 966 | 925, 969, 1,240 |
  | Decisions an SR (last 20 games) | 36 | 58 | 73 |
  | Ended at the bank (12 games each) | 8 | 7 | 5 |
  | ORs started, by epoch | 23, 30, 26 | 19, 19, 20 | 18, 16, 18 |

  - **Length:** games run past Section 5.4's 440-800, and STOCK is 56-71% of
    decisions. An SR ends only when every player passes in a row, and these
    priors pass 38% of the time, so SRs run long; a random player who passes over
    half the time takes 12, 22, and 34 decisions an SR. The epoch-20 rescale (7.1)
    reads trained play instead.
  - **Endings:** the last phase was 4 in 5 games, 5 in 17, 6 in 4, and D in 10.
    Track passes are 18-34% by group and epoch, and 4.3-7 corps float.
  - **Rare phases:** EMERGENCY_SELL_AMOUNT was searched twice in 36 games, as
    random play's 0.04-0.10 a game predicts.
  - **Cost:** the server ran 707-872 evaluations a second at a mean batch of 23-25
    and 27-35 ms a launch, against the 1,198 a second of a full batch of 128 (6.5).
    A search made 40-42 evaluations and 22-23 virtual backups. Workers waited on
    the server 99.5% of the time. Whole epochs made 59-71K samples and 58-88 games
    an hour.
  - **Memory:** free VRAM fell from 5.3 GB to 4.6 GB with the eval server up and
    to 1.7 GB while training.

## 8. Sub-steps

Each sub-step is a commit or a short series, leaves pytest and pyright green,
and leaves `tests/test_v2_parity.py` passing on its fixture.

| # | Scope | Acceptance |
|---|---|---|
| 8.1 | `nn/transformer-1889.py` (Sections 3-4), tested in `tests/g1889/test_model.py` | on random-playout states at 3-5 players, at capacities 3 and 5 and mixed: legal logits finite and −1e9 elsewhere; dense and flat inputs agree within fp32 tolerance, also with extra records of ids 25 and up and with value-0 records; a batch equals its states run alone; permuting the visible candidate rows, with their records, permutes TILE's logits; with the trunk's output held fixed, each pointer logit (TRACK and TOKEN 1+i, STOCK's four for corp c, SELECT_TRAIN's t) moves with its own row's raw slots and no other row's; zero-sum values sum to zero over real players, with padded slots zero; construction checks widths and blocks; parameters as Section 5.2 |
| 8.2 | relation transport per game (6.1): the `Game` field, flattening, replay, the trainer, `NNEvaluator`, the eval server, and the warmup | RSS's replay, trainer, and evaluator tests unchanged; 1889's `sample_into` records equal the direct fill; on CPU, a trainer step on a 1889 buffer of random-playout rows runs, matches its dense-plane twin, and overfits a fixed batch |
| 8.3 | the 1889 `SelfPlayStats` (6.6) | its scalars appear in TensorBoard after a 1889 smoke run |
| 8.4 | the configs (Section 7); a local smoke run | 2 epochs at 3-5 players complete through self-play, the eval server, training, a checkpoint, and a resume |
| 8.5 | the cloud run: compile and profile on the 5090, then launch | Section 5.3's and 6.5's GPU Tests recorded here; no graph breaks or recompiles; the epoch-20 rescale (7.1) done |
| 8.6 | the capacity probe (5.7): a training-only script on the saved buffer | its table recorded here at epochs 50 and 150, and the size decided by the criterion |

## 9. Hooks for build distance and the comparison

Milestone 6 compares v0 with v1, which adds build distance (7b), under
matched configs. v0 leaves:

- **Relation ids and E.** v0 consumes ids 0-24 by a module constant, gives
  later ids no weight, and takes N from its input. 7b's relations (ids 25 and
  up, E = 4,096: extraction-design §8) reach v0 unchanged, so v0's
  checkpoints keep playing on 7b's extraction.
- **Layouts.** v0 reads layout 1 and v1 reads 7b's layout 2. Each evaluator
  fills its own model's layout (`spec.layout_version`), so the two can share
  a game.
- **Constants, not code.** v1 changes the consumed relation count, the bias
  channels, and the layout id. New channels start inert, since the bias is
  zero at initialization; the comparison trains v1 from scratch, and a warm
  start from v0 stays possible later.
- **Matched configs.** v1's config is v0's with `model_path` and the
  directories changed, the seed included.
- **Readouts.** From epoch 0, v0 logs what the comparison reads: per-phase
  policy loss and KL (TRACK, TILE, TOKEN), value loss by player count, and
  the stats hook's ORs, track passes, and end reasons. Strength needs 1889
  tournaments, which milestone 6 builds.
- **A fixed epoch.** Compare at epoch 100 or so, not 300: a run to 300 takes
  2-4 weeks (Section 5.4).

## 10. Decisions

The user settled these on 2026-11-04. Each item gives the options and the decision.

1. **The tile block.** (a) One `tile_proj` for HEX and CANDIDATE slots 2-77,
   and a projection per type for the rest. (b) A projection per type over the
   whole row. *Decided (a):* candidates learn tiles from 52 hex rows in
   every state (3.2).
2. **Identity embeddings.** (a) Hexes, corps, privates, and train types, with
   candidate rows taking their hex's. (b) (a) without the candidates'. (c)
   Hexes and corps only. *Decided (a)* (3.3).
3. **Relation input mixing.** (a) None. (b) v3's module over the 35 relations,
   behind a config flag. *Decided (a):* v3's run has it off, and it would
   cost about 2.8 blocks of FLOPs (3.5).
4. **Phase conditioning.** (a) In the input, as v3. (b) adaLN in each block.
   *Decided (a)* (3.5).
5. **Head inputs.** (a) The context `[GLOBAL, actor]`, with the decision corp
   wherever the head's phases have one, the choice's rows, and the raw
   per-choice slots (4.1). (b) Embeddings alone.
   *Decided (a),* v3's pattern.
6. **Interjection slots.** (a) One private scorer shared by BUY_PRIVATE_PRE,
   BUY_PRIVATE_POST, and STOCK's DR and MF slots. (b) A scorer per phase. *Decided (a)* (4.3).
7. **Pass slots.** (a) One context MLP with a column per pass. (b) An MLP per
   phase, as v3. *Decided (a):* each reads only the context (4.3).
8. **Price grids and sale blocks.** (a) Joint heads with the raw amounts. (b)
   A shared per-point scorer. *Decided (a),* v3's BID and PAR (4.4).
9. **TILE and ER_BONUS** share one candidate scorer. *Decided yes,*
   extraction-design §7.1's sketch.
10. **v0's size.** (a) 256 × 15, 4 heads, ff_mult 3: v3's shape, 1.66 times its
    FLOPs. (b) 192 × 12: v3's cost per evaluation. (c) 384 × 18, 6 heads: 4.4
    times. *Decided (a)* (5.7).
11. **Settling the size.** (a) The capacity probe on v0's buffer at epochs 50
    and 150, with 5.7's criterion. (b) A second full run at another size. (c)
    Neither. *Decided (a):* it costs training, not self-play.
12. **Attention heads.** (a) 4, as v3. (b) 8, for 35 relation channels.
    *Decided (a) in v0,* with (b) in the probe: about 10% slower on the CPU
    on v0's bias path, and no evidence yet of a gain.
13. **Where the transport choice lives.** (a) A `Game` field, read by replay,
    the trainer, `NNEvaluator`, the eval server, and the warmup. (b) A config field. (c) The
    model declares it. *Decided (a)* (6.1).
14. **The config.** (a) Section 7.1: v3-10-01-26 with buffer, steps, and
    temperature windows scaled to game length, rescaled at epoch 20. (b)
    v3's values unchanged. *Decided (a).*
15. **Tooling in step 8.** (a) The 1889 stats hook, with analysis and
    tournaments left to milestone 6. (b) Port `train/analyze_game.py` too.
    *Decided (a)* (6.6).
16. **The bias path.** (a) Flat records with no per-state padding, and one dense tensor a
    layer. (b) v3's padded records and three tensors. *Decided (a)* (3.4, 5.3):
    the same outputs at about a quarter of the bias's CPU time, for the 5090 to
    confirm.
