# Shikoku 1889 Model Design

Status: design for step 8 of [`pipeline-refactor.md`](pipeline-refactor.md),
dated 2026-11-03; the user reviewed it and settled Section 10 on 2026-11-04.
Sub-steps 8.1-8.4 are built: the model, the transport, the stats hook, the
configs, and a local smoke run with a resume (Section 7.2). The local half of 8.5
and 8.6 is built too: the profile, the rescale helper, the capacity probe, and the
runbook (Section 11). The 8.5 profile ran on the 5090 on 2026-11-05 (Sections 5.3,
5.7, and 6.5). What remains is the run and the probe, whose results these sections
record. Section 12 describes v1, `nn/transformer-1889-v1.py`, by its differences
from v0 (2026-11-10), and 12.7 what was checked before its launch, whose sequence
is Section 11.1. Step 8 builds the first 1889 model,
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

**Measured** (`1889-cloud-profile.json`, 2026-11-05, at 1a761f0; torch 2.13.0+cu132;
that method, with random weights on 2,048 random-play states; model-only ms a batch,
and the ratio named):

| | B = 64 | B = 256 |
|---|---|---|
| v3 | 1.60 | 4.12 |
| v0 (ratio to v3) | 2.29 (1.43) | 7.02 (1.71) |
| v0 on padded records, three tensors (ratio to v0) | 2.39 (1.05) | 7.50 (1.07) |
| v0 on padded records, one tensor (ratio to v0) | 2.39 (1.04) | 7.47 (1.06) |
| v0, 8 heads (ratio to v0) | 2.59 (1.13) | 8.51 (1.21) |

- **v0 against v3:** 1.71 at a full batch, the bottom of the expected range and
  about the FLOP ratio, and 1.43 at B = 64, below it. Section 5.4 takes the new cost.
- **v3 itself** came in under `v3-cloud-profile.json`'s 1.84 and 4.35 ms, which ran on
  torch 2.14.0. These ratios come from one process, so that does not touch them.
- **The padded path** costs 4-7% on this GPU, far from the CPU's gap, and one tensor
  against three makes no difference here. Flat records still win, by less.

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
- **Measured:** model-only, 1.43-1.71 (5.3), under the table's 1.66-2.0. If the server
  follows the model, that is 23-27K evaluations a second and 240-290K samples an
  hour, not 205-250K. The Test above still decides it.
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
- **The held-out rows.** Replay records no epoch per row. Its ring index gives
  write order, every game goes in whole, a row's state shows whether it is its
  game's first decision, and an epoch is `games_per_epoch` games, so the rows are
  the newest 1,000 games, cut between games. TensorBoard's
  `self_play_aggregate/total_examples` for those epochs gives the same count, which
  `train/capacity_probe.py` checks and reports. **Measured** on the smoke buffer's
  newest epoch: 12 games, 12,210 rows, both ways. A graceful stop mid-epoch leaves a
  partial epoch that TensorBoard does not count; those rows join the held-out
  games, and the report says the two differ.
- **Its training rows.** A buffer holds 3.3 epochs, so holding out two leaves about
  1.3 epochs of rows to train on, about 390K at 300K an epoch. 40K steps at 256 make
  about 26 passes over them, where a live row sees about 3.6. The probe prints the
  passes and scores the held-out rows every 10K steps, so a model that overfits
  before the end shows in its curve; `--holdout-epochs` and `--steps` change them.

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
  CPU on v0's bias path, and 13-24% on padded records. **Measured** on the 5090: 13%
  at B = 64 and 21% at B = 256 on v0's path (5.3).
- Otherwise keep v0.

The probe costs only training, hours of GPU time against weeks of self-play.
**Measured** on the 5090, at batch 256 over 60 steps (`1889-cloud-profile.json`):

| Model | Steps a second | 40K steps | Peak allocated |
|---|---|---|---|
| 256 × 15 (two seeds) | 13.2 | 0.84 h | 8.3 GB |
| 192 × 12 | 19.2 | 0.58 h | 5.1 GB |
| 384 × 18, 6 heads | 7.0 | 1.59 h | 14.9 GB |
| 256 × 15, 8 heads | 11.1 | 1.00 h | 8.9 GB |

So the five models train in about 4.9 hours, before their held-out scoring.

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
- **Since v1's step 9.6** (v1-design §6) a state has at most 4,096 records, of 29
  relations: a slot's records take 16 KB of capacity, the filler table and the
  pinned flat buffers double, and K can reach 4,096. Nothing in the server
  changed for it: K is the next power of two over a launch's mean real records
  a state, which the largest state seen (1,714 records, human phase D) keeps at
  2,048 or below, so the shapes a run reaches are the 36 above.
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
  on a stand-in model): 18 pairs, with K = 256 on 65-100% of launches at B ≥ 8
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
- **Measured** (`1889-cloud-profile.json`; compiled, bf16, every launch shape, on
  random weights and random-play states, not self-play's traffic, which the run's
  first epochs test):
  - **Compile:** no graph breaks and two compiled graphs, batch 1 and the rest, in
    244 s of warmup. Its one guard failure is `x`'s batch going from 1 to 2.
  - **Recording:** 36 graphs for the 36 pairs, one each. K = 256 records on its first
    launch in the serve loop, since the warmup ran that shape, and the other Ks on
    their second. No launch took over 46 ms.
  - **Settled latency** at every pair; K = 2,048 costs at most 9% over K = 256:

    | ms a launch | B = 1 | B = 8 | B = 64 | B = 128 | B = 256 |
    |---|---|---|---|---|---|
    | K = 256 | 1.11 | 1.46 | 3.26 | 5.30 | 10.16 |
    | K = 2,048 | 1.11 | 1.49 | 3.37 | 5.75 | 11.12 |

  - **Memory:** over 7,200 launches, reserved stays at 992 MB and device free at
    30.2 GB of 32.1, with no new graphs or compiles.
  - **Against eager**, by the most on any launch, with the limits in brackets: logits
    0.0066 at launches 1-3, 0.0061 at replay, and 0.012 after a weight sync (0.05);
    values 0.0036 (0.01); policy TV 0.0027 (0.01). Filler changes nothing, exactly,
    across K.
  - **The profile's `FAIL` on recompiles, 36, was the counter's.** It counted Dynamo's
    `on_compile_end` callbacks, which Inductor also runs for each graph it records
    (`CallbackTrigger.CUDAGRAPH_RECORDING`, in `cudagraph_trees.py` in torch 2.13 and
    2.14). So the 36 are the 36 recordings: the whole section took 3.7 s where a
    compile takes minutes, and the count stayed flat over the 7,200 launches after it.
    The profile now counts `DYNAMO` alone. `TORCH_LOGS=recompiles` at launch (Section
    11) is the direct check.

### 6.6 Tooling

- **A 1889 `SelfPlayStats`** (`games/g1889_selfplay.py`), logged for each
  reporting group as `self_play_<group>/<name>`: the end reason (`end_bank`,
  `end_bankruptcy`), the last game phase (`last_phase_<phase>`), ORs started
  (`ors_played`), the share of TRACK decisions that pass (`track_pass_share`),
  the share of DIVIDEND decisions that withhold (`withhold_share`, and
  `withhold_share_<phase>` by game phase, added at epoch 20), corps floated
  (`corps_floated`), floated corps' mean treasury at the game's end (`corp_cash_end`,
  added with them), and new trains bought by type (`trains_bought_<type>`). It is
  small, and a first run is hard to read without it.
  - The driver plays forced decisions without a search, so `observe` never sees
    them. ORs and trains therefore come from the state: ORs from each set's
    `or_count` and the last `or_num`, and trains from the depot, or for the
    unlimited D from the Ds in play. `track_pass_share` and `withhold_share` are
    over the decisions with a choice; a corp with no revenue withholds unseen.
  - `withhold_share_<phase>` is absent, not 0, where a phase had no such decision.
  - Trains bought from the pool or another corp move a train already counted.
  - **Human baselines**, over the 87 scraped 3-5 player games: `withhold_share` 12%
    (14%, 11%, and 9% at 3, 4, and 5 players), and by phase 2 to D 1%, 10%, 17%, 19%,
    20%, and 4%; `corp_cash_end` ¥147 (¥169, ¥129, and ¥94).
- **Added for v1** (v1-design.md §4), per game unless a rate:
  - the auction: the premium paid over face for the privates (`auction_premium`),
    and the players' mean cash as SR1 opens (`auction_sr1_cash`, over players);
  - pars: the mean (`par_price`) and the share at ¥65 (`par_65_share`), over the
    pars made; and the share of SRs that end with a corp parred but not floated
    (`par_stranded_share`, over SRs ended);
  - treasuries: a corp's as it floats (`corp_cash_float`, over floats: 10 x par),
    and the richest floated corp's once the last 4 has sold and the 5s are next
    (`corp_cash_max_5s`);
  - the shares, in 10% units, players sell in SR2 (`sr2_shares_sold`), and the
    station tokens placed besides home tokens (`tokens_placed`);
  - the game OR in which each later phase starts, counted as `ors_played`
    counts (`phase_start_or_<phase>`, phases 3 to D), absent for a phase not
    reached;
  - MF's port tile: whether it is laid (`mf_port_laid`), and, where it is, the
    SR number of the round it goes down in, SR k or the ORs after it
    (`mf_port_sr`), and whether that is an OR (`mf_port_in_or`).
  - These too come from the state, between consecutive observations or at the
    end. Two are as of the next searched decision: the treasury at float, which
    is still 10 x par there, and `corp_cash_max_5s`, which forced and automated
    steps can move first, as an OR's start pays privates' revenue to corps (by
    ¥5-40 in 11 of 289 random playouts, ¥25 in 1 of 40 v0 games, and in none of
    the scraped games). A sale's price comes from the buyer's bid or the
    private's price, not from cash: forced passes can pay the privates out
    between two sales the collector sees together. Checked against every row of
    the driver's history on random playouts, 41 v0 games, and the 100 scraped
    games, the rest match exactly.
  - Each is absent where the game had none of what it counts, and in a game
    started from a mid-game position (v1-design.md §2.1) where what it counts
    began before the first observation: the auction stats unless no private had
    sold, `sr2_shares_sold` and `phase_start_or_<phase>` unless it starts before
    SR2, `tokens_placed` if a token besides a home one is down, `corp_cash_max_5s`
    if the 4s are gone, and the MF stats if the tile is down or MF is a corp's or
    closed. The pars, floats, and SRs counted are those after the first
    observation. `ors_played` makes no such allowance: a set before the first
    observation counts 1 OR.
  - **Human baselines**, the same 87 games: `auction_premium` ¥33,
    `auction_sr1_cash` ¥311, `par_price` ¥81, `par_65_share` 27%,
    `par_stranded_share` 12%, `corp_cash_float` ¥813, `corp_cash_max_5s` ¥639,
    `sr2_shares_sold` 0.29, `tokens_placed` 7.4, `phase_start_or_<phase>` for 3
    to D 3.4, 5.7, 8.0, 9.9, and 11.7, `mf_port_laid` 87%, `mf_port_sr` 2.3, and
    `mf_port_in_or` 84%. 41 of v1-review.md Section 2's checkpoint-58 games, in
    the same order: ¥62, ¥299, ¥69, 83%, 52%, ¥684, ¥358, 6.3, 4.0, 4.2, 6.4,
    and 10.3 (no 6 or D), 100%, 1.0, and 0%.
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
samples per epoch, by the same rules (`cloud/rescale_1889.py`; Section 11). A new
`buffer_capacity` keeps the saved buffer's newest rows that fit, in order, with
their priorities, whether it grows or shrinks. A grown buffer's steps scale with
its fullness until it fills (`train/PIPELINE.md`). A new
`training_steps_per_epoch` also moves the LR schedule's span under the restored
step: about 7% through the cosine at epoch 20, a change of a third moves the rate
by about 1%.

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

### 7.3 The train roster and the deviation test

**The stall.** Through epoch 58, self-play buys about half the 5-trains and almost no
6s or Ds, and its corps rarely withhold:

| Epochs 54-58 | 5s | 6s | Ds | Games ending in 6 or D | Phase-5 withhold share |
|---|---|---|---|---|---|
| Self-play | 1.28-1.57 | 0.004-0.034 | 0-0.014 | 0.4-2.2% | 1.4-2.3% |
| Humans (87 games, 3-5 players) | 3 | 1.7-1.9 | 1.1-2.5 | 57-91% end in D | 19% |

6s go on sale only when the last 5 is bought, so the run ends in phase 5, at the
bank, in about 90% of games. Game length does not show it: self-play plays one or two
more ORs than humans, with fewer decisions in each, and lands within 35 decisions of
them.

**Where it happens** (the replay buffer saved with epoch 35's checkpoint, 1,692
games). With room and the cash for a 5, the cheapest train on sale, a corp buys it
73% of the time, and 68% for the last 5. The positions where withholding this OR's
revenue pays for the next depot 5 or 6 and paying out does not, with the bank intact
and able to pay the withhold, come up 0.96 times a game, in 632 of the games, and the
model withholds at 4.0% of them. A payout pays the corp for its shares in the Open
Market, so those count toward what paying out leaves it. In the scraped human games,
the buyer had withheld its last dividend before 31% of the 5-trains. Little time is
left at these positions: a median 134 decisions in phase 4, 80 in phase 5.

**What could cause it.** Train purchases are the main flow back to the bank: a
payout pays nothing for IPO shares, and a withhold moves the whole revenue to the
corp. So each purchase moves the bank break later for everyone, and the buyer pays
for it alone. Either financing pays and the model does not see it (the payback lies
about 3 ORs, 100 decisions, past the decision), or it does not pay against players who
never finance, and every seat is right to pay out.

**The test** (`utils_18xx/g1889_deviation.py`). At each such DIVIDEND position, it plays two
branches to the end, every seat the checkpoint at its self-play settings: `model`, the
model's own choice, and `finance`, a withhold and the depot purchase, then the model.
The president's terminal value (`compute_terminal_values`, the value head's target),
paired by position, measures whether financing pays. A readout of each position records
what the model expects: the president's value at the position and after the forced
purchase, by search at the epoch's simulations and at 4 times them without root noise,
and by the raw value head. The other seats' net worth and the ORs left show whom a
purchase helps.

| Played out | The model's prediction | Reading | Direction |
|---|---|---|---|
| Financing is better | Worse | A value blind spot | More late-game signal: starts from human mid-game positions, more late search, more outcome weight in value targets |
| Financing is no better | Agrees | An equilibrium | Varied opponents, such as past checkpoints, or starts from human positions |

- **Positions:** drawn from a saved buffer, one a game at most, the same number for
  each player count and phase. With `--human`, from the scraped human games instead,
  on the networks humans built: the state before a dividend, loaded from Ruby's
  snapshot, its revenue set to the engine's best.
- **Continuations:** `train.self_play.play_game` from the position, with the run's
  config at the checkpoint's epoch (simulations, c_puct, root noise, forced playouts)
  at `temp_final` from the first move. Two a branch, the branches sharing each seed.
- **Readout:** if 4 times the search moves toward withholding, the simulation ramp
  may close the gap on its own.
- **Analysis:** differences per position, averaged over continuations, with bootstrap
  intervals over positions, overall, by player count and by phase; the predicted
  difference against the played-out one; and the spread, which sizes a full run.
- **Cost:** the local GPU plays 1.18 decisions a second at the training simulations
  (1,172 at epoch 56), fp32, uncompiled.
- **Pilot** (2026-11-08): checkpoint 55 (1,172 simulations) on the epoch-35 buffer's
  positions, 5 per player count and phase, 30 in all, 2 continuations a branch: 110
  minutes on 3 local workers, 1.9 decisions a second together. Results in
  `scratchpad/deviation/`. Intervals are 95% bootstrap intervals over positions.
  - **Financing does not pay on average:** the president's terminal value moves
    −0.025 [−0.094, +0.037]. By players, +0.037, −0.129 [−0.285, −0.003], and +0.019;
    by phase, −0.088 in 4 and +0.039 in 5.
  - **Every seat gains:** the president's net worth +¥195 [−61, +490], each other
    seat's +¥244 [+35, +487], and the game +0.27 ORs [−0.02, +0.58]. The president gains
    more than the others in 12 of 30 positions.
  - **The model expects that, on average:** −0.034 [−0.058, −0.013] by search, −0.044
    at 4 times it, and −0.013 by the value head. Position by position it does not
    see it: the signs agree 37-43%, the correlation is near 0, and its predictions spread
    0.07 against 0.19 measured. 4 times the search moves withhold's visit share from
    9.0% to 9.7%.
  - **One purchase does not open the late game:** 3% of games end in phase 6 or D
    without it and 5% with it, all at 3 players.
  - **Spread:** 0.19 between positions, 0.08 between a branch's continuations; about
    150 positions would resolve ±0.03.
  - **Reading:** so far an equilibrium more than a blind spot. Against this population
    one financed purchase lengthens the game for everyone and buys its president
    nothing, and the model expects as much, though not where.
- **Human positions** (2026-11-08, `--human`): checkpoint 55, the same settings, on
  every scraped game with a position, 44 of the 87: 12 in phase 4, all the first 5;
  18 in phase 5 for a 5, and 14 for the first 6. 218 minutes on 4 local workers, 2.7
  decisions a second together. Results in `scratchpad/deviation/human_*`.
  - **The humans financed:** they withheld at 34 of the 44, and at 27 of the 32 in
    phase 5, against withhold's 3.7% visit share there.
  - **Overall it does not pay for the model:** −0.014 [−0.123, +0.096], and +0.27 ORs,
    as in the pilot. The phases part: −0.250 [−0.434, −0.075] in phase 4 (the
    president's net worth −¥615) and +0.075 [−0.042, +0.199] in phase 5, a 5 and a 6
    alike. Pooled with the pilot: −0.160 [−0.267, −0.056] in phase 4, 27 positions,
    and +0.064 [−0.017, +0.153] in phase 5, 47.
  - **The model is right about phase 4 but too pessimistic in phase 5:** in phase 4
    it predicts −0.276 by deep search, close to what is measured. In phase 5 it
    predicts −0.071, 0.146 [0.019, 0.282] below the measured difference. Pooled with
    the pilot's phase 5, it is 0.126 [0.036, 0.222] too low. The pilot alone already
    had it 0.064 [0.001, 0.142] too low. Position by position it still does not
    follow: the signs agree 50-55%, the correlation is +0.2.
  - **The model's own play reaches the late game more on human boards:** without the
    forced purchase, 26% of games end in phase 6 or D, against 3% in the pilot, and
    with it 55%. Where the 6 is the next train and the president's corp could finance
    it, anyone buys one in only 32% of the model branch's continuations. The humans
    withheld for it at 13 of those 14.
  - **Spread:** 0.38 between positions, 0.18 between continuations; about 600
    positions would resolve ±0.03.
  - **Reading:** two answers, one for each phase.
    - **Phase 4:** an equilibrium that the model reads correctly. Withholding for the
      first 5 costs the president.
    - **Phase 5:** a mild blind spot. Financing is neutral to positive for the
      president, but the model expects it to cost about 0.06-0.07, which is enough
      to keep withhold near 4% of the visits. The phase-5 cuts and the pooling were
      chosen after the pilot. The error interval only just clears zero, and the
      measured gain's interval does not.
- **Later, if the result calls for it:** sustained financing, a corp withholding until
  it can buy, for the positions that need more than one withhold (about 20 a game);
  and starts from the human games around their first 6, which tell whether the model
  carries a game into phase D, and how its value head reads late-game states that
  self-play never reaches.

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

**8.6 at epoch 58** (2026-11-08, `checkpoints/1889-v0/probe_e0058/probe.json`, on the
5090).

**Verdict: keep v0.** No candidate meets its margin.

**Setup:**
- 40K steps at 256 on the 437K rows before the newest 1,000 games, which makes 23
  passes over them.
- Scored on those games' 563K rows. TensorBoard counts 139 more rows in epochs 57-58
  (0.02%).

| Model | Parameters | Steps a second | Value MSE | Policy KL | Against 256 × 15 |
|---|---|---|---|---|---|
| 256 × 15, seeds 0 and 1 | 18.2M | 13.2-13.3 | 0.1785, 0.1793 | 0.1299, 0.1301 | gaps 0.0009 and 0.0002 |
| 192 × 12 | 8.8M | 19.0 | 0.1790 | 0.1344 | value +0.0%, policy +3.4% |
| 384 × 18, 6 heads | 46.6M | 7.1 | 0.1802 | 0.1272 | value +0.7%, policy −2.1% |
| 256 × 15, 8 heads | 18.2M | 11.1 | 0.1804 | 0.1299 | value +0.9%, policy −0.0% |

- **Policy follows size,** in every player count and decision phase:
  - 192 × 12 is 3.1-5.6% worse.
  - 384 × 18 is 1.4-3.1% better.
  - Both are many times the seed gap, but under 3% overall.
  - Every model passes its best policy at 30K steps (256 × 15: 0.1282 at 30K, 0.1299 at
    40K).
- **Value does not follow size.**
  - All models are within 1.1% of each other overall, while a model's score moves
    by up to 0.009 between 10K-step scores (8 heads: 0.1737 at 10K, 0.1828 at 20K).
    That is 10 times the seed gap at 40K.
  - Most score best at 10K steps, about 6 passes.
  - The training value loss reaches 0.0035, against 0.18 held out. The value head
    memorizes its rows: it is bound by data, not capacity.
- **8 heads:** no gain, and 16% slower to train.

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
    the same outputs at about a quarter of the bias's CPU time. On the 5090 it is
    4-7% of the forward (5.3).

## 11. Runbook

The runs on the cloud RTX 5090, in order. The box is reached only by the user, so
each step that needs it writes a file to send back. Commands run from the repo
root, with the box's tmux keeping the run's TTY for `q`. 11.1 is the sequence for
the next launch, v1's. 11.2 is v0's run as it went, kept as its record and for its
notes on the box and the build.

### 11.1 v1's launch

**1. Code.** The user pushes `az-1889` from the dev box.

**2. Setup and build.** On the pod v0 and Tier C ran on, a Runpod container on
`runpod/pytorch:1.5.0-cu1320-torch2130-ubuntu2404`, with the checkout on its volume
and the image's Python (11.2):

```bash
cd /workspace/rss-az
git status --short              # nothing tracked may be modified; scratchpad/ is ignored
git fetch origin az-1889 && git checkout az-1889 && git pull --ff-only origin az-1889
python -c "import torch, Cython, numpy, rich, tensorboard; print(torch.__version__, Cython.__version__)"
python setup.py clean_build 2>&1 | grep -E "(warning|error)" || true
pytest tests/g1889 tests/test_pipeline_boundary.py -q
df -h /dev/shm /workspace       # the workers' shared buffers take 37 MB (v0's 29); the buffer saves 7.4 GB
free -g                         # and holds as much in memory at its cap (v0's: 2.9 GB)
```

- **The import line** printed `2.13.0+cu132 3.3.1` for v0. A container made anew from
  the image has lost the packages `pip` added; put them back first, with
  `install.sh`'s list and the image's torch:
  `pip install setuptools 'numpy>=2.0' 'Cython>=3.0' 'tensorboard>=2.0' 'rich>=13.0' 'matplotlib>=3.0' 'tqdm>=4.0' 'scikit-learn>=1.0' 'pytest>=8.0' && pip install --no-deps 'captum>=0.7'`.
- **The build** is the default one, with the engine's assertions, as v0's was. It can
  look hung for minutes (11.2). `setup.py --release` is not for this box: the tests
  that feed the engine invalid input crash a release build (a segmentation fault at
  `tests/g1889/phases/test_trains.py`, on the dev box), and it buys nothing a run can
  see: a leaf's extraction is 5-9% faster, of the 1-2% of a core a worker spends on it.
- **The tests** take 10 minutes on the dev box's 32 threads (1,372 of them); the replay
  tests skip without the scraped games.
- **A machine with nothing on it** takes the script instead, which makes a venv with
  `install.sh`'s torch 2.14, builds, and runs every test:
  `GH_TOKEN="ghp_..." REPO_DIR=/workspace/rss-az bash cloud/setup.sh`, then
  `cd /workspace/rss-az && source .venv/bin/activate` in every shell. Whichever torch
  it is, step 3 profiles it.
- **Nothing else to copy.** The start positions are in the repo
  (`g1889/data/human_starts.npz`, 1.5 MB), and the run reads no other file. v0's
  checkpoints stay where they are: step 7 plays against `checkpoints/1889-v0`.

**3. Profile.** With nothing else on the GPU, in the tmux session the run will keep:

```bash
tmux new -s 1889-v1
python cloud/profile_1889.py
```

It profiles `train_configs/1889-v1.json`, takes about 20 minutes, and writes
`1889-v1-cloud-profile.json` as it goes. Send that file back; 12.7 records it.

- **Stops the launch until understood:** a `FAIL` on graph breaks, compiled graphs,
  recompiles, each (B, K) recording one graph, compiled against eager, filler across
  K, the weight sync, or memory growth; or `v1/v0 model time` over 1.
- **Results to record, not blockers:** v1 against v0 away from 0.80-0.85 but under 1;
  `v0/v3` against the 1.43 and 1.71 of 2026-11-05, which says whether the box is the
  same; the two `launch sizes between the buckets` lines and `what the sizes between
  cost`, which are for step 5's padding; a leaf's extraction; and steps a second.
- **v0 again,** if a number needs its twin: `--config train_configs/1889-v0.json`,
  which writes `1889-v0-cloud-profile.json`.

**4. Launch.**

```bash
ls checkpoints/1889-v1 runs/1889-v1     # neither exists: an old run's files would be overwritten or merged
TORCH_LOGS=recompiles python -m train --config train_configs/1889-v1.json
```

The server compiles before the first game, for 4 minutes with v0. The banner shows
`nn/transformer-1889-v1.py`, 18,497,528 parameters, `replay ratio 1.1 (draws per row
over its 3 epochs in the window)`, `anneal 30%–60% of the last epoch's game length`,
and the started games; before it, `Start positions: g1889/data/human_starts.npz,
45,355 rows (3p 24,261, 4p 17,764, 5p 3,330)`.

**5. The first epochs.** `tensorboard --logdir runs/1889-v1`.

- **Compile and memory.** One `Recompiling function forward` in the warmup, batch 1 to
  batch 2, and none after. `nvidia-smi`: memory stops growing once the shapes in use
  are recorded.
- **Throughput.** `profile/server_throughput` against v0's 25.2K evaluations a second;
  the profile's v1/v0 ratio says what to expect. `epoch/duration_secs` against v0's
  1.02-1.15 hours.
- **Padding.** `eval_server/padding_share`: the share of launched rows that are
  padding, with `eval_server/launches_at_<B>` and `real_rows_mean_at_<B>` behind it.
  Local runs padded 9-17% with 8 workers (12.7) and 28% with 6 (v1-design Section 4).
  If the cloud's is near the 28%, finer launch sizes are the fix, and the profile's
  `launch_sizes` table prices them; `eval_server/forward_share`
  and `idle_share` say whether the server is the limit at all.
- **Epoch 1 differs by design.** It plays at `temp_initial` throughout, with no
  `schedule/temp_anneal_*` tag: there is no game length yet. From epoch 2 the tags are
  30% and 60% of the last epoch's `self_play_<n>p/game_length_mean`. Training runs
  about 430 steps in epoch 1, 860 in epoch 2, and 1,290 from epoch 3
  (`epoch/training_steps`: 1.1 x the epoch's rows x `buffer/window_epochs` / 3 / 256),
  and the rate warms up over epoch 1's steps.
- **Starts.** `self_play_started/games` is 126 of the 500, 42 at each player count,
  and `self_play_aggregate` and `self_play_<n>p` cover the other 374.
- **The value head.** `heldout/value_mse` beside `epoch/value_loss_avg`, their ratio
  `heldout/value_mse_over_training`, and `heldout/value_mse_q1` to `_q4`. v0's training
  loss read 0.020 where fresh games scored 0.206. **The rule** (v1-design Section 11):
  if the held-out value loss is still several times the training value loss after
  epoch 20, set `"buffer_epochs": 6` in `train_configs/1889-v1.json` at the next
  resume (step 6). The 2.5M-row cap already holds those epochs' rows.
- **What should move within 20 epochs,** if the design works (v1-design Section 9;
  v0's stall was set by epoch 10), under `self_play_aggregate/`, with the humans' figure
  and checkpoint 58's (6.6; v1-review Section 2):

  | Tag | Humans | v0 at epoch 58 |
  |---|---|---|
  | `par_65_share` | 27% | 83% |
  | `corp_cash_max_5s` | ¥639 | ¥358 |
  | `withhold_share_<phase>`, 2 to D | 1%, 10%, 17%, 19%, 20%, 4% | 3% over phases 3-5 |
  | `trains_bought_5`, `_6`, `_D` | 3, 1.84, 2.31 | 1.41, 0, 0 |
  | `par_price`, `corp_cash_float` | ¥81, ¥813 | ¥69, ¥684 |
  | `par_stranded_share`, `sr2_shares_sold`, `tokens_placed` | 12%, 0.29, 7.4 | 52%, 6.3, 4.0 |
  | `phase_start_or_<phase>`, 3 to D | 3.4, 5.7, 8.0, 9.9, 11.7 | 4.2, 6.4, 10.3, no 6 or D |

- **The rest,** as v0's: `self_play_<n>p/game_length_mean`, `end_bank`,
  `track_pass_share`, `corps_floated`, `epoch/grad_norm_avg` and
  `epoch/grad_clipped_share`, and `loss/*`, which a NaN stops with an error.

**6. Stopping and resuming.** Press `q`, then Enter, in the run's terminal, and wait
for `Shutdown complete`, which names the checkpoint. Then:

```bash
TORCH_LOGS=recompiles python -m train --config train_configs/1889-v1.json --resume latest
```

- **Stop just after an epoch's summary.** A stop mid-epoch waits 120 s for the games
  in play, drops the rest, and saves the finished ones in the buffer. The resume then
  plays that epoch's 500 games again beside them, as one epoch, with steps for all its
  rows. Nothing breaks (12.7), but 128 workers' games, about 15 minutes each, would
  mostly be dropped.
- **`Training complete.` after `Shutdown complete`** is printed on every exit.
  Ctrl-C saves nothing. A checkpoint is otherwise written every 5 epochs, so a crash
  costs up to 4.
- **Always with `--config`.** A resume takes the checkpoint's config, then the file
  over it, then flags, and prints `Config override:` for each field the file changes:
  none, unless the file was edited, as for `buffer_epochs`.
- **It continues** the rate, the temperature windows, and the start positions where
  they were (`train/PIPELINE.md`, Operations).
- **v0's steps that are gone:** the epoch-20 rescale, which `cloud/rescale_1889.py`
  refuses for a config with `replay_ratio`: the steps, the buffer, and the windows
  follow the run. The capacity probe (11.2, step 7) is not part of this launch, and
  has not been run on a v1 checkpoint.

**7. v1 against v0,** once a checkpoint is worth it (v1-design Section 8):

```bash
python -m train.tournament \
  checkpoints/1889-v1/checkpoint_epoch_0100.pt,checkpoints/1889-v0/checkpoint_epoch_0058.pt \
  --num-players 4 --simulations 1600 --search-batch-size 8 --sample-moves 20 --games-per-pair 60 \
  --workers 8 --output v1-vs-v0-4p.report
```

- **The score** is the first checkpoint's against the second's: over 0.5, v1 is
  ahead. v0 against itself scores 0.50 ± 0.04, and v0 against an untrained v1 1.00
  (12.7).
- **It runs in fp32, uncompiled,** each worker loading both models, so it shares the
  GPU with a run only by slowing it: play between a stop and a resume, or copy the
  checkpoint to the dev box. `--num-players` 3 and 5 are separate commands.

**Checked before the box** (2026-11-10; 12.7): the profile, whole, on v1 and on v0,
on the CPU and on ROCm, and on the CPU compiled through the real kwargs and warmup at
all 45 (B, K) and the 10 between; `train_configs/1889-v1-smoke.json` through
`python -m train` with a stop mid-epoch and two resumes; v0 against v1 in one game.
Inductor's CUDA graphs, bf16 under compilation, the 128-worker server, and every time
above are the 5090's to report.

### 11.2 v0's run

**1. Code.** The user pushed `az-1889` from the dev box.

**2. Setup and build.** A fresh box:

```bash
GH_TOKEN="ghp_..." bash cloud/setup.sh   # from a copy of the script; clones az-1889 to ~/rss-az
cd ~/rss-az && source .venv/bin/activate
```

The script installs torch 2.14 on CUDA 13.2 (`install.sh`), builds every extension,
`g1889` with them (`setup.py`'s `EXTENSION_DIRS`, from the committed `.pyx` files and
tables), and runs the tests, which skip what needs the submodule or a replay fixture.
Neither Ruby nor `submodules/18xx` is read at run time: Ruby only regenerates
`g1889/data/tables.*`, and the tests and 18xx.games tooling are all that read the
submodule. A box already set up:

```bash
cd ~/rss-az && source .venv/bin/activate
git fetch origin az-1889 && git checkout az-1889 && git pull --ff-only origin az-1889
python setup.py clean_build 2>&1 | grep -E "(warning|error)" || true
```

The build can look hung. `cythonize`'s dependency scan prints nothing until it ends,
and it tries each cimported name as a module in each search directory: 115K filesystem
calls on this branch, against 33K before `g1889`, or 0.6 s on the dev box. On
2026-11-05's box, in `/workspace/rss-az`, a first build was interrupted during that
silence, and a second ran to completion in place, in at least 5 minutes by the user's
estimate; the time per call there was not measured. Wait for it. If that is ever too
slow, build on another disk and copy the extensions back, from the checkout (checked on
the dev box; the 5090 did not need it):

```bash
REPO=$PWD; rm -rf /tmp/rss-build && mkdir /tmp/rss-build && git archive HEAD | tar -x -C /tmp/rss-build
cd /tmp/rss-build && python setup.py build_ext --inplace 2>&1 | grep -E "(warning|error)" || true
find core entities phases mcts g1889 -name '*.so' -exec cp --parents -t "$REPO" {} +
```

That box is a Runpod container on `runpod/pytorch:1.5.0-cu1320-torch2130-ubuntu2404`,
in `/workspace/rss-az`, on the image's system Python: torch 2.13.0+cu132 from the image,
not `install.sh`'s 2.14, and Cython 3.3.1. This branch built and passed its tests here
under Cython 3.3.1. The profile measured that torch, so a replacement box takes the same
image, or profiles again.

**3. Profile (8.5).** With nothing else on the GPU:

```bash
python cloud/profile_1889.py
```

It took under half an hour and wrote `1889-cloud-profile.json` as it went (the
script now takes the run by `--config`, and names its file for it: 11.1). Send
that file back; Sections 5.3 and 6.5 record it. A `FAIL` on graph breaks, recompiles,
compiled against eager, or the weight sync stops the launch until it is understood. A
ratio outside 1.7-2.0 or a different record pattern is a result to record, and
changes the cost estimates, not the launch. **Ran** 2026-11-05, in 25 minutes, with no
real `FAIL`: the one on recompiles was a counting bug, since fixed (6.5), and the one on
the ratio, 1.43 at B = 64, is below the range and is recorded (5.3).

**4. Launch.**

```bash
tmux new -s 1889
source .venv/bin/activate
TORCH_LOGS=recompiles python -m train --config train_configs/1889-v0.json
```

**5. Early.** `tensorboard --logdir runs/1889-v0`. In the first epochs:

- The banner shows `nn/transformer-1889.py` and no v2- or v3-only line. The log has
  one `Recompiling function forward` in the warmup, batch 1 to batch 2, and none after.
- `nvidia-smi`: GPU memory stops growing once the shapes in use are recorded.
- `profile/server_throughput` against v3's 38.7K evaluations a second, with 23-27K
  expected (5.4), and `profile/server_infer_ms` and `profile/server_batch_avg` against
  9.2 ms and 180.
- `self_play_<n>p/game_length_mean` against 440-800 (Section 5.4), and
  `self_play_aggregate/total_examples` against 300K. At 240-290K samples an hour, an
  epoch takes 1.0-1.25 hours.
- `self_play_<n>p/end_bank`, `track_pass_share`, `ors_played`, and `corps_floated`
  (Section 6.6), and `loss/*`, which a NaN stops with an error.

**6. Epoch 20: the rescale (7.1).** Once `epoch/duration_secs` has reached step 20,
press `q`, then Enter, in the run's terminal, and wait for "Shutdown complete". Then:

```bash
python cloud/rescale_1889.py    # reads runs/1889-v0, prints the fields, old and new, and writes nothing
```

Edit the printed fields into `train_configs/1889-v0.json`, then:

```bash
python -m train --config train_configs/1889-v0.json --resume latest
```

A resume takes the checkpoint's config, then the file over it, then CLI flags over
both, so every later resume needs the edited file: flags alone would be undone by the
next `--config`. Check for `Config override:` on each field and `Replay buffer
capacity changed` with the rows it kept. Send the printed fields back, for 7.1 and
the dev box's copy of the config; `git checkout` the box's edit before the pull that
brings that copy.

**7. Epochs 50 and 150: the probe (8.6).** Once the run has passed the epoch, press
`q`, then Enter, and note the checkpoint "Shutdown complete" names. A stop saves its
last finished epoch; that is the epoch to pass, at 50 or just past it:

```bash
python -m train.capacity_probe --checkpoint checkpoints/1889-v0/checkpoint_epoch_0050.pt
python -m train --config train_configs/1889-v0.json --resume latest
```

It holds the GPU for hours and writes `checkpoints/1889-v0/probe_e0050/`. Rerun the
same command after an interruption: a finished model is skipped, and an unfinished
one continues from its last save. Send back `probe.json`, whose verdict is 5.7's;
the printed table is recorded in 8.6. To act on it, when the verdict is `grow`,
`shrink`, or `8 heads`, the role is `large`, `small`, or `heads8`:

```bash
mkdir -p checkpoints/1889-v0-large
cp -r checkpoints/1889-v0/replay_buffer checkpoints/1889-v0-large/
python -m train --resume checkpoints/1889-v0/probe_e0050/large/checkpoint_epoch_0050.pt \
    --checkpoint-dir checkpoints/1889-v0-large --tensorboard-dir runs/1889-v0-large
python -m train --resume latest --checkpoint-dir checkpoints/1889-v0-large   # later resumes
```

Without `--config`: `1889-v0.json`'s sizes would override the checkpoint's. It carries
on at the run's epoch, step, LR, and RNG, with a fresh optimizer.

**Checked before the box** (2026-11-04): a clone of the branch, its submodule empty,
builds and passes its tests here beside Ruby. The profile runs whole on the CPU, compiled
through the real kwargs and warmup with no graph breaks or recompiles and compiled
outputs within 1e-7 of eager, and uncompiled on ROCm; the rescale, on
`runs/1889-v0-smoke`; a resume with its fields, on the local GPU; the probe, on the smoke
buffer at small sizes, interrupted and rerun. Inductor's CUDA graphs, bf16 under
compilation, and every time above are the 5090's to report.

## 12. v1

`nn/transformer-1889-v1.py` is v0 on token layout 2
([`token-data-1889.md`](../token-data-1889.md), "Layout 2"), built for
[`v1-design.md`](v1-design.md) Section 7. Sections 3-6 describe it wherever this
section does not say otherwise: the projections' form, the embeddings, the trunk's
blocks, the selections, the value head, the pass and `private` heads, the
initialization, and the module contract. `INPUT_LAYOUT_VERSION` is 2, and
`build_model` asserts layout 2, `token_dim` 109, and the rest as 6.2.

The file stands alone, as v0's does beside v3's (3.5). v0's stays, unchanged, as
the tournament's opponent, and can be deleted once v1 has beaten it.
`tests/g1889/test_model_v1.py` holds the two together: v0's weights, copied to the
slots both layouts hold with zeros on what v1 adds, give v0's outputs on every
state without a visible candidate.

### 12.1 What differs from v0

| | v0 | v1 | Why |
|---|---|---|---|
| Input | layout 1, 95 rows of 108 | layout 2, 95 rows of 109 | v1-design Section 5 |
| Projections | GLOBAL 43, PRICE 23, DIVIDEND 7, CORP 50, TRAIN 12, CANDIDATE 31 beside the tile block, PLAYER 37 slots | 89, 29, 12, 76, 14, 32, 39; PRIVATE, HEX, and the tile block as v0 | layout 2's widths |
| Trunk rows | all 95 | 80: every row but the 15 CANDIDATE rows | 12.3 |
| TILE and ER_BONUS | `candidate`, a pointer over candidate rows the trunk has processed | `candidate`, a head over rows the trunk ran without them | 12.3 |
| Dynamic relations | ids 0-24, 25 channels | ids 0-20 and 25-28, 25 channels | 12.4 |
| Raw head slots | 4.2 | 12.2: layout 2's numbers, and the new per-choice slots | v1-design Section 7 |
| Parameters at 256 × 15 | 18,218,872 | 18,497,528 | the candidate head, +247K; wider projections and head inputs, +31K |

`hex_mix` and `bias_init` (v1-design Section 10) are not in it.

### 12.2 Readouts by phase

4.2's table in layout 2's slot numbers. **Bold** marks a slot v1 adds to a head's
raw inputs; the rest are v0's slots where layout 2 put them.

| Phase | Head | Reads, beyond the context | Raw slots |
|---|---|---|---|
| AUCTION | `auction_private` | PRIVATE p | PRIVATE 7-13 |
| BID | `bid` | PRICE; the private | PRICE 5-23 |
| STOCK | `stock_corp` | CORP c | CORP 7-25 and 50-54 (v0's 44-48); **58-60**, the price after a payout, a withhold, and a sold-out SR; **68-71**, the float: shares still to sell, their cost, the president can pay, the actor can pay; **72**, the president sold it this SR; the actor's PLAYER 11+c and 18+c (v0's 12+c and 19+c) |
| PAR | `par` | PRICE | PRICE 5-23; **24-29**, the shares of the float out at each par |
| SELL_AMOUNT, EMERGENCY_SELL_AMOUNT | `sale` | the sold corp | its CORP 15-19, 54, 55 (v0's 48, 49) |
| BUY_PRIVATE_PRE, BUY_PRIVATE_POST, STOCK's DR and MF | `private` | PRIVATE p | PRIVATE 16, 17 |
| TRACK | `track_hex` | HEX `TRACK_HEXES[i]` | HEX 92-102 |
| TILE, ER_BONUS | `candidate` (12.3) | the hex; the hexes across the candidate's exits; CANDIDATE k as projected | CANDIDATE 1, 99-107; **108**, the route max gained with the next train; the path mask 43-63, for the exits |
| TOKEN | `token_hex` | HEX `TOKEN_HEXES[i]` | HEX 103, 104 |
| DIVIDEND | `dividend` | DIVIDEND | DIVIDEND 1-6 (v0's 1 and 3-7); **7-12**, the next train's price, the shortfall to it after a withhold and after a payout, room, and the route max with the next train and after the next rust |
| DISCARD | `discard` | TRAIN `DISCARD_TYPES[i]` | TRAIN 12 |
| BUY_TRAIN | `buy_train` | the six TRAIN rows | TRAIN 2-9 of each; **10**, the route max a purchase adds; **14**, its next depot purchase starts a phase |
| | `buy_train_corp` | CORP c | CORP 15-19, 52, 55 (v0's 46, 49) |
| SELECT_TRAIN | `select_train` | TRAIN t; the seller | TRAIN 10, 11; **13**, what the seller loses; **14**, its next depot purchase starts a phase |
| TRAIN_PRICE | `train_price` | PRICE; the train type; the seller | PRICE 5-23; **the selected TRAIN row's 10 and 13**, what the buyer gains and what the seller loses |
| PRIVATE_PRICE | `private_price` | PRICE; the private | PRICE 5-23 |
| PORT | `port` | HEX `PORT_LAY_HEX[i]` | the port tile's path mask at `PORT_LAY_ROT[i]` |

- **Slot numbers are literals in the model file,** under named constants, as v0's
  are: extraction exports the row widths and the tile block, not the slots.
  `tests/g1889/test_model_v1.py` checks every slot of every row against this table
  with the trunk's output held fixed, and the copied-weights test above checks
  the slots v0 already read.
- **Beyond v1-design Section 7's list,** three raw reads of slots layout 2 fills
  for these choices: TRAIN 10 at TRAIN_PRICE, since a price weighs what the train
  is worth to each side; TRAIN 14 at SELECT_TRAIN; and CANDIDATE 108, the delta
  layout 2 added for the case where v0's (104) is zero.
- **Left to the trunk:** the operating rank (CORP 38-44) at STOCK, and TRAIN 10 at
  DIVIDEND, where the head reads no TRAIN row. Each reaches its head through the
  rows' final embeddings.

### 12.3 The candidate head

Tier C's `cand_head` (v1-design Sections 7 and 10), adopted for its speed: with 80
rows a forward takes 0.76 of v0's time on the CPU and 0.85-0.97 on the local GPU,
by batch (12.5). On v0's replay buffer it scored TILE 2.5% and ER_BONUS 1% worse
in policy KL than v0's arrangement, and everything else the same.

- **The trunk** runs on the 80 rows that are not candidates. The input keeps all
  95 rows, since the head needs the candidates' slots, so the eval server, replay,
  and the trainer see the shapes they would with v0's arrangement.
- **States without a visible candidate,** all but TILE and ER_BONUS, lose nothing:
  v0 hides those rows as keys, and no legal slot reads them.
- **A candidate's vector** c_k is its row as projected (`tile_proj` +
  `candidate_proj` + the selected hex's identity + the type embedding), through
  `candidate_norm`, an RMSNorm that gives it the final rows' scale.
- **Exits.** A side is an exit of candidate k if a path in its tile block's mask
  (CANDIDATE 43-63) touches it. Each side's hex across from the selected hex goes
  through that side's `exit_side_proj`, Linear(d, d/4); the exits' are summed, and
  `exit_proj`, Linear(d/4, d), brings the sum to the trunk's width. A side at the
  map's edge contributes its projection's bias.
- **The head** is an MLP with two hidden layers of width d on `[g, a, c]`, the
  selected hex's final row, the pooled exits, c_k, and the raw slots of 12.2. It
  scores every candidate row alike, so permuting the candidates permutes their
  logits, and a candidate's logit depends on no other candidate.
- **Optimizers.** The head's last Linear is on AdamW with the other output layers;
  its hidden layers and the exit projections are Muon's.
- **Against the Tier C variant:** the head reads CANDIDATE 108 raw as well, and
  v0's unused candidate head is not kept.

### 12.4 Relations and rows

- **Channels.** `relation_bias_mult` is (L, H, 35), zero at initialization, as
  v0's: 25 dynamic channels, for ids 0-20 and 25-28 in order, then the 10 static
  ones. Relations 21-24 join candidate rows and have none. The shape is v0's by
  coincidence; channels 21-24 mean the SR's sales and the route hexes here.
- **Rows.** Records and planes name the input's 95 rows. The model maps each to
  its trunk row: a player's row is 15 earlier there, and a candidate's has none.
  The static planes are cut to the trunk's rows at construction.
- **What gets no weight:** a record of value 0, of an id without a channel, or
  naming a candidate row. On the flat path such a record keeps its place with
  weight 0, as v0's padding does, since removing it would make N depend on the
  data inside the compiled forward.
- **Where a record without weight is scattered:** to its own rows' cell, with the
  candidates' rows folded onto the first 15 trunk rows, so a launch's filler stays
  a few rows to a cell at most, as 6.5 builds it for v0. The Tier C variant sent
  all of them to one cell, which a launch's thousands of filler rows turn into one
  contended address. **Measured** on the local GPU at K = 1,024: that forward takes
  1.29 of v0's time at batch 16 and 1.05 at batch 32, against 0.97 and 0.90 with
  the cells spread.
- **The eval server's K** still counts the candidates' records: the server flattens
  every record above value 0 and buckets by their number (6.5). Extraction emits
  them whatever model reads the records, since v0 needs them in a game the two
  share. A TILE or ER_BONUS state carries at most 210 of them (token-data-1889.md,
  "The edge bound"); on the human start positions they are 0.5% of all records,
  with a median of 24 at TILE.

### 12.5 Cost

**Measured** (2026-11-10, `scratchpad/v1_97/forward_time.py`): forward time at
256 × 15 × 4 and capacity 5, fp32 and uncompiled, on batches of human positions
with about 500 records a state. Each figure is the median of 60 paired rounds
against v0 on a GPU with no other job (8 rounds on the CPU). The last column is a
scratch model, v1's inputs and channels on v0's 95-row trunk and head.

| Device, batch | v0 | v1 | v1 / v0 | v1's inputs on 95 rows / v0 |
|---|---|---|---|---|
| Local GPU (RX 9070 XT, ROCm), 16 | 18.0 ms | 17.4 ms | 0.97 | 1.03 |
| Local GPU, 64 | 55.4 ms | 48.5 ms | 0.88 | 1.01 |
| Local GPU, 128 | 107.4 ms | 91.5 ms | 0.85 | 1.01 |
| CPU, 8 threads, 64 | 506 ms | 385 ms | 0.76 | 0.99 |

- **The wider inputs and the four new channels cost nothing** that these runs can
  see: within 1-3% of v0 on 95 rows.
- **Dropping the candidate rows** takes the cost of a state from 0.81 ms to 0.67 ms
  on this GPU (the slope from batch 64 to 128), a ratio of 0.83, where 80 rows
  against 95 predict 0.71-0.84 for a trunk between quadratic and linear in rows.
- **A launch's fixed cost** is about 17 ms here, uncompiled, and v1's is no lower:
  it runs about 30 more small operations than v0 (the row and channel lookups, and
  the candidate head). So the ratio is 0.97 at batch 16. Compiled on the 5090 is
  the 5090's to report.
- **Padded as an eval launch** (K = 1,024 at batches 16-64, 512 at 128), the ratios
  are the same: 0.97, 0.90 at batch 32, 0.88, and 0.85.

**Smoke runs** (`scratchpad/v1_97/`): 2 epochs of 12 games at 16 simulations, 6
workers and one eval server on the local GPU, fp32, `--no-compile`, bucketed
launches of up to 32 (a mean of 15-17 states), with the replay ratio, the
temperature shares, started games, and the held-out loss on; no other job on the
GPU. Evaluations a second and the mean time of a launch, by epoch:

| Size | v0 | v1 |
|---|---|---|
| 256 × 15 | 641, 671 a second; 23.2, 24.4 ms | 712, 700 a second; 23.0, 22.1 ms |
| 64 × 3 | 2,578, 2,913 a second; 5.0, 5.0 ms | 2,368, 2,438 a second; 5.9, 5.8 ms |

At the tiny size a launch is all fixed cost, and v1's extra operations show: its
launches take 0.8 ms longer.

### 12.6 Tests

- `tests/g1889/test_model_v1.py`: 8.1's properties on layout-2 states (v0's own
  tests of them, run on v1), on random playouts and on human positions; the
  copied-weights test; the relation channels and the row map; every raw slot;
  the candidate head's dependencies, against extraction's CANDIDATE_EXIT records;
  what each layout's model refuses; the parameters against v0's.
- `tests/g1889/test_pipeline_v1.py`: `test_relation_transport.py`'s trainer,
  evaluator, eval-server, and warmup tests with v1 as their model; and a run
  through `train.main` with started games, `replay_ratio`, the held-out loss, a
  resume, and `load_model_from_checkpoint`.
- `tests/g1889/test_tournament.py`: v0 and v1 in one game at 3, 4, and 5 players,
  each evaluator filling its own layout (108 and 109 slots) under its own
  `terminal_rank_tau`, with v1's blocks on 80 rows and v0's on 95.
- `tests/g1889/test_cloud_tools.py`: the profile on v1's config and on v0's, and
  `graph_timing`'s `fullgraph` compile of each model.

### 12.7 Before the launch

Sub-step 9.9 of v1-design Section 12, on the dev box (2026-11-10; scripts and
outputs in `scratchpad/v1_99/`). The 5090's profile follows the dev box's.

**The profile.** `cloud/profile_1889.py` takes the run by `--config`, v1's by
default, and names its output for it (`1889-v1-cloud-profile.json`).

- **`model_time`** times v3, v0, and v1 on the same playout states. v1 over v0 is
  a check, expected at 0.80-0.85 and failing above 1. v0 over v3 is reported beside
  the 1.43 and 1.71 of 5.3.
- **Dropped:** v0 on padded records, as one tensor and as three, and v0 at 8 heads,
  which were 8 of v0's 12 compiles and 10 of its 25 minutes. 5.3 holds their
  results. The three-tensor variant sized its bias by the input's 95 rows, which is
  not v1's trunk.
- **`launches`, `replay`, `memory`** cover K = 4,096: 45 pairs. The eval server's
  warmup is the one they call, with the run's largest launch, 256. It runs each B
  at K = 256 only; any other K costs its shape one launch without a graph and one
  recording in the serve loop, as 6.5 measured, and that is what `launches` counts.
- **`launch_sizes`** (new) launches the serving model, all rows real, at every
  bucket and at 12, 24, 48, 96, and 192, at K = 256 and 512: the forward time a
  launch padded to the next bucket would save at the size between, and the
  compiles, recorded graphs, and memory those 10 shapes add. One compiled graph
  serves them: Dynamo's batch is dynamic from 2 up. The server launches at powers
  of two; this only measures.
- **`extraction`** (new) times a search leaf's tokens and records in both layouts,
  on children of 400 human positions a game phase, as v1-design Section 5.9
  measured it but through the policy's actions and the start-position file.
- **`train_steps`** trains at the run's `value_loss_weight` and replay priorities
  and gives the minutes an epoch's steps take.

**Measured** on the dev box (RX 9070 XT under ROCm, fp32, uncompiled, at 256 × 15;
a 9950X3D), a check of the script more than of the model:

| | v1's run | v0's run |
|---|---|---|
| Model time, batch 16: v3, v0, v1 | 15.5, 18.4, 17.5 ms (v1 / v0 0.95) | 14.8, 18.3 ms |
| Model time, batch 64 | 36.0, 55.7, 48.5 ms (0.87) | 35.8, 55.3 ms |
| Forward at B = 8, 16, 32, 64, K = 256 | 12.6, 17.4, 27.7, 48.5 ms | 12.4, 18.0, 30.7, 55.5 ms |
| Forward at B = 12, 24, 48 | 16.9, 22.3, 38.2 ms | 15.5, 24.4, 43.2 ms |
| Saved on the next bucket | 3%, 19%, 21% | 14%, 20%, 22% |
| Training steps a second, batch 64 | 5.6 | 5.0 |

| A leaf's extraction, µs, by the parent's game phase | 2 | 3 | 4 | 5 | 6 | D | All |
|---|---|---|---|---|---|---|---|
| Layout 1, mean | 6.0 | 7.8 | 8.5 | 10.8 | 17.6 | 29.5 | 14.2 |
| Layout 2, mean | 6.8 | 9.3 | 10.8 | 12.9 | 25.7 | 59.2 | 22.5 |
| Layout 2 over layout 1 | 1.12 | 1.20 | 1.27 | 1.20 | 1.46 | 2.01 | 1.58 |
| Layout 2, 99th percentile | 16 | 39 | 46 | 54 | 135 | 550 | 283 |

- **K changes little:** 1,024 for 256 adds 0-3% at every size.
- **On the CPU, compiled** through the real kwargs and warmup, at 128 × 4: at all 45
  (B, K) pairs and the 10 between, no graph break, two compiled graphs (batch 1 and
  the rest), no recompile, and outputs within 2e-7 of eager at every shape and after
  a weight sync, and the same across K. The sizes between needed no compile.
- **A release build** (`setup.py --release`, in a copy) extracts 5-9% faster, and
  its test run ends in a segmentation fault at the first test that feeds the engine
  invalid input. The cloud build stays the default one (11.1).

**Measured on the 5090** (`1889-v1-cloud-profile.json`, 2026-11-10, at f65b573;
torch 2.13.0+cu132; compiled, bf16, random weights on 2,048 random-play states).
Every check passed. v0's figures are `1889-cloud-profile.json`'s of 2026-11-05.

| ms | B = 64 | B = 256 |
|---|---|---|
| Model only: v3 | 1.62 | 4.16 |
| Model only: v0 (ratio to v3) | 2.31 (1.43) | 7.10 (1.71) |
| Model only: v1 (ratio to v0) | 2.08 (0.90) | 6.03 (0.85) |
| A served launch at K = 512: v1, v0 (ratio) | 3.12, 3.22 (0.97) | 9.24, 10.23 (0.90) |

- **The box is the same:** v0 over v3 is 1.43 and 1.71 again.
- **v1 over v0** is 0.85 at a full batch, as expected, and 0.90 at 64. A served
  launch gains less: 0.90 at 256, 0.91-0.97 from 4 to 128, and 1.01-1.04 at 1
  and 2, where v1's extra small operations show.
- **A served launch costs about 3.2 ms over the model alone at 256,** for v1 as
  for v0 (3.1). 2.6 ms of it is inside the forward and its copy back (8.66 ms
  against 6.03), which the profile does not split further; the run's
  `eval_server/*_share` tags time the stages around it.
- **Compile:** no graph breaks, two compiled graphs, 245 s of warmup (v0: 244).
  45 graphs for the 45 pairs, one each, none compiled after the warmup, and no
  launch over 19 ms. K = 4,096 costs 24% over K = 256 at B = 256 (11.3 ms
  against 9.1), 21% at 128, and 14% or less below.
- **Against eager:** logits within 8.8e-3, values 2.0e-3, policy total variation
  2.4e-3 on first launches, and the same on replays and after a weight sync
  (1.2e-2, 3.0e-3, 3.5e-3). Filler changes nothing across K, exactly.
- **Memory:** 946 MB reserved with all 45 shapes recorded, 1,850 MB for the
  process by `nvidia-smi`, and no growth over 9,000 launches. Training peaks at
  7.0 GB.
- **Training:** 14.9 steps a second at batch 256 (v0's profile: 13.2), 1.4 minutes
  for an epoch of 1,289 steps.

| Forward ms at the size between, K = 256 | 12 for 16 | 24 for 32 | 48 for 64 | 96 for 128 | 192 for 256 |
|---|---|---|---|---|---|
| Between, bucket | 1.38, 1.45 | 1.82, 2.00 | 2.42, 2.86 | 3.79, 4.67 | 6.60, 8.54 |
| Saved | 5% | 9% | 15% | 19% | 23% |

- **The sizes between cost nothing to add:** 10 shapes recorded 10 graphs with no
  compile and no more memory. K = 512 gives the same savings. Whether to build
  them is the run's `eval_server/padding_share` and `launches_at_<B>` to say.

| A leaf's extraction, µs, by the parent's game phase (a 9950X) | 2 | 3 | 4 | 5 | 6 | D | All |
|---|---|---|---|---|---|---|---|
| Layout 1, mean | 5.2 | 6.9 | 7.6 | 9.8 | 15.5 | 25.8 | 12.5 |
| Layout 2, mean | 5.9 | 8.4 | 9.6 | 11.8 | 22.3 | 51.7 | 19.8 |
| Layout 2 over layout 1 | 1.12 | 1.21 | 1.27 | 1.21 | 1.44 | 2.00 | 1.58 |
| Layout 2, 99th percentile | 14 | 38 | 42 | 51 | 121 | 463 | 243 |

**The smoke run.** `python -m train --config train_configs/1889-v1-smoke.json
--no-compile` on a pty: 8 workers and one eval server on the local GPU in fp32, 12
games an epoch at 64 simulations, batches of 64. `q` in epoch 2 after 5 games, a
resume that finished the epoch, and a second resume with `--num-epochs 3`. Read from
the event files, the buffer saved by each run, and the checkpoints:

| | Epoch 1 | Epoch 2, to the stop | Epoch 2, resumed | Epoch 3 |
|---|---|---|---|---|
| Games: opening + started | 9 + 3 | 9 + 2, and 1 dropped | 9 + 3 | 9 + 3 |
| Rows | 7,064 | 8,642 | 9,092 | 11,173 |
| Window: rows, epochs | 7,064, 1 | | 24,798, 2 | 35,971, 3 |
| Steps, run and by `replay_ratio` | 40, 40 | untrained | 203, 203 | 192, 192 |
| Rate at the first and last step | 2.5e-5, 1e-3 | | 1e-3, 0.9999754e-3 | 0.9999752e-3, 0.9999012e-3 |
| Windows at 3, 4, 5 players | none | not logged | 152-305, 251-502, 227-455 | 237-473, 244-488, 294-588 |
| The lengths they are 30% and 60% of | | | epoch 1's 507.7, 836.7, 757.7 | epoch 2's 788.7, 813.7, 980.0 |
| `heldout/` tags; rows | 53; 2,048 | | 55; 2,048 | 51; 2,048 |
| Held-out value MSE over training's | 0.415 / 0.328 = 1.26 | | 0.465 / 0.147 = 3.15 | 0.506 / 0.054 = 9.46 |
| `eval_server/` shares' sum; padding | 1.000; 16.9% | | 1.000; 12.7% | 1.000; 8.9% |
| Evaluations a second; ms a launch | 877; 26.3 | | 841; 24.4 | 913; 24.2 |

- **Steps** are `round(1.1 × rows × window epochs / 3 / 64)`, with epoch 2's rows the
  17,734 of both its parts.
- **The rate** is the epoch schedule's at each of the 43 logged steps, within 6e-11,
  through both resumes.
- **The replayed epoch stayed one epoch:** the buffer's record is
  `[[0, 0], [1, 12]]` after the stop and after the resume, with games 12-34 in epoch
  2, and `[[0, 0], [1, 12], [2, 35]]` after epoch 3. The stop left
  `checkpoint_epoch_0001.pt` at epoch 1 with epoch 1's lengths, so the replay played
  in the windows the first attempt had; its started games began from the same
  positions.
- **`heldout/`** has every split each epoch, and fewer tags where a decision phase
  has no row in the sample. The ratio climbs as 12 games an epoch are memorised,
  which says nothing of the run's.
- **Padding** is under the 28% of v1-design Section 4's 6-worker run: here launches
  at 32 carried 27.7-28.9 real rows.
- **Two things `train/PIPELINE.md` did not say,** now added to it: the drain waits
  120 s and then drops the games still in play (1 of 7 here); and a replayed epoch's
  lengths, for the next epoch's windows, and its `self_play_*` scalars cover the
  games after the resume alone (epoch 3's windows above come from 9 of epoch 2's 18
  opening games). `Training complete.` is also printed after `Shutdown complete`.

**v0 against v1** (`train/tournament.py`, Section 9's hook): checkpoint 58 against a
256 × 15 v1 at its initial weights, saved as `train/main.py` saves a run stopped
before it trains; 4 players, 32 simulations in leaf batches of 8, the first 20
decisions sampled, 4 workers on the local GPU. **Measured:** v0 scores 1.000 ± 0.000
over 24 games, first and second in every one, with v0 on layout 1 and v1 on layout
2 in the same games. With v1 listed first, the score reads 0.000 over 6 games: it is
the first checkpoint's.
