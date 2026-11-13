# Shikoku 1889 v2 Design: inputs and the policy readout

Status: draft for the user's review, dated 2026-11-13. Nothing here is built.
Decision 1 is settled: the trunk has 45 rows (Section 11).
It follows the review of v1 at epoch 10 (Section 2) and replaces token layout
2, the relation records, and v1's policy heads. It owes v0 and v1 no
compatibility.

Paths are relative to the repo. **Measured** marks a number from this review's
scripts (`scratchpad/v1_overall_review/`, `scratchpad/v2_experiments/`);
**Estimated** marks arithmetic from such numbers; **Pending** marks a result
still being collected; **Open** marks a decision for the user (Section 11);
**Test** marks a claim for implementation to check.

## 1. Summary

1. **The trunk holds the economic entities and the 25 revenue centers: 45
   rows, not 80.** GLOBAL, 7 CORP, 7 PRIVATE, 25 CENTER, and the players. The
   27 plain HEX rows, 15 CANDIDATE rows, 6 TRAIN rows, PRICE, and DIVIDEND
   leave it.
2. **Every action is scored the same way,** from a *choice table*: one row per
   action of the current decision, filled by the engine with what the action
   does. One head serves all 19 phases (Section 5).
3. **No relation records.** Facts about a pair of entities travel once, as a
   small dense *pair block*, and the model uses each block twice: as features
   on both rows, and as attention-bias channels (Section 4.3).
4. **The map is in the trunk as its 25 revenue centers and nothing else.**
   A lay or a token is a choice with engine-computed outcomes, and a corp's
   tokens, reach, and route stops are a pair block between CORP and CENTER
   rows. **Measured** at full size on v1's buffer (Section 2.4): the engine's
   lay outcomes replace the HEX rows at TRACK; with the 25 centers' rows as
   well, the model is 0.3% behind v1 on held-out policy, inside the seed
   gap. With no location rows it is 1.5% behind, mostly at STOCK.
5. **What it buys is throughput:** about half of v1's trunk time
   (**Measured** on a 5090: 0.55 at 45 rows and batch 256), about a fifth of
   the bytes per state on the wire, and no flatten, filler, or K buckets in
   the eval server. It does not address the economy stall (Section 2.1),
   which needs its own change.

## 2. Evidence

### 2.1 v1 at epoch 10

**Measured** from `runs/1889-v1` against `runs/1889-v0` at the same epoch
(open games: 6s 0.05 against 0.03 a game, Ds 0.02 against 0.00, 88% of games
ending in phase 5 in both, DIVIDEND prior top-1 0.87 against 0.86). v1's own
statistics move the wrong way each epoch: pars at ¥65 from 35% to 58%, phase-5
withholding from 44% to 3%, the richest corp when 5s arrive from ¥740 to ¥426.
Held-out value MSE by quarter is 0.43, 0.28, 0.12, 0.04, which is v0's at
epoch 58.

**Inferred:** the stall is not an input gap, so no layout fixes it. This
design is about cost and about making map plans visible; the stall's remedy
(committed exploration, auxiliary value targets) is a separate document.

### 2.2 Which relations a trained model uses

**Measured** (`ablate.py`): drop a group's records and measure the change in
policy (KL from the unablated output) and in value (mean absolute change), on
600 four-player human positions. This is what a checkpoint uses, not what a
model trained without the relation would lose. The positions are
off-distribution for v0; v1 at epoch 10 is early.

| Group | Share of records | v0@58: KL at TRACK | v0@58: value | v1@10: value |
|---|---|---|---|---|
| HOLDS + SHARES | 10% | 0.035 | 0.13 | 0.04 |
| PRESIDES | 2% | 0.021 | 0.11 | 0.05 |
| Private ownership | 2% | 0.011 | 0.13 | 0.08 |
| REACHES_HEX | 38% | 0.047 | 0.03 | 0.003 |
| REACHES_CENTER | 18% | 0.0004 | 0.005 | 0.001 |
| TRACK_LINK | 9% | 0.0000 | 0.001 | 0.001 |
| STATION | 3% | 0.0000 | 0.001 | 0.000 |
| CORP_TRAINS | 2% | 0.0000 | 0.000 | 0.000 |
| ROUTE (v1 only) | 15% | | | 0.002 |
| Static HEX_SIDE × 6, HOME, ABILITY | (buffers) | ≤ 0.0001 | ≤ 0.001 | ≤ 0.002 |
| Every dynamic relation | 100% | 0.110 | 0.23 | 0.13 |

- **Containment** (`rel_overlap.py`, 3,000 positions): HOLDS and SHARES have
  the same support; STATION ⊂ REACHES_CENTER ⊂ REACHES_HEX; ROUTE ⊂
  REACHES_HEX. Corp-to-hex and hex-to-hex records are 80% of all records.
- **CORP_TRAINS duplicates CORP 26-31** (trains by type), which is why it
  carries nothing.

### 2.3 What the trunk does with HEX and TRAIN rows

**Measured** (`row_ablate.py`, `row_ablate2.py`, 360 positions): forbid
attention from one group of rows to another.

| Cut | v1@10: STOCK KL | v1@10: value | v0@58: STOCK KL | v0@58: value |
|---|---|---|---|---|
| Economic rows cannot read HEX rows | 0.087 | 0.20 | 0.064 | 0.16 |
| ... the 25 center hexes only | 0.024 | 0.13 | 0.043 | 0.14 |
| ... the 27 plain hexes only | 0.039 | 0.14 | 0.020 | 0.10 |
| HEX rows cannot read other HEX rows | 0.011 | 0.07 | 0.006 | 0.06 |
| Nobody reads TRAIN rows | 0.001 | 0.02 | 0.012 | 0.06 |
| Nobody reads PRIVATE rows | 0.032 | 0.15 | 0.059 | 0.13 |

- **The trunk reads the hexes as a bag of locations, not as a graph:**
  cutting them moves outputs more than every relation does, while the
  adjacency relations move nothing.
- **A hard mask overstates** what is read, since attention must go
  elsewhere. Halves are sub-additive (0.13 and 0.14 against 0.20), which fits
  a broad pool more than specific facts.
- **So removing the hex rows is a hypothesis to test by training** (2.4), not
  something these numbers license.

### 2.4 A trunk trained without HEX rows

**Measured** (`scratchpad/v2_experiments/rows/runs/`): three arms trained
from scratch on v0's epoch-58 buffer, as Tier C was, two seeds each, scored on
150,000 held-out rows.

- **Base:** v1's arrangement, with the 52 HEX rows in the trunk (80 rows).
- **No HEX rows** (28 rows): the TRACK, TOKEN, and TILE heads read a hex's
  input features where the base reads its final row.
- **Centers only** (53 rows): the 25 revenue-center hexes stay in the trunk.

The models are small and briefly trained: 128 wide, 6 layers, 512,000 samples
(1.2 passes over the training rows). Each arm starts from its base seed's
weights and data order, so differences are paired by seed and averaged.

| Held out, step 2,000 | Base, two seeds | Seed gap | No HEX rows | Centers only |
|---|---|---|---|---|
| Policy KL, all | 0.1367, 0.1361 | 0.5% | −0.8% | −0.4% |
| STOCK | 0.1601, 0.1594 | 0.5% | −0.8% | −0.8% |
| TRACK | 0.1329, 0.1305 | 1.8% | +0.6% | +0.4% |
| TILE | 0.1659, 0.1663 | 0.2% | +0.9% | +0.9% |
| TOKEN | 0.0409, 0.0422 | 3.0% | −2.3% | −1.5% |
| Value MSE | 0.1504, 0.1500 | 0.3% | −3.8% | +3.3% |

- **No policy loss from removing the HEX rows.** TRACK's 0.6% is inside its
  seed gap. TILE's 0.9% repeats on both seeds and is outside its gap, but is
  a hundredth of the score.
- **Keeping the 25 centers buys nothing** over keeping none.
- **The value differences mostly rank overfitting.** Held-out value MSE rises
  through training in every arm (0.12 at step 500, 0.15 at 2,000) as the
  value head memorizes outcomes, which Tier C also saw at this loss weight.
- **This fits 2.3 and 2.5.** A model trained with hex rows leans on them. A
  model trained without them gets the same result from CORP's map summaries
  and the hex's own features at the head.
- **A run four times longer does not settle it** (`runs_long/`, 8,000 steps,
  4.7 passes, one seed an arm against two base seeds). The models memorize:
  training KL falls to 0.035 while held-out KL rises from 0.137 to 0.151, so
  step 2,000 above is the better-generalizing point. In that regime the arms
  separate a little:

  | Held out, step 8,000 | Base, two seeds | Seed gap | No HEX rows | Centers only |
  |---|---|---|---|---|
  | Policy KL, all | 0.1513, 0.1516 | 0.2% | +1.1% | −0.5% |
  | TRACK | 0.1469, 0.1491 | 1.5% | +4.9% | +4.6% |
  | TILE | 0.2120, 0.2092 | 1.3% | +3.7% | −0.2% |
  | TOKEN | 0.0435, 0.0429 | 1.3% | +1.6% | +3.1% |
  | Value MSE | 0.1608, 0.1633 | 1.5% | −2.3% | +2.5% |

  Without HEX rows, TRACK and TILE degrade about 5% and 4% more than the
  base. The 25 centers recover TILE and not TRACK, so what TRACK misses is
  the plain hexes' rows, which only v1's 80-row trunk has.
- **Two limits of this test.** More passes over 437,000 rows overfit instead
  of fitting better, and the target is v0's play, which seldom uses the map
  well. The full-size run below removes the first.

**Measured at full size** (`scratchpad/v2_cloud/`, results copied from the
cloud box's `results/REPORT.md`): v1's own model, 256 wide and 15 layers,
trained from scratch on v1's replay buffer at epoch 18 with the run's
trainer settings. 2,140,450 training rows; the newest 738 games held out and
150,000 of their rows scored; 30,000 steps at batch 256 (3.6 passes).

- **Base, No HEX rows, Centers only:** as above, on v1 (80, 28, and 53 rows).
  An arm that drops HEX rows also drops every record that touches them, so
  its CORP rows no longer know which hexes the corp tokens, reaches, or runs.
- **+ lay features:** each legal hex's TRACK logit also reads the outcome of
  the best tile to lay there, computed by the engine, which is what v2's
  TRACK choice rows carry (5.2).
- Two seeds for base, No HEX rows, and No HEX rows + lay; one for the others.
  Differences are paired by seed and averaged.

| Held out, step 30,000 | Base, two seeds | Seed gap | No HEX rows | No HEX rows + lay | Centers only | Base + lay |
|---|---|---|---|---|---|---|
| Policy KL, all | 0.1443, 0.1437 | 0.4% | +2.7% | +2.6% | +0.9% | +0.2% |
| STOCK | 0.1800, 0.1795 | 0.3% | +3.2% | +3.5% | +1.4% | +0.5% |
| TRACK | 0.1080, 0.1060 | 1.9% | +5.4% | +1.3% | +1.6% | −1.9% |
| TILE | 0.1564, 0.1560 | 0.3% | +2.1% | +2.5% | +1.4% | +0.9% |
| TOKEN | 0.0298, 0.0306 | 2.6% | +1.7% | +1.0% | +3.0% | −0.2% |
| Value MSE | 0.1452, 0.1458 | 0.5% | +0.9% | +1.1% | +0.3% | +0.1% |

| Held out, step 10,000 | Base, two seeds | Seed gap | No HEX rows | No HEX rows + lay | Centers only | Base + lay |
|---|---|---|---|---|---|---|
| Policy KL, all | 0.1380, 0.1341 | 2.9% | +2.0% | +1.2% | +1.0% | −0.1% |
| STOCK | 0.1734, 0.1680 | 3.2% | +2.1% | +1.3% | +0.9% | −0.7% |
| TRACK | 0.0990, 0.0963 | 2.7% | +4.3% | −0.0% | +6.4% | −0.2% |
| TILE | 0.1416, 0.1392 | 1.7% | +0.7% | +1.1% | +1.0% | +3.3% |
| TOKEN | 0.0313, 0.0396 | 23.5% | −5.4% | −3.5% | +3.1% | +4.0% |
| Value MSE | 0.1302, 0.1364 | 4.6% | +2.1% | +3.3% | +3.9% | +5.8% |

- **The lay features close TRACK's gap.** Without HEX rows TRACK is 4-5%
  worse at steps 10,000, 20,000, and 30,000 on both seeds. With the lay
  features it is −0.0%, +0.6%, and +1.3%, inside the seed gap each time.
  Added to the base they change little (−0.2% to −1.9%), so they stand in
  for the HEX rows at TRACK and do not add to them.
- **A gap of 2-3% remains, and STOCK carries it.** STOCK is 48% of the rows.
  Its gap is +0.3% at step 5,000 (0.6 passes), about 2% at 10,000, and about
  3% from 20,000 on, with both seeds agreeing and the lay features making no
  difference. SELL_AMOUNT (+4% to +8%) and TILE (+2%) are short as well.
- **It is not an effect of memorizing.** All arms memorize after step 10,000:
  training KL falls from 0.110 to 0.037 while the base's held-out KL rises
  from 0.136 to 0.144. But the No-HEX arms' training KL is also about 3%
  higher (0.0377 and 0.0381 against 0.0367 and 0.0368 at step 30,000). The
  base fits the search's STOCK policy better from something the HEX rows or
  their records carry.
- **The 25 centers recover about half of it** (one seed): all +0.9%, STOCK
  +1.4% at step 30,000. Their arm keeps the corp-to-center records.
- **These arms have no stand-in for those records.** The follow-up below
  adds one, as slots on the CORP and GLOBAL rows, and compares it with
  keeping the centers' rows.
- **Value is within about 1%** at step 30,000 and inside the seed gap before.
  The two No HEX rows + lay seeds disagree there (+3.4%, −1.2%).
- **Limits.** The target is v1's search up to epoch 18, which is early. Two
  seeds screen a difference and do not test it. The arms are v1 with rows
  removed, not v2.

**Measured, the follow-up** (`run_more.sh`, the same split, settings, and
base runs): two more arms, two seeds each.

- **No HEX rows + lay + summaries** (28 rows): per-center slots, built in
  the forward from the batch's own inputs. CORP gets its token, reach,
  and route flags for each of the 25 centers; GLOBAL gets each center's
  revenue and free token slots. This is the nearest stand-in for a trunk
  with no location rows.
- **Centers + lay** (53 rows): the 25 centers' rows and their records, with
  the lay features.

| Held-out policy KL against base | Seed gap | No HEX rows + lay | + summaries | Centers + lay |
|---|---|---|---|---|
| Step 10,000, all | 2.9% | +1.2% | +1.0% | +0.6% |
| Step 10,000, STOCK | 3.2% | +1.3% | +1.0% | +0.6% |
| Step 20,000, all | 0.6% | +3.3% | +2.0% | +0.4% |
| Step 20,000, STOCK | 0.8% | +4.4% | +3.3% | +0.9% |
| Step 30,000, all | 0.4% | +2.6% | +1.5% | +0.3% |
| Step 30,000, STOCK | 0.3% | +3.5% | +2.0% | +0.3% |
| Step 30,000, SELL_AMOUNT | 1.1% | +7.6% | +6.9% | +3.2% |
| Step 30,000, TRACK | 1.9% | +1.3% | +0.2% | −0.5% |
| Step 30,000, TILE | 0.3% | +2.5% | +1.8% | +1.3% |
| Step 30,000, TOKEN | 2.6% | +1.0% | +3.2% | +3.1% |
| Step 30,000, value MSE | 0.5% | +1.1% | +1.5% | +1.0% |

- **The summaries close about 40% of the gap, the centers' rows about 90%.**
  At step 30,000 the summaries arm is 1.5% behind the base and the centers
  arm 0.3%, which is inside the seed gap. Directly compared, the summaries
  arm is 1.2% behind the centers arm overall and 1.7% at STOCK, with both
  seeds agreeing (+1.0% and +1.4%; +1.7% and +1.7%).
- **Nothing separates the arms at step 5,000 or 10,000.** Every difference
  there is inside the seed gap. The ordering appears once the models are
  trained further, as the first run's gap did.
- **TRACK is level in all three,** so the lay features do their job with or
  without location rows. They also help the centers arm: its TRACK is 3-6%
  better with them than without.
- **Value does not separate them:** all three are 1.0-1.5% behind the base
  at step 30,000 and about 2.5% at 20,000.
- **What the centers' rows hold that the summaries do not:** each center's
  tile and its upgrade state, and attention between a corp and a center
  along the token, reach, and route records. The run does not say which of
  these matters.
- **The centers arm is v1's HEX rows and records** for those 25 hexes, not
  layout 3's CENTER rows. Layout 3 carries the same facts: the rows' own
  slots (4.2) and a CORP × CENTER pair block (4.3).

### 2.5 The choice head's structure

**Measured** (`scratchpad/v2_experiments/head/results_v2/`): heads retrained
on v0@58's frozen trunk against the buffer's visit targets, split by game,
scored by held-out policy KL. Three forms of choice head were tried:

- **Independent:** each choice scored alone, from its own features and the
  context rows.
- **Rank:** independent, plus each feature's standing among the legal choices
  (its gap to the best and the worst, and its rank).
- **Decoder:** one block of self-attention among the choices and
  cross-attention to the trunk's final rows.

Each is compared with v0's own head form retrained the same way, since the
checkpoint's head has already seen the test rows. A positive difference is
worse; a star marks a 95% interval that excludes zero.

| Phase | v0's form, KL | Independent | Rank | Decoder |
|---|---|---|---|---|
| TRACK | 0.0825 | +0.0166* | +0.0171* | +0.0255* |
| BID | 0.1700 | +0.0007 | +0.0025 | +0.0062 |
| PAR | 0.0489 | +0.0028* | +0.0025* | +0.0035* |
| DIVIDEND | 0.0127 | −0.0000 | −0.0001 | +0.0006* |
| TRAIN_PRICE | 0.0252 | +0.0002 | +0.0001 | +0.0007 |
| SELL_AMOUNT | 0.1030 | −0.0021 | −0.0037* | +0.0074* |
| PRIVATE_PRICE | 0.0120 | −0.0008* | −0.0008* | −0.0008 |

- **Rank features add nothing.** Rank minus independent spans zero in every
  phase (TRACK: +0.0005, interval −0.0003 to +0.0013). Scoring a choice does
  not need the other choices' features.
- **The decoder never beats independent scoring.** It is worse by 0.009 on
  TRACK and 0.010 on SELL_AMOUNT, with four to six times the parameters. A
  second block, a width of 128, and HEX rows in the cross-attention change
  nothing.
- **That is weaker than it looks.** The heads saw 7,000-60,000 rows and the
  decoders overfit (TRACK: train KL 0.038 against 0.108 held out). Their gap
  to independent scoring fell from 0.013 to 0.009 as the rows doubled. So this
  shows no benefit at this size, not that a decoder would lose in a full run.
- **Per-choice rows can replace the joint heads.** Level on BID, DIVIDEND, and
  TRAIN_PRICE, slightly better on SELL_AMOUNT and PRIVATE_PRICE, 6% worse on
  PAR.
- **TRACK loses 20% without the hex's own trunk row.** v0's form points at the
  chosen hex's final row; the others read only the hex's raw slots. No head
  form gets the loss back. Two things qualify it: this trunk was trained to
  put a hex's context in its row, and the raw slots were layout 1's, without
  the outcomes of the lay that 5.2 adds. 2.4 trains the trunk without those
  rows: TRACK is 4-5% short without the lay outcomes and level with them.

### 2.6 Cost

- **The eval server's thread, not the GPU, limits the current run.**
  **Measured** from `runs/1889-v1` (2 servers, 256 workers, GPUs at 50-60% by
  the user's reading): `build` 58% of the serve loop, `gather` 16%, `forward`
  18%; workers wait 96% of the time.
- **A state is about 37 KB on the wire today:** 95 × 109 fp16 token slots
  (20.7 KB) and a 4,096 × 4 record buffer (16.4 KB) that holds 475 records on
  average (1.9 KB).
- **Rows cost linearly.** v1-review Section 8 puts the quadratic term near 6%
  of a forward at 95 rows, and v1's 80 rows measured 0.85 of v0's 95 at batch
  256 on the 5090.

**Measured** (`scratchpad/v2_experiments/cost/`, 2026-11-12; all 45,355 human
positions and v0's 1M-row buffer; one core of a 9950X3D with other jobs
running):

- **Forward time by rows** (CPU, 8 threads, batch 64, v1's block at 256 × 15):
  45 rows take 0.54 of 80's time, 28 rows 0.29, 20 rows 0.21.
- **Forward time by rows on a 5090** (`scratchpad/v2_cloud/`, 2026-11-13;
  compiled, bf16, a CUDA graph, v1's block at 256 × 15 with a pair-block
  bias; ms a launch, and the ratio to 80 rows):

  | Rows | Batch 64 | Batch 128 | Batch 256 | Batch 512 |
  |---|---|---|---|---|
  | 80 | 1.72 | 2.90 | 5.56 | 11.87 |
  | 53 | 1.15 (0.67) | 1.87 (0.64) | 3.57 (0.64) | 7.19 (0.61) |
  | 45 | 1.12 (0.65) | 1.73 (0.60) | 3.03 (0.55) | 5.92 (0.50) |
  | 28 | 0.81 (0.47) | 1.25 (0.43) | 2.02 (0.36) | 3.75 (0.32) |
  | 20 | 0.76 (0.44) | 1.03 (0.36) | 1.64 (0.29) | 2.86 (0.24) |

  The saving grows with the batch: small launches are bound by per-launch
  overhead, not by rows.
- **A larger trunk at 20 rows,** as a ratio to v1's 80 rows at 256 × 15,
  batch 256: 256 × 24 costs 0.47, 384 × 15 costs 0.52, 384 × 20 costs 0.69,
  512 × 15 costs 0.75. Larger shapes were not timed at 45 rows.
- **The eval server's relation work on the cloud's CPU** (a Xeon Gold 6530):
  at 256 states a launch, 3.5 ms for the gather and flatten v1 ran with,
  0.09 ms for the fused gather (commit 2b67811), against a 5.6 ms forward.
- **The head, as a share of the 45-row trunk:** an MLP over 46 choices 2.0%;
  one decoder block 10.9% at width 256 and 4.1% at width 128; two blocks
  19.7% and 7.3%.
- **Choice sets are small.** Legal actions a decision: mean 4.5 in self-play
  (5.4 human), 99th percentile 16 (19), maximum 28 (31). Every decision over
  17 is a TRACK. No recorded decision has one action: the driver plays those.
- **Applying an action is cheap; chaining is not.** Per decision, for the
  whole choice set, weighted by self-play's phases:

  | | µs, mean |
  |---|---|
  | Today's leaf extraction (layout 2 tokens and records) | 26.2 |
  | Copy, apply the action alone, read 15 observables | 1.6 |
  | ... and the corp's route max where it went stale | 2.4 (3.6 on leaves) |
  | Apply through the driver's forced follow-ups, and route max | 9.4 |

  A first, cold run costs about twice these.
- **No PRIVATE row is needed** in 16% of self-play rows (28% of human ones):
  none before phase 5, about 80% from it. Seats: a third each of 3, 4, and 5
  players in self-play.

## 3. Scope

**In scope:** token layout 3 (Section 4); the choice table and its engine
fills (Section 5); `nn/transformer-1889-v2.py` (Section 6); the transport and
contract changes that follow (Section 7).

**Out of scope:** the action space (the 19 phases and 228 unified slots stay
as `games/g1889.py` has them); search; the value target; exploration and
auxiliary targets; live play. RSS does not change, and
`tests/test_v2_parity.py` passes on its fixture.

**Build distance** is in the layout but staged: the engine module is
[`build-distance-design.md`](build-distance-design.md) Sections 3 and 5 as
written, and this document replaces that design's Sections 4 and 6 (where its
outputs go). Layout 3 has the slots from the start, zero until the module
lands (Section 10, step 6).

## 4. Token layout 3

### 4.1 Rows

| Type | Count | Was | Notes |
|---|---|---|---|
| GLOBAL | 1 | GLOBAL, the six TRAIN rows' state, PRICE's and DIVIDEND's context | 4.2 |
| CORP | 7 | CORP | 4.2 |
| PRIVATE | 7 | PRIVATE | unchanged but for the per-choice slots (16, 17), which move to choices |
| CENTER | 25 | the HEX rows of the cities, towns, and offboards | 4.2 |
| PLAYER | max_players | PLAYER | 4.2 |

- **45 rows at capacity 5** (43 and 44 at 3 and 4). Players stay last.
- **Gone from the trunk:** the 27 plain HEX rows, CANDIDATE, TRAIN, PRICE,
  DIVIDEND. What they held is in GLOBAL, on CORP rows, in pair blocks, or in
  the choice table (4.4 maps every slot).
- **Why the centers and only the centers** (2.4): with the lay outcomes on
  TRACK's choices, the 25 centers' rows bring the model to within 0.3% of
  v1, and no location rows leave it 1.5% behind. The 27 plain hexes' rows
  add nothing measurable beyond the centers', and would take the trunk from
  45 rows to 72. Build distance (Section 10, step 6) needs the CENTER rows as well.
- **Hidden rows** are as today: privates not in play or closed, and padded
  seats, with slot 0 as the key mask. They still cost a query each; trimming
  them needs launches of two lengths, which is not planned (Section 11,
  item 5).

### 4.2 Row contents

Layout 2's slots carry over unless listed. Final numbers are assigned in
`token-data-1889.md` at implementation.

**GLOBAL** gains, per train type (6 × 7 slots): depot copies, pool copies,
the depot sells it now, cheapest available, rusted, rusts at the next phase,
its next depot purchase starts a phase. List prices and rust tables are
constants and become model buffers. GLOBAL keeps the tile-supply counts.

**CORP** loses the per-choice slots (15-19 sale prices, the STOCK and
BUY_TRAIN choice slots), which become choice features. Its facts about each
revenue center are the CORP × CENTER pair block (4.3), which the model
appends to the row as it does every pair block. It gains the six
build-distance summaries of build-distance-design 4.2, zero until step 6.

**CENTER,** one row per city, town, and offboard, in `HEX_RC` order, with a
learned identity per row. About 21 slots:

- **The tile:** its id and rotation, as integers. The model looks the tile
  block up, as it does for TILE's choices (5.1).
- **Revenue:** now /100 (an offboard's by phase), and to a D train /100.
- **Tokens:** tokens /3, slots open to any corp /3, a home reservation still
  open.
- **The hex:** mountain, water, the current tile's lay cost /100, blocked by
  a private, dangling exits by side, upgrade targets now and in supply.
  These are layout 2's HEX 78-93.
- **is_selected:** TILE's pending hex, and C4 at ER_BONUS, as in layout 2.
- **Not on the row:** anything relative to the decision corp (HEX 94-104).
  That is on the choices (5.2) and in the pair block.

**PLAYER** loses shares by corp, sold-this-SR, and bids by private (12-32),
which are pair blocks now.

### 4.3 Pair blocks

A pair block is a dense array over two row types. It crosses the wire once.
The model appends each entity's slice to its row's features (a PLAYER row gets
its 7 corps' values; a CORP row gets its players'), and mixes the same
values into the attention bias in both directions, each direction with its
own learned weight per layer and head, as `relation_bias_mult` does today.

| Block | Shape | Channels | Replaces |
|---|---|---|---|
| PLAYER × CORP | P × 7 | shares /10; is president; sold this SR | relations 0-5, 25, 26; PLAYER 12-25 |
| PLAYER × PRIVATE | P × 7 | owns; bid /100 | relations 6, 7, 10, 11; PLAYER 26-32 |
| CORP × PRIVATE | 7 × 7 | owns | relations 8, 9 |
| CORP × CENTER | 7 × 25 | has a token there; reaches it; it is a stop on the corp's best routes; build distance D_c /6 (step 6) | relations 14, 15, 18, 19, 27, 28 |
| CENTER × CENTER | 25 × 25 | build distance /6 (step 6) | nothing in layout 2 |

- **About 1,550 values a state** at capacity 5: 224 for the first three
  blocks (105 + 70 + 49), 700 for CORP × CENTER, 625 for CENTER × CENTER.
- **The first three are the relations 2.2 found in use:** holdings,
  presidency, and private ownership.
- **CORP × CENTER is what 2.4's centers arm kept and its summaries arm
  flattened.** As a pair block it reaches a CORP row as its 25 centers'
  values, a CENTER row as its 7 corps' values (which corps token it, reach
  it, and run it), and the attention between the two as bias.
- **The distance channels are zero until step 6.** The bias weights and the
  feature projections' columns for them start at zero and get no gradient
  until the channels are filled, so build distance can be switched on in a
  running model. A distance is 1 where there is none within 5. For the bias a
  distance is bucketed in the model (one-hot of 0, 1, 2, 3, 4-5), so the
  bias is not linear in distance (build-distance-design 4.1's reason).
- **Dropped with no replacement:** TRACK_LINK, the static HEX_SIDE, HOME, and
  ABILITY planes, CORP_TRAINS, and REACHES_HEX as a relation. A corp's reach
  over plain hexes survives as CORP's reach counts and as each TRACK choice's
  reach features.
- **Build distance arrives both ways:** corp-to-center in CORP × CENTER,
  because the one map relation v0 uses is corp-to-hex and no checkpoint
  composes through STATION (2.2); and center-to-center, the edges only
  location rows can carry.

### 4.4 Where layout 2's rows went

| Layout 2 | Layout 3 |
|---|---|
| HEX, a center's row | its CENTER row (tile id and rotation, revenue now, slots 78-93); slots 94-104 to its TRACK and TOKEN choices |
| HEX, a plain hex's row | nothing in the trunk; its tile and slots 78-104 to its TRACK choice |
| CANDIDATE k | TILE or ER_BONUS choice k |
| TRAIN t: state (2-9, 14) | GLOBAL's per-type block |
| TRAIN t: 10-13 | BUY_TRAIN, SELECT_TRAIN, DISCARD, TRAIN_PRICE choices |
| PRICE 5-29 | BID, PAR, TRAIN_PRICE, PRIVATE_PRICE choices |
| PRICE 22, 23 (reference, payer's funds) | GLOBAL |
| DIVIDEND | DIVIDEND's two choices |

## 5. The choice table

### 5.1 Shape

`choices`: `(46, F)` per state. Row k is the current decision's phase-local
action k; 46 is TRACK's size, the largest. A row whose action is illegal is
zero, and the legal mask excludes it as now. F is about 24 (**Test:** fixed
at implementation).

**Not compacted to the legal actions.** A table of 32 rows would hold every
decision seen (2.6), but TRACK's bound is its 46 slots, so a smaller table
needs a fallback for the decision that overflows it, and rows indexed by
action id keep what each row names static. The 14 spare rows cost 0.6% of the
trunk under an MLP head.

What a row names is static per action slot, so it never crosses the wire:

- **A slot embedding,** one learned vector for each of the 228 unified slots.
  For TRACK's slot of hex h it is that hex's identity, which is where the
  fixed map's geometry lives.
- **A named row,** for slots that point at an entity: CORP c (STOCK,
  BUY_TRAIN's corp slots), PRIVATE p (AUCTION, the private purchases),
  CENTER r (TRACK, TOKEN, and PORT on a revenue center). A plain hex has no
  row to name; its choices carry its features (5.2).
- **A tile,** for TILE, ER_BONUS, and TRACK: tile id and rotation ride in the
  row as integers and the model looks up the tile block.

### 5.2 Features

**A shared block,** the same slots for every action: the change in the
actor's cash, in the affected corp's treasury, in its share price, in the
actor's certificates against the limit, in the corp's route max now and with
the next train, and the cost paid. Zero where an action leaves one unchanged.

**A phase block,** whose slots mean what the phase says:

| Decision | Phase block |
|---|---|
| AUCTION, BID | the price, the minimum bid, the high bid, cash left after |
| STOCK | kind (IPO, pool, par, sell); becomes or loses president; floats the corp; shares still to sell for the float; the actor's largest sale; price after the sale |
| PAR | the price; shares the actor can afford; treasury at float |
| SELL_AMOUNT, EMERGENCY_SELL_AMOUNT | shares sold; price after; cash raised; cash still to raise |
| BUY_PRIVATE_*, PRIVATE_PRICE | the price; face; revenue; the buyer's cash after |
| TRACK | the hex's layout-2 slots 78-104; the best over its tiles of TILE's outcomes |
| TILE, ER_BONUS | layout 2's CANDIDATE 99-108; paths added; build-distance-design 4.3's nine slots |
| TOKEN | the corp's route max gained, with its trains now and with the next train; opponents' route max lost, summed and largest, likewise now and with each one's next train; the price; slots left |
| DIVIDEND | per share and total paid; shortfall to the next train after; the president's and each rival's income |
| BUY_TRAIN, SELECT_TRAIN, TRAIN_PRICE | source; price; starts a phase; the buyer's trains rusted; opponents' trains rusted; the seller's route max lost; the president's contribution |
| DISCARD | route max lost |
| PORT | none beyond the tile and the shared block |

- **New against layout 2:** TOKEN's opponents' loss (blocking is invisible
  today; v0 places 4-5 tokens a game against humans' 7.4), the trains a
  purchase rusts for each side, and rivals' income at DIVIDEND.
- **What blocking the model still cannot see** until build distance (step 6):
  a lay or a token that costs a rival no route it runs now or with its next
  train, but puts a center farther from it. TILE's rows say only whether a
  lay *helps* other corps (CANDIDATE 106, 107). Build-distance-design 4.3's
  slots 113-115 add the centers a lay puts farther, for the corp and for the
  others; TOKEN's rows take the same three slots at step 6.
- **Pass, drop, and done** are rows like any other, with a slot embedding and
  an all-zero phase block.

### 5.3 How the engine fills it

- **Generic:** copy the row to the scratch state, apply the action alone,
  and diff the shared block's observables. `_fill_candidates` does this for
  lays and layout 2's train features for purchases (v1-design 5.9).
- **The action alone, never the driver's chain.** **Measured** (2.6): with
  forced follow-ups played, 21% of tile lays, 35% of BUY_TRAIN actions, and
  30% of BUY_PRIVATE_POST passes show a diff that is partly a later event (a
  forced train purchase, the next stock round's price moves), and the actor
  is often no longer the one deciding. It also costs six times as much. So
  the fill calls the driver's dispatch without its chain, which needs a
  `cdef` entry in `g1889/driver.pxd` for extraction to use.
- **After the action alone the state may rest mid-step** (the SR's end after
  the last pass, ROUTE after a token pass). The observables are still raw
  reads; the row is not a decision state and nothing else is read from it.
- **An action that only opens a sub-decision has a zero diff:** selecting a
  bid, a par, a sale, a hex, a train's source, or a private. That is correct.
  Its outcome is the phase block's preview, as layout 2's STOCK and TRAIN
  slots give it today.
- **Route max is the one observable that can force a solve,** and only where
  the action changes track, tokens, or trains: every tile lay, token, discard,
  and train price; never in an auction, a stock round, PAR, or DIVIDEND.
- **One solve per resulting position, not per choice:** every price point of
  TRAIN_PRICE, PRIVATE_PRICE, and BID reaches the same holdings, so the route
  delta is computed once for the decision. Unshared, TRAIN_PRICE's table costs
  2.5 times today's extraction of the leaf.
- **Closed form** where applying is wasteful: price grids and sale amounts.
- **Per phase:** the phase block's own fills, most of which exist as layout
  2's per-choice slots.
- **The apply holds the GIL:** phase handlers are `except *` and take Python
  objects, where today's fills run `nogil` throughout. Self-play workers are
  single-threaded processes, so they lose nothing. **Test:** that no caller
  (the trainer's batch re-extraction, the eval tools) relies on extraction
  releasing it.
- **Test:** per-choice truth, as v1-design 5.9: each feature against the
  state after the real action, for every phase.
- **Test:** purity: the same table whatever the row's caches hold.

### 5.4 The head

One module scores all phases:

    h_k   = slot_embed[u_k] + W[phase] · f_k + named_proj(row[ptr_k]) + tile_proj(tile_k)
    logit = MLP([g, a, c, s, h_k])

- `W[phase]` is a first layer indexed by decision phase (19 × F × d), since a
  phase block's slots mean different things by phase.
- `g, a, c, s` are GLOBAL, the actor, the decision corp, and the decision's
  subject (the private bid on, the corp sold, the train's seller), found by
  the `is_selected` slots as v1 finds them.
- **Decided: each choice is scored independently,** with no attention among
  choices or back to the trunk, and no rank features. 2.5 found no gain from
  either, and a decoder block costs 4-11% of the trunk against the MLP's 2%.
- **Not built:** a decoder. If Section 9's offline parity shows a phase short
  of v1, a block between the two lines above is the first thing to try there.
- The scores scatter into the phase's block of the unified logits; fp32
  before the −1e9 mask, as now.

## 6. The model

`nn/transformer-1889-v2.py`, `INPUT_LAYOUT_VERSION = 3`.

**Unchanged from v1:** d = 256, 4 heads, 15 layers, ff_mult 3.0; the block;
identity and type embeddings; the shared tile projection; zero-sum values and
the value head; initialization; Muon and AdamW's split.

**Changed:**

- **Input:** one packed vector per state, cut by type into rows, pair blocks,
  and the choice table (Section 7.1). Row types keep their own projections.
- **Attention bias:** per layer, zeros of `(B, H, 45, 45)`, with each pair
  block's rectangle filled by an einsum of its channels with that layer's
  weights, both directions; then the hidden keys. No scatter. Weights are
  zero at initialization.
- **Heads:** Section 5.4's module replaces v1's sixteen.
- **Parameters:** **Estimated** about v1's 18.5M: the heads shrink (one MLP
  where there were sixteen), and the slot embeddings (228 × 256), `W` (19 × F
  × 256) is added.

**Test:** v1-design Section 7's list on layout 3, plus: a choice's logit
moves with its own row's features; a hidden row or an illegal choice changes
nothing; pair-block features and the bias built from them agree with the
entity facts on fixture states.

## 7. Transport and contracts

### 7.1 The wire

| Part | Values at capacity 5 (**Estimated**) |
|---|---|
| Rows (GLOBAL ~130, 7 CORP × ~80, 7 PRIVATE × ~14, 25 CENTER × ~21, 5 PLAYER × ~20) | ~1,410 |
| Pair blocks (4.3) | ~1,550 |
| Choice table (46 × 24) | ~1,100 |
| Total | ~4,100, or 8 KB as fp16 |

Against about 37 KB today. Types are packed at their own widths; nothing is
padded to the widest row.

### 7.2 What changes

Following AGENTS.md's active path:

- **`Game` adapter:** `relation_format` gains `"none"`; the fills become one
  `fill_inputs(state, out, layout_version)` for layout 3. RSS's and layouts 1
  and 2's paths stay until v2 has beaten v1 (as v1 kept layout 1 for v0).
- **`nn/model_contract.py`:** `ModelInputSpec` describes a packed input by
  its parts' offsets. The forward takes `(inputs, legal_mask)`.
- **Eval server and IPC:** one input buffer a state. The flat-record path
  (flatten, filler, K buckets) is not used by layout 3: launches bucket by
  batch alone, 9 shapes where v1 has 45.
- **`mcts/evaluator.py`, the trainer, replay:** replay stores raw states and
  re-extracts, so only the extraction call changes.
- **`train/analyze_game.py`, the profile, the tournament:** each evaluator
  fills its own model's layout, as v0 and v1 do in one game today.

**Test:** the transport tests of `tests/g1889/test_relation_transport.py` and
`test_pipeline_v1.py`, on layout 3; a tournament game of v1 against v2.

## 8. Cost

From 2.6's measurements; the profile of v2 itself is Section 9's:

- **Trunk:** on a 5090, 0.55 of v1's forward at 45 rows and batch 256 (0.50
  at 512, 0.65 at 64), with the bias built as an einsum over pair blocks.
  The 20 rows of a trunk without CENTER rows would cost 0.29.
- **Head:** 2.0% of the 45-row trunk.
- **On the current cloud box the GPUs are not the limit** (2.1), so the
  smaller trunk shows up there as GPU headroom, not as games per hour. The
  headroom can be spent on a wider or deeper trunk.
- **Extraction:** the choice table's generic part adds about 2-4 µs a leaf
  against 26 µs today. The relation emitter and the 27 plain HEX fills go,
  and a CENTER row is a fifth of a HEX row's width; the phase blocks' fills
  are mostly layout 2's per-choice work moved.
- **Server:** no flatten, about a fifth of the bytes to gather and copy.

## 9. How to judge it

1. **Offline parity** on v1's replay buffer, as v1-design Section 10 ran Tier
   C: v1 against v2, at full size. Held-out policy KL by phase and value
   MSE. 2.4's full-size run is this test for v1 with rows removed; this one
   is for v2 itself, with its CENTER rows, pair blocks, and choice table.
   The phases to watch are STOCK and SELL_AMOUNT, where 2.4's arms fell
   short without location rows. If v2 is short of v1 in a phase, the first
   thing to try is a decoder block at the head for that phase (5.4); the
   plain hexes' rows are the other thing v1 has and v2 does not, and 2.4
   found nothing from them. Removal is what this harness measures well:
   parity against v1's own search is the bar, as it was for `cand_head`.
2. **The profile** (`cloud/profile_1889.py`) on v2: model time, a served
   launch, extraction by phase, and the eval server's stage shares.
3. **In play:** v2 trained from scratch beside v1's curve, then a tournament.
   The comparison is confounded with whatever stall remedy lands with it, as
   v1 against v0 was.

## 10. Sub-steps

| # | Scope | Acceptance |
|---|---|---|
| 1 | The choice-table fills in the engine, per phase, with the generic apply-and-diff | per-choice truth and purity on every phase; the cost within 2.6's budget |
| 2 | Layout 3's rows and pair blocks; `token-data-1889.md` | fixture states against the entity APIs; layouts 1 and 2 unchanged on their fixtures |
| 3 | `nn/transformer-1889-v2.py` | Section 6's Tests |
| 4 | The packed input through the adapter, evaluator, eval server, trainer | Section 7's Tests; a smoke run with a resume |
| 5 | Offline parity and the profile (Section 9, items 1 and 2) | v2 within v1's seed spread by phase, or the user's decision on what it is short of |
| 6 | Build distance: `g1889/entities/build.pyx`, its CORP summaries, the distance channels of the two CENTER pair blocks, and its choice slots | build-distance-design Section 5's Tests; gated as its Section 9 says |

Steps 1-3 can be checked on v1's saved buffer before any self-play runs.

## 11. Decisions

Mine to propose, the user's to settle:

1. **Location rows: the 25 CENTER rows, a 45-row trunk, and no rows for
   plain hexes.** *Settled by the user, 2026-11-13.* An earlier draft
   proposed 20 rows with no location rows, on the condition that per-center
   slots on CORP matched the centers' rows. They did not (2.4):
   - With the lay outcomes and the per-center slots, a trunk without
     location rows is 1.5% behind v1 on held-out policy and 2.0% at STOCK.
   - With the 25 centers' rows it is 0.3% behind, inside the seed gap.
   - The 27 plain hexes' rows add nothing measurable beyond that.
   - The cost is 0.55 of v1's forward where 20 rows would cost 0.29 (2.6).
     On the cloud box the GPUs are not the limit, so the trunk time 20 rows
     would save would not become throughput.
   - Build distance (step 6) needs these rows anyway.
   - The limits of the evidence: the gap between the two is 1.2% of held-out
     KL, it appears only after the first pass over the data, and value does
     not separate the arms.
2. **The choice head scores each choice independently,** with no decoder and
   no rank features. *Proposed yes:* neither helped in any phase (2.5), and
   the simplest form is also the cheapest.
3. **Relations as pair blocks,** with no records on the wire. *Proposed yes.*
4. **TRAIN, PRICE, and DIVIDEND fold** into GLOBAL and the choice table.
   *Proposed yes.*
5. **No launches of two lengths** for closed privates and empty seats.
   *Proposed no.* Seven PRIVATE rows are idle in 16% of self-play rows and
   one seat on average, so a second length would save about 5% of trunk work
   and split every launch in two.
6. **Build distance staged** behind the layout, not in the first v2 run.
   *Proposed yes:* its gate (the economy) has not opened.
7. **TRACK and TILE stay two decisions,** and TRACK's row carries the best
   outcome over its hex's tiles. One decision over (hex,
   tile, rotation) would score every lay by its own outcome and take one
   search ply where two are taken now.
   - **Measured** legal lays a TRACK decision: mean 10, 99th percentile 47,
     maximum 85 in self-play (24,472 decisions from v0's buffer); mean 24, 99th
     percentile 80, maximum 102 in human games (5,546 decisions).
   - The block needs a fixed size: 102 is the most seen, and 45 hexes × 15
     candidates = 675 is the only bound the engine proves today.
   - It changes the action space, which Section 3 leaves alone: legality, the
     driver, the replay importers, and live play all follow.
   - *Settled by the user, 2026-11-13:* they stay two decisions for now. The
     choice table makes merging them a later change of one phase block, not
     of the model.

**Open:**

- Whether v2 starts a fresh run or waits for the stall remedy, so the two are
  not confounded more than they must be.
