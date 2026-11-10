# 1889-v0 review and v1 proposal

Status: review of the v0 run at epoch 58, dated 2026-11-09. The user settled
Section 9's decisions the same day. Nothing here is built. Sources: `nn/transformer-1889.py`,
[`model-design.md`](model-design.md), [`../token-data-1889.md`](../token-data-1889.md),
`train/PIPELINE.md`, the run's TensorBoard logs, checkpoint 58, the 87 scraped
3-5 player human games, and 43 games played locally by checkpoint 58. Scripts
and full tables are in `scratchpad/v1_review*/`. **Measured** marks a number
from those; **Inferred** marks a reading of them; **Not checked** marks a gap.

## 1. Summary

1. **v0 has learned a coherent but degenerate game, and it starts in the first
   stock round, not in phase 5.** Self-play pars almost every corp at the ¥65
   minimum, floats fewer corps later, never withholds, and builds half the
   stations. When the 5-trains go on sale no corp can afford one, so phase 4
   lasts 5-7 ORs against humans' 2.3 and phases 6 and D are never reached.
2. **The model's size is not the problem.** The capacity probe already said
   so. Nothing in this review argues for a larger or smaller network.
3. **The value head is the weakest part of the pipeline.** Its training loss is
   0.020, but on fresh games it scores 0.206, and in the first quarter of a
   game it is worse than predicting zero. Two causes: its target is almost
   pure rank, which flips on ¥20 differences, and a buffer holds only about
   1,800 games of 560 rows each.
4. **The inputs are correct but omit what the stalled decisions turn on.** The
   extraction audit found no bug in any slot or relation. But nothing at PAR,
   DIVIDEND, or BUY_TRAIN shows what treasury buys: revenue with the next
   train, the shortfall to it, or what the next rust removes.
5. **The trunk does not use the map as a graph.** No attention head in any
   layer passes information between neighbouring hexes. The 52 hex rows are
   55% of the trunk's work.
6. **Self-play is bound by the eval server, and extraction is nearly free.**
   Workers wait on the server 98.8% of the time and use about 1% of their CPU.
   So richer engine-computed features cost nothing that matters, and games
   per hour is the lever.

The proposal in Section 7 follows from these: change how self-play explores
and what the value head learns from first, add the missing inputs second, and
leave the architecture mostly alone.

## 2. What v0 plays

**Measured** (checkpoint 58 against the 87 human games; self-play in three
groups: T, 17 games with training's move selection at 200 simulations; S, 16
strict games at 200; F, 10 strict games at the full 1,197. † marks a
difference whose 95% bootstrap interval excludes zero):

| | Human | T | F |
|---|---|---|---|
| Pars at ¥65 | 27% | 77%† | 92%† |
| Treasury at float | ¥813 | ¥681† | ¥667† |
| Shares sold in SR2 | 0.3 | 7.7† | 8.7† |
| Station tokens per game | 7.4 | 4.8† | 4.0† |
| Tile spending per game | ¥755 | ¥584† | ¥424† |
| Richest corp when 5s first go on sale | ¥658 | ¥327† | ¥236† |
| Corps able to buy a 5 then | 1.43 | 0.19† | 0.00† |
| ORs spent in phase 4 | 2.3 | 4.8† | 6.7† |
| Withhold share, phases 3-5 | 16% | 3%† | 0%† |
| 5s / 6s / Ds bought | 3 / 1.84 / 2.31 | 1.41 / 0 / 0 | 0.70 / 0 / 0 |
| Total revenue per game | ¥18.2K | ¥11.8K† | ¥9.8K† |

- **Group T reproduces training:** 1.41 5-trains a game against the run's
  1.28-1.57.
- **Capitalization is the earliest robust divergence.** Of the ¥1,080-1,290
  treasury gap when 5s arrive, about ¥680-770 is the lower par and ¥180-530 is
  fewer floats. The withholding gap by then is only ¥200-290.
- **Early train tempo is not the cause.** Self-play games that reached phase 4
  on human pace stalled there just as long (5.8 ORs against 5.9).
- **Track is secondary.** Revenue per train at a matched phase is 10-15% below
  humans', and self-play lays sharp curves 3.6 times as often. Low treasury may
  be the cause of the thin track rather than its effect.
- **Not divergent:** auction prices, private sales to corps (about 1.9 times
  face in every group), and corp-to-corp train sales.

**Stranded floats** (the user's note 3). **Measured** on 37 checkpoint-58 games:
49% of stock rounds end with a corp parred but not floated, 76% of games have
one at the end of SR1, the money tied up averages ¥267, and 24% of games end
with a corp that never floated.

**When it set in.** **Measured** from TensorBoard: near-random epochs 1-2
bought 0.5 6s and 0.5 Ds a game. 6s fell to 0.03 by epoch 10 as the DIVIDEND
prior locked onto payout (top choice 0.86 at epoch 10, 0.97 at 58). Since
epoch 24 nothing has moved: 5s are flat at 1.4, and search overturns the
DIVIDEND prior in 0.5% of decisions.

**Inferred: why it holds.** Treasury is worth something only through trains
bought later. A value head that never sees those trains pay off prices
treasury at nothing, so the policy pars low and pays out, and then no corp can
buy the trains. Section 7.3's deviation test fits this: one financed purchase
does not pay against a population with no treasury, which is why the model
reads it as a loss, and why the model reaches phase 6 or D eight times as
often from human boards, which carry 2.5-3 times the treasury.

**Why it matters even if it is an equilibrium.** A human opponent will buy the
6s and Ds. The bot will then sit in phases it has seen in 0.4% of its games,
holding 4-trains about to rust. The D trade-in input has a weight norm of 0.53
against a typical 2.8: it has never been trained.

## 3. The value head

**Measured** (checkpoint 58's raw value head on 23 fresh games it played, one
at full simulations and 22 at 200; MSE against the terminal target):

| | MSE |
|---|---|
| Training rows, epoch 58 (TensorBoard) | 0.020 |
| Fresh games, overall | 0.206 |
| Predicting zero everywhere | 0.374 |
| Fresh games by quarter of the game | 0.432, 0.254, 0.087, 0.050 |

- **The first quarter is worse than a constant.** The head recalls the
  outcomes of similar openings in the buffer. This is where search most needs
  it: AUCTION has the highest policy loss of the common phases, and search
  overturns its prior 52% of the time.
- **About 8-10% of value outputs sit at |tanh| > 0.99.**
- **Caveat:** 23 games, most at reduced search, which makes outcomes somewhat
  less predictable. The full-simulation game scored 0.227, and the capacity
  probe's held-out 0.18 agrees in size.

**Cause 1: the target is almost pure rank.** With `terminal_blend` 0.75, and
the margin divided by the top net worth, the margin term is worth about ±0.04
against ±0.8 for rank. **Measured:** in `game-058`, seats at ¥6,302 and ¥6,282
got +0.26 and −0.24. 35% of the fresh self-play games have two adjacent seats
within 2% of mean net worth (humans: 24%). RSS net worths spread widely, so
the same formula worked there; 1889's finish within about 11-14% of each other.

**Cause 2: few games, many rows each.** A buffer of 1M rows is about 1,800
games, and every row of a game carries the same outcome.

- **The memorising is by game, not by row.** An epoch trains 332,800 samples
  against 280-300K new rows, so a row is drawn about 1.15 times in its life.
  (model-design §5.7's "a live row sees about 3.6" multiplies by the buffer's
  3.3 epochs a second time.) A loss of 0.020 on rows mostly never seen before
  means the head recognises the game from its 560 sibling rows.
- **So fewer rows per game helps but does not cure it.** What counts is the
  number of distinct games in the buffer. A5's held-out loss is the measure.

## 4. The model

**Size: keep 256 × 15, 4 heads.** The probe's verdict stands.

**The trunk does not pass messages along the map.** **Measured** on 371
self-play states from `game-055` and `game-058`, over all 60 layer-head pairs:

- The most any head puts on a hex's six neighbours is 0.10 of its attention,
  against 0.06 for uniform attention; on track-linked neighbours, 0.15.
- Hex rows attend broadly to all hexes in layers 0-2 and mostly to corp and
  player rows from layer 5 on (up to 0.95 on corps).
- The learned bias weights agree: the six adjacency planes and TRACK_LINK peak
  at 1.0-1.5, while PLAYER_SHARES reaches 5.9, PRESIDES 5.3, and
  HEX_REACHED_BY 4.2.

**Inferred:** the model's map knowledge is the engine's features (reach,
candidate deltas) plus what the hex identity embeddings memorise. Multi-hop
reasoning ("where two lays lead") is not happening. The map signals the model
does use are corp-to-hex relations, which is the form build distance takes.

**Hidden rows are processed in every state.** The 15 CANDIDATE rows are
visible in 4-8% of states, yet they run through all 15 blocks as queries in
every one. They are 16% of the rows, and token-wise work is 94% of the
forward.

**Late-game inputs leave the trained range.** **Measured** on human phase-D
states: player cash reaches 6.4 and net worth 10.1 (both /1000) against
self-play maxima near 2; share price 3.5 against 1.0; revenue 10 against 3.7.
All pass through plain linear projections.

**Sound as built:** the shared tile projection, identity embeddings, the flat
relation path, pointer and joint heads, and zero-sum values. No change
proposed.

## 5. The inputs

The extraction audit recomputed every non-choice slot from the spec on 13,668
states and every relation on 32,886, and applied the real action to check
every per-choice feature. **It found no bug.** Extraction is a pure function
of the row. What follows is what the layout does not show.

### 5.1 State the engine uses and the model cannot see

Each was confirmed by changing one raw field, getting byte-identical inputs,
and showing that the game continues differently.

| Hidden state | Effect | Fix |
|---|---|---|
| Market stack order (`market_stamp`) | Breaks operating-order ties for corps on one cell. 48% of self-play STOCK decisions have such a tie, and CORP 38 is 0 in an SR | Prospective operating rank, one-hot, in SRs and ORs |
| `return_step` at PRIVATE_PRICE and ER_BONUS | Whether the corp's turn is still ahead (PRE) or over (POST) | Two GLOBAL slots |
| `auction_turn` during a closed-auction BID | Who resumes, and who opens SR1 | A PLAYER flag |
| Tile supply counts | Only "in supply or not" is visible, and 27 tile ids have one copy | 40 counts on GLOBAL |

### 5.2 Train economics

- **The revenue a purchase brings is shown only at SELECT_TRAIN, which the
  driver almost always plays itself:** 97 of 108 in the four full-search
  self-play logs, every depot purchase among them.
- **Nothing at DIVIDEND or BUY_TRAIN shows** revenue with the next train, room
  under the limit, the shortfall after a withhold against after a payout, or
  what the next rust takes from this corp.
- **Measured** on human DIVIDEND positions: a 6 in phase 5 lifts revenue from
  ¥251 to ¥320 and is affordable now in 9% of them; a D in phase 6 lifts it
  from ¥241 to ¥433 and is affordable in none. Financing takes several ORs, so
  search does not reach the payoff.
- **TRAIN 10 and 11 are wrong when a purchase changes the phase**, as the spec
  admits: 23 of 44 first-5 purchases, 10 of 46 first-6, 6 of 38 D trade-ins,
  two with the wrong sign. Their test builds its expectation the way the code
  does, so it cannot catch this.

### 5.3 Track value

In 69% of the non-forced TILE decisions in self-play, Δ route max is 0 on
every candidate, because it is computed for the trains the corp holds now.
What a lay is worth to a longer train is invisible.

### 5.4 Smaller items

- **CORP 49** overstates the legal emergency sale in 71 of 373 human cases.
- **GLOBAL 39 and 40** (the auction's pass payouts and discount) are never
  cleared and stay set all game in 10-22% of later states.
- **Duplicates:** PLAYER 1 and 2; DIVIDEND 2 is DIVIDEND 1 / 10; PLAYER 10 and
  11; HEX 78 equals the tile's revenue except on three offboards.
- **Dead slots:** PLAYER 26 (TR is never bid on) and PLAYER 34 (bankruptcy
  ends the game).
- **Divisors:** bids and private face are /100 in PRIVATE and PLAYER and /1000
  in PRICE; payouts are /100 beside cash at /1000. CORP 43, CORP 50, PLAYER 36,
  and CANDIDATE 107 exceed 1.
- **The spec's claim that `auction_pass_payout_cap` 1 bounds the auction slots
  is wrong.** The cap forces only a player who can afford the topmost private.
  The smoke buffer shows 10 all-pass payouts and bids of ¥410.
- **Caps hold:** candidates peak at 14 of 15 and relation records at 1,374 of
  2,048. The static maximum of lay candidates is exactly 15.

### 5.5 Cost

Extraction averages 8.8 µs a leaf on self-play states. A worker's cycle is
about 40 ms for 8 leaves, of which it computes for 0.5 ms. The step-7 budget
(10 µs median, set against RSS's extraction) rejected features at 7 µs a
candidate; the binding budget is two orders of magnitude looser.

Human phase-D states are the exception to plan for: 72 µs mean and 3 ms at
worst a leaf, because every lay marks all seven corps' caches stale
(`g1889/entities/hexmap.pyx:46-48`). Marking only the corps whose graph
touches the hex pays for the features above. New solves should be cached in
the row behind `route_dirty`, as the route max is, so the trainer's
re-extraction stays cheap.

## 6. The pipeline and the forward pass

**Measured** from TensorBoard, epochs 11-58:

- An epoch is 1.02-1.15 h: self-play 95-97%, training 2.5%.
- The server runs 25.2K evaluations a second and is never idle. A search makes
  345 evaluations at 1,189 simulations; subtree reuse replays the rest (834
  virtual backups), so the simulation ramp has cost almost nothing so far.
- STOCK is 47% of searched decisions, 265 a game.

**Realized throughput is 69% of the model's.** The profile's model-only time
at a batch of 256 is 7.02 ms, or 36.5K a second. RSS shows the same ratio
(38.7K against 58.8K). The run's 25.2K matches full batches at the profile's
10.16 ms a launch, which would put about 3 ms a launch outside the model. The
mean batch is 197, and bucketed mode pads each launch to the next power of
two, so some launches are padded too. **Not checked:** which of these it is.
Launch sizes and per-stage server times are not logged.

**The epoch-20 rescale was never applied.** The resume came after epoch 23
with the buffer, steps, and temperature windows unchanged. The proposed
changes were under 2%, so nothing was lost, but step 8.5's acceptance line
says it was done.

**Late-game play is nearly greedy.** Temperature falls to 0.5 by move 200-230
of about 560. An action with 4% of the visits is then played 0.2% of the time.

## 7. Proposal for v1

Ordered by how much I expect each to matter. Tier A changes the outcome of the
run; Tier B is correct and cheap; Tier C should be tested offline first.

### Tier A: self-play and the value target

| # | Change | Why | Cost |
|---|---|---|---|
| A1 | Start a share of self-play games (suggest 25%) from positions in the 87 human games, sampled evenly over game phases | Puts treasury-rich boards and phases 6 and D in the buffer, which is the only way the value head learns what treasury buys. The model already carries human boards to phase 6 or D 26% of the time | The loader exists (`utils_18xx/g1889_deviation.py --human`); a start-state sampler in the worker |
| A2 | Replace the rank term with a soft rank: the mean over opponents of tanh((nw_i − nw_j) / (τ · mean nw)), τ about 0.05 | Equals today's rank reward when gaps are large and is continuous when they are small. Stays zero-sum and within ±1 | A config option on `compute_terminal_values`; RSS keeps its formula |
| A3 | **Withdrawn** (see below). Was: playout cap randomization, as KataGo | | |
| A4 | Store each row's final net worths in replay | Lets a saved buffer be re-targeted offline, which today's blended value target does not allow | 5 ints a row |
| A5 | Log a held-out value loss: score each epoch's new rows before training on them, by quarter of the game | The training loss hides a tenfold gap | Small |

**A3 is withdrawn: my cost estimate for it was wrong.** It assumed a full
search keeps costing 345 evaluations. It costs that only because every search
inherits the previous one's subtree: 834 of a search's 1,189 simulations are
replayed visits. Under playout cap randomization a full search follows a run
of fast ones, inherits almost nothing, and costs nearly its whole budget. At a
25% full-search share the cost per move comes to about 0.3 times the budget,
the same as today, for a quarter of the recorded rows. KataGo gains from it
because Go's branching leaves little to inherit; here the chosen move holds
70% of the visits, so reuse already does that job. What buys games per hour is
the simulation budget itself, and what puts more games in front of the value
head at no cost is a longer buffer. [`v1-design.md`](v1-design.md) §2.3
takes both up.

Also worth deciding:

- **Temperature.** Hold the final temperature nearer 0.8, or start the anneal
  later. Less certain than A1-A3.
- **The value blend.** It reaches 0.7 search value by epoch 300. While the
  value head has a known blind spot, bootstrapping from it reinforces the
  blind spot. Consider capping it near 0.3 until the late game is healthy.

**A cheap check before the fresh run.** Resume v0 from epoch 58 with A1 alone
for about 10 epochs (11 hours). If 6s and Ds start being bought and pars rise,
A1 is confirmed before v1 depends on it.

### Tier B: inputs (layout 2) and bookkeeping

| # | Change |
|---|---|
| B1 | Section 5.1's four hidden-state fixes |
| B2 | **Trains.** On every floated corp: route max after the next rust, route max with the depot's next train (through the phase machinery), trains against the limit now and after the next phase. On DIVIDEND: the shortfall to the cheapest train after a withhold and after a payout. On TRAIN at BUY_TRAIN and DIVIDEND: the post-purchase Δ for each offered type. Fix TRAIN 10-11 the same way, and add the seller's Δ |
| B3 | **Game end.** ORs left in the set, and an estimate of ORs until the bank breaks, on GLOBAL |
| B4 | **Floating** (user's note 3). On CORP: shares still needed to float and their cost, and whether the president can pay it now. On PLAYER: can float a corp alone. On PRICE at PAR, per par price: the treasury at float and how many shares the actor can afford |
| B5 | **Market** (note 5). On every corp, at every decision: the price after a payout, after a withhold, and after selling out. Fill the five sale prices at every decision, not only STOCK and BUY_TRAIN |
| B6 | **Order** (note 6). The operating rank as a one-hot (B1), plus an "is operating" flag |
| B7 | **Relations.** SOLD_THIS_SR, player to corp and back (note 6). ROUTE, corp to each hex on its best routes, and back |
| B8 | **Track value.** A second candidate Δ under the depot's next train |
| B9 | Section 5.4's cleanups. Add log-scaled companions for player cash, net worth, share price, and revenue, so late-game values stay in range |
| B10 | Mark only the affected corps' caches stale on a lay or token |
| B11 | The heads read the new raw slots: `dividend`, `buy_train`, `par`, `stock_corp` |
| B12 | **Telemetry** (note 2 and Section 2). Per game: the premium paid over face at auction, player cash entering SR1, par prices, treasury at float, stranded floats, shares sold in SR2, tokens placed, the richest corp when 5s first go on sale. Also the gradient norm and a histogram of launch sizes |

### Tier C: test offline first

Replay stores raw states and re-extracts them, so a saved v0 buffer can score
a new layout or model change with `train/capacity_probe.py`: policy KL by
phase and value loss on held-out games, in about an hour each on the 5090.
This works for anything that helps predict v0's own search. It cannot
validate B2, since v0's DIVIDEND targets are 97% payout.

| # | Candidate | Measure |
|---|---|---|
| C1 | Build distance, as extraction-design §8 plans | TRACK, TILE, TOKEN policy KL |
| C2 | The ROUTE relations (B7) | The same, and value loss |
| C3 | Local mixing for hex rows: before the trunk, add to each hex a projection of its six neighbours' rows, gated by track; or start two heads' adjacency biases near +3 | The same |
| C4 | Candidate rows after the player rows, with launches of 80 rows when no candidate is visible | Must be output-identical; saves about 15% of GPU time |
| C5 | Auxiliary value targets: each player's share of final net worth, each corp's final share price plus dividends still to come, ORs remaining | Held-out value loss; needs A4-style storage |

**Throughput first step:** log launch sizes and per-stage server times (B12).
The 31% gap between the model's rate and the server's is as large as anything
in this table, in both games.

**Not proposed:** a larger model, 8 heads, relation input mixing, or separate
revenue-center tokens. A two-stream design (a short map encoder, with the
economic rows cross-attending to it) could halve the trunk's work, but it is a
new architecture and should wait for C1-C3's answers.

### What v1 gives up

Milestone 6 planned v1 as v0 plus build distance under matched configs. This
proposal changes self-play, the value target, and the layout together, so v1
against v0 will not isolate any one of them. I think that is right: v0's stall
would dominate any feature comparison made on top of it. Strength still needs
1889 tournaments (`train/tournament.py`), which v1 should have from the start,
with v0 at epoch 58 as the opponent.

## 8. The user's notes

1. **PORT has no pass.** The lay is not forced. It is offered as STOCK's slot
   30 beside the turn's other choices, and PORT only chooses where. But the
   behaviour is as observed: in all 21 games checked, MF's owner lays the tile
   within their first few SR1 decisions, with a prior of 0.82-0.92 where
   priors were saved. Where it goes is the least settled decision in the game:
   search overturns the prior 65% of the time. The model sees no value in
   waiting, which is the same long-horizon weakness as Section 3.
2. **Auction telemetry.** Agreed; in B12. The pacing comparison found auction
   prices close to humans'.
3. **Stranded floats.** Confirmed (Section 2). B4 extends the proposed flags
   to the PAR decision, where the stranding is chosen.
4. **CORP slot 3.** It cannot fold into slot 1. At EMERGENCY_SELL_AMOUNT the
   decision corp is the operating corp in the emergency, and slot 3 marks the
   different corp whose shares are sold.
5. **The market.** Agreed; B5. At a column's bottom the price stays, so the
   remaining sale slots repeat the bottom price
   (`g1889/entities/market.pyx:84-97`). A withhold at column 0 moves down, and
   a payout at a row's end moves up.
6. **Order and sales.** The one-hot is B6. PLAYER 19-25 already mark each
   player's sales this SR on every turn; what is missing is a way for the corp
   row to find them, which is the SOLD_THIS_SR relation in B7.
7. **Hex tokens.** Every hex already has its own always-visible row: 52 of
   the 95. CANDIDATE rows are 15 more, visible only at TILE and ER_BONUS,
   each a tile as it would be laid. The cost concern is smaller than feared:
   at 95 rows the quadratic term is about 6% of the forward, so cost grows
   linearly with rows. The failure is that neighbouring hex rows do not
   exchange information (Section 4). Options, in the order I would try them:
   - give the model the engine's graph answers as corp-to-hex relations, the
     form it already uses most: routes (B7) and build distance (C1);
   - make neighbours exchange information directly (C3);
   - take the candidate rows out of the common path (C4).

## 9. Decisions

The user settled these on 2026-11-09.

1. **Human-position starts (A1).** *Decided yes, at 25% of an epoch's games,*
   from a pre-extracted file (9.1). To watch: memorising the positions.
2. **The terminal value (A2).** *Decided soft rank with τ = 0.05,* as a config
   option; RSS keeps its formula.
3. **Playout cap randomization (A3).** *Decided yes, then withdrawn by the
   review:* the estimate the decision rested on was wrong (Section 7, Tier A).
   Reopened as the simulation budget and buffer length in `v1-design.md` §2.3.
4. **The 10-epoch check on v0 before the fresh run.** *Decided no:* go
   straight to v1. Most of v1's changes are needed either way, and v1's own
   train roster is the test. The check stays available later.
5. **Layout 2's scope.** *Decided all of Tier B.*
6. **A matched comparison with v0** (v1 trained on v0's exact config and seed
   with one change, so a strength difference could be credited to it).
   *Decided no:* v1 owes v0 no compatibility.
7. **The v0 replay buffer.** On the cloud box, which is shut down; pull it
   when the box next starts, for Tier C.
8. **No more rescale.** *Decided:* the buffer, the training steps, and the
   temperature windows follow the run's own measurements (9.3).

### 9.1 Human-position starts

- **The file.** A script on the dev box replays the human logs through the
  engine and writes one `.npz`: the raw state rows at decisions, with each
  row's player count, source game, decision index, decisions left, and game
  phase, and a fingerprint of the state layout. Training loads it and refuses
  a fingerprint that does not match. The logs stay out of the repo (they name
  players); the rows carry no names. Layout 2 will change the row if it caches
  new solves there (B2), so the file is rebuilt whenever the layout changes.
- **Which positions.** Only where self-play could stand: a decision the
  policy's flags leave at least one action for, with no pending step that the
  same-president flags forbid. Not after the bank has broken, and not within
  the last 50 decisions of the source game.
  - 50, not 100: 85-91% of human games end in phase D, so the last 100
    decisions hold most of the phase-D positions, which are the ones self-play
    cannot make for itself.
- **Sampling.** Uniform over positions, by player count. Human decisions are
  already spread evenly over the game phases (6.2K-9.6K each).
- **Memorising.** A run to epoch 100 makes about 12,500 starts from about
  45,000 positions, so few are used twice, and each continuation is its own
  game. The exposure is 87 boards, 7 of them 5-player. The check is
  A5's held-out loss, split by how the game started. More scraped games are
  the remedy if it shows.
- **Temperature.** A started game counts moves from its position's index in
  the source game, so the schedule means what it does elsewhere, after a short
  stretch at the initial temperature so continuations part early.

### 9.2 Games per epoch

Moot with A3 withdrawn: every searched decision is still recorded, so 500
games make about 245K rows with a quarter of them started mid-game. With
9.3's settings the number no longer has to be exact.

### 9.3 Settings that follow the run

`cloud/rescale_1889.py` reset three things by hand: the buffer, the training
steps, and the temperature windows. Each becomes a setting in the run's own
units:

| Setting | Meaning | Replaces |
|---|---|---|
| `buffer_epochs` (3) | Sample from the rows of the last N epochs. The buffer records where each epoch's rows begin; `buffer_capacity` stays as the memory cap | A row count tuned to game length |
| `replay_ratio` (1.1) | Steps in an epoch = ratio × the epoch's new rows / batch size | `training_steps_per_epoch` |
| `temp_anneal_start_pct`, `_end_pct`, per player count | Fractions of the last finished epoch's mean game length at that count, sent to workers in `EpochConfig` | Move numbers |

- **The first epoch** has no length to read, so it plays at the initial
  temperature throughout. Its games are near-random anyway.
- **Lengths come from games played from the start,** not from started
  positions, and are saved in the checkpoint so a resume keeps them.
- **The LR schedule** spans `lr_decay_end_epoch × training_steps_per_epoch`
  steps today. With steps varying by epoch it needs an epoch-based span.
- **Epoch boundaries in the buffer** also give the capacity probe its held-out
  epochs directly, where it now reconstructs them from game starts.

## 10. Not checked

- Whether the server's gap is launch overhead or padding (Section 6).
- The value head's fresh-game error at full simulations beyond one game.
- Whether A1-A3 fix the stall. The 10-epoch check is the test.
- Human strength in the scraped games, which is mixed and unknown; they are a
  reference for what play reaches, not a target.
- Extraction of one row from two threads at once, and anything on the GPU.
