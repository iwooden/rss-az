# Shikoku 1889 v1 Design

Status: draft for the user's review, dated 2026-11-09. Nothing here is built.
It turns [`v1-review.md`](v1-review.md)'s findings and the decisions in its
Section 9 into a design for a fresh training run: the self-play and training
changes, token layout 2, the v1 model, and the order to build them in.

Paths are relative to the repo, and `path:line` cites the code at 7f7b9fa.
**Tier C:** marks a part that the offline tests on v0's buffer (Section 10)
may change, with what happens under each result. **Open:** marked a decision
for the user (Section 11). **Test:** marks a claim for implementation to
check.

## 1. Scope

**In scope:**

- self-play: starts from human-game positions, the soft-rank terminal value,
  the simulation budget, and temperature as a share of game length (Section 2);
- replay and training: game ids, final net worths, a buffer measured in
  epochs, steps from a replay ratio, and an epoch-based LR schedule (Section 3);
- telemetry (Section 4);
- token layout 2 and its relations (Sections 5 and 6);
- `nn/transformer-1889-v1.py` (Section 7);
- 1889 tournaments, with v0 at epoch 58 as the opponent (Section 8);
- `train_configs/1889-v1.json` and the run (Section 9).

**Out of scope:** build distance as extraction-design §8 specifies it (Tier C
tests a proxy first), auxiliary value targets, a two-stream architecture, and
live play. RSS, `nn/transformer-v2.py`, and `nn/transformer-v3.py` do not
change, and `tests/test_v2_parity.py` passes on its fixture. Every new setting
defaults to today's behaviour, so RSS configs run as they do now.

## 2. Self-play

### 2.1 Starts from human positions

A quarter of each epoch's games start from a decision in one of the 87 scraped
3-5 player games (v1-review §9.1).

**The file.**

- **Builder:** `utils_18xx/g1889_start_positions.py`, run on the dev box
  (built, commit `6445a99`). It replays each game through the
  differential-replay harness (`tests/g1889/replay/harness.py`) and records
  every row the engine rested at. That gives every engine decision, counted
  in `play_game`'s `move_count` units, where loading each Ruby snapshot would
  give only Ruby's stops. It reads the extract cache in
  `tests/g1889/replay/.extract_cache/` and needs `submodules/18xx`, so the
  cloud box cannot run it.
- **Contents:** one `.npz` holding the raw int16 rows, and for each row its
  player count, source game, decision index, decisions left, game phase,
  decision phase, and round; plus the row size, the engine flags, and a hash
  of the state layout.
- **Rows are normalised** before saving: the extraction prologue runs on each
  (`g1889/token_data.pyx:347-354`), so its graph and route caches are fresh.
  At a DIVIDEND decision the revenue is set to the engine's best, as the
  deviation test sets it, since humans may have run a lesser route.
- **Where it lives:** `g1889/data/human_starts.npz`, checked in. It holds
  45,355 rows of 850 slots: 78 MB raw, 1.4 MB compressed. By player count:
  24,261 at 3, 17,764 at 4, 3,330 at 5. By game phase: 5,700 to 9,200 each.
- **Loading:** training refuses a file whose layout hash or row size differs
  from the engine's. Layout 2 adds cache slots to the row (5.9), so the file
  is rebuilt once after step 9.6.

**Which positions.**

- 3-5 players; the bank not yet broken; more than 50 decisions left in the
  source game.
- The driver rests on a decision with at least one policy action under the
  self-play flags.
- **Test:** from every kept position, a uniformly random policy playout
  reaches the end of the game in a debug build with no assertion. This is the
  filter for positions the self-play flags cannot continue from (for example a
  pending train sale between presidents).

**Playing from one.**

- `play_game` (`train/self_play.py:226-237`) gains `start_row` and
  `start_move`. With a row it builds its state as
  `game.scratch_state(game.new_game(...), row)`, which carries the run's
  flags, where `utils_18xx/g1889_deviation.py:291-331` patches `get_game` to do so.
- The main process loads the file once, and the task a worker takes
  (`train/main.py:1246-1249`) carries its row (1.7 KB) and decision index.
  128 workers each loading 78 MB would cost 10 GB.
- The loader and the layout hash move from the builder's module to where the
  pipeline may import them: `train/` cannot import the engine
  (`tests/test_pipeline_boundary.py`), so they go behind the `Game` adapter.
- `human_start_fraction` (0.25) of each player count's games in the epoch's
  schedule get a start (`build_epoch_start_schedule`): the nearest whole
  number, spaced evenly through that count's games, each drawn uniformly and
  with replacement from that count's positions. The draw is a function of the
  seed, the epoch, and the player count, so it leaves the run's other random
  streams as they are, and a replayed epoch draws the same starts.
- **Repeats:** at 500 games an epoch, each count starts 42. Over 300 epochs
  that is 12,500 draws a count: 0.5 a position at 3 players, 0.7 at 4, and 3.8
  at 5, where the file holds only 3,330 positions.
- **Moves count from the position's decision index** in its source game, so
  the temperature schedule (2.4) reads the same in a started game. The first
  `start_explore_moves` (20) decisions play at the initial temperature, so
  continuations of one position part early.

**What a started game changes downstream.**

- Its rows carry a flag in replay (3.1), and the held-out loss reports them
  apart (Section 4).
- Its stats go to their own TensorBoard groups, `self_play_started` and
  `self_play_started_<n>p`, with the number of started games and their mean
  start move, and stay out of the groups for games from the opening. Several
  1889 stats read the final state (`games/g1889_selfplay.py:74-99`), so a
  started game's train counts include the humans' purchases.
- It does not count toward the game lengths that set the temperature windows.

### 2.2 The terminal value

`compute_terminal_values` (`mcts/evaluator.py:117`) gains `rank_tau`. At 0 the
rank term is today's. Above 0 it is the soft rank:

    rank_i = mean over j ≠ i of tanh((nw_i − nw_j) / (rank_tau · mean(nw)))

- **It equals today's rank reward when gaps are large,** since the mean of the
  signs of the gaps is the evenly spaced rank value, and it gives tied players
  the average of their places, as today's does.
- **At τ = 0.05 and a mean net worth of ¥5,000,** a ¥20 gap is worth 0.08 of a
  place and a ¥250 gap 0.76.
- It is zero-sum and within ±1. The margin term and `terminal_blend` (0.75)
  stay.
- **Config:** `terminal_rank_tau`, default 0. Evaluators carry it beside
  `terminal_rank_weight`, set wherever that is set (`train/self_play.py:539`,
  `train/main.py:1153`, and the tools that build evaluators:
  `utils_18xx/g1889_deviation.py`, `train/tournament.py`, `train/analyze_game.py`,
  `train/strategy_data.py`, and `utils_18xx/`). In-tree terminal nodes use
  the same reward through
  `evaluate_terminal` (`mcts/evaluator.py:224`).
- **Test:** tau 0 reproduces today's values exactly on the existing tests.

### 2.3 The search budget

**Playout cap randomization is not adopted.** v1-review Section 7 gives the
correction: with subtree reuse, a search's cost is its budget less what it
inherits, and it inherits about 70% (834 of 1,189 simulations at epoch 58,
`mcts/search.py:600`). A full search after fast ones inherits little, so the
scheme costs what uniform search costs and records a quarter of the rows.

Two settings do what it was meant to do:

- **The simulation ramp.** v0 ramps 800 to 3,200 by epoch 300, RSS's schedule.
  A search costs about 0.3 of its budget in evaluations, so the ramp's end
  would cost nearly three times what a search costs at epoch 58. v1 ramps 800
  to 1,600. The value
  head is short of games, not of search depth, and search overturns the prior
  in only 10.7% of decisions at 1,189.
- **The buffer's length.** `buffer_epochs` (3.2) at 6 would put about 3,000
  games in front of the value head, not 1,800, at no cost in self-play. The
  price is older policy targets. v1 starts at 3 and raises it only if the
  held-out value loss calls for it (Section 11, item 2).

**Tier C** (optional, Section 10): how much a policy target improves with
simulations, measured against a deep reference search on buffer states. It
would replace the judgment above with a number.

### 2.4 Temperature as a share of game length

- **Config:** `temp_anneal_start_pcts` and `temp_anneal_end_pcts`, one per
  player count. When set they replace `temp_anneal_starts` and `_ends`
  (`train/config.py:224-242`).
- **Each epoch,** the main process turns them into move numbers from the last
  finished epoch's mean game length at each player count
  (`_OpeningGameLengths`, `train/main.py`) and sends them in
  `EpochConfig.temp_anneal_windows`, which `_compute_temperature` reads in
  place of the config's windows. Built in sub-step 9.3.
- **The shares are fractions,** 0 to 1, whatever the field names say.
- **The first epoch** has no length, so it plays at the initial temperature
  throughout.
- **Lengths** are over games played from the start, and the checkpoint stores
  them, so a resume keeps its windows.
- **Values:** 30% and 60% at every player count, with the final temperature
  still 0.5. v0's were 17.5% and 35%. The later window keeps play varied
  through the phase 4 to 5 transition, at 40-65% of a game.
- The policy-target temperature stays 1.0 throughout, as in v0.

### 2.5 The value blend

`value_blend_final` falls from 0.7 to 0.3, on the same ramp over epochs
10-300. Targets that lean on the value head's own search reinforce what the
head already believes, and v0's head believes treasury is worthless.

## 3. Replay and training

### 3.1 What a row stores

`ReplayBuffer` (`train/replay_buffer.py`) gains three arrays (built, commit
`0b2aba3`). `GameRecord` carries the flags and the net worths; the buffer
assigns the id as it inserts a game, so ids never decrease in write order:

| Array | Type | Holds |
|---|---|---|
| `_game_ids` | int32 | the game's serial number in the run |
| `_flags` | uint8 | bit 0: the game started from a position |
| `_net_worths` | int32 × max_players | the game's final scores |

- **Game ids** make game boundaries exact. 1889 reconstructs them today from
  each row's state (`games/g1889.py:90-113`), which the capacity probe relies
  on.
- **Net worths** let a saved buffer be re-targeted offline. Today's value
  target is a blend that cannot be undone.
- `load` fills the arrays with "unknown" for a buffer saved without them, so
  v0's buffer and RSS's still load.

### 3.2 A buffer measured in epochs

- **Config:** `buffer_epochs`, default 0 (off: today's behaviour).
- The buffer's metadata records `(epoch, first game id)` for each epoch
  (`begin_epoch`). Sampling draws from the rows of the newest `buffer_epochs`
  epochs that still hold rows. Rows are written in game order, so those are
  the newest contiguous run of rows.
- **A resume into a partly played epoch** (after a graceful shutdown) keeps
  that epoch's earlier games in it, so the epoch holds them plus a full quota.
  Rows from before the record, as in v0's buffer, count as one epoch.
- The `min_buffer_size` check, the priority stats, and step scaling read the
  window, not the whole buffer.
- `buffer_capacity` stays as the memory cap: 2.5M rows, 7.3 GB. If the window
  outgrows it the oldest rows fall off as now, with a warning.
- The capacity probe takes its held-out epochs from the same record.

### 3.3 Steps from a replay ratio

- **Config:** `replay_ratio`, default 0 (off).
- Steps in an epoch = `replay_ratio` × the epoch's rows in the buffer, by the
  record / `batch_size`,
  scaled by the share of `buffer_epochs` the buffer holds so far. That scaling
  is what `_scaled_training_steps` does by fullness today
  (`train/main.py:713-724`): without it, the first epochs' rows would be drawn
  two or more times as often as later ones.
- At 1.1, a row is drawn about 1.1 times in its life, as in v0.

### 3.4 The LR schedule

Today's schedule counts optimizer steps over `lr_decay_end_epoch ×
training_steps_per_epoch` (`train/trainer.py:133-157`). With steps varying by
epoch, v1's reads progress in epochs: (epochs finished + the share of this
epoch's steps done) / `lr_decay_end_epoch`, with the same warmup and cosine.
It applies when `replay_ratio` is set; otherwise the schedule is today's.

As built (sub-step 9.3), the trainer's saved state gains nothing. The main
loop gives the trainer the epoch and its step count (`Trainer.begin_epoch`),
so the rate depends only on the checkpoint's epoch and this epoch's steps,
and a resume is exact by construction. The rate is today's formula at the
step this would be had every epoch this one's steps, so at constant steps the
two schedules agree at every step.

## 4. Telemetry

- **Held-out loss.** Before an epoch trains, score a sample of its new rows
  (32K) with the current weights: value MSE and policy KL overall, by quarter
  of the game, by decision phase, and for started games apart. This is the
  number v0 lacked: its training loss read 0.020 while fresh games scored
  0.206.
- **Gradient norm** before clipping, and the share of steps clipped. The value
  `clip_grad_norm_` returns is discarded today.
- **1889 self-play stats** (`games/g1889_selfplay.py`), per game:
  - the premium paid over face at auction, and player cash entering SR1 (the
    user's note 2);
  - par prices: the mean, and the share at ¥65;
  - treasury at float; stock rounds ending with a parred corp unfloated;
  - shares sold in SR2; station tokens placed;
  - the richest corp's treasury when a 5 is first the next depot train;
  - the OR at which each phase starts;
  - the round in which MF's port tile is laid.
- **As built** (sub-step 9.4): `heldout/*` tags, from `heldout_rows` sampled
  rows of the epoch's own games, with `_opening`, `_q1`-`_q4`, per-phase, and
  `_started` splits and `heldout/value_mse_over_training`;
  `epoch/grad_norm_avg`, `_max`, and `epoch/grad_clipped_share`; and
  `eval_server/*`: real and launched rows, `padding_share`, launches and mean
  real rows at each launched size, and the serve loop's time split between
  gather, build, forward, scatter, and idle. The flatten is inside build.
  **A local 6-worker run padded 28% of launched rows** (launches at 32 carried
  21 real rows on average). If the cloud's `padding_share` is similar, padding
  explains most of the gap below, and finer launch sizes are the fix to test
  in step 9.9's profile.
- **Eval server.** A histogram of real against launched rows per launch, and
  time by stage: gather, flatten, forward, scatter. `EvalServerStats` already
  holds a size histogram that only its mean and maximum reach TensorBoard from
  (`train/main.py:398-401`). This is what explains, or not, the 31% gap
  between the model's rate and the server's.

## 5. Token layout 2

Layout 2 is layout 1 with the changes below. Final slot numbers are assigned
in `token-data-1889.md` at implementation, which
`tests/g1889/test_token_data_doc.py` pins. Divisors follow layout 1's
(¥1000 for treasury-scale money, ¥100 for share prices and revenues) unless
stated. "Next train" means the cheapest train the depot sells now (TRAIN 7).

### 5.1 Rows

The row order is layout 1's, so token indices in relation records do not
change: GLOBAL, PRICE, DIVIDEND, 7 CORP, 7 PRIVATE, 6 TRAIN, 52 HEX, 15
CANDIDATE, then the players.

| Type | Width, layout 1 | Layout 2 (about) |
|---|---|---|
| GLOBAL | 44 | 90 |
| PRICE | 24 | 30 |
| DIVIDEND | 8 | 13 |
| CORP | 51 | 77 |
| PRIVATE | 18 | 18 |
| TRAIN | 13 | 15 |
| HEX | 105 | 105 |
| CANDIDATE | 108 | 109 |
| PLAYER | 38 | 40 |

### 5.2 GLOBAL

| Change | Feature |
|---|---|
| Changed | Slots 39-40 (the auction's pass payouts and the actor's discount) are 0 once the auction ends |
| Added, 2 | The interrupted step, at PRIVATE_PRICE, ER_BONUS, and an OR's PORT: before the corp's turn (PRE), after it (POST). Today the two give identical inputs and continue differently |
| Added, 1 | ORs left in the set after this one /2 |
| Added, 1 | ORs until the bank breaks at the current drain, capped at 10, /10. The drain per OR is the sum over floated corps of route max × (shares outside the IPO) / 10 |
| Added, 2 | The train limit now /4, and in the next phase /4 |
| Added, 40 | Copies left of each supply tile / the largest `TILE_COUNT` |

### 5.3 PRICE and DIVIDEND

| Row | Change | Feature |
|---|---|---|
| PRICE | Added, 6 | At PAR, for each par price: how many of the five shares a float needs the actor can pay for, /5 |
| DIVIDEND | Removed | Slot 2, per-share payout: it is slot 1 / 10 |
| DIVIDEND | Added, 1 | The next train's price /1000; 0 with no room under the limit |
| DIVIDEND | Added, 2 | The shortfall to that price after a withhold, and after a payout (the corp keeps its pool shares' payout), /1000 |
| DIVIDEND | Added, 1 | Room under the train limit |
| DIVIDEND | Added, 2 | Route max with that train bought, and route max after the next rust with no purchase, /100 (5.9) |

### 5.4 CORP

| Change | Feature |
|---|---|
| Replaced | Slot 38, the OR position as a scalar, becomes a one-hot of 7: the corp's rank in the operating order. In an OR it is this OR's order; in an SR it is the order the next OR would take, market stack order included (`market_sort_by_price`). Today's scalar is 0 in an SR and hides the stack order that breaks ties |
| Added, 1 | Is the operating corp. At DISCARD the decision corp is the crowded corp, and nothing marks the operator |
| Changed | Slots 15-19, the price after selling 1-5 shares, are filled at every decision for a parred corp, not only at STOCK and BUY_TRAIN. At a column's bottom the price repeats |
| Added, 3 | The share price after a payout, after a withhold, and after ending an SR sold out, /100 (the user's note 5) |
| Added, 4 | Trains held / the limit; room under it; trains that survive the next rust /4; room under the next phase's limit |
| Added, 3 | Route max after the next rust /100; route max with the next train bought /100; the shortfall from treasury to the next train's price /1000 (5.9) |
| Added, 4 | Shares still to sell before the corp floats /5; their cost /1000; the president can pay it now; the actor can pay it now (note 3) |
| Added, 1 | The president has sold this corp this SR, and so cannot buy it back (note 6) |
| Changed | Slot 49, the actor's largest emergency sale, uses the legal limit (`_emergency_sale_limit`, `g1889/actions.pyx:954-966`); it overstated it in 71 of 373 human cases |
| Added, 1 | On the decision corp in an emergency: the cash still to raise /1000 |
| Changed | Slot 43 (reached cities with a slot) /15, and slot 50 (privates owned) /7: both exceeded 1 |
| Added, 3 | Log-scaled share price, treasury, and route max: ln(1 + ¥/100) / 5 |

### 5.5 TRAIN

| Change | Feature |
|---|---|
| Changed | Slots 10 and 11, the Δ route max of buying one and of trading one in for a D, are computed on the state after the purchase, with the phase change it causes (5.9). Today's ignore the phase and are wrong in 23 of 44 first-5 purchases |
| Changed | Slot 10 is also filled at BUY_TRAIN and DIVIDEND, for each type the depot or the pool offers the decision corp. Today it is filled only at SELECT_TRAIN, which the driver plays itself in 97 of 108 cases |
| Added, 1 | At a purchase from a corp: the seller's Δ route max /100 |
| Added, 1 | Its next depot purchase starts a new phase |

### 5.6 HEX and CANDIDATE

- **HEX:** unchanged.
- **CANDIDATE, added, 1:** Δ route max /100 if the corp also held the next
  train. Today's Δ is 0 on every candidate in 69% of non-forced TILE
  decisions, because it is for the trains held now.
- **CANDIDATE, changed:** slot 107 (other corps' summed Δ centers) /50.

### 5.7 PLAYER

| Change | Feature |
|---|---|
| Removed | Slot 11 (certificates /25, a rescaling of slot 10); slot 26 (a bid on TR, which is never bid on); slot 34 (bankrupt: the game has ended) |
| Added, 1 | Can float a corp alone: cash of at least five shares at the lowest par, ¥325 (note 3) |
| Added, 1 | In a closed-auction BID: this player acts when the auction resumes (`auction_turn`) |
| Added, 1 | Income per OR at current route maxes /1000: dividends on shares held, and private revenue |
| Added, 2 | Log-scaled cash and net worth: ln(1 + ¥/100) / 5 |

### 5.8 PRIVATE

Slot 15 (the port tile is used) is set on MF's row only. No other change.

### 5.9 Engine work for extraction

**Purchases on a scratch state.** "Route max with the next train bought" and
TRAIN 10-11 are computed by copying the row to the scratch state, adding the
train, calling `Phase.on_train_bought` (`g1889/entities/phase.pyx:69-77`), and
solving. "After the next rust" advances the phase with no purchase.

**New caches in the row.** Per corp: route max after the next rust, and route
max with the next train; and one dirty mask for the pair. They are marked
stale wherever `route_dirty` is, and when the depot's next train changes
without a phase change (the last copy of a type sells). The row grows by 15
slots, to 865.

**Narrower invalidation.** `invalidate_connectivity`
(`g1889/entities/hexmap.pyx:46-48`) marks every corp stale on every lay and
token. v1 marks only corps whose walks touch the hex, the argument
`_fill_candidates` already relies on (`g1889/token_data.pyx:785-787`).

**Test:**

- **Purity:** tokens and records with caches as left equal those with every
  cache marked stale, on self-play, human, and random states. This is what
  catches a missed invalidation.
- **Per-choice truth:** each per-choice feature against the state after the
  real action, as the audit did, phase-changing purchases included.
- **Cost:** on human phase-D states, a leaf's extraction averages under
  100 µs and its 99th percentile is under 1 ms. Today's are 72 µs and 547 µs.

## 6. Relations

| Id | Relation | Query → key | Value |
|---|---|---|---|
| 0-24 | Layout 1's | | |
| 25 | SOLD_THIS_SR | player → corp it sold this SR | 1 |
| 26 | SOLD_BY | corp → player | 1 |
| 27 | ROUTE | corp → each hex on its best routes now | 1 |
| 28 | ROUTE_OF | hex → corp | 1 |

- **SOLD_THIS_SR** carries what PLAYER 19-25 hold in a form a corp row can
  attend along (note 6).
- **ROUTE** shows which hexes each corp's revenue runs through: what a token
  would block, and what track is shared. The solver already produces the
  routes (`utils_18xx/g1889_selfplay_log.py`'s `route_intent` reads them).
  **Tier C** (`route_rel`): if it shows no gain on TRACK, TILE, and TOKEN, it
  stays in, as the review's Tier B decided, but the next item changes.
- **The edge bound** rises from 1,844 by 70 for sales and 728 for routes, to
  2,642, so `max_relation_edges` becomes 4,096 and bucketed launches gain a
  fifth K. **Tier C:** if ROUTE were dropped, 1,914 fits in today's 2,048.
- **Build distance. Tier C** (`frontier_rel`) tests a proxy: hex-graph
  distance from each corp's reach, in buckets 1, 2, and 3. A clear gain makes
  the real build distance (extraction-design §8) the first addition after v1
  launches; its ids append after 28, and E is already 4,096. No gain leaves it
  out.
- Relation extraction has no layout version. v0 ignores ids past 24
  (`nn/transformer-1889.py:491`), so it reads layout 2's records unchanged.

## 7. The model

`nn/transformer-1889-v1.py`: `INPUT_LAYOUT_VERSION = 2`, `build_model(config,
spec)`. It is `nn/transformer-1889.py` with the changes below. v0's file
stays, for the tournament's opponent (Section 8).

**Unchanged:** d = 256, 4 heads, 15 layers, ff_mult 3.0; the shared tile
projection; identity and type embeddings; flat relation records; zero-sum
values; the value head; initialization.

**Changed:**

- **Projections** take layout 2's widths.
- **Relations:** 29 dynamic channels and the 10 static ones.
- **Heads read the new raw slots:**
  - `dividend`: the DIVIDEND row's six new slots;
  - `buy_train`: TRAIN 10 and the new phase flag, for all six rows;
  - `par`: the six affordability slots;
  - `stock_corp`: the float slots, the three market neighbours, and the
    president-sold flag;
  - `select_train` and `train_price`: the seller's Δ.

**Tier C decides four more:**

| Test | If it holds | If not |
|---|---|---|
| `cand_head`: the trunk runs on the 80 rows that are not candidates, and TILE and ER_BONUS are scored by a head over each candidate's own slots, the selected hex's final row, and the final rows of the hexes across its exits | v1 adopts it: 16% fewer rows in every state, and relations 21-24 are no longer emitted for layout 2 | v1 keeps v0's arrangement, 95 rows. Splitting launches by row count in the eval server is not planned: it doubles the launch shapes for about 9% |
| `hex_mix`: a module that adds to each hex row a projection of its six neighbours' rows, gated by track, before blocks 0, 5, and 10 | Adopted | Left out |
| `bias_init`: two heads in layers 0-5 start with a bias toward neighbours and track links | Adopted if `hex_mix` is not | Left out |
| `value_w025`: the value loss weighted 0.25 | `value_loss_weight` 0.25 in v1's config | 1.0 |

"Holds" means: better than base on the measure named in Section 10 by more
than twice the gap between base's two seeds, and no worse elsewhere by that
margin. For `cand_head` the bar is parity on TILE and ER_BONUS, not a gain.

**Test** (as v0's 8.1): legal logits finite and the rest masked; dense and
flat inputs agree; a batch equals its states alone; each pointer logit moves
with its own row's raw slots only; zero-sum values; construction checks widths
and blocks.

## 8. Tournaments and v0

A healthier train roster shows that v1 plays more like the humans, not that it
plays better. The measure of strength is games between v1 and v0 at epoch 58.

- **`train/tournament.py` for 1889:** it imports the RSS engine directly today
  (`train/tournament.py:46-57`). It moves onto the `Game` adapter, as search
  and self-play already have.
- **v0 stays playable.** Each evaluator fills its own model's layout
  (`spec.layout_version`, model-design §9), so extraction keeps layout 1
  beside layout 2, as RSS keeps v2's and v3's.
  - Layout 1's tokens must not change. The engine fixes in Section 5 (TRAIN
    10-11, CORP 49, GLOBAL 39-40) apply to layout 2 only.
  - A fixture test pins layout 1's tokens on fixed states, as
    `tests/test_v2_parity.py` pins v2's.
  - Relations need no version (Section 6).
- This is the one place v1 keeps compatibility with v0, and it costs a second
  fill path in `g1889/token_data.pyx`. It stays until v1 has beaten v0; then
  layout 1 and v0's model file can go.

## 9. The run

`train_configs/1889-v1.json` is `1889-v0.json` with these changes:

| Field | v0 | v1 | Section |
|---|---|---|---|
| `model_path` | `nn/transformer-1889.py` | `nn/transformer-1889-v1.py` | 7 |
| `terminal_rank_tau` | (0) | 0.05 | 2.2 |
| `start_positions_path`, `human_start_fraction`, `start_explore_moves` | | the file, 0.25, 20 | 2.1 |
| `mcts_sims_end` | 3,200 | 1,600 | 2.3 |
| `buffer_epochs` | | 3 | 3.2 |
| `buffer_capacity` | 1,000,000 | 2,500,000, as a cap | 3.2 |
| `replay_ratio` | | 1.1 | 3.3 |
| `training_steps_per_epoch` | 1,300 | unused | 3.3 |
| `temp_anneal_start_pcts`, `_end_pcts` | | [0.3, 0.3, 0.3], [0.6, 0.6, 0.6] | 2.4 |
| `temp_anneal_starts`, `_ends` and the policy-target ones | move numbers | removed | 2.4 |
| `value_blend_final` | 0.7 | 0.3 | 2.5 |
| `heldout_rows` | (0) | 32,768 | 4 |
| `value_loss_weight` | (1.0) | **Tier C** | 7 |
| `checkpoint_dir`, `tensorboard_dir` | `*/1889-v0` | `*/1889-v1` | |

The rest stays: 500 games an epoch, 128 workers, bf16 bucketed evaluation,
Muon, c_puct 1.7, the noise and forced-playout settings, and the seed.

**What to watch,** against v1-review Section 2's human figures: the share of
pars at ¥65, the richest corp's treasury when 5s go on sale, withhold share by
phase, trains bought by type, and the held-out value loss by quarter. The
first three should move within 20 epochs if the design works; v0's stall was
set by epoch 10.

## 10. Tier C tests

Each trains v0's shape from scratch on v0's epoch-58 buffer, with the newest
1,000 games held out, and reports held-out policy KL by phase and value MSE.
Two seeds of the unchanged model give the noise floor.

**Prepared and launched by hand** (2026-11-09, `scratchpad/v1_tier_c/`).
Every variant is built and checked on the CPU: the driver reproduces
`train/capacity_probe.py` exactly on the unchanged model, and each variant's
outputs at step 0 equal the unchanged model's. The bundle is on the 5090 pod's
volume, where a 60-step run of five variants passed. One command, from
`/workspace/rss-az`, runs all nine in order and resumes after an interruption:

    scratchpad/v1_tier_c/launch.sh --device cuda --steps 30000 --batch-size 256 --buffer checkpoints/1889-v0/replay_buffer

- **Time, estimated:** nine variants at 30K steps take 6-7 hours on the 5090
  (11-15 steps a second in the smoke run).
- **Results:** `scratchpad/v1_tier_c/runs/summary.json` on the pod.
- **Measured on the CPU with checkpoint 58:** on 600 states with no visible
  candidate, a trunk without the 15 candidate rows gives the same outputs as
  the full one (legal logits within 7.6e-6, values within 2.4e-6), at 0.81-0.83
  of the forward time. So `cand_head`'s 80-row trunk loses nothing in the 92%
  of states with no candidate; the test is about TILE and ER_BONUS only.
- **Measured:** ROUTE adds a median 36 records a state (maximum 178), and
  route revenue equals the cached route max on all 1M rows. The frontier proxy
  adds a median 198, 71% more than v0's own.

| Test | Measure | Decides |
|---|---|---|
| `cand_head` | TILE and ER_BONUS policy KL; forward time | 80 or 95 rows (Section 7) |
| `route_rel` | TRACK, TILE, TOKEN policy KL; value MSE | Whether ROUTE earns E = 4,096 (Section 6) |
| `hex_mix`, `bias_init` | TRACK, TILE, TOKEN policy KL | The module (Section 7) |
| `value_w025` | Value MSE and policy KL over steps | `value_loss_weight` (Section 9) |
| `frontier_rel` | TRACK, TILE, TOKEN policy KL | Whether build distance follows v1 (Section 6) |
| Target quality by simulations (optional; needs search, not training) | KL of a search's visits at 400, 800, and 1,600 simulations from a 6,400 reference | The ramp (2.3) |

**What they cannot show.** The targets are v0's own search, on v0's boards. A
feature that matters only in positions v0 never reaches, or for decisions
v0's search has given up on (DIVIDEND is 97% payout), will show nothing here.
That is why Section 5's train and capital features are not on this list: they
are adopted on the review's evidence, and v1's run is their test.

## 11. Decisions

The user settled these on 2026-11-09.

1. **The simulation ramp** (2.3). *Decided 800 to 1,600.*
2. **The buffer's length** (2.3, 3.2). *Decided 3 epochs to start.* The
   user's concern is that a longer buffer dilutes each epoch's new games, and
   that Muon did better on RSS with a smaller buffer.
   - With `replay_ratio` fixed, a row is still drawn about 1.1 times in its
     life at any length. A longer buffer delays those draws; it does not
     remove them. At 6 epochs the average sampled row is about 3 epochs old,
     against 1.7 at 3.3.
   - That lag costs most early, when the policy moves fastest.
   - `buffer_epochs` can change at any resume, and the 2.5M-row cap keeps
     about 10 epochs of rows whatever the setting, so raising it takes effect
     at once on rows already held.
   - **The rule for revisiting:** if the held-out value loss (Section 4) is
     still several times the training value loss after epoch 20, raise
     `buffer_epochs` to 6 at the next resume.
3. **Keeping layout 1 for the tournament** (Section 8). *Decided yes.*
4. **Where the Tier C runs happen.** *Decided the 5090.* The bundle is on its
   volume, and a 60-step run of five variants passed there on 2026-11-09.
5. **Whether the build waits on Tier C.** *Decided no:* steps 9.1-9.5 start
   now.

## 12. Sub-steps

Each is a commit or a short series, leaves pytest and pyright green, and
leaves `tests/test_v2_parity.py` passing on its fixture.

| # | Scope | Acceptance |
|---|---|---|
| 9.1 | The soft-rank terminal value (2.2) | tau 0 reproduces today's values; at 0.05 the values are zero-sum, within ±1, and match the formula on hand cases, ties included |
| 9.2 | Replay: game ids, flags, net worths, the epoch record, `buffer_epochs` (3.1, 3.2) | v0's and RSS's saved buffers load; a window of N epochs samples only their rows; the capacity probe reads its held-out epochs from the record |
| 9.3 | `replay_ratio`, the epoch-based LR schedule, temperature shares through `EpochConfig`, lengths in the checkpoint (3.3, 3.4, 2.4) | a resume mid-run continues the LR and the windows exactly; with the settings off, RSS's schedule and steps are today's |
| 9.4 | Telemetry (Section 4) | the tags appear after a local 1889 smoke run; the held-out loss on a buffer of random rows matches a direct computation |
| 9.5 | Start positions: the builder, the file, `play_game`'s start, the task's index, the started group (2.1) | every kept position plays out at random; a smoke epoch plays a quarter of its games from positions, and their rows carry the flag |
| 9.6 | Layout 2: the caches and narrower invalidation, tokens, relations 25-28, `token-data-1889.md`, the layout-1 fixture (Sections 5, 6, 8) | 5.9's Tests; layout 1's tokens unchanged on the fixture; the start-position file rebuilt |
| 9.7 | `nn/transformer-1889-v1.py` (Section 7), after Tier C | Section 7's Tests |
| 9.8 | 1889 tournaments (Section 8) | v0 at epoch 58 against itself scores even within noise over 3-5 player games |
| 9.9 | The config, a local smoke run with a resume, the cloud profile, the launch | as model-design §8's 8.4 and 8.5, on layout 2 |

9.1-9.5 run on layout 1 and v0's model, so each can be smoke-tested locally
before layout 2 exists.
