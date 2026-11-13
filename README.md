# Rolling Stock Stars AlphaZero

This repository contains a high-performance Rolling Stock Stars engine plus an
AlphaZero-style self-play, search, and training stack. It is primarily a
research/workbench repo for building models that can play Rolling Stock Stars,
inspect their decisions, and connect trained models to 18xx.games-compatible
live play tooling.

Rolling Stock Stars is its own game. The `utils_18xx/` code exists because RSS
is playable on 18xx.games and that ecosystem is useful for replays and live
integration, but the source of truth for RSS rules in this repo is
[`RULES.md`](RULES.md).

The current practical model path is the transformer v2 architecture
(`nn/transformer-v2.py`), usually through `train_configs/bigger-multi.json`.
This branch develops a successor to that model using lessons from its games
against top-level human players. Backward compatibility with retired model
APIs, checkpoints, and configs is not required. See [`AGENTS.md`](AGENTS.md)
for development guidance.

## What Is Here

- `RULES.md` - rules reference for Rolling Stock Stars.
- `core/` - compact Cython game state, static data, action encoding, driver,
  token extraction, and relation extraction.
- `entities/` - stateless handles for players, corporations, companies, market,
  deck, and turn state.
- `phases/` - phase implementations for investment, bidding, acquisitions,
  dividends, IPO/PAR, issuing, closing, income, and wrap-up.
- `nn/` - PyTorch model definitions and model factory. Current practical work
  uses transformer v2 via `model_path`.
- `mcts/` - batched Monte Carlo tree search, neural network evaluators, and
  Cython search helpers.
- `train/` - self-play workers, replay buffer, trainer, checkpointing,
  TensorBoard logging, analysis-game rendering, and tournament tools.
- `train_configs/` - JSON training configs. `bigger-multi.json` is the current
  transformer-v2 mixed 3-5 player config; `1889-v0.json` is the first Shikoku 1889
  run, and `1889-v0-smoke.json` its local smoke test (run with `--no-compile`).
  `1889-v1.json` and `1889-v1-smoke.json` are the second run's, on
  `nn/transformer-1889-v1.py` ([`1889/v1-design.md`](1889/v1-design.md)), and
  `1889-v2.json` and `1889-v2-smoke.json` the third's, on
  `nn/transformer-1889-v2.py` ([`1889/v2-design.md`](1889/v2-design.md)).
  [`1889/model-design.md`](1889/model-design.md) Section 11 is the runs' runbook:
  11.1 for v1's launch, with `cloud/setup.sh` and `cloud/profile_1889.py`, 11.2
  for v0's run, with `cloud/rescale_1889.py` and `train/capacity_probe.py`, and
  11.3 for v2's launch.
- `utils_18xx/` - 18xx.games replay parsing, replay analysis, API client, and
  live-play server.
- `tests/` - phase tests, engine invariants, model contract tests, MCTS tests,
  training tests, and 18xx compatibility checks.
- [`token-data.md`](token-data.md) - transformer v2 input layout specification.
- [`token-data-v3.md`](token-data-v3.md) - transformer v3 input layout specification.
- [`VECTORS.md`](VECTORS.md) - shared raw state and action layout.

## Model Input Layouts

Each model version has a standalone token specification. Update the matching
document when changing a model's layout; each describes its complete current
layout rather than differences from another version.

| Model implementation (`model_path`) | Input `layout_version` | Token specification |
| --- | --- | --- |
| `nn/transformer-v2.py` | 2 | [V2 token data](token-data.md) |
| `nn/transformer-v3.py` | 3 | [V3 token data](token-data-v3.md) |
| `nn/transformer-1889.py` | 1 (game `1889`) | [1889 token data](token-data-1889.md) |
| `nn/transformer-1889-v1.py` | 2 (game `1889`) | [1889 token data](token-data-1889.md), "Layout 2" |
| `nn/transformer-1889-v2.py` | 3 (game `1889`), packed | [1889 token data](token-data-1889.md), "Layout 3" |

Set `model_path` in the training config to select the model; there is no
default. Every model module declares its input version as
`INPUT_LAYOUT_VERSION`; evaluators, training, IPC, and diagnostics use that
contract to select the layout. Every extraction call names its layout: query
`get_token_dim(layout_version)` and
`get_token_widths(max_players, layout_version)` for dimensions. A packed
layout (1889's layout 3) is one vector a state with no relations: its module
also defines `packed_input_size`, and the game's `input_size` and
`fill_inputs` serve it. V2 is retained as the evaluation baseline for v3.

V3 enables relation input mixing before type embeddings by default. For an
ablation, set `"relation_input_mixing": false` in the training JSON or pass
`--no-relation-input-mixing` (including when resuming). Use
`--relation-input-mixing` to enable it again. The setting is checkpointed;
mixing weights are retained in either mode so the same checkpoint can be used
for both runs. Disabling mixing skips its computation while retaining relation
attention biases and type embeddings. This option does not affect v2.

V3 can also make value outputs zero-sum, matching the terminal targets. With
`"zero_sum_values": true` (set in `train_configs/v3-09-20-26.json`), the value
head subtracts the mean over the state's real players after tanh, so outputs
may exceed ±1 (bound ±2(n-1)/n). `--zero-sum-values` and
`--no-zero-sum-values` override it, including when resuming. The setting is
checkpointed and adds no weights; checkpoints saved without it load with it
off. This option does not affect v2.

`core/token_data.pyx/.pxd` is a thin dispatcher. Each model owns its extractor
and constants in `core/token_data_v2.pyx/.pxd` or `core/token_data_v3.pyx/.pxd`;
feature changes belong in that version's files. V2 was restored from `v2-final`
and retains nominal dividend/pending movement; v3 uses resolved market movement.

`GameState` flags control engine behavior independently of model inputs. Both
models read the same raw state, currently use the same token order and relation
planes, and play under the same chosen rules. Input layout selection does not
change the engine's two-round-trip INVEST cap.

Set `"v3_behavior": true` in the training config (or `--v3-behavior`) to enable
v3 engine behavior; the default is `false` for legacy behavior. This is a
general engine switch, independent of `model_path`. Its first distinction is
trade-history lifetime: legacy clears counters when INVEST ends, while v3
retains them through the rest of the turn and clears them on entry to the next
turn's INVEST. Returning from BID to the same INVEST does not clear counters.
Use `--no-v3-behavior` to select legacy behavior explicitly.

To train cross-player acquisition offers, also set `"acq_same_president": false`
or pass `--no-acq-same-president`. The default remains `true`. With v3 behavior,
each new offer must exceed that player's highest rejected price for the company
in the current acquisition phase. History is tracked in all modes, but legacy
mode permits repeated or lower offers for 18xx.games replay. V3 company tokens
include the active player's rejection threshold, including when that player is
responding to someone else's offer. V3 also limits each player to one rejected
cross-president offer per acquisition phase. Their own companies and FI remain
available after the cap, and FI intervention declines do not count. V3 player
tokens expose each player's rejection count divided by the cap. Both histories
reset at acquisition phase exit; legacy replay remains uncapped. V2 inputs
remain unchanged.

Self-play TensorBoard groups (`self_play_aggregate` and each player-count group)
report `acq_decisions_per_phase`, `acq_offers_per_phase`,
`acq_offer_acceptance_rate`, and `acq_cap_hits_per_phase`. These count actual
played decisions, not search simulations. A phase is counted when self-play
makes an acquisition decision; fully automated phases are excluded. Offer
statistics exclude FI interventions. Cap hits count players reaching the
rejection cap once per phase, including in uncapped legacy mode.
`acq_offer_accept_prior_mean` averages the seller's unnoised root prior on
accept, which the proposer's search also saw. Rejections split into
`acq_rejections_expected_fraction` (accept prior below 0.2) and
`acq_rejections_surprise_fraction` (above 0.5). An offer is a top choice when
each searched proposer decision leading to it (corp, company, price) was the
most-visited action; `acq_offers_top_choice_fraction` and
`acq_top_choice_acceptance_rate` separate those offers from offers that
self-play's temperature sampling picked.

The setting is checkpointed with the config and applied by self-play, analysis,
and live replay. Tournament games default to the most permissive checkpoint's
setting, with the same CLI override, and each checkpoint searches under its own
(see the tournament section). On `GameState`, it is a runtime
flag alongside the other engine options, outside the raw int16 array; callers
reconstructing a game for execution supply `v3_behavior` to `from_array` or
`from_buffer`. Rebinding retains the wrapper's mode, and MCTS keeps it across
subtree reuse, including the `acq_same_president` scope flag. Tournament
acquisition scope defaults to cross-president offers if any checkpoint was
trained with them and accepts `--[no-]acq-same-president` as an override.

18xx.games does not enforce v3's acquisition offer limits, so its replay
(`utils_18xx`) clears the `GameState.acq_offer_limits` runtime flag while
keeping the rest of v3 behavior. Rejections are still recorded, and live
decisions and analysis continuations restore the limits.

Replay stores raw counters, so newly generated replay can supply either layout.
Older replay rows saved after INVEST already lost that turn's trade counters;
the persistent history features cannot be recovered from those rows.

## Setup

Use a virtual environment. The installer handles PyTorch wheels plus the
NumPy/Captum dependency wrinkle, then builds the Cython extensions.

```bash
python3 -m venv .venv
source .venv/bin/activate
./install.sh cpu
```

For GPU installs, use one of:

```bash
./install.sh cuda
./install.sh rocm
```

If dependencies are already installed and you only need to rebuild compiled
extensions:

```bash
.venv/bin/python setup.py build_ext --inplace
```

Run commands from the repo root. The Cython extensions must be built before
importing most engine modules.

## Common Commands

Run a focused smoke test:

```bash
.venv/bin/pytest tests/test_random_game.py tests/test_mcts.py -q
```

Run the phase test suite:

```bash
.venv/bin/pytest tests/phases -q
```

Replay every scraped game on disk, where the default suite replays a few smoke
games. RSS and 1889 keep their game files in gitignored `data/` directories, so
the games aren't in the repo, and the 1889 tests skip without them:

```bash
RSS_AZ_REPLAY_FULL=1 .venv/bin/pytest tests/games_18xx/test_replay.py -q
RSS_AZ_REPLAY_FULL=1 .venv/bin/pytest tests/g1889/games -q
```

Start a transformer-v2 training run:

```bash
.venv/bin/python -m train --config train_configs/bigger-multi.json
```

Resume the newest checkpoint from the configured checkpoint directory:

```bash
.venv/bin/python -m train --config train_configs/bigger-multi.json --resume latest
```

A 1889 run launches the same way, on the cloud box by the runbook above
(`train_configs/1889-v2.json`, the next run's), or here as a smoke test on the
local GPU:

```bash
.venv/bin/python -m train --config train_configs/1889-v2-smoke.json --no-compile
```

Watch training metrics:

```bash
.venv/bin/tensorboard --logdir runs
```

[Policy-learning diagnostics](train/POLICY_METRICS.md) describe the automatic
unnoised-prior confidence, search-disagreement, and phase-specific policy-fit
metrics, including their denominators and interpretation.

Play and inspect one analysis game from the latest RSS checkpoint (a 1889
checkpoint's game is `utils_18xx.g1889_selfplay_log`'s, below):

```bash
.venv/bin/python -m train.analyze_game latest \
  --checkpoint-dir checkpoints \
  --num-players 3 \
  --simulations 200 \
  --output game_log.md
```

Run the same analyzer with a fresh untrained model:

```bash
.venv/bin/python -m train.analyze_game new --num-players 3 --simulations 50
```

Analyze an 18xx.games replay JSON:

```bash
.venv/bin/python -m utils_18xx.analyze_replay game.json latest \
  --checkpoint-dir checkpoints \
  --output replay.html
```

Continue an unfinished 18xx.games game from its exported JSON with AI players:

```bash
.venv/bin/python -m train.analyze_game latest \
  --checkpoint-dir checkpoints \
  --18xx-game-json game.json \
  --simulations 800 \
  --output continuation.md
```

The player count and display names come from the game JSON. The
`--18xx-game-json` and `--18xx-seed` starting modes are mutually exclusive.

Write a 1889 self-play game from a checkpoint as an 18xx.games hotseat log, to
step through its map and tile lays with "Import hotseat game" on 18xx.games or a
local instance. `--verify` checks it against `submodules/18xx`, which the site may
have moved past:

```bash
.venv/bin/python -m utils_18xx.g1889_selfplay_log latest \
  --checkpoint-dir checkpoints/1889-v0 \
  --num-players 4 --seed 1 \
  --output game.json --verify
```

It plays with `train.self_play.play_game` at the self-play settings of the
checkpoint's epoch, in fp32 uncompiled, keeping the root noise but always taking the
most visited move, with no forced playouts, as `train/analyze_game.py` does. Each
searched decision adds a chat line with the root value and the top choices' priors
and visit shares (`--no-eval` omits them). Playing saves `game.played.json` first; `--played game.played.json
--output game.json` converts it again without replaying. Ruby and
`submodules/18xx` are needed here, not in training.

Test whether financing a 1889 train pays (`1889/model-design.md` §7.3): draw
DIVIDEND positions where only withholding buys the next 5 or 6 from a saved
replay buffer (copied off any slow mount first), or with `--human` instead of
`--buffer` from the scraped human games, play each out with and without the
withhold-and-buy, and report the paired difference. `play` resumes from its
results file:

```bash
.venv/bin/python -m utils_18xx.g1889_deviation positions --buffer checkpoints/1889-v0/replay_buffer \
  --out positions.json --per-cell 5
.venv/bin/python -m utils_18xx.g1889_deviation play --checkpoint checkpoints/1889-v0/checkpoint_epoch_0055.pt \
  --positions positions.json --results results.jsonl --workers 3
.venv/bin/python -m utils_18xx.g1889_deviation report --positions positions.json --results results.jsonl
```

Compare checkpoints of one game (RSS or 1889) in a small tournament:

```bash
.venv/bin/python -m train.tournament \
  checkpoints/checkpoint_epoch_0100.pt,checkpoints/checkpoint_epoch_0120.pt \
  --games-per-pair 20 \
  --simulations 200

# 1889: v2 against v1 at 4 players, sampling each game's first 20 decisions
.venv/bin/python -m train.tournament \
  checkpoints/1889-v2/checkpoint_epoch_0100.pt,checkpoints/1889-v1/checkpoint_epoch_0100.pt \
  --num-players 4 --simulations 1600 --search-batch-size 8 --sample-moves 20 --games-per-pair 60 \
  --workers 8
```

Append `@c_puct=VALUE` to a checkpoint path to override its c_puct; the same
checkpoint can appear twice to compare settings
(`cp.pt@c_puct=1.2,cp.pt@c_puct=1.7`). Each checkpoint searches with its own
model, input layout, and terminal value settings, in fp32, uncompiled. Games
run one at a time, or with `--workers K` in K processes that each load every
checkpoint (a seed then gives the same games for any K > 1, though not the
ones one process plays). Moves are the most-visited action after a search with
the checkpoint's root noise; `--sample-moves N` samples each game's first N
decisions from the visits, so near-deterministic models still play different
games.

Seats count from the player who acts first (`P0`; 1889's seed picks who that
is). A 1889 schedule plays in seat-balanced rounds: with fewer models than
seats, every arrangement of the most even splits (2 models: 6 games at 3 and 4
players, 20 at 5), otherwise every rotation of every group of models in both
directions; `--games-per-pair` rounds up to whole rounds. RSS keeps its
original schedule. Besides per-place counts, the report gives each pair's
pairwise score (the share of seat pairs in which the first model's seat
finished ahead, ties half; 0.5 is even) with its standard error over games, and
games and mean score by seat.

RSS checkpoints trained under different engine rules (e.g. v2 and v3) can play
together. Games default to the most permissive checkpoint rules, and each
checkpoint searches under the rules stored in its own config; the startup
output lists both. 1889 checkpoints must share their engine options
(same-president sale flags and the auction pass cap), and the games use them.

## Live 18xx.games Play

The live server follows games, synchronizes them into this engine, runs MCTS
on the bots' turns, and posts the chosen actions back. Each followed game is
downloaded from the 18xx.games API once; after that the server follows the
actions 18xx.games publishes on its MessageBus (the same long-poll updates the
browser uses), recomputes whose turn it is locally, and only re-downloads when
the update stream has a gap. Followed games are saved under
`runtime/game_feeds/` so restarts resume following them without downloading.

Live decisions use the checkpoint's `v3_behavior`, acquisition scope, and ACQ
price cap (`max_acq_price_actions`; `--max-acq-price-actions` overrides it).
Checkpoints trained with `"acq_same_president": false` make cross-president
offers, stop planning after posting one, and answer offers made to them with
their own priors. While any acquisition offer is open, bots answer offers made
to them but otherwise wait instead of offering or passing: 18xx.games ends
Acquisition once every player has passed, dropping unanswered offers, so an
unpassed bot keeps the round open for the responder. 18xx.games players can
also make offers v3 forbids, such as
offers below the high price. V3 checkpoints never trained on those, and their
priors ignore the price, so search answers them from neutral accept/reject
priors. Same-president checkpoints auto-reject cross-president offers unless
`--allow-cross-president-offers` is set, which lets search answer them from
neutral priors.

Create a private runtime directory. It is gitignored.

`runtime/models.json` maps player counts to checkpoints. A mixed-player
transformer checkpoint can serve 3-5 player games:

```json
{
  "3-5": "latest"
}
```

`runtime/auth.json` maps bot names to 18xx.games session tokens:

```json
{
  "rss-az-1": {
    "token": "YOUR_18XX_SESSION_TOKEN"
  }
}
```

Start the live server:

```bash
.venv/bin/python -m utils_18xx.live \
  --runtime-dir runtime \
  --checkpoint-dir checkpoints \
  --base-url http://localhost:9292 \
  --api-min-interval 0 \
  --host 127.0.0.1 \
  --port 8080 \
  --simulations 400 \
  --model-output
```

When `--base-url` points at `https://18xx.games`, outbound API requests are
throttled by default to one request start every 10 seconds. Override with
`--api-min-interval SECONDS`; local URLs default to no throttling.

Follow games with the loopback-only listen endpoints. Starting a game
publishes nothing on its MessageBus channel, so call `listen` after the game
has started:

```bash
curl http://localhost:8080/listen                  # followed games and state
curl -X POST http://localhost:8080/listen/GAME_ID  # download once, then follow
curl -X POST 'http://localhost:8080/listen/GAME_ID?resync=1'  # force a re-download
curl -X POST http://localhost:8080/unlisten/GAME_ID
```

Following a game queues any bot whose turn it currently is, and later queues
bots whenever they become an acting player or, in the simultaneous
Acquisition and Closing rounds, whenever another player acts while they are
still acting. Finished games stop being followed automatically; blacklisted
games are followed but never queued.

If another player moves or undoes (undos are new actions) while a bot is
searching or posting in a followed game, the bot drops its plan and re-plans
from the new state, or stops if it is no longer acting. Games the bot plays
without following them (`poke`) have no such check.

For manual testing, the loopback-only poke endpoint queues every configured
bot for a game, followed or not:

```bash
curl http://localhost:8080/poke/GAME_ID
```

The loopback-only eval endpoint can evaluate either a live game or an exported
18xx.games JSON file without posting an action. File evaluations read a direct
child of `/tmp`:

```bash
curl http://localhost:8080/eval/GAME_ID
curl http://localhost:8080/eval/file/game.json
```

The existing `player`, `player_id`, and `player_index` query parameters can
also be used with the file route.

## Notes

- Checkpoints are written to `checkpoints/` and TensorBoard logs to `runs/` by
  default. Both are gitignored.
- Training is compute-heavy. The default current config is intended for serious
  self-play runs, not a quick laptop demo.
- The engine state supports 2-6 players, but model/search/training paths are
  scoped to 3-5 players.
- Replay examples store compact int16 game states plus dense unified policy
  masks/targets. Model-specific inputs are materialized at evaluation/training
  time.
