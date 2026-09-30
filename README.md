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
  transformer-v2 mixed 3-5 player config.
- `utils_18xx/` - 18xx.games replay parsing, replay analysis, API client, and
  live-play webhook server.
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

Set `model_path` in the training config to select the model. The model declares
its input version; evaluators, training, IPC, and diagnostics use that contract
to select the layout. Query `get_token_dim(layout_version)` and
`get_token_widths(max_players, layout_version)` for dimensions. The default
model and extraction calls without a version remain v2 for checkpoint
compatibility; v2 is retained as the evaluation baseline for v3.

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

The setting is checkpointed with the config and applied by self-play, analysis,
and live replay. Tournaments use one mode for all seats, defaulting to the first
checkpoint's setting, with the same CLI override. On `GameState`, it is a runtime
flag alongside the other engine options, outside the raw int16 array; callers
reconstructing a game for execution supply `v3_behavior` to `from_array` or
`from_buffer`. Rebinding retains the wrapper's mode, and MCTS keeps it across
subtree reuse, including the `acq_same_president` scope flag. Tournament
acquisition scope defaults to the first checkpoint's config and accepts
`--[no-]acq-same-president` as an override.

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

Start a transformer-v2 training run:

```bash
.venv/bin/python -m train --config train_configs/bigger-multi.json
```

Resume the newest checkpoint from the configured checkpoint directory:

```bash
.venv/bin/python -m train --config train_configs/bigger-multi.json --resume latest
```

Watch training metrics:

```bash
.venv/bin/tensorboard --logdir runs
```

[Policy-learning diagnostics](train/POLICY_METRICS.md) describe the automatic
unnoised-prior confidence, search-disagreement, and phase-specific policy-fit
metrics, including their denominators and interpretation.

Play and inspect one analysis game from the latest checkpoint:

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

Compare checkpoints in a small tournament:

```bash
.venv/bin/python -m train.tournament \
  checkpoints/checkpoint_epoch_0100.pt,checkpoints/checkpoint_epoch_0120.pt \
  --games-per-pair 20 \
  --simulations 200
```

Checkpoints trained under different engine rules (e.g. v2 and v3) can play
together. Games default to the most permissive checkpoint rules, and each
checkpoint searches under the rules stored in its own config; the startup
output lists both.

## Live 18xx.games Play

The live server receives webhook notifications, synchronizes the game into
this engine, runs MCTS, and posts the chosen action back. Each game is
downloaded from the 18xx.games API once; after that the server follows the
actions 18xx.games publishes on its MessageBus (the same long-poll updates the
browser uses), recomputes whose turn it is locally, and only re-downloads when
the update stream has a gap. Followed games are saved under
`runtime/game_feeds/` so restarts resume without downloading.
`--no-message-bus` restores the old behavior of downloading the full game on
every turn and after every post.

Live decisions use the checkpoint's `v3_behavior` and acquisition scope.
Checkpoints trained with `"acq_same_president": false` make cross-president
offers, stop planning after posting one, and answer offers made to them with
their own priors. 18xx.games players can also make offers v3 forbids, such as
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
  --host 0.0.0.0 \
  --port 8080 \
  --simulations 400 \
  --model-output
```

When `--base-url` points at `https://18xx.games`, outbound API requests are
throttled by default to one request start every 10 seconds. Override with
`--api-min-interval SECONDS`; local URLs default to no throttling.

Configure the webhook URL for bot `rss-az-1` as:

```text
http://YOUR_HOST:8080/webhook/rss-az-1
```

For local manual testing, the server also supports a loopback-only poke
endpoint:

```bash
curl http://localhost:8080/poke/GAME_ID
```

Webhooks start following a game automatically. Loopback-only endpoints also
manage followed games:

```bash
curl http://localhost:8080/listen                  # followed games and state
curl -X POST http://localhost:8080/listen/GAME_ID  # download once, then follow
curl -X POST 'http://localhost:8080/listen/GAME_ID?resync=1'  # force a re-download
curl -X POST http://localhost:8080/unlisten/GAME_ID
```

Following a game queues any bot whose turn it currently is, and later queues
bots whenever they become an acting player or, in the simultaneous
Acquisition and Closing rounds, whenever another player acts while they are
still acting. Webhooks are optional for followed games; note that 18xx.games
only delivers webhooks to `https` URLs on public addresses. Finished games stop being followed automatically; blacklisted
games are followed but never queued.

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
