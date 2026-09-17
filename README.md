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
against top-level human players. Backward compatibility with old models,
checkpoints, and configs is not required; the older transformer and ResNet
paths are not maintenance targets. See [`AGENTS.md`](AGENTS.md) for development
guidance.

## What Is Here

- `RULES.md` - rules reference for Rolling Stock Stars.
- `core/` - compact Cython game state, static data, action encoding, driver,
  token extraction, relation extraction, and ResNet vector extraction.
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
- `token-data.md`, `resnet-data.md`, `VECTORS.md` - implementation notes for
  model inputs and state/action layout.

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

## Live 18xx.games Play

The live server receives webhook notifications, fetches the game from the
18xx.games API, synchronizes it into this engine, runs MCTS, and posts the
chosen action back.

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
