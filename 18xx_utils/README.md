# 18xx_utils — Human vs AI Play Server

Play Rolling Stock Stars against trained AlphaZero checkpoints using the
18xx.games web UI as the frontend.

## Architecture

```
Browser (18xx.games, localhost:9292)
    │  POST /api/ai-move  { game_data: {...} }
    │  ← 200 OK           { actions: [...] }
Python API Server (localhost:5050)
    ├── server.py         Flask API, MCTS search, checkpoint loading
    ├── game_session.py   Replay game_data → Cython GameState
    └── action_mapper.py  Engine action_idx → 18xx intent dict
```

The frontend creates a 3-player hotseat game. One player is "Human"
(randomly assigned to slot 0, 1, or 2), the others are "AZ 1" / "AZ 2".
When an AI player is acting, the frontend's AI bridge module POSTs the
game state to the Python server, which runs MCTS and returns action
intents that the bridge translates into proper 18xx Engine::Action objects.

## Usage

```bash
# Terminal 1: Start the AI server
.venv/bin/python -m 18xx_utils.server latest --simulations 400

# Terminal 2: Start the 18xx frontend (Docker)
cd submodules/18xx && make dev_up_b
# → http://localhost:9292
```

### Server CLI args

Same flags as `train/analyze_game.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `checkpoint` | (required) | Path or `"latest"` |
| `--checkpoint-dir` | `checkpoints` | Where to find `latest` |
| `--device` | auto | `cpu`, `cuda`, etc. |
| `--simulations` | 400 | MCTS simulations per move |
| `--search-batch-size` | 1 | Leaves per MCTS batch |
| `--c-puct` | from checkpoint | Exploration constant |
| `--dirichlet-epsilon` | from checkpoint | Root noise epsilon |
| `--no-dirichlet-noise` | — | Disable root noise |
| `--terminal-blend` | from checkpoint | Rank vs margin weight |
| `--port` | 5050 | Server port |
| `--host` | 127.0.0.1 | Server bind address |

The server hot-reloads checkpoints — if a newer one appears in the
checkpoint directory, it's loaded on the next request.

## Module details

### `action_mapper.py`

Reverse maps engine action indices to simplified intent dicts using
`decode_action_py()`. Covers all phases: INVEST (bid/buy/sell/pass),
BID (raise/leave), IPO, PAR, DIVIDENDS, ISSUE, ACQUISITION, CLOSING.

### `game_session.py`

Synchronizes 18xx `game_data` JSON with a Cython `GameState` by replaying
the full action history from scratch on each call. Uses the same
forward-mapping functions from `tests/18xx_games/action_parser.py` that
the replay harness uses (battle-tested against 143 real games). The Ruby
extractor (`tests/18xx_games/extract_states.rb`) runs once per game to
get deck order; subsequent replays are pure Cython (~ms).

### `server.py`

Flask API with `POST /api/ai-move`. Handles:

- **Simple phases** (INVEST, BID, DIVIDENDS, ISSUE): single MCTS search
- **IPO + PAR**: two sequential MCTS searches, combined into one `par` intent
- **ACQ / CLOSING sequences**: loops MCTS until phase ends or human acts
- **Phase mismatch**: when the 18xx frontend is in ACQ/CLOSING but our
  engine has auto-advanced past it (our engine doesn't support cross-president
  acquisitions), returns pass actions so the frontend can catch up

## 18xx submodule changes (branch `rss-az`)

All changes are on the `rss-az` branch of `submodules/18xx`
(fork: `github.com/iwooden/18xx`).

### `api.rb`
- Added `http://localhost:5050` to CSP `connect-src` so the frontend
  can reach the Python server.

### `assets/app/view/welcome.rb`
- Replaced the default 18xx.games homepage with a seed input dialog
  for Rolling Stock Stars.
- Randomizes which player slot (0–2) is the human.
- Names players "Human", "AZ 1", "AZ 2".
- Stores `human_player_index` in `game_data.settings`.

### `assets/app/view/home.rb`
- Stripped down to just render the Welcome component with
  "Rolling Stock Stars" as the page title.

### `assets/app/view/game_page.rb`
- Includes `Game::AiBridge` module.
- Calls `maybe_trigger_ai_move` in both `insert` and `postpatch` hooks.

### `assets/app/view/game/ai_bridge.rb` (new)
- Detects when an AI player is acting in a hotseat game.
- POSTs game_data to the Python server.
- Translates intent dicts into proper `Engine::Action` hashes
  (`action_from_h`), resolving share IDs and entity references
  from the live game state.
- Applies actions with a 600ms delay between moves.
- Guards against duplicate requests via `@ai_pending` flag.
