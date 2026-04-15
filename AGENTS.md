# AGENTS.md

Concise injected context for future Hermes/Codex sessions working in this repo.

## Project in one screen

- This repo is a high-performance Cython engine plus AlphaZero-style search/training stack for **Rolling Stock Stars**.
- **Rolling Stock Stars is not Rolling Stock.** If behavior or rules matter, read `RULES.md` first.
- The core game state is a single contiguous compact `int16` array wrapped by `GameState`.
- The NN does **not** read state directly. The live engine→NN bridge is `core/token_data.pyx:get_token_data(...)`, which fills token buffers for `nn/transformer.py`.
- Search/eval/self-play/trainer are now on the sparse/token contract. Do not assume the old visible/hidden float32 + rotated-state + residual-MLP architecture is still current.

## Source of truth order

When docs disagree, trust these in roughly this order:

1. `RULES.md` for game rules.
2. Current code in `core/`, `entities/`, `phases/`, `nn/`, `mcts/`, `train/`.
3. Runtime-verified behavior (build/import/tests/smokes) over prose.
4. Obsidian cartography notes if present:
   - `/mnt/c/Users/Isaac/Documents/obsidian/Archivum Cogitata/Hermes/rss-az-cython2/`
   - start with `00 Index.md`, `01 Source of Truth.md`, `03 Architecture.md`, `04 Build, Test, and Dev Loop.md`.
5. `CLAUDE.md`, `VECTORS.md`, `README.md`, `transformers.md` only after checking them against code.

Important: several repo docs are stale in specific ways. In particular, old references to float32 visible/hidden state, token width `63`, stub legal enumeration, or pre-refactor MCTS/training status are not current.

## Current architecture snapshot

- `core/state.pyx`: compact int16 layout. Use `core.state.get_layout(num_players)` from Python for exact offsets/sizes.
- `entities/*`: stateless handles over the state array; they own coherence/invalidation semantics.
- `core/actions.pyx`: live phase-local action encoding + sparse legal enumeration.
- `core/driver.pyx`: stateless dispatch + auto-chaining through automated phases and forced actions; step-mode helpers also exist.
- `core/token_data.pyx`: live token extraction bridge.
- `nn/transformer.py`: active model; token-based, dense policy head over `MAX_ACTION_SIZE`, canonical-order values.
- `mcts/*`: live sparse/token search stack.
- `train/eval_server.py`, `train/self_play.py`, `train/replay_buffer.py`, `train/trainer.py`, `train/main.py`: live sparse/token path.
- `train/analyze_game.py`: current checkpoint-analysis/debugging entry point; if search/evaluator return shapes change, re-check it explicitly.

## Read order by task

- Rules / phase behavior:
  - `RULES.md`
  - relevant `phases/*.pyx`
  - `core/driver.pyx`
  - relevant tests in `tests/phases/`
- State/layout/invariants:
  - `core/state.pyx`
  - relevant `entities/*.pyx`
  - `core/token_data.pyx` if NN-visible features are affected
- Action-space changes:
  - `core/data.pxd`
  - `core/actions.pyx`
  - relevant phase module
  - `nn/transformer.py` if action slicing/head assumptions change
- Search/eval changes:
  - `mcts/mcts_core.pyx`
  - `mcts/node.py`
  - `mcts/evaluator.py`
  - `mcts/search.py`
  - `train/eval_server.py`
- Self-play / trainer changes:
  - `train/self_play.py`
  - `train/replay_buffer.py`
  - `train/trainer.py`
  - `train/main.py`

## Build / test loop

- Use `.venv/bin/python`, not bare `python`.
- Run commands from the repo root. If you need explicit imports from ad hoc shell commands, set `PYTHONPATH` to the current repo/worktree root.
- Build extensions before running code that imports compiled modules:

```bash
.venv/bin/python setup.py build_ext --inplace
```

- If you changed `.pxd` files or low-level signatures/layout, clean first:

```bash
.venv/bin/python setup.py clean
.venv/bin/python setup.py build_ext --inplace
```

- Useful targeted tests/smokes:

```bash
.venv/bin/pytest tests/phases -q
.venv/bin/pytest tests/test_mcts.py -q
.venv/bin/pytest tests/test_driver_step_mode.py -q
.venv/bin/pytest tests/test_self_play.py -q
```

## Working rules

- Do not trust stale prose over live code.
- Do not hardcode action counts or layout numbers; query helpers/constants.
- Do not bypass entity handles with raw state-array writes unless you are deliberately working in tightly scoped test/setup code.
- For replay/training/search work, assume the sparse legal-list contract is the default until the code proves otherwise.
- If you need more context than this file can carry, check the Obsidian notes before doing a broad rediscovery pass.

## Environment / workflow notes

- `setup.py` is the build entry point; there is no `pyproject.toml`.
- `pyright` is the system binary, not `.venv/bin/pyright`.
- The repo uses `bd` / beads for issue tracking.
- Use `scratchpad/` for temporary investigation scripts.
