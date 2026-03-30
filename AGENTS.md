# AGENTS.md

Short operational guide for future Codex sessions. This file is intentionally narrower than `CLAUDE.md`: it focuses on the context you usually need before editing, debugging, or verifying work.

## Project In One Screen

- This repo is a high-performance Cython engine plus AlphaZero-style training stack for **Rolling Stock Stars**.
- **Rolling Stock Stars is not Rolling Stock.** If a task touches rules or game behavior, read `RULES.md` first.
- The central data structure is a single contiguous `float32` game-state array. Most core logic is pointer arithmetic over that array, not Python objects.
- The state is split into **visible** NN input and **hidden** bookkeeping. The NN sees the active player rotated into slot 0; hidden state remains canonical/private.
- The engine supports 2-6 players, but the current search / NN / training defaults are centered on **3-player** play.

## Source Of Truth Order

When docs disagree, trust these in roughly this order:

1. `RULES.md` for game rules.
2. `VECTORS.md` plus `core/state.pyx` for state/action layout details.
3. Current source code (`core/`, `phases/`, `mcts/`, `train/`, `nn/`) for behavior and defaults.
4. `CLAUDE.md` for repo-specific workflow notes.
5. `README.md` for high-level orientation only.

## Repo Map

- `core/`
  - `state.pyx`: layout computation, visible/hidden split, rotation-related offsets.
  - `actions.pyx`: action-space layout; use helpers instead of hardcoding counts.
  - `driver.pyx`: stateless `apply_action(...)` loop, phase dispatch, forced-action chaining.
  - `data.pyx`: enums, constants, divisors, market data, company/corp metadata.
- `entities/`
  - Handles over state-array regions (`PLAYERS`, `CORPS`, `TURN`, `FI`, etc.).
  - Hot-path access lives here; IDs are plain ints, handles are singleton objects.
- `phases/`
  - Per-phase rules and transitions: invest, bid, acquisition, closing, dividends, issue, ipo, wrap_up, income, end_card.
- `mcts/`
  - `search.py`: PUCT search, batched leaf eval, leaf locking, subtree reuse, A0GB target extraction.
  - `node.py`: tree structure.
  - `evaluator.py`: state rotation, NN eval, terminal values.
  - `mcts_core.pyx`: Cython hot helpers and worker/server signaling primitives.
- `nn/`
  - `model_3p.py`: current residual MLP with phase-specific policy heads.
  - `__init__.py`: dynamic model factory via `--model-path`.
- `train/`
  - `config.py`: all training + MCTS defaults and validation.
  - `main.py`: CLI / orchestration.
  - `self_play.py`: game generation and worker entry points.
  - `eval_server.py`: shared-memory eval servers.
  - `trainer.py`, `replay_buffer.py`, `checkpoint.py`, `logging.py`: training loop plumbing.
- `tests/`
  - `tests/phases/`: rules / phase coverage.
  - `tests/test_mcts.py`, `tests/test_training.py`, `tests/test_state_layout.py`: search, training, layout.
  - `tests/18xx_games/`: replay tests against 18xx.games plus reusable debug scripts.
- `interp/`
  - Interpretability tooling. Start with `interp/README.md` if the task touches analysis or reports.

## Mental Model That Matters

- State layout is flat and normalized. Writes usually divide by a constant; reads usually multiply and round back to ints.
- Many visible fields are derived views of hidden/canonical state. If you change state logic, expect both low-level and visible representations to matter.
- Action space size depends on player count. Use `get_total_action_count(num_players)` and `get_action_layout(...)`; do not hardcode totals.
- Some phases are automated (`WRAP_UP`, `INCOME`, `END_CARD`). The driver may auto-advance through them or auto-apply forced actions.
- `GameState._layout` is a Cython-only struct. From Python, use `core.state.get_layout(num_players)`.
- Acquisition and closing use hidden offer buffers and present offers one at a time. If those phases misbehave, inspect buffer setup and revalidation logic before changing action masks.

## Read Order By Task

- Game-rule bug or behavior change:
  - `RULES.md`
  - relevant `phases/*.pyx`
  - `core/driver.pyx`
  - matching tests in `tests/phases/`
- State/layout/vector issue:
  - `VECTORS.md`
  - `core/state.pyx`
  - relevant entity modules
  - `tests/test_state_layout.py`
- Action-space change:
  - `core/actions.pyx`
  - `core/driver.pyx`
  - relevant phase file
  - tests covering that phase
- Search / MCTS work:
  - `mcts/search.py`
  - `mcts/node.py`
  - `mcts/evaluator.py`
  - `tests/test_mcts.py`
- Training / self-play / checkpointing:
  - `train/config.py`
  - `train/main.py`
  - `train/self_play.py`
  - `train/eval_server.py`
  - `tests/test_training.py`
- Interpretability:
  - `interp/README.md`
  - specific scripts under `interp/`
- Replay mismatch against 18xx.games:
  - `tests/18xx_games/README.md`
  - `tests/18xx_games/replay_harness.py`
  - `tests/18xx_games/action_parser.py`
  - existing scripts in `tests/18xx_games/debug/`

## Build And Verification

- Use `.venv/bin/python`, not bare `python`.
- Build extensions before running code that imports `core`, `entities`, `phases`, or `mcts`:

```bash
.venv/bin/python setup.py build_ext --inplace
```

- If you changed `.pxd` files or low-level layout/signature details, clean first:

```bash
.venv/bin/python setup.py clean
.venv/bin/python setup.py build_ext --inplace
```

- Full test gate:

```bash
pytest tests/
```

- Targeted commands that are often useful:

```bash
pyright <path>
.venv/bin/python setup.py trace_game
.venv/bin/python setup.py benchmark --device=cpu --batch-size=4
.venv/bin/python -m train --num-workers 0 --games-per-epoch 10
```

## Testing Notes

- `tests/phases/conftest.py` is important. Its fixtures and `assert_invariants(...)` cover many derived-state expectations, so not every feature needs a dedicated test file.
- For engine/rules changes, run the relevant phase tests first, then `pytest tests/`.
- For MCTS or training changes, start with `tests/test_mcts.py` and `tests/test_training.py`, then run the full suite.
- Replay tests in `tests/18xx_games/` require Ruby and the checked-in `18xx/` submodule content. Read that README before assuming a replay mismatch is an engine bug.
- Existing replay debug scripts are usually better than ad-hoc one-off scripts.

## Environment / Workflow Gotchas

- `install.sh` expects an **activated** venv and installs platform-specific PyTorch for `cpu`, `cuda`, or `rocm`.
- There is no `pyproject.toml`; `setup.py` is the build entry point.
- `pyright` is expected to be the system binary, not `.venv/bin/pyright`.
- Search/training worker partitioning is constrained by a uint64 bitmap design: each eval-server partition may cover at most 64 workers. `TrainingConfig` validates this.
- The repo uses `bd` / beads for issue tracking. If you discover follow-up work that is out of scope, record it instead of letting it disappear.
- Use `scratchpad/` for temporary investigation scripts if you need local throwaway code.
