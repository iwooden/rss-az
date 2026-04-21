# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project

High-performance Cython engine for "Rolling Stock Stars" — AlphaZero self-play target of thousands of games/min. Single contiguous `int16` `GameState`, nogil hot paths, entity-readout transformer NN. Per-token NN features are produced lazily by `get_token_data()`, separate from state storage.

> **"Rolling Stock Stars" is NOT "Rolling Stock"** — rules differ. `RULES.md` is authoritative.

- Engine supports 2–6 players; NN/MCTS/training scoped to 3–5p.
- Working branch: `transformer-refactor`. `main` preserves pre-refactor code as a reference for **rule intent only** — do not copy state layout, field offsets, action indices, or signatures from it.

## Authoritative docs

Don't auto-load; open when the task needs them.

| Doc | Scope |
|-----|-------|
| `RULES.md` | Game rules — before touching game logic or phase handlers |
| `VECTORS.md` | State buffer layout, action encoding, enums — before touching `core/state`, `core/actions`, entity handles |
| `transformers.md` | Model architecture, eval pipeline — before touching `nn/` or the evaluator |
| `token-data.md` | Per-token feature spec — paired with `core/token_data.{pyx,pxd}` |
| `sparse-refactor.md` | Historical design notes (deep reference only) |

## Directory map

```
core/        state.pyx, data.pyx, actions.pyx, driver.pyx, token_data.pyx
entities/    player, corp, company, deck, turn, market, fi — 7 stateless handles
phases/      14 phase handlers: invest, bid, acq_select_corp, acq_select_company,
             acq_select_price, acq_offer, closing, dividends, income, issue,
             ipo, par, wrap_up, end_card
nn/          transformer.py — token-based model
mcts/        search.py, node.py, evaluator.py, mcts_core.pyx
train/       main, self_play, eval_server, trainer, replay_buffer,
             analyze_game, tournament; gpu/{nvidia,amd}.py
tests/       phases/ (full phase suite), test_driver_step_mode, test_mcts,
             test_self_play; games_18xx/data/ (JSON fixtures, replay harness pending)
scratchpad/  Ad-hoc scripts (gitignored)
```

## Architecture in one breath

- **State.** `GameState` wraps one contiguous `int16` numpy array. Raw integers only — no normalization, no one-hot, no visible/hidden split. Layout offsets live as Cython `cdef` structs at module scope on `core.state` (`LAYOUT`, `PLAYER_FIELDS`, `CORP_FIELDS`, `TURN_OFFSETS`, `COMPANY_OFFSETS`, `DECK_OFFSETS`, `FI_OFFSETS`). Full layout in `VECTORS.md`.
- **Entity handles.** `PLAYERS[i]`, `CORPS[c]`, `COMPANIES[i]`, `TURN`, `FI`, `MARKET`, `DECK` are stateless singletons — one set, reused with any `GameState` at any player count. All state reads and writes go through them.
- **Actions.** Phase-local integer ids (see `VECTORS.md` Action Space). 11 decision phases (`DecisionPhase` in `core/data.pxd`); engine's 15 `GamePhases` fold to them via `ENGINE_TO_DECISION_PHASE`. Sparse legal-action enumeration via `enumerate_legal_actions(state, uint16_t* ids)`; buffers pad to `MAX_LEGAL_ACTIONS` and overflow is a loud assert.
- **Driver.** `core/driver.pyx::GameDriver` routes a phase-local `action_id` through legality check → phase handler → auto-chain (fast-forward through automated phases + forced decisions until a real multi-choice decision or `PHASE_GAME_OVER`). Optional `history` list records `(state._array.copy(), phase_id, action_id)` tuples pre-mutation.
- **Token extraction.** `core/token_data.get_token_data(state, buffer)` fills `(num_players + 56, TOKEN_DIM=97)` float32 inside one nogil block; `get_token_data_batch` amortizes Python dispatch via `GameState.rebind`. Sole engine → NN bridge.
- **Model.** `nn/transformer.py::RSSTransformerNet` — pre-RMSNorm + SwiGLU, entity-readout heads, unified-head policy dispatch via static action LUT, per-row logit gather. `TransformerConfig` defaults: 3p, d_model=128, 10 layers, 2 heads, ~2.39M params. `forward(x, action_ids, n_legals, phase_ids) → (policy_logits[B, K_MAX], values[B, N])`.
- **Evaluator / MCTS / training.** `mcts/evaluator.py` (NNEvaluator, RemoteEvaluator) speaks token buffers end-to-end. `mcts/search.py` is sparse / subtree-reusing. `train/*` runs full self-play + training loop on 3p.

## Code conventions

- **All state access goes through entity handles.** Non-negotiable. Use `PLAYERS[i].get_cash(state)`, `CORPS[c].get_share_price(state)`, `COMPANIES[i].get_location(state)`, etc. Do **not** cimport `LAYOUT` / `*_FIELDS` / `*_OFFSETS` outside `entities/` or `core/token_data.pyx`. Do **not** index `state._data` directly. Do **not** write ad-hoc field accessors in phase code. Handles own dirty-mask invalidation, cached-star bookkeeping, and ownership-location sync — reaching around them corrupts state.
- **Entity method surfaces are mostly finalized.** If you want a method that doesn't exist, **stop and ask the user** before adding one. Same for changing existing semantics.
- **`core/data` is data-only.** Static arrays + enums + normalization divisors. No field-level helpers. Per-entity helpers (synergy aggregation, CoO lookup, par validity) live as private `cdef` functions in the entity that uses them.
- **Naming.** `player_id` / `corp_id` / `company_id` are indices; `PLAYERS[i]` / `CORPS[c]` / `COMPANIES[i]` are the singleton handles. `PHASE_*` = `GamePhases`, `DPHASE_*` = `DecisionPhase`, `LOC_*` = `CompanyLocation`.
- **Assert, don't fall back silently.** In Cython, validate invariants with `assert` + an f-string message — not `if bad: return 0`. Out-of-range ids, inactive entities that should be active, malformed state: crash loudly. `assert` compiles out under `python -O`. **Exception:** genuine business-logic branches ("can't afford the share, so the action is illegal") still return cleanly.
- **No method-level imports.** All imports at file scope — production, tests, scripts.

## Build

- Python: always `.venv/bin/python` (venv may not be activated).
- Pyright: system `/usr/bin/pyright`, NOT `.venv/bin/pyright`.
- Submodules: absolute paths or single-line `cd /home/icebreaker/rss-az-cython2/submodules/18xx && ...`.

```bash
.venv/bin/python setup.py clean # especially needed if you've edited any .pxd files
.venv/bin/python setup.py build_ext --inplace 2>&1 | grep -E "(warning|error)" || true
```

- **Warning-free builds required.** File a beads issue if warnings appear.
- **Clean before final verification** — incremental builds miss `.pxd` header changes.
- **Fix pyright errors before moving on**, even pre-existing. Run `pyright <file>` via Bash for the definitive result (auto-injected diagnostics can be stale).

## Testing

- `pytest tests/` — phase suite (`tests/phases/`), driver step-mode, MCTS, end-to-end self-play smoke. 1400+ tests; runs clean.
- Prefer invariant assertions (cash conservation, share counts, ownership consistency) at transitions over narrow field-level checks.
- **When a test fails, assume the implementation is broken** until proven otherwise.
- `tests/games_18xx/data/` holds 18xx.games JSON fixtures; the replay harness for the new driver is pending. Known rule divergences to remember when it returns: (1) cross-president ACQ transfers, (2) directly offering positive-income company closes in CLOSING.

## Key files by task

| Task | Primary | Secondary |
|------|---------|-----------|
| Layout / field offsets | `core/state.{pyx,pxd}` | `VECTORS.md`, `entities/*.pxd` |
| Action encoding | `core/actions.{pyx,pxd}` | `core/data.pxd::ActionSize`, `VECTORS.md` |
| Static data (synergies, CoO, par) | `core/data.{pyx,pxd}` | `RULES.md` |
| Entity field access | `entities/<entity>.{pyx,pxd}` | — |
| Phase logic | `phases/<phase>.{pyx,pxd}` | entity handles, `RULES.md` (ask before adding handle methods) |
| Driver / game loop | `core/driver.{pyx,pxd}` | `phases/*.pyx` |
| Token features | `core/token_data.{pyx,pxd}` | `token-data.md` |
| Model | `nn/transformer.py` | `transformers.md` |
| Evaluator / MCTS | `mcts/evaluator.py`, `mcts/search.py` | `transformers.md` |
| Training loop | `train/main.py`, `train/trainer.py`, `train/self_play.py` | `train/replay_buffer.py`, `train/eval_server.py` |

## Devbox

WSL2 on Windows. AMD Ryzen 9 9950X3D (32 cores). AMD Radeon RX 9070 XT (ROCm 7.2.0). Training defaults: 96 self-play workers, 2 eval servers.

---

# Agent Instructions

Issue tracking: **bd** (beads). `bd prime` is the authoritative workflow reference — the `SessionStart` hook runs it automatically, so treat its output as source of truth for commands and the session-close protocol.

- File beads issues for any out-of-scope discoveries. **No insights lost.** Always include `--description` — a title alone is not enough.
- Use `bd create --parent=<id>` for related subtasks (dot-notation ids), independent issues for unrelated bugs.
- Claim work with `bd update <id> --claim` (the old `--status=in_progress` still works but `--claim` is preferred).
- Remote sync is `bd dolt pull` / `bd dolt push` (the old `bd sync` no longer exists).
- Persistent cross-session knowledge lives in `bd remember "insight"` / `bd memories <keyword>`, injected by `bd prime`. Prefer this over ad-hoc MEMORY.md files for project-wide context.
- **Ad-hoc scripts:** write to `scratchpad/` (gitignored), iterate with `Edit`, run with `PYTHONPATH=/home/icebreaker/rss-az-cython2 .venv/bin/python scratchpad/<script>.py`. Don't inline 100-line scripts in Bash.
- **Session close:** follow the protocol from `bd prime`. Minimum gate is a clean + `build_ext`. This branch is ephemeral (no upstream) — code is merged to `main` locally, not pushed.
