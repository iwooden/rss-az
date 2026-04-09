# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance Cython game engine for "Rolling Stock Stars" board game, optimized for AlphaZero-style self-play training. Game state is stored as a single contiguous int16 numpy array. Per-token features for the NN are produced lazily by `get_token_data()`, separate from state storage.

**Important:** "Rolling Stock Stars" is a different game than "Rolling Stock". They share similarities but many rules differ. Do NOT rely on knowledge of "Rolling Stock" — always consult `RULES.md` as the authoritative source for game rules.

**Key characteristics:**
- 2-6 player support with dynamic state sizing
- 454-610 int16 values per game state (3p = 493)
- No Python object overhead in hot paths (nogil execution)
- Benchmark target: thousands of games per minute

## Devbox Hardware

- **Platform:** WSL2 on Windows
- **CPU:** AMD Ryzen 9 9950X3D (32 usable cores)
- **GPU:** AMD Radeon RX 9070 XT (ROCm 7.2.0)

## Directory Structure

```
rss-az-cython2/
├── core/              # Low-level engine: state.pyx, driver.pyx, actions.pyx, data.pyx
├── entities/          # Entity handles: player, corp, company, deck, turn, market, fi
├── phases/            # Phase handlers: invest, bid, acquisition, closing, dividends, income, issue, ipo, wrap_up, end_card
├── mcts/              # MCTS search: node.py, evaluator.py, search.py, mcts_core.pyx (Cython hot functions + signaling)
├── nn/                # Neural network: model_3p.py (residual MLP, policy + value heads)
├── train/             # Self-play training: config, eval_server, self_play, replay_buffer, trainer, checkpoint, logging, main
│   └── gpu/           # Vendor-specific GPU optimizations: nvidia.py, amd.py (auto-detected)
├── tests/             # Test suite: phases/, games_18xx/ replay tests, conftest.py
├── RULES.md           # Complete game rules (authoritative)
├── VECTORS.md         # State/action vector documentation
```

## Architecture Overview

### Entity Handles Pattern

Global singleton instances provide clean access to state array regions:

```python
PLAYERS = [Player(i) for i in range(6)]  # Player handles
CORPS = [Corporation(i) for i in range(8)]  # Corporation handles
TURN = TurnState()  # Turn tracking
FI = ForeignInvestor()  # Foreign investor
```

Each entity provides:
- **cdef methods** (nogil): Direct pointer arithmetic, max performance
- **cpdef methods**: Python-accessible wrappers for testing
- **Access pattern**: `entity.get_field(state)` indexes the state buffer at a constant offset derived from the module-level `LAYOUT` / `PLAYER_FIELDS` / `CORP_FIELDS` / `TURN_OFFSETS` structs on `core.state`. **Every** handle (Player, Corp, Company, TurnState, Market, FI, Deck) is fully stateless — no per-instance offset cache, no `initialize()` step. The singleton handles in `PLAYERS` / `CORPS` / `COMPANIES` are reused with any `GameState` at any player count; the only per-instance fields are display identifiers (`player_id`, `corp_id`, `name`).

Low-level (`cdef` nogil) wraps high-level (cpdef/def) with no code duplication.

## Key Modules

### GameState (`core/state.pyx`)

Single contiguous int16 numpy array. Raw integers only — no normalization, no one-hot encoding, no visible/hidden split.

**Sizes by player count:**

| Players | total_size | player_stride | corp_stride |
|---------|-----------|---------------|-------------|
| 2 | 454 | 39 | 17 |
| 3 | 493 | 39 | 17 |
| 4 | 532 | 39 | 17 |
| 5 | 571 | 39 | 17 |
| 6 | 610 | 39 | 17 |

The fixed prefix (everything except the players section) is **376** int16 slots and is identical across player counts. Only `total_size` scales: `total_size = 376 + 39 * num_players`. The players section lives at the end of the buffer for exactly this reason — every other section's offset is constant.

`GameState` is basically just a thin wrapper around the state vector. There are no per-instance layout fields — entity handles read offsets directly from the module-level constants below. All field-level reads and writes go through entity handles in `entities/`.

`GameState.from_array(array, num_players)` is the copy-in path and accepts non-contiguous 1-D `int16` views. `GameState.from_buffer(buffer, num_players)` and `state.rebind(buffer, num_players)` are the zero-copy paths and require writable C-contiguous `int16` buffers whose canonical `turn.num_players` slot already matches the claimed player count.

**Module-level layout constants** on `core.state` (computed once at import, shared by every `GameState`):

- `LAYOUT` — `cdef StateLayout`: top-level section offsets (`players_offset`, `fi_offset`, `corps_offset`, `turn_offset`, …) plus `player_stride` and `corp_stride`. `total_size` is *not* in this struct because it depends on `num_players`; compute it inline as `LAYOUT.players_offset + LAYOUT.player_stride * num_players` at the few sites that need it (allocation, length validation).
- `PLAYER_FIELDS` — `cdef PlayerFieldOffsets`: relative offsets within a player block (`cash`, `owned_shares`, `auction_passed`, `stride`, …).
- `CORP_FIELDS` — `cdef CorpFieldOffsets`: relative offsets within a corp block.
- `TURN_OFFSETS` — `cdef TurnStateOffsets`: relative offsets within the (fixed-size) turn block.

These are Cython `cdef` structs — **NOT accessible from Python directly**. Cython code uses `from core.state cimport LAYOUT, PLAYER_FIELDS, CORP_FIELDS, TURN_OFFSETS`. Python code uses the namedtuple accessors `core.state.get_layout(num_players)`, `get_player_fields()`, `get_corp_fields()`, `get_turn_fields()`. See `VECTORS.md` for the full layout.

### Actions (`core/actions.pyx`)

**Subject to change** as part of the transformers refactor.

### Driver (`core/driver.pyx`)

Stateless game loop: `apply_action(state, action_idx, history)` dispatches to phase handlers, auto-applies forced actions. Returns `STATUS_OK`, `STATUS_INVALID`, or `STATUS_GAME_OVER`.

### Data (`core/data.pyx`)

Pure data + constants module. Holds the static game tables (36 companies, 8 corporations, 27 market prices, par/CoO tables, synergy matrix), the shared enums (`GameConstants`, `GamePhases`, `CorpIndices`), and the normalization divisors used by NN token extraction. **No accessor functions, no computational helpers** — other modules `cimport` the underlying arrays directly. All gamestate reads/writes go through entity handles, never through helpers in `core/data`. Computational helpers (synergy aggregation, required-stars formula, cost-of-ownership lookup, par-price validity, etc.) live as private cdef functions in the module that uses them — e.g. `_aggregate_synergies` and `_required_stars` are inlined in `entities/corp.pyx`. If a helper ends up needed by multiple modules, promote it to a `cimport`-able symbol in whichever entity owns the concept rather than recreating an accessor layer in `core/data`.

## MCTS Search

Pure-Python AlphaZero-style MCTS for 3-player games. MCTSConfig lives in `train/config.py`.

### Value Representation

Value head outputs scalars in [-1, 1] for each player via tanh: `[v_player0, v_player1, ... ]`. 

**Terminal values:** Blend of rank-based and zero-sum net-worth-deviation rewards (`--terminal-blend`, default 0.75). Rank: `linspace(+1, -1)` by placement. Margin: `(n/(n-1)) * (nw_i - mean_nw) / max_nw`. Both zero-sum, stays in [-1, +1]. Use `--terminal-blend 1.0` for pure rank.

### PUCT Selection

```
UCB(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
```

Q(a) is the mean value for the **active player** at the parent node.

### A0GB Greedy Backup (Value Targets)

Instead of soft-Z or game outcome, we use **A0GB** (Willemsen et al., 2022): follow max-visit child from root to leaf/terminal, return that node's value as training target. Removes exploration bias, converges faster. Implementation: `get_greedy_leaf_value(root)` in `search.py`. Stops when best child has `visit_count == 0`.

### Search Flow

1. **Root setup:** Evaluate with NN, expand, add Dirichlet noise
2. **Per batch** (`search_batch_size` leaves): Select via PUCT → lock leaf (Q=-inf prevents re-selection) → batch evaluate → unlock + expand + backup
3. **Output:** `get_action_probabilities(root, temperature)` → policy target

**Subtree reuse:** Child subtree becomes new root after action. `prepare_reuse_root()` compacts state pool. Saves 40-60% GPU evals. Always enabled.

**Memory:** Search uses a preallocated `StatePool`; nodes store `state_idx` values into pool rows, and expansion rebinds one scratch `GameState` across those rows instead of allocating wrappers per node.

### NN Model (`nn/transformer.py`)

**Subject to change** as part of the transformers refactor.

## Self-Play Training

AlphaZero-style loop in `train/`. Each epoch: play N games via MCTS → store in replay buffer → train NN → checkpoint.

### Architecture

Worker processes (MCTS is CPU-bound, need own GIL) send states to eval server processes via shared memory (`SharedEvalBuffers`). Model shared zero-copy via CUDA IPC. `spawn` context, `torch.compile` per-process, daemon workers. `num_workers=0` for single-process debug.

**Worker ↔ eval server communication** uses per-server uint64 bitmaps for lockfree request submission and per-worker `mp.Event`s for done signaling. Workers atomically set a bit in their server's bitmap (`fetch_or`, release); servers atomically exchange the bitmap to zero (`exchange`, acquire) to claim all pending work in O(1). A per-server doorbell `mp.Event` wakes idle servers. Each server owns a static partition of workers `[worker_start, worker_end)` — max 64 workers per partition (bitmap width). Gather/scatter between per-worker slots and contiguous inference buffers uses Cython `nogil` memcpy. Signaling primitives live in `mcts/mcts_core.pyx`.

Key files: `train/eval_server.py` (EvaluationServer + RemoteEvaluator), `train/self_play.py` (play_game + worker entry), `train/main.py` (orchestration).

### Configuration

All hyperparameters in `TrainingConfig` dataclass (`train/config.py`). `TrainingConfig.to_mcts_config()` creates MCTSConfig. `--resume latest` to continue from checkpoint. Config validation enforces `num_eval_servers <= num_workers` and max 64 workers per eval server partition (uint64 bitmap width).

**Replay buffer** (3p, 500K capacity): ~4.2 GB. Saved to `checkpoints/replay_buffer/` via `ReplayBuffer.save()/load()`.

### Training Examples & Loss

At each decision point: state, legal_mask, policy_target (MCTS visits), value_target (A0GB).

**Loss:** Policy cross-entropy with MCTS targets + Value MSE with A0GB targets.

### Graceful Shutdown

- **Ctrl-C**: Hard exit
- **q + Enter**: Graceful — drains workers, saves checkpoint + replay buffer, then exits

## State Representation

See `VECTORS.md` for the full buffer layout. Key points:

- Single contiguous int16 array per `GameState`. Raw integers, no normalization, no one-hot encoding, no visible/hidden split.
- Sections in order: `FI (2) | companies (108) | market (27) | corps (17 × 8) | turn (66) | deck (37) | players (39 × N)`. The players section is **last** because it is the only section whose size depends on `num_players`; every other offset is constant.
- The companies section packs three parallel 36-slot sub-arrays — `incomes`, `locations`, `owner_ids` — reachable via `LAYOUT.companies_offset + COMPANY_OFFSETS.<field>`. The deck section packs `top` (1) + `order` (36) reachable via `LAYOUT.deck_offset + DECK_OFFSETS.<field>`.
- The turn block (66 slots, fixed across player counts) carries the game-wide metadata at the front (`active_player`, `num_players`, `phase`, `coo_level`, `turn_number`) followed by global scalars, the phase-remaining flag arrays, and two internal dirty-mask slots for the lazy player/corp caches. The per-player `auction_passed` flag lives in the player block, not the turn block.
- The corp block carries its star count split into three slots (`total_stars`, `cash_stars`, `company_stars`) and stores other derived corp values lazily behind a dirty bit. `share_price` is not stored; it is derived from `price_index`.
- All per-player data — cash, shares, presidencies, this-turn share buys/sells, and the per-player `auction_passed` flag — lives inside one player block, so `_player_ptr(i)` reaches everything for player `i` in a single pointer hop.
- `companies.locations` (`CompanyLocation` enum, 0–8) plus `companies.owner_ids` are the single source of truth for "who owns what". `LOC_DECK = 0` is the zero-init default; `__cinit__` explicitly seeds `companies.owner_ids` to `-1`.

## Game Flow & Phases

**12 Phases** (indices 0-11):
| Index | Phase | Description |
|-------|-------|-------------|
| 0 | INVEST | Buy/sell shares, start auctions |
| 1 | BID_IN_AUCTION | Bidding for a company |
| 2 | WRAP_UP | FI buying companies at face value |
| 3 | ACQUISITION | Corps acquiring companies |
| 4 | CLOSING | Player-owned companies closing |
| 5 | INCOME | Dividend payments |
| 6 | DIVIDENDS | Dividend calculation |
| 7 | END_CARD | Game end triggered |
| 8 | ISSUE_SHARES | Corp issuing shares |
| 9 | IPO | Select corp charter for company |
| 10 | PAR | Select par price for new corp |
| 11 | GAME_OVER | Terminal state |

**Automated phases** (no player input): WRAP_UP, INCOME, END_CARD

### Offer Buffer Pattern (acquisition.pyx, closing.pyx)

Both phases use **one-by-one offer presentation**: offers generated at phase entry, sorted by priority, presented sequentially, re-validated dynamically. Keeps action space constant.

**ACQUISITION:** Priority: OS→FI (face value DESC) → Corp→FI (price DESC) → Corp→Corp → Corp→Player. Receivership corps auto-buy from FI at HIGH if affordable, auto-pass otherwise. Actions: 51 price offsets, FI_BUY, PASS.

**CLOSING:** Two stages: (1) auto-close (FI negative-income, receivership red/orange above CoO), (2) player offers sorted by face value ASC. Actions: CLOSE, PASS. Mandatory close at end if player has negative income+cash.

## Code Conventions

- **Naming:** `corp_id`/`company_id`/`player_id` = indices; `CORPS[i]`/`PLAYERS[i]` = singletons; `PHASE_*` from `GamePhases`; `LOC_*` from `CompanyLocation`
- **Field access:** Use entity handles (`PLAYERS[i].get_cash(state)`, `CORPS[c].get_share_price(state)`) for all field reads/writes. `GameState` exposes only structural primitives. `core/data.{pxd,pyx}` is data-only (static arrays + enums + normalization constants) — there are no field-level helpers there; modules that need static game data `cimport` the underlying arrays directly.
- **Pointer safety:** `nogil` functions take pointers/offsets. Every entity handle reads offsets directly from `LAYOUT` / `PLAYER_FIELDS` / `CORP_FIELDS` / `TURN_OFFSETS` (cimported from `core.state`) and caches nothing per-instance. Player and Corp use a small inline `_slot(field)` helper that computes `LAYOUT.<section>_offset + id * LAYOUT.<section>_stride + field` — Cython inlines it away in nogil hot paths.
- **Guard with `assert`, not silent fallbacks:** In Cython code, validate parameters and invariants with `assert` rather than `if ...: return`/`return False`/`return 0` style guards. Out-of-range IDs, inactive entities that should be active, malformed state — all of these should crash loudly in development. Silently propagating a default value hides the real bug at the call site and lets corrupt state spread. `assert` statements compile out under `python -O`, so there is zero overhead in production. Examples: `assert 0 <= player_id < state._num_players`, `assert corp.is_active(state)`. Include a descriptive f-string message so failures are debuggable. This rule does *not* apply to genuine business-logic branches (e.g. "this player can't afford the share, so the action is illegal") — those still return cleanly. The rule is about *defensive* checks for things that should never happen if callers are correct.

## Build Commands

> ⚠️ **Refactor in progress:** A breaking refactor of `core/state.*` and `entities/` is underway. `setup.py` currently builds `core/data.pyx`, `core/state.pyx`, `core/actions.pyx`, and `entities/*.pyx`; `phases/`, `mcts/`, and root-level `.pyx` files are intentionally excluded and will not compile against the new state layout. The build target list will be expanded as each refactor phase lands. For this period, ignore everything outside `core/state.*`, `core/data.*`, `core/actions.*`, and `entities/` unless explicitly told otherwise; tests, training, and benchmarks are also expected to be broken until later phases.

**Python binary:** Always use `.venv/bin/python` (not `python` or `python3`). The venv may not be activated in the shell.

**Pyright:** Use `pyright` (system-installed at `/usr/bin/pyright`), NOT `.venv/bin/pyright`.

**Submodules:** When running commands in `submodules/18xx/`, use absolute paths or `cd /home/icebreaker/rss-az-cython2/submodules/18xx && ...` in a single Bash call. Do NOT `cd` into a submodule and forget — subsequent commands will run in the wrong directory.

```bash
# Build Cython extensions (required before running any Python code)
.venv/bin/python setup.py build_ext --inplace 2>&1 | grep -E "(warning|error)" || true

# Run all tests
pytest tests/

# Clean build artifacts
.venv/bin/python setup.py clean

# Run self-play training
.venv/bin/python -m train --device cuda --num-workers 4 --search-batch-size 8
.venv/bin/python -m train --num-workers 0 --games-per-epoch 10  # single-process debug

# Run MCTS benchmark
.venv/bin/python setup.py benchmark --device=cuda --batch-size=4
```

**Warning-free builds:** No compiler warnings expected. If warnings appear, create a beads issue.

**Pyright errors:** Fix before moving on. Run `pyright <file>` via Bash for definitive results (auto-injected diagnostics can be stale). Note: if you encounter a Pyright error, don't ignore it just because it wasn't caused by your change - fix pre-existing errors to keep the codebase clean!

## Testing Approach

**Key fixtures** (conftest.py): `game_state` (3-player, seed=42), `invest_state`, `bid_state`, `trade_state`, `apply_and_track`.

**Patterns:** Validate action effects, check mask validity, verify phase transitions, assert invariants (cash conservation, share counts).

**Derived state fields** (e.g. `pending_price_move`): Test via invariants in `assert_invariants()` (`tests/phases/conftest.py`), not dedicated test files. This validates correctness at every state transition across all phases and replay tests automatically. Only add a dedicated test file if the feature has complex standalone logic that isn't covered by invariant checks.

**When a test fails, assume the implementation is broken** until proven otherwise. Only "fix" a test after confirming the implementation is correct and the test setup was invalid.

**Status codes:** STATUS_OK (0), STATUS_INVALID (1), STATUS_GAME_OVER (2)

### 18xx.games Replay Tests

Validate our engine against completed games from 18xx.games by replaying every action and comparing state. **Requires Ruby** — `extract_states.rb` runs as subprocess.

Key files: `extract_states.rb` (state extraction), `action_parser.py` (action mapping with undo/redo handling), `replay_harness.py` (replay + comparison), `test_replay.py` (pytest entry).

**Action space differences:** Our engine doesn't support: (1) cross-president ACQ transfers, (2) directly offering positive-income company closes in CLO. Check for these before investigating engine bugs if replay fails.

**Adding a game:** Export JSON → save to `tests/games_18xx/data/<id>.json` → add to `@pytest.mark.parametrize` in `test_replay.py`.

## Key Files by Task

| Task | Primary Files | Secondary Files |
|------|---------------|-----------------|
| Add game rule | `phases/*.pyx` | `core/data.pyx`, `RULES.md` |
| Modify action space | `core/actions.pyx` | `core/driver.pyx`, `phases/*.pyx` |
| Debug state | `core/state.pyx`, `VECTORS.md` | Entity files |
| Optimize performance | Any `.pyx` | Check compiler directives, nogil |
| Fix bug | Tests first | Phase/entity files |
| MCTS / search | `mcts/search.py`, `mcts/node.py` | `mcts/evaluator.py`, `train/config.py` |
| Self-play / training | `train/main.py`, `train/config.py` | `train/self_play.py`, `train/eval_server.py`, `mcts/mcts_core.pyx` |
| Interpretability | `interp/README.md` | `interp/*.py` |

---

# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

**Before working on any game logic**, read `RULES.md` — authoritative source for rules.

## Agent Work Standards

Create beads issues for anything discovered that needs follow-up but is out of scope. **No insights or work should be lost.**

**Always include `--description="..."` when creating a `bd` issue.** A title alone is not enough — descriptions provide the context needed to pick up work later.

## Using Subtasks

Use subtasks (`bd create --parent=<id> --title="..." --type=task`) for related work. Results in dot-notation IDs: `abc.1`, `abc.2`. Use independent issues for unrelated bugs or cross-cutting concerns.

## Ad-hoc Test Scripts

Write ad-hoc scripts to the project `scratchpad` directory, not inline in Bash. Use `Edit` for iteration. Always prepend `PYTHONPATH=/home/icebreaker/rss-az-cython2` when running. This will save you a ton of tokens by making sure you don't have to write 100-line scripts from scratch on each invocation!

## Verification Before Closing

```bash
.venv/bin/python setup.py clean && .venv/bin/python setup.py build_ext --inplace 2>&1 | grep -E "(warning|error)" || true
pytest tests/
```

**IMPORTANT:** Always run `pytest tests/` without `--ignore`. All tests must pass.

## Landing the Plane (Session Completion)

Work is NOT complete until `git push` succeeds. **MANDATORY:**

1. File issues for remaining work
2. Run quality gates (tests, builds)
3. Update issue status (`bd close`)
4. Push: `git pull --rebase && bd sync && git push && git status`
5. Verify all changes committed AND pushed

**NEVER stop before pushing. NEVER say "ready to push when you are" — YOU must push.**
