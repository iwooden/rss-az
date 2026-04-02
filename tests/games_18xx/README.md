# 18xx Replay Tests

Validates our Cython game engine against completed games from [18xx.games](https://18xx.games) by replaying every action and comparing state at phase boundaries.

## Architecture

- **`utils_18xx/extract_states.rb`** — Ruby script that replays a game through the 18xx engine and extracts a state snapshot after every action. Requires the 18xx engine submodule at `submodules/18xx/`. This is the **preferred place for preprocessing** — the Ruby side has direct access to the 18xx engine's internal state, making it much easier to annotate snapshots with metadata (forced actions, round labels, round summaries, etc.) than reconstructing that information on the Python side.
- **`utils_18xx/action_parser.py`** — Converts 18xx action streams into our engine's integer action indices. Handles committed-action filtering (via IDs from the extractor), auto-action flattening, per-phase mapping, and shared utilities (deck override, action dispatch).
- **`replay_harness.py`** — Orchestrates replay: initializes our engine with the 18xx deck order, replays actions through the parser, and compares state snapshots.
- **`test_replay.py`** — Pytest entry point. Dynamically discovers game JSON files in `data/`.

## 18xx.games Submodule Orientation

The vendored `submodules/18xx/` submodule is the full 18xx.games repository. For Rolling Stock Stars replay work, most of it is irrelevant. Start with these files:

- **`submodules/18xx/lib/engine.rb`** — top-level loader that requires all game definitions and resolves game titles.
- **`submodules/18xx/lib/engine/game/base.rb`** — `Engine::Game.load(...)`, which `extract_states.rb` uses to load a JSON export at a given action boundary.
- **`submodules/18xx/lib/engine/game/g_rolling_stock/game.rb`** — the base Rolling Stock implementation. This is the main source for round sequencing, wrap-up / FI buying, ACQ, CLO, INCOME, DIV, ISSUE, and IPO behavior.
- **`submodules/18xx/lib/engine/game/g_rolling_stock_stars/game.rb`** — Rolling Stock Stars overrides on top of base Rolling Stock: market, par table, cost-of-ownership table, phase count, and star-price movement.
- **`submodules/18xx/lib/engine/game/g_rolling_stock/round/*.rb`** — round labels and round-local control flow. In replay work, `investment.rb`, `acquisition.rb`, `closing.rb`, `dividends.rb`, `issue.rb`, and `ipo.rb` are the important ones because the extractor records `round.class.short_name`.
- **`submodules/18xx/lib/engine/game/g_rolling_stock/step/*.rb`** — action semantics. The most relevant files are:
  - `buy_sell_shares_bid_companies.rb` for INVEST and auction behavior
  - `propose_and_purchase.rb` and `receiver_propose_and_purchase.rb` for ACQ offers, right-of-first-refusal, and receivership FI buying
  - `close_companies.rb` for CLO behavior and program close-pass handling
  - `dividend.rb` for explicit vs forced dividends
  - `issue_shares.rb` for ISSUE behavior, including receivership auto-issue
  - `ipo_company.rb` for IPO/PAR semantics and player cash cost to convert
- **`submodules/18xx/public/fixtures/RollingStockStars/`** — upstream sample game exports from 18xx.games. Useful as Ruby-side reference examples, but our replay tests use the local ignored files under `tests/games_18xx/data/`.

If you are debugging a replay mismatch, read the Ruby game flow in this order:

1. `utils_18xx/extract_states.rb`
2. `submodules/18xx/lib/engine/game/g_rolling_stock/game.rb`
3. the specific `round/*.rb` file for the mismatch round
4. the specific `step/*.rb` file that processes the action type

That is usually enough. You rarely need to inspect unrelated titles or the web app/frontend parts of the submodule.

## Engine Differences

Our Cython engine makes several intentional design choices that differ from the 18xx.games implementation. The replay harness translation layer must account for each of these.

### 1. Acquisition Offer Scope

**18xx:** During Phase 3 (Acquisition), any player can offer any company to any corporation. Offers are open negotiation — a player can offer a company they own to a corp presided by a different player, and that president accepts or rejects.

**Our engine:** Acquisition offers are constrained to same-president transactions. A corp can only acquire companies when its president also controls the selling entity (owns the private company, or is president of the selling corp). Cross-president offers are excluded from the action space.

**Replay handling:** The Ruby extractor post-processes snapshots to compute ACQ outcomes for each round — diffing corp company ownership before vs after, with offer prices scoped to the round's action ID range, seller info (type, name, player ID), and a `cross_president` flag. The Python ACQ adapter replays same-president offers normally, pauses at the ACQ->CLO boundary, then patches unresolved cross-president outcomes into the acquisition zone before resuming phase advancement.

### 2. Acquisition Offer Ordering

**18xx:** Players take turns offering companies in any order they choose. The action stream records offers as they happened.

**Our engine:** Offers are generated into a hidden buffer and presented one-by-one in a fixed priority order: OS→FI (face DESC) → Corp→FI (price DESC) → Corp→Corp → Corp→Player.

**Replay handling:** The extractor emits ACQ round summaries, and the ACQ adapter consumes those summaries round-by-round rather than rediscovering boundaries from the flat action stream. Same-president corp-to-corp transfers are still pre-applied before the buffer walk (using `acquisition_proceeds` for correct cash flow), because fixed offer ordering can otherwise diverge. The adapter then walks the remaining offer buffer, accepts or passes offers based on the round summary, pauses at ACQ exhaustion, patches deferred outcomes, and only then advances into CLO.

### 3. Closing Offer Scope

**18xx:** During Phase 4 (Closing), any player or president may close any company they control, regardless of income. Companies with positive, zero, or negative adjusted income can all be closed.

**Our engine:** By default, optional close offers are generated only for companies with **negative adjusted income** (income − cost of ownership < 0). For replay, the harness enables `allow_closing_positive_income`, so the engine offers the full 18xx closing scope instead of only negative-income companies.

**Replay handling:** The extractor emits a CLO round summary containing `closed_companies`. The CLO adapter uses that summary to decide which offers to take while walking our engine's close-offer buffer. No direct close pre-apply or CLO look-ahead is needed when replay runs with `allow_closing_positive_income`.

### 4. Closing Offer Ordering

**18xx:** Players choose which companies to close in any order.

**Our engine:** Close offers are sorted by face value ascending and presented one-by-one from a hidden buffer.

**Replay handling:** The CLO adapter reads `closed_companies` from the extractor's round summary, then matches that set against our engine's ordered offer buffer.

### 5. IPO / PAR Phase Split

**18xx:** IPO is a single action: the player selects both the target corporation and the par price in one `par` action.

**Our engine:** IPO is split into two sequential phases: IPO (select corporation) → PAR (select par price). This keeps the action space smaller and more uniform.

**Replay handling:** The action parser's `map_ipo_action()` maps a single 18xx `par` action to two engine actions: `[ipo_base + corp_id, par_base + par_index]`. If the engine auto-applies the PAR action (only one valid par price), the second action is skipped.

### 6. Auto-Pass (program_share_pass / program_close_pass)

**18xx:** Players can enable "auto-pass" programs that automatically pass for them in future INVEST or CLOSING phases. These generate `program_share_pass` and `program_close_pass` actions in the stream, and cause additional `pass` auto-actions to be inserted.

**Our engine:** No auto-pass concept. Every pass is an explicit action.

**Replay handling:** Program actions are filtered out by `filter_actions()` (in `SKIP_ACTIONS`). Their auto-actions are preserved only if the program action itself is committed (not undone) — undone program actions must not leak auto-action passes into the stream, as that would advance the active player and desync the replay. The Ruby extractor includes committed skip-type action IDs in `committed_action_ids` so the Python side can distinguish committed from undone program actions. Committed auto-actions are flattened into the main stream by `flatten_auto_actions()` and replayed as normal pass actions.

### 7. Undo / Redo

**18xx:** Players can undo and redo actions during a game. The action stream contains `undo` and `redo` actions that revert or re-apply previous actions.

**Our engine:** No undo/redo support. Actions are final once applied.

**Replay handling:** The Ruby extractor (`extract_states.rb`) resolves undo/redo at extraction time. It maintains an `engine_action_stack` that tracks all processed actions (including skip-types like `program_*`), so that each undo correctly identifies what the engine actually undid. When an undo reverts a skip-type action (which has no snapshot), no snapshot is popped — only real actions cause snapshot pops. Undo groups store both engine-stack entries and snapshots so that redo correctly restores both. The extractor emits `committed_action_ids` in the initial record (including committed skip-type IDs), which `filter_actions()` uses to drop undone actions from the raw stream without reimplementing undo logic in Python.

The extractor mirrors the 18xx engine's `filtered_actions` semantics (base.rb line 804): new non-undo/redo actions clear `undo_groups` (the redo stack), and `undo action_id=X` always pushes a group (even if empty, when X is already the stack top). Without these, stale undo groups leak into subsequent redo operations and produce incorrect `committed_action_ids`.

### 8. Auto-Applied Forced Actions

**18xx:** Every action is explicit in the action stream, even when there's only one valid choice (e.g., dividend 0 for a corp with insufficient cash).

**Our engine:** Forced actions (only one valid choice) are auto-applied without player input. Examples:
- Dividends when `cash < issued_shares` (only dividend 0 is valid)
- Receivership dividends (always 0)
- IPO passes when a company's owner can't afford any valid par price
- PAR price selection when only one par price is valid for the company's star tier
- Any single-option offer in ACQ/CLO phases

**Replay handling:** The Ruby extractor tags forced actions with `forced: true` in the snapshot (dividends where `max_dividend_per_share == 0` or corp is in receivership; IPO passes where the owner can't afford any valid par price). The replay harness checks this flag and skips both the comparison and the action mapping. For PAR, it checks whether the engine already advanced past `PHASE_PAR`.

### 9. Round Label Timing in Extractor

The Ruby extractor captures the round label **before** processing each action, not after. This ensures the last action of a round (which causes a phase transition) is labeled with the round it was *taken in*, not the round the engine transitions *to*.

The extractor also emits explicit round metadata so replay can operate at the round level instead of rescanning the flat action stream:

- `round_seq` on snapshots to distinguish adjacent rounds of the same type
- `round_start_action_id` / `round_end_action_id` and `round_final_snapshot`
- top-level `rounds` entries with `compare_after_action_id`
- round summaries such as `acq_outcomes` and `closed_companies`

### 10. Cost of Ownership Level Numbering

**18xx (Ruby):** Uses levels 1-5, 7, 8 (skipping 6). Level 7 = deck empty / game end card front (7-coin side). Level 8 = game end card flipped (10-coin side).

**Our engine:** Uses contiguous levels 1-7. Level 6 = game end card front. Level 7 = game end card flipped.

**Replay handling:** The Ruby extractor normalizes levels at extraction time (7 → 6, 8 → 7), so the Python comparator receives our engine's numbering directly.

## State Comparison

The harness compares these fields at action boundaries (before applying each action):

- **Players:** cash, net worth (value), owned companies, shares per corp
- **Corporations:** active/floated status, share price, treasury cash, owned companies, shares in market (bank)
- **Foreign Investor:** cash, owned companies
- **Offering:** companies available for auction + revealed/unavailable
- **Deck:** remaining card count, cost of ownership level
- **Active entity:** active player (INVEST/BID/IPO) or active corp (DIVIDENDS/ISSUE), only when phases are aligned between engines

## Debug Scripts

The `debug/` directory contains reusable scripts for investigating replay failures. **Always prefer extending these over writing ad-hoc inline scripts.** If you need a new debug pattern, add it as a script here.

All scripts operate on extract files (`data/<game_id>_extract.json`) and/or raw game JSONs (`data/<game_id>.json`). Run from the repo root with `python tests/games_18xx/debug/<script>.py`.

| Script | Purpose | Example |
|--------|---------|---------|
| `track_company.py` | Track a company's ownership across snapshots | `python ... 210560 SJ --transitions-only` |
| `show_acq_outcomes.py` | List all ACQ outcomes (with cross-president flags) | `python ... 210560 --cross-only` |
| `dump_state.py` | Dump full state at specific action IDs | `python ... 210560 280 282` |
| `diff_states.py` | Diff two snapshots to see what changed | `python ... 210560 280 282` |
| `show_actions.py` | Show raw 18xx actions with committed/undone status | `python ... 210560 --range 278-290 --expand-autos` |
| `show_round.py` | Show all snapshots for a round type (ACQ, CLO, etc.) | `python ... 210560 ACQ --range 260-290` |
| `find_gaps.py` | Find phase-transition gaps from undone actions | `python ... 213447` |
| `check_committed_ids.py` | Compare old vs new undo tracking to find affected games | `python ... 213447` or `python ...` (all) |
| `replay_to_action.py` | Replay to a specific action and dump engine state | `python ... 213447 322` |

### Common workflows

**First mismatch in a test:** Run the test with `-x --tb=long` to get the first mismatch action ID and field. Then use `dump_state.py` to inspect the reference state at that point, and `diff_states.py` to see what changed in the turn leading up to it.

**Company in wrong place:** Use `track_company.py --transitions-only` to find when ownership diverged. Then `diff_states.py` on the two surrounding action IDs to see the full delta.

**ACQ/CLO issues:** Use `show_acq_outcomes.py` to see all transfers and cross-president flags. Use `show_round.py ACQ` to see the full ACQ round state. Use `show_actions.py` to see the raw 18xx actions (offer/respond/pass).

**Invisible turn cycles:** When two adjacent extract snapshots are both INV-round but state changed significantly (receivership auto-buys, INCOME, etc.), use `diff_states.py` to see everything that happened during the automated phases.

**Undo gaps:** When the engine is stuck in a phase but the action stream jumps ahead, use `find_gaps.py` to find places where undone actions created phase-transition holes. Use `show_actions.py --expand-autos` to see the full undo/redo chain and auto_actions in the gap.

## Adding a Game

1. Export game JSON from 18xx.games → save to `data/<game_id>.json`
2. Extracts are auto-generated: the pytest session fixture refreshes any missing or stale `_extract.json` files before tests run. You can also run `ruby utils_18xx/extract_states.rb tests/games_18xx/data/<game_id>.json > tests/games_18xx/data/<game_id>_extract.json` manually to regenerate a specific extract.
3. The test auto-discovers all game JSONs in `data/` (excluding `*_extract.json`).
4. If a game exposes a confirmed **18xx.games engine bug** (not a bug in our engine), add its ID to `SKIP_GAMES` in `test_replay.py` with a comment explaining the bug.
