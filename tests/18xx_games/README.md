# 18xx Replay Tests

Validates our Cython game engine against completed games from [18xx.games](https://18xx.games) by replaying every action and comparing state at phase boundaries.

## Architecture

- **`extract_states.rb`** — Ruby script that replays a game through the 18xx engine and extracts a state snapshot after every action. Requires the 18xx engine submodule at `18xx/`.
- **`action_parser.py`** — Converts 18xx action streams into our engine's integer action indices. Handles undo/redo, auto-action flattening, and per-phase mapping.
- **`replay_harness.py`** — Orchestrates replay: initializes our engine with the 18xx deck order, replays actions through the parser, and compares state snapshots.
- **`test_replay.py`** — Pytest entry point. Dynamically discovers game JSON files in `data/`.

## Engine Differences

Our Cython engine makes several intentional design choices that differ from the 18xx.games implementation. The replay harness translation layer must account for each of these.

### 1. Acquisition Offer Scope

**18xx:** During Phase 3 (Acquisition), any player can offer any company to any corporation. Offers are open negotiation — a player can offer a company they own to a corp presided by a different player, and that president accepts or rejects.

**Our engine:** Acquisition offers are constrained to same-president transactions. A corp can only acquire companies when its president also controls the selling entity (owns the private company, or is president of the selling corp). Cross-president offers are excluded from the action space.

**Replay handling:** The ACQ adapter builds outcomes by diffing corp company ownership before vs after the 18xx ACQ round. Transfers that our engine wouldn't offer (cross-president corp-to-corp, player-to-different-president corp) are pre-applied as direct state patches before the engine's offer buffer is walked.

### 2. Acquisition Offer Ordering

**18xx:** Players take turns offering companies in any order they choose. The action stream records offers as they happened.

**Our engine:** Offers are generated into a hidden buffer and presented one-by-one in a fixed priority order: OS→FI (face DESC) → Corp→FI (price DESC) → Corp→Corp → Corp→Player.

**Replay handling:** The ACQ adapter matches our engine's offers against the computed outcomes, accepting or passing each offer based on whether it matches a reference transfer.

### 3. Closing Offer Scope

**18xx:** During Phase 4 (Closing), any player or president may close any company they control, regardless of income. Companies with positive, zero, or negative adjusted income can all be closed.

**Our engine:** Optional close offers are generated only for companies with **negative adjusted income** (income − cost of ownership < 0). Zero and positive income companies are never offered.

**Replay handling:** The CLO adapter scans for `sell_company` actions in the CLO round. Companies with non-negative adjusted income that our engine won't offer are pre-applied as direct state patches (calling `remove_from_game()` with JS scrapping bonus if applicable).

### 4. Closing Offer Ordering

**18xx:** Players choose which companies to close in any order.

**Our engine:** Close offers are sorted by face value ascending and presented one-by-one from a hidden buffer.

**Replay handling:** The CLO adapter builds a set of closed company names from the reference, then matches against our engine's ordered offer buffer.

### 5. IPO / PAR Phase Split

**18xx:** IPO is a single action: the player selects both the target corporation and the par price in one `par` action.

**Our engine:** IPO is split into two sequential phases: IPO (select corporation) → PAR (select par price). This keeps the action space smaller and more uniform.

**Replay handling:** The action parser's `map_ipo_action()` maps a single 18xx `par` action to two engine actions: `[ipo_base + corp_id, par_base + par_index]`. If the engine auto-applies the PAR action (only one valid par price), the second action is skipped.

### 6. Auto-Pass (program_share_pass / program_close_pass)

**18xx:** Players can enable "auto-pass" programs that automatically pass for them in future INVEST or CLOSING phases. These generate `program_share_pass` and `program_close_pass` actions in the stream, and cause additional `pass` auto-actions to be inserted.

**Our engine:** No auto-pass concept. Every pass is an explicit action.

**Replay handling:** Program actions are filtered out by `filter_actions()` (in `SKIP_ACTIONS`). The auto-actions they generate are flattened into the main stream by `flatten_auto_actions()` and replayed as normal pass actions. An `AutoPassTracker` is maintained for potential future use but is not currently queried.

### 7. Undo / Redo

**18xx:** Players can undo and redo actions during a game. The action stream contains `undo` and `redo` actions that revert or re-apply previous actions.

**Our engine:** No undo/redo support. Actions are final once applied.

**Replay handling:** `filter_actions()` processes undos by maintaining committed and undone stacks. An `undo` action reverts all committed actions after the specified `action_id`. A `redo` re-commits the most recently undone action. Only the final committed actions are included in the filtered stream.

### 8. Round Label Timing in Extractor

The Ruby extractor captures the round label **before** processing each action, not after. This ensures the last action of a round (which causes a phase transition) is labeled with the round it was *taken in*, not the round the engine transitions *to*.

## Adding a Game

1. Export game JSON from 18xx.games → save to `data/<game_id>.json`
2. Run `ruby extract_states.rb data/` to generate the `_extract.json` file
3. The test will auto-discover it via `test_replay.py`'s dynamic game ID collection
