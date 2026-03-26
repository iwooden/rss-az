# 18xx Replay Tests

Validates our Cython game engine against completed games from [18xx.games](https://18xx.games) by replaying every action and comparing state at phase boundaries.

## Architecture

- **`extract_states.rb`** — Ruby script that replays a game through the 18xx engine and extracts a state snapshot after every action. Requires the 18xx engine submodule at `18xx/`. This is the **preferred place for preprocessing** — the Ruby side has direct access to the 18xx engine's internal state, making it much easier to annotate snapshots with metadata (forced actions, round labels, etc.) than reconstructing that information on the Python side.
- **`action_parser.py`** — Converts 18xx action streams into our engine's integer action indices. Handles committed-action filtering (via IDs from the extractor), auto-action flattening, and per-phase mapping.
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

**Replay handling:** The Ruby extractor (`extract_states.rb`) resolves undo/redo at extraction time — undone snapshots are popped from the output, and redos restore them. The extractor emits `committed_action_ids` in the initial record, which `filter_actions()` uses to drop undone actions from the raw stream without reimplementing undo logic in Python.

### 8. Auto-Applied Forced Actions

**18xx:** Every action is explicit in the action stream, even when there's only one valid choice (e.g., dividend 0 for a corp with insufficient cash).

**Our engine:** Forced actions (only one valid choice) are auto-applied without player input. Examples:
- Dividends when `cash < issued_shares` (only dividend 0 is valid)
- Receivership dividends (always 0)
- IPO passes when a company's owner can't afford any valid par price
- PAR price selection when only one par price is valid for the company's star tier
- Any single-option offer in ACQ/CLO phases

**Replay handling:** The replay harness detects each case and skips both the comparison and the action mapping for actions the engine already auto-applied. For dividends, it checks `cash < issued_shares` or bankrupt status. For IPO, it checks whether the owner can afford any par. For PAR, it checks whether the engine already advanced past `PHASE_PAR`.

### 9. Round Label Timing in Extractor

The Ruby extractor captures the round label **before** processing each action, not after. This ensures the last action of a round (which causes a phase transition) is labeled with the round it was *taken in*, not the round the engine transitions *to*.

### 10. Cost of Ownership Level Numbering

**18xx (Ruby):** Uses levels 1-5, 7, 8 (skipping 6). Level 7 = deck empty / game end card front (7-coin side). Level 8 = game end card flipped (10-coin side).

**Our engine:** Uses contiguous levels 1-7. Level 6 = game end card front. Level 7 = game end card flipped.

**Replay handling:** The state comparator remaps Ruby levels: 7 → 6, 8 → 7.

## State Comparison

The harness compares these fields at action boundaries (before applying each action):

- **Players:** cash, net worth (value), owned companies, shares per corp
- **Corporations:** active/floated status, share price, treasury cash, owned companies, shares in market (bank)
- **Foreign Investor:** cash, owned companies
- **Offering:** companies available for auction + revealed/unavailable
- **Deck:** remaining card count, cost of ownership level
- **Active entity:** active player (INVEST/BID/IPO) or active corp (DIVIDENDS/ISSUE), only when phases are aligned between engines

## Adding a Game

1. Export game JSON from 18xx.games → save to `data/<game_id>.json`
2. Extracts are auto-generated: the pytest session fixture runs `ruby extract_states.rb data/` to create any missing `_extract.json` files. You can also run it manually to pre-generate or regenerate extracts.
3. The test auto-discovers all game JSONs in `data/` (excluding `*_extract.json`).
4. If a game exposes a confirmed **18xx.games engine bug** (not a bug in our engine), add its ID to `SKIP_GAMES` in `test_replay.py` with a comment explaining the bug.
