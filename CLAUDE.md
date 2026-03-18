# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance Cython game engine for "Rolling Stock Stars" board game, optimized for AlphaZero-style self-play training. Game state is stored as a single contiguous float32 array that can be passed directly to PyTorch without serialization overhead.

**Important:** "Rolling Stock Stars" is a different game than "Rolling Stock". They share similarities but many rules differ. Do NOT rely on knowledge of "Rolling Stock" — always consult `RULES.md` as the authoritative source for game rules.

**Key characteristics:**
- 2-6 player support with dynamic state sizing
- ~2660-3010 floats per game state (varies by player count)
- No Python object overhead in hot paths (nogil execution)
- Benchmark target: thousands of games per minute

## Directory Structure

```
rss-az-cython2/
├── core/              # Low-level game engine
│   ├── state.pyx      # GameState: the central float32 array
│   ├── driver.pyx     # GameDriver: action dispatch and game loop
│   ├── actions.pyx    # Action space layout and decoding
│   └── data.pyx       # Static game constants (companies, corps, prices)
├── entities/          # Entity handles for state access
│   ├── player.pyx     # Player cash, shares, companies
│   ├── corp.pyx       # Corporation state (IPO'd companies)
│   ├── company.pyx    # Company state (auction deck items)
│   ├── deck.pyx       # Company deck management
│   ├── turn.pyx       # Turn and phase tracking
│   ├── market.pyx     # Share price market spaces
│   ├── fi.pyx         # Foreign Investor entity
│   └── encoding.pyx   # One-hot encoding utilities
├── phases/            # Game phase handlers
│   ├── invest.pyx     # Investment phase (buy/sell/auction)
│   ├── bid.pyx        # Auction bidding
│   ├── acquisition.pyx # Company acquisition offers
│   ├── closing.pyx    # Company closure logic
│   ├── dividends.pyx  # Dividend calculations
│   ├── income.pyx     # Income distribution
│   ├── issue.pyx      # Share issuance
│   ├── ipo.pyx        # IPO conversions
│   ├── wrap_up.pyx    # Turn wrap-up (FI buying)
│   └── end_card.pyx   # Game-end handling
├── mcts/              # MCTS search for AlphaZero training
│   ├── node.py        # MCTSNode: tree node with visit stats
│   ├── evaluator.py   # NN wrapper (state rotation, single/batch inference, value un-rotation)
│   ├── eval_cache.py  # Per-game NN eval cache (alternative to subtree reuse)
│   └── search.py      # PUCT selection, batched search with virtual loss, A0GB value targets
├── nn/                # Neural network models
│   └── model_3p.py    # Residual MLP with policy + per-player value heads
├── train/             # Self-play training harness
│   ├── config.py      # TrainingConfig (all hyperparameters)
│   ├── eval_server.py # Centralized GPU evaluator + RemoteEvaluator proxy
│   ├── self_play.py   # Game generation via MCTS + worker process entry point
│   ├── replay_buffer.py # Ring buffer for training examples
│   ├── trainer.py     # Loss computation, optimizer, LR schedule
│   ├── checkpoint.py  # Save/load model checkpoints
│   ├── logging.py     # Rich live UI + Tensorboard integration
│   ├── main.py        # Training loop orchestration
│   └── __main__.py    # python -m train entry point
├── tests/             # Test suite
│   ├── phases/        # Phase-specific tests
│   ├── 18xx_games/    # Replay tests against 18xx.games engine
│   │   ├── 18xx/      # 18xx.games Ruby engine (git submodule)
│   │   ├── data/      # Game JSON files from 18xx.games
│   │   ├── extract_states.rb   # Ruby state extractor
│   │   ├── replay_harness.py   # Python replay + comparison engine
│   │   ├── action_parser.py    # 18xx action → engine action mapper
│   │   └── test_replay.py      # Pytest entry point
│   └── conftest.py    # Pytest fixtures
├── RULES.md           # Complete game rules (authoritative)
├── VECTORS.md         # State/action vector documentation
└── RSS.pdf            # Original board game rulebook
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
- **Access pattern**: `entity.get_field(state)` reads from cached offset

### Low-level vs High-level Access

- **Low-level** (`cdef` nogil): Performance-critical loops, action validation
- **High-level** (cpdef/def): Python code, testing, debugging
- High-level wraps low-level (no code duplication)

## Key Modules

### GameState (`core/state.pyx`)

Central data structure: single contiguous float32 numpy array.

**Two-part layout:**
- **Visible state**: NN input (player-rotated so active player first)
- **Hidden state**: Internal bookkeeping (truncated before NN sees state)

**Hidden state purposes:**
- **Information hiding**: Data the model shouldn't see (deck order, canonical active player)
- **Bookkeeping**: Offer buffers for acquisition/closing phases
- **Performance**: Compact storage for O(1) access to one-hot values (phase, auction company, corp price indices, etc.)

**Sizes by player count:**
| Players | Visible | Hidden | Total |
|---------|---------|--------|-------|
| 2 | 1473 | 1184 | 2657 |
| 3 | 1559 | 1184 | 2743 |
| 6 | 1829 | 1184 | 3013 |

### Actions (`core/actions.pyx`)

Dynamic action space: `186 + (num_players * 20)` total actions.
- 3 players: 246 actions
- 6 players: 306 actions

**Action layout by phase:**
- INVEST: 1 pass + auction slots + 8 buy + 8 sell
- BID: 1 leave + 19 raise bid amounts
- ACQUISITION: 51 price offsets + 2 FI actions + 1 pass
- CLOSING: 1 close + 1 pass
- DIVIDENDS: 26 dividend amounts
- ISSUE: 1 issue + 1 pass
- IPO: 1 pass + 64 corp/par combinations

### Driver (`core/driver.pyx`)

Stateless game loop engine:
- `apply_action(state, action_idx, history)`: Main entry point
- Auto-applies forced actions when only 1 legal choice
- Returns: `STATUS_OK`, `STATUS_INVALID`, or `STATUS_GAME_OVER`

### Data (`core/data.pyx`)

Static game constants:
- 36 companies, 8 corporations, 27 market prices
- **Normalization divisors:**
  - `CASH_DIVISOR = 100.0` (prices, cash)
  - `SHARE_DIVISOR = 7.0` (share counts)
  - `STAR_DIVISOR = 20.0` (star ratings)
  - `MAX_ROUNDTRIPS = 2.0` (buy/sell tracking)

## MCTS Search

Pure-Python AlphaZero-style MCTS for 3-player games.

### Architecture

```
mcts/
├── node.py        # MCTSNode: visit_count, value_sum, prior, children dict
├── evaluator.py   # State rotation, NN inference, value un-rotation, terminal values
├── eval_cache.py  # Per-game NN eval cache (matrix-backed, MD5 hash index)
└── search.py      # PUCT selection, search loop, action probabilities, A0GB targets
```

MCTSConfig lives in `train/config.py` alongside TrainingConfig.

### State Rotation

The NN always sees the active player's data at slot 0. Before inference, the visible state is rotated so the active player's block comes first. After inference, the per-player value output is un-rotated back to canonical order.

**What gets rotated:**
- Player data blocks (contiguous, each `player_stride` floats)
- Per-player turn state fields: `auction_high_bidder`, `auction_starter`, `auction_passed`

**What does NOT get rotated:** phase, CoO, FI, companies, corporations, market, static data

**Important:** `GameState._layout` is a Cython `cdef` struct — NOT accessible from Python. Use `core.state.get_layout(num_players)` to get a Python-accessible `LayoutInfo` namedtuple with the same offsets (cached wrapper also available at `mcts.evaluator.get_layout()`).

### Value Representation

The value head outputs 3 scalars in [-1, 1] via tanh, representing per-player expected outcomes: `[v_active, v_next, v_next_next]`. These are un-rotated to canonical order via `np.roll(values, active_player_id)`.

**Terminal values:** Hybrid of rank-based and net-worth-ratio rewards, blended 50/50. The rank component (`linspace(+1, -1)` by placement) provides sharp signal at rank boundaries. The margin component (`2 * nw_i / max_nw - 1`) provides continuous gradient within ranks. Both are in [-1, +1], so the convex combination is always bounded. Winner always gets +1.0. All-zero net worths yield 0.0.

### PUCT Selection

```
UCB(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
```

Where Q(a) is the mean value for the **active player** at the parent node. This ensures each player maximizes their own expected outcome.

### A0GB Greedy Backup (Value Targets)

Instead of using the root node's mean value (soft-Z) or the game outcome as training targets, we use **A0GB** (Willemsen et al., "Value targets in off-policy AlphaZero: a new greedy backup", ALA 2020 / Neural Computing and Applications, 2022).

**Algorithm:** Starting from the root, follow the child with the highest visit count at each level until reaching a leaf (unexpanded node) or terminal. Return that node's value as the training target.

**Why A0GB:**
- At a leaf node visited once, the value equals V_NN (the neural network's evaluation)
- At a terminal node, the value equals the game outcome
- Removes exploration bias that contaminates soft-Z targets
- Converges faster than soft-Z or game-outcome targets in practice

**Implementation:** `get_greedy_leaf_value(root)` in `search.py`. Stops when the best child has `visit_count == 0` (the current node is the tree-edge leaf).

### Search Flow

1. **Root setup:** Evaluate root state with NN, expand, add Dirichlet noise to priors
2. **Per batch of simulations** (`search_batch_size` leaves per batch):
   - **Select:** Traverse tree using PUCT until reaching unexpanded or terminal node. Increment visit counts along the path and **lock** the selected leaf by setting its Q in the parent to -inf (preventing re-selection).
   - Terminal nodes have visits incremented and values backed up immediately.
   - **Batch evaluate:** All non-terminal leaves in the batch are evaluated in a single NN forward pass via `evaluate_batch()`.
   - **Unlock + Expand + Backup:** Restore parent Q, expand each leaf, propagate values up the tree. Visit counts were already incremented at selection time.
3. **Output:** `get_action_probabilities(root, temperature)` converts visit counts to policy target

**Leaf lock:** When a leaf is queued for batch evaluation, its Q in the parent edge is set to -inf so PUCT cannot re-select it. This is surgical — only the specific leaf edge is locked, not the entire ancestor path. Subsequent selections can still explore deep into the same subtree via different frontier nodes, avoiding the width bias of traditional virtual loss.

**Subtree reuse:** After choosing an action, the child's subtree is preserved as the root for the next move's search. `prepare_reuse_root(root, action, pool)` compacts the state pool and returns the child. `run_search(..., reuse_root=child)` then runs only `max(0, num_simulations - child.visit_count)` additional simulations. This saves 40-60% of GPU forward passes per game. Fresh Dirichlet noise is added to the reused root each time. Controlled by `--reuse-subtree-after-epoch`.

**Eval cache (alternative to subtree reuse):** When subtree reuse is disabled, `play_game()` creates a per-game `EvalCache` that caches NN evaluation results (policy + values) keyed by state hash (MD5). Each move starts with a fresh MCTS tree so Dirichlet noise is fully effective, but cached evals from prior moves avoid redundant GPU calls. Cache hits are resolved inline during MCTS selection — they don't consume batch slots, so the evaluator always gets full batches of cache misses. The cache is matrix-backed with a doubling growth strategy (~1.1KB per entry), cleared per game. Recovers ~70% of subtree reuse's throughput benefit while maintaining full exploration.

**When to use which:** Subtree reuse is faster but carries over visit counts that can drown out Dirichlet noise, causing the model to lock into narrow lines. The eval cache sacrifices some throughput for better exploration. To disable subtree reuse and activate the cache: `--reuse-subtree-after-epoch 9999`.

**Memory efficiency:** States are NOT stored in tree nodes. The root state is cloned and actions replayed to reach each leaf.

### NN Model (`nn/model_3p.py`)

Residual MLP (~25.4M parameters):
- **Input:** 1559 floats (3-player) (visible state, active player rotated to slot 0)
- **Trunk:** Linear → 10 residual blocks (pre-LN, GELU, expansion=2) → LayerNorm
- **Policy head:** Linear(768→256) → GELU → Linear(256→246) logits (masked by legal actions before softmax)
- **Value head:** Linear(768→384) → GELU → Linear(384→192) → GELU → Linear(192→3) → Tanh
- **Init:** Xavier uniform for all linear layers; residual block fc2 layers zero-initialized (blocks start as identity)

### Key APIs

```python
from train.config import MCTSConfig
from mcts.evaluator import NNEvaluator
from mcts.eval_cache import EvalCache
from mcts.search import run_search, get_action_probabilities, get_greedy_leaf_value, prepare_reuse_root

# Setup
evaluator = NNEvaluator(model, device, num_players=3)
config = MCTSConfig(num_simulations=800, c_puct=2.5, search_batch_size=4)  # c_puct set per-epoch via EpochConfig

# Search (batches 4 leaves per NN call → 200 inference calls instead of 800)
root = run_search(game_state, evaluator, config)
policy = get_action_probabilities(root, temperature=1.0, action_dim=config.action_dim)
value_target = get_greedy_leaf_value(root, num_players=config.num_players)

# Subtree reuse: reuse chosen child as next root (saves ~40-60% GPU evals)
reuse_root = prepare_reuse_root(root, chosen_action, state_pool)
root = run_search(next_state, evaluator, config, state_pool=state_pool, reuse_root=reuse_root)

# Eval cache (alternative to subtree reuse): fresh tree each move, cached NN evals
cache = EvalCache(config.action_dim, config.num_players)  # created once per game
root = run_search(game_state, evaluator, config, eval_cache=cache)
# cache persists across searches within the game; cleared per game via cache.clear()
```

## Self-Play Training

AlphaZero-style self-play training loop in `train/`.

### Multi-Process Self-Play Architecture

Self-play uses one or more centralized evaluation server threads for GPU throughput:

```
┌──────────────────────────────────────────────────────────┐
│  Main Process                                            │
│  - Owns model, device, replay buffer, trainer            │
│  - M EvaluationServer threads (batched GPU inference)    │
│  - Spawns N worker processes                             │
│  - Collects GameRecords, runs training                   │
└────────────────────┬─────────────────────────────────────┘
                     │ 1 shared Queue + N Events
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│  Worker 0  │ │  Worker 1  │ │  Worker K  │
│ play_game()│ │ play_game()│ │ play_game()│
│ RemoteEval │ │ RemoteEval │ │ RemoteEval │
└────────────┘ └────────────┘ └────────────┘
```

**Key design decisions:**
- **Workers are processes** (not threads) because MCTS is CPU-bound — GIL would serialize them
- **Evaluators are threads** in the main process — the GIL is released during CUDA kernels, and the model stays in one process
- **Multiple eval servers** (`--num-eval-servers M`) consume from a shared request queue, naturally double-buffering GPU access: one server gathers while another is on GPU
- **Plain `multiprocessing`** (not `torch.multiprocessing`) — avoids file descriptor exhaustion from shared tensor overhead on small arrays (~12KB per state)
- **`spawn` context** — avoids CUDA fork issues
- **Workers are daemon processes** — auto-killed on main process exit for clean Ctrl-C shutdown
- **`num_workers=0`** falls back to single-process sequential self-play (useful for debugging)

**Communication:** Workers write pre-rotated states into per-worker slots in `SharedEvalBuffers` (shared memory via `multiprocessing.RawArray`). A shared `multiprocessing.Queue` carries lightweight request tuples `(worker_idx, state_count)`. Each EvaluationServer thread blocks on `queue.get()`, then drains additional requests with `get_nowait()` to build larger batches. After inference, results are scattered to shared memory and per-worker `multiprocessing.Event` objects are set to signal completion. Uses pinned memory and pre-allocated GPU tensors for zero-alloc H2D/D2H transfers. With 24 workers each sending batch-8 requests, the GPU sees batches of up to ~192 states.

**Files:**
- `train/eval_server.py` — `EvaluationServer` (thread) + `RemoteEvaluator` (worker-side proxy with same interface as `NNEvaluator`)
- `train/self_play.py` — `play_game()` (takes an evaluator object) + `self_play_worker()` (worker process entry point)
- `train/main.py` — Orchestration: spawns workers, feeds game seeds via `mp.Queue`, collects `GameRecord`s

### Training Loop

Each epoch: (1) play N games via MCTS self-play → (2) store examples in replay buffer → (3) train NN on batched samples → (4) checkpoint and log.

```bash
# Run training (builds Cython extensions must be done first)
.venv/bin/python -m train

# With options
.venv/bin/python -m train --device cuda --games-per-epoch 100 --num-workers 4 --search-batch-size 8
.venv/bin/python -m train --config config.json --resume latest

# Single-process mode (for debugging)
.venv/bin/python -m train --num-workers 0 --games-per-epoch 10
```

### Training Configuration (`train/config.py`)

`TrainingConfig` dataclass holds all hyperparameters. Key defaults:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_simulations` | 800 | MCTS simulations per move |
| `search_batch_size` | 1 | Leaves per NN call (virtual loss batching) |
| `num_workers` | 4 | Self-play worker processes (0 = single-process) |
| `num_eval_servers` | 1 | Eval server threads (2 = double-buffer GPU) |
| `games_per_epoch` | 1000 | Self-play games per epoch |
| `learning_rate` | 1e-3 | AdamW, cosine decay to `lr_min` |
| `lr_min` | 1e-4 | Cosine schedule floor |
| `warmup_steps` | 1000 | Linear warmup from 0 to LR |
| `temp_anneal_start` | 60 | Move where temperature starts decreasing |
| `temp_anneal_end` | 120 | Move where temperature reaches `temp_final` |
| `temp_final` | 0.5 | Temperature floor after anneal |
| `c_puct_initial` | 3.5 | Starting c_puct (anneals to `c_puct_final`) |
| `c_puct_final` | 2.5 | Final c_puct after annealing |
| `c_puct_anneal_epochs` | 20 | Epochs over which c_puct anneals |
| `value_blend_start_epoch` | 10 | Epoch where A0GB blending begins |
| `value_blend_end_epoch` | 40 | Epoch where blend reaches pure A0GB |
| `reuse_subtree_after_epoch` | 15 | Subtree reuse disabled before this epoch |
| `buffer_capacity` | 500,000 | Replay buffer size (~4.2 GB) |
| `batch_size` | 256 | Training batch size |

`TrainingConfig.to_mcts_config()` creates an `MCTSConfig` from the relevant fields.

**Replay buffer memory** (3 players, 500K capacity): states ~3.3GB + masks ~470MB + policies ~470MB + values ~6MB = **~4.2 GB total**. Reduce `buffer_capacity` if memory is tight.

**Checkpointing:** The replay buffer is NOT checkpointed (too large). On resume, it starts empty and refills during self-play. This is standard AlphaZero practice.

### Training Examples

At each decision point during self-play, a `TrainingExample` is stored:
- **state**: Visible state rotated so active player is at slot 0 (shape `(1639,)` for 3 players)
- **legal_mask**: Binary mask of legal actions (shape `(246,)`)
- **policy_target**: MCTS visit probabilities (shape `(246,)`)
- **value_target**: A0GB values rotated to active-player-first (shape `(3,)`)

**Value target rotation:** `get_greedy_leaf_value()` returns canonical order `[p0, p1, p2]`. The NN outputs active-player-first `[active, next, next_next]`. Training targets are rotated to match: `np.roll(canonical_values, -active_player_id)`.

### Loss Functions

- **Policy**: Cross-entropy with MCTS targets: `-(pi * log_softmax(logits)).sum(-1).mean()`. Legal action mask is passed to the model so softmax only covers legal actions.
- **Value**: MSE between NN output and A0GB target.
- **Total**: `policy_loss_weight * policy_loss + value_loss_weight * value_loss`

### Key Training APIs

```python
from train.config import TrainingConfig
from train.self_play import play_game, GameRecord
from train.eval_server import EvaluationServer, RemoteEvaluator
from train.replay_buffer import ReplayBuffer, TrainingExample
from train.trainer import Trainer
from train.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from train.logging import TrainingLogger
from mcts.evaluator import NNEvaluator

# Single-process usage:
evaluator = NNEvaluator(model, device, num_players=3)
record = play_game(evaluator, config, game_seed=42, rng=rng)

# Multi-process: play_game is called inside self_play_worker with a RemoteEvaluator
```

## State Representation

**Normalization strategy:**
- Integers divided by divisor before storage
- Retrieved by multiplying back: `<int>(value * DIVISOR + 0.5)`

**One-hot encodings for:**
- Phase (11 values)
- Cost of ownership (7 levels)
- Turn order (per player)
- Price index (27 market spaces per corp)

**State layout:**
```
[Phase (11) | CoO Level (7) | Players (repeated) | FI (37) | Companies (108) |
 Company Incomes (36) | Market (27) | Corporations (872) | Turn (complex) |
 Auction Slot Info (5*num_players) | HIDDEN: Active Player, Deck, Offer Buffers]
```

**Player stride** = `73 + num_players` floats per player

**Corporation stride** = 109 floats per corp

See `VECTORS.md` for exact offsets.

## Game Flow & Phases

**11 Phases** (indices 0-10):
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
| 9 | IPO | Company → Corporation conversion |
| 10 | GAME_OVER | Terminal state |

**Automated phases** (no player input): WRAP_UP, INCOME, END_CARD

### Offer Buffer Pattern (acquisition.pyx, closing.pyx)

Both ACQUISITION and CLOSING phases use a **one-by-one offer presentation** pattern to keep the action space small. Instead of exposing all possible offers simultaneously (which would create a combinatorial explosion), offers are:

1. **Generated once** at phase entry into a hidden state buffer
2. **Sorted** by priority rules (varies by phase)
3. **Presented one at a time** to the active player
4. **Advanced sequentially** after each accept/pass decision
5. **Re-validated dynamically** since earlier decisions may invalidate later offers

**Why this pattern?**
- Action space stays constant regardless of game state complexity
- Model sees one decision at a time (cleaner learning signal)
- Priority ordering is deterministic (no hidden information)
- Buffer lives in hidden state (not visible to NN, but persists across actions)

#### ACQUISITION Phase Offers

**Priority order** (RULES-compliant):
1. **OS→FI**: OS corporation buys from Foreign Investor at face value (special ability), sorted by face value DESC
2. **Other Corp→FI**: By (share price DESC, face value DESC) - higher-valued corps and more expensive companies first
3. **Corp→Corp**: Same president controls buyer and seller, sorted by (buyer price DESC, face value DESC)
4. **Corp→Player**: President's corp buying their private companies, sorted by (buyer price DESC, face value DESC)

**Hidden buffer layout:**
```
[offer_count][offer_index][owner_type₀, corp_id₀, company_id₀][owner_type₁, corp_id₁, company_id₁]...
```

**Receivership handling**: Corps without a president (in receivership) have automated behavior:
- Auto-buy from FI at HIGH price if affordable (most expensive company first)
- Auto-pass on all other offers (receivership can only buy from FI)

**Actions**: 51 price offsets (low to high), FI_HIGH, FI_FACE (OS only), PASS

#### CLOSING Phase Offers

**Two-stage process:**
1. **Auto-close** (deterministic, no player input):
   - FI closes companies with negative adjusted income
   - Receivership corps close red/orange companies above CoO thresholds
2. **Offer-based close** (player decisions):
   - Only for player-owned and player-presided corps
   - Sorted by face value ascending (cheapest first)

**Hidden buffer layout:**
```
[offer_count][offer_index][owner_type₀, owner_id₀, company_id₀]...
```
Where `owner_type` is OWNER_PLAYER (0) or OWNER_CORP (1).

**Validation rules:**
- Company not already closed this phase
- Ownership unchanged since buffer generation
- Corp last-company rule: can't close if corp would have 0 companies

**Mandatory close** (after all offers processed): If a player would have negative income+cash, their cheapest negative-income private company is force-closed repeatedly until safe.

**Actions**: CLOSE, PASS

## Code Conventions

### Naming
- `corp_id`, `company_id`, `player_id` = indices (0-7, 0-35, 0-5)
- `CORPS[corp_id]`, `PLAYERS[player_id]` = global singletons
- `PHASE_*` constants from `GamePhases` enum

### Normalization
```cython
# Store: divide by divisor
state[offset] = cash / CASH_DIVISOR

# Retrieve: multiply and round
cash = <int>(state[offset] * CASH_DIVISOR + 0.5)
```

### One-hot patterns
```cython
# Find set bit
for i in range(N):
    if array[offset + i] == 1.0:
        return i

# Set bit
set_one_hot(array, offset, index, size)
```

### Pointer safety
- All `nogil` functions take pointers or direct offsets
- GameState provides `_player_ptr()`, `_corp_ptr()` helpers
- Entities compute offsets once and cache them

## Build Commands

**Python binary:** Always use `.venv/bin/python` (not `python` or `python3`). The venv may not be activated in the shell.

**Pyright:** Use `pyright` (system-installed at `/usr/bin/pyright`), NOT `.venv/bin/pyright`.

```bash
# Build Cython extensions (required before running any Python code)
# Pipe to grep to avoid 200+ lines of output consuming context
.venv/bin/python setup.py build_ext --inplace 2>&1 | grep -E "(warning|error)" || true

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_invest.py -v

# Clean build artifacts (.c, .so, .html, build/, *.egg-info)
.venv/bin/python setup.py clean

# Run self-play training (requires Cython build first)
.venv/bin/python -m train --device cuda --num-workers 4 --search-batch-size 8
.venv/bin/python -m train --device cuda --num-workers 24 --num-eval-servers 2  # double-buffer GPU
.venv/bin/python -m train --config config.json --resume latest
.venv/bin/python -m train --num-workers 0 --games-per-epoch 10  # single-process debug

# Run MCTS benchmark
.venv/bin/python setup.py benchmark --device=cuda
.venv/bin/python setup.py benchmark --device=cuda --batch-size=4
```

**Warning-free builds:** The build should produce no compiler warnings. If warnings appear, create a beads issue to fix them.

**Pyright errors:** When Pyright diagnostics appear after reading or editing a file, fix them before moving on. Unused imports, unused variables, and type errors should be resolved immediately rather than left for later. The auto-injected diagnostics can be stale after edits — run `pyright <file>` via Bash to get definitive results.

## Testing Approach

**Organization:** Tests in `tests/`, phase tests in `tests/phases/`

**Key fixtures** (conftest.py):
- `game_state`: 3-player initialized game (seed=42)
- `invest_state`, `bid_state`, `trade_state`: Pre-configured states
- `apply_and_track`: Helper for action history and invariants

**Test patterns:**
- Validate action effects on state
- Check mask validity after actions
- Verify phase transitions
- Assert game invariants (cash conservation, share counts)

**When a test fails, assume the implementation is broken** until proven otherwise. Investigate the root cause thoroughly before concluding that a test is wrong. Only "fix" a test to make it pass after confirming the implementation is correct and the test setup was invalid.

**Status codes:** STATUS_OK (0), STATUS_INVALID (1), STATUS_GAME_OVER (2)

### 18xx.games Replay Tests

Replay tests validate our engine against real completed games from [18xx.games](https://18xx.games). They replay every action from a game log through our engine and compare state at each action boundary against reference snapshots extracted from the Ruby 18xx.games engine.

**Requires Ruby** — the state extractor (`extract_states.rb`) runs inline as a subprocess during tests. No pre-generated state files are stored; reference states are extracted on-the-fly and held in memory.

**Architecture:**
- `extract_states.rb` — Loads game JSON into the Ruby 18xx engine (git submodule at `tests/18xx_games/18xx/`), replays all actions, and emits a JSON snapshot after each action
- `action_parser.py` — Maps 18xx action format (bid, par, sell_shares, etc.) to our engine's integer action indices, handling undo/redo, auto-actions, and program_* auto-pass flattening
- `replay_harness.py` — Drives the replay loop: initializes our engine with the 18xx deck order, replays actions phase-by-phase, and compares state against reference snapshots
- `test_replay.py` — Pytest parametrized entry point

**State compared at each action boundary:**
- **Players**: cash, net worth, companies owned, shares held
- **Corporations**: active/floated status, share price, cash, companies owned, shares in market
- **Foreign Investor**: cash, companies owned
- **Global**: offering (auction + revealed), deck size, cost of ownership level
- **Active entity**: active player (INVEST/BID/IPO) and active corp (DIVIDENDS/ISSUE), gated on phase alignment

**ACQ and CLOSING phases** use special adapters because our engine structures offers differently (sequential offer buffer vs. the 18xx engine's proposal model). These phases diff the before/after reference state to determine outcomes rather than mapping actions 1:1.

**Action space differences from 18xx.games:** Our engine intentionally restricts certain actions that 18xx.games allows, to simplify the model's decision space:
- **ACQUISITION**: 18xx.games allows player-owned corp-to-corp transfers via an offer/accept system. Our engine does not support these transfers. Game logs containing such offers will need the ACQ adapter to handle them (typically by passing).
- **CLOSING**: 18xx.games allows players to close companies with positive income. Our engine never offers this since there is no strategic reason to do so. Game logs where a player closes a positive-income company will fail replay.

When adding a new game, verify it doesn't rely on these unsupported actions. If a replay fails, check whether the game log contains corp-to-corp ACQ transfers or positive-income closures before investigating engine bugs.

**Adding a new game:**
1. Export the game JSON from 18xx.games (game data API or browser)
2. Save to `tests/18xx_games/data/<game_id>.json`
3. Add the game ID to the `@pytest.mark.parametrize` list in `test_replay.py`

```bash
# Run replay tests (requires Ruby)
pytest tests/18xx_games/test_replay.py -v
```

## Key Files by Task

| Task | Primary Files | Secondary Files |
|------|---------------|-----------------|
| Add game rule | `phases/*.pyx` | `core/data.pyx`, `RULES.md` |
| Modify action space | `core/actions.pyx` | `core/driver.pyx`, `phases/*.pyx` |
| Debug state | `core/state.pyx`, `VECTORS.md` | Entity files |
| Optimize performance | Any `.pyx` | Check compiler directives, nogil |
| Add phase | Create `phases/new.pyx` | `core/driver.pyx`, `core/actions.pyx` |
| Fix bug | Tests first | Phase/entity files |
| MCTS / search | `mcts/search.py`, `mcts/node.py` | `mcts/evaluator.py`, `mcts/eval_cache.py`, `train/config.py` |
| NN model | `nn/model_3p.py` | `mcts/evaluator.py` |
| Self-play / training | `train/main.py`, `train/config.py` | `train/self_play.py`, `train/eval_server.py`, `train/trainer.py` |

## Documentation

- **CLAUDE.md**: This file - agent instructions
- **RULES.md**: Complete game rules (24KB)
- **VECTORS.md**: State/action vector layouts with exact offsets
- **RSS.pdf**: Original board game rulebook

---

# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Understanding Game Rules

**Before working on any game logic**, read `RULES.md` to understand the correct behavior. This is the authoritative source for how the game should work. Common areas where rules matter:
- Share buying/selling price calculations
- Phase transitions and action ordering
- Dividend payments and constraints
- Acquisition and closing logic
- Receivership and bankruptcy handling

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Agent Work Standards

When working on any task, create beads issues (`bd create`) for:
- Bugs or problems discovered
- Rule discrepancies or ambiguities
- Missing functionality or gaps
- Performance concerns
- Code quality issues
- Anything that needs follow-up but is out of scope for current task

**No insights or work should be lost.** When in doubt, create an issue.

## Using Subtasks

When breaking down a feature into multiple steps, **use subtasks instead of independent issues**. Subtasks keep related work grouped and make progress easier to track.

**Subtask IDs** use dot notation: `parent.1`, `parent.2`, etc. Nesting is supported up to two levels (`parent.1.1`).

```bash
# Create subtasks for a feature (e.g., rss-az-cython2-abc)
bd create --parent=rss-az-cython2-abc --title="Create phase handler" --type=task
bd create --parent=rss-az-cython2-abc --title="Integrate into driver" --type=task
bd create --parent=rss-az-cython2-abc --title="Add tests" --type=task

# Results in: abc.1, abc.2, abc.3
```

**When to use subtasks:**
- Implementation steps for a feature (handler, integration, tests)
- Bug fix with multiple related changes
- Any work that logically belongs to a parent issue

**When to use independent issues:**
- Unrelated bugs discovered during work
- Cross-cutting concerns affecting multiple features
- Work that should be tracked separately for prioritization

## Ad-hoc Test Scripts

When writing ad-hoc Python scripts to test or debug code, **write them to the scratchpad directory** instead of inlining them in Bash commands. This is more token-efficient when iterating:

```bash
# Write script to scratchpad
Write scratchpad/test_something.py

# Run it (PYTHONPATH required since script runs outside the project directory)
PYTHONPATH=/home/icebreaker/rss-az-cython2 .venv/bin/python /path/to/scratchpad/test_something.py

# If it fails, use Edit to make small changes instead of rewriting
```

Benefits:
- Small edits are cheaper than rewriting 50+ line scripts inline
- Script persists for re-running after fixes
- Easier to read and debug

**PYTHONPATH:** Always prepend `PYTHONPATH=/home/icebreaker/rss-az-cython2` when running scratchpad scripts, since they live outside the project tree and won't find `core/`, `entities/`, `phases/` etc. without it.

The scratchpad path is provided in the system prompt at session start.

## Verification Before Closing

Before closing a task, run a **full clean rebuild** to catch stale artifacts:

```bash
.venv/bin/python setup.py clean && .venv/bin/python setup.py build_ext --inplace 2>&1 | grep -E "(warning|error)" || true
pytest tests/
```

A regular `build_ext --inplace` is incremental and may miss issues caused by changed `.pxd` headers or removed symbols. Always clean first when verifying.

**IMPORTANT:** Always run the full test suite (`pytest tests/`). Do NOT use `--ignore` to skip the 18xx replay tests or any other test directory. All tests must pass before closing a task.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
