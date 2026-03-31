# Rolling Stock Stars — AlphaZero Engine

A high-performance game engine and AlphaZero training pipeline for Rolling Stock Stars, a 2–6 player financial strategy card game. The engine is written in Cython for zero-overhead self-play, with game state stored as a single contiguous float32 array that passes directly to PyTorch without serialization.

## What is Rolling Stock Stars?

Players act as investors: buying companies at auction, converting them into corporations via IPO, trading shares, and managing acquisitions and dividends. The player with the most wealth (cash + company face values + share values) at game end wins. The game features 36 companies across 5 tiers, 8 corporations with unique abilities, and a 27-step share price track.

> **Note:** Rolling Stock Stars is a distinct game from "Rolling Stock." Many rules differ. See [`RULES.md`](RULES.md) for the complete rules specification.

## Architecture

```
rss-az-cython2/
├── core/        # Cython engine: state array, action dispatch, driver loop, static data
├── entities/    # Entity handles: Player, Corporation, Company, Deck, Market, etc.
├── phases/      # Phase handlers: invest, bid, acquisition, closing, dividends, ...
├── mcts/        # AlphaZero MCTS: batched search, PUCT selection, subtree reuse
├── nn/          # Neural network: residual MLP (~2.3M params), policy + value heads
├── train/       # Self-play training loop: workers, eval servers, replay buffer
│   └── gpu/     # Vendor-specific GPU optimizations (NVIDIA / AMD auto-detected)
├── interp/      # Interpretability analysis (probing, conductance, SVD)
├── tests/       # Unit tests + 18xx.games replay integration tests
└── benchmarks/  # MCTS performance benchmarks
```

~16K lines of Cython/Python across the engine, search, and training code.

### Game Engine

The entire game state lives in a single `float32` numpy array (2285–2641 floats depending on player count). This flat representation eliminates Python object overhead and enables `nogil` execution in Cython hot paths.

**State layout:** `[Phase | CoO | Players | FI | Companies | Incomes | Market | Corps | Turn | Auction | Hidden]`

- **Visible region** — fed to the neural network (player-rotated so the active player is always slot 0)
- **Hidden region** — internal bookkeeping (deck order, offer buffers, canonical indices), truncated before inference

The stateless driver (`core/driver.pyx`) dispatches actions to 12 phase handlers, auto-applies forced moves, and returns status codes. See [`VECTORS.md`](VECTORS.md) for the full state/action vector specification.

### Neural Network

Residual MLP (~2.3M parameters), architecture guided by interpretability analysis:

| Component | Design |
|-----------|--------|
| Input | 2-layer preprocessing (input → 768 → 256) |
| Trunk | 8 residual blocks, 256-dim, pre-LayerNorm, GELU |
| Policy head | 3 hidden layers → action logits |
| Value head | 1 hidden layer → 3 per-player values (tanh) |

### Self-Play Training

AlphaZero loop: play N games via MCTS → store in replay buffer → train → checkpoint.

- Multi-process architecture: CPU-bound MCTS workers + GPU eval server processes
- Zero-copy model sharing via CUDA IPC, shared-memory state buffers
- Lockfree worker-server IPC via GCC atomic uint64 bitmaps (one bit per worker, O(1) batch claim via exchange-to-zero)
- Graceful shutdown: `q + Enter` drains workers and saves state

## Beyond Standard AlphaZero

This implementation diverges from vanilla AlphaZero in several ways — motivated by the multiplayer setting, by training stability, and by squeezing more signal out of each MCTS simulation.

### Batched MCTS with Leaf Locking

Standard AlphaZero runs one tree traversal per simulation, evaluating a single leaf at a time. We collect up to `search_batch_size` leaves per GPU call, amortizing kernel launch overhead.

The challenge: if two traversals in the same batch reach the same leaf, the second is wasted. We solve this with a **leaf-lock mechanism**: when a leaf is queued for evaluation, its parent edge's Q value is overwritten with `-inf`, making PUCT never select it again within the batch. The original Q is saved and restored after the NN returns.

This lock **propagates upward**: if all children of a node are locked, the node itself is locked at its parent, recursively up to the root. When every root edge is locked, the batch is submitted early. The effect is that batch filling never wastes traversals into fully-explored subtrees.

Visit counts are also incremented at selection time (before evaluation), gently nudging subsequent in-batch traversals toward less-explored branches.

### Subtree Reuse with Correct Dirichlet Noise

After each real game move, the chosen child's subtree becomes the new root — saving 40-60% of GPU evaluations. The tricky part: Dirichlet exploration noise must have real influence at the new root, but the reused child already has hundreds of accumulated visits that would drown out any noise in the PUCT formula.

Our solution: **reset the root's per-action visit counts to zero while preserving the children's subtrees intact**. Fresh Dirichlet noise is mixed into the root priors, and a **virtual backup catch-up phase** replays each child's mean Q value back into the root one visit at a time until the root's per-action counts match. This gives the noise full influence over early selections while smoothly incorporating the reused statistics — no tree information is lost, and exploration isn't suppressed.

The state pool itself is compacted in-place (retained nodes sorted ascending by index, `memcpy`'d to the front) so there is no memory fragmentation across reuses.

### A0GB Greedy Backup (Value Targets)

Standard AlphaZero uses the game outcome as the value target for every position in the game. This is high-variance: early moves are trained against a signal that depends on 100+ subsequent decisions.

We use **A0GB** (Willemsen et al., 2022): from the root, follow the max-visit child at each level until reaching a node whose best child has zero visits — that node's neural network value estimate becomes the training target. This produces a lower-variance, forward-looking target that converges faster than game outcomes.

### Blended Terminal Rewards

In a 3-player game, "who won" is less informative than "by how much." Our terminal signal blends two components (controlled by `--terminal-blend`, default 0.5):

- **Rank-based:** evenly spaced values in [-1, +1] by final placement (e.g., 1st/2nd/3rd → +1/0/-1), with ties averaged
- **Margin-based:** zero-sum net-worth deviation, scaled by `n/(n-1)` to guarantee the output stays in [-1, +1] regardless of wealth distribution

The blend gives the value head both ordinal (who won) and cardinal (by how much) signal to learn from.

### Value Target Annealing

Early in training the neural network is unreliable, so A0GB targets would just be noise training on noise. We linearly anneal from pure game-outcome targets to pure A0GB targets over epochs 10-40:

```
value_target = alpha * a0gb_target + (1 - alpha) * game_outcome
```

This bootstraps the network on reliable (if lagged) outcomes before transitioning to the lower-variance MCTS-derived signal.

### Training Schedules

Several hyperparameters are annealed rather than fixed:

| Parameter | Schedule | Rationale |
|-----------|----------|-----------|
| Temperature | 1.0 → 0.5 over moves 60-120 | Broad exploration early in each game, sharper play in endgame |
| c_puct | 3.5 → 2.5 over first 20 epochs | Explore broadly when the policy head is uninformative, trust it more as training progresses |
| Value blend | 0% → 100% A0GB over epochs 10-40 | Stable outcome targets first, then lower-variance MCTS targets |
| Learning rate | Cosine decay with 1K-step warmup | Standard, but the warmup matters when replaying from buffer |

### Multiplayer Value Representation

Standard AlphaZero stores a single scalar value per node (the game is zero-sum between two players). With 3 players, we store a full `(num_players,)` value vector at every node in canonical player order. PUCT indexes into the active player's component for Q. The value head outputs 3 tanh scalars — one per player's expected outcome — rather than a single win probability.

### Interpretability-Driven Architecture

The neural network architecture was shaped by probing and conductance analysis on earlier checkpoints. The [`interp/`](interp/README.md) directory contains a full analysis toolkit; example reports from epoch 375 are checked in under [`interp/examples/`](interp/examples/):

| Report | What it shows |
|--------|---------------|
| [Sensitivity](interp/examples/sensitivity_epoch375.html) | Feature ablation — which input groups the policy and value heads rely on, broken down by game phase |
| [Probing](interp/examples/probing_epoch375.html) | Linear probes at each trunk layer — where game concepts (who's winning, action type, value estimates) become linearly decodable |
| [Architecture](interp/examples/arch_epoch375.html) | Residual block contribution norms, effective rank (SVD), and trunk/head conductance — is depth or width the bottleneck? |
| [Decision Attribution](interp/examples/decisions_epoch375.html) | Integrated Gradients on high-uncertainty states — which features tip the model between its top-2 candidate actions |
| [Action-Conditioned IG](interp/examples/acig_epoch375.html) | Feature importance grouped by chosen action type — reveals the conditional logic within each phase |
| [Preprocessing](interp/examples/preprocess_epoch375.html) | Signal attenuation through the 768→256 input compression, expanded probing, and SVD projection analysis |
| [Policy Head](interp/examples/policy_head_epoch375.html) | Logit lens (when decisions crystallize), neuron specialization by action type, and layer causal necessity |
| [Value Head](interp/examples/value_head_epoch375.html) | Value lens (trunk-only vs full head), per-player neuron specialization, and phase-stratified value characteristics |
| [Normalization](interp/examples/norm_epoch375.html) | Per-feature statistics — out-of-range values, sparsity, and distribution health across the state vector |

Key architectural decisions that came from this analysis:

- SVD showed expanded residual block widths at only 13-16% utilization — we removed the inner expansion factor, halving trunk parameters with no representational loss
- Conductance revealed the policy head bottleneck was depth (not enough nonlinear transforms), not width — so we gave it 3 hidden layers instead of 1
- The value head was already nearly linear through the trunk (R^2 = 0.97 for a linear probe), so it only needs 1 hidden layer
- Input preprocessing uses a 2-layer projection (input → 3x hidden → hidden) to avoid bottlenecking the initial transform

## Getting Started

### Prerequisites

- Python 3.10+
- C compiler (gcc/clang)
- Ruby (only for 18xx.games replay tests)

### Installation

```bash
git clone https://github.com/<you>/rss-az-cython2.git
cd rss-az-cython2
python -m venv .venv
source .venv/bin/activate

# Install dependencies (picks correct PyTorch build)
./install.sh          # CPU-only
./install.sh cuda     # CUDA 12.4
./install.sh rocm     # ROCm 6.2
```

The install script handles PyTorch platform selection, installs all dependencies, and builds the Cython extensions.

### Build

```bash
# Rebuild Cython extensions after changing .pyx files
python setup.py build_ext --inplace

# Clean build artifacts (do this before full rebuilds — incremental builds miss .pxd changes)
python setup.py clean
```

### Run Tests

```bash
pytest tests/
```

### Train

```bash
# Full training run
python -m train --device cuda --num-workers 4 --search-batch-size 8

# Single-process debug mode
python -m train --num-workers 0 --games-per-epoch 10

# Resume from latest checkpoint
python -m train --resume latest
```

Graceful shutdown: press `q + Enter` to drain workers and save checkpoint + replay buffer. `Ctrl-C` for hard exit.

### Benchmark

```bash
python setup.py benchmark --device=cuda --batch-size=4
```

### Trace a Game

```bash
python setup.py trace_game
```

Plays a random game and prints a human-readable move-by-move trace.

## Key Documentation

| File | Description |
|------|-------------|
| [`RULES.md`](RULES.md) | Complete game rules (authoritative) |
| [`VECTORS.md`](VECTORS.md) | State and action vector layouts with offsets and normalization |
| [`interp/README.md`](interp/README.md) | Interpretability analysis pipeline |
| [`tests/games_18xx/README.md`](tests/games_18xx/README.md) | Integration tests against 18xx.games replays |

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy >= 2.0 | State arrays |
| Cython >= 3.0 | Engine compilation |
| torch >= 2.0 | Neural network + training |
| tensorboard >= 2.0 | Training metrics |
| rich >= 13.0 | Terminal UI |
| pytest >= 8.0 | Testing |
| matplotlib, scikit-learn, captum | Interpretability |
