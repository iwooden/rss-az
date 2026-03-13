# Self-Play Training Harness — Implementation Plan

## Overview

Standard AlphaZero self-play training loop for the RSS game engine. Each epoch:
1. Play N games via MCTS self-play, collecting training examples
2. Store examples in a replay buffer
3. Train the neural network on batched samples from the buffer
4. Checkpoint, log metrics, repeat

## Architecture

```
train/
├── __init__.py          # Package init, version
├── __main__.py          # python -m train entry point
├── config.py            # TrainingConfig dataclass
├── self_play.py         # Self-play game generation
├── replay_buffer.py     # Ring buffer for training examples
├── trainer.py           # Training loop + loss computation
├── checkpoint.py        # Save/load model + optimizer state
└── logging.py           # Tensorboard + CLI output
```

## Detailed Component Design

### 1. Configuration (`train/config.py`)

Single `TrainingConfig` dataclass with all hyperparameters. Should support
serialization to/from JSON for reproducibility (save config alongside checkpoints).

```python
@dataclass
class TrainingConfig:
    # --- Game ---
    num_players: int = 3

    # --- Self-Play ---
    games_per_epoch: int = 1000
    num_simulations: int = 800
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # --- Temperature Schedule ---
    # temp=1.0 for the first `temp_threshold` decision points (MCTS searches),
    # then drops to `temp_final`. Measured in total game decisions, not per-player.
    # With ~200 total moves per game, 60 gives ~30% exploration.
    temp_threshold: int = 60
    temp_initial: float = 1.0
    temp_final: float = 0.1

    # --- Replay Buffer ---
    buffer_capacity: int = 500_000    # ~5 epochs at 100K examples/epoch
    min_buffer_size: int = 10_000     # Min examples before training starts

    # --- Training ---
    batch_size: int = 256
    learning_rate: float = 2e-4       # AdamW
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0        # Gradient clipping
    training_steps_per_epoch: int = 1000

    # --- LR Schedule ---
    # Cosine annealing with linear warmup
    warmup_steps: int = 500
    lr_min: float = 1e-5              # Floor for cosine schedule

    # --- Loss ---
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5      # Every N epochs
    keep_last_n: int = 10             # Retain last N checkpoints

    # --- Logging ---
    tensorboard_dir: str = "runs"
    log_interval: int = 100           # Log every N training steps

    # --- Overall ---
    num_epochs: int = 100
    seed: int = 42                    # Master RNG seed
```

**Computed properties** (via `__post_init__` or properties):
- `action_dim = 186 + num_players * 20`
- `visible_size` from `core.state.get_layout(num_players).visible_size`

Should also have `to_json()` / `from_json()` for saving with checkpoints, and
a `to_mcts_config()` convenience method that returns a `MCTSConfig`.

### 2. Replay Buffer (`train/replay_buffer.py`)

Pre-allocated numpy ring buffer. Each training example stores:

| Field | Shape | Dtype | Notes |
|-------|-------|-------|-------|
| `states` | `(capacity, visible_size)` | float32 | Rotated visible state (active player at slot 0) |
| `legal_masks` | `(capacity, action_dim)` | float32 | Binary mask for legal actions |
| `policy_targets` | `(capacity, action_dim)` | float32 | MCTS visit probabilities |
| `value_targets` | `(capacity, num_players)` | float32 | A0GB values, rotated to active-player-first |

**Memory estimate** (3 players, 500K capacity):
- States: 500K * 3023 * 4B = ~5.7 GB
- Masks: 500K * 246 * 4B = ~470 MB
- Policies: 500K * 246 * 4B = ~470 MB
- Values: 500K * 3 * 4B = ~5.7 MB
- **Total: ~6.6 GB**

This is significant but acceptable for a GPU training machine. Can reduce
`buffer_capacity` if memory is tight.

```python
class ReplayBuffer:
    def __init__(self, capacity: int, visible_size: int, action_dim: int, num_players: int):
        """Pre-allocate numpy arrays."""

    def add_examples(self, examples: list[TrainingExample]) -> None:
        """Add batch of examples from a completed game."""

    def sample(self, batch_size: int, rng: np.random.Generator) -> TrainingBatch:
        """Random sample without replacement. Returns dict of tensors."""

    def __len__(self) -> int:
        """Current number of examples in buffer."""

    @property
    def is_ready(self) -> bool:
        """True if buffer has enough examples for training."""
```

`TrainingExample` is a lightweight NamedTuple:
```python
class TrainingExample(NamedTuple):
    state: np.ndarray          # (visible_size,)
    legal_mask: np.ndarray     # (action_dim,)
    policy_target: np.ndarray  # (action_dim,)
    value_target: np.ndarray   # (num_players,)
```

`TrainingBatch` is a dict of torch.Tensors, ready for the model.

### 3. Self-Play (`train/self_play.py`)

Plays a single game to completion, running MCTS at each decision point.

```python
@dataclass
class GameRecord:
    """Results from a single self-play game."""
    examples: list[TrainingExample]
    total_moves: int          # Decision points (MCTS searches)
    game_length: int          # Total actions including auto-forced
    winner_id: int            # Canonical player ID of 1st place (-1 if tie)
    net_worths: list[int]     # Final net worths per player
    duration_secs: float      # Wall-clock time

def play_game(
    model: torch.nn.Module,
    device: torch.device,
    config: TrainingConfig,
    game_seed: int,
    rng: np.random.Generator,
) -> GameRecord:
    """Play one self-play game, returning training examples."""
```

**Algorithm:**
1. `state = GameState(num_players); state.initialize_game(seed=game_seed)`
2. `evaluator = NNEvaluator(model, device, num_players)`
3. Loop until game over:
   a. `root = run_search(state, evaluator, mcts_config, rng)`
   b. `temp = temp_initial if move_count < temp_threshold else temp_final`
   c. `policy = get_action_probabilities(root, temp, action_dim)`
   d. `value_target = get_greedy_leaf_value(root, num_players)`
   e. Rotate state for storage: `rotated = rotate_visible_state(state._array, active_player, num_players)`
   f. Rotate value target to active-player-first: `rotated_value = np.roll(value_target, -active_player)`
   g. Store `TrainingExample(rotated, legal_mask, policy, rotated_value)`
   h. Sample action from policy, apply via `DRIVER.apply_action(state, action)`
   i. Increment `move_count`
4. Return `GameRecord` with all examples + game stats

**Key detail — value target rotation:**
- `get_greedy_leaf_value()` returns values in canonical order `[p0, p1, p2]`
- NN outputs in active-player-first order `[active, next, next_next]`
- Training target must match NN output order
- So: `value_target = np.roll(canonical_values, -active_player_id)`

### 4. Training (`train/trainer.py`)

```python
class Trainer:
    def __init__(self, model: RSSAlphaZeroNet, config: TrainingConfig, device: torch.device):
        """Set up optimizer, LR scheduler."""

    def train_step(self, batch: TrainingBatch) -> dict[str, float]:
        """Single training step. Returns loss dict."""

    @property
    def global_step(self) -> int:
        """Total training steps across all epochs."""
```

**Loss computation:**
```python
# Policy loss: cross-entropy with MCTS targets
# Use log_softmax on raw logits, then dot with target probs
log_probs = F.log_softmax(policy_logits, dim=-1)  # mask already applied in model
policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()

# Value loss: MSE
value_loss = F.mse_loss(value_preds, value_targets)

# Total
total_loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss
```

**Wait — masking in loss vs model:**
The model's `forward()` can optionally mask illegal actions. But for training,
we want to compute loss over ALL logits (the policy target is already zero for
illegal actions, so those terms vanish naturally). We should NOT mask in the
model during training — just pass `legal_action_mask=None` and apply masking
only in the loss via the target distribution (which is zero for illegal actions).

Actually, re-examining: the policy target from MCTS is a proper probability
distribution over legal actions only (sums to 1.0, zero elsewhere). If we use
cross-entropy `-(pi * log_softmax(logits))`, the illegal action terms are
`0 * log_softmax(logit_j)` which is 0, so they don't contribute to the loss.
However, the softmax denominator includes illegal action logits, which could
dilute the probability mass. It's better to mask illegal logits to `-inf`
before the softmax so the model learns to put zero probability there. So we
SHOULD pass the legal mask during training.

**Optimizer:** AdamW (standard for transformers and MLPs)

**LR Schedule:** Cosine annealing with linear warmup
```python
# Warmup: linear from 0 to lr over warmup_steps
# Cosine: decay from lr to lr_min over remaining steps
# Total steps = num_epochs * training_steps_per_epoch
```

**Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)`

### 5. Checkpointing (`train/checkpoint.py`)

```python
@dataclass
class CheckpointData:
    epoch: int
    global_step: int
    model_state_dict: dict
    optimizer_state_dict: dict
    scheduler_state_dict: dict
    config: TrainingConfig
    buffer_stats: dict  # size, total_examples_added
    metrics: dict       # Latest loss values, game stats

def save_checkpoint(path: Path, data: CheckpointData) -> None:
    """Save checkpoint to disk via torch.save."""

def load_checkpoint(path: Path, device: torch.device) -> CheckpointData:
    """Load checkpoint from disk."""

def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find most recent checkpoint in directory."""

def cleanup_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    """Remove old checkpoints, keeping the N most recent."""
```

**Checkpoint file naming:** `checkpoint_epoch_{epoch:04d}.pt`

**What we DON'T checkpoint:** The full replay buffer (too large, ~6GB). On resume,
the buffer starts empty and refills during self-play. This is standard practice —
AlphaZero doesn't persist the replay buffer either.

### 6. Logging (`train/logging.py`)

Two output channels: Tensorboard (detailed metrics) and CLI (progress + summaries).

**Tensorboard metrics:**
- `loss/total`, `loss/policy`, `loss/value` — per training step
- `lr` — learning rate per step
- `self_play/game_length_mean` — average decision points per game
- `self_play/duration_mean` — average seconds per game
- `self_play/examples_per_game` — training examples generated
- `buffer/size`, `buffer/utilization` — replay buffer stats
- `epoch` — current epoch number

**CLI output:**
```
Epoch 1/100
  Self-play: 1000 games | 98,432 examples | avg 98.4 moves/game | 8.2 sec/game
  Buffer: 98,432 / 500,000 (19.7%)
  Training: 1000 steps | policy=4.821 value=0.332 total=5.153 | lr=1.2e-4
  Checkpoint saved: checkpoints/checkpoint_epoch_0001.pt
  Epoch time: 2h 31m 14s
```

Use a simple logger class:
```python
class TrainingLogger:
    def __init__(self, tensorboard_dir: str, log_interval: int):
        """Initialize Tensorboard SummaryWriter and CLI formatting."""

    def log_training_step(self, step: int, losses: dict[str, float], lr: float) -> None:
        """Log a single training step (Tensorboard + periodic CLI)."""

    def log_self_play_epoch(self, epoch: int, records: list[GameRecord]) -> None:
        """Log self-play statistics for an epoch."""

    def log_epoch_summary(self, epoch: int, train_losses: dict, duration: float) -> None:
        """Print epoch summary to CLI."""

    def close(self) -> None:
        """Flush and close Tensorboard writer."""
```

### 7. Main Loop (`train/main.py`)

```python
def main(args: argparse.Namespace) -> None:
    """Main training entry point."""

    # 1. Load or create config
    config = TrainingConfig(...)  # from args or config file

    # 2. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RSSAlphaZeroNet(model_config).to(device)
    trainer = Trainer(model, config, device)
    buffer = ReplayBuffer(config.buffer_capacity, ...)
    logger = TrainingLogger(config.tensorboard_dir, config.log_interval)

    # 3. Optionally resume from checkpoint
    if resume_path:
        checkpoint = load_checkpoint(resume_path, device)
        model.load_state_dict(checkpoint.model_state_dict)
        trainer.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        trainer.scheduler.load_state_dict(checkpoint.scheduler_state_dict)
        start_epoch = checkpoint.epoch + 1

    # 4. Training loop
    for epoch in range(start_epoch, config.num_epochs):
        # Phase 1: Self-play
        model.eval()
        records = []
        for game_idx in range(config.games_per_epoch):
            record = play_game(model, device, config, game_seed=..., rng=rng)
            buffer.add_examples(record.examples)
            records.append(record)
            # Print progress periodically

        logger.log_self_play_epoch(epoch, records)

        # Phase 2: Training
        if buffer.is_ready:
            model.train()
            epoch_losses = defaultdict(float)
            for step in range(config.training_steps_per_epoch):
                batch = buffer.sample(config.batch_size, rng)
                losses = trainer.train_step(batch)
                logger.log_training_step(trainer.global_step, losses, trainer.lr)

            logger.log_epoch_summary(epoch, epoch_losses, ...)

        # Phase 3: Checkpoint
        if (epoch + 1) % config.checkpoint_interval == 0:
            save_checkpoint(...)
            cleanup_checkpoints(...)

    logger.close()
```

**CLI arguments:**
```
python -m train [OPTIONS]

Options:
  --config PATH          Load config from JSON file
  --resume PATH          Resume from checkpoint
  --device DEVICE        cuda / cpu (default: auto)
  --games-per-epoch N    Override games per epoch
  --num-epochs N         Override total epochs
  --checkpoint-dir PATH  Override checkpoint directory
  --tensorboard-dir PATH Override Tensorboard directory
  --seed N               Override master seed
```

### 8. `train/__main__.py`

Simple entry point:
```python
from train.main import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(...)
    # ... add args ...
    main(parser.parse_args())
```

## Hyperparameter Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_simulations=800` | Standard AlphaZero default | Balances quality vs speed |
| `c_puct=2.5` | Already tuned in MCTSConfig | |
| `dirichlet_alpha=0.3` | ~10/avg_legal_actions | RSS has ~10-30 legal actions |
| `temp_threshold=60` | ~30% of game | Explore early, exploit late |
| `batch_size=256` | Standard | Good GPU utilization |
| `learning_rate=2e-4` | Conservative AdamW | Stable training |
| `warmup_steps=500` | ~0.5 epochs | Prevents early instability |
| `training_steps=1000/epoch` | Match games_per_epoch | Each example seen ~2.5x |
| `buffer_capacity=500K` | ~5 epochs | Prevents overfitting to recent data |
| `max_grad_norm=1.0` | Standard | Prevents gradient explosions |

## Dependencies

Existing (already in project):
- `torch` — model, training
- `numpy` — data storage, MCTS

New:
- `tensorboard` — logging (standard PyTorch companion)

## Issue Breakdown

7 implementation issues, roughly in dependency order:

1. **Config** — standalone, no deps
2. **Replay Buffer** — depends on config for types/sizes
3. **Self-Play** — depends on config, uses existing MCTS/engine
4. **Trainer** — depends on config, model
5. **Checkpointing** — depends on config, trainer
6. **Logging** — depends on config
7. **Main Loop + CLI** — integrates everything

Issues 1, 2, 3, 4, 5, 6 can be developed somewhat in parallel (they share
the config interface but are otherwise independent). Issue 7 ties them together.
