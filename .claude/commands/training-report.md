Analyze the Tensorboard training logs and produce a structured assessment of how training is progressing.

## Instructions

Read the Tensorboard event files from the `runs/` directory using Python and the `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` API. Merge data across all event files in the directory (there may be multiple from training restarts).

Extract all scalar time series. The key tags are:

**Loss curves (per training step):**
- `loss/total`, `loss/policy`, `loss/value` — per-step losses logged every 100 steps
- `lr` — learning rate

**Loss curves (per epoch):**
- `epoch/total_loss_avg`, `epoch/policy_loss_avg`, `epoch/value_loss_avg`

**Self-play stats (per epoch):**
- `self_play/game_length_mean` — average decision points per game
- `self_play/duration_mean` — average wall-clock seconds per game
- `self_play/examples_per_game` — training examples generated per game
- `self_play/total_examples` — total examples generated in the epoch
- `self_play/net_worth_1st`, `self_play/net_worth_2nd`, `self_play/net_worth_3rd` — average net worth by final rank
- `self_play/net_worth_*_min`, `self_play/net_worth_*_max` — min/max net worth by rank

**Buffer stats (per epoch):**
- `buffer/size`, `buffer/utilization`

**Timing (per epoch):**
- `epoch/duration_secs`

**Profile stats (per epoch, only present if `--profile` was used):**
- `profile/search_*` — MCTS search time breakdown
- `profile/eval_client_*` — worker-side eval timing
- `profile/server_*` — GPU eval server stats

Use the Bash tool to run a Python script that extracts the data. Example pattern:

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, os

files = sorted(glob.glob('runs/events.out.tfevents.*'), key=os.path.getmtime)
merged = {}  # tag -> [(step, value), ...]
for f in files:
    ea = EventAccumulator(f)
    ea.Reload()
    for tag in ea.Tags().get('scalars', []):
        if tag not in merged:
            merged[tag] = []
        merged[tag].extend((s.step, s.value) for s in ea.Scalars(tag))
# Deduplicate by step (later files win)
for tag in merged:
    by_step = dict(merged[tag])
    merged[tag] = sorted(by_step.items())
```

## Analysis to produce

### 1. Training Overview

| Metric | Value |
|--------|-------|
| Epochs completed | N |
| Total training steps | N |
| Current learning rate | X |
| Buffer size / capacity | N / M (X%) |
| Avg epoch duration | Xm Ys |
| Avg game length | N moves |
| Avg game duration | Xs |

### 2. Loss Curves

Analyze the trajectory of policy and value losses:

- **Overall trend**: Are losses decreasing? Plateauing? Increasing (bad)?
- **Recent trend**: Compare the last 5 epochs to the preceding 5. Is learning still progressing or stalling?
- **Policy vs value balance**: Is one loss dominating? Are they decreasing at similar rates? A value loss that plateaus while policy loss keeps dropping (or vice versa) may indicate an imbalance.
- **Per-step noise**: How noisy are the per-step losses? High variance might suggest the batch size or learning rate needs adjustment.

Present the epoch-level loss values as a compact table showing the trend:

```
Epoch  Total   Policy  Value   LR
  1    X.XXX   X.XXX   X.XXX   X.Xe-X
  5    X.XXX   X.XXX   X.XXX   X.Xe-X
 10    ...
```

Show every epoch if there are fewer than 20, otherwise sample ~15-20 representative epochs (first few, evenly spaced middle, last few).

### 3. Self-Play Quality

Analyze how game quality evolves as the model improves:

- **Game length trend**: Are games getting longer (model learning to play more moves before game over)? Shorter? Stable? Early in training, random play produces short games; better play should produce longer, more strategic games — up to a point.
- **Net worth trends**: Are 1st-place net worths increasing over epochs? Is the spread between 1st and 3rd widening (model learning to differentiate skill) or narrowing?
- **Net worth ranges**: Are the min/max ranges tightening (more consistent play) or staying wide?

Present as a compact table:

```
Epoch  AvgMoves  1st$    2nd$    3rd$    Spread
  1      XX     $XXX    $XXX    $XXX     $XXX
  5      XX     $XXX    $XXX    $XXX     $XXX
 ...
```

### 4. Throughput & Efficiency

- **Games/minute**: Derive from `epoch/duration_secs` and games_per_epoch (from config or total_examples/examples_per_game).
- **Epoch time trend**: Is epoch duration increasing (expected as games get longer)?
- **Buffer fill rate**: How quickly is the buffer filling? Is it full yet?

If profile stats are present, also analyze:
- **GPU utilization**: What fraction of search time is spent waiting for GPU eval? What are the batch sizes?
- **Throughput**: States evaluated per second.
- **Bottleneck identification**: Is the system CPU-bound (low GPU utilization) or GPU-bound (high eval wait)?

### 5. Training Health Assessment

Based on the data, give a brief overall assessment:
- **Learning status**: Is the model actively learning, plateauing, or showing signs of divergence?
- **Key concerns**: Any red flags? (Loss spikes, value loss not decreasing, game lengths stuck, etc.)
- **Recommendations**: Any obvious hyperparameter adjustments suggested by the data? (LR too high/low, buffer too small, etc.)

## Output format

Use markdown with the section headers above. Be data-driven — cite specific numbers, epochs, and trends rather than vague statements. The user is an experienced ML practitioner who understands AlphaZero; don't explain basic concepts.
