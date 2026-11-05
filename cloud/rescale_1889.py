"""The epoch-20 rescale of a 1889 run (1889/model-design.md §7.1); run from the repo root.

Reads the run's TensorBoard logs and prints the config fields to change, with
their old and new values, and the resume procedure. It writes nothing.

From the means over the last ``--window`` epochs logged through ``--epoch``:

- the temperature windows start at 17.5% of ``self_play_<n>p/game_length_mean``, to
  the nearest 5 moves, and end at twice that, 35%, as ``train_configs/1889-v0.json``'s;
- ``buffer_capacity`` is 3.3 epochs of ``self_play_aggregate/total_examples``, and
  ``training_steps_per_epoch`` trains 1.1 times an epoch's samples, each to 3
  significant figures.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path.cwd()))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

WINDOW_START_SHARE = 0.175
WINDOW_END_OVER_START = 2  # 35% of the game
BUFFER_EPOCHS = 3.3
SAMPLES_TRAINED_PER_EPOCH = 1.1
WINDOW_ROUNDING = 5

SAMPLES_TAG = "self_play_aggregate/total_examples"


def _fail(message: str) -> NoReturn:
    sys.exit(f"rescale_1889: {message}")


def _significant(value: float, digits: int = 3) -> int:
    """``value`` as an integer rounded to ``digits`` significant figures."""
    return max(1, round(float(f"{value:.{digits}g}")))


def measure(
    scalars: dict[str, list[tuple[int, float]]],
    player_counts: range,
    *,
    epoch: int | None,
    window: int,
) -> tuple[list[int], dict[int, float], float]:
    """The epochs read, the mean game length by player count, and the mean
    samples per epoch."""
    tags = [SAMPLES_TAG, *(f"self_play_{n}p/game_length_mean" for n in player_counts)]
    missing = [tag for tag in tags if not scalars.get(tag)]
    if missing:
        _fail(f"the logs have no {', '.join(missing)}: is this a run's log directory?")
    last = epoch if epoch is not None else scalars[SAMPLES_TAG][-1][0]
    epochs = [step for step, _ in scalars[SAMPLES_TAG] if step <= last][-window:]
    if not epochs:
        _fail(f"the logs have no epochs through {last}")

    def mean_over_epochs(tag: str) -> float:
        by_epoch = dict(scalars[tag])
        found = [by_epoch[e] for e in epochs if e in by_epoch]
        if len(found) != len(epochs):
            _fail(f"{tag} has only {len(found)} of epochs {epochs[0]}-{epochs[-1]}")
        return sum(found) / len(found)

    lengths = {n: mean_over_epochs(f"self_play_{n}p/game_length_mean") for n in player_counts}
    return epochs, lengths, mean_over_epochs(SAMPLES_TAG)


def rescale(lengths: dict[int, float], samples: float, batch_size: int) -> dict[str, object]:
    """Section 7.1's rules: the new config fields, players in ascending order."""
    ordered = [lengths[n] for n in sorted(lengths)]
    starts = [WINDOW_ROUNDING * round(WINDOW_START_SHARE * x / WINDOW_ROUNDING) for x in ordered]
    ends = [WINDOW_END_OVER_START * start for start in starts]
    return {
        "temp_anneal_starts": starts,
        "temp_anneal_ends": ends,
        # Inert while the target temperature is flat; the config keeps them equal.
        "policy_target_temp_anneal_starts": starts,
        "policy_target_temp_anneal_ends": ends,
        "buffer_capacity": _significant(BUFFER_EPOCHS * samples),
        "training_steps_per_epoch": _significant(SAMPLES_TRAINED_PER_EPOCH * samples / batch_size),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="train_configs/1889-v0.json", help="The run's config file")
    parser.add_argument("--logdir", help="The run's TensorBoard directory (default: the config's tensorboard_dir)")
    parser.add_argument("--epoch", type=int, help="The last epoch to read, 1-indexed (default: the latest logged)")
    parser.add_argument("--window", type=int, default=5, help="Epochs to average (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.window < 1:
        parser.error("--window must be positive")

    config_path = Path(args.config)
    if not config_path.exists():
        _fail(f"no config file {config_path}; run from the repo root, or pass --config")
    try:
        from train.config import TrainingConfig
        from train.tb_reader import read_tb_scalars
    except ImportError as exc:
        _fail(f"cannot import {exc.name}; activate the venv (install.sh installs it)")
    config = TrainingConfig.from_json(config_path.read_text())
    logdir = args.logdir or config.tensorboard_dir
    scalars = read_tb_scalars(logdir)
    if not scalars:
        _fail(f"no TensorBoard events in {logdir}; pass --logdir")

    epochs, lengths, samples = measure(scalars, config.iter_player_counts(), epoch=args.epoch, window=args.window)
    new_fields = rescale(lengths, samples, config.batch_size)
    try:
        new_config = dataclasses.replace(config, **new_fields)  # type: ignore[arg-type]
    except ValueError as exc:
        _fail(f"the new values don't make a valid config: {exc}")

    print(f"{config.game} rescale from {logdir}: epochs {epochs[0]}-{epochs[-1]} ({len(epochs)} averaged)")
    if epochs[-1] < 20:
        print(f"  Note: Section 7.1 reads epoch 20; these logs reach {epochs[-1]}.")
    print("  Game length: " + ", ".join(f"{n}p {lengths[n]:,.1f}" for n in sorted(lengths)))
    print(f"  Samples per epoch: {samples:,.0f}")
    print()
    rows = [(name, getattr(config, name), new) for name, new in new_fields.items()]
    changed = {name: new for name, old, new in rows if old != new}
    width = max(len(name) for name, _, _ in rows)
    old_width = max(len(json.dumps(old)) for _, old, _ in rows)
    print(f"  {'Field':<{width}}  {'Old':<{old_width}}  New")
    for name, old, new in rows:
        mark = "" if old != new else "  (unchanged)"
        print(f"  {name:<{width}}  {json.dumps(old):<{old_width}}  {json.dumps(new)}{mark}")
    print(
        f"  The LR schedule's span follows: {config.lr_decay_end_epoch} x steps = "
        f"{(config.lr_decay_end_epoch or config.num_epochs) * config.training_steps_per_epoch:,} -> "
        f"{(new_config.lr_decay_end_epoch or new_config.num_epochs) * new_config.training_steps_per_epoch:,} "
        "optimizer steps."
    )
    if not changed:
        print("\nNothing to change.")
        return
    print(f"\nTo resume, stop the run (q, then Enter, in its terminal) and set in {config_path}:\n")
    print(",\n".join(f"  {json.dumps(name)}: {json.dumps(value)}" for name, value in changed.items()))
    print(
        f"\nthen run:\n\n  python -m train --config {config_path} --resume latest\n\n"
        "A resume takes the checkpoint's config, then this file over it, then CLI flags over both,\n"
        "so edit the file: on a later resume an unedited file would undo flags given once.\n"
        "The banner shows the new values, and a changed capacity prints the rows it kept."
    )


if __name__ == "__main__":
    main()
