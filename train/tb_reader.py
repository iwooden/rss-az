"""Read and merge Tensorboard event files from a log directory.

Provides a single entry point for loading all scalar data from a training
run, handling multiple event files (from restarts) and step deduplication.

Usage:
    from train.tb_reader import read_tb_scalars

    data = read_tb_scalars("runs/")
    for step, value in data["loss/total"]:
        ...
"""

from __future__ import annotations

import glob
import os

from tensorboard.backend.event_processing.event_accumulator import (  # pyright: ignore[reportMissingImports]
    EventAccumulator,
)


def read_tb_scalars(
    logdir: str = "runs",
) -> dict[str, list[tuple[int, float]]]:
    """Read all scalar data from Tensorboard event files in a directory.

    Merges data across multiple event files (from training restarts).
    When the same tag has data at the same step in multiple files,
    the later file's value wins (assumed to be the more recent run).

    Args:
        logdir: Directory containing event files.

    Returns:
        Dict mapping tag name to sorted list of (step, value) tuples.
    """
    files = sorted(
        glob.glob(os.path.join(logdir, "events.out.tfevents.*")),
        key=os.path.getmtime,
    )
    if not files:
        return {}

    merged: dict[str, dict[int, float]] = {}
    for f in files:
        ea = EventAccumulator(f)
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            if tag not in merged:
                merged[tag] = {}
            for s in ea.Scalars(tag):
                merged[tag][s.step] = s.value

    return {
        tag: sorted(step_vals.items())
        for tag, step_vals in merged.items()
    }


def latest_value(
    data: dict[str, list[tuple[int, float]]], tag: str,
) -> float | None:
    """Return the most recent value for a tag, or None if missing."""
    series = data.get(tag)
    if not series:
        return None
    return series[-1][1]


def values_at_epochs(
    data: dict[str, list[tuple[int, float]]], tag: str,
) -> list[tuple[int, float]]:
    """Return all (epoch, value) pairs for a tag. Alias for data[tag]."""
    return data.get(tag, [])


def sample_epochs(
    series: list[tuple[int, float]], max_rows: int = 20,
) -> list[tuple[int, float]]:
    """Sample a series down to max_rows representative points.

    Keeps the first 3, last 3, and evenly spaced points in between.
    Returns the full series if it has fewer than max_rows points.
    """
    n = len(series)
    if n <= max_rows:
        return list(series)

    # Always include first 3 and last 3
    head = 3
    tail = 3
    middle_budget = max_rows - head - tail
    if middle_budget <= 0:
        return list(series[:head]) + list(series[-tail:])

    middle = series[head : n - tail]
    step = max(1, len(middle) // middle_budget)
    sampled_middle = middle[::step][:middle_budget]

    result = list(series[:head]) + sampled_middle + list(series[-tail:])
    # Deduplicate (in case sampling overlaps with head/tail)
    seen: set[int] = set()
    deduped = []
    for item in result:
        if item[0] not in seen:
            seen.add(item[0])
            deduped.append(item)
    return deduped
