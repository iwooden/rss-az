"""Profiling statistics for self-play performance analysis.

Gated behind --profile flag; zero overhead when disabled.
All timing uses time.perf_counter() for sub-millisecond accuracy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class SearchStats:
    """Accumulated timing from run_search calls within a game."""

    selection_secs: float = 0.0
    eval_secs: float = 0.0
    backup_secs: float = 0.0
    num_searches: int = 0
    num_eval_batches: int = 0
    total_leaves: int = 0
    cache_hits: int = 0


@dataclass
class EvalClientStats:
    """Accumulated timing from RemoteEvaluator.evaluate_leaves within a game."""

    prepare_secs: float = 0.0
    wait_secs: float = 0.0
    result_secs: float = 0.0
    num_calls: int = 0
    total_states: int = 0


@dataclass
class GameProfileData:
    """Profile data from a single self-play game."""

    search: SearchStats
    eval_client: EvalClientStats | None = None
    game_duration: float = 0.0


class EvalServerStats:
    """Accumulated stats from eval server, reset per epoch."""

    __slots__ = (
        "batch_count", "batch_size_sum", "batch_size_min", "batch_size_max",
        "batch_size_hist", "inference_secs_sum", "inference_secs_min",
        "inference_secs_max", "idle_polls", "idle_secs", "total_states",
    )

    def __init__(self) -> None:
        self.batch_count: int = 0
        self.batch_size_sum: int = 0
        self.batch_size_min: int = 999999
        self.batch_size_max: int = 0
        self.batch_size_hist: dict[int, int] = defaultdict(int)
        self.inference_secs_sum: float = 0.0
        self.inference_secs_min: float = float("inf")
        self.inference_secs_max: float = 0.0
        self.idle_polls: int = 0
        self.idle_secs: float = 0.0
        self.total_states: int = 0

    def record_batch(self, batch_size: int, inference_secs: float) -> None:
        self.batch_count += 1
        self.batch_size_sum += batch_size
        self.total_states += batch_size
        if batch_size < self.batch_size_min:
            self.batch_size_min = batch_size
        if batch_size > self.batch_size_max:
            self.batch_size_max = batch_size
        self.batch_size_hist[batch_size] += 1
        self.inference_secs_sum += inference_secs
        if inference_secs < self.inference_secs_min:
            self.inference_secs_min = inference_secs
        if inference_secs > self.inference_secs_max:
            self.inference_secs_max = inference_secs

    def record_idle(self, idle_secs: float) -> None:
        self.idle_polls += 1
        self.idle_secs += idle_secs

    def reset(self) -> None:
        self.batch_count = 0
        self.batch_size_sum = 0
        self.batch_size_min = 999999
        self.batch_size_max = 0
        self.batch_size_hist = defaultdict(int)
        self.inference_secs_sum = 0.0
        self.inference_secs_min = float("inf")
        self.inference_secs_max = 0.0
        self.idle_polls = 0
        self.idle_secs = 0.0
        self.total_states = 0

    @classmethod
    def merge(cls, stats_list: list[EvalServerStats]) -> EvalServerStats:
        """Merge stats from multiple eval servers into one."""
        merged = cls()
        for s in stats_list:
            merged.batch_count += s.batch_count
            merged.batch_size_sum += s.batch_size_sum
            if s.batch_size_min < merged.batch_size_min:
                merged.batch_size_min = s.batch_size_min
            if s.batch_size_max > merged.batch_size_max:
                merged.batch_size_max = s.batch_size_max
            for size, count in s.batch_size_hist.items():
                merged.batch_size_hist[size] += count
            merged.inference_secs_sum += s.inference_secs_sum
            if s.inference_secs_min < merged.inference_secs_min:
                merged.inference_secs_min = s.inference_secs_min
            if s.inference_secs_max > merged.inference_secs_max:
                merged.inference_secs_max = s.inference_secs_max
            merged.idle_polls += s.idle_polls
            merged.idle_secs += s.idle_secs
            merged.total_states += s.total_states
        return merged


def _pct(value: float, total: float) -> str:
    return f"{value / total * 100:.0f}%" if total > 0 else "-%"


def format_epoch_profile(
    game_profiles: list[GameProfileData],
    server_stats: EvalServerStats | None,
    epoch_duration: float,
) -> str:
    """Format profile summary for epoch-end display."""
    lines: list[str] = []
    n = len(game_profiles)
    if n == 0:
        return "  Profile: no games"

    # Aggregate search stats (averages per game)
    sel = sum(g.search.selection_secs for g in game_profiles) / n
    evl = sum(g.search.eval_secs for g in game_profiles) / n
    bak = sum(g.search.backup_secs for g in game_profiles) / n
    total_search = sel + evl + bak
    searches = sum(g.search.num_searches for g in game_profiles) / n
    batches = sum(g.search.num_eval_batches for g in game_profiles) / n
    leaves = sum(g.search.total_leaves for g in game_profiles) / n

    cache_hits = sum(g.search.cache_hits for g in game_profiles) / n
    cache_misses = leaves - cache_hits
    hit_rate = cache_hits / leaves * 100 if leaves > 0 else 0

    lines.append(
        f"  Profile: Search avg/game: "
        f"select={sel:.3f}s({_pct(sel, total_search)}) "
        f"eval={evl:.3f}s({_pct(evl, total_search)}) "
        f"backup={bak:.3f}s({_pct(bak, total_search)}) "
        f"| {searches:.0f} searches, {batches:.0f} batches, {leaves:.0f} leaves"
    )
    lines.append(
        f"           Cache avg/game: "
        f"{cache_hits:.0f} hits, {cache_misses:.0f} misses "
        f"({hit_rate:.1f}% hit rate)"
    )

    # Aggregate eval client stats
    clients = [g.eval_client for g in game_profiles if g.eval_client is not None]
    if clients:
        nc = len(clients)
        prep = sum(c.prepare_secs for c in clients) / nc
        wt = sum(c.wait_secs for c in clients) / nc
        res = sum(c.result_secs for c in clients) / nc
        total_client = prep + wt + res
        lines.append(
            f"           EvalClient avg/game: "
            f"prepare={prep:.3f}s({_pct(prep, total_client)}) "
            f"wait={wt:.3f}s({_pct(wt, total_client)}) "
            f"result={res:.3f}s({_pct(res, total_client)})"
        )

    # Eval server stats
    if server_stats is not None and server_stats.batch_count > 0:
        s = server_stats
        avg_bs = s.batch_size_sum / s.batch_count
        avg_infer_ms = s.inference_secs_sum / s.batch_count * 1000
        throughput = s.total_states / epoch_duration if epoch_duration > 0 else 0

        # Compute p50 and p95 batch size from histogram
        sorted_sizes = sorted(s.batch_size_hist.items())
        cumsum = 0
        total = s.batch_count
        p50 = p95 = 0
        for size, count in sorted_sizes:
            cumsum += count
            if p50 == 0 and cumsum >= total * 0.5:
                p50 = size
            if p95 == 0 and cumsum >= total * 0.95:
                p95 = size
                break

        lines.append(
            f"           EvalServer: "
            f"batch avg={avg_bs:.1f} [{s.batch_size_min}-{s.batch_size_max}] "
            f"p50={p50} p95={p95} "
            f"| infer avg={avg_infer_ms:.2f}ms "
            f"[{s.inference_secs_min * 1000:.2f}-{s.inference_secs_max * 1000:.2f}ms] "
            f"| {throughput:.0f} states/s "
            f"| {s.idle_polls:,} idle polls ({s.idle_secs:.1f}s)"
        )

    return "\n".join(lines)
