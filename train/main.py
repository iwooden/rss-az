"""Main training orchestration loop."""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.multiprocessing as mp
from torch._dynamo.decorators import mark_unbacked

from games import Game, StartRows, get_game
from nn import create_model, get_model_input_spec
from nn.model_contract import FLAT_RELATION_RECORD_WIDTH
from train.checkpoint import (
    Checkpoint,
    cleanup_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from train.config import POLICY_TARGET_PRUNING_MODES, TrainingConfig
from train.eval_server import (
    EvaluationServer,
    SharedEvalBuffers,
    get_worker_result,
)
from train.logging import TrainingLogger
from train.profile_stats import STAGES, EvalServerStats, GameProfileData, format_epoch_profile
from train.policy_metrics import PolicyMetrics
from train.replay_buffer import FLAG_FROM_POSITION, ReplayBuffer
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool
from train.self_play import (
    build_epoch_player_count_schedule,
    build_epoch_start_schedule,
    play_game,
    self_play_worker,
)
from train.trainer import Trainer, average_training_metrics, summarize_scores


def _parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list for per-player-count schedules."""
    try:
        parts = [part.strip() for part in value.split(",")]
        return [int(part) for part in parts if part]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from exc


def _parse_str_list(value: str) -> list[str]:
    """Parse a comma-separated string list."""
    parts = [part.strip() for part in value.split(",")]
    values = [part for part in parts if part]
    if not values:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated non-empty strings, got {value!r}"
        )
    return values


def _parse_positive_int(value: str) -> int:
    """Parse an integer of at least 1."""
    try:
        parsed = int(value)
    except ValueError:
        parsed = 0
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AlphaZero self-play training"
    )
    parser.add_argument("--config", type=str, help="Load config from JSON file")
    parser.add_argument(
        "--resume",
        type=str,
        help='Resume from checkpoint file, or "latest" to auto-find',
    )
    parser.add_argument("--device", type=str, help="Force device: cuda, cpu")
    parser.add_argument("--num-players", type=int,
                        help="Number of players (default: 3)")
    parser.add_argument(
        "--v3-behavior", action=argparse.BooleanOptionalAction, default=None,
        help="Enable v3 engine behavior, independent of model input layout (RSS only)",
    )
    parser.add_argument(
        "--acq-same-president", action=argparse.BooleanOptionalAction, default=None,
        help="Restrict acquisition to companies controlled by the buyer's president (RSS only)",
    )
    parser.add_argument(
        "--min-players",
        type=int,
        help="Minimum player count for mixed player-count training",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        help="Maximum player count for mixed player-count training",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["transformer"],
        help="Model family to instantiate",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Model module or Python file, e.g. nn/transformer-v2.py; required "
             "unless --config or the resumed checkpoint sets model_path",
    )
    parser.add_argument("--d-model", type=int, help="Transformer trunk width")
    parser.add_argument("--d-proj", type=int, help="Transformer policy projection width")
    parser.add_argument("--num-heads", type=int, help="Transformer attention head count")
    parser.add_argument("--num-layers", type=int, help="Transformer block count")
    parser.add_argument("--ff-mult", type=float, help="Transformer SwiGLU FFN multiplier")
    parser.add_argument("--games-per-epoch", type=int)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--training-steps-per-epoch", type=int)
    parser.add_argument("--num-simulations", type=int)
    parser.add_argument("--mcts-sims-start", type=int,
                        help="Sim count at ramp start epoch (enables linear ramp)")
    parser.add_argument("--mcts-sims-end", type=int,
                        help="Sim count at ramp end epoch")
    parser.add_argument("--mcts-ramp-start-epoch", type=int,
                        help="Epoch where sim ramp begins")
    parser.add_argument("--mcts-ramp-end-epoch", type=int,
                        help="Epoch where sim ramp ends")
    parser.add_argument("--search-batch-size", type=int)
    parser.add_argument(
        "--max-acq-price-actions",
        type=int,
        help="Cap ACQ_SELECT_PRICE to equal low/high price edges, 0 disables (RSS only)",
    )
    nonfinite_group = parser.add_mutually_exclusive_group()
    nonfinite_group.add_argument(
        "--check-nonfinite-mcts",
        dest="check_nonfinite_mcts",
        action="store_true",
        default=None,
        help="Fail fast if MCTS observes NaN/inf values or priors",
    )
    nonfinite_group.add_argument(
        "--no-check-nonfinite-mcts",
        dest="check_nonfinite_mcts",
        action="store_false",
        help="Disable MCTS NaN/inf checks",
    )
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--num-eval-servers", type=int)
    parser.add_argument(
        "--eval-devices",
        type=_parse_str_list,
        help="Comma-separated eval-server devices, one per server "
             "(for example: cuda:0,cuda:1). Defaults to the training device.",
    )
    parser.add_argument("--buffer-capacity", type=int)
    parser.add_argument(
        "--eval-min-batch-size", type=int,
        help="Minimum GPU batch size in states before launching a forward "
             "pass. 0 (default) disables — submit on every drain.",
    )
    parser.add_argument(
        "--eval-min-batch-timeout-ms", type=float,
        help="Timeout (ms) the min-batch loop waits for more arrivals "
             "before flushing a partial batch (default: 10)",
    )
    parser.add_argument(
        "--eval-batch-shape-mode",
        type=str,
        choices=["dynamic", "bucketed"],
        help="Eval GPU batch-shape policy: current fully dynamic behavior or "
             "power-of-2 bucketed launches",
    )
    parser.add_argument(
        "--eval-max-batch-size",
        type=int,
        help="Bucketed mode only: maximum actual states per eval launch before "
             "padding (0 uses the partition max)",
    )
    parser.add_argument("--checkpoint-dir", type=str)
    parser.add_argument("--tensorboard-dir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temp-initial", type=float)
    parser.add_argument("--temp-anneal-start", type=int)
    parser.add_argument("--temp-anneal-end", type=int)
    parser.add_argument("--temp-anneal-starts", type=_parse_int_list)
    parser.add_argument("--temp-anneal-ends", type=_parse_int_list)
    parser.add_argument("--temp-final", type=float)
    parser.add_argument("--policy-target-temp-initial", type=float)
    parser.add_argument("--policy-target-temp-anneal-start", type=int)
    parser.add_argument("--policy-target-temp-anneal-end", type=int)
    parser.add_argument(
        "--policy-target-temp-anneal-starts", type=_parse_int_list,
    )
    parser.add_argument(
        "--policy-target-temp-anneal-ends", type=_parse_int_list,
    )
    parser.add_argument("--policy-target-temp-final", type=float)
    parser.add_argument("--c-puct-initial", type=float)
    parser.add_argument("--c-puct-final", type=float)
    parser.add_argument("--c-puct-anneal-epochs", type=int)
    parser.add_argument("--value-blend-start-epoch", type=int)
    parser.add_argument("--value-blend-end-epoch", type=int)
    parser.add_argument(
        "--value-blend-final", type=float,
        help="A0GB share of value targets after the blend ramp (default: 1.0)",
    )
    parser.add_argument(
        "--terminal-blend", type=float,
        help="Rank vs margin weight for terminal rewards (0=margin, 1=rank, default 0.5)",
    )
    parser.add_argument("--lr-min", type=float, help="Minimum learning rate for cosine decay")
    parser.add_argument(
        "--warmup-epochs", type=float,
        help="Linear LR warmup duration in training epochs (default: 1.0)",
    )
    parser.add_argument(
        "--lr-decay-end-epoch", type=int,
        help="Epoch at which LR reaches lr_min and stays constant (default: num_epochs)",
    )
    parser.add_argument(
        "--optimizer", type=str, choices=["adamw", "muon"],
        help="Optimizer type (default: muon)",
    )
    parser.add_argument(
        "--weight-decay", type=float,
        help="Decoupled weight decay for decayed optimizer param groups (default: 1e-2)",
    )
    parser.add_argument(
        "--grad-clip", type=float,
        help="Global-norm gradient clip (default: 1.0; pass 0 to disable)",
    )
    parser.add_argument("--dirichlet-alpha", type=float)
    parser.add_argument("--dirichlet-epsilon", type=float)
    dyn_group = parser.add_mutually_exclusive_group()
    dyn_group.add_argument(
        "--dirichlet-dynamic", action="store_true", default=None,
        help="Use dynamic alpha = numerator / n_legal_actions",
    )
    dyn_group.add_argument(
        "--no-dirichlet-dynamic", dest="dirichlet_dynamic",
        action="store_false",
        help="Use static alpha (override checkpoint that has dynamic enabled)",
    )
    parser.add_argument(
        "--dirichlet-alpha-numerator", type=float,
        help="Numerator for dynamic alpha: alpha = N / n_legal (default: 10.0)",
    )
    parser.add_argument(
        "--replay-priority-fraction", type=float,
        help="Share of each training batch sampled by policy KL priority; "
             "0 disables (default: 0)",
    )
    parser.add_argument(
        "--replay-priority-exponent", type=float,
        help="Exponent on the KL priority (default: 0.5)",
    )
    parser.add_argument(
        "--forced-playouts-k", type=float,
        help="KataGo forced playouts at self-play roots, sqrt(k*P*N) visits "
             "per child; 0 disables (default: 0)",
    )
    parser.add_argument(
        "--policy-target-pruning", choices=POLICY_TARGET_PRUNING_MODES,
        help="Prune root visits before building policy targets (default: none)",
    )
    parser.add_argument(
        "--profile", action="store_true", default=None,
        help="Enable per-epoch self-play performance profiling",
    )
    parser.add_argument(
        "--no-compile", action="store_true", default=False,
        help="Disable torch.compile model optimization",
    )
    parser.add_argument(
        "--openblas-threads", type=_parse_positive_int,
        help="OPENBLAS_NUM_THREADS for the self-play worker and eval-server "
             "processes (default: the environment's value, else 1)",
    )
    parser.add_argument(
        "--eval-dtype", type=str, choices=["bfloat16", "float16"],
        help="Enable autocast for eval inference (default: disabled, fp32)",
    )
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--phase-conditioning",
        dest="phase_conditioning",
        action="store_true",
        default=None,
        help="Enable per-block adaLN phase conditioning (v2 only)",
    )
    phase_group.add_argument(
        "--no-phase-conditioning",
        dest="phase_conditioning",
        action="store_false",
        help="Disable per-block adaLN phase conditioning (v2 only)",
    )
    mixing_group = parser.add_mutually_exclusive_group()
    mixing_group.add_argument(
        "--relation-input-mixing", action="store_true", default=None,
        help="Enable relation input mixing before type embeddings (v3 only; default)",
    )
    mixing_group.add_argument(
        "--no-relation-input-mixing", dest="relation_input_mixing", action="store_false",
        help="Disable relation input mixing, retaining relation attention biases (v3 only)",
    )
    zero_sum_group = parser.add_mutually_exclusive_group()
    zero_sum_group.add_argument(
        "--zero-sum-values", action="store_true", default=None,
        help="Subtract the mean over real players from value outputs (v3 and 1889)",
    )
    zero_sum_group.add_argument(
        "--no-zero-sum-values", dest="zero_sum_values", action="store_false",
        help="Use raw per-player tanh value outputs (v3 and 1889)",
    )
    parser.add_argument(
        "--price-slot-fourier-bands",
        type=int,
        help="Number of fixed Fourier bands for price-like policy slot keys",
    )
    return parser


_CLI_FIELDS = (
    "num_players", "min_players", "max_players", "v3_behavior", "acq_same_president",
    "eval_dtype", "model_type", "model_path", "phase_conditioning", "relation_input_mixing",
    "zero_sum_values",
    "d_model", "d_proj", "num_heads", "num_layers", "ff_mult",
    "price_slot_fourier_bands",
    "games_per_epoch", "num_epochs", "training_steps_per_epoch",
    "num_simulations", "search_batch_size", "check_nonfinite_mcts",
    "max_acq_price_actions",
    "mcts_sims_start", "mcts_sims_end", "mcts_ramp_start_epoch", "mcts_ramp_end_epoch",
    "num_workers", "num_eval_servers", "eval_devices",
    "buffer_capacity",
    "eval_min_batch_size", "eval_min_batch_timeout_ms",
    "eval_batch_shape_mode", "eval_max_batch_size",
    "checkpoint_dir", "tensorboard_dir", "seed",
    "temp_initial", "temp_anneal_start", "temp_anneal_end",
    "temp_anneal_starts", "temp_anneal_ends", "temp_final",
    "policy_target_temp_initial", "policy_target_temp_anneal_start",
    "policy_target_temp_anneal_end",
    "policy_target_temp_anneal_starts", "policy_target_temp_anneal_ends",
    "policy_target_temp_final",
    "c_puct_initial", "c_puct_final", "c_puct_anneal_epochs",
    "value_blend_start_epoch", "value_blend_end_epoch", "value_blend_final",
    "terminal_blend", "lr_min", "warmup_epochs", "lr_decay_end_epoch",
    "optimizer", "weight_decay", "grad_clip",
    "dirichlet_alpha", "dirichlet_epsilon",
    "dirichlet_dynamic", "dirichlet_alpha_numerator",
    "forced_playouts_k", "policy_target_pruning",
    "replay_priority_fraction", "replay_priority_exponent",
)


def _build_profile_scalars(
    game_profiles: list[GameProfileData],
    server_stats: EvalServerStats | None,
    sp_duration: float,
    num_eval_servers: int = 1,
) -> dict[str, float]:
    """Build Tensorboard scalars from profile data."""
    n = len(game_profiles)
    sel = sum(g.search.selection_secs for g in game_profiles) / n
    evl = sum(g.search.eval_secs for g in game_profiles) / n
    bak = sum(g.search.backup_secs for g in game_profiles) / n
    total = sel + evl + bak

    scalars: dict[str, float] = {}
    if total > 0:
        scalars["profile/search_select_pct"] = sel / total * 100
        scalars["profile/search_eval_pct"] = evl / total * 100
        scalars["profile/search_backup_pct"] = bak / total * 100
    scalars["profile/search_select_secs"] = sel
    scalars["profile/search_eval_secs"] = evl
    scalars["profile/search_backup_secs"] = bak

    searches = sum(g.search.num_searches for g in game_profiles) / n
    batches = sum(g.search.num_eval_batches for g in game_profiles) / n
    leaves = sum(g.search.total_leaves for g in game_profiles) / n
    vbackups = sum(g.search.virtual_backups for g in game_profiles) / n
    scalars["profile/searches_per_game"] = searches
    scalars["profile/eval_batches_per_game"] = batches
    scalars["profile/leaves_per_game"] = leaves
    scalars["profile/virtual_backups_per_game"] = vbackups

    clients = [g.eval_client for g in game_profiles if g.eval_client is not None]
    if clients:
        nc = len(clients)
        prep = sum(c.prepare_secs for c in clients) / nc
        wt = sum(c.wait_secs for c in clients) / nc
        res = sum(c.result_secs for c in clients) / nc
        ct = prep + wt + res
        if ct > 0:
            scalars["profile/eval_client_prepare_pct"] = prep / ct * 100
            scalars["profile/eval_client_wait_pct"] = wt / ct * 100
            scalars["profile/eval_client_result_pct"] = res / ct * 100

    if server_stats is not None and server_stats.batch_count > 0:
        s = server_stats
        scalars["profile/server_batch_avg"] = s.batch_size_sum / s.batch_count
        scalars["profile/server_batch_max"] = float(s.batch_size_max)
        scalars["profile/server_infer_ms"] = (
            s.inference_secs_sum / s.batch_count * 1000
        )
        if sp_duration > 0:
            scalars["profile/server_throughput"] = s.total_states / sp_duration
        scalars["profile/server_idle_pct"] = (
            s.idle_secs / num_eval_servers / sp_duration * 100
            if sp_duration > 0 else 0
        )

    return scalars


def _build_eval_server_scalars(stats: EvalServerStats) -> dict[str, float]:
    """The eval servers' epoch: real against launched rows, launches by
    launched size, and the serve loop's wall time by stage, as shares and as
    milliseconds a launch. Merged servers sum their rows and times."""
    scalars = {
        "eval_server/launches": float(stats.batch_count),
        "eval_server/real_rows": float(stats.total_states),
        "eval_server/launched_rows": float(stats.launched_rows),
        "eval_server/wall_secs": stats.wall_secs,
    }
    if stats.launched_rows:
        scalars["eval_server/padding_share"] = 1.0 - stats.total_states / stats.launched_rows
    for size, count in sorted(stats.launch_counts.items()):
        scalars[f"eval_server/launches_at_{size}"] = float(count)
        scalars[f"eval_server/real_rows_mean_at_{size}"] = stats.launch_real_rows[size] / count
    for stage, secs in zip(STAGES, stats.stage_secs, strict=True):
        if stats.wall_secs > 0:
            scalars[f"eval_server/{stage}_share"] = secs / stats.wall_secs
        if stats.batch_count:
            scalars[f"eval_server/{stage}_ms_per_launch"] = secs / stats.batch_count * 1000
    return scalars


_RANK_LABELS = ("1st", "2nd", "3rd", "4th", "5th", "6th")


class _SelfPlayMetricBucket:
    """Accumulate self-play game metrics for one reporting group."""

    def __init__(self) -> None:
        self.games = 0
        self.examples = 0
        self.moves = 0
        self.duration = 0.0
        self.target_entropy = 0.0
        self.target_top1 = 0.0
        self.sample_entropy = 0.0
        self.sample_top1 = 0.0
        self.total_net_worth = 0.0
        self.policy_metrics = PolicyMetrics()
        # The game's diagnostics: name -> [numerator, denominator] totals.
        self.game_stats: dict[str, list[float]] = {}

        self.rank_counts: list[int] = []
        self.rank_net_worths: list[float] = []
        self.rank_net_worth_mins: list[float] = []
        self.rank_net_worth_maxs: list[float] = []

    def _ensure_rank_capacity(self, num_ranks: int) -> None:
        while len(self.rank_counts) < num_ranks:
            self.rank_counts.append(0)
            self.rank_net_worths.append(0.0)
            self.rank_net_worth_mins.append(float("inf"))
            self.rank_net_worth_maxs.append(float("-inf"))

    def add_record(self, record: Any) -> None:
        num_players = int(record.num_players)
        self._ensure_rank_capacity(num_players)

        self.games += 1
        self.examples += int(record.num_examples)
        self.moves += int(record.total_moves)
        self.duration += float(record.duration_secs)
        self.policy_metrics.add(record.policy_metrics)
        for name, (numerator, denominator) in record.game_stats.items():
            totals = self.game_stats.setdefault(name, [0.0, 0.0])
            totals[0] += numerator
            totals[1] += denominator
        self.target_entropy += float(record.policy_target_entropy_mean)
        self.target_top1 += float(record.policy_target_top1_fraction)
        self.sample_entropy += float(record.sample_policy_entropy_mean)
        self.sample_top1 += float(record.sample_top1_action_fraction)
        self.total_net_worth += float(sum(record.net_worths[:num_players]))

        ranked = sorted(
            range(num_players),
            key=lambda p: record.net_worths[p],
            reverse=True,
        )
        for rank, player_id in enumerate(ranked):
            net_worth = float(record.net_worths[player_id])
            self.rank_counts[rank] += 1
            self.rank_net_worths[rank] += net_worth
            self.rank_net_worth_mins[rank] = min(
                self.rank_net_worth_mins[rank], net_worth,
            )
            self.rank_net_worth_maxs[rank] = max(
                self.rank_net_worth_maxs[rank], net_worth,
            )

    def snapshot(self) -> dict[str, Any]:
        games = self.games
        if games == 0:
            return {
                "games": 0.0,
                "game_stats": {},
                "examples": 0.0,
                "avg_moves": 0.0,
                "avg_duration": 0.0,
                "rank_net_worths": [],
                "rank_net_worths_min": [],
                "rank_net_worths_max": [],
                "policy_entropy": 0.0,
                "top1_visit_frac": 0.0,
                "policy_target_entropy": 0.0,
                "policy_target_top1_frac": 0.0,
                "sample_policy_entropy": 0.0,
                "sample_top1_frac": 0.0,
                "total_net_worth": 0.0,
            }

        def rank_avg(values: list[float]) -> list[float]:
            return [
                value / count if count > 0 else 0.0
                for value, count in zip(values, self.rank_counts)
            ]

        return {
            "games": float(games),
            "policy_metrics": self.policy_metrics.scalars(),
            "game_stats": {
                name: numerator / max(denominator, 1)
                for name, (numerator, denominator) in self.game_stats.items()
            },
            "examples": float(self.examples),
            "avg_moves": self.moves / games,
            "avg_duration": self.duration / games,
            "rank_net_worths": rank_avg(self.rank_net_worths),
            "rank_net_worths_min": [
                value if count > 0 else 0.0
                for value, count in zip(
                    self.rank_net_worth_mins, self.rank_counts,
                )
            ],
            "rank_net_worths_max": [
                value if count > 0 else 0.0
                for value, count in zip(
                    self.rank_net_worth_maxs, self.rank_counts,
                )
            ],
            "policy_entropy": self.target_entropy / games,
            "top1_visit_frac": self.sample_top1 / games,
            "policy_target_entropy": self.target_entropy / games,
            "policy_target_top1_frac": self.target_top1 / games,
            "sample_policy_entropy": self.sample_entropy / games,
            "sample_top1_frac": self.sample_top1 / games,
            "total_net_worth": self.total_net_worth / games,
        }


class _SelfPlayMetricAccumulator:
    """Accumulate aggregate and per-player-count self-play metrics."""

    def __init__(self) -> None:
        self.aggregate = _SelfPlayMetricBucket()
        self.by_player_count: dict[int, _SelfPlayMetricBucket] = {}

    def add_record(self, record: Any) -> None:
        num_players = int(record.num_players)
        self.aggregate.add_record(record)
        bucket = self.by_player_count.setdefault(
            num_players, _SelfPlayMetricBucket(),
        )
        bucket.add_record(record)

    def aggregate_snapshot(self) -> dict[str, Any]:
        return self.aggregate.snapshot()

    def count_snapshots(self) -> dict[int, dict[str, Any]]:
        return {
            num_players: bucket.snapshot()
            for num_players, bucket in sorted(self.by_player_count.items())
            if bucket.games > 0
        }


class _EpochSelfPlayMetrics:
    """An epoch's self-play metrics for TensorBoard, the games started from a
    position apart from those played from the opening: a started game's
    length, trains, and end differ, and the opening games' groups stay
    comparable with runs that have no starts. ``every`` holds all the games,
    for the console."""

    def __init__(self) -> None:
        self.opening = _SelfPlayMetricAccumulator()
        self.started = _SelfPlayMetricAccumulator()
        self.every = _SelfPlayMetricAccumulator()
        self._start_moves: dict[int, int] = {}  # started games' start moves, summed by player count

    def add_record(self, record: Any) -> None:
        self.every.add_record(record)
        if record.flags & FLAG_FROM_POSITION:
            self.started.add_record(record)
            num_players = int(record.num_players)
            self._start_moves[num_players] = self._start_moves.get(num_players, 0) + int(record.start_move)
        else:
            self.opening.add_record(record)

    def scalars(self) -> dict[str, float]:
        """``self_play_aggregate`` and ``self_play_<n>p`` over the opening
        games; ``self_play_started`` and ``self_play_started_<n>p`` over the
        started ones, with their number and mean start move."""
        scalars = _build_epoch_self_play_scalars(self.opening)
        groups = {"self_play_started": (self.started.aggregate_snapshot(), sum(self._start_moves.values()))}
        groups.update(
            (f"self_play_started_{num_players}p", (stats, self._start_moves[num_players]))
            for num_players, stats in self.started.count_snapshots().items()
        )
        for prefix, (stats, start_moves) in groups.items():
            if stats["games"] > 0:
                scalars.update(_build_self_play_scalars(prefix, stats))
                scalars[f"{prefix}/games"] = stats["games"]
                scalars[f"{prefix}/start_move_mean"] = start_moves / stats["games"]
        return scalars


def _load_start_positions(config: TrainingConfig, game: Game) -> StartRows | None:
    """The positions ``human_start_fraction`` of the games start from, loaded
    once for the main process to hand out; None with the feature off."""
    path = config.start_positions_path
    if path is None:
        return None
    if game.load_start_positions is None:
        raise ValueError(
            f"start_positions_path is set, but game {game.name!r} has no start positions"
        )
    if config.human_start_fraction <= 0:
        return None
    starts = game.load_start_positions(path)
    by_count = {n: int((starts.num_players == n).sum()) for n in config.iter_player_counts()}
    print(
        f"Start positions: {path}, {len(starts.rows):,} rows ("
        + ", ".join(f"{n}p {count:,}" for n, count in by_count.items())
        + f"); {config.human_start_fraction:.0%} of each player count's games start from one"
    )
    for n, count in by_count.items():
        if count == 0:
            print(f"  No start positions at {n} players: its games all start from the opening")
    return starts


class _OpeningGameLengths:
    """An epoch's game lengths in moves by player count, over the games played
    from the opening: a game started from a position measures only its
    remainder."""

    def __init__(self) -> None:
        self._totals: dict[int, list[int]] = {}  # [games, moves]

    def add_record(self, record: Any) -> None:
        if record.flags & FLAG_FROM_POSITION:
            return
        totals = self._totals.setdefault(int(record.num_players), [0, 0])
        totals[0] += 1
        totals[1] += int(record.total_moves)

    def means(self) -> dict[int, float]:
        """Each player count's mean length; a count with no game is absent."""
        return {
            num_players: moves / games
            for num_players, (games, moves) in sorted(self._totals.items())
        }


def _build_self_play_scalars(
    prefix: str, stats: dict[str, Any],
) -> dict[str, float]:
    games = float(stats.get("games", 0.0))
    if games <= 0:
        return {}

    scalars: dict[str, float] = {
        f"{prefix}/game_length_mean": float(stats.get("avg_moves", 0.0)),
        f"{prefix}/duration_mean": float(stats.get("avg_duration", 0.0)),
        f"{prefix}/total_examples": float(stats.get("examples", 0.0)),
        f"{prefix}/policy_entropy_mean": float(stats.get("policy_entropy", 0.0)),
        f"{prefix}/top1_visit_fraction": float(stats.get("top1_visit_frac", 0.0)),
        f"{prefix}/policy_target_entropy_mean": float(
            stats.get("policy_target_entropy", 0.0),
        ),
        f"{prefix}/policy_target_top1_fraction": float(
            stats.get("policy_target_top1_frac", 0.0),
        ),
        f"{prefix}/sample_policy_entropy_mean": float(
            stats.get("sample_policy_entropy", 0.0),
        ),
        f"{prefix}/sample_top1_action_fraction": float(
            stats.get("sample_top1_frac", 0.0),
        ),
        f"{prefix}/total_net_worth": float(stats.get("total_net_worth", 0.0)),
    }

    rank_net_worths = list(stats.get("rank_net_worths", []))
    scalars.update({f"{prefix}/{name}": value for name, value in stats.get("game_stats", {}).items()})
    scalars.update({f"{prefix}/{name}": value for name, value in stats.get("policy_metrics", {}).items()})
    rank_mins = list(stats.get("rank_net_worths_min", []))
    rank_maxs = list(stats.get("rank_net_worths_max", []))

    for rank, avg in enumerate(rank_net_worths):
        label = _RANK_LABELS[rank]
        scalars[f"{prefix}/net_worth_{label}"] = float(avg)
        if rank < len(rank_mins):
            scalars[f"{prefix}/net_worth_{label}_min"] = float(rank_mins[rank])
        if rank < len(rank_maxs):
            scalars[f"{prefix}/net_worth_{label}_max"] = float(rank_maxs[rank])

    return scalars


def _build_epoch_self_play_scalars(
    metrics: _SelfPlayMetricAccumulator,
) -> dict[str, float]:
    scalars = _build_self_play_scalars(
        "self_play_aggregate", metrics.aggregate_snapshot(),
    )
    for num_players, stats in metrics.count_snapshots().items():
        scalars.update(_build_self_play_scalars(f"self_play_{num_players}p", stats))
    return scalars


def _apply_overrides(
    config: TrainingConfig, args: argparse.Namespace, *, log_changes: bool = False,
) -> None:
    """Apply CLI overrides to config in-place.

    When log_changes is True (resume), prints overridden checkpoint values.
    """
    for field in _CLI_FIELDS:
        val = getattr(args, field, None)
        if val is not None:
            if log_changes:
                old = getattr(config, field)
                if old != val:
                    print(f"  CLI override: {field} = {val} (was {old})")
            setattr(config, field, val)


def _resolve_config(args: argparse.Namespace, cp: Checkpoint | None) -> TrainingConfig:
    """The run's config: the checkpoint's on resume, then the ``--config``
    JSON, then CLI overrides. A resumed run keeps the checkpoint's game."""
    if cp is not None:
        config = TrainingConfig.from_json(cp["config_json"])  # type: ignore[arg-type]
        trained_game = config.game
        if args.config:
            changes = config.apply_json_overrides(Path(args.config).read_text())
            for c in changes:
                print(f"  Config override: {c}")
        _apply_overrides(config, args, log_changes=True)
        if config.game != trained_game:
            raise ValueError(
                f"the checkpoint plays game {trained_game!r}, but the run's config "
                f"sets game {config.game!r}; a run resumes only its own game"
            )
    elif args.config:
        config = TrainingConfig.from_json(Path(args.config).read_text())
        _apply_overrides(config, args)
    else:
        config = TrainingConfig()
        _apply_overrides(config, args)
    config.validate()
    return config


def _latest_checkpoint_dir(args: argparse.Namespace) -> Path:
    """Directory searched by ``--resume latest``, with config precedence."""
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    if args.config:
        return Path(TrainingConfig.from_json(Path(args.config).read_text()).checkpoint_dir)
    return Path(TrainingConfig.checkpoint_dir)


def _resolve_eval_devices(
    config: TrainingConfig,
    training_device: torch.device,
) -> list[torch.device]:
    """Return one eval-server device per configured eval server."""
    if config.num_workers <= 0:
        return []
    if config.eval_devices:
        return [torch.device(device) for device in config.eval_devices]
    return [training_device for _ in range(config.num_eval_servers)]


def _model_state_dict_cpu_snapshot(
    model: torch.nn.Module,
) -> dict[str, np.ndarray]:
    """Clone model state to CPU for eval-server replica sync."""
    base_model = getattr(model, "_orig_mod", model)
    snapshot: dict[str, np.ndarray] = {}
    for name, tensor in base_model.state_dict().items():
        snapshot[name] = tensor.detach().cpu().numpy().copy()
    return snapshot


def _sync_eval_servers(
    eval_servers: list[EvaluationServer],
    model: torch.nn.Module,
    *,
    timeout: float = 300.0,
) -> None:
    """Push the trainer's current weights into all eval-server replicas."""
    if not eval_servers:
        return
    state_dict = _model_state_dict_cpu_snapshot(model)
    for server in eval_servers:
        server.sync_weights(state_dict, timeout=timeout)


def _scaled_training_steps(
    config: TrainingConfig, buffer_size: int, window_epochs: int = 0,
) -> int:
    """Scale per-epoch training updates by replay-buffer fullness.

    Early epochs have less data diversity in replay. Scale the configured
    update budget by the occupied fraction of the buffer so, for example,
    100k rows in a 500k buffer runs 20% of the configured steps. With
    ``buffer_epochs`` set, ``buffer_size`` is the sampling window's rows and
    fullness is the share of ``buffer_epochs`` the window holds
    (``window_epochs``): the capacity is then only a cap.
    """
    if buffer_size < config.min_buffer_size or config.training_steps_per_epoch < 1:
        return 0
    if config.buffer_epochs > 0:
        held, full = min(window_epochs, config.buffer_epochs), config.buffer_epochs
    else:
        held, full = min(max(buffer_size, 0), config.buffer_capacity), config.buffer_capacity
    return max(1, config.training_steps_per_epoch * held // full)


def _epoch_training_steps(config: TrainingConfig, buffer: ReplayBuffer) -> int:
    """Training steps for the epoch whose games ``buffer`` just took; 0 while
    its sampling window is under ``min_buffer_size``.

    With ``replay_ratio``, steps follow the epoch's own rows: ``replay_ratio *
    rows / batch_size``, times the share of ``buffer_epochs`` the window holds.
    An epoch then draws each row of the window ``replay_ratio / buffer_epochs``
    times on average, ``replay_ratio`` times over the ``buffer_epochs`` epochs
    the row spends there. Without the share, the first epochs' rows, sampled
    from a window still filling, would be drawn about twice as often. The
    epoch's rows include those a graceful shutdown left it before a resume.
    Without ``replay_ratio``, ``_scaled_training_steps``.
    """
    if config.replay_ratio <= 0:
        return _scaled_training_steps(config, buffer.window_size, buffer.window_epochs)
    if buffer.window_size < config.min_buffer_size:
        return 0
    draws = config.replay_ratio * buffer.epoch_rows * buffer.window_epochs / config.buffer_epochs
    return max(1, round(draws / config.batch_size))


_HELDOUT_STREAM = 0x4E1D  # keeps the held-out draw apart from other uses of the seed


def _heldout_scalars(
    trainer: Trainer,
    buffer: ReplayBuffer,
    config: TrainingConfig,
    epoch: int,
    phase_names: Sequence[str],
) -> dict[str, float]:
    """The held-out loss: up to ``heldout_rows`` of the epoch's new rows,
    scored by the weights that have yet to train on them.

    The rows are drawn uniformly without replacement from the epoch's
    (``epoch_row_positions``), by a generator of ``(seed, epoch)`` alone, so
    the run's other random streams are as they are without it. The trainer
    scores them by its own input path and loss functions, in eval mode, with
    no update and no importance weights. Value MSE and policy KL cover every
    row; then games from the opening alone, together, by quarter of their
    game's rows, and by decision phase; then games started from a position.
    Empty without ``heldout_rows``.
    """
    if config.heldout_rows <= 0:
        return {}
    indices, positions, lengths = buffer.epoch_row_positions()
    count = min(config.heldout_rows, len(indices))
    if count == 0:
        return {}
    rng = np.random.default_rng((config.seed, epoch, _HELDOUT_STREAM))
    sample = np.sort(rng.choice(len(indices), size=count, replace=False))
    rows = indices[sample]
    scores = trainer.score_batches(
        buffer.rows_at(rows[start:start + config.batch_size])
        for start in range(0, count, config.batch_size)
    )
    labels = buffer.rows_at(rows, ("phase_ids", "player_counts", "flags"))
    started = (labels["flags"] & FLAG_FROM_POSITION) != 0
    opening = ~started
    quarters = 4 * positions[sample] // lengths[sample]
    groups = {"all": np.ones(count, dtype=bool), "opening": opening, "started": started}
    groups.update({f"q{q + 1}": opening & (quarters == q) for q in range(4)})
    groups.update({name: opening & (labels["phase_ids"] == i) for i, name in enumerate(phase_names)})
    scalars = {"heldout/rows": float(count), "heldout/rows_started": float(started.sum())}
    for group, measures in summarize_scores(*scores, labels["player_counts"], groups).items():
        suffix = "" if group == "all" else f"_{group}"
        scalars[f"heldout/value_mse{suffix}"] = measures["value_mse"]
        scalars[f"heldout/policy_kl{suffix}"] = measures["policy_kl"]
    return scalars


def _heldout_over_train(heldout: dict[str, float], avg_losses: dict[str, float]) -> dict[str, float]:
    """The held-out value MSE over the epoch's mean training value loss."""
    if "heldout/value_mse" not in heldout or not avg_losses.get("value_loss"):
        return {}
    return {"heldout/value_mse_over_training": heldout["heldout/value_mse"] / avg_losses["value_loss"]}


def _capture_rng_state(master_rng: np.random.Generator) -> dict[str, object]:
    """Capture all RNG states for checkpoint reproducibility."""
    state: dict[str, object] = {
        "numpy": master_rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(
    master_rng: np.random.Generator, state: dict[str, object],
) -> None:
    """Restore RNG states from a checkpoint.

    The checkpoint may have been loaded with map_location=cuda, which moves
    the RNG state tensors to GPU. torch.set_rng_state requires CPU ByteTensors,
    so we explicitly move them back.
    """
    master_rng.bit_generator.state = state["numpy"]  # type: ignore[assignment]
    torch.set_rng_state(state["torch_cpu"].cpu().byte())  # type: ignore[union-attr]
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(  # type: ignore[arg-type]
            [s.cpu().byte() for s in state["torch_cuda"]]  # type: ignore[union-attr]
        )


def _start_shutdown_listener() -> threading.Event:
    """Start a background thread that listens for 'q' + Enter on stdin.

    Returns an Event that is set when the user requests graceful shutdown.
    If stdin is not a TTY (piped input), returns an unset Event with no thread.
    """
    event = threading.Event()
    if not sys.stdin.isatty():
        return event

    def _listen() -> None:
        try:
            for line in sys.stdin:
                if line.strip().lower() == "q":
                    event.set()
                    return
        except (EOFError, OSError):
            pass

    t = threading.Thread(target=_listen, daemon=True)
    t.start()
    return event


def _drain_workers(
    task_queue: Any,
    result_queue: Any,
    workers: list[Any],
    collect_fn: Any,
    games_collected: int,
    timeout: float = 120.0,
) -> int:
    """Drain in-flight games from worker processes.

    Clears remaining game assignments from the task queue, sends None
    sentinels to workers, and collects results until all workers have exited
    or the timeout is reached.

    Returns the total number of games collected (including pre-drain).
    """
    # Clear remaining game assignments from task queue.
    cleared = 0
    while True:
        try:
            task_queue.get_nowait()
            cleared += 1
        except queue.Empty:
            break

    # Send None sentinels so workers exit after their current game
    for _ in workers:
        task_queue.put(None)

    if cleared > 0:
        print(f"  Cleared {cleared} pending game assignments from queue")

    # Collect results from in-flight games
    deadline = time.perf_counter() + timeout
    drained = 0
    while time.perf_counter() < deadline:
        alive = [w for w in workers if w.is_alive()]
        if not alive and result_queue.empty():
            break
        try:
            record = result_queue.get(timeout=1.0)
            collect_fn(record, games_collected + drained)
            drained += 1
            print(f"  Drained game {drained} (from {len(alive)} live workers)")
        except queue.Empty:
            if not alive:
                break

    # Join workers
    for w in workers:
        w.join(timeout=3.0)
        if w.is_alive():
            w.terminate()

    if drained > 0:
        print(f"  Collected {drained} in-flight games")

    return games_collected + drained


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --- Device (resolved early, needed for checkpoint loading) ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- GPU vendor detection and optimizations ---
    from train.gpu import detect_gpu

    gpu = detect_gpu(device.type)

    if gpu.vendor != "cpu":
        gpu_info = gpu.apply_optimizations()
        print(
            f"GPU optimizations ({gpu.vendor}): "
            + ", ".join(f"{k}={v}" for k, v in gpu_info.items())
        )

    # --- Resolve checkpoint for resume ---
    cp: Checkpoint | None = None
    if args.resume:
        cp_path: Path | None = None
        if args.resume == "latest":
            cp_dir = _latest_checkpoint_dir(args)
            cp_path = find_latest_checkpoint(cp_dir)
            if cp_path is None:
                print(f"No checkpoint found in {cp_dir}, starting from scratch.")
        else:
            cp_path = Path(args.resume)

        if cp_path is not None:
            cp = load_checkpoint(cp_path, device)
            print(f"Loaded checkpoint: {cp_path}")

    # --- Config ---
    config = _resolve_config(args, cp)

    # --- Operational flags (not persisted in config JSON) ---
    if args.profile:
        config.profile = True

    eval_devices = _resolve_eval_devices(config, device)

    # --- RNG ---
    master_rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Restore RNG state from checkpoint (after base init, before any draws)
    if cp is not None and "rng_state" in cp:
        _restore_rng_state(master_rng, cp["rng_state"])  # type: ignore[arg-type]
        print("  Restored RNG state from checkpoint")

    # --- Model ---
    model = create_model(config).to(device)
    model_input_spec = get_model_input_spec(config)
    param_count = sum(p.numel() for p in model.parameters())

    # --- Resume: restore model weights (before compile + Trainer creation) ---
    start_epoch = 0
    # Mean game length by player count over the last finished epoch's games
    # from the opening: what temp_anneal_*_pcts are shares of.
    game_lengths: dict[int, float] = {}
    if cp is not None:
        model.load_state_dict(cp["model_state_dict"])  # type: ignore[arg-type]
        start_epoch = cp["epoch"] + 1  # type: ignore[operator]
        game_lengths = dict(cp.get("game_lengths", {}))
        if config.temp_anneal_start_pcts and "game_lengths" not in cp:
            print(
                "  Checkpoint has no game lengths for temp_anneal_*_pcts: the "
                "next epoch plays at temp_initial throughout"
            )

    # --- Components (model-independent) ---
    # Compact int16 state — replay buffer stores raw state rows, the
    # trainer builds token buffers per-batch through the game's adapter.
    game = get_game(config)
    start_positions = _load_start_positions(config, game)
    max_players = config.effective_max_players
    state_size_int16 = game.state_size(max_players)
    num_tokens = game.num_tokens(max_players)
    buffer = ReplayBuffer(
        game,
        config.buffer_capacity,
        state_size_int16,
        max_players,
        min_players=config.effective_min_players,
        max_players=max_players,
        priority_fraction=config.replay_priority_fraction,
        priority_exponent=config.replay_priority_exponent,
        buffer_epochs=config.buffer_epochs,
    )
    logger = TrainingLogger(config.tensorboard_dir)

    # --- Log startup ---
    logger.log_training_start(config, device=str(device))
    print(f"Model parameters: {param_count:,}")

    # --- Resume: load replay buffer if available ---
    buffer_dir = Path(config.checkpoint_dir) / "replay_buffer"
    if cp is not None:
        loaded = buffer.load(buffer_dir)
        if loaded > 0:
            print(f"Loaded {loaded:,} examples into replay buffer")

    # --- Graceful shutdown listener ---
    shutdown_event = _start_shutdown_listener()
    print("Press q + Enter for graceful shutdown\n")

    # --- Multi-process self-play setup ---
    # Eval servers must be spawned BEFORE torch.compile — they receive the
    # uncompiled model via CUDA IPC and compile independently per-process.
    workers: list[Any] = []  # mp.Process (SpawnProcess when using spawn context)
    eval_servers: list[EvaluationServer] = []
    task_queue: Any = None  # mp.Queue
    result_queue: Any = None  # mp.Queue

    if config.num_workers > 0:
        # numpy's OpenBLAS starts a thread per core (up to 64) at import, in
        # every spawned process, before a worker's set_num_threads(1) can run.
        # 128 workers of those exhaust a container's thread limit. Set here,
        # after this process's own pool exists, it reaches only the children:
        # --openblas-threads when given, else the environment's value, else 1.
        if args.openblas_threads is not None:
            os.environ["OPENBLAS_NUM_THREADS"] = str(args.openblas_threads)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        task_queue.cancel_join_thread()
        result_queue = ctx.Queue()
        result_queue.cancel_join_thread()

        shared_bufs = SharedEvalBuffers(
            game,
            num_workers=config.num_workers,
            batch_size=config.search_batch_size,
            num_players=max_players,
            input_spec=model_input_spec,
        )
        # Partition workers across eval servers. Each server owns a
        # contiguous range and scans its partition's bitmap.
        # E.g., 96 workers / 2 servers → server 0: [0, 48), server 1: [48, 96)
        n_servers = config.num_eval_servers
        workers_per_server = config.num_workers // n_servers
        remainder = config.num_workers % n_servers

        partitions: list[tuple[int, int]] = []
        w_offset = 0
        for i in range(n_servers):
            partition = workers_per_server + (1 if i < remainder else 0)
            partitions.append((w_offset, w_offset + partition))
            w_offset += partition

        shared_bufs.init_bitmap(partitions, ctx)

        eval_compile_kwargs = gpu.get_compile_kwargs(
            for_training=False,
            eval_batch_shape_mode=config.eval_batch_shape_mode,
        )
        for i, (ws, we) in enumerate(partitions):
            server_device = eval_devices[i]
            server = EvaluationServer(
                model, server_device, shared_bufs,
                server_id=i,
                worker_start=ws,
                worker_end=we,
                # Its stats feed the eval_server/ tags every epoch.
                profile=True,
                mp_context=ctx,
                no_compile=args.no_compile,
                compile_kwargs=eval_compile_kwargs,
                gpu_vendor=gpu.vendor,
                min_batch_size=config.eval_min_batch_size,
                min_batch_timeout_ms=config.eval_min_batch_timeout_ms,
                batch_shape_mode=config.eval_batch_shape_mode,
                max_batch_size=config.eval_max_batch_size,
                eval_dtype=config.eval_dtype,
            )
            server.start()
            eval_servers.append(server)

        # Wait for all eval servers to finish compilation + warmup before
        # spawning workers.  This prevents workers from timing out on
        # unresponsive servers and avoids CPU contention between server
        # compilation and the main-process compile below.
        n_servers = config.num_eval_servers
        print(
            f"Waiting for {n_servers} eval server{'s' if n_servers > 1 else ''} "
            f"to compile..."
        )
        ready_timeout = 2400.0
        for server in eval_servers:
            if not server.wait_ready(timeout=ready_timeout):
                server.check_alive()
                raise RuntimeError(
                    f"Eval server did not become ready within "
                    f"{ready_timeout:g}s — compilation may be stuck "
                    "(check stderr)"
                )
        print(f"  {n_servers} eval server{'s' if n_servers > 1 else ''} ready.")
        print(
            "Eval server devices: "
            + ", ".join(
                f"{i}:{eval_devices[i]}" for i in range(n_servers)
            )
        )

        # Training-side torch.compile is disabled pending rss-az-mpxc: the AMD
        # Triton backend crashes in make_ttgir lowering the fused backward
        # RMSNorm kernel on gfx1201 (RDNA 4).  Self-play dominates wall time
        # so leaving the trainer uncompiled is near-free.  Eval servers still
        # compile (inference_mode avoids the offending backward partition).

        # Spawn workers now that eval servers are ready to serve requests.
        for i in range(config.num_workers):
            p = ctx.Process(
                target=self_play_worker,
                args=(
                    task_queue, result_queue, config,
                    shared_bufs, i,
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)

        if config.eval_min_batch_size > 0:
            launch_policy = (
                f"min-batch floor {config.eval_min_batch_size} states "
                f"(timeout {config.eval_min_batch_timeout_ms:g}ms)"
            )
        else:
            launch_policy = "greedy"
        if config.eval_batch_shape_mode == "bucketed":
            max_actual = (
                str(config.eval_max_batch_size)
                if config.eval_max_batch_size > 0 else "partition max"
            )
            batch_detail = (
                f", eval batching: bucketed powers-of-2, {launch_policy}, "
                f"max actual batch {max_actual}"
            )
            if gpu.vendor == "nvidia" and not args.no_compile:
                batch_detail += ", CUDA graphs enabled"
        else:
            batch_detail = f", eval batching: dynamic, {launch_policy}"
            if gpu.vendor == "nvidia" and not args.no_compile:
                batch_detail += ", no CUDA graphs"
        print(
            f"Started {config.num_workers} self-play workers, "
            f"{n_servers} eval server{'s' if n_servers > 1 else ''}"
            f"{batch_detail}"
        )

        # Write process IDs for external profiling tools.
        with open("procids.txt", "w") as f:
            f.write(f"main {os.getpid()}\n")
            for i, server in enumerate(eval_servers):
                f.write(f"eval_server_{i} {server._process.pid}\n")  # type: ignore[union-attr]
            for i, w in enumerate(workers):
                f.write(f"worker_{i} {w.pid}\n")
        print("Wrote procids.txt")
    else:
        # Single-process: compile for both training and self-play evaluation.
        if not args.no_compile and device.type == "cuda":
            sp_compile_kwargs = gpu.get_compile_kwargs(for_training=False)
            print(f"Compiling model with torch.compile ({sp_compile_kwargs})...")
            model = cast(torch.nn.Module, torch.compile(model, **sp_compile_kwargs))  # type: ignore[call-overload]
            model.train()
            # Warmup mirrors train/eval_server.py for transformer inputs:
            # one row per phase so every per-row policy head is traced against
            # a real legal mask, and
            # ``mark_unbacked`` on every model input so the batch dim stays
            # symbolic. If any input is left unmarked, Dynamo bakes a static
            # guard on that input's shape and recompiles on the first
            # batch != warmup_n; warming up at batch=1 also wastes a
            # specialization that the runtime never reuses.
            policy = game.policy
            warmup_n = policy.num_phases
            lut = policy.build_action_lut()
            with torch.inference_mode():
                dummy_mask = torch.zeros(
                    warmup_n, policy.unified_dim, dtype=torch.bool, device=device,
                )
                for i in range(warmup_n):
                    n = policy.phase_sizes[i]
                    dummy_mask[i, lut[i, :n].to(device)] = True
                # Tokens, or a packed layout's vectors, which its model's
                # clamped lookups take at random.
                dummy_tokens = torch.randn(
                    warmup_n, *model_input_spec.input_shape, device=device,
                )
                # The trainer and NNEvaluator send this one format. A flat
                # game's N comes from its records: value-0 rows, with N past 1
                # so that ``mark_unbacked`` leaves it unspecialized. A packed
                # layout has no relations: its forward is (inputs, mask).
                relation_format = game.relation_format_of(model_input_spec.layout_version)
                dummy_relations = (
                    torch.zeros(
                        2 * warmup_n, FLAT_RELATION_RECORD_WIDTH,
                        dtype=torch.int16, device=device,
                    )
                    if relation_format == "flat"
                    else torch.zeros(
                        warmup_n, game.num_relations, num_tokens, num_tokens,
                        dtype=torch.uint8, device=device,
                    )
                    if relation_format == "dense"
                    else None
                )
                inputs = (
                    (dummy_tokens, dummy_mask) if dummy_relations is None
                    else (dummy_tokens, dummy_mask, dummy_relations)
                )
                for _t in inputs:
                    mark_unbacked(_t, 0)
                model(*inputs)
                del dummy_tokens, dummy_relations, inputs
                del dummy_mask
            torch.cuda.synchronize()
            print("  Model compiled.")

    # --- Trainer (receives possibly-compiled model) ---
    trainer = Trainer(model, config, device)

    # --- Resume: restore trainer state (optimizer + scheduler) ---
    if cp is not None:
        trainer.load_state_dict(cp["trainer_state"])  # type: ignore[arg-type]
        print(
            f"Resumed from epoch {cp['epoch']}, "
            f"step {trainer.global_step}"
        )

    # --- Single-process evaluator (receives possibly-compiled model) ---
    evaluator: NNEvaluator | None = None
    state_pool: StatePool | None = None

    if config.num_workers == 0:
        evaluator = NNEvaluator(
            game, model, device, num_players=max_players,
            terminal_rank_weight=config.terminal_blend,
            terminal_rank_tau=config.terminal_rank_tau,
            eval_dtype=config.eval_dtype,
            input_spec=model_input_spec,
        )
        state_pool = StatePool(
            game, 2 * (config.max_simulations + 1), state_size_int16,
        )

    # --- Training loop ---
    avg_losses: dict[str, float] = {}

    try:
        for epoch in range(start_epoch, config.num_epochs):
            epoch_start = time.perf_counter()
            epoch_num = epoch + 1  # 1-indexed for display

            # --- Phase 1: Self-play ---
            model.eval()
            self_play_metrics = _EpochSelfPlayMetrics()
            opening_lengths = _OpeningGameLengths()
            game_profiles: list[GameProfileData] = []

            def _collect_record(record: object, game_idx: int) -> None:
                buffer.add_stacked(  # type: ignore[union-attr]
                    record.states,  # type: ignore[union-attr]
                    record.phase_ids,  # type: ignore[union-attr]
                    record.legal_masks,  # type: ignore[union-attr]
                    record.policy_targets,  # type: ignore[union-attr]
                    record.value_targets,  # type: ignore[union-attr]
                    num_players=record.num_players,  # type: ignore[union-attr]
                    priorities=record.policy_priorities,  # type: ignore[union-attr]
                    flags=record.flags,  # type: ignore[union-attr]
                    net_worths=record.net_worths,  # type: ignore[union-attr]
                )
                self_play_metrics.add_record(record)
                opening_lengths.add_record(record)
                if record.profile is not None:  # type: ignore[union-attr]
                    game_profiles.append(record.profile)  # type: ignore[union-attr]
                # The console counts every game, started or not.
                aggregate_stats = self_play_metrics.every.aggregate_snapshot()
                count_stats = self_play_metrics.every.count_snapshots()
                logger.update_self_play(
                    games_done=game_idx + 1,
                    total_examples=int(aggregate_stats["examples"]),
                    avg_moves=float(aggregate_stats["avg_moves"]),
                    rank_net_worths=list(aggregate_stats["rank_net_worths"]),
                    rank_mins=list(aggregate_stats["rank_net_worths_min"]),
                    rank_maxs=list(aggregate_stats["rank_net_worths_max"]),
                    target_entropy=float(aggregate_stats["policy_target_entropy"]),
                    target_top1_frac=float(
                        aggregate_stats["policy_target_top1_frac"]
                    ),
                    sample_entropy=float(aggregate_stats["sample_policy_entropy"]),
                    sample_top1_frac=float(aggregate_stats["sample_top1_frac"]),
                    count_rank_net_worths={
                        n: list(stats["rank_net_worths"])
                        for n, stats in count_stats.items()
                    },
                    count_rank_mins={
                        n: list(stats["rank_net_worths_min"])
                        for n, stats in count_stats.items()
                    },
                    count_rank_maxs={
                        n: list(stats["rank_net_worths_max"])
                        for n, stats in count_stats.items()
                    },
                    count_games={
                        n: int(stats["games"])
                        for n, stats in count_stats.items()
                    },
                    count_avg_moves={
                        n: float(stats["avg_moves"])
                        for n, stats in count_stats.items()
                    },
                    count_total_net_worths={
                        n: float(stats["total_net_worth"])
                        for n, stats in count_stats.items()
                    },
                )

            # Eval-server stats cover this epoch's self-play alone.
            for es in eval_servers:
                es.reset_profile_stats()

            # Compute per-epoch annealing parameters
            epoch_cfg = config.compute_epoch_config(epoch, game_lengths)

            logger.begin_self_play(epoch_num, config.num_epochs, config.games_per_epoch)

            games_collected = 0
            player_count_schedule = build_epoch_player_count_schedule(config)
            # Each game's start position, -1 for the opening. The draw leaves
            # master_rng alone: seeds are the same with starts or without.
            start_schedule = build_epoch_start_schedule(
                player_count_schedule,
                start_positions.num_players if start_positions is not None else None,
                config.human_start_fraction, config.seed, epoch,
            )
            buffer.begin_epoch(epoch)

            def _game_start(game_idx: int) -> tuple[np.ndarray, int] | None:
                position = start_schedule[game_idx]
                if position < 0:
                    return None
                assert start_positions is not None
                return start_positions.rows[position], int(start_positions.start_moves[position])

            if config.num_workers > 0:
                assert task_queue is not None and result_queue is not None
                # Feed all game assignments to workers (with epoch config).
                for game_idx, num_players in enumerate(player_count_schedule):
                    game_seed = int(master_rng.integers(0, 2**31))
                    rng_seed = int(master_rng.integers(0, 2**63))
                    task = (game_seed, rng_seed, num_players, epoch_cfg)
                    # A started game's task carries its row and move number.
                    task_queue.put(task + (_game_start(game_idx) or ()))

                # Collect results as they complete
                for game_idx in range(config.games_per_epoch):
                    try:
                        record = get_worker_result(
                            result_queue, eval_servers, timeout=1200.0,
                        )
                    except queue.Empty:
                        alive = sum(1 for w in workers if w.is_alive())
                        raise RuntimeError(
                            f"Timed out waiting for game result "
                            f"({alive}/{config.num_workers} workers alive)"
                        )
                    _collect_record(record, game_idx)
                    games_collected = game_idx + 1
                    # Min-batch mode drains partial accumulations via its
                    # own doorbell timeout, so no epoch-end flag handshake
                    # is needed here.
                    if shutdown_event.is_set():
                        logger.end_self_play()
                        print(
                            "\nGraceful shutdown requested "
                            "— draining in-flight games..."
                        )
                        games_collected = _drain_workers(
                            task_queue, result_queue, workers,
                            _collect_record, games_collected,
                        )
                        break
            else:
                assert evaluator is not None
                for game_idx, num_players in enumerate(player_count_schedule):
                    game_seed = int(master_rng.integers(0, 2**31))
                    game_rng = np.random.default_rng(master_rng.integers(0, 2**63))

                    start_row, start_move = _game_start(game_idx) or (None, 0)
                    record = play_game(
                        evaluator, config, game_seed, game_rng,
                        state_pool=state_pool, epoch_config=epoch_cfg,
                        num_players=num_players,
                        start_row=start_row, start_move=start_move,
                    )
                    _collect_record(record, game_idx)
                    games_collected = game_idx + 1
                    if shutdown_event.is_set():
                        break

            logger.end_self_play()  # idempotent if already stopped

            # --- Graceful shutdown: save state and exit ---
            if shutdown_event.is_set():
                if config.num_workers == 0:
                    print("\nGraceful shutdown requested...")

                did_train = False
                if (
                    games_collected >= config.games_per_epoch
                    and buffer.window_size >= config.min_buffer_size
                ):
                    # Met quota — run scaled training phase before exiting
                    print("Game quota met — running training before shutdown...")
                    heldout = _heldout_scalars(trainer, buffer, config, epoch, game.policy.phase_names)
                    logger.log_scalars(epoch_num, heldout)
                    model.train()
                    shutdown_losses: dict[str, list[float]] = defaultdict(list)
                    training_steps = _epoch_training_steps(config, buffer)
                    trainer.begin_epoch(epoch, training_steps)
                    logger.begin_training(
                        epoch_num, config.num_epochs,
                        training_steps,
                    )
                    for step in range(training_steps):
                        losses = trainer.train_step(
                            buffer, config.batch_size, master_rng,
                        )
                        for k, v in losses.items():
                            shutdown_losses[k].append(v)
                        logger.update_training(step + 1, losses, trainer.lr)
                    logger.end_training()
                    avg_losses = average_training_metrics(shutdown_losses)
                    logger.log_scalars(epoch_num, _heldout_over_train(heldout, avg_losses))
                    did_train = True
                    game_lengths.update(opening_lengths.means())

                # Save checkpoint. Untrained, the epoch is replayed on resume,
                # under the game lengths it began with.
                save_epoch = epoch if did_train else epoch - 1
                shutdown_cp = (
                    Path(config.checkpoint_dir)
                    / f"checkpoint_epoch_{save_epoch + 1:04d}.pt"
                )
                save_checkpoint(
                    path=shutdown_cp,
                    epoch=save_epoch,
                    model=model,
                    trainer_state=trainer.state_dict(),
                    config=config,
                    metrics=avg_losses,
                    buffer_stats={
                        "size": len(buffer),
                        "capacity": config.buffer_capacity,
                    },
                    rng_state=_capture_rng_state(master_rng),
                    game_lengths=game_lengths,
                )

                # Save replay buffer
                buf_dir = Path(config.checkpoint_dir) / "replay_buffer"
                print(f"Saving replay buffer ({len(buffer):,} examples)...")
                buffer.save(buf_dir)

                print(
                    f"\nShutdown complete:"
                    f"\n  Checkpoint: {shutdown_cp}"
                    f"\n  Replay buffer: {buf_dir}/"
                    f"\n  Games collected: {games_collected}"
                    f"\n  Buffer size: {len(buffer):,}"
                    f"\n  Trained: {'yes' if did_train else 'no'}"
                    f"\n  Resume with: python -m train --resume latest"
                    f" --checkpoint-dir {config.checkpoint_dir}"
                )
                break

            # The next epoch's temperature windows read this one's lengths; a
            # player count with no game from the opening keeps its last.
            game_lengths.update(opening_lengths.means())

            # Self-play Tensorboard metrics
            self_play_stats = self_play_metrics.every.aggregate_snapshot()
            self_play_by_count_stats = self_play_metrics.every.count_snapshots()
            logger.log_scalars(
                epoch_num,
                {
                    "buffer/size": float(len(buffer)),
                    "buffer/utilization": len(buffer) / config.buffer_capacity,
                    **(
                        {
                            "buffer/window_size": float(buffer.window_size),
                            "buffer/window_epochs": float(buffer.window_epochs),
                        }
                        if config.buffer_epochs > 0 else {}
                    ),
                    **{
                        f"buffer/{key}": value
                        for key, value in buffer.priority_stats().items()
                    },
                    "schedule/c_puct": epoch_cfg.c_puct,
                    "schedule/value_blend_alpha": epoch_cfg.value_blend_alpha,
                    "schedule/num_simulations": epoch_cfg.num_simulations,
                    **{
                        f"schedule/temp_anneal_{edge}_{num_players}p": float(move)
                        for num_players, window in (epoch_cfg.temp_anneal_windows or {}).items()
                        for edge, move in zip(("start", "end"), window)
                    },
                    **self_play_metrics.scalars(),
                },
            )

            # --- Eval-server stats, and the profile summary ---
            sp_duration = time.perf_counter() - epoch_start
            server_stats: EvalServerStats | None = None
            all_stats = [
                s for s in (es.get_profile_stats() for es in eval_servers)
                if s is not None
            ]
            if all_stats:
                server_stats = EvalServerStats.merge(all_stats)
                logger.log_scalars(epoch_num, _build_eval_server_scalars(server_stats))
            if config.profile and game_profiles:
                print(format_epoch_profile(game_profiles, server_stats, sp_duration))

                # Tensorboard: profile scalars
                profile_scalars = _build_profile_scalars(
                    game_profiles, server_stats, sp_duration,
                    num_eval_servers=config.num_eval_servers,
                )
                logger.log_scalars(epoch_num, profile_scalars)

            # --- Held-out loss: the epoch's new rows, before it trains on them ---
            heldout = _heldout_scalars(trainer, buffer, config, epoch, game.policy.phase_names)
            logger.log_scalars(epoch_num, heldout)

            # --- Phase 2: Training ---
            training_steps_done = 0
            if buffer.window_size >= config.min_buffer_size:
                model.train()
                epoch_losses: dict[str, list[float]] = defaultdict(list)
                training_steps = _epoch_training_steps(config, buffer)
                trainer.begin_epoch(epoch, training_steps)
                training_steps_done = training_steps

                logger.begin_training(
                    epoch_num, config.num_epochs, training_steps
                )

                for step in range(training_steps):
                    # train_step samples directly into pinned host scratch
                    # and raises on NaN in policy / value targets.
                    losses = trainer.train_step(
                        buffer, config.batch_size, master_rng,
                    )
                    for k, v in losses.items():
                        epoch_losses[k].append(v)

                    logger.update_training(step + 1, losses, trainer.lr)

                    # Per-step Tensorboard
                    if (step + 1) % config.log_interval == 0:
                        logger.log_scalars(
                            trainer.global_step,
                            {
                                "loss/total": losses["total_loss"],
                                "loss/policy": losses["policy_loss"],
                                "loss/value": losses["value_loss"],
                                "loss/policy_target_entropy": losses[
                                    "policy_target_entropy"
                                ],
                                "loss/policy_loss_residual": losses[
                                    "policy_loss_residual"
                                ],
                                "loss/policy_kl": losses["policy_kl"],
                                "lr": trainer.lr,
                            },
                        )

                logger.end_training()
                if eval_servers:
                    print("Syncing eval server weights...")
                    _sync_eval_servers(eval_servers, model)

                avg_losses = average_training_metrics(epoch_losses)

                # Epoch-level Tensorboard
                epoch_scalars = {
                    "epoch/total_loss_avg": avg_losses["total_loss"],
                    "epoch/policy_loss_avg": avg_losses["policy_loss"],
                    "epoch/value_loss_avg": avg_losses["value_loss"],
                    "epoch/policy_target_entropy_avg": avg_losses[
                        "policy_target_entropy"
                    ],
                    "epoch/policy_loss_residual_avg": avg_losses[
                        "policy_loss_residual"
                    ],
                    "epoch/policy_kl_avg": avg_losses["policy_kl"],
                    "epoch/training_steps": float(training_steps_done),
                    **_heldout_over_train(heldout, avg_losses),
                }
                if "grad_norm" in avg_losses:
                    epoch_scalars["epoch/grad_norm_avg"] = avg_losses["grad_norm"]
                    epoch_scalars["epoch/grad_norm_max"] = avg_losses["grad_norm_max"]
                    epoch_scalars["epoch/grad_clipped_share"] = avg_losses["grad_clipped"]
                for k, v in avg_losses.items():
                    if k.startswith((
                        "policy_loss_",
                        "value_loss_",
                        "value_raw_",
                        "pass_logit_abs_",
                        "action_logit_abs_",
                        "policy_target_entropy_",
                        "policy_kl_",
                    )):
                        epoch_scalars[f"epoch/{k}_avg"] = v
                    elif k.startswith("policy_samples_"):
                        epoch_scalars[f"epoch/{k}"] = v
                logger.log_scalars(epoch_num, epoch_scalars)
            else:
                avg_losses = {}
                print(
                    f"  Skipping training: buffer has {buffer.window_size} examples "
                    f"(need {config.min_buffer_size})"
                )

            # --- Phase 3: Checkpoint ---
            checkpoint_path: str | None = None
            if epoch_num % config.checkpoint_interval == 0:
                cp_path = (
                    Path(config.checkpoint_dir) / f"checkpoint_epoch_{epoch_num:04d}.pt"
                )
                save_checkpoint(
                    path=cp_path,
                    epoch=epoch,
                    model=model,
                    trainer_state=trainer.state_dict(),
                    config=config,
                    metrics=avg_losses,
                    buffer_stats={"size": len(buffer), "capacity": config.buffer_capacity},
                    rng_state=_capture_rng_state(master_rng),
                    game_lengths=game_lengths,
                )
                buffer.save(buffer_dir)
                cleanup_checkpoints(Path(config.checkpoint_dir), config.keep_last_n)
                checkpoint_path = str(cp_path)

            # --- Phase 4: Epoch summary ---
            epoch_duration = time.perf_counter() - epoch_start
            logger.log_scalars(epoch_num, {"epoch/duration_secs": epoch_duration})
            base_model = getattr(model, "_orig_mod", model)
            diagnostic = getattr(base_model, "phase_mod_diagnostics", None)
            if callable(diagnostic):
                diagnostics = diagnostic()
                if not isinstance(diagnostics, dict):
                    raise TypeError("phase_mod_diagnostics must return a dict")
                if diagnostics:
                    logger.log_scalars(epoch_num, diagnostics)
            logger.log_epoch_summary(
                epoch=epoch_num,
                num_epochs=config.num_epochs,
                self_play_stats={
                    **self_play_stats,
                    "by_player_count": self_play_by_count_stats,
                },
                train_stats={
                    "steps": float(training_steps_done) if avg_losses else 0.0,
                    "lr": trainer.lr,
                    **avg_losses,
                },
                buffer_size=len(buffer),
                buffer_capacity=config.buffer_capacity,
                epoch_duration=epoch_duration,
                checkpoint_path=checkpoint_path,
            )

        # --- Final checkpoint (skip if we exited via graceful shutdown) ---
        final_cp = Path(config.checkpoint_dir) / f"checkpoint_epoch_{config.num_epochs:04d}.pt"
        if not shutdown_event.is_set() and not final_cp.exists():
            save_checkpoint(
                final_cp,
                config.num_epochs - 1,
                model,
                trainer.state_dict(),
                config,
                avg_losses,
                {"size": len(buffer), "capacity": config.buffer_capacity},
                rng_state=_capture_rng_state(master_rng),
                game_lengths=game_lengths,
            )
            buffer.save(buffer_dir)

        print("\nTraining complete.")

    except KeyboardInterrupt:
        print("\nInterrupted — shutting down...")

    finally:
        # Clean shutdown: stop eval servers, terminate workers
        for es in eval_servers:
            es.stop()
        for w in workers:
            w.join(timeout=3.0)
            if w.is_alive():
                w.terminate()
        logger.close()
