"""Main training orchestration loop."""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.multiprocessing as mp
from torch._dynamo.decorators import mark_unbacked

from core.attention_relations import NUM_ATTENTION_RELATIONS
from core.data import PHASE_ACTION_SIZES
from core.state import get_layout
from core.token_data import TokenDataSize, get_num_tokens
from nn import create_model, get_model_input_spec
from nn.model_contract import ModelKind
from nn.transformer import (
    NUM_PHASES,
    UNIFIED_LOGIT_DIM,
    build_action_lut,
)
from train.checkpoint import (
    cleanup_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from train.config import TrainingConfig
from train.eval_server import EvaluationServer, SharedEvalBuffers
from train.logging import TrainingLogger
from train.profile_stats import EvalServerStats, GameProfileData, format_epoch_profile
from train.replay_buffer import ReplayBuffer
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool
from train.self_play import play_game, self_play_worker
from train.trainer import Trainer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AlphaZero self-play training for Rolling Stock Stars"
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
        choices=["transformer", "resnet"],
        help="Model family to instantiate",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Optional Python module/file path that provides the selected model implementation",
    )
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
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--num-eval-servers", type=int)
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
    parser.add_argument("--temp-final", type=float)
    parser.add_argument("--policy-target-temp-initial", type=float)
    parser.add_argument("--policy-target-temp-anneal-start", type=int)
    parser.add_argument("--policy-target-temp-anneal-end", type=int)
    parser.add_argument("--policy-target-temp-final", type=float)
    parser.add_argument("--c-puct-initial", type=float)
    parser.add_argument("--c-puct-final", type=float)
    parser.add_argument("--c-puct-anneal-epochs", type=int)
    parser.add_argument("--value-blend-start-epoch", type=int)
    parser.add_argument("--value-blend-end-epoch", type=int)
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
        "--profile", action="store_true", default=None,
        help="Enable per-epoch self-play performance profiling",
    )
    parser.add_argument(
        "--no-compile", action="store_true", default=False,
        help="Disable torch.compile model optimization",
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
        help="Enable per-block adaLN phase conditioning",
    )
    phase_group.add_argument(
        "--no-phase-conditioning",
        dest="phase_conditioning",
        action="store_false",
        help="Disable per-block adaLN phase conditioning",
    )
    parser.add_argument(
        "--price-slot-fourier-bands",
        type=int,
        help="Number of fixed Fourier bands for price-like policy slot keys",
    )
    parser.add_argument(
        "--price-slot-residual-scale",
        type=float,
        help="Blend weight for learned price-slot embeddings: 0=pure Fourier, 1=pure embedding",
    )
    parser.add_argument("--resnet-hidden-dim", type=int)
    parser.add_argument("--resnet-num-blocks", type=int)
    return parser


_CLI_FIELDS = (
    "num_players", "min_players", "max_players",
    "eval_dtype", "model_type", "model_path", "phase_conditioning",
    "price_slot_fourier_bands", "price_slot_residual_scale",
    "resnet_hidden_dim", "resnet_num_blocks",
    "games_per_epoch", "num_epochs", "training_steps_per_epoch",
    "num_simulations", "search_batch_size",
    "mcts_sims_start", "mcts_sims_end", "mcts_ramp_start_epoch", "mcts_ramp_end_epoch",
    "num_workers", "num_eval_servers",
    "buffer_capacity",
    "eval_min_batch_size", "eval_min_batch_timeout_ms",
    "eval_batch_shape_mode", "eval_max_batch_size",
    "checkpoint_dir", "tensorboard_dir", "seed",
    "temp_initial", "temp_anneal_start", "temp_anneal_end", "temp_final",
    "policy_target_temp_initial", "policy_target_temp_anneal_start",
    "policy_target_temp_anneal_end", "policy_target_temp_final",
    "c_puct_initial", "c_puct_final", "c_puct_anneal_epochs",
    "value_blend_start_epoch", "value_blend_end_epoch",
    "terminal_blend", "lr_min", "warmup_epochs", "lr_decay_end_epoch",
    "optimizer", "weight_decay", "grad_clip",
    "dirichlet_alpha", "dirichlet_epsilon",
    "dirichlet_dynamic", "dirichlet_alpha_numerator",
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
        self.total_shares = 0.0
        self.total_companies = 0.0
        self.avg_active_corp_price = 0.0
        self.corps_in_receivership = 0.0
        self.games_with_max_price_corp = 0

        self.rank_counts: list[int] = []
        self.rank_net_worths: list[float] = []
        self.rank_net_worth_mins: list[float] = []
        self.rank_net_worth_maxs: list[float] = []
        self.rank_shares: list[float] = []
        self.rank_companies: list[float] = []
        self.rank_pres_share_values: list[float] = []
        self.rank_nw_cash_pct: list[float] = []
        self.rank_nw_companies_pct: list[float] = []
        self.rank_nw_shares_pct: list[float] = []

    def _ensure_rank_capacity(self, num_ranks: int) -> None:
        while len(self.rank_counts) < num_ranks:
            self.rank_counts.append(0)
            self.rank_net_worths.append(0.0)
            self.rank_net_worth_mins.append(float("inf"))
            self.rank_net_worth_maxs.append(float("-inf"))
            self.rank_shares.append(0.0)
            self.rank_companies.append(0.0)
            self.rank_pres_share_values.append(0.0)
            self.rank_nw_cash_pct.append(0.0)
            self.rank_nw_companies_pct.append(0.0)
            self.rank_nw_shares_pct.append(0.0)

    def add_record(self, record: Any) -> None:
        num_players = int(record.num_players)
        self._ensure_rank_capacity(num_players)

        self.games += 1
        self.examples += int(record.num_examples)
        self.moves += int(record.total_moves)
        self.duration += float(record.duration_secs)
        self.target_entropy += float(record.policy_target_entropy_mean)
        self.target_top1 += float(record.policy_target_top1_fraction)
        self.sample_entropy += float(record.sample_policy_entropy_mean)
        self.sample_top1 += float(record.sample_top1_action_fraction)
        self.total_net_worth += float(sum(record.net_worths[:num_players]))
        self.total_shares += float(sum(record.shares_per_player[:num_players]))
        self.total_companies += float(sum(record.companies_per_player[:num_players]))
        self.avg_active_corp_price += float(record.avg_active_corp_price)
        self.corps_in_receivership += float(record.corps_in_receivership)
        if record.has_max_price_corp:
            self.games_with_max_price_corp += 1

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
            self.rank_shares[rank] += float(record.shares_per_player[player_id])
            self.rank_companies[rank] += float(record.companies_per_player[player_id])
            self.rank_pres_share_values[rank] += float(
                record.pres_share_values[player_id]
            )
            self.rank_nw_cash_pct[rank] += float(record.nw_cash_pct[player_id])
            self.rank_nw_companies_pct[rank] += float(
                record.nw_companies_pct[player_id]
            )
            self.rank_nw_shares_pct[rank] += float(record.nw_shares_pct[player_id])

    def snapshot(self) -> dict[str, Any]:
        games = self.games
        if games == 0:
            return {
                "games": 0.0,
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
                "avg_shares_per_player": [],
                "avg_companies_per_player": [],
                "avg_pres_share_values": [],
                "avg_nw_cash_pct": [],
                "avg_nw_companies_pct": [],
                "avg_nw_shares_pct": [],
                "total_shares": 0.0,
                "total_companies": 0.0,
                "avg_active_corp_price": 0.0,
                "corps_in_receivership": 0.0,
                "pct_games_max_price_corp": 0.0,
            }

        def rank_avg(values: list[float]) -> list[float]:
            return [
                value / count if count > 0 else 0.0
                for value, count in zip(values, self.rank_counts)
            ]

        return {
            "games": float(games),
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
            "avg_shares_per_player": rank_avg(self.rank_shares),
            "avg_companies_per_player": rank_avg(self.rank_companies),
            "avg_pres_share_values": rank_avg(self.rank_pres_share_values),
            "avg_nw_cash_pct": rank_avg(self.rank_nw_cash_pct),
            "avg_nw_companies_pct": rank_avg(self.rank_nw_companies_pct),
            "avg_nw_shares_pct": rank_avg(self.rank_nw_shares_pct),
            "total_shares": self.total_shares / games,
            "total_companies": self.total_companies / games,
            "avg_active_corp_price": self.avg_active_corp_price / games,
            "corps_in_receivership": self.corps_in_receivership / games,
            "pct_games_max_price_corp": self.games_with_max_price_corp / games,
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
        f"{prefix}/total_shares": float(stats.get("total_shares", 0.0)),
        f"{prefix}/total_companies": float(stats.get("total_companies", 0.0)),
        f"{prefix}/avg_active_corp_price": float(
            stats.get("avg_active_corp_price", 0.0),
        ),
        f"{prefix}/corps_in_receivership": float(
            stats.get("corps_in_receivership", 0.0),
        ),
        f"{prefix}/pct_games_max_price_corp": float(
            stats.get("pct_games_max_price_corp", 0.0),
        ),
    }

    rank_net_worths = list(stats.get("rank_net_worths", []))
    rank_mins = list(stats.get("rank_net_worths_min", []))
    rank_maxs = list(stats.get("rank_net_worths_max", []))
    avg_shares = list(stats.get("avg_shares_per_player", []))
    avg_companies = list(stats.get("avg_companies_per_player", []))
    avg_pres_share_values = list(stats.get("avg_pres_share_values", []))
    avg_nw_cash_pct = list(stats.get("avg_nw_cash_pct", []))
    avg_nw_companies_pct = list(stats.get("avg_nw_companies_pct", []))
    avg_nw_shares_pct = list(stats.get("avg_nw_shares_pct", []))

    for rank, avg in enumerate(rank_net_worths):
        label = _RANK_LABELS[rank]
        scalars[f"{prefix}/net_worth_{label}"] = float(avg)
        if rank < len(rank_mins):
            scalars[f"{prefix}/net_worth_{label}_min"] = float(rank_mins[rank])
        if rank < len(rank_maxs):
            scalars[f"{prefix}/net_worth_{label}_max"] = float(rank_maxs[rank])
    for rank, shares in enumerate(avg_shares):
        label = _RANK_LABELS[rank]
        scalars[f"{prefix}/shares_{label}"] = float(shares)
        if rank < len(avg_companies):
            scalars[f"{prefix}/companies_{label}"] = float(avg_companies[rank])
        if rank < len(avg_pres_share_values):
            scalars[f"{prefix}/pres_share_value_{label}"] = float(
                avg_pres_share_values[rank]
            )
        if rank < len(avg_nw_cash_pct):
            scalars[f"{prefix}/nw_cash_pct_{label}"] = float(avg_nw_cash_pct[rank])
        if rank < len(avg_nw_companies_pct):
            scalars[f"{prefix}/nw_companies_pct_{label}"] = float(
                avg_nw_companies_pct[rank]
            )
        if rank < len(avg_nw_shares_pct):
            scalars[f"{prefix}/nw_shares_pct_{label}"] = float(
                avg_nw_shares_pct[rank]
            )

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


def _scaled_training_steps(config: TrainingConfig, buffer_size: int) -> int:
    """Scale per-epoch training updates by replay-buffer fullness.

    Early epochs have less data diversity in replay. Scale the configured
    update budget by the occupied fraction of the buffer so, for example,
    100k rows in a 500k buffer runs 20% of the configured steps.
    """
    if buffer_size < config.min_buffer_size or config.training_steps_per_epoch < 1:
        return 0
    capped_size = min(max(buffer_size, 0), config.buffer_capacity)
    scaled = config.training_steps_per_epoch * capped_size // config.buffer_capacity
    return max(1, scaled)


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

    Clears remaining seeds from the task queue, sends None sentinels to
    workers, and collects results until all workers have exited or the
    timeout is reached.

    Returns the total number of games collected (including pre-drain).
    """
    # Clear remaining seeds from task queue
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
        print(f"  Cleared {cleared} pending game seeds from queue")

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
    cp: dict[str, object] | None = None
    if args.resume:
        cp_path: Path | None = None
        if args.resume == "latest":
            cp_dir = args.checkpoint_dir or "checkpoints"
            cp_path = find_latest_checkpoint(Path(cp_dir))
            if cp_path is None:
                print("No checkpoint found, starting from scratch.")
        else:
            cp_path = Path(args.resume)

        if cp_path is not None:
            cp = load_checkpoint(cp_path, device)
            print(f"Loaded checkpoint: {cp_path}")

    # --- Config ---
    if cp is not None:
        # Resume: checkpoint config as base, then JSON overrides, then CLI overrides
        config = TrainingConfig.from_json(cp["config_json"])  # type: ignore[arg-type]
        if args.config:
            changes = config.apply_json_overrides(Path(args.config).read_text())
            for c in changes:
                print(f"  Config override: {c}")
        _apply_overrides(config, args, log_changes=True)
        config.validate()
    elif args.config:
        config = TrainingConfig.from_json(Path(args.config).read_text())
        _apply_overrides(config, args)
        config.validate()
    else:
        config = TrainingConfig()
        _apply_overrides(config, args)
        config.validate()

    # --- Operational flags (not persisted in config JSON) ---
    if args.profile:
        config.profile = True

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
    if cp is not None:
        model.load_state_dict(cp["model_state_dict"])  # type: ignore[arg-type]
        start_epoch = cp["epoch"] + 1  # type: ignore[operator]

    # --- Components (model-independent) ---
    # Compact int16 state — replay buffer stores raw GameState rows, the
    # trainer builds token buffers per-batch via core.token_data.
    max_players = config.effective_max_players
    state_size_int16 = get_layout(max_players).total_size
    num_tokens = get_num_tokens(max_players)
    token_dim = int(TokenDataSize.TOKEN_DIM)
    buffer = ReplayBuffer(
        config.buffer_capacity,
        state_size_int16,
        max_players,
        min_players=config.effective_min_players,
        max_players=max_players,
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
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        task_queue.cancel_join_thread()
        result_queue = ctx.Queue()
        result_queue.cancel_join_thread()

        shared_bufs = SharedEvalBuffers(
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
            server = EvaluationServer(
                model, device, shared_bufs,
                server_id=i,
                worker_start=ws,
                worker_end=we,
                profile=config.profile,
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
        for server in eval_servers:
            if not server.wait_ready(timeout=2400.0):
                raise RuntimeError(
                    "Eval server did not become ready within 300s — "
                    "compilation may have failed (check stderr)"
                )
        print(f"  {n_servers} eval server{'s' if n_servers > 1 else ''} ready.")

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
            # NUM_PHASES rows so every per-row policy head is traced against
            # a real legal mask, and
            # ``mark_unbacked`` on every model input so the batch dim stays
            # symbolic. If any input is left unmarked, Dynamo bakes a static
            # guard on that input's shape and recompiles on the first
            # batch != warmup_n; warming up at batch=1 also wastes a
            # specialization that the runtime never reuses.
            warmup_n = NUM_PHASES
            lut = build_action_lut()
            with torch.inference_mode():
                dummy_mask = torch.zeros(
                    warmup_n, UNIFIED_LOGIT_DIM, dtype=torch.bool, device=device,
                )
                for i in range(warmup_n):
                    n = PHASE_ACTION_SIZES[i]
                    dummy_mask[i, lut[i, :n].to(device)] = True
                if config.model_type == ModelKind.TRANSFORMER.value:
                    dummy_tokens = torch.randn(
                        warmup_n, num_tokens, token_dim, device=device,
                    )
                    dummy_relations = torch.zeros(
                        warmup_n, NUM_ATTENTION_RELATIONS, num_tokens, num_tokens,
                        dtype=torch.uint8, device=device,
                    )
                    for _t in (dummy_tokens, dummy_mask, dummy_relations):
                        mark_unbacked(_t, 0)
                    model(dummy_tokens, dummy_mask, dummy_relations)
                    del dummy_tokens, dummy_relations
                else:
                    base_model = getattr(model, "_orig_mod", model)
                    input_dim = int(getattr(base_model.cfg, "input_dim"))
                    dummy_vectors = torch.randn(warmup_n, input_dim, device=device)
                    for _t in (dummy_vectors, dummy_mask):
                        mark_unbacked(_t, 0)
                    model(dummy_vectors, dummy_mask)
                    del dummy_vectors
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
            model, device, num_players=max_players,
            terminal_rank_weight=config.terminal_blend,
            eval_dtype=config.eval_dtype,
            input_spec=model_input_spec,
        )
        state_pool = StatePool(
            2 * (config.max_simulations + 1), state_size_int16,
        )

    # --- Training loop ---
    avg_losses: dict[str, float] = {}

    try:
        for epoch in range(start_epoch, config.num_epochs):
            epoch_start = time.perf_counter()
            epoch_num = epoch + 1  # 1-indexed for display

            # --- Phase 1: Self-play ---
            model.eval()
            self_play_metrics = _SelfPlayMetricAccumulator()
            game_profiles: list[GameProfileData] = []

            def _collect_record(record: object, game_idx: int) -> None:
                buffer.add_stacked(  # type: ignore[union-attr]
                    record.states,  # type: ignore[union-attr]
                    record.phase_ids,  # type: ignore[union-attr]
                    record.legal_masks,  # type: ignore[union-attr]
                    record.policy_targets,  # type: ignore[union-attr]
                    record.value_targets,  # type: ignore[union-attr]
                    num_players=record.num_players,  # type: ignore[union-attr]
                )
                self_play_metrics.add_record(record)
                if record.profile is not None:  # type: ignore[union-attr]
                    game_profiles.append(record.profile)  # type: ignore[union-attr]
                aggregate_stats = self_play_metrics.aggregate_snapshot()
                count_stats = self_play_metrics.count_snapshots()
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
                    count_total_net_worths={
                        n: float(stats["total_net_worth"])
                        for n, stats in count_stats.items()
                    },
                )

            # Reset eval server profile stats
            if config.profile and eval_servers:
                for es in eval_servers:
                    es.reset_profile_stats()

            # Compute per-epoch annealing parameters
            epoch_cfg = config.compute_epoch_config(epoch)

            logger.begin_self_play(epoch_num, config.num_epochs, config.games_per_epoch)

            games_collected = 0

            if config.num_workers > 0:
                assert task_queue is not None and result_queue is not None
                # Feed all game seeds to workers (with epoch config)
                for _ in range(config.games_per_epoch):
                    game_seed = int(master_rng.integers(0, 2**31))
                    rng_seed = int(master_rng.integers(0, 2**63))
                    task_queue.put((game_seed, rng_seed, epoch_cfg))

                # Collect results as they complete
                for game_idx in range(config.games_per_epoch):
                    try:
                        record = result_queue.get(timeout=1200.0)
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
                for game_idx in range(config.games_per_epoch):
                    game_seed = int(master_rng.integers(0, 2**31))
                    game_rng = np.random.default_rng(master_rng.integers(0, 2**63))

                    record = play_game(
                        evaluator, config, game_seed, game_rng,
                        state_pool=state_pool, epoch_config=epoch_cfg,
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
                    and len(buffer) >= config.min_buffer_size
                ):
                    # Met quota — run scaled training phase before exiting
                    print("Game quota met — running training before shutdown...")
                    model.train()
                    shutdown_losses: dict[str, list[float]] = defaultdict(list)
                    training_steps = _scaled_training_steps(config, len(buffer))
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
                    avg_losses = {
                        k: sum(v) / len(v) for k, v in shutdown_losses.items()
                    }
                    did_train = True

                # Save checkpoint
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
                )
                break

            # Self-play Tensorboard metrics
            self_play_stats = self_play_metrics.aggregate_snapshot()
            self_play_by_count_stats = self_play_metrics.count_snapshots()
            logger.log_scalars(
                epoch_num,
                {
                    "buffer/size": float(len(buffer)),
                    "buffer/utilization": len(buffer) / config.buffer_capacity,
                    "schedule/c_puct": epoch_cfg.c_puct,
                    "schedule/value_blend_alpha": epoch_cfg.value_blend_alpha,
                    "schedule/num_simulations": epoch_cfg.num_simulations,
                    **_build_epoch_self_play_scalars(self_play_metrics),
                },
            )

            # --- Profile summary ---
            if config.profile and game_profiles:
                sp_duration = time.perf_counter() - epoch_start
                server_stats: EvalServerStats | None = None
                if eval_servers:
                    all_stats = [
                        s for s in (es.get_profile_stats() for es in eval_servers)
                        if s is not None
                    ]
                    if all_stats:
                        server_stats = EvalServerStats.merge(all_stats)
                print(format_epoch_profile(game_profiles, server_stats, sp_duration))

                # Tensorboard: profile scalars
                profile_scalars = _build_profile_scalars(
                    game_profiles, server_stats, sp_duration,
                    num_eval_servers=config.num_eval_servers,
                )
                logger.log_scalars(epoch_num, profile_scalars)

            # --- Phase 2: Training ---
            training_steps_done = 0
            if len(buffer) >= config.min_buffer_size:
                model.train()
                epoch_losses: dict[str, list[float]] = defaultdict(list)
                training_steps = _scaled_training_steps(config, len(buffer))
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

                avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}

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
                }
                for k, v in avg_losses.items():
                    if k.startswith((
                        "policy_loss_",
                        "value_loss_",
                        "pass_logit_abs_",
                        "action_logit_abs_",
                    )):
                        epoch_scalars[f"epoch/{k}_avg"] = v
                logger.log_scalars(epoch_num, epoch_scalars)
            else:
                avg_losses = {}
                print(
                    f"  Skipping training: buffer has {len(buffer)} examples "
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
                )
                buffer.save(buffer_dir)
                cleanup_checkpoints(Path(config.checkpoint_dir), config.keep_last_n)
                checkpoint_path = str(cp_path)

            # --- Phase 4: Epoch summary ---
            epoch_duration = time.perf_counter() - epoch_start
            logger.log_scalars(epoch_num, {"epoch/duration_secs": epoch_duration})
            base_model = getattr(model, "_orig_mod", model)
            diagnostics = base_model.phase_mod_diagnostics()
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
