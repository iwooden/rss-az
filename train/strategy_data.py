"""Offline strategy-data collection from checkpointed self-play.

This entry point reuses the same shared-memory eval server and self-play
workers as training, but asks ``play_game`` for the heavier per-decision
strategy trace and writes sharded ``.npz`` files for notebook analysis.
"""

from __future__ import annotations

import argparse
import copy
import json
import queue
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp

from core.data import (
    ALL_PAR_PRICES,
    COMPANY_NAMES,
    CORP_NAMES,
    PHASE_ACTION_SIZES,
)
from core.state import get_layout
from entities.company import COMPANIES
from entities.corp import CORPS
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool
from nn import create_model, get_model_input_spec
from nn.model_contract import ModelKind
from nn.transformer import UNIFIED_LOGIT_DIM, build_action_lut
from train.checkpoint import find_latest_checkpoint, load_checkpoint
from train.config import EpochConfig, TrainingConfig
from train.eval_server import EvaluationServer, RemoteEvaluator, SharedEvalBuffers
from train.self_play import GameRecord, play_game


def _parse_int_csv(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one player count")
    for value in values:
        if value not in (3, 4, 5):
            raise argparse.ArgumentTypeError(
                f"supported strategy counts are 3, 4, and 5; got {value}"
            )
    return sorted(set(values))


def _parse_str_csv(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated non-empty strings, got {value!r}"
        )
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect sharded strategy-analysis data from an RSS checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help='Checkpoint path, or "latest" to use --checkpoint-dir',
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help='Directory used when --checkpoint is "latest"',
    )
    parser.add_argument(
        "--output-dir",
        default="strategy_data",
        help="Directory for metadata.json and strategy_*_shard_*.npz files",
    )
    parser.add_argument(
        "--player-counts",
        type=_parse_int_csv,
        default=[3, 4, 5],
        help="Comma-separated actual player counts to collect (default: 3,4,5)",
    )
    parser.add_argument(
        "--games-per-count",
        type=int,
        default=1000,
        help="Completed games to collect for each requested player count",
    )
    parser.add_argument(
        "--games-per-shard",
        type=int,
        default=50,
        help="Completed games per output .npz shard",
    )
    parser.add_argument("--device", help="Eval device for model loading, e.g. cuda:0")
    parser.add_argument("--seed", type=int, help="Master RNG seed override")
    parser.add_argument("--num-workers", type=int, help="Self-play worker count")
    parser.add_argument("--num-eval-servers", type=int, help="Eval server count")
    parser.add_argument(
        "--eval-devices",
        type=_parse_str_csv,
        help="Comma-separated eval-server devices, one per server",
    )
    parser.add_argument("--search-batch-size", type=int)
    parser.add_argument("--num-simulations", type=int)
    parser.add_argument("--c-puct", type=float)
    parser.add_argument("--dirichlet-epsilon", type=float)
    parser.add_argument(
        "--eval-min-batch-size",
        type=int,
        help="Minimum pending states before eval-server launch",
    )
    parser.add_argument("--eval-min-batch-timeout-ms", type=float)
    parser.add_argument(
        "--eval-batch-shape-mode",
        choices=["dynamic", "bucketed"],
    )
    parser.add_argument("--eval-max-batch-size", type=int)
    parser.add_argument("--eval-dtype", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile in eval servers",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use uncompressed npz shards for faster writes/loads",
    )
    return parser


def _resolve_checkpoint(path_arg: str, checkpoint_dir: str) -> Path:
    if path_arg == "latest":
        latest = find_latest_checkpoint(Path(checkpoint_dir))
        if latest is None:
            raise FileNotFoundError(
                f"no checkpoint_epoch_*.pt found in {checkpoint_dir!r}"
            )
        return latest
    return Path(path_arg)


def _collection_config(
    base: TrainingConfig,
    player_counts: list[int],
) -> TrainingConfig:
    config = copy.deepcopy(base)
    max_capacity = int(base.effective_max_players)
    min_requested = min(player_counts)
    max_requested = max(player_counts)
    if max_requested > max_capacity:
        raise ValueError(
            f"checkpoint model capacity is {max_capacity} players, but "
            f"requested {max_requested}p collection"
        )

    if config.model_type == ModelKind.RESNET.value and len(player_counts) != 1:
        raise ValueError("ResNet checkpoints can only collect one player count")

    if config.model_type == ModelKind.RESNET.value:
        config.num_players = player_counts[0]
        config.min_players = 0
        config.max_players = 0
        config.validate()
        return config

    # Preserve transformer model/storage capacity so checkpoint weights load.
    if min_requested == max_capacity and len(player_counts) == 1:
        config.num_players = max_capacity
        config.min_players = 0
        config.max_players = 0
    else:
        config.num_players = 0
        config.min_players = min_requested
        config.max_players = max_capacity
    config.validate()
    return config


def _apply_runtime_overrides(config: TrainingConfig, args: argparse.Namespace) -> None:
    for attr in (
        "num_workers",
        "num_eval_servers",
        "eval_devices",
        "search_batch_size",
        "dirichlet_epsilon",
        "eval_min_batch_size",
        "eval_min_batch_timeout_ms",
        "eval_batch_shape_mode",
        "eval_max_batch_size",
        "eval_dtype",
        "seed",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(config, attr, value)

    if args.num_simulations is not None:
        config.num_simulations = args.num_simulations
        config.mcts_sims_start = None
        config.mcts_sims_end = None
        config.mcts_ramp_start_epoch = None
        config.mcts_ramp_end_epoch = None

    config.games_per_epoch = args.games_per_count * len(args.player_counts)
    config.validate()


def _epoch_config_for_collection(
    config: TrainingConfig,
    args: argparse.Namespace,
) -> EpochConfig:
    num_simulations = (
        int(args.num_simulations)
        if args.num_simulations is not None
        else int(config.max_simulations)
    )
    c_puct = float(args.c_puct) if args.c_puct is not None else config.c_puct_final
    return EpochConfig(
        c_puct=c_puct,
        value_blend_alpha=1.0,
        num_simulations=num_simulations,
    )


def _eval_devices_for(config: TrainingConfig, device: torch.device) -> list[torch.device]:
    if config.num_workers <= 0:
        return []
    if config.eval_devices:
        return [torch.device(value) for value in config.eval_devices]
    return [device for _ in range(config.num_eval_servers)]


def _strategy_data_worker(
    task_queue: Any,
    result_queue: Any,
    config: TrainingConfig,
    shared_bufs: SharedEvalBuffers,
    worker_idx: int,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)

    evaluator = RemoteEvaluator(
        config.effective_max_players,
        shared_bufs,
        worker_idx,
        terminal_rank_weight=config.terminal_blend,
    )
    total_size = get_layout(config.effective_max_players).total_size
    state_pool = StatePool(2 * (config.max_simulations + 1), total_size)

    try:
        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                break
            game_id, game_seed, rng_seed, num_players, epoch_config = task
            rng = np.random.default_rng(int(rng_seed))
            record = play_game(
                evaluator,
                config,
                int(game_seed),
                rng,
                state_pool=state_pool,
                epoch_config=epoch_config,
                num_players=int(num_players),
                collect_strategy_trace=True,
                game_id=int(game_id),
                rng_seed=int(rng_seed),
            )
            result_queue.put(record)
    except (KeyboardInterrupt, EOFError, BrokenPipeError, OSError):
        pass


def _event_with_game_id(records: list[GameRecord], attr: str, width: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for record in records:
        trace = record.strategy_trace
        if trace is None:
            raise ValueError("strategy shard received a record without trace data")
        rows = getattr(trace, attr)
        if rows.shape[0] == 0:
            continue
        game_col = np.full((rows.shape[0], 1), record.game_id, dtype=np.int32)
        chunks.append(np.concatenate([game_col, rows.astype(np.int32)], axis=1))
    if not chunks:
        return np.empty((0, width + 1), dtype=np.int32)
    return np.concatenate(chunks, axis=0)


class _StrategyShardWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        games_per_shard: int,
        compress: bool,
    ) -> None:
        self.output_dir = output_dir
        self.games_per_shard = int(games_per_shard)
        self.compress = bool(compress)
        self._pending: dict[int, list[GameRecord]] = {}
        self._shard_indices: dict[int, int] = {}
        self.files: list[str] = []

    def add(self, record: GameRecord) -> None:
        records = self._pending.setdefault(record.num_players, [])
        records.append(record)
        if len(records) >= self.games_per_shard:
            self.flush_count(record.num_players)

    def flush_all(self) -> None:
        for num_players in list(self._pending):
            self.flush_count(num_players)

    def flush_count(self, num_players: int) -> None:
        records = self._pending.get(num_players, [])
        if not records:
            return
        shard_idx = self._shard_indices.get(num_players, 0)
        path = self.output_dir / f"strategy_{num_players}p_shard_{shard_idx:05d}.npz"
        payload = self._payload(records)
        save_fn = np.savez_compressed if self.compress else np.savez
        save_fn(path, **payload)
        self.files.append(path.name)
        total_moves = int(payload["states"].shape[0])
        print(
            f"  wrote {path.name}: {len(records)} games, {total_moves} moves"
        )
        self._pending[num_players] = []
        self._shard_indices[num_players] = shard_idx + 1

    def _payload(self, records: list[GameRecord]) -> dict[str, np.ndarray]:
        traces = []
        for record in records:
            if record.strategy_trace is None:
                raise ValueError("strategy shard received a record without trace data")
            traces.append(record.strategy_trace)

        game_lengths = np.asarray([r.num_examples for r in records], dtype=np.int32)
        starts = np.zeros(len(records), dtype=np.int64)
        if len(records) > 1:
            starts[1:] = np.cumsum(game_lengths[:-1], dtype=np.int64)

        def concat_record(name: str) -> np.ndarray:
            return np.concatenate([getattr(r, name) for r in records], axis=0)

        def concat_trace(name: str) -> np.ndarray:
            return np.concatenate([getattr(t, name) for t in traces], axis=0)

        game_ids = np.concatenate(
            [
                np.full(r.num_examples, r.game_id, dtype=np.int32)
                for r in records
            ]
        )
        move_numbers = np.concatenate(
            [
                np.arange(r.num_examples, dtype=np.int32)
                for r in records
            ]
        )

        return {
            "states": concat_record("states"),
            "final_states": np.stack([r.final_state for r in records]).astype(np.int16),
            "phase_ids": concat_record("phase_ids"),
            "legal_masks": concat_record("legal_masks"),
            "nn_policy_pct": concat_trace("nn_policy_pct"),
            "nn_values": concat_trace("nn_values"),
            "mcts_policy_pct": (concat_record("policy_targets") * 100.0).astype(np.float32),
            "mcts_visit_counts": concat_trace("mcts_visit_counts"),
            "a0gb_values": concat_trace("a0gb_values"),
            "mcts_root_values": concat_trace("mcts_root_values"),
            "game_ids": game_ids,
            "move_numbers": move_numbers,
            "game_start_offsets": starts,
            "game_num_examples": game_lengths,
            "game_ids_per_game": np.asarray([r.game_id for r in records], dtype=np.int32),
            "game_seeds": np.asarray([r.game_seed for r in records], dtype=np.int64),
            "rng_seeds": np.asarray([r.rng_seed for r in records], dtype=np.int64),
            "game_durations_sec": np.asarray(
                [r.duration_secs for r in records], dtype=np.float32,
            ),
            "final_net_worths": np.stack(
                [np.asarray(r.net_worths, dtype=np.int16) for r in records]
            ),
            "final_shares_per_player": np.stack(
                [np.asarray(r.shares_per_player, dtype=np.int16) for r in records]
            ),
            "final_companies_per_player": np.stack(
                [np.asarray(r.companies_per_player, dtype=np.int16) for r in records]
            ),
            "selected_action_ids": concat_trace("selected_action_ids"),
            "selected_unified_slots": concat_trace("selected_unified_slots"),
            "action_types": concat_trace("action_types"),
            "action_corps": concat_trace("action_corps"),
            "action_companies": concat_trace("action_companies"),
            "action_amounts": concat_trace("action_amounts"),
            "engine_phase_ids": concat_trace("engine_phase_ids"),
            "active_players": concat_trace("active_players"),
            "active_corps": concat_trace("active_corps"),
            "active_companies": concat_trace("active_companies"),
            "turn_numbers": concat_trace("turn_numbers"),
            "coo_levels": concat_trace("coo_levels"),
            "cards_remaining": concat_trace("cards_remaining"),
            "auction_prices": concat_trace("auction_prices"),
            "auction_high_bidders": concat_trace("auction_high_bidders"),
            "auction_starters": concat_trace("auction_starters"),
            "acq_offer_prices": concat_trace("acq_offer_prices"),
            "acq_offer_corps": concat_trace("acq_offer_corps"),
            "target_temperatures": concat_trace("target_temperatures"),
            "sample_temperatures": concat_trace("sample_temperatures"),
            "greedy_leaf_depths": concat_trace("greedy_leaf_depths"),
            "root_visit_counts": concat_trace("root_visit_counts"),
            "player_cash": concat_trace("player_cash"),
            "player_net_worth": concat_trace("player_net_worth"),
            "player_liquidity": concat_trace("player_liquidity"),
            "player_income": concat_trace("player_income"),
            "player_shares": concat_trace("player_shares"),
            "corp_active": concat_trace("corp_active"),
            "corp_prices": concat_trace("corp_prices"),
            "corp_cash": concat_trace("corp_cash"),
            "corp_income": concat_trace("corp_income"),
            "corp_presidents": concat_trace("corp_presidents"),
            "corp_issued_shares": concat_trace("corp_issued_shares"),
            "corp_bank_shares": concat_trace("corp_bank_shares"),
            "corp_unissued_shares": concat_trace("corp_unissued_shares"),
            "corp_receivership": concat_trace("corp_receivership"),
            "company_locations": concat_trace("company_locations"),
            "company_owners": concat_trace("company_owners"),
            "company_adjusted_income": concat_trace("company_adjusted_income"),
            "auction_events": _event_with_game_id(records, "auction_events", 9),
            "ipo_events": _event_with_game_id(records, "ipo_events", 11),
            "acquisition_events": _event_with_game_id(records, "acquisition_events", 12),
            "share_trade_events": _event_with_game_id(records, "share_trade_events", 12),
            "dividend_events": _event_with_game_id(records, "dividend_events", 9),
            "issue_events": _event_with_game_id(records, "issue_events", 12),
            "close_events": _event_with_game_id(records, "close_events", 8),
        }


def _metadata(
    *,
    checkpoint_path: Path,
    checkpoint_epoch: int,
    config: TrainingConfig,
    epoch_config: EpochConfig,
    player_counts: list[int],
    games_per_count: int,
    games_per_shard: int,
    compress: bool,
) -> dict[str, Any]:
    layout = get_layout(config.effective_max_players)
    layout_payload = (
        layout._asdict() if hasattr(layout, "_asdict") else dict(layout)
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "player_counts": player_counts,
        "games_per_count": games_per_count,
        "games_per_shard": games_per_shard,
        "compressed_npz": compress,
        "training_config": json.loads(config.to_json()),
        "collection_epoch_config": {
            "c_puct": epoch_config.c_puct,
            "value_blend_alpha": epoch_config.value_blend_alpha,
            "num_simulations": epoch_config.num_simulations,
        },
        "state_layout": layout_payload,
        "unified_logit_dim": int(UNIFIED_LOGIT_DIM),
        "phase_action_sizes": list(PHASE_ACTION_SIZES),
        "action_lut": build_action_lut().numpy().tolist(),
        "company_names": list(COMPANY_NAMES),
        "corp_names": list(CORP_NAMES),
        "all_par_prices": [int(v) for v in ALL_PAR_PRICES],
        "company_static": {
            "face_value": [COMPANIES[i].get_face_value() for i in range(len(COMPANIES))],
            "low_price": [COMPANIES[i].get_low_price() for i in range(len(COMPANIES))],
            "high_price": [COMPANIES[i].get_high_price() for i in range(len(COMPANIES))],
            "stars": [COMPANIES[i].get_stars() for i in range(len(COMPANIES))],
            "base_income": [COMPANIES[i].get_base_income() for i in range(len(COMPANIES))],
        },
        "corp_static": {
            "total_shares": [CORPS[i].get_total_shares() for i in range(len(CORPS))],
        },
        "event_columns": {
            "auction_events": [
                "game_id", "move_number", "turn_number", "company_id",
                "winner_player", "price", "starter_player", "high_bidder_before",
                "final_action_type", "final_action_id",
            ],
            "ipo_events": [
                "game_id", "move_number", "turn_number", "player_id",
                "corp_id", "company_id", "par_index", "par_price",
                "float_shares", "player_payment", "corp_cash", "issued_shares",
            ],
            "acquisition_events": [
                "game_id", "move_number", "turn_number", "player_id",
                "buyer_corp", "company_id", "price", "source_location",
                "source_owner", "post_location", "post_owner", "action_type",
                "action_id",
            ],
            "share_trade_events": [
                "game_id", "move_number", "turn_number", "player_id",
                "corp_id", "action_type", "shares_before", "shares_after",
                "cash_before", "cash_after", "price_before", "price_after",
                "action_id",
            ],
            "dividend_events": [
                "game_id", "move_number", "turn_number", "player_id",
                "corp_id", "amount", "cash_before", "cash_after",
                "price_before", "price_after",
            ],
            "issue_events": [
                "game_id", "move_number", "turn_number", "player_id",
                "corp_id", "bank_shares_before", "bank_shares_after",
                "issued_before", "issued_after", "cash_before", "cash_after",
                "price_before", "price_after",
            ],
            "close_events": [
                "game_id", "move_number", "turn_number", "player_id",
                "company_id", "source_location", "source_owner",
                "action_type", "action_id",
            ],
        },
        "event_notes": {
            "acquisition_events": (
                "Priced acquisition transactions only; phase-cleanup transfers "
                "from acquisition piles into owned corp locations are visible "
                "in company_locations/company_owners snapshots but omitted here."
            ),
        },
    }


def _write_metadata(path: Path, data: dict[str, Any], files: list[str] | None = None) -> None:
    payload = dict(data)
    if files is not None:
        payload["files"] = sorted(files)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _start_eval_stack(
    *,
    model: torch.nn.Module,
    config: TrainingConfig,
    model_input_spec: Any,
    device: torch.device,
    no_compile: bool,
) -> tuple[Any, Any, list[Any], list[EvaluationServer]]:
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    task_queue.cancel_join_thread()
    result_queue = ctx.Queue()
    result_queue.cancel_join_thread()

    shared_bufs = SharedEvalBuffers(
        num_workers=config.num_workers,
        batch_size=config.search_batch_size,
        num_players=config.effective_max_players,
        input_spec=model_input_spec,
    )
    n_servers = config.num_eval_servers
    workers_per_server = config.num_workers // n_servers
    remainder = config.num_workers % n_servers
    partitions: list[tuple[int, int]] = []
    offset = 0
    for i in range(n_servers):
        width = workers_per_server + (1 if i < remainder else 0)
        partitions.append((offset, offset + width))
        offset += width
    shared_bufs.init_bitmap(partitions, ctx)

    from train.gpu import detect_gpu

    gpu = detect_gpu(device.type)
    compile_kwargs = gpu.get_compile_kwargs(
        for_training=False,
        eval_batch_shape_mode=config.eval_batch_shape_mode,
    )
    eval_devices = _eval_devices_for(config, device)
    eval_servers: list[EvaluationServer] = []
    for i, (ws, we) in enumerate(partitions):
        server = EvaluationServer(
            model,
            eval_devices[i],
            shared_bufs,
            server_id=i,
            worker_start=ws,
            worker_end=we,
            mp_context=ctx,
            no_compile=no_compile,
            compile_kwargs=compile_kwargs,
            gpu_vendor=gpu.vendor,
            min_batch_size=config.eval_min_batch_size,
            min_batch_timeout_ms=config.eval_min_batch_timeout_ms,
            batch_shape_mode=config.eval_batch_shape_mode,
            max_batch_size=config.eval_max_batch_size,
            eval_dtype=config.eval_dtype,
        )
        server.start()
        eval_servers.append(server)

    print(f"Waiting for {n_servers} eval server{'s' if n_servers != 1 else ''}...")
    for server in eval_servers:
        if not server.wait_ready(timeout=2400.0):
            raise RuntimeError("eval server did not become ready")
    print("  eval servers ready")

    workers: list[Any] = []
    for worker_idx in range(config.num_workers):
        process = ctx.Process(
            target=_strategy_data_worker,
            args=(task_queue, result_queue, config, shared_bufs, worker_idx),
            daemon=True,
        )
        process.start()
        workers.append(process)
    return task_queue, result_queue, workers, eval_servers


def main() -> None:
    args = _build_parser().parse_args()
    if args.games_per_count < 1:
        raise ValueError("--games-per-count must be >= 1")
    if args.games_per_shard < 1:
        raise ValueError("--games-per-shard must be >= 1")

    checkpoint_path = _resolve_checkpoint(args.checkpoint, args.checkpoint_dir)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    from train.gpu import detect_gpu

    gpu = detect_gpu(device.type)
    if gpu.vendor != "cpu":
        info = gpu.apply_optimizations()
        print(
            f"GPU optimizations ({gpu.vendor}): "
            + ", ".join(f"{k}={v}" for k, v in info.items())
        )

    cp = load_checkpoint(checkpoint_path, device)
    base_config = TrainingConfig.from_json(cp["config_json"])  # type: ignore[arg-type]
    config = _collection_config(base_config, args.player_counts)
    _apply_runtime_overrides(config, args)
    epoch_config = _epoch_config_for_collection(config, args)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    master_rng = np.random.default_rng(config.seed)

    model = create_model(config).to(device)
    model.load_state_dict(cp["model_state_dict"])  # type: ignore[arg-type]
    model.eval()
    model_input_spec = get_model_input_spec(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    metadata = _metadata(
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=int(cp.get("epoch", -1)),
        config=config,
        epoch_config=epoch_config,
        player_counts=args.player_counts,
        games_per_count=args.games_per_count,
        games_per_shard=args.games_per_shard,
        compress=not args.no_compress,
    )
    _write_metadata(metadata_path, metadata)

    writer = _StrategyShardWriter(
        output_dir,
        games_per_shard=args.games_per_shard,
        compress=not args.no_compress,
    )
    counts_remaining = {
        count: args.games_per_count for count in args.player_counts
    }
    total_games = args.games_per_count * len(args.player_counts)
    games_done = 0
    started = time.perf_counter()

    task_queue: Any = None
    result_queue: Any = None
    workers: list[Any] = []
    eval_servers: list[EvaluationServer] = []

    try:
        if config.num_workers > 0:
            task_queue, result_queue, workers, eval_servers = _start_eval_stack(
                model=model,
                config=config,
                model_input_spec=model_input_spec,
                device=device,
                no_compile=args.no_compile,
            )
            game_id = 0
            for player_count in args.player_counts:
                for _ in range(args.games_per_count):
                    game_seed = int(master_rng.integers(0, 2**31))
                    rng_seed = int(master_rng.integers(0, 2**63))
                    task_queue.put((
                        game_id,
                        game_seed,
                        rng_seed,
                        player_count,
                        epoch_config,
                    ))
                    game_id += 1

            while games_done < total_games:
                try:
                    record = result_queue.get(timeout=1200.0)
                except queue.Empty:
                    alive = sum(1 for worker in workers if worker.is_alive())
                    raise RuntimeError(
                        f"timed out waiting for strategy game "
                        f"({alive}/{len(workers)} workers alive)"
                    )
                writer.add(record)
                counts_remaining[record.num_players] -= 1
                games_done += 1
                elapsed = time.perf_counter() - started
                rate = games_done / elapsed if elapsed > 0 else 0.0
                remaining = ", ".join(
                    f"{count}p={left}"
                    for count, left in sorted(counts_remaining.items())
                )
                print(
                    f"[{games_done}/{total_games}] "
                    f"{record.num_players}p game {record.game_id} "
                    f"moves={record.num_examples} rate={rate:.2f}/s "
                    f"remaining: {remaining}"
                )
        else:
            evaluator = NNEvaluator(
                model,
                device,
                num_players=config.effective_max_players,
                terminal_rank_weight=config.terminal_blend,
                eval_dtype=config.eval_dtype,
                input_spec=model_input_spec,
            )
            state_pool = StatePool(
                2 * (config.max_simulations + 1),
                get_layout(config.effective_max_players).total_size,
            )
            game_id = 0
            for player_count in args.player_counts:
                for _ in range(args.games_per_count):
                    game_seed = int(master_rng.integers(0, 2**31))
                    rng_seed = int(master_rng.integers(0, 2**63))
                    record = play_game(
                        evaluator,
                        config,
                        game_seed,
                        np.random.default_rng(rng_seed),
                        state_pool=state_pool,
                        epoch_config=epoch_config,
                        num_players=player_count,
                        collect_strategy_trace=True,
                        game_id=game_id,
                        rng_seed=rng_seed,
                    )
                    writer.add(record)
                    counts_remaining[player_count] -= 1
                    games_done += 1
                    game_id += 1
                    print(
                        f"[{games_done}/{total_games}] "
                        f"{player_count}p game {record.game_id} "
                        f"moves={record.num_examples}"
                    )
        writer.flush_all()
        _write_metadata(metadata_path, metadata, writer.files)
        elapsed = time.perf_counter() - started
        print(
            f"\nCollected {games_done} games in {elapsed:.1f}s. "
            f"Output: {output_dir}"
        )
    finally:
        if task_queue is not None:
            for _ in workers:
                task_queue.put(None)
        for server in eval_servers:
            server.stop()
        for worker in workers:
            worker.join(timeout=3.0)
            if worker.is_alive():
                worker.terminate()


if __name__ == "__main__":
    main()
