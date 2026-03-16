"""Main training orchestration loop."""

from __future__ import annotations

import argparse
import queue
import time
from collections import defaultdict
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.multiprocessing as mp

from nn.model_3p import RSSAlphaZeroNet, RSSModelConfig
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
    parser.add_argument("--games-per-epoch", type=int)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--num-simulations", type=int)
    parser.add_argument("--search-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--checkpoint-dir", type=str)
    parser.add_argument("--tensorboard-dir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temp-initial", type=float)
    parser.add_argument("--temp-anneal-start", type=int)
    parser.add_argument("--temp-anneal-end", type=int)
    parser.add_argument("--temp-final", type=float)
    parser.add_argument("--c-puct-initial", type=float)
    parser.add_argument("--c-puct-final", type=float)
    parser.add_argument("--c-puct-anneal-epochs", type=int)
    parser.add_argument("--value-blend-start-epoch", type=int)
    parser.add_argument("--value-blend-end-epoch", type=int)
    parser.add_argument("--reuse-subtree-after-epoch", type=int)
    parser.add_argument("--dirichlet-alpha", type=float)
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
    return parser


_CLI_FIELDS = (
    "games_per_epoch", "num_epochs", "num_simulations", "search_batch_size",
    "num_workers", "checkpoint_dir", "tensorboard_dir", "seed",
    "temp_initial", "temp_anneal_start", "temp_anneal_end", "temp_final",
    "c_puct_initial", "c_puct_final", "c_puct_anneal_epochs",
    "value_blend_start_epoch", "value_blend_end_epoch",
    "reuse_subtree_after_epoch",
    "dirichlet_alpha", "dirichlet_dynamic", "dirichlet_alpha_numerator",
)


def _build_profile_scalars(
    game_profiles: list[GameProfileData],
    server_stats: EvalServerStats | None,
    sp_duration: float,
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
            s.idle_secs / sp_duration * 100 if sp_duration > 0 else 0
        )

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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --- Device (resolved early, needed for checkpoint loading) ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Resume: restore checkpointed config, apply operational overrides only
        config = TrainingConfig.from_json(cp["config_json"])  # type: ignore[arg-type]
        _apply_overrides(config, args, log_changes=True)
        config.validate()
        if args.config:
            print("  Warning: --config ignored on resume (using checkpointed config)")
    elif args.config:
        config = TrainingConfig.from_json(Path(args.config).read_text())
        _apply_overrides(config, args)
        config.validate()
    else:
        config = TrainingConfig()
        _apply_overrides(config, args)
        config.validate()

    # --- Profile flag (operational, not in config JSON) ---
    if args.profile:
        config.profile = True

    # --- RNG ---
    master_rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # --- Model ---
    model_config = RSSModelConfig(
        input_dim=config.visible_size,
        action_dim=config.action_dim,
        value_dim=config.num_players,
    )
    model = RSSAlphaZeroNet(model_config).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    # --- Components ---
    trainer = Trainer(model, config, device)
    buffer = ReplayBuffer(
        config.buffer_capacity,
        config.visible_size,
        config.action_dim,
        config.num_players,
    )
    logger = TrainingLogger(config.tensorboard_dir)

    # --- Resume: restore model and trainer state ---
    start_epoch = 0
    if cp is not None:
        model.load_state_dict(cp["model_state_dict"])  # type: ignore[arg-type]
        trainer.load_state_dict(cp["trainer_state"])  # type: ignore[arg-type]
        start_epoch = cp["epoch"] + 1  # type: ignore[operator]
        print(
            f"Resumed from epoch {cp['epoch']}, "
            f"step {trainer.global_step}"
        )

    # --- torch.compile ---
    if not args.no_compile and device.type == "cuda":
        print("Compiling model with torch.compile...")
        # torch.compile returns OptimizedModule (a Module subclass at runtime),
        # but Pyright sees the return type as FunctionType | Module. Cast to keep
        # downstream type-safety for .eval(), .train(), save_checkpoint(), etc.
        model = cast(RSSAlphaZeroNet, torch.compile(model))
        # Warmup pass to trigger compilation before self-play starts
        with torch.no_grad():
            dummy = torch.randn(1, config.visible_size, device=device)
            model(dummy)
        print("  Model compiled.")

    # --- Log startup ---
    logger.log_training_start(config, device=str(device))
    print(f"Model parameters: {param_count:,}")
    print()

    # --- Multi-process self-play setup ---
    workers: list[Any] = []  # mp.Process (SpawnProcess when using spawn context)
    eval_server: EvaluationServer | None = None
    server_conns: list[Connection] = []
    task_queue: Any = None  # mp.Queue
    result_queue: Any = None  # mp.Queue

    # Single-process fallback
    evaluator: NNEvaluator | None = None
    state_pool: StatePool | None = None

    if config.num_workers > 0:
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()

        worker_conns: list[Connection] = []
        for _ in range(config.num_workers):
            s_conn, w_conn = ctx.Pipe()
            server_conns.append(s_conn)
            worker_conns.append(w_conn)

        shared_bufs = SharedEvalBuffers(
            num_workers=config.num_workers,
            batch_size=config.search_batch_size,
            visible_size=config.visible_size,
            action_dim=config.action_dim,
            num_players=config.num_players,
        )

        eval_server = EvaluationServer(
            model, device, server_conns, shared_bufs, profile=config.profile,
        )
        eval_server.start()

        for i in range(config.num_workers):
            p = ctx.Process(
                target=self_play_worker,
                args=(
                    worker_conns[i], task_queue, result_queue, config,
                    shared_bufs, i,
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)

        # Close our copies of worker-side pipe ends
        for conn in worker_conns:
            conn.close()

        print(f"Started {config.num_workers} self-play workers")
    else:
        evaluator = NNEvaluator(model, device, num_players=config.num_players)
        from core.state import get_layout

        total_state_size = get_layout(config.num_players).total_size
        state_pool = StatePool(config.num_simulations + 1, total_state_size)

    # --- Training loop ---
    avg_losses: dict[str, float] = {}

    try:
        for epoch in range(start_epoch, config.num_epochs):
            epoch_start = time.perf_counter()
            epoch_num = epoch + 1  # 1-indexed for display

            # --- Phase 1: Self-play ---
            model.eval()
            total_examples = 0
            total_moves = 0
            total_duration = 0.0
            total_entropy = 0.0
            total_top1 = 0.0
            num_players = config.num_players
            rank_totals = [0.0] * num_players
            rank_mins = [float("inf")] * num_players
            rank_maxs = [float("-inf")] * num_players
            game_profiles: list[GameProfileData] = []

            def _collect_record(record: object, game_idx: int) -> None:
                nonlocal total_examples, total_moves, total_duration
                nonlocal total_entropy, total_top1
                buffer.add_examples(record.examples)  # type: ignore[union-attr]
                total_examples += len(record.examples)  # type: ignore[union-attr]
                total_moves += record.total_moves  # type: ignore[union-attr]
                total_duration += record.duration_secs  # type: ignore[union-attr]
                total_entropy += record.policy_entropy_mean  # type: ignore[union-attr]
                total_top1 += record.top1_visit_fraction  # type: ignore[union-attr]
                for rank, nw in enumerate(sorted(record.net_worths, reverse=True)):  # type: ignore[union-attr]
                    rank_totals[rank] += nw
                    if nw < rank_mins[rank]:
                        rank_mins[rank] = nw
                    if nw > rank_maxs[rank]:
                        rank_maxs[rank] = nw
                if record.profile is not None:  # type: ignore[union-attr]
                    game_profiles.append(record.profile)  # type: ignore[union-attr]
                n = game_idx + 1
                logger.update_self_play(
                    games_done=n,
                    total_examples=total_examples,
                    avg_moves=total_examples / n,
                    rank_net_worths=[t / n for t in rank_totals],
                    rank_mins=list(rank_mins),
                    rank_maxs=list(rank_maxs),
                    policy_entropy=total_entropy / n,
                    top1_visit_frac=total_top1 / n,
                )

            # Reset eval server profile stats for this epoch
            if config.profile and eval_server is not None:
                eval_server.reset_profile_stats()

            # Compute per-epoch annealing parameters
            epoch_cfg = config.compute_epoch_config(epoch)

            logger.begin_self_play(epoch_num, config.num_epochs, config.games_per_epoch)

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
                        record = result_queue.get(timeout=300.0)
                    except queue.Empty:
                        alive = sum(1 for w in workers if w.is_alive())
                        raise RuntimeError(
                            f"Timed out waiting for game result "
                            f"({alive}/{config.num_workers} workers alive)"
                        )
                    _collect_record(record, game_idx)
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

            logger.end_self_play()

            # Self-play Tensorboard metrics
            n_games = config.games_per_epoch
            avg_game_moves = total_moves / n_games
            avg_game_dur = total_duration / n_games
            rank_avgs = [t / n_games for t in rank_totals]

            rank_labels = ["1st", "2nd", "3rd", "4th", "5th", "6th"][:num_players]
            net_worth_scalars = {}
            for k, avg, mn, mx in zip(
                rank_labels, rank_avgs, rank_mins, rank_maxs
            ):
                net_worth_scalars[f"self_play/net_worth_{k}"] = avg
                net_worth_scalars[f"self_play/net_worth_{k}_min"] = mn
                net_worth_scalars[f"self_play/net_worth_{k}_max"] = mx

            avg_entropy = total_entropy / n_games
            avg_top1 = total_top1 / n_games

            logger.log_scalars(
                epoch_num,
                {
                    "self_play/game_length_mean": avg_game_moves,
                    "self_play/duration_mean": avg_game_dur,
                    "self_play/examples_per_game": total_examples / n_games,
                    "self_play/total_examples": float(total_examples),
                    "self_play/policy_entropy_mean": avg_entropy,
                    "self_play/top1_visit_fraction": avg_top1,
                    "buffer/size": float(len(buffer)),
                    "buffer/utilization": len(buffer) / config.buffer_capacity,
                    "schedule/c_puct": epoch_cfg.c_puct,
                    "schedule/value_blend_alpha": epoch_cfg.value_blend_alpha,
                    "schedule/subtree_reuse": float(epoch_cfg.enable_subtree_reuse),
                    **net_worth_scalars,
                },
            )

            # --- Profile summary ---
            if config.profile and game_profiles:
                sp_duration = time.perf_counter() - epoch_start
                server_stats = (
                    eval_server.get_profile_stats()
                    if eval_server is not None else None
                )
                print(format_epoch_profile(game_profiles, server_stats, sp_duration))

                # Tensorboard: profile scalars
                profile_scalars = _build_profile_scalars(
                    game_profiles, server_stats, sp_duration,
                )
                logger.log_scalars(epoch_num, profile_scalars)

            # --- Phase 2: Training ---
            if len(buffer) >= config.min_buffer_size:
                model.train()
                epoch_losses: dict[str, list[float]] = defaultdict(list)

                logger.begin_training(
                    epoch_num, config.num_epochs, config.training_steps_per_epoch
                )

                for step in range(config.training_steps_per_epoch):
                    batch = buffer.sample(config.batch_size, master_rng)
                    losses = trainer.train_step(batch)
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
                                "lr": trainer.lr,
                            },
                        )

                logger.end_training()

                avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}

                # Epoch-level Tensorboard
                logger.log_scalars(
                    epoch_num,
                    {
                        "epoch/total_loss_avg": avg_losses["total_loss"],
                        "epoch/policy_loss_avg": avg_losses["policy_loss"],
                        "epoch/value_loss_avg": avg_losses["value_loss"],
                    },
                )
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
                )
                cleanup_checkpoints(Path(config.checkpoint_dir), config.keep_last_n)
                checkpoint_path = str(cp_path)

            # --- Phase 4: Epoch summary ---
            epoch_duration = time.perf_counter() - epoch_start
            logger.log_scalars(epoch_num, {"epoch/duration_secs": epoch_duration})
            logger.log_epoch_summary(
                epoch=epoch_num,
                num_epochs=config.num_epochs,
                self_play_stats={
                    "games": float(n_games),
                    "examples": float(total_examples),
                    "avg_moves": avg_game_moves,
                    "avg_duration": avg_game_dur,
                    "rank_net_worths": rank_avgs,
                    "rank_net_worths_min": rank_mins,
                    "rank_net_worths_max": rank_maxs,
                    "policy_entropy": avg_entropy,
                    "top1_visit_frac": avg_top1,
                },
                train_stats={
                    "steps": float(config.training_steps_per_epoch) if avg_losses else 0.0,
                    "lr": trainer.lr,
                    **avg_losses,
                },
                buffer_size=len(buffer),
                buffer_capacity=config.buffer_capacity,
                epoch_duration=epoch_duration,
                checkpoint_path=checkpoint_path,
            )

        # --- Final checkpoint ---
        final_cp = Path(config.checkpoint_dir) / f"checkpoint_epoch_{config.num_epochs:04d}.pt"
        if not final_cp.exists():
            save_checkpoint(
                final_cp,
                config.num_epochs - 1,
                model,
                trainer.state_dict(),
                config,
                avg_losses,
                {"size": len(buffer), "capacity": config.buffer_capacity},
            )

        print("\nTraining complete.")

    except KeyboardInterrupt:
        print("\nInterrupted — shutting down...")

    finally:
        # Clean shutdown: stop eval server, close connections, terminate workers
        if eval_server is not None:
            eval_server.stop()
        for conn in server_conns:
            try:
                conn.close()
            except OSError:
                pass
        for w in workers:
            w.join(timeout=3.0)
            if w.is_alive():
                w.terminate()
        logger.close()
