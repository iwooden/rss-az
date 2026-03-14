"""Main training orchestration loop."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import time
from collections import defaultdict
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nn.model_3p import RSSAlphaZeroNet, RSSModelConfig
from train.checkpoint import (
    cleanup_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from train.config import TrainingConfig
from train.eval_server import EvaluationServer
from train.logging import TrainingLogger
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
    return parser


def _apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> None:
    """Apply CLI overrides to config in-place."""
    if args.games_per_epoch is not None:
        config.games_per_epoch = args.games_per_epoch
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.num_simulations is not None:
        config.num_simulations = args.num_simulations
    if args.search_batch_size is not None:
        config.search_batch_size = args.search_batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.tensorboard_dir is not None:
        config.tensorboard_dir = args.tensorboard_dir
    if args.seed is not None:
        config.seed = args.seed


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --- Config ---
    if args.config:
        config = TrainingConfig.from_json(Path(args.config).read_text())
    else:
        config = TrainingConfig()
    _apply_overrides(config, args)

    # --- RNG ---
    master_rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # --- Resume ---
    start_epoch = 0
    if args.resume:
        if args.resume == "latest":
            cp_path = find_latest_checkpoint(Path(config.checkpoint_dir))
            if cp_path is None:
                print("No checkpoint found, starting from scratch.")
            else:
                args.resume = str(cp_path)
        if args.resume != "latest":
            cp = load_checkpoint(Path(args.resume), device)
            model.load_state_dict(cp["model_state_dict"])  # type: ignore[arg-type]
            trainer.load_state_dict(cp["trainer_state"])  # type: ignore[arg-type]
            start_epoch = cp["epoch"] + 1  # type: ignore[operator]
            print(
                f"Resumed from epoch {cp['epoch']}, "
                f"step {trainer.global_step}"
            )

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

        eval_server = EvaluationServer(model, device, server_conns)
        eval_server.start()

        for i in range(config.num_workers):
            p = ctx.Process(
                target=self_play_worker,
                args=(worker_conns[i], task_queue, result_queue, config),
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
            records = []
            total_examples = 0
            num_players = config.num_players
            rank_totals = [0.0] * num_players

            def _collect_record(record: object, game_idx: int) -> None:
                nonlocal total_examples
                buffer.add_examples(record.examples)  # type: ignore[union-attr]
                records.append(record)
                total_examples += len(record.examples)  # type: ignore[union-attr]
                for rank, nw in enumerate(sorted(record.net_worths, reverse=True)):  # type: ignore[union-attr]
                    rank_totals[rank] += nw
                n = game_idx + 1
                logger.update_self_play(
                    games_done=n,
                    total_examples=total_examples,
                    avg_moves=total_examples / n,
                    rank_net_worths=[t / n for t in rank_totals],
                )

            logger.begin_self_play(epoch_num, config.num_epochs, config.games_per_epoch)

            if config.num_workers > 0:
                assert task_queue is not None and result_queue is not None
                # Feed all game seeds to workers
                for _ in range(config.games_per_epoch):
                    game_seed = int(master_rng.integers(0, 2**31))
                    rng_seed = int(master_rng.integers(0, 2**63))
                    task_queue.put((game_seed, rng_seed))

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
                        state_pool=state_pool,
                    )
                    _collect_record(record, game_idx)

            logger.end_self_play()

            # Self-play Tensorboard metrics
            avg_game_moves = sum(r.total_moves for r in records) / len(records)
            avg_game_dur = sum(r.duration_secs for r in records) / len(records)

            # Net worth by finishing rank (already accumulated in rank_totals)
            n_games = len(records)
            rank_avgs = [t / n_games for t in rank_totals]

            net_worth_scalars = {
                f"self_play/net_worth_{k}": v
                for k, v in zip(
                    ["1st", "2nd", "3rd", "4th", "5th", "6th"][:num_players],
                    rank_avgs,
                )
            }

            logger.log_scalars(
                epoch_num,
                {
                    "self_play/game_length_mean": avg_game_moves,
                    "self_play/duration_mean": avg_game_dur,
                    "self_play/examples_per_game": total_examples / len(records),
                    "self_play/total_examples": float(total_examples),
                    "buffer/size": float(len(buffer)),
                    "buffer/utilization": len(buffer) / config.buffer_capacity,
                    **net_worth_scalars,
                },
            )

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
                    "games": float(len(records)),
                    "examples": float(total_examples),
                    "avg_moves": avg_game_moves,
                    "avg_duration": avg_game_dur,
                    "rank_net_worths": rank_avgs,
                },
                train_stats={
                    "steps": float(config.training_steps_per_epoch),
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
