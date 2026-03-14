"""Tests for the self-play training harness.

Uses tiny configs and small models to keep tests fast (< 30s total).
All file output uses tmp_path to avoid polluting the project directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from nn.model_3p import RSSAlphaZeroNet, RSSModelConfig
from train.checkpoint import (
    cleanup_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from train.config import MCTSConfig, TrainingConfig
from train.logging import TrainingLogger, _format_duration
from train.main import _apply_overrides, _build_parser
from train.replay_buffer import ReplayBuffer, TrainingExample
from train.self_play import play_game
from train.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_example(
    visible_size: int, action_dim: int, num_players: int
) -> TrainingExample:
    """Create a random valid TrainingExample."""
    rng = np.random.default_rng()
    state = rng.standard_normal(visible_size).astype(np.float32)
    legal_mask = np.zeros(action_dim, dtype=np.float32)
    n_legal = max(1, rng.integers(1, min(10, action_dim + 1)))
    legal_indices = rng.choice(action_dim, n_legal, replace=False)
    legal_mask[legal_indices] = 1.0
    policy = np.zeros(action_dim, dtype=np.float32)
    policy[legal_indices] = rng.dirichlet(np.ones(n_legal)).astype(np.float32)
    value = rng.uniform(-1, 1, size=num_players).astype(np.float32)
    return TrainingExample(
        state=state, legal_mask=legal_mask, policy_target=policy, value_target=value
    )


def _make_batch(
    batch_size: int, visible_size: int, action_dim: int, num_players: int
) -> dict[str, torch.Tensor]:
    """Create a synthetic training batch."""
    examples = [_make_example(visible_size, action_dim, num_players) for _ in range(batch_size)]
    return {
        "states": torch.from_numpy(np.stack([e.state for e in examples])),
        "legal_masks": torch.from_numpy(np.stack([e.legal_mask for e in examples])),
        "policy_targets": torch.from_numpy(np.stack([e.policy_target for e in examples])),
        "value_targets": torch.from_numpy(np.stack([e.value_target for e in examples])),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        num_players=3,
        games_per_epoch=1,
        num_simulations=2,
        temp_threshold=5,
        buffer_capacity=1000,
        min_buffer_size=10,
        batch_size=4,
        learning_rate=1e-3,
        training_steps_per_epoch=2,
        warmup_steps=1,
        num_epochs=2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        tensorboard_dir=str(tmp_path / "tb"),
        seed=42,
    )


@pytest.fixture
def small_model() -> RSSAlphaZeroNet:
    cfg = RSSModelConfig(
        input_dim=3023,
        action_dim=246,
        value_dim=3,
        hidden_dim=32,
        num_blocks=1,
        expansion=1,
    )
    return RSSAlphaZeroNet(cfg)


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_computed_properties(self) -> None:
        cfg = TrainingConfig(num_players=3)
        assert cfg.action_dim == 246
        assert cfg.visible_size == 3023

    def test_json_roundtrip(self) -> None:
        cfg = TrainingConfig(learning_rate=0.005, num_epochs=50)
        restored = TrainingConfig.from_json(cfg.to_json())
        assert restored.learning_rate == cfg.learning_rate
        assert restored.num_epochs == cfg.num_epochs
        assert restored.action_dim == cfg.action_dim
        assert restored.visible_size == cfg.visible_size

    def test_validation_num_players(self) -> None:
        with pytest.raises(ValueError, match="num_players"):
            TrainingConfig(num_players=0)

    def test_validation_buffer_capacity(self) -> None:
        with pytest.raises(ValueError, match="buffer_capacity"):
            TrainingConfig(buffer_capacity=100, min_buffer_size=200)

    def test_validation_min_buffer_vs_batch(self) -> None:
        with pytest.raises(ValueError, match="min_buffer_size"):
            TrainingConfig(min_buffer_size=100, batch_size=256)

    def test_to_mcts_config(self) -> None:
        cfg = TrainingConfig(num_simulations=100, c_puct=3.0, num_players=3)
        mcts_cfg = cfg.to_mcts_config()
        assert isinstance(mcts_cfg, MCTSConfig)
        assert mcts_cfg.num_simulations == 100
        assert mcts_cfg.c_puct == 3.0
        assert mcts_cfg.action_dim == 246


# ---------------------------------------------------------------------------
# Replay Buffer Tests
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    def test_add_and_len(self) -> None:
        buf = ReplayBuffer(capacity=100, visible_size=4, action_dim=2, num_players=3)
        assert len(buf) == 0
        buf.add_examples([_make_example(4, 2, 3) for _ in range(10)])
        assert len(buf) == 10

    def test_sample_shapes(self) -> None:
        buf = ReplayBuffer(capacity=100, visible_size=3023, action_dim=246, num_players=3)
        buf.add_examples([_make_example(3023, 246, 3) for _ in range(20)])
        rng = np.random.default_rng(0)
        batch = buf.sample(batch_size=8, rng=rng)
        assert batch["states"].shape == (8, 3023)
        assert batch["legal_masks"].shape == (8, 246)
        assert batch["policy_targets"].shape == (8, 246)
        assert batch["value_targets"].shape == (8, 3)
        assert isinstance(batch["states"], torch.Tensor)

    def test_ring_wrap(self) -> None:
        buf = ReplayBuffer(capacity=10, visible_size=4, action_dim=2, num_players=3)
        for i in range(15):
            ex = TrainingExample(
                state=np.full(4, float(i), dtype=np.float32),
                legal_mask=np.ones(2, dtype=np.float32),
                policy_target=np.array([0.5, 0.5], dtype=np.float32),
                value_target=np.zeros(3, dtype=np.float32),
            )
            buf.add_examples([ex])
        assert len(buf) == 10
        rng = np.random.default_rng(0)
        batch = buf.sample(10, rng)
        vals = sorted(batch["states"][:, 0].tolist())
        assert vals == list(range(5, 15))

    def test_batch_overflow_keeps_last_capacity(self) -> None:
        """Adding more examples than capacity should keep only the last `capacity`."""
        buf = ReplayBuffer(capacity=5, visible_size=4, action_dim=2, num_players=3)
        examples = [
            TrainingExample(
                state=np.full(4, float(i), dtype=np.float32),
                legal_mask=np.ones(2, dtype=np.float32),
                policy_target=np.array([0.5, 0.5], dtype=np.float32),
                value_target=np.zeros(3, dtype=np.float32),
            )
            for i in range(10)
        ]
        buf.add_examples(examples)
        assert len(buf) == 5
        rng = np.random.default_rng(0)
        batch = buf.sample(5, rng)
        vals = sorted(batch["states"][:, 0].tolist())
        assert vals == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_sample_raises_when_batch_exceeds_size(self) -> None:
        buf = ReplayBuffer(capacity=100, visible_size=4, action_dim=2, num_players=3)
        buf.add_examples([_make_example(4, 2, 3) for _ in range(3)])
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="batch_size.*exceeds"):
            buf.sample(batch_size=10, rng=rng)

    def test_sample_exact_size(self) -> None:
        """Sampling exactly len(buffer) items should work."""
        buf = ReplayBuffer(capacity=100, visible_size=4, action_dim=2, num_players=3)
        buf.add_examples([_make_example(4, 2, 3) for _ in range(7)])
        rng = np.random.default_rng(0)
        batch = buf.sample(batch_size=7, rng=rng)
        assert batch["states"].shape == (7, 4)

    def test_sample_only_from_filled(self) -> None:
        buf = ReplayBuffer(capacity=100, visible_size=4, action_dim=2, num_players=3)
        buf.add_examples([_make_example(4, 2, 3) for _ in range(5)])
        rng = np.random.default_rng(0)
        batch = buf.sample(batch_size=3, rng=rng)
        assert batch["states"].shape == (3, 4)


# ---------------------------------------------------------------------------
# Trainer Tests
# ---------------------------------------------------------------------------


class TestTrainer:
    def test_train_step_returns_losses(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        device = torch.device("cpu")
        trainer = Trainer(small_model, tiny_config, device)
        batch = _make_batch(4, 3023, 246, 3)
        losses = trainer.train_step(batch)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "total_loss" in losses
        assert all(np.isfinite(v) for v in losses.values())

    def test_global_step_increments(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        device = torch.device("cpu")
        trainer = Trainer(small_model, tiny_config, device)
        assert trainer.global_step == 0
        batch = _make_batch(4, 3023, 246, 3)
        trainer.train_step(batch)
        assert trainer.global_step == 1
        trainer.train_step(batch)
        assert trainer.global_step == 2

    def test_lr_schedule_warmup_and_decay(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        device = torch.device("cpu")
        tiny_config.warmup_steps = 10
        tiny_config.training_steps_per_epoch = 100
        tiny_config.num_epochs = 10
        trainer = Trainer(small_model, tiny_config, device)
        batch = _make_batch(4, 3023, 246, 3)
        lrs = []
        for _ in range(50):
            trainer.train_step(batch)
            lrs.append(trainer.lr)
        # Warmup: LR should increase over first 10 steps
        assert lrs[9] > lrs[0]
        # After warmup: LR should decrease (cosine decay)
        assert lrs[49] < lrs[10]
        # Should not go below lr_min (with small tolerance)
        assert all(lr >= tiny_config.lr_min * 0.99 for lr in lrs)

    def test_parameters_stay_finite(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        device = torch.device("cpu")
        trainer = Trainer(small_model, tiny_config, device)
        batch = _make_batch(4, 3023, 246, 3)
        trainer.train_step(batch)
        assert all(torch.isfinite(p).all() for p in small_model.parameters())

    def test_state_dict_roundtrip(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        device = torch.device("cpu")
        trainer = Trainer(small_model, tiny_config, device)
        batch = _make_batch(4, 3023, 246, 3)
        trainer.train_step(batch)
        trainer.train_step(batch)
        state = trainer.state_dict()
        assert state["global_step"] == 2
        trainer2 = Trainer(small_model, tiny_config, device)
        trainer2.load_state_dict(state)
        assert trainer2.global_step == 2


# ---------------------------------------------------------------------------
# Checkpoint Tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_load_roundtrip(
        self,
        small_model: RSSAlphaZeroNet,
        tiny_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        device = torch.device("cpu")
        trainer = Trainer(small_model, tiny_config, device)
        cp_path = tmp_path / "test_checkpoint.pt"
        save_checkpoint(
            cp_path,
            epoch=5,
            model=small_model,
            trainer_state=trainer.state_dict(),
            config=tiny_config,
            metrics={"total_loss": 1.23},
            buffer_stats={"size": 100, "capacity": 1000},
        )
        loaded = load_checkpoint(cp_path, device)
        assert loaded["epoch"] == 5
        assert loaded["metrics"]["total_loss"] == 1.23  # type: ignore[index]
        restored = TrainingConfig.from_json(loaded["config_json"])  # type: ignore[arg-type]
        assert restored.num_epochs == tiny_config.num_epochs

    def test_find_latest(self, tmp_path: Path) -> None:
        for epoch in [1, 5, 3]:
            (tmp_path / f"checkpoint_epoch_{epoch:04d}.pt").touch()
        latest = find_latest_checkpoint(tmp_path)
        assert latest is not None
        assert "0005" in latest.name

    def test_find_latest_empty(self, tmp_path: Path) -> None:
        assert find_latest_checkpoint(tmp_path) is None

    def test_resume_continues_training(
        self,
        small_model: RSSAlphaZeroNet,
        tiny_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """After loading a checkpoint, training should continue correctly."""
        device = torch.device("cpu")
        trainer = Trainer(small_model, tiny_config, device)
        batch = _make_batch(4, 3023, 246, 3)
        trainer.train_step(batch)
        trainer.train_step(batch)
        original_step = trainer.global_step
        original_lr = trainer.lr

        # Save
        cp_path = tmp_path / "resume_test.pt"
        save_checkpoint(
            cp_path,
            epoch=0,
            model=small_model,
            trainer_state=trainer.state_dict(),
            config=tiny_config,
            metrics={"total_loss": 1.0},
            buffer_stats={"size": 10, "capacity": 1000},
        )

        # Create fresh model and trainer, load checkpoint
        model2 = RSSAlphaZeroNet(RSSModelConfig(
            input_dim=3023, action_dim=246, value_dim=3,
            hidden_dim=32, num_blocks=1, expansion=1,
        ))
        trainer2 = Trainer(model2, tiny_config, device)
        cp = load_checkpoint(cp_path, device)
        model2.load_state_dict(cp["model_state_dict"])  # type: ignore[arg-type]
        trainer2.load_state_dict(cp["trainer_state"])  # type: ignore[arg-type]

        assert trainer2.global_step == original_step
        assert trainer2.lr == pytest.approx(original_lr)

        # Training step after resume should work
        losses = trainer2.train_step(batch)
        assert trainer2.global_step == original_step + 1
        assert np.isfinite(losses["total_loss"])

    def test_cleanup(self, tmp_path: Path) -> None:
        for epoch in range(1, 8):
            (tmp_path / f"checkpoint_epoch_{epoch:04d}.pt").touch()
        cleanup_checkpoints(tmp_path, keep_last_n=3)
        remaining = sorted(tmp_path.glob("checkpoint_epoch_*.pt"))
        assert len(remaining) == 3
        assert "0005" in remaining[0].name


# ---------------------------------------------------------------------------
# Self-Play Integration Test
# ---------------------------------------------------------------------------


class TestSelfPlay:
    def test_play_game_produces_valid_record(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        from mcts.evaluator import NNEvaluator

        device = torch.device("cpu")
        small_model.eval()
        evaluator = NNEvaluator(small_model, device, num_players=tiny_config.num_players)
        rng = np.random.default_rng(42)
        record = play_game(evaluator, tiny_config, game_seed=123, rng=rng)

        assert record.total_moves > 0
        assert len(record.examples) == record.total_moves
        assert len(record.net_worths) == tiny_config.num_players
        assert record.duration_secs > 0

        ex = record.examples[0]
        assert ex.state.shape == (tiny_config.visible_size,)
        assert ex.legal_mask.shape == (tiny_config.action_dim,)
        assert ex.policy_target.shape == (tiny_config.action_dim,)
        assert ex.value_target.shape == (tiny_config.num_players,)

        for ex in record.examples:
            assert abs(ex.policy_target.sum() - 1.0) < 1e-5
            assert (ex.policy_target >= 0).all()
            assert (ex.value_target >= -1.0 - 1e-5).all()
            assert (ex.value_target <= 1.0 + 1e-5).all()

    def test_remote_evaluator_matches_local(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        """RemoteEvaluator through EvaluationServer produces same results as NNEvaluator."""
        from multiprocessing import Pipe

        from mcts.evaluator import NNEvaluator

        from core.state import GameState
        from train.eval_server import EvaluationServer, RemoteEvaluator

        device = torch.device("cpu")
        small_model.eval()
        num_players = tiny_config.num_players

        # Set up a game state to evaluate
        state = GameState(num_players)
        state.initialize_game(seed=42)

        # Local evaluation
        local_eval = NNEvaluator(small_model, device, num_players=num_players)
        local_policy, local_values = local_eval.evaluate(state)

        # Remote evaluation through server
        server_conn, worker_conn = Pipe()
        server = EvaluationServer(small_model, device, [server_conn])
        server.start()
        try:
            remote_eval = RemoteEvaluator(worker_conn, num_players)
            remote_policy, remote_values = remote_eval.evaluate(state)
        finally:
            server.stop()
            server_conn.close()
            worker_conn.close()

        np.testing.assert_allclose(remote_policy, local_policy, atol=1e-6)
        np.testing.assert_allclose(remote_values, local_values, atol=1e-6)

    def test_remote_evaluator_batch(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        """RemoteEvaluator.evaluate_batch matches NNEvaluator.evaluate_batch."""
        from multiprocessing import Pipe

        from mcts.evaluator import NNEvaluator

        from core.driver import DRIVER
        from core.state import GameState
        from train.eval_server import EvaluationServer, RemoteEvaluator

        device = torch.device("cpu")
        small_model.eval()
        num_players = tiny_config.num_players

        # Create a few different game states
        states = []
        state = GameState(num_players)
        state.initialize_game(seed=42)
        states.append(state)
        # Advance a few actions to get different states
        for seed in [99, 77]:
            s = GameState(num_players)
            s.initialize_game(seed=seed)
            legal = DRIVER.get_legal_moves(s)
            action = int(np.argmax(legal))
            DRIVER.apply_action(s, action)
            states.append(s)

        # Local batch evaluation
        local_eval = NNEvaluator(small_model, device, num_players=num_players)
        local_results = local_eval.evaluate_batch(states)

        # Remote batch evaluation
        server_conn, worker_conn = Pipe()
        server = EvaluationServer(small_model, device, [server_conn])
        server.start()
        try:
            remote_eval = RemoteEvaluator(worker_conn, num_players)
            remote_results = remote_eval.evaluate_batch(states)
        finally:
            server.stop()
            server_conn.close()
            worker_conn.close()

        assert len(remote_results) == len(local_results)
        for (rp, rv), (lp, lv) in zip(remote_results, local_results):
            np.testing.assert_allclose(rp, lp, atol=1e-6)
            np.testing.assert_allclose(rv, lv, atol=1e-6)

    def test_play_game_with_remote_evaluator(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        """play_game produces valid results when using RemoteEvaluator."""
        from multiprocessing import Pipe

        from train.eval_server import EvaluationServer, RemoteEvaluator

        device = torch.device("cpu")
        small_model.eval()

        server_conn, worker_conn = Pipe()
        server = EvaluationServer(small_model, device, [server_conn])
        server.start()
        try:
            remote_eval = RemoteEvaluator(worker_conn, tiny_config.num_players)
            rng = np.random.default_rng(42)
            record = play_game(remote_eval, tiny_config, game_seed=123, rng=rng)
        finally:
            server.stop()
            server_conn.close()
            worker_conn.close()

        assert record.total_moves > 0
        assert len(record.examples) == record.total_moves
        assert len(record.net_worths) == tiny_config.num_players
        for ex in record.examples:
            assert abs(ex.policy_target.sum() - 1.0) < 1e-5

    def test_multiprocess_workers(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        """End-to-end test: spawn actual worker processes, play games, collect results."""
        import multiprocessing as mp

        from train.eval_server import EvaluationServer
        from train.self_play import self_play_worker

        device = torch.device("cpu")
        small_model.eval()
        num_workers = 2
        games_per_worker = 1  # 2 games total

        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()

        server_conns = []
        worker_conns = []
        for _ in range(num_workers):
            s_conn, w_conn = ctx.Pipe()
            server_conns.append(s_conn)
            worker_conns.append(w_conn)

        server = EvaluationServer(small_model, device, server_conns)
        server.start()

        workers = []
        for i in range(num_workers):
            p = ctx.Process(
                target=self_play_worker,
                args=(worker_conns[i], task_queue, result_queue, tiny_config),
                daemon=True,
            )
            p.start()
            workers.append(p)

        for conn in worker_conns:
            conn.close()

        try:
            # Feed game seeds
            total_games = num_workers * games_per_worker
            for i in range(total_games):
                task_queue.put((42 + i, 100 + i))

            # Collect results
            records = []
            for _ in range(total_games):
                record = result_queue.get(timeout=120.0)
                records.append(record)

            assert len(records) == total_games
            for record in records:
                assert record.total_moves > 0
                assert len(record.examples) == record.total_moves
                assert len(record.net_worths) == tiny_config.num_players
        finally:
            # Clean shutdown
            for _ in workers:
                task_queue.put(None)
            server.stop()
            for conn in server_conns:
                conn.close()
            for w in workers:
                w.join(timeout=5.0)
                if w.is_alive():
                    w.terminate()


# ---------------------------------------------------------------------------
# End-to-End Integration Test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_game_to_buffer_to_training(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        device = torch.device("cpu")

        # Self-play
        from mcts.evaluator import NNEvaluator

        small_model.eval()
        evaluator = NNEvaluator(small_model, device, num_players=tiny_config.num_players)
        rng = np.random.default_rng(42)
        record = play_game(evaluator, tiny_config, game_seed=1, rng=rng)

        # Buffer
        buf = ReplayBuffer(
            tiny_config.buffer_capacity,
            tiny_config.visible_size,
            tiny_config.action_dim,
            tiny_config.num_players,
        )
        buf.add_examples(record.examples)
        assert len(buf) > 0

        # Training
        small_model.train()
        trainer = Trainer(small_model, tiny_config, device)
        batch = buf.sample(
            min(tiny_config.batch_size, len(buf)), rng=np.random.default_rng(0)
        )
        losses = trainer.train_step(batch)
        assert np.isfinite(losses["total_loss"])


# ---------------------------------------------------------------------------
# Logging Smoke Test
# ---------------------------------------------------------------------------


class TestLogging:
    def test_logger_does_not_crash(
        self, tiny_config: TrainingConfig, tmp_path: Path
    ) -> None:
        logger = TrainingLogger(str(tmp_path / "tb"))
        logger.log_training_start(tiny_config, device="cpu")

        logger.begin_self_play(epoch=1, num_epochs=2, total_games=1)
        logger.update_self_play(
            games_done=1, total_examples=50, avg_moves=50.0
        )
        logger.end_self_play()

        logger.begin_training(epoch=1, num_epochs=2, total_steps=2)
        logger.update_training(
            step=1,
            losses={"policy_loss": 1.0, "value_loss": 0.5, "total_loss": 1.5},
            lr=1e-3,
        )
        logger.end_training()

        logger.log_scalars(1, {"loss/total": 1.5, "lr": 1e-3})

        logger.log_epoch_summary(
            epoch=1,
            num_epochs=2,
            self_play_stats={"games": 1.0, "examples": 50.0, "avg_moves": 50.0, "avg_duration": 1.0},
            train_stats={"steps": 2.0, "total_loss": 1.5, "policy_loss": 1.0, "value_loss": 0.5, "lr": 1e-3},
            buffer_size=50,
            buffer_capacity=1000,
            epoch_duration=10.0,
        )
        logger.close()

        # Tensorboard files should exist
        assert any((tmp_path / "tb").iterdir())


# ---------------------------------------------------------------------------
# CLI Override Tests
# ---------------------------------------------------------------------------


class TestCLIOverrides:
    def test_apply_overrides(self) -> None:
        config = TrainingConfig()
        parser = _build_parser()
        args = parser.parse_args([
            "--games-per-epoch", "5",
            "--num-epochs", "3",
            "--num-simulations", "10",
            "--search-batch-size", "4",
            "--seed", "99",
        ])
        _apply_overrides(config, args)
        assert config.games_per_epoch == 5
        assert config.num_epochs == 3
        assert config.num_simulations == 10
        assert config.search_batch_size == 4
        assert config.seed == 99

    def test_no_overrides_leaves_defaults(self) -> None:
        config = TrainingConfig()
        original_games = config.games_per_epoch
        parser = _build_parser()
        args = parser.parse_args([])
        _apply_overrides(config, args)
        assert config.games_per_epoch == original_games


# ---------------------------------------------------------------------------
# Duration Formatting Tests
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_sub_second(self) -> None:
        assert _format_duration(0.3) == "0.3s"

    def test_zero(self) -> None:
        assert _format_duration(0.0) == "0.0s"

    def test_seconds(self) -> None:
        assert _format_duration(45.0) == "45s"

    def test_minutes_and_seconds(self) -> None:
        assert _format_duration(125.0) == "2m 05s"

    def test_hours(self) -> None:
        assert _format_duration(3661.0) == "1h 01m 01s"
