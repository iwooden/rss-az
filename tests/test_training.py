"""Tests for the self-play training harness.

Uses tiny configs and small models to keep tests fast (< 30s total).
All file output uses tmp_path to avoid polluting the project directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from nn.model_3p import RSSAlphaZeroNet, RSSModelConfig
from nn.model_3p_plus import RSSAlphaZeroNet as RSSAlphaZeroNetPlus
from nn.model_3p_plus import RSSModelConfig as RSSModelConfigPlus
from nn.model_4p import RSSAlphaZeroNet as RSSAlphaZeroNet4P
from nn.model_4p import RSSModelConfig as RSSModelConfig4P
from train.gpu.amd import get_compile_kwargs as get_amd_compile_kwargs
from train.gpu.nvidia import get_compile_kwargs as get_nvidia_compile_kwargs
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

from core.actions import get_total_action_count
from core.state import get_layout

# Computed layout constants (single source of truth)
_L3 = get_layout(3)
_VIS = _L3.visible_size
_ACT = get_total_action_count(3)

_L4 = get_layout(4)
_VIS4 = _L4.visible_size
_ACT4 = get_total_action_count(4)


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


def _stack_examples(examples: list[TrainingExample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack a list of TrainingExamples into 4 contiguous arrays."""
    return (
        np.stack([e.state for e in examples]),
        np.stack([e.legal_mask for e in examples]),
        np.stack([e.policy_target for e in examples]),
        np.stack([e.value_target for e in examples]),
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
        temp_anneal_start=3,
        temp_anneal_end=5,
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
        input_dim=_VIS,
        action_dim=_ACT,
        value_dim=3,
        hidden_dim=32,
        num_blocks=1,
    )
    return RSSAlphaZeroNet(cfg)


@pytest.fixture
def small_model_plus() -> RSSAlphaZeroNetPlus:
    cfg = RSSModelConfigPlus(
        input_dim=_VIS,
        action_dim=_ACT,
        value_dim=3,
        hidden_dim=32,
        num_blocks=1,
    )
    return RSSAlphaZeroNetPlus(cfg)


@pytest.fixture
def small_model_4p() -> RSSAlphaZeroNet4P:
    cfg = RSSModelConfig4P(
        input_dim=_VIS4,
        action_dim=_ACT4,
        value_dim=4,
        hidden_dim=32,
        num_blocks=1,
    )
    return RSSAlphaZeroNet4P(cfg)


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_computed_properties(self) -> None:
        cfg = TrainingConfig(num_players=3)
        assert cfg.action_dim == _ACT
        assert cfg.visible_size == _VIS

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

    def test_validation_num_simulations(self) -> None:
        with pytest.raises(ValueError, match="num_simulations"):
            TrainingConfig(num_simulations=0)

    def test_validation_search_batch_size(self) -> None:
        with pytest.raises(ValueError, match="search_batch_size"):
            TrainingConfig(search_batch_size=0)

    def test_validation_c_puct(self) -> None:
        with pytest.raises(ValueError, match="c_puct_initial"):
            TrainingConfig(c_puct_initial=-1.0)
        with pytest.raises(ValueError, match="c_puct_final"):
            TrainingConfig(c_puct_final=-1.0)

    def test_validation_dirichlet(self) -> None:
        with pytest.raises(ValueError, match="dirichlet_alpha"):
            TrainingConfig(dirichlet_alpha=0)
        with pytest.raises(ValueError, match="dirichlet_epsilon"):
            TrainingConfig(dirichlet_epsilon=2.0)
        with pytest.raises(ValueError, match="dirichlet_alpha_numerator"):
            TrainingConfig(dirichlet_alpha_numerator=0)

    def test_validation_num_workers(self) -> None:
        with pytest.raises(ValueError, match="num_workers"):
            TrainingConfig(num_workers=-1)

    def test_validation_num_eval_servers(self) -> None:
        with pytest.raises(ValueError, match="num_eval_servers"):
            TrainingConfig(num_eval_servers=0)

    def test_validation_servers_gt_workers(self) -> None:
        with pytest.raises(ValueError, match="num_eval_servers.*must be <="):
            TrainingConfig(num_workers=2, num_eval_servers=3)

    def test_validation_partition_size_over_64(self) -> None:
        with pytest.raises(ValueError, match="partition size.*exceeds 64"):
            TrainingConfig(num_workers=65, num_eval_servers=1)

    def test_validation_partition_size_at_64(self) -> None:
        # Exactly 64 should be accepted (fits in uint64)
        cfg = TrainingConfig(num_workers=64, num_eval_servers=1)
        assert cfg.num_workers == 64

    def test_validation_partition_size_split(self) -> None:
        # 65 workers / 2 servers → partitions of 33 and 32, both <= 64
        cfg = TrainingConfig(num_workers=65, num_eval_servers=2)
        assert cfg.num_workers == 65

    def test_to_mcts_config(self) -> None:
        cfg = TrainingConfig(num_simulations=100, c_puct_final=3.0, num_players=3)
        mcts_cfg = cfg.to_mcts_config()
        assert isinstance(mcts_cfg, MCTSConfig)
        assert mcts_cfg.num_simulations == 100
        assert mcts_cfg.c_puct == 3.0
        assert mcts_cfg.action_dim == _ACT
        assert mcts_cfg.dirichlet_dynamic is True
        assert mcts_cfg.dirichlet_alpha_numerator == 10.0

    def test_to_mcts_config_c_puct_override(self) -> None:
        cfg = TrainingConfig(c_puct_final=2.5)
        mcts_cfg = cfg.to_mcts_config(c_puct_override=3.5)
        assert mcts_cfg.c_puct == 3.5

    def test_compute_epoch_config(self) -> None:
        cfg = TrainingConfig(
            c_puct_initial=4.0, c_puct_final=2.0, c_puct_anneal_epochs=20,
            value_blend_start_epoch=10, value_blend_end_epoch=40,
        )
        # Epoch 0: c_puct at initial, no blend
        ec = cfg.compute_epoch_config(0)
        assert ec.c_puct == pytest.approx(4.0)
        assert ec.value_blend_alpha == pytest.approx(0.0)

        # Epoch 10: c_puct halfway, blend starts
        ec = cfg.compute_epoch_config(10)
        assert ec.c_puct == pytest.approx(3.0)
        assert ec.value_blend_alpha == pytest.approx(0.0)

        # Epoch 20: c_puct at final
        ec = cfg.compute_epoch_config(20)
        assert ec.c_puct == pytest.approx(2.0)

        # Epoch 25: halfway through blend
        ec = cfg.compute_epoch_config(25)
        assert ec.value_blend_alpha == pytest.approx(0.5)

        # Epoch 40+: fully A0GB
        ec = cfg.compute_epoch_config(50)
        assert ec.c_puct == pytest.approx(2.0)
        assert ec.value_blend_alpha == pytest.approx(1.0)

    def test_compute_epoch_config_sim_ramp(self) -> None:
        cfg = TrainingConfig(
            num_simulations=800,
            mcts_sims_start=800, mcts_sims_end=1600,
            mcts_ramp_start_epoch=100, mcts_ramp_end_epoch=200,
        )
        # Before ramp: use start value
        assert cfg.compute_epoch_config(0).num_simulations == 800
        assert cfg.compute_epoch_config(50).num_simulations == 800
        assert cfg.compute_epoch_config(100).num_simulations == 800

        # During ramp: linear interpolation
        assert cfg.compute_epoch_config(150).num_simulations == 1200

        # After ramp: use end value
        assert cfg.compute_epoch_config(200).num_simulations == 1600
        assert cfg.compute_epoch_config(300).num_simulations == 1600

    def test_compute_epoch_config_no_sim_ramp(self) -> None:
        cfg = TrainingConfig(num_simulations=800)
        # Without ramp fields, num_simulations stays fixed
        assert cfg.compute_epoch_config(0).num_simulations == 800
        assert cfg.compute_epoch_config(100).num_simulations == 800

    def test_max_simulations_with_ramp(self) -> None:
        cfg = TrainingConfig(
            num_simulations=800,
            mcts_sims_start=800, mcts_sims_end=1600,
            mcts_ramp_start_epoch=100, mcts_ramp_end_epoch=200,
        )
        assert cfg.max_simulations == 1600

    def test_max_simulations_without_ramp(self) -> None:
        cfg = TrainingConfig(num_simulations=800)
        assert cfg.max_simulations == 800

    def test_to_mcts_config_num_simulations_override(self) -> None:
        cfg = TrainingConfig(num_simulations=800)
        mcts_cfg = cfg.to_mcts_config(num_simulations_override=1200)
        assert mcts_cfg.num_simulations == 1200

    def test_sim_ramp_partial_fields_raises(self) -> None:
        with pytest.raises(ValueError, match="must all be set or all be None"):
            TrainingConfig(mcts_sims_start=800, mcts_sims_end=1600)

    def test_to_mcts_config_dynamic_dirichlet(self) -> None:
        cfg = TrainingConfig(
            dirichlet_dynamic=True,
            dirichlet_alpha_numerator=15.0,
        )
        mcts_cfg = cfg.to_mcts_config()
        assert mcts_cfg.dirichlet_dynamic is True
        assert mcts_cfg.dirichlet_alpha_numerator == 15.0

    # --- Auto model path and multi-player support ---

    def test_auto_model_path_3p(self) -> None:
        cfg = TrainingConfig(num_players=3)
        assert cfg.model_path == "nn.model_3p"

    def test_auto_model_path_4p(self) -> None:
        cfg = TrainingConfig(num_players=4)
        assert cfg.model_path == "nn.model_4p"

    def test_explicit_model_path_preserved(self) -> None:
        cfg = TrainingConfig(num_players=4, model_path="nn.model_custom")
        assert cfg.model_path == "nn.model_custom"

    def test_computed_properties_4p(self) -> None:
        cfg = TrainingConfig(num_players=4)
        assert cfg.action_dim == _ACT4
        assert cfg.visible_size == _VIS4
        assert cfg.action_dim != _ACT, "4p action_dim should differ from 3p"
        assert cfg.visible_size != _VIS, "4p visible_size should differ from 3p"

    def test_json_roundtrip_4p(self) -> None:
        cfg = TrainingConfig(num_players=4)
        restored = TrainingConfig.from_json(cfg.to_json())
        assert restored.num_players == 4
        assert restored.model_path == "nn.model_4p"
        assert restored.action_dim == _ACT4
        assert restored.visible_size == _VIS4

    def test_json_roundtrip_preserves_explicit_model_path(self) -> None:
        cfg = TrainingConfig(num_players=3, model_path="nn.model_3p_plus")
        restored = TrainingConfig.from_json(cfg.to_json())
        assert restored.model_path == "nn.model_3p_plus"


# ---------------------------------------------------------------------------
# Bitmap Signaling Tests
# ---------------------------------------------------------------------------


class TestBitmapSignaling:
    """Tests for the atomic bitmap primitives in mcts_core.pyx."""

    def test_publish_and_drain_single(self) -> None:
        """Single worker publish + server drain round-trip."""
        from mcts.mcts_core import worker_publish_request, server_drain_bitmap
        masks = np.zeros(1, dtype=np.uint64)
        counts = np.zeros(4, dtype=np.int32)
        out_widx = np.zeros(4, dtype=np.int32)
        out_cnt = np.zeros(4, dtype=np.int32)

        became_nonempty = worker_publish_request(masks, counts, 2, 0, 2, 5)
        assert became_nonempty is True

        n = server_drain_bitmap(masks, counts, out_widx, out_cnt, 0, 0)
        assert n == 1
        assert out_widx[0] == 2
        assert out_cnt[0] == 5
        assert masks[0] == 0  # exchanged to zero

    def test_publish_multiple_workers(self) -> None:
        """Multiple workers publish, drain gets all of them."""
        from mcts.mcts_core import worker_publish_request, server_drain_bitmap
        masks = np.zeros(1, dtype=np.uint64)
        counts = np.zeros(8, dtype=np.int32)
        out_widx = np.zeros(8, dtype=np.int32)
        out_cnt = np.zeros(8, dtype=np.int32)

        # First publish transitions 0 -> non-zero
        assert worker_publish_request(masks, counts, 1, 0, 1, 3) is True
        # Subsequent publishes see non-zero mask
        assert worker_publish_request(masks, counts, 5, 0, 5, 7) is False
        assert worker_publish_request(masks, counts, 0, 0, 0, 1) is False

        n = server_drain_bitmap(masks, counts, out_widx, out_cnt, 0, 0)
        assert n == 3
        result = sorted(zip(out_widx[:n].tolist(), out_cnt[:n].tolist()))
        assert result == [(0, 1), (1, 3), (5, 7)]

    def test_drain_empty_bitmap(self) -> None:
        """Draining an empty bitmap returns 0."""
        from mcts.mcts_core import server_drain_bitmap
        masks = np.zeros(1, dtype=np.uint64)
        counts = np.zeros(4, dtype=np.int32)
        out_widx = np.zeros(4, dtype=np.int32)
        out_cnt = np.zeros(4, dtype=np.int32)

        n = server_drain_bitmap(masks, counts, out_widx, out_cnt, 0, 0)
        assert n == 0

    def test_peek_bitmap(self) -> None:
        """Peek returns True when work is pending, False when empty."""
        from mcts.mcts_core import (
            worker_publish_request, server_drain_bitmap, server_peek_bitmap,
        )
        masks = np.zeros(1, dtype=np.uint64)
        counts = np.zeros(4, dtype=np.int32)
        out_widx = np.zeros(4, dtype=np.int32)
        out_cnt = np.zeros(4, dtype=np.int32)

        assert server_peek_bitmap(masks, 0) is False
        worker_publish_request(masks, counts, 0, 0, 0, 1)
        assert server_peek_bitmap(masks, 0) is True
        # Peek doesn't consume — drain still finds it
        server_drain_bitmap(masks, counts, out_widx, out_cnt, 0, 0)
        assert server_peek_bitmap(masks, 0) is False

    def test_partition_offset(self) -> None:
        """Drain maps local bit index back to global worker index."""
        from mcts.mcts_core import worker_publish_request, server_drain_bitmap
        masks = np.zeros(2, dtype=np.uint64)
        counts = np.zeros(8, dtype=np.int32)
        out_widx = np.zeros(4, dtype=np.int32)
        out_cnt = np.zeros(4, dtype=np.int32)

        # Server 1 owns workers [4, 8). Worker 5 is local_idx=1.
        worker_publish_request(masks, counts, 5, 1, 1, 10)
        n = server_drain_bitmap(masks, counts, out_widx, out_cnt, 1, 4)
        assert n == 1
        assert out_widx[0] == 5  # global index, not local
        assert out_cnt[0] == 10

    def test_high_bit_index(self) -> None:
        """Bit 63 (max for uint64) works correctly."""
        from mcts.mcts_core import worker_publish_request, server_drain_bitmap
        masks = np.zeros(1, dtype=np.uint64)
        counts = np.zeros(64, dtype=np.int32)
        out_widx = np.zeros(64, dtype=np.int32)
        out_cnt = np.zeros(64, dtype=np.int32)

        worker_publish_request(masks, counts, 63, 0, 63, 2)
        n = server_drain_bitmap(masks, counts, out_widx, out_cnt, 0, 0)
        assert n == 1
        assert out_widx[0] == 63
        assert out_cnt[0] == 2

    def test_all_64_bits(self) -> None:
        """All 64 workers in a partition can publish and be drained."""
        from mcts.mcts_core import worker_publish_request, server_drain_bitmap
        masks = np.zeros(1, dtype=np.uint64)
        counts = np.zeros(64, dtype=np.int32)
        out_widx = np.zeros(64, dtype=np.int32)
        out_cnt = np.zeros(64, dtype=np.int32)

        for i in range(64):
            worker_publish_request(masks, counts, i, 0, i, i + 1)

        n = server_drain_bitmap(masks, counts, out_widx, out_cnt, 0, 0)
        assert n == 64
        result = sorted(zip(out_widx[:n].tolist(), out_cnt[:n].tolist()))
        assert result == [(i, i + 1) for i in range(64)]

    def test_repeated_publish_drain_cycles(self) -> None:
        """Multiple publish/drain cycles don't leave stale state."""
        from mcts.mcts_core import worker_publish_request, server_drain_bitmap
        masks = np.zeros(1, dtype=np.uint64)
        counts = np.zeros(4, dtype=np.int32)
        out_widx = np.zeros(4, dtype=np.int32)
        out_cnt = np.zeros(4, dtype=np.int32)

        for cycle in range(5):
            became_nonempty = worker_publish_request(
                masks, counts, 0, 0, 0, cycle + 1,
            )
            assert became_nonempty is True  # always 0->non-zero since we drain each time
            n = server_drain_bitmap(masks, counts, out_widx, out_cnt, 0, 0)
            assert n == 1
            assert out_cnt[0] == cycle + 1

    def test_concurrent_publish_no_lost_bits(self) -> None:
        """Concurrent publishes from multiple threads don't lose any bits."""
        import threading
        from mcts.mcts_core import worker_publish_request, server_drain_bitmap

        num_workers = 48
        masks = np.zeros(1, dtype=np.uint64)
        counts = np.zeros(num_workers, dtype=np.int32)
        wakeups = [False] * num_workers

        barrier = threading.Barrier(num_workers)

        def publish(w: int) -> None:
            barrier.wait()
            wakeups[w] = worker_publish_request(masks, counts, w, 0, w, 1)

        threads = [threading.Thread(target=publish, args=(i,)) for i in range(num_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one thread should have seen the 0->non-zero transition
        assert sum(wakeups) == 1

        # Drain should find all workers
        out_widx = np.zeros(num_workers, dtype=np.int32)
        out_cnt = np.zeros(num_workers, dtype=np.int32)
        n = server_drain_bitmap(masks, counts, out_widx, out_cnt, 0, 0)
        assert n == num_workers
        assert sorted(out_widx[:n].tolist()) == list(range(num_workers))


# ---------------------------------------------------------------------------
# Replay Buffer Tests
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    def test_add_and_len(self) -> None:
        buf = ReplayBuffer(capacity=100, visible_size=4, action_dim=2, num_players=3)
        assert len(buf) == 0
        buf.add_stacked(*_stack_examples([_make_example(4, 2, 3) for _ in range(10)]))
        assert len(buf) == 10

    def test_sample_shapes(self) -> None:
        buf = ReplayBuffer(capacity=100, visible_size=_VIS, action_dim=_ACT, num_players=3)
        buf.add_stacked(*_stack_examples([_make_example(_VIS, _ACT, 3) for _ in range(20)]))
        rng = np.random.default_rng(0)
        batch = buf.sample(batch_size=8, rng=rng)
        assert batch["states"].shape == (8, _VIS)
        assert batch["legal_masks"].shape == (8, _ACT)
        assert batch["policy_targets"].shape == (8, _ACT)
        assert batch["value_targets"].shape == (8, 3)
        assert isinstance(batch["states"], torch.Tensor)

    def test_ring_wrap(self) -> None:
        buf = ReplayBuffer(capacity=10, visible_size=4, action_dim=2, num_players=3)
        for i in range(15):
            buf.add_stacked(
                np.full((1, 4), float(i), dtype=np.float32),
                np.ones((1, 2), dtype=np.float32),
                np.array([[0.5, 0.5]], dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
            )
        assert len(buf) == 10
        rng = np.random.default_rng(0)
        batch = buf.sample(10, rng)
        vals = sorted(batch["states"][:, 0].tolist())
        assert vals == list(range(5, 15))

    def test_batch_overflow_keeps_last_capacity(self) -> None:
        """Adding more examples than capacity should keep only the last `capacity`."""
        buf = ReplayBuffer(capacity=5, visible_size=4, action_dim=2, num_players=3)
        buf.add_stacked(
            np.stack([np.full(4, float(i), dtype=np.float32) for i in range(10)]),
            np.ones((10, 2), dtype=np.float32),
            np.full((10, 2), 0.5, dtype=np.float32),
            np.zeros((10, 3), dtype=np.float32),
        )
        assert len(buf) == 5
        rng = np.random.default_rng(0)
        batch = buf.sample(5, rng)
        vals = sorted(batch["states"][:, 0].tolist())
        assert vals == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_sample_raises_when_batch_exceeds_size(self) -> None:
        buf = ReplayBuffer(capacity=100, visible_size=4, action_dim=2, num_players=3)
        buf.add_stacked(*_stack_examples([_make_example(4, 2, 3) for _ in range(3)]))
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="batch_size.*exceeds"):
            buf.sample(batch_size=10, rng=rng)

    def test_sample_exact_size(self) -> None:
        """Sampling exactly len(buffer) items should work."""
        buf = ReplayBuffer(capacity=100, visible_size=4, action_dim=2, num_players=3)
        buf.add_stacked(*_stack_examples([_make_example(4, 2, 3) for _ in range(7)]))
        rng = np.random.default_rng(0)
        batch = buf.sample(batch_size=7, rng=rng)
        assert batch["states"].shape == (7, 4)

    def test_sample_only_from_filled(self) -> None:
        buf = ReplayBuffer(capacity=100, visible_size=4, action_dim=2, num_players=3)
        buf.add_stacked(*_stack_examples([_make_example(4, 2, 3) for _ in range(5)]))
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
        batch = _make_batch(4, _VIS, _ACT, 3)
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
        batch = _make_batch(4, _VIS, _ACT, 3)
        trainer.train_step(batch)
        assert trainer.global_step == 1
        trainer.train_step(batch)
        assert trainer.global_step == 2

    def test_lr_warmup_step_zero_nonzero(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        device = torch.device("cpu")
        tiny_config.warmup_steps = 10
        trainer = Trainer(small_model, tiny_config, device)
        batch = _make_batch(4, _VIS, _ACT, 3)
        trainer.train_step(batch)
        # First step LR must be non-zero (step+1)/warmup_steps = 1/10
        assert trainer.lr > 0, "LR at step 0 must not be zero"

    def test_lr_schedule_warmup_and_decay(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        device = torch.device("cpu")
        tiny_config.warmup_steps = 10
        tiny_config.training_steps_per_epoch = 100
        tiny_config.num_epochs = 10
        trainer = Trainer(small_model, tiny_config, device)
        batch = _make_batch(4, _VIS, _ACT, 3)
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

    def test_lr_decay_end_epoch(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        """LR reaches lr_min at decay end epoch, then stays constant."""
        device = torch.device("cpu")
        tiny_config.warmup_steps = 0
        tiny_config.training_steps_per_epoch = 10
        tiny_config.num_epochs = 100
        tiny_config.lr_decay_end_epoch = 5  # decay over 50 steps
        trainer = Trainer(small_model, tiny_config, device)
        batch = _make_batch(4, _VIS, _ACT, 3)
        lrs = []
        for _ in range(80):
            trainer.train_step(batch)
            lrs.append(trainer.lr)
        # Should decay over first 50 steps
        assert lrs[0] > lrs[49]
        # At step 50 should be at lr_min
        assert abs(lrs[49] - tiny_config.lr_min) < 1e-6
        # After decay end, LR stays constant at lr_min
        assert abs(lrs[79] - tiny_config.lr_min) < 1e-6
        assert abs(lrs[60] - tiny_config.lr_min) < 1e-6

    def test_parameters_stay_finite(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        device = torch.device("cpu")
        trainer = Trainer(small_model, tiny_config, device)
        batch = _make_batch(4, _VIS, _ACT, 3)
        trainer.train_step(batch)
        assert all(torch.isfinite(p).all() for p in small_model.parameters())

    def test_state_dict_roundtrip(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        device = torch.device("cpu")
        trainer = Trainer(small_model, tiny_config, device)
        batch = _make_batch(4, _VIS, _ACT, 3)
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
        batch = _make_batch(4, _VIS, _ACT, 3)
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
            input_dim=_VIS, action_dim=_ACT, value_dim=3,
            hidden_dim=32, num_blocks=1,
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

    def test_checkpoint_preserves_rng_state(
        self,
        small_model: RSSAlphaZeroNet,
        tiny_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """RNG state should round-trip through checkpoint save/load."""
        from train.main import _capture_rng_state, _restore_rng_state

        # Advance the RNG so it's not at initial state
        rng = np.random.default_rng(42)
        for _ in range(100):
            rng.integers(0, 2**31)

        # Capture state and generate reference values
        rng_state = _capture_rng_state(rng)
        expected = [int(rng.integers(0, 2**31)) for _ in range(10)]

        # Save checkpoint with RNG state
        cp_path = tmp_path / "rng_test.pt"
        save_checkpoint(
            cp_path,
            epoch=0,
            model=small_model,
            trainer_state=Trainer(small_model, tiny_config, torch.device("cpu")).state_dict(),
            config=tiny_config,
            metrics={},
            buffer_stats={"size": 0, "capacity": 1000},
            rng_state=rng_state,
        )

        # Load and restore into fresh RNG
        cp = load_checkpoint(cp_path, torch.device("cpu"))
        rng2 = np.random.default_rng(0)  # different seed
        _restore_rng_state(rng2, cp["rng_state"])  # type: ignore[arg-type]

        # Should produce identical sequence
        actual = [int(rng2.integers(0, 2**31)) for _ in range(10)]
        assert actual == expected

    def test_cleanup(self, tmp_path: Path) -> None:
        for epoch in range(1, 8):
            (tmp_path / f"checkpoint_epoch_{epoch:04d}.pt").touch()
        cleanup_checkpoints(tmp_path, keep_last_n=3)
        remaining = sorted(tmp_path.glob("checkpoint_epoch_*.pt"))
        assert len(remaining) == 3
        assert "0005" in remaining[0].name

    def test_resume_restores_config(
        self,
        small_model: RSSAlphaZeroNet,
        tmp_path: Path,
    ) -> None:
        """Resume should use checkpointed config, not fresh defaults."""
        device = torch.device("cpu")

        # Save checkpoint with non-default config values
        original_config = TrainingConfig(
            games_per_epoch=7,
            num_simulations=50,
            search_batch_size=4,
            learning_rate=5e-4,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            tensorboard_dir=str(tmp_path / "tb"),
        )
        trainer = Trainer(small_model, original_config, device)
        cp_path = tmp_path / "config_test.pt"
        save_checkpoint(
            cp_path,
            epoch=2,
            model=small_model,
            trainer_state=trainer.state_dict(),
            config=original_config,
            metrics={},
            buffer_stats={"size": 0, "capacity": 1000},
        )

        # Load checkpoint and restore config (simulating what main() does)
        cp = load_checkpoint(cp_path, device)
        restored = TrainingConfig.from_json(cp["config_json"])  # type: ignore[arg-type]

        # Verify non-default values were preserved
        assert restored.games_per_epoch == 7
        assert restored.num_simulations == 50
        assert restored.search_batch_size == 4
        assert restored.learning_rate == 5e-4

    def test_resume_overrides_all_cli_fields(self) -> None:
        """_apply_overrides should apply all CLI fields on resume."""
        from train.main import _apply_overrides

        config = TrainingConfig(
            games_per_epoch=7,
            num_simulations=50,
            num_workers=2,
            checkpoint_dir="original_dir",
            num_epochs=10,
        )

        args = argparse.Namespace(
            num_workers=8,
            checkpoint_dir="new_dir",
            tensorboard_dir="new_tb",
            num_epochs=20,
            games_per_epoch=500,
            num_simulations=400,
            search_batch_size=16,
            seed=123,
        )
        _apply_overrides(config, args, log_changes=True)

        # All overrides applied
        assert config.num_workers == 8
        assert config.checkpoint_dir == "new_dir"
        assert config.tensorboard_dir == "new_tb"
        assert config.num_epochs == 20
        assert config.games_per_epoch == 500
        assert config.num_simulations == 400
        assert config.search_batch_size == 16
        assert config.seed == 123

    def test_resume_overrides_only_specified(self) -> None:
        """_apply_overrides should not modify fields not specified on CLI."""
        from train.main import _apply_overrides

        config = TrainingConfig(
            games_per_epoch=7,
            num_simulations=50,
            num_workers=2,
        )

        # Only override num_workers, leave everything else as None
        args = argparse.Namespace(
            num_workers=8,
            checkpoint_dir=None,
            tensorboard_dir=None,
            num_epochs=None,
            games_per_epoch=None,
            num_simulations=None,
            search_batch_size=None,
            seed=None,
        )
        _apply_overrides(config, args, log_changes=True)

        assert config.num_workers == 8
        # Unspecified fields unchanged
        assert config.games_per_epoch == 7
        assert config.num_simulations == 50


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
        assert record.num_examples == record.total_moves
        assert len(record.net_worths) == tiny_config.num_players
        assert record.duration_secs > 0

        assert record.states.shape == (record.num_examples, tiny_config.visible_size)
        assert record.legal_masks.shape == (record.num_examples, tiny_config.action_dim)
        assert record.policy_targets.shape == (record.num_examples, tiny_config.action_dim)
        assert record.value_targets.shape == (record.num_examples, tiny_config.num_players)

        for i in range(record.num_examples):
            assert abs(record.policy_targets[i].sum() - 1.0) < 1e-5
            assert (record.policy_targets[i] >= 0).all()
            assert (record.value_targets[i] >= -1.0 - 1e-5).all()
            assert (record.value_targets[i] <= 1.0 + 1e-5).all()

    def test_value_blend_modifies_targets(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        """Value blending (blend_alpha < 1) should produce targets between pure A0GB and terminal."""
        from mcts.evaluator import NNEvaluator
        from train.config import EpochConfig

        device = torch.device("cpu")
        small_model.eval()
        evaluator = NNEvaluator(small_model, device, num_players=tiny_config.num_players)
        rng_seed = 42

        # Play same game twice: once without blending, once with
        pure_record = play_game(
            evaluator, tiny_config, game_seed=123,
            rng=np.random.default_rng(rng_seed),
            epoch_config=EpochConfig(c_puct=2.5, value_blend_alpha=1.0),
        )
        blended_record = play_game(
            evaluator, tiny_config, game_seed=123,
            rng=np.random.default_rng(rng_seed),
            epoch_config=EpochConfig(c_puct=2.5, value_blend_alpha=0.5),
        )

        # Same game, same moves → same number of examples
        assert pure_record.num_examples == blended_record.num_examples
        # States and policies should be identical (same game trajectory)
        np.testing.assert_array_equal(pure_record.states, blended_record.states)
        np.testing.assert_array_equal(pure_record.policy_targets, blended_record.policy_targets)
        # Value targets should differ due to blending
        assert not np.allclose(pure_record.value_targets, blended_record.value_targets), \
            "Blended values should differ from pure A0GB values"
        # Blended values should still be in valid range
        assert (blended_record.value_targets >= -1.0 - 1e-5).all()
        assert (blended_record.value_targets <= 1.0 + 1e-5).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_remote_evaluator_matches_local(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        """RemoteEvaluator through EvaluationServer produces same results as NNEvaluator."""
        import torch.multiprocessing as mp

        from mcts.evaluator import NNEvaluator

        from core.state import GameState
        from train.eval_server import EvaluationServer, RemoteEvaluator, SharedEvalBuffers

        device = torch.device("cuda")
        model = small_model.to(device)
        model.eval()
        num_players = tiny_config.num_players
        ctx = mp.get_context("spawn")

        # Set up a game state to evaluate
        state = GameState(num_players)
        state.initialize_game(seed=42)

        # Local evaluation
        local_eval = NNEvaluator(model, device, num_players=num_players)
        local_policy, local_values, local_mask = local_eval.evaluate(state)

        # Remote evaluation through server
        shared_bufs = SharedEvalBuffers(
            num_workers=1,
            batch_size=tiny_config.search_batch_size,
            visible_size=tiny_config.visible_size,
            action_dim=tiny_config.action_dim,
            num_players=num_players,
        )
        shared_bufs.init_bitmap([(0, 1)], ctx)
        server = EvaluationServer(
            model, device, shared_bufs,
            worker_start=0, worker_end=1,
            mp_context=ctx, no_compile=True,
        )
        server.start()
        try:
            remote_eval = RemoteEvaluator(
                num_players, shared_bufs, 0,
            )
            remote_policy, remote_values, remote_mask = remote_eval.evaluate(state)
        finally:
            server.stop()

        np.testing.assert_allclose(remote_policy, local_policy, atol=1e-6)
        np.testing.assert_allclose(remote_values, local_values, atol=1e-6)
        np.testing.assert_array_equal(remote_mask, local_mask)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_remote_evaluator_batch(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        """RemoteEvaluator.evaluate_batch matches NNEvaluator.evaluate_batch."""
        import torch.multiprocessing as mp

        from mcts.evaluator import NNEvaluator

        from core.driver import DRIVER
        from core.state import GameState
        from train.eval_server import (
            EvaluationServer,
            RemoteEvaluator,
            SharedEvalBuffers,
        )

        device = torch.device("cuda")
        model = small_model.to(device)
        model.eval()
        num_players = tiny_config.num_players
        ctx = mp.get_context("spawn")

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
        local_eval = NNEvaluator(model, device, num_players=num_players)
        local_results = local_eval.evaluate_batch(states)

        # Remote batch evaluation (batch_size must fit all states)
        shared_bufs = SharedEvalBuffers(
            num_workers=1,
            batch_size=max(tiny_config.search_batch_size, len(states)),
            visible_size=tiny_config.visible_size,
            action_dim=tiny_config.action_dim,
            num_players=num_players,
        )
        shared_bufs.init_bitmap([(0, 1)], ctx)
        server = EvaluationServer(
            model, device, shared_bufs,
            worker_start=0, worker_end=1,
            mp_context=ctx, no_compile=True,
        )
        server.start()
        try:
            remote_eval = RemoteEvaluator(
                num_players, shared_bufs, 0,
            )
            remote_results = remote_eval.evaluate_batch(states)
        finally:
            server.stop()

        assert len(remote_results) == len(local_results)
        for (rp, rv, rm), (lp, lv, lm) in zip(remote_results, local_results):
            np.testing.assert_allclose(rp, lp, atol=1e-6)
            np.testing.assert_allclose(rv, lv, atol=1e-6)
            np.testing.assert_array_equal(rm, lm)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_play_game_with_remote_evaluator(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        """play_game produces valid results when using RemoteEvaluator."""
        import torch.multiprocessing as mp

        from train.eval_server import (
            EvaluationServer,
            RemoteEvaluator,
            SharedEvalBuffers,
        )

        device = torch.device("cuda")
        model = small_model.to(device)
        model.eval()
        ctx = mp.get_context("spawn")

        shared_bufs = SharedEvalBuffers(
            num_workers=1,
            batch_size=tiny_config.search_batch_size,
            visible_size=tiny_config.visible_size,
            action_dim=tiny_config.action_dim,
            num_players=tiny_config.num_players,
        )
        shared_bufs.init_bitmap([(0, 1)], ctx)
        server = EvaluationServer(
            model, device, shared_bufs,
            worker_start=0, worker_end=1,
            mp_context=ctx, no_compile=True,
        )
        server.start()
        try:
            remote_eval = RemoteEvaluator(
                tiny_config.num_players, shared_bufs, 0,
            )
            rng = np.random.default_rng(42)
            record = play_game(remote_eval, tiny_config, game_seed=123, rng=rng)
        finally:
            server.stop()

        assert record.total_moves > 0
        assert record.num_examples == record.total_moves
        assert len(record.net_worths) == tiny_config.num_players
        for i in range(record.num_examples):
            assert abs(record.policy_targets[i].sum() - 1.0) < 1e-5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_multiprocess_workers(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        """End-to-end test: spawn actual worker processes, play games, collect results."""
        import torch.multiprocessing as mp

        from train.eval_server import EvaluationServer, SharedEvalBuffers
        from train.self_play import self_play_worker

        device = torch.device("cuda")
        model = small_model.to(device)
        model.eval()
        num_workers = 2
        games_per_worker = 1  # 2 games total

        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()

        shared_bufs = SharedEvalBuffers(
            num_workers=num_workers,
            batch_size=tiny_config.search_batch_size,
            visible_size=tiny_config.visible_size,
            action_dim=tiny_config.action_dim,
            num_players=tiny_config.num_players,
        )
        shared_bufs.init_bitmap([(0, num_workers)], ctx)

        server = EvaluationServer(
            model, device, shared_bufs,
            worker_start=0, worker_end=num_workers,
            mp_context=ctx, no_compile=True,
        )
        server.start()

        workers = []
        for i in range(num_workers):
            p = ctx.Process(
                target=self_play_worker,
                args=(
                    task_queue, result_queue, tiny_config,
                    shared_bufs, i,
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)

        try:
            # Feed game seeds (with epoch config)
            from train.config import EpochConfig
            epoch_cfg = EpochConfig(c_puct=2.5, value_blend_alpha=1.0)
            total_games = num_workers * games_per_worker
            for i in range(total_games):
                task_queue.put((42 + i, 100 + i, epoch_cfg))

            # Collect results
            records = []
            for _ in range(total_games):
                record = result_queue.get(timeout=120.0)
                records.append(record)

            assert len(records) == total_games
            for record in records:
                assert record.total_moves > 0
                assert record.num_examples == record.total_moves
                assert len(record.net_worths) == tiny_config.num_players
        finally:
            # Clean shutdown
            for _ in workers:
                task_queue.put(None)
            server.stop()
            for w in workers:
                w.join(timeout=5.0)
                if w.is_alive():
                    w.terminate()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_multi_server_eval(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        """Multiple EvaluationServer processes handle concurrent requests correctly."""
        import threading

        import torch.multiprocessing as mp

        from core.state import GameState
        from train.eval_server import (
            EvaluationServer,
            RemoteEvaluator,
            SharedEvalBuffers,
        )

        device = torch.device("cuda")
        model = small_model.to(device)
        model.eval()
        num_workers = 4
        num_servers = 2
        num_players = tiny_config.num_players
        num_rounds = 10
        ctx = mp.get_context("spawn")

        shared_bufs = SharedEvalBuffers(
            num_workers=num_workers,
            batch_size=tiny_config.search_batch_size,
            visible_size=tiny_config.visible_size,
            action_dim=tiny_config.action_dim,
            num_players=num_players,
        )
        workers_per_server = num_workers // num_servers
        partitions = [
            (i * workers_per_server, (i + 1) * workers_per_server)
            for i in range(num_servers)
        ]
        shared_bufs.init_bitmap(partitions, ctx)

        servers = []
        for i, (ws, we) in enumerate(partitions):
            server = EvaluationServer(
                model, device, shared_bufs,
                worker_start=ws, worker_end=we,
                server_id=i,
                mp_context=ctx, no_compile=True,
            )
            server.start()
            servers.append(server)

        try:
            evals = [
                RemoteEvaluator(
                    num_players, shared_bufs, i,
                )
                for i in range(num_workers)
            ]

            state = GameState(num_players)
            state.initialize_game(seed=42)

            # Get reference result from a single sequential call
            ref_policy, ref_values, ref_mask = evals[0].evaluate(state)

            # Fire concurrent requests from all workers across multiple rounds
            errors: list[Exception] = []
            results: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [
                [] for _ in range(num_workers)
            ]

            def _worker_fn(wid: int) -> None:
                try:
                    for _ in range(num_rounds):
                        r = evals[wid].evaluate(state)
                        results[wid].append(r)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=_worker_fn, args=(i,))
                for i in range(num_workers)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30.0)

            assert not errors, f"Worker threads raised: {errors}"

            # Every result from every worker must match the reference.
            # bf16 autocast with different batch compositions (reference=1
            # vs concurrent=2-4) causes minor precision differences from
            # Tensor Core accumulation patterns.
            for wid in range(num_workers):
                assert len(results[wid]) == num_rounds
                for rp, rv, rm in results[wid]:
                    np.testing.assert_allclose(rp, ref_policy, atol=0.01)
                    np.testing.assert_allclose(rv, ref_values, atol=0.01)
                    np.testing.assert_array_equal(rm, ref_mask)
        finally:
            for s in servers:
                s.stop()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_eval_server_sees_weight_updates_with_autocast(
        self, small_model: RSSAlphaZeroNet, tiny_config: TrainingConfig
    ) -> None:
        """Eval server with bfloat16 autocast sees in-place weight updates.

        Regression test for a bug where the persistent autocast context in
        the eval server cached bf16 weight casts, making optimizer.step()
        weight updates via CUDA IPC invisible for the entire training run.
        """
        import torch.multiprocessing as mp

        from core.state import GameState
        from train.eval_server import EvaluationServer, RemoteEvaluator, SharedEvalBuffers

        device = torch.device("cuda")
        model = small_model.to(device)
        model.eval()
        num_players = tiny_config.num_players
        ctx = mp.get_context("spawn")

        shared_bufs = SharedEvalBuffers(
            num_workers=1,
            batch_size=tiny_config.search_batch_size,
            visible_size=tiny_config.visible_size,
            action_dim=tiny_config.action_dim,
            num_players=num_players,
        )
        shared_bufs.init_bitmap([(0, 1)], ctx)
        server = EvaluationServer(
            model, device, shared_bufs,
            worker_start=0, worker_end=1,
            mp_context=ctx, no_compile=True,
            eval_dtype="bfloat16",
        )
        server.start()
        try:
            remote_eval = RemoteEvaluator(num_players, shared_bufs, 0)

            state = GameState(num_players)
            state.initialize_game(seed=42)

            # Get baseline output with original weights
            _, values_before, _ = remote_eval.evaluate(state)

            # Simulate optimizer.step(): zero out all weights in-place
            with torch.inference_mode():
                for p in model.parameters():
                    p.zero_()

            # Eval server must see the zeroed weights
            _, values_after, _ = remote_eval.evaluate(state)

            # With all-zero weights, the value head outputs tanh(0) = 0
            # for all players. If the autocast cache is stale, values_after
            # would still match values_before (non-zero).
            assert not np.allclose(values_before, values_after, atol=1e-3), (
                "Eval server did not see weight update — autocast cache is stale"
            )
            np.testing.assert_allclose(values_after, 0.0, atol=1e-3)
        finally:
            server.stop()


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
        buf.add_stacked(record.states, record.legal_masks, record.policy_targets, record.value_targets)
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

    def test_num_players_override_recomputes(self) -> None:
        config = TrainingConfig()  # defaults: num_players=3, model_path="auto" → "nn.model_3p"
        assert config.model_path == "nn.model_3p"
        assert config.action_dim == _ACT
        assert config.visible_size == _VIS

        parser = _build_parser()
        args = parser.parse_args(["--num-players", "4"])
        _apply_overrides(config, args)
        config.validate()

        assert config.num_players == 4
        assert config.model_path == "nn.model_4p"
        assert config.action_dim == _ACT4
        assert config.visible_size == _VIS4

    def test_num_players_override_with_explicit_model_path(self) -> None:
        config = TrainingConfig()
        parser = _build_parser()
        args = parser.parse_args([
            "--num-players", "4",
            "--model-path", "nn.model_custom",
        ])
        _apply_overrides(config, args)
        config.validate()

        assert config.num_players == 4
        assert config.model_path == "nn.model_custom"
        assert config.action_dim == _ACT4


# ---------------------------------------------------------------------------
# Model Creation Tests
# ---------------------------------------------------------------------------


class TestModelCreation:
    """Verify create_model produces working models for different player counts."""

    @pytest.mark.parametrize("num_players,vis,act", [
        (3, _VIS, _ACT),
        (4, _VIS4, _ACT4),
    ])
    def test_create_model(self, num_players: int, vis: int, act: int) -> None:
        from nn import create_model
        model = create_model(
            f"nn.model_{num_players}p",
            input_dim=vis,
            action_dim=act,
            value_dim=num_players,
        )
        # Smoke test: forward pass with phase-aware input
        x = torch.randn(2, vis)
        x[:, :8] = 0
        x[0, 0] = 1.0  # INVEST phase
        x[1, 1] = 1.0  # BID phase
        policy, values = model(x)
        assert policy.shape == (2, act)
        assert values.shape == (2, num_players)
        assert values.min() >= -1.0 and values.max() <= 1.0


class TestCompileConfig:
    def test_gpu_compile_kwargs_do_not_force_dynamic_shapes(self) -> None:
        assert get_nvidia_compile_kwargs(for_training=False) == {"mode": "reduce-overhead"}
        assert get_nvidia_compile_kwargs(for_training=True) == {"mode": "reduce-overhead"}
        assert get_amd_compile_kwargs(for_training=False) == {"mode": "reduce-overhead"}
        assert get_amd_compile_kwargs(for_training=True) == {"mode": "reduce-overhead"}


class TestModelForward:
    def test_forward_does_not_call_tensor_tolist(
        self, small_model: RSSAlphaZeroNet
    ) -> None:
        x = torch.randn(8, small_model.cfg.input_dim)
        x[:, :8] = 0.0
        for i in range(8):
            x[i, i] = 1.0

        original_tolist = torch.Tensor.tolist

        def _forbid_tolist(self: torch.Tensor) -> list[object]:
            raise AssertionError("model forward should not call Tensor.tolist()")

        try:
            torch.Tensor.tolist = _forbid_tolist  # type: ignore[method-assign]
            policy_logits, values = small_model(x)
        finally:
            torch.Tensor.tolist = original_tolist  # type: ignore[method-assign]

        assert policy_logits.shape == (8, small_model.cfg.action_dim)
        assert values.shape == (8, small_model.cfg.value_dim)

    def test_forward_plus_does_not_call_tensor_tolist(
        self, small_model_plus: RSSAlphaZeroNetPlus
    ) -> None:
        x = torch.randn(8, small_model_plus.cfg.input_dim)
        x[:, :8] = 0.0
        for i in range(8):
            x[i, i] = 1.0

        original_tolist = torch.Tensor.tolist

        def _forbid_tolist(self: torch.Tensor) -> list[object]:
            raise AssertionError("model forward should not call Tensor.tolist()")

        try:
            torch.Tensor.tolist = _forbid_tolist  # type: ignore[method-assign]
            policy_logits, values = small_model_plus(x)
        finally:
            torch.Tensor.tolist = original_tolist  # type: ignore[method-assign]

        assert policy_logits.shape == (8, small_model_plus.cfg.action_dim)
        assert values.shape == (8, small_model_plus.cfg.value_dim)

    def test_forward_4p_does_not_call_tensor_tolist(
        self, small_model_4p: RSSAlphaZeroNet4P
    ) -> None:
        x = torch.randn(8, small_model_4p.cfg.input_dim)
        x[:, :8] = 0.0
        for i in range(8):
            x[i, i] = 1.0

        original_tolist = torch.Tensor.tolist

        def _forbid_tolist(self: torch.Tensor) -> list[object]:
            raise AssertionError("model forward should not call Tensor.tolist()")

        try:
            torch.Tensor.tolist = _forbid_tolist  # type: ignore[method-assign]
            policy_logits, values = small_model_4p(x)
        finally:
            torch.Tensor.tolist = original_tolist  # type: ignore[method-assign]

        assert policy_logits.shape == (8, small_model_4p.cfg.action_dim)
        assert values.shape == (8, small_model_4p.cfg.value_dim)

    def test_grouped_forward_all_phases(
        self, small_model: RSSAlphaZeroNet
    ) -> None:
        """Grouped bmm forward handles all 8 phases in one batch."""
        x = torch.randn(16, small_model.cfg.input_dim)
        x[:, :8] = 0.0
        for i in range(16):
            x[i, i % 8] = 1.0

        policy_logits, values = small_model(x)
        assert policy_logits.shape == (16, small_model.cfg.action_dim)
        assert values.shape == (16, small_model.cfg.value_dim)
        assert torch.isfinite(policy_logits).all()


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
