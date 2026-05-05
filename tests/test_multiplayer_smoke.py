from __future__ import annotations

import multiprocessing as mp
from types import SimpleNamespace

import numpy as np
import torch

from core.state import get_layout
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool
from nn import get_model_input_spec
from nn.transformer import UNIFIED_LOGIT_DIM
from train.config import TrainingConfig
from train.eval_server import EvaluationServer, SharedEvalBuffers
from train.main import (
    _SelfPlayMetricAccumulator,
    _build_epoch_self_play_scalars,
    _scaled_training_steps,
)
from train.replay_buffer import ReplayBuffer
from train.self_play import play_game, self_play_worker
from train.trainer import Trainer


MAX_PLAYERS = 5
U_DIM = int(UNIFIED_LOGIT_DIM)


class SmokeTransformerModel(torch.nn.Module):
    """Small transformer-contract model for mixed player-count smoke tests."""

    def __init__(self, num_players: int = MAX_PLAYERS) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(num_players=num_players)
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.values = torch.nn.Parameter(torch.zeros(num_players))
        self.register_buffer(
            "_slot_ids",
            torch.arange(U_DIM, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        legal_mask: torch.Tensor,
        relations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del tokens, relations
        legal = legal_mask.to(torch.bool)
        logits = self.logit_scale * self._slot_ids.to(legal.device)
        logits = logits.expand(legal.shape[0], -1).clone()
        logits = logits.masked_fill(~legal, -1e9)
        values = torch.tanh(self.values).expand(legal.shape[0], -1)
        return logits, values

    def phase_mod_diagnostics(self) -> dict[str, float]:
        return {}


def _smoke_config(*, num_workers: int) -> TrainingConfig:
    return TrainingConfig(
        num_players=0,
        min_players=3,
        max_players=MAX_PLAYERS,
        games_per_epoch=3,
        num_epochs=1,
        num_simulations=2,
        search_batch_size=2,
        num_workers=num_workers,
        num_eval_servers=1,
        dirichlet_epsilon=0.0,
        buffer_capacity=512,
        min_buffer_size=4,
        batch_size=4,
        training_steps_per_epoch=1,
        optimizer="adamw",
        value_blend_start_epoch=0,
        value_blend_end_epoch=0,
        seed=123,
    )


def _replay_buffer_for(config: TrainingConfig) -> ReplayBuffer:
    max_players = config.effective_max_players
    return ReplayBuffer(
        config.buffer_capacity,
        get_layout(max_players).total_size,
        max_players,
        min_players=config.effective_min_players,
        max_players=max_players,
    )


def test_local_mixed_mini_epoch_runs_one_game_per_count_and_trains() -> None:
    config = _smoke_config(num_workers=0)
    device = torch.device("cpu")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    model = SmokeTransformerModel().to(device)
    input_spec = get_model_input_spec(config)
    evaluator = NNEvaluator(
        model,
        device,
        num_players=MAX_PLAYERS,
        input_spec=input_spec,
        terminal_rank_weight=config.terminal_blend,
    )
    state_pool = StatePool(
        2 * (config.max_simulations + 1),
        get_layout(MAX_PLAYERS).total_size,
    )
    buffer = _replay_buffer_for(config)
    metrics = _SelfPlayMetricAccumulator()
    epoch_config = config.compute_epoch_config(0)

    records = [
        play_game(
            evaluator,
            config,
            game_seed=seed,
            rng=np.random.default_rng(10_000 + seed),
            state_pool=state_pool,
            epoch_config=epoch_config,
        )
        for seed in (0, 1, 2)
    ]

    assert [record.num_players for record in records] == [3, 4, 5]
    for record in records:
        assert record.states.shape[1] == get_layout(MAX_PLAYERS).total_size
        assert record.value_targets.shape == (
            record.num_examples,
            record.num_players,
        )
        assert record.legal_masks.shape == (record.num_examples, U_DIM)
        assert record.policy_targets.shape == (record.num_examples, U_DIM)
        buffer.add_stacked(
            record.states,
            record.phase_ids,
            record.legal_masks,
            record.policy_targets,
            record.value_targets,
            num_players=record.num_players,
        )
        metrics.add_record(record)

    assert len(buffer) >= config.min_buffer_size
    assert set(metrics.count_snapshots()) == {3, 4, 5}
    scalars = _build_epoch_self_play_scalars(metrics)
    assert "self_play_aggregate/game_length_mean" in scalars
    assert "self_play_3p/game_length_mean" in scalars
    assert "self_play_4p/game_length_mean" in scalars
    assert "self_play_5p/game_length_mean" in scalars

    trainer = Trainer(model, config, device)
    losses: dict[str, list[float]] = {}
    for _ in range(_scaled_training_steps(config, len(buffer))):
        step_losses = trainer.train_step(buffer, config.batch_size, rng)
        for key, value in step_losses.items():
            losses.setdefault(key, []).append(value)

    assert trainer.global_step == 1
    assert losses
    for values in losses.values():
        assert all(np.isfinite(value) for value in values)


def test_eval_server_worker_runs_mixed_player_count_games() -> None:
    config = _smoke_config(num_workers=1)
    ctx = mp.get_context("spawn")
    torch.manual_seed(1)
    model = SmokeTransformerModel()
    shared_bufs = SharedEvalBuffers(
        num_workers=config.num_workers,
        batch_size=config.search_batch_size,
        num_players=MAX_PLAYERS,
        input_spec=get_model_input_spec(config),
    )
    shared_bufs.init_bitmap([(0, 1)], ctx)
    server = EvaluationServer(
        model,
        torch.device("cpu"),
        shared_bufs,
        mp_context=ctx,
        no_compile=True,
    )
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    worker = ctx.Process(
        target=self_play_worker,
        args=(task_queue, result_queue, config, shared_bufs, 0),
        daemon=True,
    )

    records = []
    try:
        server.start()
        assert server.wait_ready(timeout=15.0)
        worker.start()
        epoch_config = config.compute_epoch_config(0)
        for seed in (0, 1, 2):
            task_queue.put((seed, 20_000 + seed, epoch_config))
        for _ in range(3):
            records.append(result_queue.get(timeout=60.0))
        task_queue.put(None)
        worker.join(timeout=10.0)
        assert not worker.is_alive()
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5.0)
        server.stop()

    assert [record.num_players for record in records] == [3, 4, 5]
    for record in records:
        assert record.num_examples > 0
        assert record.states.shape[1] == get_layout(MAX_PLAYERS).total_size
        assert record.value_targets.shape == (
            record.num_examples,
            record.num_players,
        )
        assert np.isfinite(record.policy_targets).all()
        assert np.isfinite(record.value_targets).all()
