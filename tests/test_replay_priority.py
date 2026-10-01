"""Prioritized replay: sampling mixture, importance weights, persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from core.state import GameState, get_layout
from nn.policy_layout import PHASE_OFFSETS, UNIFIED_LOGIT_DIM
from train.config import TrainingConfig
from train.policy_metrics import kl_to_prior
from train.replay_buffer import ReplayBuffer
from train.trainer import Trainer

U_DIM = int(UNIFIED_LOGIT_DIM)


def _buffer(capacity: int, **kwargs) -> ReplayBuffer:
    return ReplayBuffer(capacity, get_layout(3).total_size, 3, **kwargs)


def _add_rows(
    buffer: ReplayBuffer,
    ids: list[int],
    priorities: list[float] | None = None,
    *,
    policy_targets: np.ndarray | None = None,
    value_targets: np.ndarray | None = None,
) -> None:
    """Add 3p rows whose first value target is the row id unless given."""
    n = len(ids)
    state = GameState(3)
    state.initialize_game(3, seed=0)
    masks = np.zeros((n, U_DIM), dtype=np.uint8)
    masks[:, :2] = 1
    if policy_targets is None:
        policy_targets = np.zeros((n, U_DIM), dtype=np.float32)
        policy_targets[:, 0] = 1.0
    if value_targets is None:
        value_targets = np.zeros((n, 3), dtype=np.float32)
        value_targets[:, 0] = ids
    buffer.add_stacked(
        states=np.repeat(state._array[None], n, axis=0),
        phase_ids=np.zeros(n, dtype=np.int8),
        legal_masks=masks,
        policy_targets=policy_targets,
        value_targets=value_targets,
        priorities=None if priorities is None else np.array(priorities, dtype=np.float32),
    )


def _draws(buffer: ReplayBuffer, batches: int, batch_size: int, seed: int = 0):
    """Return sampled row ids and importance weights, flattened."""
    rng = np.random.default_rng(seed)
    ids, weights = [], []
    for _ in range(batches):
        batch = buffer.sample(batch_size, rng)
        ids.append(batch["value_targets"][:, 0].numpy().astype(int))
        weights.append(batch["is_weights"].numpy())
    return np.concatenate(ids), np.concatenate(weights)


def test_default_sampling_is_unchanged_uniform_without_replacement():
    buffer = _buffer(10)
    _add_rows(buffer, list(range(10)), [5.0] * 9 + [0.0])
    ids, weights = _draws(buffer, 1, 4, seed=5)
    np.testing.assert_array_equal(ids, np.random.default_rng(5).choice(10, 4, replace=False))
    np.testing.assert_array_equal(weights, np.ones(4))


def test_mixture_frequencies_and_weights_estimate_uniform_means():
    # Exponent 0.5 turns KL [0, 1, 4, 9] into weights [0, 1, 2, 3]. With half
    # of each 4-row batch prioritized, row i appears 0.5 + w_i / 3 times per
    # batch, and its importance weight is uniform / mixture probability.
    buffer = _buffer(4, priority_fraction=0.5, priority_exponent=0.5)
    _add_rows(buffer, [0, 1, 2, 3], [0.0, 1.0, 4.0, 9.0])
    batches = 20_000
    ids, weights = _draws(buffer, batches, 4)

    counts = np.bincount(ids, minlength=4) / batches
    np.testing.assert_allclose(counts, [0.5, 5 / 6, 7 / 6, 1.5], rtol=0.03)
    expected_weight = 1.0 / (0.5 + 0.5 * np.array([0.0, 2 / 3, 4 / 3, 2.0]))
    np.testing.assert_allclose(weights, expected_weight[ids], rtol=1e-6)
    # Weighted averages of a per-row quantity (the id) recover its mean, 1.5.
    assert np.mean(weights * ids) == pytest.approx(1.5, rel=0.02)


def test_rows_without_priority_sample_at_the_average_weight():
    buffer = _buffer(4, priority_fraction=0.5, priority_exponent=1.0)
    _add_rows(buffer, [0], None)
    _add_rows(buffer, [1, 2, 3], [1.0, 1.0, 4.0])
    batches = 20_000
    ids, _ = _draws(buffer, batches, 4)
    # Weights [2, 1, 1, 4] (unknown = mean of known), so counts 0.5 + w / 4.
    counts = np.bincount(ids, minlength=4) / batches
    np.testing.assert_allclose(counts, [1.0, 0.75, 0.75, 1.5], rtol=0.03)
    assert buffer.priority_stats()["priority_unknown_fraction"] == 0.25


def test_all_zero_priorities_fall_back_to_uniform():
    buffer = _buffer(4, priority_fraction=0.5)
    _add_rows(buffer, [0, 1, 2, 3], [0.0] * 4)
    _, weights = _draws(buffer, 10, 4)
    np.testing.assert_allclose(weights, 1.0)


def _wrapped_buffer() -> ReplayBuffer:
    # Second add wraps: slot 3 <- id 3, slots 0-1 <- ids 4-5. Only id 2 has
    # zero priority, so only it has weight 1 / (1 - f) = 2.
    buffer = _buffer(4, priority_fraction=0.5, priority_exponent=1.0)
    _add_rows(buffer, [0, 1, 2], [0.0, 0.0, 0.0])
    _add_rows(buffer, [3, 4, 5], [9.0, 9.0, 9.0])
    return buffer


def _assert_wrapped_weights(buffer: ReplayBuffer) -> None:
    ids, weights = _draws(buffer, 200, 4)
    assert set(ids.tolist()) == {2, 3, 4, 5}
    np.testing.assert_allclose(weights[ids == 2], 2.0)
    np.testing.assert_allclose(weights[ids != 2], 1.0 / (0.5 + 0.5 * 4 / 3), rtol=1e-6)


def test_ring_wrap_keeps_priorities_with_their_rows():
    _assert_wrapped_weights(_wrapped_buffer())


def test_save_load_round_trips_priorities_and_old_buffers_load_unknown(tmp_path: Path):
    _wrapped_buffer().save(tmp_path)
    loaded = _buffer(4, priority_fraction=0.5, priority_exponent=1.0)
    assert loaded.load(tmp_path) == 4
    _assert_wrapped_weights(loaded)

    (tmp_path / "priorities.npy").unlink()
    old = _buffer(4, priority_fraction=0.5, priority_exponent=1.0)
    assert old.load(tmp_path) == 4
    assert old.priority_stats()["priority_unknown_fraction"] == 1.0
    _, weights = _draws(old, 10, 4)
    np.testing.assert_allclose(weights, 1.0)


def test_priority_stats():
    buffer = _buffer(4, priority_fraction=0.5, priority_exponent=0.5)
    _add_rows(buffer, [0, 1, 2, 3], [0.0, 1.0, 4.0, 9.0])
    rate = np.array([0.5, 5 / 6, 7 / 6, 1.5])
    assert buffer.priority_stats() == pytest.approx({
        "priority_unknown_fraction": 0.0,
        "priority_kl_mean": 3.5,
        "priority_max_rate": 1.5,
        "priority_ess_fraction": 1.0 / np.mean(rate**2),
    })
    uniform = _buffer(4)
    _add_rows(uniform, [0, 1, 2, 3], [0.0, 1.0, 4.0, 9.0])
    assert set(uniform.priority_stats()) == {"priority_unknown_fraction", "priority_kl_mean"}


@pytest.mark.parametrize("fraction", [0.0, 0.5])
def test_trainer_reports_uniform_estimates_and_trains_policy_on_priority_mix(fraction):
    """Reported losses track uniform sampling; the policy trains on the mix.

    The model's fixed prior is [.98, .02]. Six of eight rows agree with it
    (KL 0, value error 0); two overturn it (target [.2, .8], value target
    [1, 0, 0]). With zero loss weights and no weight decay, nothing changes
    between steps, so step averages converge to batch expectations.
    """

    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(U_DIM))
            with torch.no_grad():
                start = PHASE_OFFSETS[0]
                self.logits[start:start + 2] = torch.tensor([.98, .02]).log()

        def forward(self, tokens, masks, relations):
            logits = self.logits.expand(len(tokens), -1).masked_fill(~masks, -1e9)
            return logits, torch.zeros(len(tokens), 3, device=tokens.device)

    config = TrainingConfig(
        num_players=3, optimizer="adamw", weight_decay=0.0, grad_clip=0,
        policy_loss_weight=0.0, value_loss_weight=0.0,
        replay_priority_fraction=fraction, replay_priority_exponent=1.0,
    )
    trainer = Trainer(FixedModel(), config, torch.device("cpu"))
    buffer = _buffer(8, priority_fraction=fraction, priority_exponent=1.0)
    prior = np.array([.98, .02])
    targets = np.zeros((8, U_DIM), dtype=np.float32)
    start = PHASE_OFFSETS[0]
    targets[:6, start:start + 2] = prior
    targets[6:, start:start + 2] = [.2, .8]
    values = np.zeros((8, 3), dtype=np.float32)
    values[6:, 0] = 1.0
    kl = [kl_to_prior(row[start:start + 2], prior) for row in targets]
    _add_rows(buffer, list(range(8)), kl, policy_targets=targets, value_targets=values)
    # _add_rows marks the first two slots legal; move them to phase 0's slots.
    buffer._legal_masks[:8] = 0
    buffer._legal_masks[:8, start:start + 2] = 1

    rng = np.random.default_rng(0)
    steps = [trainer.train_step(buffer, 8, rng) for _ in range(400)]

    def mean(key: str) -> float:
        return float(np.mean([step[key] for step in steps]))

    cross_entropy = -(targets[:, start:start + 2] * np.log(prior)).sum(axis=1)
    assert mean("policy_loss") == pytest.approx(cross_entropy.mean(), rel=0.05)
    assert mean("value_loss") == pytest.approx(2 / 24, rel=0.1)
    if fraction == 0.0:
        assert "policy_kl_sampled" not in steps[0]
    else:
        # Half of each batch is prioritized and only overturned rows (a quarter
        # of the buffer) have KL, so the trained KL is (0.5 * 0.25 + 0.5) / 0.25
        # = 2.5x the uniform KL.
        assert mean("policy_kl_sampled") == pytest.approx(2.5 * mean("policy_kl"), rel=0.1)
