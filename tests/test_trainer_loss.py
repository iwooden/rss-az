from __future__ import annotations

import numpy as np
import pytest
import torch

from core.attention_relations import NUM_ATTENTION_RELATIONS
from core.state import GameState, get_layout
from core.token_data import TokenDataSize, get_num_tokens
from entities.turn import TURN
from nn.policy_layout import PHASES_WITH_PASS_SLOT, UNIFIED_LOGIT_DIM
from train.config import TrainingConfig
from train.replay_buffer import ReplayBuffer
from train.trainer import Trainer, average_training_metrics

NUM_PLAYERS = 3


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_phase_policy_entropy_kl_counts_and_gradients(device):
    from nn.policy_layout import PHASE_OFFSETS

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    class FixedPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(int(UNIFIED_LOGIT_DIM)))
            with torch.no_grad():
                for phase, probabilities in ((0, [.98, .02]), (1, [.6, .4])):
                    start = PHASE_OFFSETS[phase]
                    self.logits[start:start + 2] = torch.tensor(probabilities).log()

        def forward(self, tokens, masks, relations):
            return self.logits.expand(len(tokens), -1).masked_fill(~masks, -1e9), torch.zeros(
                len(tokens), 3, device=tokens.device,
            )

    config = TrainingConfig(num_players=3, optimizer="adamw", value_loss_weight=0, grad_clip=0)
    model = FixedPolicy().to(device)
    trainer = Trainer(model, config, torch.device(device))
    buffer = ReplayBuffer(3, get_layout(3).total_size, 3)
    for phase in (0, 0, 1):
        mask = np.zeros(int(UNIFIED_LOGIT_DIM), dtype=np.uint8)
        target = np.zeros(int(UNIFIED_LOGIT_DIM), dtype=np.float32)
        start = PHASE_OFFSETS[phase]
        mask[start:start + 2] = 1
        target[start:start + 2] = [.8, .2]
        buffer.add_stacked(
            states=_make_initialized_state(3)[None], phase_ids=np.array([phase], dtype=np.int8),
            legal_masks=mask[None], policy_targets=target[None],
            value_targets=np.zeros((1, 3), dtype=np.float32),
        )
    result = trainer.train_step(buffer, 3, np.random.default_rng(1))
    target = np.array([.8, .2])
    entropy = -(target * np.log(target)).sum()
    for phase, name, p, count in ((0, "invest", [.98, .02], 2), (1, "bid", [.6, .4], 1)):
        assert result[f"policy_samples_{name}"] == count
        assert result[f"policy_target_entropy_{name}"] == pytest.approx(entropy)
        assert result[f"policy_kl_{name}"] == pytest.approx((target * np.log(target / p)).sum())
        start = PHASE_OFFSETS[phase]
        # Additional diagnostics must not contribute to the policy gradient.
        assert model.logits.grad is not None
        np.testing.assert_allclose(
            model.logits.grad[start:start + 2].cpu().numpy(), (np.array(p) - target) * count / 3,
            atol=1e-7,
        )
    assert "policy_samples_ipo" not in result


def test_phase_epoch_means_weight_samples_not_steps():
    metrics = {
        "total_loss": [1., 2.],
        "policy_samples_invest": [9., 1.],
        "policy_loss_invest": [1., 2.],
        "policy_target_entropy_invest": [.7, .8],
        "policy_kl_invest": [.3, 1.2],
        "policy_samples_bid": [2.],
        "policy_loss_bid": [1.],
        "policy_target_entropy_bid": [.8],
        "policy_kl_bid": [.2],
    }
    result = average_training_metrics(metrics)
    assert result["total_loss"] == 1.5
    assert result["policy_samples_invest"] == 10
    assert result["policy_loss_invest"] == pytest.approx(1.1)
    assert result["policy_target_entropy_invest"] == pytest.approx(.71)
    assert result["policy_kl_invest"] == pytest.approx(.39)
    assert result["policy_kl_bid"] == .2


class ConstantValueModel(torch.nn.Module):
    def __init__(self, num_players: int) -> None:
        super().__init__()
        self.value_bias = torch.nn.Parameter(torch.zeros(num_players))
        self.last_tokens_shape: tuple[int, ...] | None = None
        self.last_tokens_dtype: torch.dtype | None = None

    def forward(
        self,
        tokens: torch.Tensor,
        _legal_masks: torch.Tensor,
        relations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_tokens_shape = tuple(tokens.shape)
        self.last_tokens_dtype = tokens.dtype
        self.last_relations_shape = tuple(relations.shape)
        self.last_relations_dtype = relations.dtype
        batch = tokens.shape[0]
        policy_logits = torch.zeros(
            (batch, int(UNIFIED_LOGIT_DIM)), dtype=tokens.dtype, device=tokens.device
        )
        values = self.value_bias.unsqueeze(0).expand(batch, -1)
        return policy_logits, values

    def pass_action_logit_abs(
        self,
        policy_logits: torch.Tensor,
        _legal_mask: torch.Tensor,
        _phase_ids: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros(
            2 * len(PHASES_WITH_PASS_SLOT),
            dtype=policy_logits.dtype,
            device=policy_logits.device,
        )


def _make_initialized_state(
    num_players: int,
    *,
    active_player: int = 0,
    max_players: int | None = None,
) -> np.ndarray:
    if max_players is None:
        state = GameState(num_players)
        state.initialize_game(num_players, seed=0)
    else:
        state = GameState(num_players, max_players=max_players)
        state.initialize_game(num_players, seed=0, max_players=max_players)
    TURN.set_active_player(state, active_player)
    return state._array.copy()


def _one_hot_policy_targets() -> tuple[np.ndarray, np.ndarray]:
    legal_mask = np.zeros(int(UNIFIED_LOGIT_DIM), dtype=np.uint8)
    legal_mask[0] = 1
    policy_target = np.zeros(int(UNIFIED_LOGIT_DIM), dtype=np.float32)
    policy_target[0] = 1.0
    return legal_mask, policy_target


def _mixed_training_config() -> TrainingConfig:
    return TrainingConfig(
        num_players=0,
        min_players=3,
        max_players=5,
        optimizer="adamw",
        batch_size=1,
        num_epochs=1,
        training_steps_per_epoch=1,
        warmup_epochs=0,
        learning_rate=1e-3,
        weight_decay=0.0,
        policy_loss_weight=0.0,
        value_loss_weight=1.0,
    )


def test_train_step_value_loss_uses_mean_over_player_dimension() -> None:
    cfg = TrainingConfig(
        num_players=NUM_PLAYERS,
        optimizer="adamw",
        batch_size=1,
        num_epochs=1,
        training_steps_per_epoch=1,
        warmup_epochs=0,
        learning_rate=1e-3,
        policy_loss_weight=0.0,
        value_loss_weight=1.0,
    )
    model = ConstantValueModel(NUM_PLAYERS)
    trainer = Trainer(model, cfg, torch.device("cpu"))

    layout = get_layout(NUM_PLAYERS)
    buffer = ReplayBuffer(1, layout.total_size, NUM_PLAYERS)

    state = _make_initialized_state(NUM_PLAYERS)
    legal_mask, policy_target = _one_hot_policy_targets()
    value_target = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    buffer.add_stacked(
        states=state[None, :],
        phase_ids=np.array([0], dtype=np.int8),
        legal_masks=legal_mask[None, :],
        policy_targets=policy_target[None, :],
        value_targets=value_target[None, :],
    )

    losses = trainer.train_step(buffer, batch_size=1, rng=np.random.default_rng(0))

    assert losses["value_loss"] == pytest.approx(1.0 / NUM_PLAYERS)
    assert losses["total_loss"] == pytest.approx(1.0 / NUM_PLAYERS)
    assert model.last_relations_shape == (
        1,
        NUM_ATTENTION_RELATIONS,
        get_num_tokens(NUM_PLAYERS),
        get_num_tokens(NUM_PLAYERS),
    )
    assert model.last_relations_dtype == torch.uint8


def test_train_step_value_loss_ignores_padded_player_slots() -> None:
    cfg = _mixed_training_config()
    model = ConstantValueModel(5)
    with torch.no_grad():
        model.value_bias.copy_(torch.tensor([0.0, 0.0, 0.0, 100.0, 100.0]))
    trainer = Trainer(model, cfg, torch.device("cpu"))

    buffer = ReplayBuffer(
        1,
        get_layout(5).total_size,
        5,
        min_players=3,
        max_players=5,
    )

    state = _make_initialized_state(3, max_players=5)
    legal_mask, policy_target = _one_hot_policy_targets()
    value_target = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    buffer.add_stacked(
        states=state[None, :],
        phase_ids=np.array([0], dtype=np.int8),
        legal_masks=legal_mask[None, :],
        policy_targets=policy_target[None, :],
        value_targets=value_target[None, :],
        num_players=3,
    )

    losses = trainer.train_step(buffer, batch_size=1, rng=np.random.default_rng(0))

    assert losses["value_loss"] == pytest.approx(1.0 / 3.0)
    assert losses["value_loss_3p"] == pytest.approx(1.0 / 3.0)
    assert losses["policy_loss_3p"] == pytest.approx(np.log(int(UNIFIED_LOGIT_DIM)))
    assert "value_loss_4p" not in losses
    assert losses["total_loss"] == pytest.approx(1.0 / 3.0)
    torch.testing.assert_close(
        model.value_bias.detach()[3:],
        torch.tensor([100.0, 100.0]),
    )


def test_train_step_mixed_batch_uses_masked_value_mse_and_max_width_inputs() -> None:
    cfg = _mixed_training_config()
    cfg.batch_size = 2
    model = ConstantValueModel(5)
    trainer = Trainer(model, cfg, torch.device("cpu"))

    buffer = ReplayBuffer(
        2,
        get_layout(5).total_size,
        5,
        min_players=3,
        max_players=5,
    )
    legal_mask, policy_target = _one_hot_policy_targets()
    rows = [
        (3, np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        (5, np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)),
    ]
    for num_players, value_target in rows:
        buffer.add_stacked(
            states=_make_initialized_state(num_players, max_players=5)[None, :],
            phase_ids=np.array([0], dtype=np.int8),
            legal_masks=legal_mask[None, :],
            policy_targets=policy_target[None, :],
            value_targets=value_target[None, :],
            num_players=num_players,
        )

    losses = trainer.train_step(buffer, batch_size=2, rng=np.random.default_rng(0))

    expected_sqerr_sum = 1.0 + 4.0 + 9.0 + 1.0 + 4.0 + 9.0 + 16.0 + 25.0
    assert losses["value_loss"] == pytest.approx(expected_sqerr_sum / 8.0)
    assert losses["value_loss_3p"] == pytest.approx((1.0 + 4.0 + 9.0) / 3.0)
    assert losses["value_loss_5p"] == pytest.approx(
        (1.0 + 4.0 + 9.0 + 16.0 + 25.0) / 5.0
    )
    assert losses["policy_loss_3p"] == pytest.approx(np.log(int(UNIFIED_LOGIT_DIM)))
    assert losses["policy_loss_5p"] == pytest.approx(np.log(int(UNIFIED_LOGIT_DIM)))
    assert losses["total_loss"] == pytest.approx(expected_sqerr_sum / 8.0)
    assert model.last_tokens_shape == (
        2,
        get_num_tokens(5),
        int(TokenDataSize.TOKEN_DIM),
    )
    assert model.last_tokens_dtype == torch.float32
    assert model.last_relations_shape == (
        2,
        NUM_ATTENTION_RELATIONS,
        get_num_tokens(5),
        get_num_tokens(5),
    )
    assert model.last_relations_dtype == torch.uint8
