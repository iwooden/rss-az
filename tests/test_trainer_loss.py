from __future__ import annotations

import numpy as np
import pytest
import torch

from core.state import GameState, get_layout
from nn.transformer import UNIFIED_LOGIT_DIM
from train.config import TrainingConfig
from train.replay_buffer import ReplayBuffer
from train.trainer import Trainer

NUM_PLAYERS = 3


class ConstantValueModel(torch.nn.Module):
    def __init__(self, num_players: int) -> None:
        super().__init__()
        self.value_bias = torch.nn.Parameter(torch.zeros(num_players))

    def forward(
        self, tokens: torch.Tensor, legal_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = tokens.shape[0]
        policy_logits = torch.zeros(
            (batch, int(UNIFIED_LOGIT_DIM)), dtype=tokens.dtype, device=tokens.device
        )
        values = self.value_bias.unsqueeze(0).expand(batch, -1)
        return policy_logits, values


def _make_initialized_state(num_players: int) -> np.ndarray:
    state = GameState(num_players)
    state.initialize_game(num_players, seed=0)
    return state._array.copy()


def test_train_step_value_loss_uses_mean_over_player_dimension() -> None:
    cfg = TrainingConfig(
        num_players=NUM_PLAYERS,
        optimizer="adamw",
        batch_size=1,
        num_epochs=1,
        training_steps_per_epoch=1,
        warmup_steps=0,
        learning_rate=1e-3,
        policy_loss_weight=0.0,
        value_loss_weight=1.0,
    )
    model = ConstantValueModel(NUM_PLAYERS)
    trainer = Trainer(model, cfg, torch.device("cpu"))

    layout = get_layout(NUM_PLAYERS)
    buffer = ReplayBuffer(1, layout.total_size, NUM_PLAYERS)

    state = _make_initialized_state(NUM_PLAYERS)
    legal_mask = np.zeros(int(UNIFIED_LOGIT_DIM), dtype=np.uint8)
    legal_mask[0] = 1
    policy_target = np.zeros(int(UNIFIED_LOGIT_DIM), dtype=np.float32)
    policy_target[0] = 1.0
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
