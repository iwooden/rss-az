from __future__ import annotations

import numpy as np
import pytest
import torch

from core.attention_relations import NUM_ATTENTION_RELATIONS
from core.resnet_data import get_resnet_vector_size
from core.state import GameState, get_layout
from core.token_data import get_num_tokens
from entities.turn import TURN
from nn.transformer import PHASES_WITH_PASS_HEAD, UNIFIED_LOGIT_DIM
from train.config import TrainingConfig
from train.replay_buffer import ReplayBuffer
from train.trainer import Trainer

NUM_PLAYERS = 3


class ConstantValueModel(torch.nn.Module):
    def __init__(self, num_players: int) -> None:
        super().__init__()
        self.value_bias = torch.nn.Parameter(torch.zeros(num_players))

    def forward(
        self,
        tokens: torch.Tensor,
        _legal_masks: torch.Tensor,
        relations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            2 * len(PHASES_WITH_PASS_HEAD),
            dtype=policy_logits.dtype,
            device=policy_logits.device,
        )


class RelativeValueVectorModel(torch.nn.Module):
    def __init__(self, num_players: int, relative_values: list[float]) -> None:
        super().__init__()
        self.value_bias = torch.nn.Parameter(torch.tensor(relative_values))
        self.num_players = num_players
        self.last_input_shape: tuple[int, ...] | None = None
        self.last_input_dtype: torch.dtype | None = None
        self.last_legal_mask_dtype: torch.dtype | None = None

    def forward(
        self,
        vectors: torch.Tensor,
        legal_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_input_shape = tuple(vectors.shape)
        self.last_input_dtype = vectors.dtype
        self.last_legal_mask_dtype = legal_masks.dtype
        batch = vectors.shape[0]
        policy_logits = torch.zeros(
            (batch, int(UNIFIED_LOGIT_DIM)),
            dtype=vectors.dtype,
            device=vectors.device,
        )
        values = self.value_bias.to(vectors.device).unsqueeze(0).expand(batch, -1)
        return policy_logits, values


def _make_initialized_state(
    num_players: int,
    *,
    active_player: int = 0,
) -> np.ndarray:
    state = GameState(num_players)
    state.initialize_game(num_players, seed=0)
    TURN.set_active_player(state, active_player)
    return state._array.copy()


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
    assert model.last_relations_shape == (
        1,
        NUM_ATTENTION_RELATIONS,
        get_num_tokens(NUM_PLAYERS),
        get_num_tokens(NUM_PLAYERS),
    )
    assert model.last_relations_dtype == torch.uint8


def test_resnet_train_step_uses_vectors_and_rotates_value_targets() -> None:
    cfg = TrainingConfig(
        num_players=NUM_PLAYERS,
        model_type="resnet",
        optimizer="adamw",
        batch_size=1,
        num_epochs=1,
        training_steps_per_epoch=1,
        warmup_epochs=0,
        learning_rate=1e-3,
        policy_loss_weight=0.0,
        value_loss_weight=1.0,
    )
    # Canonical [-1, 0, 1] rotated for active player 1 becomes [0, 1, -1].
    model = RelativeValueVectorModel(NUM_PLAYERS, [0.0, 1.0, -1.0])
    trainer = Trainer(model, cfg, torch.device("cpu"))

    layout = get_layout(NUM_PLAYERS)
    buffer = ReplayBuffer(1, layout.total_size, NUM_PLAYERS)

    state = _make_initialized_state(NUM_PLAYERS, active_player=1)
    legal_mask = np.zeros(int(UNIFIED_LOGIT_DIM), dtype=np.uint8)
    legal_mask[0] = 1
    policy_target = np.zeros(int(UNIFIED_LOGIT_DIM), dtype=np.float32)
    policy_target[0] = 1.0
    value_target = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    buffer.add_stacked(
        states=state[None, :],
        phase_ids=np.array([0], dtype=np.int8),
        legal_masks=legal_mask[None, :],
        policy_targets=policy_target[None, :],
        value_targets=value_target[None, :],
    )

    losses = trainer.train_step(buffer, batch_size=1, rng=np.random.default_rng(0))

    assert losses["value_loss"] == pytest.approx(0.0)
    assert losses["total_loss"] == pytest.approx(0.0)
    assert model.last_input_shape == (1, get_resnet_vector_size(NUM_PLAYERS))
    assert model.last_input_dtype == torch.float32
    assert model.last_legal_mask_dtype == torch.bool
