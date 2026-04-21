from __future__ import annotations

import pytest
import torch

from nn.transformer import (
    UNIFIED_LOGIT_DIM,
    RSSTransformerNet,
    TransformerConfig,
    build_action_lut,
)
from core.data import PHASE_ACTION_SIZES


NUM_PLAYERS = 3
U_DIM = int(UNIFIED_LOGIT_DIM)


@pytest.fixture(scope="module")
def model() -> RSSTransformerNet:
    torch.manual_seed(0)
    return RSSTransformerNet(TransformerConfig(num_players=NUM_PLAYERS)).to(torch.device("cpu"))


@pytest.fixture(scope="module")
def valid_inputs(model: RSSTransformerNet) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = model.cfg
    x = torch.randn(2, cfg.num_tokens, cfg.token_dim)
    lut = build_action_lut()
    legal_mask = torch.zeros(2, U_DIM, dtype=torch.bool)
    legal_mask[0, lut[0, : int(PHASE_ACTION_SIZES[0])]] = True
    legal_mask[1, lut[1, : int(PHASE_ACTION_SIZES[1])]] = True
    return x, legal_mask


def test_forward_rejects_wrong_legal_mask_shape(model: RSSTransformerNet, valid_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, _ = valid_inputs
    wrong_mask = torch.zeros(1, U_DIM, dtype=torch.bool)

    with pytest.raises(AssertionError, match="legal_mask shape"):
        model(x, wrong_mask)


def test_forward_rejects_wrong_num_tokens(model: RSSTransformerNet, valid_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, legal_mask = valid_inputs
    wrong_x = torch.randn(x.shape[0], x.shape[1] - 1, x.shape[2])

    with pytest.raises(AssertionError, match="x shape"):
        model(wrong_x, legal_mask)


def test_forward_rejects_wrong_token_dim(model: RSSTransformerNet, valid_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, legal_mask = valid_inputs
    wrong_x = torch.randn(x.shape[0], x.shape[1], x.shape[2] - 1)

    with pytest.raises(AssertionError, match="x shape"):
        model(wrong_x, legal_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for device mismatch test")
def test_forward_rejects_legal_mask_on_different_device() -> None:
    model = RSSTransformerNet(TransformerConfig(num_players=NUM_PLAYERS)).to(torch.device("cuda"))
    cfg = model.cfg
    x = torch.randn(1, cfg.num_tokens, cfg.token_dim, device=torch.device("cuda"))
    lut = build_action_lut()
    legal_mask = torch.zeros(1, U_DIM, dtype=torch.bool)
    legal_mask[0, lut[0, : int(PHASE_ACTION_SIZES[0])]] = True

    with pytest.raises(AssertionError, match="legal_mask device"):
        model(x, legal_mask)
