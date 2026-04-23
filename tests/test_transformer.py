from __future__ import annotations

import pytest
import torch

from nn.transformer import (
    NUM_PASS_PHASES,
    UNIFIED_LOGIT_DIM,
    RSSTransformerNet,
    TransformerConfig,
    build_action_lut,
)
from core.data import GameConstants, PHASE_ACTION_SIZES
from core.token_data import TokenWidth


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


def test_project_tokens_preserves_autocast_dtype(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    x = torch.randn(2, cfg.num_tokens, cfg.token_dim)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        tokens = model._project_tokens(x)

    assert tokens.dtype == torch.bfloat16
    assert tokens.shape == (2, cfg.num_tokens + NUM_PASS_PHASES, cfg.d_model)


def test_policy_layout_matches_phase_action_sizes(model: RSSTransformerNet) -> None:
    model._validate_policy_layout()


def test_corp_projection_uses_learned_identity_embedding(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    num_corps = int(GameConstants.NUM_CORPS)
    assert model.corp_proj.in_features == int(TokenWidth.TW_CORP) - num_corps
    assert tuple(model.corp_id_embed.weight.shape) == (num_corps, cfg.d_model)

    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x_with_ids = x.clone()
    x_with_ids[:, model._corp_slice, :num_corps] = torch.eye(num_corps).unsqueeze(0)

    projected_without_ids = model._project_tokens(x)[:, model._corp_slice]
    projected_with_ids = model._project_tokens(x_with_ids)[:, model._corp_slice]
    assert torch.allclose(projected_without_ids, projected_with_ids)

    expected_delta = model.corp_id_embed.weight[1] - model.corp_id_embed.weight[0]
    actual_delta = projected_without_ids[0, 1] - projected_without_ids[0, 0]
    assert torch.allclose(actual_delta, expected_delta)


def test_company_projection_uses_shared_corp_owner_embedding(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    num_corps = int(GameConstants.NUM_CORPS)
    owner_start = model._company_owner_corp_offset
    owner_stop = owner_start + num_corps

    assert model.company_proj.in_features == int(TokenWidth.TW_COMPANY) - num_corps

    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x_with_owner = x.clone()
    x_with_owner[:, model._company_slice, owner_start:owner_stop] = (
        torch.eye(num_corps).repeat(5, 1)[:36].unsqueeze(0)
    )

    company_without_owner = model._project_tokens(x)[:, model._company_slice]
    company_with_owner = model._project_tokens(x_with_owner)[:, model._company_slice]
    actual_delta = company_with_owner - company_without_owner

    expected_delta = model.corp_id_embed.weight[
        torch.arange(36) % num_corps
    ].unsqueeze(0)
    assert torch.allclose(actual_delta, expected_delta)


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
