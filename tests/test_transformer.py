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
    assert model.corp_proj.in_features == int(TokenWidth.TW_CORP)
    assert tuple(model.corp_id_embed.weight.shape) == (num_corps, cfg.d_model)

    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    projected_without_ids = model._project_tokens(x)[:, model._corp_slice]

    expected_delta = model.corp_id_embed.weight[1] - model.corp_id_embed.weight[0]
    actual_delta = projected_without_ids[0, 1] - projected_without_ids[0, 0]
    assert torch.allclose(actual_delta, expected_delta)


def test_player_projection_uses_learned_identity_embedding(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    num_player_slots = 5
    assert model.player_proj.in_features == int(TokenWidth.TW_PLAYER)
    assert tuple(model.player_id_embed.weight.shape) == (num_player_slots, cfg.d_model)

    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    projected_without_ids = model._project_tokens(x)[:, model._player_slice]

    expected_delta = model.player_id_embed.weight[1] - model.player_id_embed.weight[0]
    actual_delta = projected_without_ids[0, 1] - projected_without_ids[0, 0]
    assert torch.allclose(actual_delta, expected_delta)


def test_company_projection_uses_learned_identity_embedding(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    num_companies = int(GameConstants.NUM_COMPANIES)
    assert model.company_proj.in_features == model._company_owner_offset
    assert tuple(model.company_id_embed.weight.shape) == (num_companies, cfg.d_model)

    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    projected_without_ids = model._project_tokens(x)[:, model._company_slice]

    expected_delta = model.company_id_embed.weight[1] - model.company_id_embed.weight[0]
    actual_delta = projected_without_ids[0, 1] - projected_without_ids[0, 0]
    assert torch.allclose(actual_delta, expected_delta)


def test_company_projection_uses_shared_owner_embeddings(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    num_corps = int(GameConstants.NUM_CORPS)
    num_player_slots = 5
    owner_start = model._company_owner_offset
    owner_width = num_corps + num_player_slots + 1
    owner_stop = owner_start + owner_width

    assert model._company_owner_corp_offset == owner_start
    assert model._company_owner_player_offset == owner_start + num_corps
    assert model._company_owner_fi_offset == owner_start + num_corps + num_player_slots
    assert owner_stop == int(TokenWidth.TW_COMPANY)
    assert model.company_proj.in_features == owner_start

    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x_with_owner = x.clone()
    owner_indices = torch.arange(36) % owner_width
    x_with_owner[:, model._company_slice, owner_start:owner_stop] = torch.eye(
        owner_width
    )[owner_indices].unsqueeze(0)

    company_without_owner = model._project_tokens(x)[:, model._company_slice]
    company_with_owner = model._project_tokens(x_with_owner)[:, model._company_slice]
    actual_delta = company_with_owner - company_without_owner

    fi_type_id = int(model._type_ids[model._fi_idx].item())
    owner_ref_table = torch.cat(
        [
            model.corp_id_embed.weight,
            model.player_id_embed.weight,
            model.type_embeds[fi_type_id].unsqueeze(0),
        ],
        dim=0,
    )
    expected_delta = owner_ref_table[owner_indices].unsqueeze(0)
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
