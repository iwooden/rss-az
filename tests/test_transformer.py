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
    assert model.corp_proj.in_features == model._corp_rel_offset
    assert model._corp_president_offset == model._corp_rel_offset
    assert model._corp_companies_offset == model._corp_president_offset + model._corp_president_width
    assert (
        model._corp_companies_offset + model._corp_companies_width
        == int(TokenWidth.TW_CORP)
    )
    assert tuple(model.corp_id_embed.weight.shape) == (num_corps, cfg.d_model)

    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    projected_without_ids = model._project_tokens(x)[:, model._corp_slice]

    expected_delta = model.corp_id_embed.weight[1] - model.corp_id_embed.weight[0]
    actual_delta = projected_without_ids[0, 1] - projected_without_ids[0, 0]
    assert torch.allclose(actual_delta, expected_delta)


def test_corp_projection_uses_shared_relation_embeddings(model: RSSTransformerNet) -> None:
    num_corps = int(GameConstants.NUM_CORPS)
    num_companies = int(GameConstants.NUM_COMPANIES)
    num_player_slots = 5

    president_start = model._corp_president_offset
    president_stop = president_start + model._corp_president_width
    companies_start = model._corp_companies_offset
    companies_stop = companies_start + model._corp_companies_width

    assert model._corp_president_width == num_player_slots
    assert model._corp_companies_width == num_companies
    assert companies_stop == int(TokenWidth.TW_CORP)
    assert model.corp_proj.in_features == president_start

    x = torch.zeros(1, model.cfg.num_tokens, model.cfg.token_dim)
    x_with_relations = x.clone()
    president_indices = torch.arange(num_corps) % num_player_slots
    x_with_relations[:, model._corp_slice, president_start:president_stop] = torch.eye(
        num_player_slots
    )[president_indices].unsqueeze(0)

    owned_company_bitmap = torch.zeros(num_corps, num_companies)
    corp_indices = torch.arange(num_corps)
    owned_company_bitmap[corp_indices, corp_indices] = 1.0
    owned_company_bitmap[corp_indices, corp_indices + num_corps] = 1.0
    x_with_relations[:, model._corp_slice, companies_start:companies_stop] = (
        owned_company_bitmap.unsqueeze(0)
    )

    corp_without_relations = model._project_tokens(x)[:, model._corp_slice]
    corp_with_relations = model._project_tokens(x_with_relations)[:, model._corp_slice]
    actual_delta = corp_with_relations - corp_without_relations

    expected_delta = (
        model.player_id_embed.weight[president_indices]
        + owned_company_bitmap.to(model.company_id_embed.weight.dtype)
        @ model.company_id_embed.weight
    ).unsqueeze(0)
    assert torch.allclose(actual_delta, expected_delta)


def test_player_projection_uses_learned_identity_embedding(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    num_player_slots = 5
    assert model.player_proj.in_features == model._player_rel_offset
    assert model._player_shares_offset == model._player_rel_offset
    assert model._player_companies_offset == model._player_shares_offset + model._player_shares_width
    assert (
        model._player_companies_offset + model._player_companies_width
        == int(TokenWidth.TW_PLAYER)
    )
    assert tuple(model.player_id_embed.weight.shape) == (num_player_slots, cfg.d_model)

    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    projected_without_ids = model._project_tokens(x)[:, model._player_slice]

    expected_delta = model.player_id_embed.weight[1] - model.player_id_embed.weight[0]
    actual_delta = projected_without_ids[0, 1] - projected_without_ids[0, 0]
    assert torch.allclose(actual_delta, expected_delta)


def test_player_projection_uses_shared_relation_embeddings(model: RSSTransformerNet) -> None:
    num_corps = int(GameConstants.NUM_CORPS)
    num_companies = int(GameConstants.NUM_COMPANIES)

    shares_start = model._player_shares_offset
    shares_stop = shares_start + model._player_shares_width
    companies_start = model._player_companies_offset
    companies_stop = companies_start + model._player_companies_width

    assert model._player_shares_width == num_corps
    assert model._player_companies_width == num_companies
    assert companies_stop == int(TokenWidth.TW_PLAYER)
    assert model.player_proj.in_features == shares_start

    x = torch.zeros(1, model.cfg.num_tokens, model.cfg.token_dim)
    x_with_relations = x.clone()

    owned_shares = torch.zeros(model.cfg.num_players, num_corps)
    owned_shares[0, 0] = 0.2
    owned_shares[0, 1] = 0.4
    owned_shares[1, 2] = 0.6
    owned_shares[2, 7] = 1.0
    x_with_relations[:, model._player_slice, shares_start:shares_stop] = (
        owned_shares.unsqueeze(0)
    )

    owned_company_bitmap = torch.zeros(model.cfg.num_players, num_companies)
    owned_company_bitmap[0, [0, 5]] = 1.0
    owned_company_bitmap[1, [7, 13]] = 1.0
    owned_company_bitmap[2, [35]] = 1.0
    x_with_relations[:, model._player_slice, companies_start:companies_stop] = (
        owned_company_bitmap.unsqueeze(0)
    )

    player_without_relations = model._project_tokens(x)[:, model._player_slice]
    player_with_relations = model._project_tokens(x_with_relations)[:, model._player_slice]
    actual_delta = player_with_relations - player_without_relations

    expected_delta = (
        owned_shares.to(model.corp_id_embed.weight.dtype)
        @ model.corp_id_embed.weight
        + owned_company_bitmap.to(model.company_id_embed.weight.dtype)
        @ model.company_id_embed.weight
    ).unsqueeze(0)
    assert torch.allclose(actual_delta, expected_delta)


def test_active_entity_refs_broadcast_to_eligible_tokens(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x_active = x.clone()

    active_company = 0
    active_corp = 1
    active_player = 2
    x_active[:, model._company_slice.start + active_company, 0] = 1.0
    x_active[:, model._corp_slice.start + active_corp, 0] = 1.0
    x_active[:, model._player_slice.start + active_player, 0] = 1.0

    without_active = model._project_tokens(x)
    with_active = model._project_tokens(x_active)
    delta = with_active - without_active

    active_company_ref = model.company_id_embed.weight[active_company]
    active_corp_ref = model.corp_id_embed.weight[active_corp]
    active_player_ref = model.player_id_embed.weight[active_player]
    all_refs = active_company_ref + active_corp_ref + active_player_ref

    assert torch.allclose(delta[0, model._market_info_idx], torch.zeros_like(all_refs))
    assert torch.allclose(delta[0, model._global_info_idx], torch.zeros_like(all_refs))
    assert torch.allclose(delta[0, model._invest_idx], all_refs)
    assert torch.allclose(delta[0, model._pass_idxs[0]], all_refs)

    inactive_company_idx = model._company_slice.start + 1
    assert torch.allclose(
        delta[0, inactive_company_idx],
        active_player_ref + active_corp_ref,
    )
    active_company_idx = model._company_slice.start + active_company
    expected_active_company_delta = (
        model.company_proj.weight[:, 0]
        + active_player_ref
        + active_corp_ref
    )
    assert torch.allclose(delta[0, active_company_idx], expected_active_company_delta)

    inactive_corp_idx = model._corp_slice.start
    assert torch.allclose(
        delta[0, inactive_corp_idx],
        active_player_ref + active_company_ref,
    )
    active_corp_idx = model._corp_slice.start + active_corp
    expected_active_corp_delta = (
        model.corp_proj.weight[:, 0]
        + active_player_ref
        + active_company_ref
    )
    assert torch.allclose(delta[0, active_corp_idx], expected_active_corp_delta)

    inactive_player_idx = model._player_slice.start
    assert torch.allclose(
        delta[0, inactive_player_idx],
        active_corp_ref + active_company_ref,
    )
    active_player_idx = model._player_slice.start + active_player
    expected_active_player_delta = (
        model.player_proj.weight[:, 0]
        + active_corp_ref
        + active_company_ref
    )
    assert torch.allclose(delta[0, active_player_idx], expected_active_player_delta)


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


def test_fi_projection_uses_owned_company_embeddings(model: RSSTransformerNet) -> None:
    num_companies = int(GameConstants.NUM_COMPANIES)
    companies_start = model._fi_companies_offset
    companies_stop = companies_start + model._fi_companies_width

    assert model.fi_proj.in_features == companies_start
    assert model._fi_companies_width == num_companies
    assert companies_stop == int(TokenWidth.TW_FI)

    x = torch.zeros(1, model.cfg.num_tokens, model.cfg.token_dim)
    x_with_companies = x.clone()
    owned_company_bitmap = torch.zeros(num_companies)
    owned_company_bitmap[[0, 7, 35]] = 1.0
    x_with_companies[:, model._fi_idx, companies_start:companies_stop] = (
        owned_company_bitmap.unsqueeze(0)
    )

    fi_without_companies = model._project_tokens(x)[:, model._fi_idx]
    fi_with_companies = model._project_tokens(x_with_companies)[:, model._fi_idx]
    actual_delta = fi_with_companies - fi_without_companies

    expected_delta = (
        owned_company_bitmap.to(model.company_id_embed.weight.dtype)
        @ model.company_id_embed.weight
    ).unsqueeze(0)
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
