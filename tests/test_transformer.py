from __future__ import annotations

import pytest
import torch

from nn.transformer import (
    NUM_PASS_PHASES,
    NUM_SYNTHETIC_TOKENS,
    PASS_PHASE_IDS,
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
    x[:, :, 0] = 1.0
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
    assert tokens.shape == (
        2,
        cfg.num_tokens + NUM_PASS_PHASES + NUM_SYNTHETIC_TOKENS,
        cfg.d_model,
    )


def test_project_tokens_ignores_attention_mask_slot(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x_with_masks = x.clone()
    x_with_masks[:, :, 0] = 1.0

    assert torch.allclose(model._project_tokens(x), model._project_tokens(x_with_masks))


def test_attention_mask_combines_input_rows_and_pass_anchors(
    model: RSSTransformerNet,
) -> None:
    cfg = model.cfg
    x = torch.zeros(2, cfg.num_tokens, cfg.token_dim)

    x[0, model._global_info_idx, 0] = 1.0
    x[0, model._fi_idx, 0] = 1.0
    x[0, model._global_info_idx, model._global_phase_offset + PASS_PHASE_IDS[0]] = 1.0

    x[1, model._market_info_idx, 0] = 1.0
    x[1, model._global_info_idx, 0] = 1.0
    x[1, model._global_info_idx, model._global_phase_offset + PASS_PHASE_IDS[2]] = 1.0

    attn_mask = model._attention_mask(x)

    assert attn_mask.dtype == torch.bool
    assert attn_mask.shape == (
        2,
        1,
        1,
        cfg.num_tokens + NUM_PASS_PHASES + NUM_SYNTHETIC_TOKENS,
    )

    flat_mask = attn_mask[:, 0, 0, :]
    assert torch.equal(flat_mask[:, :cfg.num_tokens], x[:, :, 0].to(torch.bool))

    expected_pass_mask = torch.zeros(2, NUM_PASS_PHASES, dtype=torch.bool)
    expected_pass_mask[0, 0] = True
    expected_pass_mask[1, 2] = True
    assert torch.equal(
        flat_mask[:, cfg.num_tokens:cfg.num_tokens + NUM_PASS_PHASES],
        expected_pass_mask,
    )
    assert flat_mask[:, -NUM_SYNTHETIC_TOKENS:].all()


def test_policy_layout_matches_phase_action_sizes(model: RSSTransformerNet) -> None:
    model._validate_policy_layout()


def test_corp_projection_uses_learned_identity_embedding(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    num_corps = int(GameConstants.NUM_CORPS)
    assert model.corp_proj.in_features == (
        model._corp_rel_offset - model._token_feature_start
    )
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


def test_corp_projection_uses_gated_relation_embeddings(model: RSSTransformerNet) -> None:
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
    assert model.corp_proj.in_features == president_start - model._token_feature_start
    assert tuple(model.company_ownership_gate.shape) == (model.cfg.d_model,)
    assert tuple(model.corp_president_gate.shape) == (model.cfg.d_model,)

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

    original_company_gate = model.company_ownership_gate.detach().clone()
    original_president_gate = model.corp_president_gate.detach().clone()
    company_gate = torch.linspace(
        0.5,
        1.5,
        model.cfg.d_model,
        dtype=model.company_ownership_gate.dtype,
        device=model.company_ownership_gate.device,
    )
    president_gate = torch.linspace(
        1.5,
        0.5,
        model.cfg.d_model,
        dtype=model.corp_president_gate.dtype,
        device=model.corp_president_gate.device,
    )
    with torch.no_grad():
        model.company_ownership_gate.copy_(company_gate)
        model.corp_president_gate.copy_(president_gate)
    try:
        corp_without_relations = model._project_tokens(x)[:, model._corp_slice]
        corp_with_relations = model._project_tokens(x_with_relations)[:, model._corp_slice]
        actual_delta = corp_with_relations - corp_without_relations

        owned_company_delta = (
            owned_company_bitmap.to(model.company_id_embed.weight.dtype)
            @ model.company_id_embed.weight
        ) / owned_company_bitmap.sum(dim=-1, keepdim=True).sqrt()
        expected_delta = (
            model.player_id_embed.weight[president_indices]
            * president_gate.view(1, -1)
            + owned_company_delta * company_gate.view(1, -1)
        ).unsqueeze(0)
        assert torch.allclose(actual_delta, expected_delta)
    finally:
        with torch.no_grad():
            model.company_ownership_gate.copy_(original_company_gate)
            model.corp_president_gate.copy_(original_president_gate)


def test_player_projection_uses_learned_identity_embedding(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    num_player_slots = 5
    assert model.player_proj.in_features == (
        model._player_rel_offset - model._token_feature_start
    )
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


def test_player_projection_uses_gated_relation_embeddings(model: RSSTransformerNet) -> None:
    num_corps = int(GameConstants.NUM_CORPS)
    num_companies = int(GameConstants.NUM_COMPANIES)

    shares_start = model._player_shares_offset
    shares_stop = shares_start + model._player_shares_width
    companies_start = model._player_companies_offset
    companies_stop = companies_start + model._player_companies_width

    assert model._player_shares_width == num_corps
    assert model._player_companies_width == num_companies
    assert companies_stop == int(TokenWidth.TW_PLAYER)
    assert model.player_proj.in_features == shares_start - model._token_feature_start
    assert tuple(model.company_ownership_gate.shape) == (model.cfg.d_model,)
    assert tuple(model.share_ownership_gate.shape) == (model.cfg.d_model,)

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

    original_company_gate = model.company_ownership_gate.detach().clone()
    original_share_gate = model.share_ownership_gate.detach().clone()
    company_gate = torch.linspace(
        0.5,
        1.5,
        model.cfg.d_model,
        dtype=model.company_ownership_gate.dtype,
        device=model.company_ownership_gate.device,
    )
    share_gate = torch.linspace(
        1.5,
        0.5,
        model.cfg.d_model,
        dtype=model.share_ownership_gate.dtype,
        device=model.share_ownership_gate.device,
    )
    with torch.no_grad():
        model.company_ownership_gate.copy_(company_gate)
        model.share_ownership_gate.copy_(share_gate)
    try:
        player_without_relations = model._project_tokens(x)[:, model._player_slice]
        player_with_relations = model._project_tokens(x_with_relations)[:, model._player_slice]
        actual_delta = player_with_relations - player_without_relations

        owned_company_delta = (
            owned_company_bitmap.to(model.company_id_embed.weight.dtype)
            @ model.company_id_embed.weight
        ) / owned_company_bitmap.sum(dim=-1, keepdim=True).sqrt()
        share_delta = (
            owned_shares.to(model.corp_id_embed.weight.dtype)
            @ model.corp_id_embed.weight
        )
        expected_delta = (
            share_delta * share_gate.view(1, -1)
            + owned_company_delta * company_gate.view(1, -1)
        ).unsqueeze(0)
        assert torch.allclose(actual_delta, expected_delta)
    finally:
        with torch.no_grad():
            model.company_ownership_gate.copy_(original_company_gate)
            model.share_ownership_gate.copy_(original_share_gate)


def test_active_entity_refs_broadcast_to_phase_tokens_and_pass_anchors(
    model: RSSTransformerNet,
) -> None:
    cfg = model.cfg
    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x_active = x.clone()

    active_company = 0
    active_corp = 1
    active_player = 2
    x_active[
        :, model._company_slice.start + active_company, model._is_selected_offset
    ] = 1.0
    x_active[
        :, model._corp_slice.start + active_corp, model._is_selected_offset
    ] = 1.0
    x_active[
        :, model._player_slice.start + active_player, model._is_selected_offset
    ] = 1.0

    active_company_ref = model.company_id_embed.weight[active_company]
    active_corp_ref = model.corp_id_embed.weight[active_corp]
    active_player_ref = model.player_id_embed.weight[active_player]
    assert tuple(model.active_player_gate.shape) == (cfg.d_model,)
    assert tuple(model.active_corp_gate.shape) == (cfg.d_model,)
    assert tuple(model.active_company_gate.shape) == (cfg.d_model,)

    original_player_gate = model.active_player_gate.detach().clone()
    original_corp_gate = model.active_corp_gate.detach().clone()
    original_company_gate = model.active_company_gate.detach().clone()
    player_gate = torch.linspace(
        0.5,
        1.5,
        cfg.d_model,
        dtype=model.active_player_gate.dtype,
        device=model.active_player_gate.device,
    )
    corp_gate = torch.linspace(
        1.5,
        0.5,
        cfg.d_model,
        dtype=model.active_corp_gate.dtype,
        device=model.active_corp_gate.device,
    )
    company_gate = torch.linspace(
        0.75,
        1.25,
        cfg.d_model,
        dtype=model.active_company_gate.dtype,
        device=model.active_company_gate.device,
    )
    with torch.no_grad():
        model.active_player_gate.copy_(player_gate)
        model.active_corp_gate.copy_(corp_gate)
        model.active_company_gate.copy_(company_gate)

    try:
        without_active = model._project_tokens(x)
        with_active = model._project_tokens(x_active)
        delta = with_active - without_active

        all_refs = (
            active_player_ref * player_gate
            + active_corp_ref * corp_gate
            + active_company_ref * company_gate
        )

        assert torch.allclose(delta[0, model._market_info_idx], torch.zeros_like(all_refs))
        assert torch.allclose(delta[0, model._global_info_idx], torch.zeros_like(all_refs))
        assert torch.allclose(delta[0, model._fi_idx], torch.zeros_like(all_refs))
        assert torch.allclose(delta[0, model._invest_idx], all_refs)
        assert torch.allclose(delta[0, model._acq_price_info_idx], all_refs)
        assert torch.allclose(delta[0, model._pass_idxs[0]], all_refs)

        inactive_company_idx = model._company_slice.start + 1
        assert torch.allclose(
            delta[0, inactive_company_idx],
            torch.zeros_like(all_refs),
        )
        active_company_idx = model._company_slice.start + active_company
        expected_active_company_delta = model.company_proj.weight[:, 0]
        assert torch.allclose(delta[0, active_company_idx], expected_active_company_delta)

        inactive_corp_idx = model._corp_slice.start
        assert torch.allclose(
            delta[0, inactive_corp_idx],
            torch.zeros_like(all_refs),
        )
        active_corp_idx = model._corp_slice.start + active_corp
        expected_active_corp_delta = model.corp_proj.weight[:, 0]
        assert torch.allclose(delta[0, active_corp_idx], expected_active_corp_delta)

        inactive_player_idx = model._player_slice.start
        assert torch.allclose(
            delta[0, inactive_player_idx],
            torch.zeros_like(all_refs),
        )
        active_player_idx = model._player_slice.start + active_player
        expected_active_player_delta = model.player_proj.weight[:, 0]
        assert torch.allclose(delta[0, active_player_idx], expected_active_player_delta)
    finally:
        with torch.no_grad():
            model.active_player_gate.copy_(original_player_gate)
            model.active_corp_gate.copy_(original_corp_gate)
            model.active_company_gate.copy_(original_company_gate)


def test_phase_ref_broadcasts_to_all_but_global_info(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    phase_width = len(PHASE_ACTION_SIZES)
    assert model._global_phase_width == phase_width
    assert model.global_info_proj.in_features == (
        int(TokenWidth.TW_GLOBAL_INFO)
        - model._global_phase_offset
        - phase_width
    )
    assert tuple(model.phase_embed.weight.shape) == (phase_width, cfg.d_model)

    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    phase_id = 1
    x_with_phase = x.clone()
    x_with_phase[:, model._global_info_idx, model._global_phase_offset + phase_id] = 1.0

    without_phase = model._project_tokens(x)
    with_phase = model._project_tokens(x_with_phase)
    delta = with_phase - without_phase

    phase_ref = model.phase_embed.weight[phase_id]
    assert torch.allclose(delta[0, model._global_info_idx], torch.zeros_like(phase_ref))
    assert torch.allclose(delta[0, model._market_info_idx], phase_ref)
    assert torch.allclose(delta[0, model._company_slice.start], phase_ref)
    assert torch.allclose(delta[0, model._invest_idx], phase_ref)
    assert torch.allclose(delta[0, model._pass_idxs[0]], phase_ref)

    x_with_global_suffix = x.clone()
    x_with_global_suffix[
        :,
        model._global_info_idx,
        model._global_phase_offset + phase_width,
    ] = 1.0
    with_global_suffix = model._project_tokens(x_with_global_suffix)
    suffix_delta = with_global_suffix - without_phase
    expected_global_delta = model.global_info_proj.weight[:, 0]
    assert torch.allclose(suffix_delta[0, model._global_info_idx], expected_global_delta)
    assert torch.allclose(suffix_delta[0, model._market_info_idx], torch.zeros_like(phase_ref))


def test_company_projection_uses_learned_identity_embedding(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    num_companies = int(GameConstants.NUM_COMPANIES)
    assert model.company_proj.in_features == (
        model._company_owner_offset - model._token_feature_start
    )
    assert tuple(model.company_id_embed.weight.shape) == (num_companies, cfg.d_model)

    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    projected_without_ids = model._project_tokens(x)[:, model._company_slice]

    expected_delta = model.company_id_embed.weight[1] - model.company_id_embed.weight[0]
    actual_delta = projected_without_ids[0, 1] - projected_without_ids[0, 0]
    assert torch.allclose(actual_delta, expected_delta)


def test_removed_companies_token_uses_scaled_company_id_embeddings(
    model: RSSTransformerNet,
) -> None:
    cfg = model.cfg
    removed_company_ids = torch.tensor([0, 7, 35], dtype=torch.long)
    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x_removed = x.clone()
    x_removed[
        :,
        model._company_slice.start + removed_company_ids,
        model._company_removed_offset,
    ] = 1.0

    without_removed = model._project_tokens(x)
    with_removed = model._project_tokens(x_removed)
    actual_delta = (
        with_removed[:, model._removed_companies_idx]
        - without_removed[:, model._removed_companies_idx]
    )

    expected_delta = (
        model.company_id_embed.weight[removed_company_ids].sum(dim=0)
        / float(len(removed_company_ids)) ** 0.5
    ).unsqueeze(0)
    assert torch.allclose(actual_delta, expected_delta)


def test_company_projection_uses_gated_owned_by_embeddings(model: RSSTransformerNet) -> None:
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
    assert model.company_proj.in_features == owner_start - model._token_feature_start
    assert tuple(model.company_owned_by_gate.shape) == (cfg.d_model,)

    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x_with_owner = x.clone()
    owner_indices = torch.arange(36) % owner_width
    x_with_owner[:, model._company_slice, owner_start:owner_stop] = torch.eye(
        owner_width
    )[owner_indices].unsqueeze(0)

    original_gate = model.company_owned_by_gate.detach().clone()
    test_gate = torch.linspace(
        0.5,
        1.5,
        cfg.d_model,
        dtype=model.company_owned_by_gate.dtype,
        device=model.company_owned_by_gate.device,
    )
    with torch.no_grad():
        model.company_owned_by_gate.copy_(test_gate)
    try:
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
        expected_delta = (
            owner_ref_table[owner_indices].unsqueeze(0)
            * test_gate.view(1, 1, -1)
        )
        assert torch.allclose(actual_delta, expected_delta)
    finally:
        with torch.no_grad():
            model.company_owned_by_gate.copy_(original_gate)


def test_fi_projection_uses_gated_owned_company_embeddings(model: RSSTransformerNet) -> None:
    num_companies = int(GameConstants.NUM_COMPANIES)
    companies_start = model._fi_companies_offset
    companies_stop = companies_start + model._fi_companies_width

    assert model.fi_proj.in_features == companies_start - model._token_feature_start
    assert model._fi_companies_width == num_companies
    assert companies_stop == int(TokenWidth.TW_FI)
    assert tuple(model.company_ownership_gate.shape) == (model.cfg.d_model,)

    x = torch.zeros(1, model.cfg.num_tokens, model.cfg.token_dim)
    x_with_companies = x.clone()
    owned_company_bitmap = torch.zeros(num_companies)
    owned_company_bitmap[[0, 7, 35]] = 1.0
    x_with_companies[:, model._fi_idx, companies_start:companies_stop] = (
        owned_company_bitmap.unsqueeze(0)
    )

    original_gate = model.company_ownership_gate.detach().clone()
    test_gate = torch.linspace(
        0.5,
        1.5,
        model.cfg.d_model,
        dtype=model.company_ownership_gate.dtype,
        device=model.company_ownership_gate.device,
    )
    with torch.no_grad():
        model.company_ownership_gate.copy_(test_gate)
    try:
        fi_without_companies = model._project_tokens(x)[:, model._fi_idx]
        fi_with_companies = model._project_tokens(x_with_companies)[:, model._fi_idx]
        actual_delta = fi_with_companies - fi_without_companies

        expected_delta = (
            owned_company_bitmap.to(model.company_id_embed.weight.dtype)
            @ model.company_id_embed.weight
        ) / owned_company_bitmap.sum(dim=-1, keepdim=True).sqrt()
        expected_delta = (expected_delta * test_gate).unsqueeze(0)
        assert torch.allclose(actual_delta, expected_delta)
    finally:
        with torch.no_grad():
            model.company_ownership_gate.copy_(original_gate)


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
