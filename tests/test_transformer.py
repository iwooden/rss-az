from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from core.actions import get_decision_phase_py
from core.attention_relations import NUM_ATTENTION_RELATIONS
from nn.transformer import (
    UNIFIED_LOGIT_DIM,
    RSSTransformerNet,
    TransformerConfig,
    build_action_lut,
)
from core.data import DecisionPhase, GameConstants, PHASE_ACTION_SIZES
from core.state import GameState
from core.token_data import TokenDataSize, TokenWidth, get_num_tokens, get_token_data


NUM_PLAYERS = 3
U_DIM = int(UNIFIED_LOGIT_DIM)
STATES_NPZ = Path(__file__).with_name("states.npz")


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


@pytest.fixture(scope="module")
def attention_mask_model() -> RSSTransformerNet:
    torch.manual_seed(123)
    model = RSSTransformerNet(
        TransformerConfig(
            num_players=NUM_PLAYERS,
            d_model=48,
            num_heads=3,
            num_layers=2,
            ff_mult=2.0,
        )
    ).to(torch.device("cpu"))
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.trunc_normal_(block.out_proj.weight, std=0.02)
            torch.nn.init.trunc_normal_(block.ffn_down.weight, std=0.02)
    model.eval()
    return model


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


def test_forward_accepts_relation_planes(model: RSSTransformerNet, valid_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, legal_mask = valid_inputs
    relations = torch.zeros(
        x.shape[0],
        NUM_ATTENTION_RELATIONS,
        model.cfg.num_tokens,
        model.cfg.num_tokens,
        dtype=torch.uint8,
    )

    logits_with_relations, values_with_relations = model(x, legal_mask, relations)
    logits_without_relations, values_without_relations = model(x, legal_mask)

    assert torch.allclose(logits_with_relations, logits_without_relations)
    assert torch.allclose(values_with_relations, values_without_relations)


def test_forward_rejects_wrong_relation_shape(model: RSSTransformerNet, valid_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, legal_mask = valid_inputs
    relations = torch.zeros(
        x.shape[0],
        NUM_ATTENTION_RELATIONS,
        model.cfg.num_tokens,
        model.cfg.num_tokens - 1,
        dtype=torch.uint8,
    )

    with pytest.raises(AssertionError, match="relations shape"):
        model(x, legal_mask, relations)


def test_forward_rejects_wrong_relation_dtype(model: RSSTransformerNet, valid_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, legal_mask = valid_inputs
    relations = torch.zeros(
        x.shape[0],
        NUM_ATTENTION_RELATIONS,
        model.cfg.num_tokens,
        model.cfg.num_tokens,
        dtype=torch.float32,
    )

    with pytest.raises(AssertionError, match="relations must be bool or uint8"):
        model(x, legal_mask, relations)


def test_project_tokens_preserves_autocast_dtype(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    x = torch.randn(2, cfg.num_tokens, cfg.token_dim)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        tokens = model._project_tokens(x)

    assert tokens.dtype == torch.bfloat16
    assert tokens.shape == (
        2,
        cfg.num_tokens,
        cfg.d_model,
    )


def test_project_tokens_ignores_attention_mask_slot(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x_with_masks = x.clone()
    x_with_masks[:, :, 0] = 1.0

    assert torch.allclose(model._project_tokens(x), model._project_tokens(x_with_masks))


def test_attention_mask_matches_input_attention_rows(
    model: RSSTransformerNet,
) -> None:
    cfg = model.cfg
    x = torch.zeros(2, cfg.num_tokens, cfg.token_dim)

    x[0, model._global_info_idx, 0] = 1.0
    x[0, model._fi_idx, 0] = 1.0

    x[1, model._market_info_idx, 0] = 1.0
    x[1, model._global_info_idx, 0] = 1.0

    attn_mask = model._attention_mask(x)

    assert attn_mask.dtype == torch.bool
    assert attn_mask.shape == (
        2,
        1,
        1,
        cfg.num_tokens,
    )

    flat_mask = attn_mask[:, 0, 0, :]
    assert torch.equal(flat_mask, x[:, :, 0].to(torch.bool))


def _phase_state_cases() -> list[tuple[str, np.ndarray]]:
    with np.load(STATES_NPZ, allow_pickle=False) as data:
        assert int(data["num_players"]) == NUM_PLAYERS
        phase_names = [str(name) for name in data["phase_names"].tolist()]
        states = [state.copy() for state in data["states"]]
    return list(zip(phase_names, states, strict=True))


def _token_buffer_for_state(state: GameState) -> torch.Tensor:
    num_tokens = get_num_tokens(NUM_PLAYERS)
    buf = np.zeros(
        (num_tokens, int(TokenDataSize.TOKEN_DIM)),
        dtype=np.float32,
    )
    get_token_data(state, buf)
    return torch.from_numpy(buf).unsqueeze(0)


def _token_labels(model: RSSTransformerNet) -> list[str]:
    labels = [""] * model.cfg.num_tokens
    labels[model._market_info_idx] = "MarketInfo"
    for cid in range(int(GameConstants.NUM_COMPANIES)):
        labels[model._company_slice.start + cid] = f"Company[{cid}]"
    labels[model._fi_idx] = "FI"
    labels[model._global_info_idx] = "GlobalInfo"
    labels[model._invest_idx] = "Invest"
    labels[model._auction_idx] = "Auction"
    labels[model._dividend_idx] = "Dividend"
    labels[model._issue_idx] = "Issue"
    labels[model._par_idx] = "ParIPO"
    labels[model._acq_offer_idx] = "AcqOffer"
    labels[model._acq_price_info_idx] = "AcqPriceInfo"
    for corp_id in range(int(GameConstants.NUM_CORPS)):
        labels[model._corp_slice.start + corp_id] = f"Corp[{corp_id}]"
    for player_id in range(NUM_PLAYERS):
        labels[model._player_slice.start + player_id] = f"Player[{player_id}]"
    assert all(labels), labels
    return labels


def _expected_attention_mask(
    model: RSSTransformerNet,
    state: GameState,
    phase_id: int,
) -> torch.Tensor:
    expected = torch.zeros(model.cfg.num_tokens, dtype=torch.bool)

    expected[model._market_info_idx] = True
    expected[model._fi_idx] = True
    expected[model._global_info_idx] = True
    expected[model._company_slice] = True
    expected[model._corp_slice] = True
    expected[model._player_slice] = True

    phase_token_indices: dict[int, int] = {
        int(DecisionPhase.DPHASE_INVEST): model._invest_idx,
        int(DecisionPhase.DPHASE_BID): model._auction_idx,
        int(DecisionPhase.DPHASE_DIVIDENDS): model._dividend_idx,
        int(DecisionPhase.DPHASE_ISSUE): model._issue_idx,
        int(DecisionPhase.DPHASE_IPO): model._par_idx,
        int(DecisionPhase.DPHASE_PAR): model._par_idx,
        int(DecisionPhase.DPHASE_ACQ_OFFER): model._acq_offer_idx,
        int(DecisionPhase.DPHASE_ACQ_SELECT_PRICE): model._acq_price_info_idx,
    }
    phase_token_idx = phase_token_indices.get(phase_id)
    if phase_token_idx is not None:
        expected[phase_token_idx] = True

    return expected


def _format_mask_diff(
    *,
    actual: torch.Tensor,
    expected: torch.Tensor,
    labels: list[str],
) -> str:
    bad = torch.nonzero(actual != expected, as_tuple=False).flatten().tolist()
    return ", ".join(
        f"{idx}:{labels[idx]} actual={bool(actual[idx])} expected={bool(expected[idx])}"
        for idx in bad
    )


def _run_trunk_from_projected(
    model: RSSTransformerNet,
    tokens: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    for block in model.blocks:
        tokens = block(tokens, attn_mask)
    return model.final_norm(tokens)


@pytest.mark.parametrize(("phase_name", "state_array"), _phase_state_cases())
def test_saved_phase_states_attention_mask_matches_visibility_invariants(
    attention_mask_model: RSSTransformerNet,
    phase_name: str,
    state_array: np.ndarray,
) -> None:
    state = GameState.from_array(state_array, NUM_PLAYERS)
    phase_id = int(get_decision_phase_py(state))
    x = _token_buffer_for_state(state)

    actual = attention_mask_model._attention_mask(x)[0, 0, 0].cpu()
    expected = _expected_attention_mask(attention_mask_model, state, phase_id)

    assert torch.equal(actual, expected), (
        f"{phase_name} attention mask mismatch: "
        f"{_format_mask_diff(actual=actual, expected=expected, labels=_token_labels(attention_mask_model))}"
    )


@pytest.mark.parametrize(("phase_name", "state_array"), _phase_state_cases())
def test_masked_tokens_do_not_affect_visible_trunk_outputs(
    attention_mask_model: RSSTransformerNet,
    phase_name: str,
    state_array: np.ndarray,
) -> None:
    state = GameState.from_array(state_array, NUM_PLAYERS)
    phase_id = int(get_decision_phase_py(state))
    x = _token_buffer_for_state(state)
    expected = _expected_attention_mask(attention_mask_model, state, phase_id)
    labels = _token_labels(attention_mask_model)

    with torch.no_grad():
        projected = attention_mask_model._project_tokens(x)
        attn_mask = attention_mask_model._attention_mask(x)
        baseline = _run_trunk_from_projected(
            attention_mask_model,
            projected,
            attn_mask,
        )

        perturb = torch.linspace(
            -3.0,
            3.0,
            attention_mask_model.cfg.d_model,
            dtype=projected.dtype,
        ).view(1, 1, -1)

        visible_indices = torch.nonzero(expected, as_tuple=False).flatten()
        visible_probe = int(visible_indices[0])
        visible_perturbed = projected.clone()
        visible_perturbed[:, visible_probe:visible_probe + 1, :] += perturb
        visible_out = _run_trunk_from_projected(
            attention_mask_model,
            visible_perturbed,
            attn_mask,
        )
        visible_rows_except_probe = visible_indices[visible_indices != visible_probe]
        visible_delta = (
            visible_out[:, visible_rows_except_probe, :]
            - baseline[:, visible_rows_except_probe, :]
        ).abs().max()
        assert visible_delta > 1e-5, (
            f"{phase_name}: perturbing visible key {labels[visible_probe]} "
            "did not affect any other visible trunk output; test model is not sensitive"
        )

        for token_idx in torch.nonzero(~expected, as_tuple=False).flatten().tolist():
            perturbed = projected.clone()
            perturbed[:, token_idx:token_idx + 1, :] += perturb
            out = _run_trunk_from_projected(
                attention_mask_model,
                perturbed,
                attn_mask,
            )
            max_visible_delta = (
                out[:, visible_indices, :] - baseline[:, visible_indices, :]
            ).abs().max()
            assert max_visible_delta <= 1e-6, (
                f"{phase_name}: masked key {token_idx}:{labels[token_idx]} "
                f"changed visible trunk output by {float(max_visible_delta)}"
            )


def test_policy_layout_matches_phase_action_sizes(model: RSSTransformerNet) -> None:
    model._validate_policy_layout()


def test_projection_widths_consume_declared_token_features(model: RSSTransformerNet) -> None:
    start = model._token_feature_start
    assert model.player_proj.in_features == int(TokenWidth.TW_PLAYER) - start
    assert model.corp_proj.in_features == int(TokenWidth.TW_CORP) - start
    assert model.company_proj.in_features == int(TokenWidth.TW_COMPANY) - start
    assert model.fi_proj.in_features == int(TokenWidth.TW_FI) - start
    assert model.market_info_proj.in_features == int(TokenWidth.TW_MARKET_INFO) - start
    assert model.global_info_proj.in_features == int(TokenWidth.TW_GLOBAL_INFO) - start
    assert model.invest_proj.in_features == int(TokenWidth.TW_INVEST) - start
    assert model.auction_proj.in_features == int(TokenWidth.TW_AUCTION) - start
    assert model.dividend_proj.in_features == int(TokenWidth.TW_DIVIDEND) - start
    assert model.issue_proj.in_features == int(TokenWidth.TW_ISSUE) - start
    assert model.par_proj.in_features == int(TokenWidth.TW_PAR) - start
    assert model.acq_offer_proj.in_features == int(TokenWidth.TW_ACQ_OFFER) - start
    assert model.acq_price_proj.in_features == int(TokenWidth.TW_ACQ_PRICE) - start


def test_type_embedding_table_matches_token_layout(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    type_ids = model._type_ids

    assert tuple(type_ids.shape) == (cfg.num_tokens,)
    assert tuple(model.type_embeds.weight.shape) == (
        int(type_ids.max().item()) + 1,
        cfg.d_model,
    )
    assert len(torch.unique(type_ids)) == model.type_embeds.num_embeddings
    assert torch.equal(
        type_ids[model._company_slice],
        type_ids.new_full(
            (int(GameConstants.NUM_COMPANIES),),
            type_ids[model._company_slice.start],
        ),
    )
    assert torch.equal(
        type_ids[model._corp_slice],
        type_ids.new_full(
            (int(GameConstants.NUM_CORPS),),
            type_ids[model._corp_slice.start],
        ),
    )
    assert torch.equal(
        type_ids[model._player_slice],
        type_ids.new_full((cfg.num_players,), type_ids[model._player_slice.start]),
    )


def test_project_tokens_matches_manual_type_and_corp_embedding_addition(
    model: RSSTransformerNet,
) -> None:
    cfg = model.cfg
    x = torch.randn(2, cfg.num_tokens, cfg.token_dim)

    manual_parts = [
        model.market_info_proj(
            x[
                :,
                model._market_info_idx,
                model._token_feature_start:int(TokenWidth.TW_MARKET_INFO),
            ]
        ).unsqueeze(1),
        model._project_company_tokens(x),
        model._project_fi_token(x).unsqueeze(1),
        model._project_global_info_token(x).unsqueeze(1),
        model.invest_proj(
            x[:, model._invest_idx, model._token_feature_start:int(TokenWidth.TW_INVEST)]
        ).unsqueeze(1),
        model.auction_proj(
            x[
                :,
                model._auction_idx,
                model._token_feature_start:int(TokenWidth.TW_AUCTION),
            ]
        ).unsqueeze(1),
        model.dividend_proj(
            x[
                :,
                model._dividend_idx,
                model._token_feature_start:int(TokenWidth.TW_DIVIDEND),
            ]
        ).unsqueeze(1),
        model.issue_proj(
            x[:, model._issue_idx, model._token_feature_start:int(TokenWidth.TW_ISSUE)]
        ).unsqueeze(1),
        model.par_proj(
            x[:, model._par_idx, model._token_feature_start:int(TokenWidth.TW_PAR)]
        ).unsqueeze(1),
        model.acq_offer_proj(
            x[
                :,
                model._acq_offer_idx,
                model._token_feature_start:int(TokenWidth.TW_ACQ_OFFER),
            ]
        ).unsqueeze(1),
        model.acq_price_proj(
            x[
                :,
                model._acq_price_info_idx,
                model._token_feature_start:int(TokenWidth.TW_ACQ_PRICE),
            ]
        ).unsqueeze(1),
        model._project_corp_tokens(x),
        model._project_player_tokens(x),
    ]
    expected = torch.cat(manual_parts, dim=1)
    expected = expected + model.type_embeds(model._type_ids).to(expected.dtype)

    assert torch.allclose(model._project_tokens(x), expected)


def test_corp_projection_uses_learned_identity_embedding(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    num_corps = int(GameConstants.NUM_CORPS)
    assert tuple(model.corp_id_embed.weight.shape) == (num_corps, cfg.d_model)

    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    projected = model._project_tokens(x)[:, model._corp_slice]

    expected_delta = model.corp_id_embed.weight[1] - model.corp_id_embed.weight[0]
    actual_delta = projected[0, 1] - projected[0, 0]
    assert torch.allclose(actual_delta, expected_delta)


def test_company_and_player_rows_have_no_learned_row_identity(
    model: RSSTransformerNet,
) -> None:
    cfg = model.cfg
    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    projected = model._project_tokens(x)

    company_rows = projected[0, model._company_slice]
    player_rows = projected[0, model._player_slice]

    assert torch.allclose(company_rows, company_rows[:1].expand_as(company_rows))
    assert torch.allclose(player_rows, player_rows[:1].expand_as(player_rows))


def test_removed_additive_embedding_state_is_absent(model: RSSTransformerNet) -> None:
    assert isinstance(model.type_embeds, torch.nn.Embedding)
    assert isinstance(model.corp_id_embed, torch.nn.Embedding)

    removed_names = [
        "company_id_embed",
        "player_id_embed",
        "active_player_gate",
        "active_corp_gate",
        "active_company_gate",
        "company_owned_by_gate",
        "company_ownership_gate",
        "corp_president_gate",
        "share_ownership_gate",
        "phase_embed",
        "pass_embeds",
    ]
    for name in removed_names:
        assert not hasattr(model, name), name


def test_selected_entity_flags_are_projected_only_on_selected_rows(
    model: RSSTransformerNet,
) -> None:
    cfg = model.cfg
    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x_selected = x.clone()

    active_company = 0
    active_corp = 1
    active_player = 2
    x_selected[
        :, model._company_slice.start + active_company, model._is_selected_offset
    ] = 1.0
    x_selected[
        :, model._corp_slice.start + active_corp, model._is_selected_offset
    ] = 1.0
    x_selected[
        :, model._player_slice.start + active_player, model._is_selected_offset
    ] = 1.0

    delta = model._project_tokens(x_selected) - model._project_tokens(x)
    zero = torch.zeros(cfg.d_model, dtype=delta.dtype)

    assert torch.allclose(delta[0, model._invest_idx], zero)
    assert torch.allclose(delta[0, model._company_slice.start + 1], zero)
    assert torch.allclose(delta[0, model._corp_slice.start], zero)
    assert torch.allclose(delta[0, model._player_slice.start], zero)

    assert torch.allclose(
        delta[0, model._company_slice.start + active_company],
        model.company_proj.weight[:, 0],
    )
    assert torch.allclose(
        delta[0, model._corp_slice.start + active_corp],
        model.corp_proj.weight[:, 0],
    )
    assert torch.allclose(
        delta[0, model._player_slice.start + active_player],
        model.player_proj.weight[:, 0],
    )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for device mismatch test")
def test_forward_rejects_relations_on_different_device() -> None:
    model = RSSTransformerNet(TransformerConfig(num_players=NUM_PLAYERS)).to(torch.device("cuda"))
    cfg = model.cfg
    x = torch.randn(1, cfg.num_tokens, cfg.token_dim, device=torch.device("cuda"))
    lut = build_action_lut()
    legal_mask = torch.zeros(1, U_DIM, dtype=torch.bool, device=torch.device("cuda"))
    legal_mask[0, lut[0, : int(PHASE_ACTION_SIZES[0])].to(torch.device("cuda"))] = True
    relations = torch.zeros(
        1,
        NUM_ATTENTION_RELATIONS,
        cfg.num_tokens,
        cfg.num_tokens,
        dtype=torch.uint8,
    )

    with pytest.raises(AssertionError, match="relations device"):
        model(x, legal_mask, relations)
