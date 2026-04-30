from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from core.actions import get_decision_phase_py
from core.attention_relations import (
    ATTENTION_RELATION_COORD_WIDTH,
    MAX_ATTENTION_RELATION_EDGES,
    NUM_ATTENTION_RELATIONS,
    AttentionRelation,
)
from nn.transformer import (
    UNIFIED_LOGIT_DIM,
    RSSTransformerNet,
    TransformerBlock,
    TransformerConfig,
    build_action_lut,
)
from core.data import (
    ALL_PAR_PRICES,
    PY_COMPANY_PRICE_DIVISOR,
    DecisionPhase,
    GameConstants,
    PHASE_ACTION_SIZES,
)
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


def _zero_relations(
    model: RSSTransformerNet,
    batch_size: int,
    *,
    dtype: torch.dtype = torch.uint8,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.zeros(
        batch_size,
        NUM_ATTENTION_RELATIONS,
        model.cfg.num_tokens,
        model.cfg.num_tokens,
        dtype=dtype,
        device=device,
    )


def _zero_relation_coords(
    batch_size: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.zeros(
        batch_size,
        MAX_ATTENTION_RELATION_EDGES,
        ATTENTION_RELATION_COORD_WIDTH,
        dtype=torch.uint8,
        device=device,
    )


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
            assert isinstance(block, TransformerBlock)
            torch.nn.init.trunc_normal_(block.out_proj.weight, std=0.02)
            torch.nn.init.trunc_normal_(block.ffn_down.weight, std=0.02)
            d_model = model.cfg.d_model
            phase_mod = block.phase_mod
            assert phase_mod is not None
            phase_mod.weight[:, 2 * d_model:3 * d_model] = 1.0
            phase_mod.weight[:, 5 * d_model:6 * d_model] = 1.0
    model.eval()
    return model


def test_forward_rejects_wrong_legal_mask_shape(model: RSSTransformerNet, valid_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, _ = valid_inputs
    wrong_mask = torch.zeros(1, U_DIM, dtype=torch.bool)
    relations = _zero_relations(model, x.shape[0])

    with pytest.raises(AssertionError, match="legal_mask shape"):
        model(x, wrong_mask, relations)


def test_forward_rejects_wrong_num_tokens(model: RSSTransformerNet, valid_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, legal_mask = valid_inputs
    wrong_x = torch.randn(x.shape[0], x.shape[1] - 1, x.shape[2])
    relations = _zero_relations(model, x.shape[0])

    with pytest.raises(AssertionError, match="x shape"):
        model(wrong_x, legal_mask, relations)


def test_forward_rejects_wrong_token_dim(model: RSSTransformerNet, valid_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, legal_mask = valid_inputs
    wrong_x = torch.randn(x.shape[0], x.shape[1], x.shape[2] - 1)
    relations = _zero_relations(model, x.shape[0])

    with pytest.raises(AssertionError, match="x shape"):
        model(wrong_x, legal_mask, relations)


def test_forward_accepts_relation_planes(model: RSSTransformerNet, valid_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, legal_mask = valid_inputs
    uint8_relations = _zero_relations(model, x.shape[0])
    bool_relations = _zero_relations(model, x.shape[0], dtype=torch.bool)

    logits_with_uint8, values_with_uint8 = model(x, legal_mask, uint8_relations)
    logits_with_bool, values_with_bool = model(x, legal_mask, bool_relations)

    assert torch.allclose(logits_with_uint8, logits_with_bool)
    assert torch.allclose(values_with_uint8, values_with_bool)


def test_forward_requires_relation_planes(model: RSSTransformerNet, valid_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, legal_mask = valid_inputs

    with pytest.raises(TypeError):
        model(x, legal_mask)  # type: ignore[call-arg]


def test_forward_rejects_wrong_relation_shape(model: RSSTransformerNet, valid_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, legal_mask = valid_inputs
    relations = _zero_relations(model, x.shape[0])[:, :, :, :-1]

    with pytest.raises(AssertionError, match="relations shape"):
        model(x, legal_mask, relations)


def test_forward_rejects_wrong_relation_dtype(model: RSSTransformerNet, valid_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, legal_mask = valid_inputs
    relations = _zero_relations(model, x.shape[0], dtype=torch.float32)

    with pytest.raises(AssertionError, match="dense relation planes must be bool or uint8"):
        model(x, legal_mask, relations)


def test_relation_bias_multiplier_shape_and_zero_init() -> None:
    model = RSSTransformerNet(
        TransformerConfig(
            num_players=NUM_PLAYERS,
            d_model=48,
            num_heads=3,
            num_layers=2,
            ff_mult=2.0,
        )
    )

    assert tuple(model.relation_bias_mult.shape) == (
        model.cfg.num_layers,
        model.cfg.num_heads,
        NUM_ATTENTION_RELATIONS,
    )
    assert torch.count_nonzero(model.relation_bias_mult).item() == 0


def test_relation_attention_bias_combines_planes_per_layer_and_head() -> None:
    model = RSSTransformerNet(
        TransformerConfig(
            num_players=NUM_PLAYERS,
            d_model=48,
            num_heads=3,
            num_layers=2,
            ff_mult=2.0,
        )
    )
    cfg = model.cfg
    relation_a = int(AttentionRelation.CORP_OWNS_COMPANY)
    relation_b = int(AttentionRelation.PLAYER_OWNS_COMPANY)
    relation_flags = torch.zeros(
        1,
        NUM_ATTENTION_RELATIONS,
        cfg.num_tokens,
        cfg.num_tokens,
    )
    relation_flags[0, relation_a, 2, 3] = 1.0
    relation_flags[0, relation_b, 4, 5] = 1.0
    ref = torch.empty(1, cfg.num_tokens, cfg.d_model)

    with torch.no_grad():
        model.relation_bias_mult.zero_()
        model.relation_bias_mult[1, 0, relation_a] = 2.5
        model.relation_bias_mult[1, 2, relation_b] = -1.25

    bias = model._relation_attention_bias(relation_flags, 1, ref)

    assert tuple(bias.shape) == (
        1,
        cfg.num_heads,
        cfg.num_tokens,
        cfg.num_tokens,
    )
    assert bias[0, 0, 2, 3].item() == pytest.approx(2.5)
    assert bias[0, 2, 4, 5].item() == pytest.approx(-1.25)
    assert bias[0, 1].abs().sum().item() == pytest.approx(0.0)


def test_sparse_relation_attention_bias_matches_dense_planes_with_duplicates() -> None:
    model = RSSTransformerNet(
        TransformerConfig(
            num_players=NUM_PLAYERS,
            d_model=48,
            num_heads=3,
            num_layers=2,
            ff_mult=2.0,
        )
    )
    cfg = model.cfg
    relation_a = int(AttentionRelation.CORP_OWNS_COMPANY)
    relation_b = int(AttentionRelation.PLAYER_OWNS_COMPANY)
    relation_flags = torch.zeros(
        1,
        NUM_ATTENTION_RELATIONS,
        cfg.num_tokens,
        cfg.num_tokens,
    )
    relation_flags[0, relation_a, 2, 3] = 1.0
    relation_flags[0, relation_b, 2, 3] = 1.0
    relation_flags[0, relation_a, 4, 5] = 1.0
    relation_coords = _zero_relation_coords(1)
    relation_coords[0, 0] = torch.tensor([relation_a, 2, 3], dtype=torch.uint8)
    relation_coords[0, 1] = torch.tensor([relation_b, 2, 3], dtype=torch.uint8)
    relation_coords[0, 2] = torch.tensor([relation_a, 4, 5], dtype=torch.uint8)
    ref = torch.empty(1, cfg.num_tokens, cfg.d_model)

    with torch.no_grad():
        model.relation_bias_mult.zero_()
        model.relation_bias_mult[1, 0, relation_a] = 2.5
        model.relation_bias_mult[1, 0, relation_b] = 0.75
        model.relation_bias_mult[1, 2, relation_a] = -1.25

    dense_bias = model._relation_attention_bias(relation_flags, 1, ref)
    sparse_ctx = model._prepare_sparse_relation_context(relation_coords)
    sparse_bias = model._sparse_relation_attention_bias(sparse_ctx, 1, ref)

    assert torch.allclose(sparse_bias, dense_bias)
    assert sparse_bias[0, 0, 2, 3].item() == pytest.approx(3.25)
    assert sparse_bias[0, 2, 4, 5].item() == pytest.approx(-1.25)


def test_forward_accepts_sparse_relation_coords(
    attention_mask_model: RSSTransformerNet,
    valid_inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    model = attention_mask_model
    x, legal_mask = valid_inputs
    relation_id = int(AttentionRelation.CORP_OWNS_COMPANY)
    dense_relations = _zero_relations(model, x.shape[0])
    dense_relations[0, relation_id, 2, 3] = 1
    dense_relations[1, relation_id, 4, 5] = 1
    sparse_relations = _zero_relation_coords(x.shape[0])
    sparse_relations[0, 0] = torch.tensor([relation_id, 2, 3], dtype=torch.uint8)
    sparse_relations[1, 0] = torch.tensor([relation_id, 4, 5], dtype=torch.uint8)

    with torch.no_grad():
        model.relation_bias_mult.zero_()
        model.relation_bias_mult[:, :, relation_id] = 0.5

    logits_with_dense, values_with_dense = model(x, legal_mask, dense_relations)
    logits_with_sparse, values_with_sparse = model(x, legal_mask, sparse_relations)

    assert torch.allclose(logits_with_sparse, logits_with_dense)
    assert torch.allclose(values_with_sparse, values_with_dense)


def test_phase_ids_use_global_info_decision_phase_onehot(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    phase_a = int(DecisionPhase.DPHASE_INVEST)
    phase_b = int(DecisionPhase.DPHASE_PAR)
    x = torch.zeros(2, cfg.num_tokens, cfg.token_dim)
    x[0, model._global_info_idx, 1 + phase_a] = 1.0
    x[1, model._global_info_idx, 1 + phase_b] = 1.0

    phase_ids = model._phase_ids(x)
    expected = torch.tensor([phase_a, phase_b])

    assert torch.equal(phase_ids, expected)


def test_phase_modulation_is_zero_initialized() -> None:
    model = RSSTransformerNet(
        TransformerConfig(
            num_players=NUM_PLAYERS,
            d_model=48,
            num_heads=3,
            num_layers=2,
            ff_mult=2.0,
        )
    )
    cfg = model.cfg

    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        phase_mod = block.phase_mod
        assert phase_mod is not None
        assert isinstance(phase_mod, torch.nn.Embedding)
        assert phase_mod.num_embeddings == len(DecisionPhase)
        assert phase_mod.embedding_dim == 6 * cfg.d_model
        assert torch.count_nonzero(phase_mod.weight).item() == 0
        assert torch.count_nonzero(block.out_proj.weight).item() > 0
        assert torch.count_nonzero(block.ffn_down.weight).item() > 0


def test_transformer_trunk_linears_are_biasless() -> None:
    model = RSSTransformerNet(
        TransformerConfig(
            num_players=NUM_PLAYERS,
            d_model=48,
            num_heads=3,
            num_layers=2,
            ff_mult=2.0,
        )
    )

    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        assert block.qkv_proj.bias is None
        assert block.out_proj.bias is None
        assert block.ffn_gate.bias is None
        assert block.ffn_up.bias is None
        assert block.ffn_down.bias is None


def test_disabled_phase_conditioning_omits_phase_mod_and_skips_phase_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RSSTransformerNet(
        TransformerConfig(
            num_players=NUM_PLAYERS,
            d_model=48,
            num_heads=3,
            num_layers=2,
            ff_mult=2.0,
            phase_conditioning=False,
        )
    )
    cfg = model.cfg

    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        assert block.phase_mod is None
    assert all("phase_mod" not in name for name, _ in model.named_parameters())
    assert model.phase_mod_diagnostics() == {}

    def fail_phase_ids(_x: torch.Tensor) -> torch.Tensor:
        raise AssertionError("phase ids should not be read when conditioning is disabled")

    monkeypatch.setattr(model, "_phase_ids", fail_phase_ids)
    x = torch.randn(1, cfg.num_tokens, cfg.token_dim)
    x[:, :, 0] = 1.0
    lut = build_action_lut()
    legal_mask = torch.zeros(1, U_DIM, dtype=torch.bool)
    legal_mask[0, lut[0, : int(PHASE_ACTION_SIZES[0])]] = True
    relations = _zero_relations(model, 1)

    policy_logits, values = model(x, legal_mask, relations)

    assert policy_logits.shape == (1, U_DIM)
    assert values.shape == (1, NUM_PLAYERS)


def test_zero_init_phase_modulation_preserves_block_outputs() -> None:
    torch.manual_seed(789)
    model = RSSTransformerNet(
        TransformerConfig(
            num_players=NUM_PLAYERS,
            d_model=48,
            num_heads=3,
            num_layers=1,
            ff_mult=2.0,
        )
    )
    block = model.blocks[0]
    assert isinstance(block, TransformerBlock)
    with torch.no_grad():
        torch.nn.init.trunc_normal_(block.out_proj.weight, std=0.02)
        torch.nn.init.trunc_normal_(block.ffn_down.weight, std=0.02)

    cfg = model.cfg
    tokens = torch.randn(2, cfg.num_tokens, cfg.d_model)
    attn_mask = torch.ones(2, 1, 1, cfg.num_tokens, dtype=torch.bool)
    relation_bias = torch.zeros(2, cfg.num_heads, cfg.num_tokens, cfg.num_tokens)
    phase_a = torch.tensor([0, 0])
    phase_b = torch.tensor([1, 1])

    out_a = block(tokens, attn_mask, relation_bias, phase_a)
    out_b = block(tokens, attn_mask, relation_bias, phase_b)

    assert torch.allclose(out_a, out_b)
    assert torch.allclose(out_a, tokens)


def test_nonzero_phase_modulation_changes_block_outputs() -> None:
    torch.manual_seed(790)
    model = RSSTransformerNet(
        TransformerConfig(
            num_players=NUM_PLAYERS,
            d_model=48,
            num_heads=3,
            num_layers=1,
            ff_mult=2.0,
        )
    )
    block = model.blocks[0]
    assert isinstance(block, TransformerBlock)
    with torch.no_grad():
        torch.nn.init.trunc_normal_(block.out_proj.weight, std=0.02)
        torch.nn.init.trunc_normal_(block.ffn_down.weight, std=0.02)
        d_model = model.cfg.d_model
        phase_mod = block.phase_mod
        assert phase_mod is not None
        phase_mod.weight[1, 2 * d_model:3 * d_model] = 1.0

    cfg = model.cfg
    tokens = torch.randn(2, cfg.num_tokens, cfg.d_model)
    attn_mask = torch.ones(2, 1, 1, cfg.num_tokens, dtype=torch.bool)
    relation_bias = torch.zeros(2, cfg.num_heads, cfg.num_tokens, cfg.num_tokens)

    out_a = block(tokens, attn_mask, relation_bias, torch.tensor([0, 0]))
    out_b = block(tokens, attn_mask, relation_bias, torch.tensor([1, 1]))

    assert (out_a - out_b).abs().max().item() > 1e-6


def test_nonzero_relation_bias_changes_forward_outputs() -> None:
    torch.manual_seed(456)
    model = RSSTransformerNet(
        TransformerConfig(
            num_players=NUM_PLAYERS,
            d_model=48,
            num_heads=3,
            num_layers=2,
            ff_mult=2.0,
        )
    )
    model.eval()
    with torch.no_grad():
        for block in model.blocks:
            assert isinstance(block, TransformerBlock)
            torch.nn.init.trunc_normal_(block.out_proj.weight, std=0.02)
            d_model = model.cfg.d_model
            phase_mod = block.phase_mod
            assert phase_mod is not None
            phase_mod.weight[0, 2 * d_model:3 * d_model] = 1.0
        relation_id = int(AttentionRelation.PLAYER_OWNS_COMPANY)
        model.relation_bias_mult[0, :, relation_id] = 8.0

    cfg = model.cfg
    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x[:, :, 0] = 1.0
    x[:, model._player_slice, 1] = 0.0
    x[:, model._player_slice.start, 1] = 1.0
    x[:, model._company_slice.start, 1] = 1.0

    lut = build_action_lut()
    legal_mask = torch.zeros(1, U_DIM, dtype=torch.bool)
    legal_mask[0, lut[0, : int(PHASE_ACTION_SIZES[0])]] = True

    zero_relations = torch.zeros(
        1,
        NUM_ATTENTION_RELATIONS,
        cfg.num_tokens,
        cfg.num_tokens,
        dtype=torch.uint8,
    )
    edge_relations = zero_relations.clone()
    edge_relations[
        0,
        relation_id,
        model._player_slice.start,
        model._company_slice.start,
    ] = 1

    logits_without_edge, values_without_edge = model(x, legal_mask, zero_relations)
    logits_with_edge, values_with_edge = model(x, legal_mask, edge_relations)

    max_logit_delta = (logits_with_edge - logits_without_edge).abs().max()
    max_value_delta = (values_with_edge - values_without_edge).abs().max()
    assert max(max_logit_delta.item(), max_value_delta.item()) > 1e-6


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
    phase_ids = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
    relation_flags = tokens.new_zeros(
        tokens.shape[0],
        NUM_ATTENTION_RELATIONS,
        model.cfg.num_tokens,
        model.cfg.num_tokens,
    )
    for layer_idx, block in enumerate(model.blocks):
        relation_bias = model._relation_attention_bias(
            relation_flags,
            layer_idx,
            tokens,
        )
        tokens = block(tokens, attn_mask, relation_bias, phase_ids)
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
    expected = _expected_attention_mask(attention_mask_model, phase_id)

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
    expected = _expected_attention_mask(attention_mask_model, phase_id)
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


def test_price_slot_key_heads_use_fourier_widths_and_slot_embeddings(
    model: RSSTransformerNet,
) -> None:
    bands = model.cfg.price_slot_fourier_bands
    scalar_width = 1 + 2 * bands
    pair_width = 2 * scalar_width

    assert model.dividend_amount_proj.in_features == scalar_width
    assert model.bid_offset_proj.in_features == pair_width
    assert model.acq_price_offset_proj.in_features == pair_width
    assert model.par_price_proj.in_features == pair_width

    assert model.bid_offset_embed.num_embeddings == int(GameConstants.AUCTION_CAP)
    assert model.dividend_amount_embed.num_embeddings == int(
        PHASE_ACTION_SIZES[int(DecisionPhase.DPHASE_DIVIDENDS)]
    )
    assert model.acq_price_offset_embed.num_embeddings == int(
        PHASE_ACTION_SIZES[int(DecisionPhase.DPHASE_ACQ_SELECT_PRICE)]
    )
    assert model.par_price_embed.num_embeddings == int(
        PHASE_ACTION_SIZES[int(DecisionPhase.DPHASE_PAR)]
    )


def test_price_slot_static_features_match_action_semantics(
    model: RSSTransformerNet,
) -> None:
    assert model._dividend_amount_features[0, 0, 0].item() == pytest.approx(0.0)
    assert model._dividend_amount_features[0, -1, 0].item() == pytest.approx(1.0)

    expected_par = torch.tensor(
        [
            [idx / (len(ALL_PAR_PRICES) - 1), price / max(ALL_PAR_PRICES)]
            for idx, price in enumerate(ALL_PAR_PRICES)
        ],
        dtype=model._par_price_features.dtype,
    )
    assert torch.allclose(model._par_price_features, expected_par)

    # ACQ_SELECT_PRICE buffer carries two channels: 0-indexed slot position
    # (feeds the Fourier slot key, matching BID/PAR/DIVIDENDS, AND pairs with
    # engine-side ``max_offset = (high - low) / 50`` for ``remaining_after_offset``),
    # and the price delta added to ``low_price`` to recover the candidate price.
    K = int(PHASE_ACTION_SIZES[int(DecisionPhase.DPHASE_ACQ_SELECT_PRICE)])
    expected_acq = torch.tensor(
        [
            [k / (K - 1), k / float(PY_COMPANY_PRICE_DIVISOR)]
            for k in range(K)
        ],
        dtype=model._acq_price_offset_features.dtype,
    ).view(1, K, 2)
    assert torch.allclose(model._acq_price_offset_features, expected_acq)


def test_slot_fourier_features_keep_raw_scalars_first(model: RSSTransformerNet) -> None:
    features = torch.tensor([[[0.25, 0.5]]], dtype=torch.float32)
    encoded = model._slot_fourier_features(features)

    expected_width = features.shape[-1] * (1 + 2 * model.cfg.price_slot_fourier_bands)
    assert tuple(encoded.shape) == (1, 1, expected_width)
    assert torch.allclose(encoded[..., : features.shape[-1]], features)


def test_price_slot_residual_scale_blends_fourier_keys_and_embeddings() -> None:
    cfg = TransformerConfig(
        num_players=NUM_PLAYERS,
        d_model=48,
        num_heads=3,
        num_layers=1,
        ff_mult=2.0,
        price_slot_residual_scale=0.5,
    )
    model = RSSTransformerNet(cfg)
    assert "price_slot_residual_scale" not in dict(model.named_parameters())

    keys = torch.randn(2, int(GameConstants.AUCTION_CAP), cfg.d_proj)
    learned = model.bid_offset_embed.weight.unsqueeze(0).expand_as(keys)
    actual = model._blend_price_slot_keys(keys, model.bid_offset_embed)
    expected = 0.5 * keys + 0.5 * learned
    assert torch.allclose(actual, expected)

    zero_cfg = TransformerConfig(
        num_players=NUM_PLAYERS,
        d_model=48,
        num_heads=3,
        num_layers=1,
        ff_mult=2.0,
        price_slot_residual_scale=0.0,
    )
    zero_model = RSSTransformerNet(zero_cfg)
    untouched = zero_model._blend_price_slot_keys(keys, zero_model.bid_offset_embed)
    assert torch.equal(untouched, keys)

    one_cfg = TransformerConfig(
        num_players=NUM_PLAYERS,
        d_model=48,
        num_heads=3,
        num_layers=1,
        ff_mult=2.0,
        price_slot_residual_scale=1.0,
    )
    one_model = RSSTransformerNet(one_cfg)
    pure_embedding = one_model._blend_price_slot_keys(keys, one_model.bid_offset_embed)
    expected_embedding = one_model.bid_offset_embed.weight.unsqueeze(0).expand_as(keys)
    assert torch.equal(pure_embedding, expected_embedding)


def test_price_slot_residual_scale_must_be_blend_weight() -> None:
    with pytest.raises(AssertionError, match="price_slot_residual_scale"):
        TransformerConfig(num_players=NUM_PLAYERS, price_slot_residual_scale=-0.1)

    with pytest.raises(AssertionError, match="price_slot_residual_scale"):
        TransformerConfig(num_players=NUM_PLAYERS, price_slot_residual_scale=1.1)


def test_projection_widths_consume_declared_token_features(model: RSSTransformerNet) -> None:
    start = model._token_feature_start
    assert model.player_proj.in_features == model._player_rel_tail_start - start
    assert model.corp_proj.in_features == model._corp_rel_tail_start - start
    assert model.company_proj.in_features == model._company_rel_tail_start - start
    assert model.fi_proj.in_features == model._fi_rel_tail_start - start
    assert model.market_info_proj.in_features == int(TokenWidth.TW_MARKET_INFO) - start
    assert model.global_info_proj.in_features == (
        int(TokenWidth.TW_GLOBAL_INFO) - model._global_info_feature_start
    )
    assert model.invest_proj.in_features == int(TokenWidth.TW_INVEST) - start
    assert model.auction_proj.in_features == int(TokenWidth.TW_AUCTION) - start
    assert model.dividend_proj.in_features == int(TokenWidth.TW_DIVIDEND) - start
    assert model.issue_proj.in_features == int(TokenWidth.TW_ISSUE) - start
    assert model.par_proj.in_features == int(TokenWidth.TW_PAR) - start
    assert model.acq_offer_proj.in_features == int(TokenWidth.TW_ACQ_OFFER) - start
    assert model.acq_price_proj.in_features == int(TokenWidth.TW_ACQ_PRICE) - start


def test_project_tokens_ignores_entity_relation_tails(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x_with_tails = x.clone()

    x_with_tails[
        :,
        model._company_slice,
        model._company_rel_tail_start:int(TokenWidth.TW_COMPANY),
    ] = torch.randn(
        1,
        int(GameConstants.NUM_COMPANIES),
        int(TokenWidth.TW_COMPANY) - model._company_rel_tail_start,
    )
    x_with_tails[
        :,
        model._fi_idx,
        model._fi_rel_tail_start:int(TokenWidth.TW_FI),
    ] = torch.randn(1, int(TokenWidth.TW_FI) - model._fi_rel_tail_start)
    x_with_tails[
        :,
        model._corp_slice,
        model._corp_rel_tail_start:int(TokenWidth.TW_CORP),
    ] = torch.randn(
        1,
        int(GameConstants.NUM_CORPS),
        int(TokenWidth.TW_CORP) - model._corp_rel_tail_start,
    )
    x_with_tails[
        :,
        model._player_slice,
        model._player_rel_tail_start:int(TokenWidth.TW_PLAYER),
    ] = torch.randn(
        1,
        cfg.num_players,
        int(TokenWidth.TW_PLAYER) - model._player_rel_tail_start,
    )

    assert torch.allclose(model._project_tokens(x), model._project_tokens(x_with_tails))


def test_project_tokens_keeps_player_share_amount_features(model: RSSTransformerNet) -> None:
    cfg = model.cfg
    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x_with_shares = x.clone()
    share_feature_start = 15
    share_feature_stop = model._player_rel_tail_start
    x_with_shares[
        :,
        model._player_slice,
        share_feature_start:share_feature_stop,
    ] = torch.randn(
        1,
        cfg.num_players,
        share_feature_stop - share_feature_start,
    )

    delta = model._project_tokens(x_with_shares) - model._project_tokens(x)
    player_delta = delta[:, model._player_slice]

    assert player_delta.abs().max().item() > 0.0
    assert delta[:, :model._player_slice.start].abs().max().item() == pytest.approx(0.0)


def test_project_tokens_ignores_global_info_phase_onehot_when_conditioned(
    model: RSSTransformerNet,
) -> None:
    cfg = model.cfg
    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x_with_phase = x.clone()
    phase_start = 1
    phase_stop = phase_start + len(DecisionPhase)
    x_with_phase[:, model._global_info_idx, phase_start:phase_stop] = torch.randn(
        1,
        phase_stop - phase_start,
    )

    assert torch.allclose(model._project_tokens(x), model._project_tokens(x_with_phase))


def test_project_tokens_keeps_global_info_phase_onehot_without_conditioning() -> None:
    model = RSSTransformerNet(
        TransformerConfig(
            num_players=NUM_PLAYERS,
            d_model=48,
            num_heads=3,
            num_layers=1,
            ff_mult=2.0,
            phase_conditioning=False,
        )
    )
    cfg = model.cfg
    x = torch.zeros(1, cfg.num_tokens, cfg.token_dim)
    x_with_phase = x.clone()
    phase_start = 1
    phase_stop = phase_start + len(DecisionPhase)
    x_with_phase[:, model._global_info_idx, phase_start] = 1.0

    with torch.no_grad():
        model.global_info_proj.weight.zero_()
        model.global_info_proj.bias.zero_()
        model.global_info_proj.weight[:, 0] = 1.0

    assert model.global_info_proj.in_features == int(TokenWidth.TW_GLOBAL_INFO) - phase_start

    delta = model._project_tokens(x_with_phase) - model._project_tokens(x)
    global_delta = delta[:, model._global_info_idx]

    assert global_delta.abs().max().item() > 0.0
    assert delta[:, :model._global_info_idx].abs().max().item() == pytest.approx(0.0)
    assert delta[:, model._global_info_idx + 1:].abs().max().item() == pytest.approx(0.0)


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
            int(type_ids[model._company_slice.start].item()),
        ),
    )
    assert torch.equal(
        type_ids[model._corp_slice],
        type_ids.new_full(
            (int(GameConstants.NUM_CORPS),),
            int(type_ids[model._corp_slice.start].item()),
        ),
    )
    assert torch.equal(
        type_ids[model._player_slice],
        type_ids.new_full(
            (cfg.num_players,),
            int(type_ids[model._player_slice.start].item()),
        ),
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
    relations = _zero_relations(model, 1, device=torch.device("cuda"))

    with pytest.raises(AssertionError, match="legal_mask device"):
        model(x, legal_mask, relations)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for device mismatch test")
def test_forward_rejects_relations_on_different_device() -> None:
    model = RSSTransformerNet(TransformerConfig(num_players=NUM_PLAYERS)).to(torch.device("cuda"))
    cfg = model.cfg
    x = torch.randn(1, cfg.num_tokens, cfg.token_dim, device=torch.device("cuda"))
    lut = build_action_lut()
    legal_mask = torch.zeros(1, U_DIM, dtype=torch.bool, device=torch.device("cuda"))
    legal_mask[0, lut[0, : int(PHASE_ACTION_SIZES[0])].to(torch.device("cuda"))] = True
    relations = _zero_relations(model, 1)

    with pytest.raises(AssertionError, match="relations device"):
        model(x, legal_mask, relations)
