from __future__ import annotations

import pytest
import torch

from core.data import PHASE_ACTION_SIZES
from core.resnet_data import get_resnet_vector_size
from nn.resnet import (
    RSSResNet,
    RSSResNetConfig,
    count_parameters,
    parameter_breakdown,
    policy_head_breakdown,
)
from nn.transformer import UNIFIED_LOGIT_DIM, build_action_lut


def _small_model(num_players: int = 3) -> RSSResNet:
    return RSSResNet(
        RSSResNetConfig(
            num_players=num_players,
            input_dim=get_resnet_vector_size(num_players),
            hidden_dim=32,
            num_blocks=1,
        )
    )


def _head_final_linear(head: torch.nn.Sequential) -> torch.nn.Linear:
    final = head[-1]
    assert isinstance(final, torch.nn.Linear)
    return final


def _assert_head_architecture(
    head: torch.nn.Sequential,
    hidden_dim: int,
    output_dim: int,
) -> None:
    assert len(head) == 5
    assert isinstance(head[0], torch.nn.Linear)
    assert head[0].in_features == hidden_dim
    assert head[0].out_features == hidden_dim
    assert isinstance(head[1], torch.nn.GELU)
    assert isinstance(head[2], torch.nn.Linear)
    assert head[2].in_features == hidden_dim
    assert head[2].out_features == hidden_dim
    assert isinstance(head[3], torch.nn.GELU)
    final = _head_final_linear(head)
    assert final.in_features == hidden_dim
    assert final.out_features == output_dim


def test_resnet_policy_heads_match_phase_action_sizes() -> None:
    model = _small_model()

    assert len(model.policy_heads) == len(PHASE_ACTION_SIZES)
    for head in model.policy_heads:
        assert isinstance(head, torch.nn.Sequential)
    head_widths = [_head_final_linear(head).out_features for head in model.policy_heads]
    assert head_widths == [int(size) for size in PHASE_ACTION_SIZES]
    assert sum(head_widths) == int(UNIFIED_LOGIT_DIM)


def test_resnet_policy_and_value_heads_share_mlp_architecture() -> None:
    model = _small_model()

    for phase_size, head in zip(PHASE_ACTION_SIZES, model.policy_heads, strict=True):
        assert isinstance(head, torch.nn.Sequential)
        _assert_head_architecture(head, model.cfg.hidden_dim, int(phase_size))

    value_body = model.value_head[:-1]
    assert isinstance(value_body, torch.nn.Sequential)
    _assert_head_architecture(value_body, model.cfg.hidden_dim, model.cfg.num_players)
    assert isinstance(model.value_head[-1], torch.nn.Tanh)


def test_resnet_parameter_breakdowns_sum_to_total() -> None:
    model = _small_model()

    total = count_parameters(model)
    by_category = parameter_breakdown(model)
    by_policy_phase = policy_head_breakdown(model)

    assert set(by_category) == {
        "Input preprocessing",
        "Residual blocks",
        "Trunk norm",
        "Policy heads",
        "Value head",
    }
    assert sum(by_category.values()) == total
    assert sum(by_policy_phase.values()) == by_category["Policy heads"]
    assert all(count > 0 for count in by_category.values())
    assert all(count > 0 for count in by_policy_phase.values())


def test_resnet_preprocess_has_no_input_layer_norm() -> None:
    model = _small_model()

    assert isinstance(model.input_preprocess[0], torch.nn.Linear)
    assert model.input_preprocess[0].in_features == model.cfg.input_dim
    assert not any(
        isinstance(layer, torch.nn.LayerNorm)
        and layer.normalized_shape == (model.cfg.input_dim,)
        for layer in model.input_preprocess
    )


def test_resnet_policy_heads_concat_all_phase_blocks() -> None:
    model = _small_model()
    model.eval()

    expected_blocks: list[torch.Tensor] = []
    next_value = 0.0
    with torch.no_grad():
        for head in model.policy_heads:
            assert isinstance(head, torch.nn.Sequential)
            final = _head_final_linear(head)
            width = final.out_features
            block = torch.arange(
                next_value,
                next_value + width,
                dtype=final.bias.dtype,
            )
            final.weight.zero_()
            final.bias.copy_(block)
            expected_blocks.append(block)
            next_value += width

    x = torch.randn(2, model.cfg.input_dim)
    legal_mask = torch.ones(2, int(UNIFIED_LOGIT_DIM), dtype=torch.bool)

    with torch.inference_mode():
        policy_logits, _values = model(x, legal_mask)

    expected = torch.cat(expected_blocks).expand_as(policy_logits)
    assert torch.equal(policy_logits, expected)


def test_resnet_forward_shapes_masking_and_value_range() -> None:
    model = _small_model()
    model.eval()
    batch = 2
    x = torch.randn(batch, model.cfg.input_dim)
    legal_mask = torch.zeros(batch, int(UNIFIED_LOGIT_DIM), dtype=torch.bool)
    lut = build_action_lut()
    legal_mask[0, lut[0, : int(PHASE_ACTION_SIZES[0])]] = True
    legal_mask[1, lut[1, : int(PHASE_ACTION_SIZES[1])]] = True

    with torch.inference_mode():
        policy_logits, values = model(x, legal_mask)

    assert policy_logits.shape == (batch, int(UNIFIED_LOGIT_DIM))
    assert values.shape == (batch, model.cfg.num_players)
    assert torch.isfinite(policy_logits[legal_mask]).all()
    assert torch.equal(
        policy_logits[~legal_mask],
        torch.full_like(policy_logits[~legal_mask], -1e9),
    )
    assert torch.isfinite(values).all()
    assert torch.all(values >= -1.0)
    assert torch.all(values <= 1.0)


def test_resnet_forward_rejects_wrong_vector_width() -> None:
    model = _small_model()
    x = torch.randn(1, model.cfg.input_dim - 1)
    legal_mask = torch.ones(1, int(UNIFIED_LOGIT_DIM), dtype=torch.bool)

    with pytest.raises(AssertionError, match="input width"):
        model(x, legal_mask)


def test_resnet_forward_rejects_rank3_input() -> None:
    model = _small_model()
    x = torch.randn(1, 1, model.cfg.input_dim)
    legal_mask = torch.ones(1, int(UNIFIED_LOGIT_DIM), dtype=torch.bool)

    with pytest.raises(AssertionError, match="rank 2"):
        model(x, legal_mask)


def test_resnet_forward_rejects_non_bool_legal_mask() -> None:
    model = _small_model()
    x = torch.randn(1, model.cfg.input_dim)
    legal_mask = torch.ones(1, int(UNIFIED_LOGIT_DIM), dtype=torch.uint8)

    with pytest.raises(AssertionError, match="legal_mask must be bool"):
        model(x, legal_mask)
