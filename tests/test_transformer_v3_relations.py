"""Input relation messages: routing, degree scaling, sparse parity, and training."""

import pytest
import torch

from core.attention_relations import (
    ATTENTION_RELATION_COORD_WIDTH,
    MAX_ATTENTION_RELATION_EDGES,
    NUM_ATTENTION_RELATIONS,
    AttentionRelation,
)
from nn import _load_model_module
from nn.policy_layout import UNIFIED_LOGIT_DIM


@pytest.fixture
def model():
    torch.manual_seed(42)
    module = _load_model_module("nn/transformer-v3.py")
    return module.RSSTransformerNet(module.TransformerConfig(
        num_players=5, d_model=32, num_heads=4, num_layers=1,
    ))


def _inputs(model, device="cpu"):
    n = model.cfg.num_tokens
    tokens = torch.randn(2, n, model.cfg.d_model, device=device)
    visible = torch.ones(2, n, dtype=torch.bool, device=device)
    visible[0, -2:] = False  # Mixed 3p/5p batch.
    dense = torch.zeros(2, NUM_ATTENTION_RELATIONS, n, n, dtype=torch.uint8, device=device)
    coords = torch.zeros(
        2, MAX_ATTENTION_RELATION_EDGES, ATTENTION_RELATION_COORD_WIDTH,
        dtype=torch.uint8, device=device,
    )
    c, p, a, fi = model._corp_slice.start, model._player_slice.start, 1, model._fi_idx
    edges = [
        (AttentionRelation.CORP_OWNS_COMPANY, c, a),
        (AttentionRelation.COMPANY_OWNED_BY_CORP, a, c),
        (AttentionRelation.PLAYER_OWNS_COMPANY, p, a + 1),
        (AttentionRelation.COMPANY_OWNED_BY_PLAYER, a + 1, p),
        (AttentionRelation.FI_OWNS_COMPANY, fi, a + 2),
        (AttentionRelation.COMPANY_OWNED_BY_FI, a + 2, fi),
        (AttentionRelation.PLAYER_OWNS_CORP_SHARES, p, c),
        (AttentionRelation.CORP_HAS_PLAYER_SHAREHOLDER, c, p),
        (AttentionRelation.PLAYER_PRESIDENT_OF_CORP, p, c),
        (AttentionRelation.CORP_PRESIDENT_PLAYER, c, p),
        (AttentionRelation.CORP_OWNS_COMPANY, c, a + 3),
    ]
    for i, (r, query, key) in enumerate(edges):
        dense[:, r, query, key] = 1
        coords[:, i] = torch.tensor([r, query, key], dtype=torch.uint8, device=device)
    return tokens, visible, dense, coords


def test_zero_edges_are_identity_and_unrelated_tokens_are_unchanged(model):
    tokens, visible, dense, coords = _inputs(model)
    mixing = model.relation_input_mixing
    torch.testing.assert_close(mixing(tokens, visible, dense * 0, None), tokens, rtol=0, atol=0)
    empty_ctx = model._prepare_sparse_relation_context(coords * 0)
    torch.testing.assert_close(mixing(tokens, visible, None, empty_ctx), tokens, rtol=0, atol=0)
    actual = mixing(tokens, visible, dense, None)
    recipients = dense.bool().any(dim=1).any(dim=-1) & visible
    torch.testing.assert_close(actual[~recipients], tokens[~recipients], rtol=0, atol=0)
    assert torch.all((actual - tokens)[recipients].abs().sum(-1) > 0)


def test_degree_scaling_and_independent_gains(model):
    tokens, visible, dense, _ = _inputs(model)
    mixing = model.relation_input_mixing
    r = int(AttentionRelation.CORP_OWNS_COMPANY)
    receiver = model._corp_slice.start
    dense.zero_()
    dense[:, r, receiver, 1] = 1
    tokens[:, 2] = tokens[:, 1]
    one = mixing(tokens, visible, dense, None) - tokens
    dense[:, r, receiver, 2] = 1
    two = mixing(tokens, visible, dense, None) - tokens
    torch.testing.assert_close(two, one * (2 ** 0.5), rtol=1e-3, atol=3e-7)

    # A different relation's gain does not affect this neighborhood.
    with torch.no_grad():
        mixing.relation_gains[r + 1] = 3
    torch.testing.assert_close(mixing(tokens, visible, dense, None) - tokens, two)
    with torch.no_grad():
        mixing.relation_gains[r] *= 2
    torch.testing.assert_close(
        mixing(tokens, visible, dense, None) - tokens, two * 2, rtol=1e-3, atol=3e-7,
    )
    with torch.no_grad():
        mixing.relation_gains[r] = 0
    torch.testing.assert_close(mixing(tokens, visible, dense, None), tokens, rtol=0, atol=0)


@pytest.mark.parametrize("sparse", [False, True])
def test_invisible_sources_and_recipients_do_not_communicate(model, sparse):
    tokens, visible, dense, coords = _inputs(model)
    # Hide an existing source and receiver, leaving real edges to exercise masking.
    visible[:, 1] = False
    visible[:, model._corp_slice.start] = False
    ctx = model._prepare_sparse_relation_context(coords) if sparse else None
    flags = None if sparse else dense
    mixing = model.relation_input_mixing
    actual = mixing(tokens, visible, flags, ctx)
    changed = tokens.clone()
    changed[~visible] = torch.randn_like(changed[~visible]) * 100
    perturbed = mixing(changed, visible, flags, ctx)
    torch.testing.assert_close(actual[visible], perturbed[visible], rtol=0, atol=0)
    torch.testing.assert_close(actual[~visible], tokens[~visible], rtol=0, atol=0)


def test_mixing_is_equivariant_to_entity_reordering(model):
    tokens, visible, dense, _ = _inputs(model)
    permutation = torch.randperm(model.cfg.num_tokens)
    mixing = model.relation_input_mixing
    actual = mixing(
        tokens[:, permutation], visible[:, permutation],
        dense[:, :, permutation][:, :, :, permutation], None,
    )
    expected = mixing(tokens, visible, dense, None)[:, permutation]
    torch.testing.assert_close(actual, expected)


def test_dense_sparse_messages_and_parameter_gradients_match(model):
    tokens, visible, dense, coords = _inputs(model)
    tokens.requires_grad_()
    mixing = model.relation_input_mixing
    params = [tokens, *mixing.parameters()]
    target = torch.randn_like(tokens)
    dense_out = mixing(tokens, visible, dense, None)
    dense_grads = torch.autograd.grad((dense_out * target).sum(), params)
    sparse_out = mixing(tokens, visible, None, model._prepare_sparse_relation_context(coords))
    sparse_grads = torch.autograd.grad((sparse_out * target).sum(), params)
    torch.testing.assert_close(dense_out, sparse_out)
    for dense_grad, sparse_grad in zip(dense_grads, sparse_grads):
        torch.testing.assert_close(dense_grad, sparse_grad, rtol=1e-4, atol=1e-6)
        assert torch.isfinite(dense_grad).all() and dense_grad.abs().sum() > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward_backward_and_sparse_parity(model, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = model.to(device)
    _, visible, dense, coords = _inputs(model, device)
    x = torch.randn(2, model.cfg.num_tokens, model.cfg.token_dim, device=device)
    x[:, :, 0] = visible
    legal = torch.ones(2, UNIFIED_LOGIT_DIM, dtype=torch.bool, device=device)
    # CUDA exercises the eval/trainer bf16 path, including sparse scatter backward.
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        logits, values = model(x, legal, dense)
        sparse_logits, sparse_values = model(x, legal, coords)
        loss = sparse_logits.square().mean() + sparse_values.float().square().mean()
    rtol, atol = (0.03, 3e-3) if device == "cuda" else (1e-4, 1e-6)
    torch.testing.assert_close(logits, sparse_logits, rtol=rtol, atol=atol)
    torch.testing.assert_close(values, sparse_values, rtol=rtol, atol=atol)
    loss.backward()
    for parameter in model.relation_input_mixing.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("sparse", [False, True])
def test_compiled_cuda_forward(model, sparse):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = model.cuda().eval()
    _, visible, dense, coords = _inputs(model, "cuda")
    x = torch.randn(2, model.cfg.num_tokens, model.cfg.token_dim, device="cuda")
    x[:, :, 0] = visible
    legal = torch.ones(2, UNIFIED_LOGIT_DIM, dtype=torch.bool, device="cuda")
    relations = coords if sparse else dense
    compiled = torch.compile(model, fullgraph=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model(x, legal, relations)
        actual = compiled(x, legal, relations)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0.03, atol=3e-3)
