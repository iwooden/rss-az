"""Input relation messages: routing, degree scaling, sparse parity, and training."""

import pytest
import torch

from core.attention_relations import (
    ATTENTION_RELATION_COORD_WIDTH,
    MAX_ATTENTION_RELATION_EDGES,
    NUM_ATTENTION_RELATIONS,
    AttentionRelation,
)
from core.data import COMPANY_NAME_TO_ID
from entities.company import COMPANIES
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
    visible[0, model._company_slice.stop - 1] = False  # Hidden company in the deck.
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
        coords[:, i] = torch.tensor([r, query, key, 1], dtype=torch.uint8, device=device)
    for i, (r, query, key) in enumerate([
        (AttentionRelation.PLAYER_CORP_SHARE_COUNT, p, c),
        (AttentionRelation.CORP_PLAYER_SHARE_COUNT, c, p),
    ], start=len(edges)):
        dense[:, r, query, key] = 5
        coords[:, i] = torch.tensor([r, query, key, 5], dtype=torch.uint8, device=device)
    return tokens, visible, dense, coords


def test_zero_edges_are_identity_and_unrelated_tokens_are_unchanged(model):
    tokens, visible, dense, coords = _inputs(model)
    dense = model._normalize_dense_relations(dense, tokens)
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
    dense = model._normalize_dense_relations(dense, tokens)
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
    flags = None if sparse else model._normalize_dense_relations(dense, tokens)
    mixing = model.relation_input_mixing
    actual = mixing(tokens, visible, flags, ctx)
    changed = tokens.clone()
    changed[~visible] = torch.randn_like(changed[~visible]) * 100
    perturbed = mixing(changed, visible, flags, ctx)
    torch.testing.assert_close(actual[visible], perturbed[visible], rtol=0, atol=0)
    torch.testing.assert_close(actual[~visible], tokens[~visible], rtol=0, atol=0)


def test_mixing_is_equivariant_to_entity_reordering(model):
    tokens, visible, dense, _ = _inputs(model)
    dense = model._normalize_dense_relations(dense, tokens)
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
    static_args = (model._static_company_relations, model._company_slice)
    dense_out = mixing(tokens, visible, model._normalize_dense_relations(dense, tokens), None, *static_args)
    dense_grads = torch.autograd.grad((dense_out * target).sum(), params)
    sparse_out = mixing(tokens, visible, None, model._prepare_sparse_relation_context(coords), *static_args)
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
    with torch.no_grad():
        model.relation_bias_mult.normal_(std=0.3)
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
    with torch.no_grad():
        model.relation_bias_mult.normal_(std=0.3)
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


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_quantity_messages_scale_with_shares_and_count_neighbors(model, sparse, reverse):
    tokens, visible, dense, coords = _inputs(model)
    dense.zero_()
    coords.zero_()
    if reverse:
        r = int(AttentionRelation.CORP_PLAYER_SHARE_COUNT)
        receiver, source = model._corp_slice.start, model._player_slice.start
    else:
        r = int(AttentionRelation.PLAYER_CORP_SHARE_COUNT)
        receiver, source = model._player_slice.start, model._corp_slice.start
    # Zero recipient embeddings keep small message comparisons numerically clean.
    tokens[:, receiver] = 0
    tokens[:, source + 1] = tokens[:, source]

    def message(shares, neighbors):
        dense.zero_()
        coords.zero_()
        for edge in range(neighbors):
            dense[:, r, receiver, source + edge] = shares
            coords[:, edge] = torch.tensor([r, receiver, source + edge, shares], dtype=torch.uint8)
        flags = None if sparse else model._normalize_dense_relations(dense, tokens)
        context = model._prepare_sparse_relation_context(coords) if sparse else None
        return model.relation_input_mixing(tokens, visible, flags, context)[:, receiver]

    one_share = message(1, 1)
    four_shares = message(4, 1)
    two_neighbors = message(4, 2)
    assert one_share.abs().sum() > 0
    torch.testing.assert_close(four_shares, one_share * 4)
    torch.testing.assert_close(two_neighbors, four_shares * (2 ** 0.5))
    torch.testing.assert_close(message(0, 0), torch.zeros_like(one_share), atol=0, rtol=0)


@pytest.mark.parametrize("reverse", [False, True])
def test_attention_adds_independent_presence_presidency_and_normalized_quantity(reverse):
    module = _load_model_module("nn/transformer-v3.py")
    model = module.RSSTransformerNet(module.TransformerConfig(
        num_players=5, d_model=32, num_heads=4, num_layers=1,
    ))
    tokens, _, dense, coords = _inputs(model)
    dense.zero_()
    coords.zero_()
    if reverse:
        relations = (
            AttentionRelation.CORP_HAS_PLAYER_SHAREHOLDER,
            AttentionRelation.CORP_PRESIDENT_PLAYER,
            AttentionRelation.CORP_PLAYER_SHARE_COUNT,
        )
        query, key = model._corp_slice.start, model._player_slice.start
    else:
        relations = (
            AttentionRelation.PLAYER_OWNS_CORP_SHARES,
            AttentionRelation.PLAYER_PRESIDENT_OF_CORP,
            AttentionRelation.PLAYER_CORP_SHARE_COUNT,
        )
        query, key = model._player_slice.start, model._corp_slice.start
    with torch.no_grad():
        for r, coefficient in zip(relations, (2, 3, 7)):
            model.relation_bias_mult[:, :, r] = coefficient
    for shares in range(1, 8):
        for edge, (r, value) in enumerate(zip(relations, (1, 1, shares))):
            dense[:, r, query, key] = value
            coords[:, edge] = torch.tensor([r, query, key, value], dtype=torch.uint8)
        dense_bias = model._relation_attention_bias(
            model._normalize_dense_relations(dense, tokens), 0, tokens,
        )
        sparse_bias = model._sparse_relation_attention_bias(
            model._prepare_sparse_relation_context(coords), 0, tokens,
        )
        torch.testing.assert_close(dense_bias, sparse_bias)
        torch.testing.assert_close(
            dense_bias[:, :, query, key], torch.full((2, 4), float(2 + 3 + shares)),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dense_sparse_normalization_rounds_all_share_counts_identically(dtype):
    module = _load_model_module("nn/transformer-v3.py")
    model = module.RSSTransformerNet(module.TransformerConfig(
        num_players=5, d_model=32, num_heads=4, num_layers=1,
    ))
    tokens, _, dense, coords = _inputs(model)
    dense.zero_()
    coords.zero_()
    r = int(AttentionRelation.PLAYER_CORP_SHARE_COUNT)
    query, key = model._player_slice.start, model._corp_slice.start
    counts = torch.arange(1, 8, dtype=torch.uint8)
    dense[:, r, query, key:key + 7] = counts
    coords[:, :7, 0] = r
    coords[:, :7, 1] = query
    coords[:, :7, 2] = torch.arange(key, key + 7, dtype=torch.uint8)
    coords[:, :7, 3] = counts
    normalized = model._normalize_dense_relations(dense, tokens.to(dtype))
    context = model._prepare_sparse_relation_context(coords)
    torch.testing.assert_close(
        normalized[:, r, query, key:key + 7], context.edge_weights[:, :7].to(dtype),
        rtol=0, atol=0,
    )


def test_static_synergies_match_engine_pairs_without_changing_engine_table(model):
    directed = torch.tensor([
        [company.get_synergy_with(other) for other in range(len(COMPANIES))]
        for company in COMPANIES
    ])
    assert not torch.equal(directed, directed.T)
    assert not ((directed > 0) & (directed.T > 0)).any()
    expected = ((directed > 0) | (directed.T > 0)).float()
    actual = model._static_company_relations[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, actual.T, rtol=0, atol=0)
    assert not actual.diagonal().any()
    for first, second in (("KME", "BME"), ("CDG", "MAD")):
        assert actual[COMPANY_NAME_TO_ID[first], COMPANY_NAME_TO_ID[second]] == 1
    assert actual[COMPANY_NAME_TO_ID["BME"], COMPANY_NAME_TO_ID["MAD"]] == 0
    full = model._static_attention_relations[0]
    torch.testing.assert_close(full[model._company_slice, model._company_slice], actual)
    assert full.sum() == actual.sum()  # No edges to non-company tokens.
    strength = model._static_company_relations[1]
    torch.testing.assert_close(strength, (directed + directed.T).float() / 16, rtol=0, atol=0)
    torch.testing.assert_close(strength, strength.T, rtol=0, atol=0)
    assert strength[COMPANY_NAME_TO_ID["KME"], COMPANY_NAME_TO_ID["BME"]] == 1 / 16
    assert strength[COMPANY_NAME_TO_ID["CDG"], COMPANY_NAME_TO_ID["MAD"]] == 1
    full_strength = model._static_attention_relations[1]
    torch.testing.assert_close(full_strength[model._company_slice, model._company_slice], strength)
    assert full_strength.sum() == strength.sum()
    assert not any(name.startswith("_static_") for name in model.state_dict())


@pytest.mark.parametrize("sparse", [False, True])
def test_static_synergy_bias_is_symmetric_and_independent_per_head(model, sparse):
    tokens, _, dense, coords = _inputs(model)
    coefficients = torch.tensor([[2., -1.], [-3., 4.], [0., 2.], [0.5, 0.]])
    with torch.no_grad():
        model.relation_bias_mult.zero_()
        model.relation_bias_mult[0, :, NUM_ATTENTION_RELATIONS:] = coefficients
    if sparse:
        bias = model._sparse_relation_attention_bias(
            model._prepare_sparse_relation_context(coords * 0), 0, tokens,
        )
    else:
        bias = model._relation_attention_bias(
            model._normalize_dense_relations(dense * 0, tokens), 0, tokens,
        )
    presence, strength = model._static_attention_relations
    expected = (
        coefficients[None, :, 0, None, None] * presence
        + coefficients[None, :, 1, None, None] * strength
    )
    torch.testing.assert_close(bias, expected.expand_as(bias), rtol=0, atol=0)
    torch.testing.assert_close(bias, bias.transpose(-1, -2), rtol=0, atol=0)
    bias.sum().backward()
    grad = model.relation_bias_mult.grad
    assert grad is not None
    assert torch.all(grad[0, :, NUM_ATTENTION_RELATIONS:] > 0)
    assert not grad[0, :, :NUM_ATTENTION_RELATIONS].any()


def test_static_synergy_attention_cannot_read_hidden_company(model):
    tokens, visible, dense, _ = _inputs(model)
    hidden = model._company_slice.start + COMPANY_NAME_TO_ID["KME"]
    visible[:, hidden] = False
    with torch.no_grad():
        model.relation_bias_mult[:, :, NUM_ATTENTION_RELATIONS:] = 3
    bias = model._relation_attention_bias(model._normalize_dense_relations(dense * 0, tokens), 0, tokens)
    block = model.blocks[0]
    expected = block(tokens, visible[:, None, None, :], bias)
    changed = tokens.clone()
    changed[:, hidden] = torch.randn_like(changed[:, hidden]) * 100
    actual = block(changed, visible[:, None, None, :], bias)
    torch.testing.assert_close(actual[visible], expected[visible], rtol=0, atol=0)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("static_relation", [0, 1])
def test_static_synergy_messages_respect_visibility_degree_and_gain(model, sparse, static_relation):
    tokens, visible, dense, coords = _inputs(model)
    dense.zero_()
    coords.zero_()
    visible.zero_()
    receiver, source, second_source = [
        model._company_slice.start + COMPANY_NAME_TO_ID[name]
        for name in ("KME", "BME", "MHE")
    ]
    visible[:, [receiver, source]] = True
    tokens[:, receiver] = 0
    tokens[:, second_source] = tokens[:, source]
    flags = None if sparse else model._normalize_dense_relations(dense, tokens)
    ctx = model._prepare_sparse_relation_context(coords) if sparse else None
    mixing = model.relation_input_mixing
    relation_id = NUM_ATTENTION_RELATIONS + static_relation
    with torch.no_grad():
        mixing.relation_gains[NUM_ATTENTION_RELATIONS:] = 0
        mixing.relation_gains[relation_id] = 0.1

    def mix(inputs):
        return mixing(inputs, visible, flags, ctx, model._static_company_relations, model._company_slice)

    actual = mix(tokens)
    one = actual[:, receiver].clone()
    assert one.abs().sum() > 0
    reverse_tokens = tokens.clone()
    reverse_tokens[:, receiver] = torch.randn_like(tokens[:, receiver])
    assert (mix(reverse_tokens)[:, source] - tokens[:, source]).abs().sum() > 0
    torch.testing.assert_close(actual[~visible], tokens[~visible], rtol=0, atol=0)
    changed = tokens.clone()
    changed[~visible] = torch.randn_like(changed[~visible]) * 100
    torch.testing.assert_close(mix(changed)[visible], actual[visible], rtol=0, atol=0)

    visible[:, second_source] = True
    two = mix(tokens)[:, receiver]
    torch.testing.assert_close(two, one * (2 ** 0.5))
    with torch.no_grad():
        mixing.relation_gains[relation_id] *= -2
    torch.testing.assert_close(mix(tokens)[:, receiver], two * -2)
    with torch.no_grad():
        mixing.relation_gains[relation_id] = 0
    torch.testing.assert_close(mix(tokens), tokens, rtol=0, atol=0)


@pytest.mark.parametrize("sparse", [False, True])
def test_synergy_strength_scales_messages_by_bonus_without_weighted_degree(model, sparse):
    tokens, visible, dense, coords = _inputs(model)
    receiver, large_source, small_source = [
        model._company_slice.start + COMPANY_NAME_TO_ID[name]
        for name in ("CDG", "MAD", "E")  # Bonuses 16 and 8.
    ]
    tokens[:, receiver] = 0
    tokens[:, small_source] = tokens[:, large_source]
    mixing = model.relation_input_mixing
    presence_id, strength_id = NUM_ATTENTION_RELATIONS, NUM_ATTENTION_RELATIONS + 1
    # Equal projections isolate the two relations' different edge weights.
    with torch.no_grad():
        mixing.relation_projs[strength_id].weight.copy_(mixing.relation_projs[presence_id].weight)
    flags = None if sparse else model._normalize_dense_relations(dense * 0, tokens)
    ctx = model._prepare_sparse_relation_context(coords * 0) if sparse else None

    def message(relation_id, neighbors):
        visible.zero_()
        visible[:, [receiver, *neighbors]] = True
        with torch.no_grad():
            mixing.relation_gains[NUM_ATTENTION_RELATIONS:] = 0
            mixing.relation_gains[relation_id] = 0.1
        return mixing(
            tokens, visible, flags, ctx, model._static_company_relations, model._company_slice,
        )[:, receiver]

    presence = message(presence_id, [large_source])
    assert presence.abs().sum() > 0
    torch.testing.assert_close(message(presence_id, [small_source]), presence)
    torch.testing.assert_close(message(strength_id, [large_source]), presence)
    torch.testing.assert_close(message(strength_id, [small_source]), presence / 2)
    torch.testing.assert_close(
        message(strength_id, [large_source, small_source]), presence * (1.5 / (2 ** 0.5)),
    )
