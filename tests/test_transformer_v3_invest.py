"""INVEST candidate sharing, actor holdings, and joint policy training."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from core.attention_relations import NUM_ATTENTION_RELATIONS
from core.data import GameConstants, PHASE_ACTION_SIZES, DecisionPhase
from nn import _load_model_module
from nn.policy_layout import UNIFIED_LOGIT_DIM, build_action_lut


@pytest.fixture
def model():
    torch.manual_seed(42)
    module = _load_model_module("nn/transformer-v3.py")
    return module.RSSTransformerNet(module.TransformerConfig(
        num_players=5, d_model=32, d_proj=8, num_heads=4, num_layers=1,
    ))


def _inputs(model, num_players):
    # Two actors in the same batch; padded players have no selector or data.
    x = torch.zeros(2, model.cfg.num_tokens, model.cfg.token_dim)
    x[:, :model._player_slice.start + num_players, 0] = 1
    x[0, model._player_slice.start + num_players - 1, 1] = 1
    x[1, model._player_slice.start, 1] = 1
    x[:, model._player_slice.start:model._player_slice.start + num_players, 15:23] = (
        torch.rand(2, num_players, int(GameConstants.NUM_CORPS))
    )
    tokens = torch.randn(2, model.cfg.num_tokens, model.cfg.d_model)
    return x, tokens


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_raw_holdings_shortcut_uses_only_actor_and_matching_corp(model, num_players):
    x, tokens = _inputs(model, num_players)
    # Freeze contextual embeddings to isolate the raw holdings shortcut.
    baseline = model._invest_logits(model._policy_context(tokens, x))
    changed = x.clone()
    changed[0, model._player_slice.start + num_players - 1, 15 + 2] += 0.5
    changed[1, model._player_slice.start, 15 + 5] += 0.5
    actual = model._invest_logits(model._policy_context(tokens, changed))
    affected = torch.zeros_like(actual, dtype=torch.bool)
    trade_start = 1 + int(GameConstants.NUM_COMPANIES)
    affected[0, trade_start + 2 * 2:trade_start + 2 * 2 + 2] = True
    affected[1, trade_start + 2 * 5:trade_start + 2 * 5 + 2] = True
    torch.testing.assert_close(actual[~affected], baseline[~affected], rtol=0, atol=0)
    assert torch.all(actual[affected] != baseline[affected])

    # Another player's holdings must not leak into the direct shortcut.
    changed = x.clone()
    changed[0, model._player_slice.start, 15:23] += 1
    changed[1, model._player_slice.start + num_players - 1, 15:23] += 1
    torch.testing.assert_close(
        model._invest_logits(model._policy_context(tokens, changed)), baseline,
        rtol=0, atol=0,
    )


def test_candidate_permutation_preserves_scores_and_buy_sell_order(model):
    x, tokens = _inputs(model, 5)
    ctx = model._policy_context(tokens, x)
    baseline = model._invest_logits(ctx)
    company_perm = torch.randperm(int(GameConstants.NUM_COMPANIES))
    corp_perm = torch.randperm(int(GameConstants.NUM_CORPS))
    permuted_raw = x.clone()
    permuted_raw[:, model._player_slice, 15:23] = x[:, model._player_slice, 15:23][:, :, corp_perm]
    # Permute candidate embeddings and associated holdings together, keeping
    # shared actor/INVEST context fixed. Candidate scoring must be equivariant.
    permuted_ctx = replace(
        ctx, raw_tokens=permuted_raw,
        company_tokens=ctx.company_tokens[:, company_perm],
        corp_tokens=ctx.corp_tokens[:, corp_perm],
    )
    actual = model._invest_logits(permuted_ctx)
    trade_start = 1 + int(GameConstants.NUM_COMPANIES)
    expected = torch.cat([
        baseline[:, :1], baseline[:, 1:trade_start][:, company_perm],
        baseline[:, trade_start:].reshape(2, -1, 2)[:, corp_perm].flatten(1),
    ], dim=-1)
    torch.testing.assert_close(actual, expected)


def test_joint_invest_loss_reaches_all_scorers_and_trunk(model):
    x, _ = _inputs(model, 3)
    phase = int(DecisionPhase.DPHASE_INVEST)
    slots = torch.as_tensor(build_action_lut()[phase, :int(PHASE_ACTION_SIZES[phase])]).long()
    mask = torch.zeros(2, UNIFIED_LOGIT_DIM, dtype=torch.bool)
    mask[:, slots] = True
    # Verify masking still excludes individual INVEST actions as well as
    # every other phase. Keep all action classes represented in the loss.
    mask[:, slots[2]] = False
    relations = torch.zeros(
        2, NUM_ATTENTION_RELATIONS, model.cfg.num_tokens, model.cfg.num_tokens,
        dtype=torch.uint8,
    )
    logits, values = model(x, mask, relations)
    assert torch.all(logits[~mask] == -1e9)
    assert torch.isfinite(logits[mask]).all() and torch.isfinite(values).all()
    target = mask.float() / mask.sum(-1, keepdim=True)
    loss = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    loss.backward()
    for module in (
        model.invest_auction_head, model.invest_trade_head, model.invest_pass_head,
        model.invest_proj, model.player_proj, model.company_proj, model.corp_proj,
        model.blocks,
    ):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads if g is not None) > 0
