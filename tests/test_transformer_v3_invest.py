"""V3 MLP readouts: candidate sharing, actor holdings, and policy training."""

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
    x[:, model._company_slice.start + 2, 1] = 1
    x[:, model._corp_slice.start + 1, 1] = 1
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


def test_closing_candidate_permutation_and_pass_independence(model):
    x, tokens = _inputs(model, 5)
    ctx = model._policy_context(tokens, x)
    baseline = model._closing_logits(ctx)
    permutation = torch.randperm(int(GameConstants.NUM_COMPANIES))
    permuted = model._closing_logits(replace(
        ctx, company_tokens=ctx.company_tokens[:, permutation],
    ))
    torch.testing.assert_close(permuted[:, :1], baseline[:, :1])
    torch.testing.assert_close(permuted[:, 1:], baseline[:, 1:][:, permutation])

    # With the trunk frozen, changing a candidate only affects its own score.
    changed_companies = ctx.company_tokens.clone()
    changed_companies[:, 7] += 1
    changed = model._closing_logits(replace(ctx, company_tokens=changed_companies))
    unaffected = torch.arange(baseline.shape[-1]) != 8
    torch.testing.assert_close(changed[:, unaffected], baseline[:, unaffected])
    assert torch.all(changed[:, 8] != baseline[:, 8])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("phase", [
    DecisionPhase.DPHASE_INVEST, DecisionPhase.DPHASE_CLOSING, DecisionPhase.DPHASE_BID,
    DecisionPhase.DPHASE_IPO, DecisionPhase.DPHASE_PAR,
    DecisionPhase.DPHASE_DIVIDENDS,
    DecisionPhase.DPHASE_ACQ_SELECT_CORP, DecisionPhase.DPHASE_ACQ_SELECT_COMPANY,
    DecisionPhase.DPHASE_ACQ_SELECT_PRICE,
])
def test_joint_policy_loss_reaches_all_scorers_and_trunk(model, phase, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = model.to(device)
    x, _ = _inputs(model, 3)
    x = x.to(device)
    phase = int(phase)
    slots = torch.as_tensor(build_action_lut()[phase, :int(PHASE_ACTION_SIZES[phase])]).long()
    mask = torch.zeros(2, UNIFIED_LOGIT_DIM, dtype=torch.bool, device=device)
    mask[:, slots] = True
    # Verify masking still excludes individual phase actions as well as
    # every other phase. Keep all action classes represented in the loss.
    mask[:, slots[2]] = False
    relations = torch.zeros(
        2, NUM_ATTENTION_RELATIONS, model.cfg.num_tokens, model.cfg.num_tokens,
        dtype=torch.uint8, device=device,
    )
    logits, values = model(x, mask, relations)
    assert torch.all(logits[~mask] == -1e9)
    assert torch.isfinite(logits[mask]).all() and torch.isfinite(values).all()
    target = mask.float() / mask.sum(-1, keepdim=True)
    loss = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    loss.backward()
    phase_modules = {
        int(DecisionPhase.DPHASE_INVEST): [
            model.invest_auction_head, model.invest_trade_head, model.invest_pass_head,
            model.invest_proj, model.corp_proj,
        ],
        int(DecisionPhase.DPHASE_CLOSING): [model.closing_company_head, model.closing_pass_head],
        int(DecisionPhase.DPHASE_BID): [model.bid_head, model.auction_proj],
        int(DecisionPhase.DPHASE_IPO): [model.ipo_corp_head, model.ipo_pass_head, model.par_proj],
        int(DecisionPhase.DPHASE_PAR): [model.par_head, model.par_proj, model.corp_proj],
        int(DecisionPhase.DPHASE_DIVIDENDS): [
            model.dividend_head, model.dividend_proj, model.corp_proj,
        ],
        int(DecisionPhase.DPHASE_ACQ_SELECT_CORP): [model.acq_corp_head, model.acq_pass_head],
        int(DecisionPhase.DPHASE_ACQ_SELECT_COMPANY): [model.acq_company_head, model.corp_proj],
        int(DecisionPhase.DPHASE_ACQ_SELECT_PRICE): [
            model.acq_price_head, model.acq_price_proj, model.corp_proj,
        ],
    }[phase]
    for module in phase_modules + [model.player_proj, model.company_proj, model.blocks]:
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads if g is not None) > 0


def test_ipo_candidate_sharing_and_company_conditioned_pass(model):
    x, tokens = _inputs(model, 5)
    ctx = model._policy_context(tokens, x)
    baseline = model._ipo_logits(ctx)
    permutation = torch.randperm(int(GameConstants.NUM_CORPS))
    permuted = model._ipo_logits(replace(ctx, corp_tokens=ctx.corp_tokens[:, permutation]))
    torch.testing.assert_close(permuted[:, :1], baseline[:, :1])
    torch.testing.assert_close(permuted[:, 1:], baseline[:, 1:][:, permutation])
    changed_corps = ctx.corp_tokens.clone()
    changed_corps[:, 3] += 1
    changed = model._ipo_logits(replace(ctx, corp_tokens=changed_corps))
    unaffected = torch.arange(baseline.shape[-1]) != 4
    torch.testing.assert_close(changed[:, unaffected], baseline[:, unaffected])
    assert torch.all(changed[:, 4] != baseline[:, 4])
    changed = model._ipo_logits(replace(ctx, active_company=ctx.active_company + 1))
    assert torch.all(changed[:, 0] != baseline[:, 0])


def test_acquisition_candidate_sharing_and_raw_shortcuts(model):
    x, tokens = _inputs(model, 5)
    x[:, model._company_slice, 13] = torch.rand(2, int(GameConstants.NUM_COMPANIES))
    ctx = model._policy_context(tokens, x)
    baseline = model._acq_select_corp_logits(ctx)
    permutation = torch.randperm(int(GameConstants.NUM_CORPS))
    changed_raw = x.clone()
    changed_raw[:, model._player_slice, 15:23] = x[:, model._player_slice, 15:23][:, :, permutation]
    permuted = model._acq_select_corp_logits(replace(
        ctx, raw_tokens=changed_raw, corp_tokens=ctx.corp_tokens[:, permutation],
    ))
    torch.testing.assert_close(permuted[:, :1], baseline[:, :1])
    torch.testing.assert_close(permuted[:, 1:], baseline[:, 1:][:, permutation])
    changed_raw = x.clone()
    changed_raw[0, model._player_slice.start + 4, 15 + 2] += 0.5
    changed = model._acq_select_corp_logits(replace(ctx, raw_tokens=changed_raw))
    unaffected = torch.ones_like(baseline, dtype=torch.bool)
    unaffected[0, 3] = False  # Pass followed by corp IDs.
    torch.testing.assert_close(changed[unaffected], baseline[unaffected])
    assert changed[0, 3] != baseline[0, 3]

    baseline = model._acq_select_company_logits(ctx)
    permutation = torch.randperm(int(GameConstants.NUM_COMPANIES))
    changed_raw = x.clone()
    changed_raw[:, model._company_slice, 13] = x[:, model._company_slice, 13][:, permutation]
    permuted = model._acq_select_company_logits(replace(
        ctx, raw_tokens=changed_raw, company_tokens=ctx.company_tokens[:, permutation],
    ))
    torch.testing.assert_close(permuted, baseline[:, permutation])
    changed_raw = x.clone()
    changed_raw[:, model._company_slice.start + 7, 13] += 0.5
    changed = model._acq_select_company_logits(replace(ctx, raw_tokens=changed_raw))
    unaffected = torch.arange(baseline.shape[-1]) != 7
    torch.testing.assert_close(changed[:, unaffected], baseline[:, unaffected])
    assert torch.all(changed[:, 7] != baseline[:, 7])
