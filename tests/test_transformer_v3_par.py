"""IPO/PAR MLP inputs and logits against engine capitalization outcomes."""

import numpy as np
import pytest
import torch

from core.actions import ACTION_IPO_PY
from core.data import ALL_PAR_PRICES, PY_CASH_DIVISOR, DecisionPhase
from core.driver import DRIVER
from core.state import GameState
from core.token_data import get_token_data
from entities.corp import CORPS
from entities.market import MARKET
from entities.player import PLAYERS
from mcts.evaluator import NNEvaluator
from nn import _load_model_module
from tests.phases.conftest import find_legal_action, get_legal_actions
from tests.phases.test_ipo import _enter_ipo


def _model():
    module = _load_model_module("nn/transformer-v3.py")
    return module.RSSTransformerNet(module.TransformerConfig(
        num_players=5, d_model=32, d_proj=8, num_heads=4, num_layers=1,
    )).eval()


def _state(num_players, company_id, cash=100):
    state = GameState(num_players, max_players=5)
    state.initialize_game(num_players, seed=42, max_players=5)
    state.step_mode = True
    _enter_ipo(state, {company_id: num_players - 1}, {num_players - 1: cash})
    return state


def _tokens(model, state):
    raw = np.empty((1, model.cfg.num_tokens, model.cfg.token_dim), np.float32)
    get_token_data(state, raw[0], max_players=5, layout_version=3)
    return torch.from_numpy(raw)


@pytest.mark.parametrize("num_players,company_id", [(3, 5), (4, 14), (5, 35)])
def test_par_features_match_executed_floats(num_players, company_id):
    model = _model()
    state = _state(num_players, company_id)
    ipo_features = model._par_outcome_features(_tokens(model, state))
    # Select a charter through the actual IPO decision. Capitalization preview
    # is already present during IPO and must not change on entry into PAR.
    corp_id = 3
    DRIVER.apply_action(state, find_legal_action(state, action_type=ACTION_IPO_PY, corp_id=corp_id))
    raw = _tokens(model, state)
    features = model._par_outcome_features(raw)
    torch.testing.assert_close(features, ipo_features)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(model._par_outcome_features(raw), features)
    for action_id, info in get_legal_actions(state):
        after = GameState.from_array(state._array.copy(), num_players, max_players=5)
        after.step_mode = True
        DRIVER.apply_action(after, action_id)
        remaining = PLAYERS[num_players - 1].get_cash(after)
        expected = torch.tensor([
            ALL_PAR_PRICES[info.amount] / PY_CASH_DIVISOR,
            (100 - remaining) / PY_CASH_DIVISOR,
            remaining / PY_CASH_DIVISOR,
            CORPS[corp_id].get_cash(after) / PY_CASH_DIVISOR,
            CORPS[corp_id].get_issued_shares(after) / 4.0,
        ])
        torch.testing.assert_close(features[0, info.amount], expected)


def test_par_evaluator_preserves_price_slots_and_all_legality_gates():
    model = _model()
    state = _state(3, 14, cash=4)  # FV20: par20 costs0, par22 costs2, par24 costs4.
    MARKET.set_space_available(state, MARKET.get_index_for_price(22), False)
    evaluator = NNEvaluator(model, torch.device("cpu"), 5)
    ipo = evaluator.evaluate(state)
    np.testing.assert_array_equal(ipo[2], np.arange(9))  # Pass plus eight charters.
    DRIVER.apply_action(state, find_legal_action(state, action_type=ACTION_IPO_PY, corp_id=1))
    raw = _tokens(model, state)
    features = model._par_outcome_features(raw)
    assert features[0, ALL_PAR_PRICES.index(18), 2] < 0  # Unaffordable, not clipped.
    assert features[0, ALL_PAR_PRICES.index(10), 1] == 0  # Tier-ineligible preview.
    seen = []
    handle = model.par_head.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    with torch.no_grad():
        model.par_head[-1].weight.zero_()
        model.par_head[-1].bias.copy_(torch.arange(len(ALL_PAR_PRICES)) / 10)
    try:
        priors, values, ids, count, phase = evaluator.evaluate(state)
    finally:
        handle.remove()
    expected_ids = [ALL_PAR_PRICES.index(p) for p in (20, 24)]
    np.testing.assert_array_equal(ids, expected_ids)
    assert count == 2 and phase == int(DecisionPhase.DPHASE_PAR)
    expected_priors = torch.softmax(torch.tensor(expected_ids) / 10, dim=0).numpy()
    np.testing.assert_allclose(priors, expected_priors, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(seen[0][:, 4 * model.cfg.d_model:], features.flatten(1))
    assert np.isfinite(values).all()
