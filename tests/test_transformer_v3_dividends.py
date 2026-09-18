"""Dividend features against executed payouts and legal amount slots."""

import numpy as np
import pytest
import torch

from core.data import PY_CASH_DIVISOR, PY_IMPACT_DIVISOR, DecisionPhase, PHASE_ACTION_SIZES
from core.driver import DRIVER
from core.state import GameState
from core.token_data import get_token_data
from entities.corp import CORPS
from entities.market import MARKET
from entities.player import PLAYERS
from entities.turn import TURN
from mcts.evaluator import NNEvaluator
from nn import _load_model_module
from tests.phases.conftest import float_corp_for_test, get_legal_actions
from tests.phases.test_dividends import _enter_dividends


@pytest.mark.parametrize("num_players", [3, 4, 5])
@pytest.mark.parametrize("cash", [9, 60])
def test_dividend_features_match_payments_and_amount_slots(num_players, cash):
    module = _load_model_module("nn/transformer-v3.py")
    model = module.RSSTransformerNet(module.TransformerConfig(
        num_players=5, d_model=32, d_proj=8, num_heads=4, num_layers=1,
    )).eval()
    state = GameState(num_players, max_players=5)
    state.initialize_game(num_players, seed=42, max_players=5)
    state.step_mode = True
    actor, corp_id = num_players - 1, num_players - 3
    float_corp_for_test(
        state, corp_id=corp_id, player_id=actor, company_id=14,
        par_index=MARKET.get_index_for_price(20), float_shares=2,
    )
    PLAYERS[0].set_shares(state, corp_id, 1)
    CORPS[corp_id].set_cash(state, cash)
    _enter_dividends(state)
    assert TURN.get_active_player(state) == actor
    assert TURN.get_active_corp(state) == corp_id
    assert CORPS[corp_id].get_bank_shares(state) == 1
    raw = np.empty((1, model.cfg.num_tokens, model.cfg.token_dim), np.float32)
    get_token_data(state, raw[0], max_players=5, layout_version=3)
    x = torch.from_numpy(raw)
    features = model._dividend_outcome_features(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(model._dividend_outcome_features(x), features)

    amounts = int(PHASE_ACTION_SIZES[int(DecisionPhase.DPHASE_DIVIDENDS)])
    assert features.shape == (1, amounts, 7)
    assert features[0, -1, 2] < 0  # Unaffordable outcomes remain unclipped.
    torch.testing.assert_close(
        features[0, :, 1], features[0, :, 3:6].sum(-1),
    )  # Corporation payout equals actor + other players + bank.
    expected_moves = torch.tensor([
        CORPS[corp_id].simulate_dividend_price_move(state, amount) / PY_IMPACT_DIVISOR
        for amount in range(amounts)
    ])
    torch.testing.assert_close(features[0, :, 6], expected_moves)
    if cash == 60:
        assert expected_moves.unique().numel() > 1

    before_cash = [PLAYERS[p].get_cash(state) for p in range(num_players)]
    legal = get_legal_actions(state)
    for action_id, info in legal:
        after = GameState.from_array(state._array.copy(), num_players, max_players=5)
        after.step_mode = True
        DRIVER.apply_action(after, action_id)
        corp_cash = CORPS[corp_id].get_cash(after)
        received = [PLAYERS[p].get_cash(after) - before_cash[p] for p in range(num_players)]
        total = cash - corp_cash
        expected_money = torch.tensor([
            info.amount, total, corp_cash, received[actor],
            sum(received) - received[actor], total - sum(received),
        ]) / float(PY_CASH_DIVISOR)
        torch.testing.assert_close(features[0, info.amount, :6], expected_money)

    # Check the numerical features are supplied to the actual head, and its
    # output index remains the literal dividend amount through evaluation.
    seen = []
    handle = model.dividend_head.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    with torch.no_grad():
        model.dividend_head[-1].weight.zero_()
        model.dividend_head[-1].bias.copy_(torch.arange(amounts) / 10)
    try:
        priors, values, ids, count, phase = NNEvaluator(
            model, torch.device("cpu"), 5,
        ).evaluate(state)
    finally:
        handle.remove()
    expected_ids = [aid for aid, _ in legal]
    assert expected_ids == list(range(min(20 // 3, cash // 4) + 1))
    np.testing.assert_array_equal(ids, expected_ids)
    assert count == len(expected_ids) and phase == int(DecisionPhase.DPHASE_DIVIDENDS)
    expected_priors = torch.softmax(torch.tensor(expected_ids) / 10, dim=0).numpy()
    np.testing.assert_allclose(priors, expected_priors, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(seen[0][:, 3 * model.cfg.d_model:], features.flatten(1))
    assert np.isfinite(values).all()
