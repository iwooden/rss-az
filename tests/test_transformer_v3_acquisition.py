"""Acquisition readout features against transfers and canonical ownership."""

import numpy as np
import pytest
import torch

from core.actions import ACTION_ACQ_OFFER_ACCEPT_PY
from core.data import GamePhases, DecisionPhase, PHASE_ACTION_SIZES, PY_CASH_DIVISOR
from core.driver import DRIVER
from core.state import GameState
from core.token_data import get_token_data
from entities.company import COMPANIES
from entities.corp import CORPS
from entities.player import PLAYERS
from entities.turn import TURN
from mcts.evaluator import NNEvaluator
from nn import _load_model_module
from phases.acq_select_corp import setup_acquisition_phase_py
from tests.phases.conftest import find_legal_action, float_corp_for_test, get_legal_actions
from tests.phases.helpers.ownership import give_company_to_corp, give_company_to_player


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_rejection_observation_follows_deciding_player_including_offer(num_players):
    from tests.phases.test_acq_rejections import negotiation_state, select_target, TARGET
    from core.token_data import get_num_tokens, get_token_dim
    state = negotiation_state(num_players)
    # The eventual seller's own past buying attempt is deliberately different.
    COMPANIES[TARGET].record_rejected_offer(state, 0, 17)
    COMPANIES[TARGET].record_rejected_offer(state, num_players - 1, 12)
    # Test observations independently of the v3 legal floor.
    state.v3_behavior = False
    raw = np.empty((get_num_tokens(5), get_token_dim(3)), np.float32)
    select_target(state)
    get_token_data(state, raw, max_players=5, layout_version=3)
    assert raw[TARGET + 1, 14] == np.float32(12 / 80)
    assert raw[TARGET + 1, 15] == 0
    DRIVER.apply_action(state, 1)
    assert TURN.get_active_player(state) == 0
    get_token_data(state, raw, max_players=5, layout_version=3)
    assert raw[TARGET + 1, 14] == np.float32(17 / 80)
    assert raw[TARGET + 1, 15] == 1

    # The scalar reaches the company projection; ownership remains in the tail.
    module = _load_model_module("nn/transformer-v3.py")
    model = module.RSSTransformerNet(module.TransformerConfig(
        num_players=5, d_model=32, num_heads=4, num_layers=1,
    )).eval()
    x = torch.from_numpy(raw[None]).requires_grad_()
    model._project_company_tokens(x).square().sum().backward()
    assert x.grad is not None and x.grad[0, TARGET + 1, 14].abs() > 0
    assert x.grad[0, TARGET + 1, 15].abs() > 0
    assert model.company_proj.in_features == 15


@pytest.mark.parametrize("num_players", [3, 4, 5])
@pytest.mark.parametrize("seller_kind", ["player", "corp", "foreign_player"])
@pytest.mark.parametrize("v3_behavior", [False, True])
def test_acquisition_price_features_match_buyer_and_seller_balances(num_players, seller_kind, v3_behavior):
    module = _load_model_module("nn/transformer-v3.py")
    model = module.RSSTransformerNet(module.TransformerConfig(
        num_players=5, d_model=32, num_heads=4, num_layers=1,
    )).eval()
    state = GameState(num_players, max_players=5, v3_behavior=v3_behavior)
    state.initialize_game(num_players, seed=42, max_players=5)
    state.step_mode = True
    state.acq_same_president = seller_kind != "foreign_player"
    actor = num_players - 1
    seller_player = 0 if seller_kind == "foreign_player" else actor
    target = 14
    float_corp_for_test(state, corp_id=0, player_id=actor, company_id=0, par_index=10)
    buyer_cash = COMPANIES[target].get_low_price() + 3
    CORPS[0].set_cash(state, buyer_cash)
    PLAYERS[actor].set_cash(state, 95)
    if seller_kind == "corp":
        float_corp_for_test(state, corp_id=1, player_id=actor, company_id=1, par_index=12)
        give_company_to_corp(state, target, 1)
        CORPS[1].set_cash(state, 17)
    else:
        give_company_to_player(state, target, seller_player)
        PLAYERS[seller_player].set_cash(state, 45)
    setup_acquisition_phase_py(state)
    for _ in range(num_players):
        if TURN.get_active_player(state) == actor:
            break
        DRIVER.apply_action(state, 0)
    assert TURN.get_active_player(state) == actor
    DRIVER.apply_action(state, 1)  # Select buyer corporation 0.
    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_COMPANY)
    if seller_kind == "corp":
        CORPS[1].set_acquisition_proceeds(state, 9)
    evaluator = NNEvaluator(model, torch.device("cpu"), 5)

    raw = np.empty((1, model.cfg.num_tokens, model.cfg.token_dim), np.float32)
    get_token_data(state, raw[0], max_players=5, layout_version=3)
    seen_company = []
    handle = model.acq_company_head.register_forward_pre_hook(
        lambda _module, args: seen_company.append(args[0].detach().clone())
    )
    try:
        company_output = evaluator.evaluate(state)
    finally:
        handle.remove()
    np.testing.assert_array_equal(company_output[2], [aid for aid, _ in get_legal_actions(state)])
    assert target in company_output[2]
    torch.testing.assert_close(
        seen_company[0][:, :, -1], torch.from_numpy(raw[:, model._company_slice, 13]),
    )
    DRIVER.apply_action(state, target)  # Company-selection ID is company ID.
    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_PRICE)
    get_token_data(state, raw[0], max_players=5, layout_version=3)
    x = torch.from_numpy(raw)
    features = model._acq_price_features(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(model._acq_price_features(x), features)
    num_prices = int(PHASE_ACTION_SIZES[int(DecisionPhase.DPHASE_ACQ_SELECT_PRICE)])
    prices = torch.arange(num_prices) + COMPANIES[target].get_low_price()
    seller_before = 26 if seller_kind == "corp" else 45
    expected = torch.stack([prices, buyer_cash - prices, seller_before + prices], dim=-1)
    expected = expected.float().unsqueeze(0) / float(PY_CASH_DIVISOR)
    torch.testing.assert_close(features, expected, atol=1e-7, rtol=1e-5)
    assert features[0, -1, 1] < 0

    for aid, info in get_legal_actions(state):
        after = GameState.from_array(
            state._array.copy(), num_players, max_players=5, v3_behavior=v3_behavior,
        )
        after.step_mode = True
        after.acq_same_president = state.acq_same_president
        DRIVER.apply_action(after, aid)
        if seller_kind == "foreign_player":
            assert TURN.get_phase(after) == int(GamePhases.PHASE_ACQ_OFFER)
            DRIVER.apply_action(after, find_legal_action(after, action_type=ACTION_ACQ_OFFER_ACCEPT_PY))
        if seller_kind == "corp":
            assert CORPS[1].get_cash(after) == 17  # Proceeds remain locked.
            seller_after = CORPS[1].get_cash(after) + CORPS[1].get_acquisition_proceeds(after)
        else:
            seller_after = PLAYERS[seller_player].get_cash(after)
        expected_balances = torch.tensor([CORPS[0].get_cash(after), seller_after]) / float(PY_CASH_DIVISOR)
        torch.testing.assert_close(features[0, info.amount, 1:], expected_balances)

    seen_price = []
    handle = model.acq_price_head.register_forward_pre_hook(
        lambda _module, args: seen_price.append(args[0].detach().clone())
    )
    with torch.no_grad():
        model.acq_price_head[-1].weight.zero_()
        model.acq_price_head[-1].bias.copy_(torch.arange(num_prices) / 10)
    try:
        priors, values, ids, count, phase = evaluator.evaluate(state)
    finally:
        handle.remove()
    expected_ids = np.array([3]) if v3_behavior and seller_kind == "foreign_player" else np.arange(4)
    np.testing.assert_array_equal(ids, expected_ids)
    assert count == len(expected_ids) and phase == int(DecisionPhase.DPHASE_ACQ_SELECT_PRICE)
    np.testing.assert_allclose(
        priors, torch.softmax(torch.from_numpy(expected_ids) / 10, 0).numpy(), atol=1e-7,
    )
    torch.testing.assert_close(seen_price[0][:, 4 * model.cfg.d_model:], features.flatten(1))
    assert np.isfinite(values).all()
