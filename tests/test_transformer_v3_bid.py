"""BID numerical features and price-slot semantics against actual auctions."""

import numpy as np
import pytest
import torch

from core.data import AUCTION_CAP, PY_COMPANY_PRICE_DIVISOR, DecisionPhase
from core.state import GameState
from core.token_data import get_token_data
from entities.company import COMPANIES
from entities.player import PLAYERS
from entities.turn import TURN
from mcts.evaluator import NNEvaluator
from nn import _load_model_module
from tests.phases.conftest import get_legal_actions
from tests.phases.test_bid import _enter_bid_phase, _place_opening_bid


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_bid_features_and_output_slots_match_engine_auctions(num_players):
    module = _load_model_module("nn/transformer-v3.py")
    model = module.RSSTransformerNet(module.TransformerConfig(
        num_players=5, d_model=32, num_heads=4, num_layers=1,
    )).eval()
    states, expected_features = [], []
    for opening in (True, False):
        state = GameState(num_players, max_players=5)
        state.initialize_game(num_players, seed=42 if opening else 71, max_players=5)
        TURN.set_active_player(state, num_players - 1)
        _, company_id = _enter_bid_phase(state)
        if not opening:
            _place_opening_bid(state, offset=2)
        actor = TURN.get_active_player(state)
        face = COMPANIES[company_id].get_face_value()
        cash = face + 4
        PLAYERS[actor].set_cash(state, cash)
        prices = torch.arange(int(AUCTION_CAP), dtype=torch.float32) + face
        expected_features.append(torch.stack([prices, cash - prices], dim=-1))
        states.append(state)

    raw = np.empty((2, model.cfg.num_tokens, model.cfg.token_dim), np.float32)
    for row, state in enumerate(states):
        get_token_data(state, raw[row], max_players=5, layout_version=3)
    expected = torch.stack(expected_features) / float(PY_COMPANY_PRICE_DIVISOR)
    features = model._bid_price_features(torch.from_numpy(raw))
    torch.testing.assert_close(features, expected, atol=1e-7, rtol=1e-5)
    assert (features[:, 5:, 1] < 0).all()  # Do not clip unaffordable balances.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(model._bid_price_features(torch.from_numpy(raw)), features)

    # Verify these features actually enter the MLP, after all three embeddings.
    seen = []
    handle = model.bid_head.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    # Distinct output biases make each slot identifiable after the unified
    # policy layout and engine legal-action mask have been applied.
    with torch.no_grad():
        model.bid_head[-1].weight.zero_()
        model.bid_head[-1].bias.copy_(torch.arange(int(AUCTION_CAP) + 1) / 10)
    try:
        outputs = NNEvaluator(model, torch.device("cpu"), 5).evaluate_batch(states)
    finally:
        handle.remove()
    torch.testing.assert_close(seen[0][:, 3 * model.cfg.d_model:], expected.flatten(1))
    for row, (state, output) in enumerate(zip(states, outputs)):
        priors, values, action_ids, num_legal, phase = output
        expected_ids = [action_id for action_id, _ in get_legal_actions(state)]
        assert expected_ids == (list(range(1, 6)) if row == 0 else [0, 4, 5])
        assert phase == int(DecisionPhase.DPHASE_BID)
        assert num_legal == len(expected_ids)
        np.testing.assert_array_equal(action_ids, expected_ids)
        expected_priors = torch.softmax(torch.tensor(expected_ids) / 10, dim=0).numpy()
        np.testing.assert_allclose(priors, expected_priors, atol=1e-7, rtol=1e-5)
        assert np.isfinite(values).all()
