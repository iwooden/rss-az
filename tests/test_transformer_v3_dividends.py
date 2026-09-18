"""Dividend features against executed payouts and legal amount slots."""

import numpy as np
import pytest
import torch

from core.data import PY_CASH_DIVISOR, PY_IMPACT_DIVISOR, DecisionPhase, PHASE_ACTION_SIZES, GameConstants
from core.driver import DRIVER
from core.state import GameState
from core.token_data import get_token_data, get_token_data_batch, get_token_widths, get_token_dim
from core.token_data_v3 import TokenWidth
from entities.corp import CORPS
from entities.market import MARKET
from entities.player import PLAYERS
from entities.turn import TURN
from mcts.evaluator import NNEvaluator
from nn import _load_model_module
from tests.phases.conftest import float_corp_for_test, get_legal_actions
from tests.phases.test_dividends import _enter_dividends
from train.debug_trace import format_token_dump


@pytest.mark.parametrize("num_players", [3, 4, 5])
@pytest.mark.parametrize("cash", [9, 60])
def test_dividend_features_match_payments_and_amount_slots(num_players, cash):
    module = _load_model_module("nn/transformer-v3.py")
    model = module.RSSTransformerNet(module.TransformerConfig(
        num_players=5, d_model=32, num_heads=4, num_layers=1,
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
    dividend_dump = next(
        line for line in format_token_dump(state, layout_version=3).splitlines()
        if "| dividend |" in line
    )
    assert "| 53 |" in dividend_dump and "share_prices=" in dividend_dump
    features = model._dividend_outcome_features(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(model._dividend_outcome_features(x), features)

    amounts = int(PHASE_ACTION_SIZES[int(DecisionPhase.DPHASE_DIVIDENDS)])
    assert features.shape == (1, amounts, 9)
    assert features[0, -1, 2] < 0  # Unaffordable outcomes remain unclipped.
    torch.testing.assert_close(
        features[0, :, 1], features[0, :, 3:6].sum(-1),
    )  # Corporation payout equals actor + other players + bank.
    if cash == 60:
        assert features[0, :, 6].unique().numel() > 1

    before_worth = PLAYERS[actor].get_net_worth(state)
    before_index = CORPS[corp_id].get_price_index(state)
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
        assert features[0, info.amount, 6].item() == pytest.approx(
            (CORPS[corp_id].get_price_index(after) - before_index) / PY_IMPACT_DIVISOR,
        )
        assert features[0, info.amount, 7].item() == pytest.approx(
            CORPS[corp_id].get_share_price(after) / PY_CASH_DIVISOR,
        )
        assert features[0, info.amount, 8].item() == pytest.approx(
            (PLAYERS[actor].get_net_worth(after) - before_worth) / PY_CASH_DIVISOR,
            abs=1e-6,
        )

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


@pytest.mark.parametrize("index,nominal,occupied,expected", [
    (1, -2, (), 0), (1, -1, (), 0), (2, -2, (), 0),
    (7, -1, (6, 5), 4), (7, -2, (5, 4), 3),
    (3, -1, (2, 1), 0),
    (5, 0, (4, 6), 5),
    (5, 1, (6, 7), 8), (5, 2, (7, 8), 9),
    (24, 1, (25, 26), 26), (25, 2, (26,), 26),
    (26, 2, (), 26), (26, 1, (), 26), (26, 0, (), 26),
    (26, -1, (25, 24), 23),
])
def test_resolved_dividend_outcomes_match_execution(index, nominal, occupied, expected):
    """Predictions include occupied runs, both endpoints, and bankruptcy losses."""
    module = _load_model_module("nn/transformer-v3.py")
    model = module.RSSTransformerNet(module.TransformerConfig(
        num_players=5, d_model=32, num_heads=4, num_layers=1,
    )).eval()
    state = GameState(3, max_players=5)
    state.initialize_game(3, seed=42, max_players=5)
    state.step_mode = True
    actor = 2
    maximum = int(GameConstants.NUM_MARKET_SPACES) - 1

    def place_corp(corp_id, player_id, space, shares=1):
        # $75 is reachable by movement, not IPO; float below it then move.
        par = MARKET.find_next_lower_space(state, maximum) if space == maximum else space
        float_corp_for_test(
            state, corp_id=corp_id, player_id=player_id, company_id=corp_id,
            par_index=par, float_shares=shares,
        )
        if space == maximum:
            MARKET.set_space_available(state, par, True)
            CORPS[corp_id].set_price_index(state, maximum)

    place_corp(0, actor, index, shares=3)
    PLAYERS[0].set_shares(state, 0, 1)
    for corp_id, space in enumerate(occupied, start=1):
        place_corp(corp_id, 0, space)
    # Six issued shares, one company star: choose cash to induce each nominal move.
    required = int(6 * MARKET.get_price_at_index(index) / 10 + 0.5)
    CORPS[0].set_cash(state, (required - 1 + nominal) * 10)
    assert CORPS[0].get_pending_price_move(state) == nominal
    _enter_dividends(state)
    # Occupying corps have already resolved their dividends; isolate this decision.
    for corp_id in range(1, int(GameConstants.NUM_CORPS)):
        TURN.set_dividend_remaining(state, corp_id, False)
    TURN.set_active_corp(state, 0)
    TURN.set_active_player(state, actor)
    raw = np.empty((1, model.cfg.num_tokens, model.cfg.token_dim), np.float32)
    get_token_data(state, raw[0], max_players=5, layout_version=3)
    features = model._dividend_outcome_features(torch.from_numpy(raw))[0]
    assert get_token_widths(5, 3)[model._dividend_idx] == int(TokenWidth.TW_DIVIDEND)
    batched = np.empty_like(raw)
    get_token_data_batch([state._array.copy()], batched, max_players=5, layout_version=3)
    np.testing.assert_array_equal(raw, batched)
    assert features[0, 6].item() == pytest.approx((expected - index) / PY_IMPACT_DIVISOR)
    assert raw[0, model._corp_slice.start, 36] == pytest.approx(features[0, 6].item())
    # The restored v2 extractor must retain nominal movement even where v3
    # skips occupied spaces or clamps to a boundary. Check explicit nominal
    # expectations, not equality between the two versions' implementations.
    v2 = np.empty((model.cfg.num_tokens, get_token_dim(2)), np.float32)
    get_token_data(state, v2, max_players=5, layout_version=2)
    assert v2[model._corp_slice.start, 36] == pytest.approx(nominal / PY_IMPACT_DIVISOR)
    assert v2[model._dividend_idx, 1] == pytest.approx(nominal / PY_IMPACT_DIVISOR)
    assert not v2[model._dividend_idx, 27:].any()
    worth = PLAYERS[actor].get_net_worth(state)
    cash = PLAYERS[actor].get_cash(state)
    for action_id, info in get_legal_actions(state):
        after = GameState.from_array(state._array.copy(), 3, max_players=5)
        after.step_mode = True
        DRIVER.apply_action(after, action_id)
        active = CORPS[0].is_active(after)
        destination = CORPS[0].get_price_index(after) if active else 0
        price = CORPS[0].get_share_price(after) if active else 0
        assert features[info.amount, 6].item() == pytest.approx(
            (destination - index) / PY_IMPACT_DIVISOR,
        )
        assert features[info.amount, 7].item() == pytest.approx(price / PY_CASH_DIVISOR)
        assert features[info.amount, 8].item() == pytest.approx(
            (PLAYERS[actor].get_net_worth(after) - worth) / PY_CASH_DIVISOR, abs=1e-6,
        )
        assert features[info.amount, 3].item() == pytest.approx(
            (PLAYERS[actor].get_cash(after) - cash) / PY_CASH_DIVISOR,
        )

    # A change in another corporation's occupancy must update previews even
    # when this corporation's nominal-movement cache is already populated.
    for corp_id in range(1, len(occupied) + 1):
        CORPS[corp_id].go_bankrupt(state)
    get_token_data(state, raw[0], max_players=5, layout_version=3)
    unoccupied = min(max(index + nominal, 0), maximum)
    assert raw[0, model._corp_slice.start, 36] == pytest.approx(
        (unoccupied - index) / PY_IMPACT_DIVISOR,
    )
    assert raw[0, model._dividend_idx, 1] == pytest.approx(
        (unoccupied - index) / PY_IMPACT_DIVISOR,
    )
