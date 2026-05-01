from __future__ import annotations

import numpy as np
import pytest
import torch

from core.data import PY_CASH_DIVISOR
from core.resnet_data import (
    get_resnet_data,
    get_resnet_data_batch,
    get_resnet_vector_size,
)
from core.state import GameState
from core.token_data import TokenDataSize, get_num_tokens, get_token_data
from entities.company import CompanyLocation
from entities.corp import CORPS
from entities.deck import DECK
from entities.player import PLAYERS
from entities.turn import TURN
from mcts.evaluator import NNEvaluator
from nn import create_model, get_model_input_spec
from nn.model_contract import (
    canonical_player_for_relative,
    relative_slot_for_canonical,
    rotate_values_to_relative,
    unrotate_values_to_canonical,
)
from nn.transformer import UNIFIED_LOGIT_DIM
from train.config import TrainingConfig


MARKET_TOK = 0
COMPANY_TOK_BASE = 1
FI_TOK = 37
GLOBAL_TOK = 38
INVEST_TOK = 39
CORP_TOK_BASE = 46
PLAYER_TOK_BASE = 54

GLOBAL_BASE = 0
MARKET_BASE = 23
PHASE_BASE = 77
COMPANY_BASE = 156


def _company_stride(num_players: int) -> int:
    return 22 + num_players


def _fi_base(num_players: int) -> int:
    return COMPANY_BASE + 36 * _company_stride(num_players)


def _corp_stride(num_players: int) -> int:
    return 89 + num_players


def _corp_base(num_players: int) -> int:
    return _fi_base(num_players) + 39


def _player_stride(num_players: int) -> int:
    return 56 + num_players


def _player_base(num_players: int) -> int:
    return _corp_base(num_players) + 8 * _corp_stride(num_players)


def _new_state(num_players: int = 3, seed: int = 42) -> GameState:
    state = GameState(num_players)
    state.initialize_game(num_players, seed=seed)
    return state


def _resnet_vec(state: GameState, num_players: int) -> np.ndarray:
    vec = np.full(get_resnet_vector_size(num_players), -7.0, dtype=np.float32)
    get_resnet_data(state, vec)
    return vec


def _token_buf(state: GameState, num_players: int) -> np.ndarray:
    buf = np.full(
        (get_num_tokens(num_players), int(TokenDataSize.TOKEN_DIM)),
        -7.0,
        dtype=np.float32,
    )
    get_token_data(state, buf)
    return buf


@pytest.mark.parametrize("num_players, expected", [(3, 2008), (4, 2115), (5, 2224)])
def test_resnet_vector_size_matches_schema_and_factory(
    num_players: int,
    expected: int,
) -> None:
    assert get_resnet_vector_size(num_players) == expected

    config = TrainingConfig(
        num_players=num_players,
        model_type="resnet",
        resnet_hidden_dim=32,
        resnet_num_blocks=1,
    )
    spec = get_model_input_spec(config)
    model = create_model(config)

    assert spec.input_dim == expected
    assert model.cfg.input_dim == expected


@pytest.mark.parametrize("num_players", [2, 6])
def test_resnet_vector_rejects_unsupported_nn_player_counts(num_players: int) -> None:
    with pytest.raises(AssertionError, match="num_players"):
        get_resnet_vector_size(num_players)


def test_resnet_vector_matches_token_data_for_active_zero_invest_state() -> None:
    num_players = 3
    state = _new_state(num_players)
    vec = _resnet_vec(state, num_players)
    tokens = _token_buf(state, num_players)

    np.testing.assert_allclose(
        vec[GLOBAL_BASE:GLOBAL_BASE + 23],
        tokens[GLOBAL_TOK, 1:24],
    )
    np.testing.assert_allclose(
        vec[MARKET_BASE:MARKET_BASE + 54],
        tokens[MARKET_TOK, 1:55],
    )

    expected_phase = np.zeros(79, dtype=np.float32)
    expected_phase[0] = tokens[INVEST_TOK, 1]
    np.testing.assert_allclose(vec[PHASE_BASE:PHASE_BASE + 79], expected_phase)

    for company_id in range(36):
        base = COMPANY_BASE + company_id * _company_stride(num_players)
        record = vec[base:base + _company_stride(num_players)]
        token = tokens[COMPANY_TOK_BASE + company_id]

        np.testing.assert_allclose(record[:21], token[1:22])
        np.testing.assert_allclose(record[21:21 + num_players], token[22:22 + num_players])
        assert record[21 + num_players] == token[27]

    fi_base = _fi_base(num_players)
    np.testing.assert_allclose(vec[fi_base:fi_base + 39], tokens[FI_TOK, 1:40])

    for corp_id in range(8):
        base = _corp_base(num_players) + corp_id * _corp_stride(num_players)
        record = vec[base:base + _corp_stride(num_players)]
        token = tokens[CORP_TOK_BASE + corp_id]

        np.testing.assert_allclose(record[:53], token[1:54])
        np.testing.assert_allclose(record[53:53 + num_players], token[54:54 + num_players])
        np.testing.assert_allclose(record[53 + num_players:], token[59:95])

    for player_id in range(num_players):
        base = _player_base(num_players) + player_id * _player_stride(num_players)
        record = vec[base:base + _player_stride(num_players)]
        token = tokens[PLAYER_TOK_BASE + player_id]
        expected = np.zeros(_player_stride(num_players), dtype=np.float32)

        expected[0] = token[1]
        expected[1:1 + num_players] = token[2:2 + num_players]
        expected[1 + num_players] = token[7]
        expected[2 + num_players] = token[8]
        expected[3 + num_players] = token[9]
        expected[4 + num_players] = token[10]
        expected[5 + num_players] = token[11]
        expected[6 + num_players] = token[12]
        expected[7 + num_players] = token[13]
        expected[8 + num_players] = token[14]
        expected[9 + num_players:17 + num_players] = token[15:23]
        expected[17 + num_players] = token[23]
        expected[18 + num_players] = token[24]
        expected[19 + num_players] = token[25]
        expected[20 + num_players:] = token[26:62]

        np.testing.assert_allclose(record, expected)


def test_resnet_vector_rotates_players_owners_and_presidents() -> None:
    num_players = 3
    state = _new_state(num_players)

    DECK.set_company_location(
        state,
        6,
        int(CompanyLocation.LOC_PLAYER),
        2,
    )
    DECK.set_company_location(
        state,
        14,
        int(CompanyLocation.LOC_PLAYER),
        0,
    )
    CORPS[0].float_corp(
        state,
        player_id=0,
        company_id=14,
        market_index=10,
    )
    for player_id, cash in enumerate((111, 222, 333)):
        PLAYERS[player_id].set_cash(state, cash)
    TURN.set_active_player(state, 1)

    vec = _resnet_vec(state, num_players)

    player_base = _player_base(num_players)
    stride = _player_stride(num_players)
    cash_offset = 2 + num_players
    turn_order_slice = slice(1, 1 + num_players)

    assert vec[player_base + cash_offset] == pytest.approx(222 / PY_CASH_DIVISOR)
    assert vec[player_base + stride + cash_offset] == pytest.approx(333 / PY_CASH_DIVISOR)
    assert vec[player_base + 2 * stride + cash_offset] == pytest.approx(111 / PY_CASH_DIVISOR)
    np.testing.assert_array_equal(vec[player_base + turn_order_slice.start:player_base + turn_order_slice.stop], [1, 0, 0])
    np.testing.assert_array_equal(
        vec[player_base + stride + turn_order_slice.start:player_base + stride + turn_order_slice.stop],
        [0, 1, 0],
    )
    np.testing.assert_array_equal(
        vec[player_base + 2 * stride + turn_order_slice.start:player_base + 2 * stride + turn_order_slice.stop],
        [0, 0, 1],
    )

    company_base = COMPANY_BASE + 6 * _company_stride(num_players)
    np.testing.assert_array_equal(
        vec[company_base + 21:company_base + 21 + num_players],
        [0, 1, 0],
    )

    corp_base = _corp_base(num_players)
    np.testing.assert_array_equal(
        vec[corp_base + 53:corp_base + 53 + num_players],
        [0, 0, 1],
    )


def test_resnet_data_batch_matches_single_extraction() -> None:
    num_players = 3
    states = [_new_state(num_players, seed=1), _new_state(num_players, seed=2)]
    TURN.set_active_player(states[1], 2)
    state_arrays = [state._array.copy() for state in states]

    batch = np.empty((2, get_resnet_vector_size(num_players)), dtype=np.float32)
    get_resnet_data_batch(state_arrays, num_players, batch)

    for i, state in enumerate(states):
        np.testing.assert_allclose(batch[i], _resnet_vec(state, num_players))


def test_local_resnet_evaluator_unrotates_model_values_to_canonical() -> None:
    num_players = 3
    input_dim = get_resnet_vector_size(num_players)

    class RelativeValueModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cfg = type(
                "Cfg",
                (),
                {"num_players": num_players, "input_dim": input_dim},
            )()

        def forward(self, x, legal_mask, relations=None):
            del relations
            assert x.shape[1] == input_dim
            logits = torch.zeros(
                x.shape[0],
                int(UNIFIED_LOGIT_DIM),
                dtype=x.dtype,
                device=x.device,
            )
            logits = logits.masked_fill(~legal_mask.to(torch.bool), -1e9)
            values = torch.arange(num_players, dtype=x.dtype, device=x.device)
            return logits, values.expand(x.shape[0], -1).clone()

    state = _new_state(num_players)
    TURN.set_active_player(state, 1)
    evaluator = NNEvaluator(
        RelativeValueModel(),
        torch.device("cpu"),
        num_players=num_players,
    )

    priors, values, action_ids, n_legal, phase_id = evaluator.evaluate(state)

    assert len(priors) == n_legal == len(action_ids)
    assert phase_id == 0
    np.testing.assert_allclose(values, [2.0, 0.0, 1.0])


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_active_relative_value_helpers_roundtrip(num_players: int) -> None:
    values = np.arange(2 * num_players, dtype=np.float32).reshape(2, num_players)

    for active_player in range(num_players):
        for rel in range(num_players):
            canonical = canonical_player_for_relative(active_player, rel, num_players)
            assert relative_slot_for_canonical(active_player, canonical, num_players) == rel

        relative = rotate_values_to_relative(values, active_player, num_players)
        restored = unrotate_values_to_canonical(relative, active_player, num_players)
        np.testing.assert_array_equal(restored, values)
