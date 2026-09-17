"""Versioned observations of one shared game, including replay and IPC paths."""

from copy import deepcopy
from dataclasses import replace
import multiprocessing as mp

import numpy as np
import pytest
import torch

from core.data import GamePhases
from core.driver import DRIVER
from core.state import GameState, get_layout
from core.token_data import get_num_tokens, get_token_data, get_token_data_batch, get_token_dim
from entities.player import PLAYERS
from entities.turn import TURN
from mcts.evaluator import NNEvaluator
from nn import create_model, get_model_input_spec
from nn.policy_layout import UNIFIED_LOGIT_DIM
from tests.phases.conftest import get_legal_actions
from tests.phases.test_invest import _make_trade_state
from train.checkpoint import load_model_from_checkpoint, save_checkpoint
from train.config import TrainingConfig
from train.debug_trace import TokenNormalizationAccumulator, format_token_dump
from train.eval_server import EvaluationServer, RemoteEvaluator, SharedEvalBuffers
from train.replay_buffer import ReplayBuffer
from train.trainer import Trainer


def _config(version, **kwargs):
    return TrainingConfig(
        model_path=f"nn/transformer-v{version}.py",
        d_model=32, d_proj=8, num_heads=4, num_layers=1,
        optimizer="adamw", num_epochs=1, training_steps_per_epoch=1,
        warmup_epochs=0, **kwargs,
    )


def _state(n=3):
    state = GameState(n, max_players=5)
    state.initialize_game(n, seed=42, max_players=5)
    _make_trade_state(state)
    # Different player/corp histories, including unclipped counts and a
    # buy and sell in DIFFERENT corps (not a round trip).
    for _ in range(5):
        PLAYERS[0].increment_share_buys(state, 0)
    PLAYERS[0].increment_share_sells(state, 0)
    PLAYERS[1].increment_share_buys(state, 1)
    PLAYERS[1].increment_share_sells(state, 2)
    PLAYERS[2].increment_share_buys(state, 3)
    PLAYERS[2].increment_share_sells(state, 3)
    return state


def _tokens(state, version):
    result = np.full((get_num_tokens(5), get_token_dim(version)), np.nan, np.float32)
    get_token_data(state, result, max_players=5, layout_version=version)
    return result


@pytest.mark.parametrize("n", [3, 4, 5])
def test_history_is_actor_relative_but_player_flags_are_canonical(n):
    state = _state(n)
    v3 = _tokens(state, 3)
    np.testing.assert_array_equal(v3[46, 54:57], [1.25, 0.25, 1])
    np.testing.assert_array_equal(v3[54:57, 14], [1, 0, 1])
    assert not v3[54 + n:].any()
    assert np.isfinite(v3).all()

    TURN.set_active_player(state, 1)
    other = _tokens(state, 3)
    np.testing.assert_array_equal(other[46, 54:57], [0, 0, 0])
    np.testing.assert_array_equal(other[47, 54:57], [0.25, 0, 0])
    np.testing.assert_array_equal(other[48, 54:57], [0, 0.25, 0])
    np.testing.assert_array_equal(other[54:, 14], v3[54:, 14])

    # History survives non-INVEST observations, including inactive corps.
    TURN.set_phase(state, int(GamePhases.PHASE_ISSUE_SHARES))
    TURN.set_active_corp(state, 0)
    np.testing.assert_array_equal(_tokens(state, 3)[:, 54:57], other[:, 54:57])
    np.testing.assert_array_equal(_tokens(state, 3)[54:, 14], v3[54:, 14])
    assert not _tokens(state, 2)[54:, 14].any()


def test_v2_layout_retains_old_flag_and_all_other_features():
    state = _state()
    v2, v3 = _tokens(state, 2), _tokens(state, 3)
    # A single round trip sets only the v3 flag. V2 still uses >=2 OR >=2.
    np.testing.assert_array_equal(v2[54:57, 14], [1, 0, 0])
    np.testing.assert_array_equal(v2[:46], v3[:46, :95])
    np.testing.assert_array_equal(v2[46:54, :54], v3[46:54, :54])
    np.testing.assert_array_equal(v2[46:54, 54:], v3[46:54, 57:])
    np.testing.assert_array_equal(v2[54:, :14], v3[54:, :14])
    np.testing.assert_array_equal(v2[54:, 15:], v3[54:, 15:95])


@pytest.mark.parametrize("version", [2, 3])
def test_mixed_batch_reconstruction_matches_individual_extraction(version):
    states = [_state(n) for n in (3, 4, 5)]
    batch = np.empty((3, get_num_tokens(5), get_token_dim(version)), np.float32)
    arrays = [s._array.copy() for s in states]
    get_token_data_batch(arrays, batch, max_players=5, layout_version=version)
    for i, state in enumerate(states):
        np.testing.assert_array_equal(batch[i], _tokens(state, version))
        clone = GameState.from_array(arrays[i], 3 + i, max_players=5)
        np.testing.assert_array_equal(_tokens(clone, version), batch[i])


def test_history_survives_auction_return_and_resets_at_next_turn():
    state = _state()
    state.step_mode = True
    before = _tokens(state, 3)
    # Start an auction; minimum opening bid, then everybody else leaves.
    auction = next(action for action, info in get_legal_actions(state) if info.company_id >= 0)
    DRIVER.apply_action(state, auction)
    assert TURN.get_phase(state) == int(GamePhases.PHASE_BID)
    for _ in range(10):
        DRIVER.apply_action(state, get_legal_actions(state)[0][0])
        if TURN.get_phase(state) == int(GamePhases.PHASE_INVEST):
            break
    assert TURN.get_phase(state) == int(GamePhases.PHASE_INVEST)
    np.testing.assert_array_equal(_tokens(state, 3)[54:, 14], before[54:, 14])
    assert PLAYERS[0].get_share_buys(state, 0) == 5

    # Walk through the rest of this turn, including automated phases.
    saw_wrap_up = False
    for _ in range(200):
        if TURN.get_turn_number(state) == 2:
            break
        saw_wrap_up |= TURN.get_phase(state) == int(GamePhases.PHASE_WRAP_UP)
        np.testing.assert_array_equal(_tokens(state, 3)[54:, 14], before[54:, 14])
        if DRIVER.is_non_player_phase(state):
            DRIVER.advance_phase(state)
        else:
            DRIVER.apply_action(state, get_legal_actions(state)[0][0])
    assert saw_wrap_up
    assert TURN.get_turn_number(state) == 2
    assert TURN.get_phase(state) == int(GamePhases.PHASE_INVEST)
    assert not _tokens(state, 3)[54:, 14].any()
    for p in range(3):
        for c in range(8):
            assert PLAYERS[p].get_share_buys(state, c) == 0
            assert PLAYERS[p].get_share_sells(state, c) == 0


@pytest.mark.parametrize("version", [2, 3])
def test_checkpoint_evaluator_and_trainer_use_model_layout(version, tmp_path):
    config = _config(version, num_players=0, min_players=3, max_players=5)
    model = create_model(config)
    spec = get_model_input_spec(config)
    assert spec.layout_version == version
    assert getattr(model, "cfg").layout_version == version
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, 0, model, {}, config, {}, {})
    loaded, _, _ = load_model_from_checkpoint(path, torch.device("cpu"))
    evaluator = NNEvaluator(loaded, torch.device("cpu"), 5, input_spec=spec)
    states = [_state(n) for n in (3, 4, 5)]
    outputs = evaluator.evaluate_batch(states)
    for i, state in enumerate(states):
        np.testing.assert_array_equal(evaluator._tok_h_np[i], _tokens(state, version))
        assert outputs[i][1].shape == (3 + i,)
        assert np.isfinite(outputs[i][1]).all()
    evaluator.evaluate(states[0])
    np.testing.assert_array_equal(evaluator._tok_h_np[0], _tokens(states[0], version))

    replay = ReplayBuffer(3, get_layout(5).total_size, 5, min_players=3, max_players=5)
    masks = np.zeros((3, UNIFIED_LOGIT_DIM), np.uint8)
    masks[:, 0] = 1
    replay.add_stacked(
        np.stack([s._array for s in states]), np.zeros(3, np.int8), masks,
        masks.astype(np.float32), np.zeros((3, 5), np.float32),
        player_counts=np.array([3, 4, 5], np.uint8),
    )
    trainer = Trainer(loaded, config, torch.device("cpu"))
    losses = trainer.train_step(replay, 3, np.random.default_rng(0))
    assert np.isfinite(losses["total_loss"])
    assert trainer._tok_h_np.shape[-1] == spec.token_dim
    for row in range(3):
        assert trainer._tok_h_np[row, 46, 54] == (1.25 if version == 3 else 1)

    wrong = replace(spec, layout_version=5 - version, token_dim=get_token_dim(5 - version))
    with pytest.raises(ValueError, match="model input contract"):
        NNEvaluator(loaded, torch.device("cpu"), 5, input_spec=wrong)


@pytest.mark.parametrize("version", [2, 3])
def test_remote_evaluator_sends_selected_layout(version, monkeypatch):
    spec = get_model_input_spec(_config(version, num_players=5))
    shared = SharedEvalBuffers(1, 2, 5, input_spec=spec)
    shared.init_bitmap([(0, 1)])
    evaluator = RemoteEvaluator(5, shared, 0)
    monkeypatch.setattr(evaluator, "_request_eval", lambda n: None)
    state = _state()
    evaluator.evaluate(state)
    np.testing.assert_array_equal(shared.get_input_states_np(0)[0], _tokens(state, version).astype(np.float16))
    masks = np.zeros((2, UNIFIED_LOGIT_DIM), np.uint8)
    masks[:, 0] = 1
    evaluator.evaluate_leaves([state._array, state._array.copy()], masks)
    np.testing.assert_array_equal(shared.get_input_states_np(0)[1], _tokens(state, version).astype(np.float16))


def test_v3_token_diagnostics_label_and_denormalize_history():
    state = _state()
    dump = format_token_dump(state, layout_version=3)
    assert "actor_buys=5 actor_sells=1 actor_round_trip=1" in dump
    accumulator = TokenNormalizationAccumulator(3, layout_version=3)
    accumulator.add_state(state)
    assert list(accumulator._iter_field_stats())


def test_v3_eval_server_matches_local_evaluation(monkeypatch):
    monkeypatch.setattr(RemoteEvaluator, "_EVAL_TIMEOUT", 15.0)
    config = _config(3, num_players=5)
    model = create_model(config)
    shared = SharedEvalBuffers(1, 2, 5, input_spec=get_model_input_spec(config))
    ctx = mp.get_context("spawn")
    shared.init_bitmap([(0, 1)], mp_context=ctx)
    # CPU serving has no autocast: match the existing fp16 wire dtype.
    server = EvaluationServer(
        deepcopy(model).half(), torch.device("cpu"), shared,
        mp_context=ctx, no_compile=True,
    )
    state = _state()
    expected = NNEvaluator(model, torch.device("cpu"), 5).evaluate(state)
    try:
        server.start()
        assert server.wait_ready(timeout=15)
        actual = RemoteEvaluator(5, shared, 0).evaluate(state)
        # IPC quantizes token features to fp16, local evaluation uses fp32.
        np.testing.assert_allclose(actual[0], expected[0], rtol=0.01, atol=1e-4)
        np.testing.assert_allclose(actual[1], expected[1], rtol=0.01, atol=1e-4)
        np.testing.assert_array_equal(actual[2], expected[2])
    finally:
        server.stop()


def test_unsupported_layout_fails_before_extraction():
    with pytest.raises(ValueError, match="Unsupported token layout"):
        get_token_dim(4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_v3_cuda_forward_backward():
    config = _config(3, num_players=5)
    model = create_model(config).cuda()
    evaluator = NNEvaluator(model, torch.device("cuda"), 5)
    state = _state()
    priors, values, *_ = evaluator.evaluate(state)
    assert np.isfinite(priors).all() and np.isfinite(values).all()
    trainer = Trainer(model, config, torch.device("cuda"))
    trainer._ensure_scratch(1)
    trainer._states_np[0] = state._array
    trainer._fill_token_batch(1)
    np.testing.assert_array_equal(trainer._tok_h_np[0], _tokens(state, 3))
    tokens = torch.from_numpy(_tokens(state, 3)[None]).cuda()
    relations = evaluator._rel_d[:1]
    mask = evaluator._mask_d[:1]
    logits, values = model(tokens, mask, relations)
    loss = values.square().mean() + torch.logsumexp(logits, -1).mean()
    loss.backward()
    corp_proj = getattr(model, "corp_proj")
    assert isinstance(corp_proj, torch.nn.Linear)
    grad = corp_proj.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad[:, 53:56].abs().sum() > 0
