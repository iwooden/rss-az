from __future__ import annotations

import multiprocessing as mp
from types import SimpleNamespace

import numpy as np
import torch

from core.actions import enumerate_legal_actions_py, get_decision_phase_py
from core.data import MAX_ACTION_SIZE
from core.resnet_data import get_resnet_vector_size
from core.state import GameState
from entities.turn import TURN
from mcts.evaluator import NNEvaluator
from nn import get_model_input_spec
from nn.transformer import UNIFIED_LOGIT_DIM, build_action_lut
from train.config import TrainingConfig
from train.eval_server import EvaluationServer, RemoteEvaluator, SharedEvalBuffers


NUM_PLAYERS = 3


class RelativeValueModel(torch.nn.Module):
    def __init__(self, num_players: int = NUM_PLAYERS) -> None:
        super().__init__()
        input_dim = get_resnet_vector_size(num_players)
        self.cfg = SimpleNamespace(
            num_players=num_players,
            input_dim=input_dim,
        )
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, x, legal_mask, relations=None):
        del relations
        self.input_shapes.append(tuple(x.shape))
        assert x.ndim == 2
        assert x.shape[1] == self.cfg.input_dim
        logits = torch.zeros(
            x.shape[0],
            int(UNIFIED_LOGIT_DIM),
            dtype=x.dtype,
            device=x.device,
        )
        logits = logits.masked_fill(~legal_mask.to(torch.bool), -1e9)
        values = torch.arange(
            self.cfg.num_players,
            dtype=x.dtype,
            device=x.device,
        )
        return logits, values.expand(x.shape[0], -1).clone()


class CompiledLikeWrapper(torch.nn.Module):
    """Minimal wrapper that mirrors the ``torch.compile`` ``_orig_mod`` shape."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = module

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)


def _small_resnet_config() -> TrainingConfig:
    return TrainingConfig(
        num_players=NUM_PLAYERS,
        model_type="resnet",
        resnet_hidden_dim=32,
        resnet_num_blocks=0,
    )


def _new_state(active_player: int, seed: int = 42) -> GameState:
    state = GameState(NUM_PLAYERS)
    state.initialize_game(NUM_PLAYERS, seed=seed)
    TURN.set_active_player(state, active_player)
    return state


def _canonical_values_for_active(active_player: int) -> np.ndarray:
    relative = np.arange(NUM_PLAYERS, dtype=np.float32)
    return np.array(
        [
            relative[(player_id - active_player) % NUM_PLAYERS]
            for player_id in range(NUM_PLAYERS)
        ],
        dtype=np.float32,
    )


def _dense_legal_mask(states: list[GameState]) -> np.ndarray:
    lut = build_action_lut().numpy()
    scratch = np.empty(MAX_ACTION_SIZE, dtype=np.uint16)
    legal_mask = np.zeros((len(states), int(UNIFIED_LOGIT_DIM)), dtype=np.uint8)
    for row, state in enumerate(states):
        phase_id = get_decision_phase_py(state)
        n_legal = enumerate_legal_actions_py(state, scratch)
        legal_mask[row, lut[phase_id, scratch[:n_legal]]] = 1
    return legal_mask


def test_resnet_evaluator_unrotates_values_for_batch_and_leaf_paths() -> None:
    model = RelativeValueModel()
    evaluator = NNEvaluator(
        model,
        torch.device("cpu"),
        num_players=NUM_PLAYERS,
        input_spec=get_model_input_spec(_small_resnet_config()),
    )
    states = [_new_state(0, seed=1), _new_state(2, seed=2)]

    batch_results = evaluator.evaluate_batch(states)
    assert [shape[1] for shape in model.input_shapes] == [model.cfg.input_dim]
    for state, result in zip(states, batch_results, strict=True):
        _priors, values, _action_ids, _n_legal, _phase_id = result
        active_player = TURN.get_active_player(state)
        np.testing.assert_allclose(
            values,
            _canonical_values_for_active(active_player),
        )

    priors, values = evaluator.evaluate_leaves(
        [state._array for state in states],
        _dense_legal_mask(states),
    )
    assert priors.shape == (2, int(UNIFIED_LOGIT_DIM))
    assert values.shape == (2, NUM_PLAYERS)
    for row, state in enumerate(states):
        active_player = TURN.get_active_player(state)
        np.testing.assert_allclose(
            values[row],
            _canonical_values_for_active(active_player),
        )


def test_resnet_evaluator_empty_leaf_path_has_resnet_shapes() -> None:
    evaluator = NNEvaluator(
        RelativeValueModel(),
        torch.device("cpu"),
        num_players=NUM_PLAYERS,
        input_spec=get_model_input_spec(_small_resnet_config()),
    )

    priors, values = evaluator.evaluate_leaves(
        [],
        np.empty((0, int(UNIFIED_LOGIT_DIM)), dtype=np.uint8),
    )

    assert priors.shape == (0, int(UNIFIED_LOGIT_DIM))
    assert values.shape == (0, NUM_PLAYERS)


def test_resnet_evaluator_infers_spec_from_compiled_wrapper() -> None:
    model = RelativeValueModel()
    evaluator = NNEvaluator(
        CompiledLikeWrapper(model),
        torch.device("cpu"),
        num_players=NUM_PLAYERS,
    )
    state = _new_state(1)

    _priors, values, _action_ids, _n_legal, _phase_id = evaluator.evaluate(state)

    assert model.input_shapes == [(1, model.cfg.input_dim)]
    np.testing.assert_allclose(values, _canonical_values_for_active(1))


def test_resnet_remote_evaluator_roundtrip_returns_canonical_values() -> None:
    config = _small_resnet_config()
    shared_bufs = SharedEvalBuffers(
        num_workers=1,
        batch_size=2,
        num_players=NUM_PLAYERS,
        input_spec=get_model_input_spec(config),
    )
    ctx = mp.get_context("spawn")
    shared_bufs.init_bitmap([(0, 1)], ctx)
    server = EvaluationServer(
        RelativeValueModel(),
        torch.device("cpu"),
        shared_bufs,
        mp_context=ctx,
        no_compile=True,
    )

    try:
        server.start()
        assert server.wait_ready(timeout=10.0)

        evaluator = RemoteEvaluator(NUM_PLAYERS, shared_bufs, worker_idx=0)
        states = [_new_state(0, seed=3), _new_state(2, seed=4)]
        legal_mask = _dense_legal_mask(states)

        priors, values = evaluator.evaluate_leaves(
            [state._array for state in states],
            legal_mask,
        )
    finally:
        server.stop()

    assert priors.shape == (2, int(UNIFIED_LOGIT_DIM))
    assert values.shape == (2, NUM_PLAYERS)
    np.testing.assert_allclose(priors.sum(axis=1), np.ones(2, dtype=np.float32))
    for row, state in enumerate(states):
        active_player = TURN.get_active_player(state)
        np.testing.assert_allclose(
            values[row],
            _canonical_values_for_active(active_player),
        )
