"""Locked transformer-v2 baseline: inputs and outputs must not drift.

Contract (AGENTS.md): with ``v3_behavior=False``, legality, the token features
and relation planes supplied to the v2 model, and v2 policy/value outputs stay
identical, even when the raw ``GameState`` layout changes.

The fixture in ``tests/v2_parity/`` stores recorded random games as action
sequences, so it survives layout changes. It contains:

- ``checkpoint.pt``: a tiny seeded v2 checkpoint. Its config JSON is the
  live-play checkpoint's (original field set) with only model dimensions
  shrunk, and it loads through ``load_model_from_checkpoint``.
- ``expected.npz``: every decision's phase and legal mask; sampled decisions'
  tokens, v2 relation planes, logits and values; and the live-play
  checkpoint's config JSON and parameter shapes.

Games cover every decision phase under both self-play rules and the
18xx.games live-replay rules (cross-president acquisitions, positive-income
closing). Tokens, relation planes and masks must match bit for bit; outputs
match to float32 rounding so CPU kernel differences between hosts don't fail.

Regenerating the fixture redefines the v2 baseline. Do it only for an
approved behavior change:

    .venv/bin/python -m tests.test_v2_parity --capture \\
        [--reference-checkpoint PATH_TO_LIVE_V2_CHECKPOINT]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from core.actions import enumerate_legal_actions_py, get_decision_phase_py
from core.data import MAX_ACTION_SIZE, DecisionPhase
from core.driver import DRIVER, STATUS_GAME_OVER_PY, STATUS_OK_PY
from core.state import GameState
from mcts.evaluator import NNEvaluator
from nn import create_model, get_model_input_spec
from nn.policy_layout import UNIFIED_LOGIT_DIM, build_action_lut
from tests.phases.conftest import play_random_decisions
from tests.test_relations import _materialize_relation_coords_np
from train.checkpoint import Checkpoint, load_checkpoint, load_model_from_checkpoint
from train.config import TrainingConfig
from train.eval_server import RemoteEvaluator, SharedEvalBuffers


FIXTURE_DIR = Path(__file__).with_name("v2_parity")
CHECKPOINT_PATH = FIXTURE_DIR / "checkpoint.pt"
EXPECTED_PATH = FIXTURE_DIR / "expected.npz"

V2_MODEL_PATH = "nn/transformer-v2.py"
MAX_PLAYERS = 5
# nn/transformer-v2.py consumes only the original ten binary relation planes.
V2_RELATION_PLANES = 10

# Capture-only settings; the fixture records the resulting games.
SELF_PLAY_RULES = {"acq_same_president": True, "allow_positive_income_closing": False}
LIVE_REPLAY_RULES = {"acq_same_president": False, "allow_positive_income_closing": True}
CAPTURE_GAMES = (  # (num_players, seed, rules): seeds chosen for phase coverage
    (3, 1, SELF_PLAY_RULES), (4, 1, SELF_PLAY_RULES), (5, 2, SELF_PLAY_RULES),
    (3, 0, LIVE_REPLAY_RULES), (4, 0, LIVE_REPLAY_RULES), (5, 0, LIVE_REPLAY_RULES),
)
CAPTURE_DIMS = {"d_model": 32, "d_proj": 8, "num_heads": 4, "num_layers": 2}
CAPTURE_INVEST_STRIDE = 4  # sample every 4th INVEST decision, all others fully

_ACTION_LUT = build_action_lut().numpy()


@dataclass
class Game:
    num_players: int
    seed: int
    acq_same_president: bool
    allow_positive_income_closing: bool
    actions: np.ndarray

    def new_state(self) -> GameState:
        state = GameState(
            self.num_players, max_players=MAX_PLAYERS, v3_behavior=False,
            acq_same_president=self.acq_same_president,
            allow_positive_income_closing=self.allow_positive_income_closing,
        )
        state.initialize_game(self.num_players, seed=self.seed, max_players=MAX_PLAYERS)
        return state


@dataclass
class Observation:
    game: int
    decision: int
    phase: int
    state_array: np.ndarray
    tokens: torch.Tensor
    relations: torch.Tensor
    legal_mask: torch.Tensor
    logits: torch.Tensor
    values: torch.Tensor


@dataclass
class Replay:
    phases: list[list[int]] = field(default_factory=list)
    legal_masks: list[list[np.ndarray]] = field(default_factory=list)
    ended: list[bool] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)


def _legal_mask(state: GameState, phase: int) -> np.ndarray:
    ids = np.empty(int(MAX_ACTION_SIZE), dtype=np.uint16)
    count = enumerate_legal_actions_py(state, ids)
    mask = np.zeros(int(UNIFIED_LOGIT_DIM), dtype=bool)
    mask[_ACTION_LUT[phase, ids[:count]]] = True
    return mask


class _ModelIO:
    """Record every forward call's inputs and outputs."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.calls: list[tuple[torch.Tensor, ...]] = []
        self._handle = model.register_forward_hook(self._hook)

    def _hook(self, _module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
        self.calls.append(tuple(t.detach().clone() for t in (*args, *output)))

    def pop(self) -> tuple[torch.Tensor, ...]:
        assert len(self.calls) == 1, f"expected one forward call, got {len(self.calls)}"
        return self.calls.pop()

    def close(self) -> None:
        self._handle.remove()


def replay_games(
    model: torch.nn.Module, games: list[Game], observe: set[tuple[int, int]] | None,
) -> Replay:
    """Replay recorded actions, evaluating ``observe`` decisions (all if None).

    Stops a game at the first recorded action the engine rejects.
    """
    evaluator = NNEvaluator(model, torch.device("cpu"), MAX_PLAYERS)
    io = _ModelIO(model)
    replay = Replay()
    try:
        for g, game in enumerate(games):
            state = game.new_state()
            phases: list[int] = []
            masks: list[np.ndarray] = []
            ended = False
            for d, action in enumerate(game.actions.tolist()):
                phase = get_decision_phase_py(state)
                phases.append(phase)
                masks.append(_legal_mask(state, phase))
                if observe is None or (g, d) in observe:
                    evaluator.evaluate(state)
                    tokens, legal_mask, relations, logits, values = io.pop()
                    replay.observations.append(Observation(
                        g, d, phase, state._array.copy(), tokens[0],
                        relations[0, :V2_RELATION_PLANES], legal_mask[0], logits[0], values[0],
                    ))
                status = DRIVER.apply_action(state, action)
                if status != STATUS_OK_PY:
                    ended = status == STATUS_GAME_OVER_PY and d == len(game.actions) - 1
                    break
            replay.phases.append(phases)
            replay.legal_masks.append(masks)
            replay.ended.append(ended)
    finally:
        io.close()
    return replay


def _load_games(expected: Any) -> list[Game]:
    offsets = expected["action_offsets"]
    return [
        Game(
            int(expected["game_players"][g]), int(expected["game_seeds"][g]),
            bool(expected["game_acq_same_president"][g]),
            bool(expected["game_positive_income_closing"][g]),
            expected["actions"][offsets[g]:offsets[g + 1]],
        )
        for g in range(len(offsets) - 1)
    ]


def _describe(obs: Observation, games: list[Game]) -> str:
    game = games[obs.game]
    rules = "live-replay" if not game.acq_same_president else "self-play"
    return (
        f"game {obs.game} ({game.num_players}p seed={game.seed} {rules} rules), "
        f"decision {obs.decision}, {DecisionPhase(obs.phase).name}"
    )


def _assert_outputs_close(actual: torch.Tensor, expected: np.ndarray, what: str) -> None:
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), rtol=1e-5, atol=1e-5, msg=lambda m: f"{what} changed\n{m}",
    )


def _token_diff(actual: torch.Tensor, expected: torch.Tensor) -> str:
    rows, cols = torch.nonzero(actual != expected, as_tuple=True)
    shown = [
        f"token {r} feature {c}: {expected[r, c].item()!r} -> {actual[r, c].item()!r}"
        for r, c in zip(rows.tolist()[:8], cols.tolist()[:8])
    ]
    more = f" (+{len(rows) - 8} more)" if len(rows) > 8 else ""
    return "; ".join(shown) + more


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def expected() -> Any:
    with np.load(EXPECTED_PATH, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def games(expected: Any) -> list[Game]:
    return _load_games(expected)


@pytest.fixture(scope="module")
def checkpoint() -> tuple[torch.nn.Module, TrainingConfig]:
    loaded, config, _ = load_model_from_checkpoint(CHECKPOINT_PATH, torch.device("cpu"))
    assert config.model_path == V2_MODEL_PATH
    assert not config.v3_behavior
    assert get_model_input_spec(config).layout_version == 2
    return loaded.eval(), config


@pytest.fixture(scope="module")
def model(checkpoint: tuple[torch.nn.Module, TrainingConfig]) -> torch.nn.Module:
    return checkpoint[0]


@pytest.fixture(scope="module")
def replay(model: torch.nn.Module, games: list[Game], expected: Any) -> Replay:
    observe = set(zip(expected["obs_game"].tolist(), expected["obs_decision"].tolist()))
    return replay_games(model, games, observe)


@pytest.fixture(scope="module")
def observations(replay: Replay, expected: Any) -> list[Observation]:
    recorded = list(zip(expected["obs_game"].tolist(), expected["obs_decision"].tolist()))
    replayed = [(obs.game, obs.decision) for obs in replay.observations]
    if replayed != recorded:
        pytest.fail("replay diverged before reaching every recorded observation; "
                    "see test_recorded_games_replay_with_identical_legality")
    return replay.observations


def test_recorded_games_replay_with_identical_legality(
    replay: Replay, games: list[Game], expected: Any,
) -> None:
    offsets = expected["action_offsets"]
    for g, game in enumerate(games):
        phases = expected["decision_phases"][offsets[g]:offsets[g + 1]]
        masks = expected["decision_legal_masks"][offsets[g]:offsets[g + 1]]
        where = f"game {g} ({game.num_players}p seed={game.seed})"
        for d, (phase, mask) in enumerate(zip(replay.phases[g], replay.legal_masks[g])):
            assert phase == phases[d], (
                f"{where} decision {d}: phase {DecisionPhase(phase).name}, "
                f"expected {DecisionPhase(int(phases[d])).name}"
            )
            np.testing.assert_array_equal(
                mask, masks[d], err_msg=f"{where} decision {d} ({DecisionPhase(phase).name}) legal mask",
            )
        assert replay.ended[g], (
            f"{where}: replay stopped after {len(replay.phases[g])} of {len(game.actions)} "
            "recorded decisions, or the game did not end on the last one"
        )


def test_fixture_covers_every_decision_phase(expected: Any) -> None:
    assert set(expected["obs_phase"].tolist()) == {int(p) for p in DecisionPhase}


def test_evaluator_inputs_match(observations: list[Observation], games: list[Game], expected: Any) -> None:
    for i, obs in enumerate(observations):
        where = _describe(obs, games)
        tokens = torch.from_numpy(expected["tokens"][i])
        assert torch.equal(obs.tokens, tokens), f"{where}: v2 tokens changed: {_token_diff(obs.tokens, tokens)}"
        relations = torch.from_numpy(expected["relations"][i])
        assert torch.equal(obs.relations, relations), (
            f"{where}: v2 relation planes changed at (plane, query, key) "
            f"{torch.nonzero(obs.relations != relations)[:8].tolist()}"
        )
        assert torch.equal(obs.legal_mask, torch.from_numpy(expected["legal_masks"][i])), (
            f"{where}: legal mask changed"
        )


def test_evaluator_outputs_match(observations: list[Observation], games: list[Game], expected: Any) -> None:
    for i, obs in enumerate(observations):
        where = _describe(obs, games)
        _assert_outputs_close(obs.logits, expected["logits"][i], f"{where}: policy logits")
        _assert_outputs_close(obs.values, expected["values"][i], f"{where}: values")


def _game_batches(observations: list[Observation]) -> dict[int, list[int]]:
    batches: dict[int, list[int]] = {}
    for i, obs in enumerate(observations):
        batches.setdefault(obs.game, []).append(i)
    return batches


def test_batched_leaf_evaluation_matches(
    model: torch.nn.Module, observations: list[Observation], expected: Any,
) -> None:
    """MCTS's batched path extracts from raw state arrays."""
    evaluator = NNEvaluator(model, torch.device("cpu"), MAX_PLAYERS)
    io = _ModelIO(model)
    try:
        for g, rows in _game_batches(observations).items():
            evaluator.evaluate_leaves(
                [observations[i].state_array for i in rows], expected["legal_masks"][rows],
            )
            tokens, legal_mask, relations, logits, values = io.pop()
            assert torch.equal(tokens, torch.from_numpy(expected["tokens"][rows])), f"game {g}: batched tokens"
            assert torch.equal(
                relations[:, :V2_RELATION_PLANES], torch.from_numpy(expected["relations"][rows]),
            ), f"game {g}: batched relation planes"
            assert torch.equal(legal_mask, torch.from_numpy(expected["legal_masks"][rows]))
            _assert_outputs_close(logits, expected["logits"][rows], f"game {g}: batched logits")
            _assert_outputs_close(values, expected["values"][rows], f"game {g}: batched values")
    finally:
        io.close()


def test_eval_server_wire_inputs_and_sparse_forward_match(
    checkpoint: tuple[torch.nn.Module, TrainingConfig], observations: list[Observation], expected: Any,
) -> None:
    """Self-play workers send fp16 tokens, masks and sparse relation records."""
    model, config = checkpoint
    batches = _game_batches(observations)
    shared = SharedEvalBuffers(
        1, max(map(len, batches.values())), MAX_PLAYERS, input_spec=get_model_input_spec(config),
    )
    shared.init_bitmap([(0, 1)])
    worker = RemoteEvaluator(MAX_PLAYERS, shared, 0)
    setattr(worker, "_request_eval", lambda _n: None)
    for g, rows in batches.items():
        n = len(rows)
        worker.evaluate_leaves([observations[i].state_array for i in rows], expected["legal_masks"][rows])
        tokens = expected["tokens"][rows]
        np.testing.assert_array_equal(
            shared.get_input_states_np(0)[:n], tokens.astype(np.float16), err_msg=f"game {g}: wire tokens",
        )
        np.testing.assert_array_equal(
            shared.get_input_legal_mask_np(0)[:n], expected["legal_masks"][rows], err_msg=f"game {g}: wire mask",
        )
        coords = shared.get_input_relation_coords_np(0)[:n].copy()
        planes = np.stack([_materialize_relation_coords_np(c, num_tokens=tokens.shape[1]) for c in coords])
        np.testing.assert_array_equal(
            planes[:, :V2_RELATION_PLANES], expected["relations"][rows],
            err_msg=f"game {g}: wire relation records",
        )
        with torch.inference_mode():
            logits, values = model(
                torch.from_numpy(tokens), torch.from_numpy(expected["legal_masks"][rows]),
                torch.from_numpy(coords),
            )
        _assert_outputs_close(logits, expected["logits"][rows], f"game {g}: sparse-relation logits")
        _assert_outputs_close(values, expected["values"][rows], f"game {g}: sparse-relation values")


def test_live_checkpoint_config_builds_same_architecture(expected: Any) -> None:
    """The live-play checkpoint's saved config must still build its model."""
    config = TrainingConfig.from_json(str(expected["reference_config_json"]))
    assert config.model_path == V2_MODEL_PATH
    shapes = {name: list(t.shape) for name, t in create_model(config).state_dict().items()}
    assert shapes == json.loads(str(expected["reference_param_shapes"]))


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _reference(reference_checkpoint: Path | None) -> tuple[str, str]:
    if reference_checkpoint is None:
        with np.load(EXPECTED_PATH, allow_pickle=False) as data:
            return str(data["reference_config_json"]), str(data["reference_param_shapes"])
    cp = load_checkpoint(reference_checkpoint, torch.device("cpu"))
    shapes = {name: list(t.shape) for name, t in cp["model_state_dict"].items()}
    return cp["config_json"], json.dumps(shapes)


def capture(reference_checkpoint: Path | None) -> None:
    reference_config_json, reference_shapes = _reference(reference_checkpoint)
    config_values = json.loads(reference_config_json)
    assert config_values["model_path"] == V2_MODEL_PATH, config_values.get("model_path")
    config_values.update(CAPTURE_DIMS)
    config_json = json.dumps(config_values)

    torch.manual_seed(0)
    model = create_model(TrainingConfig.from_json(config_json)).eval()
    # Break zero-initialized heads and relation biases so every input matters.
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param), alpha=0.1)
    FIXTURE_DIR.mkdir(exist_ok=True)
    # Same structure as save_checkpoint, but keeping the reference config's
    # original field set so loading exercises old-config defaults.
    checkpoint: Checkpoint = {
        "epoch": 0, "model_state_dict": model.state_dict(), "trainer_state": {},
        "config_json": config_json, "metrics": {}, "buffer_stats": {},
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

    games = []
    for num_players, seed, rules in CAPTURE_GAMES:
        game = Game(num_players, seed, rules["acq_same_president"],
                    rules["allow_positive_income_closing"], np.empty(0, np.int16))
        game.actions = np.array(list(play_random_decisions(game.new_state(), seed)), dtype=np.int16)
        games.append(game)

    model, _, _ = load_model_from_checkpoint(CHECKPOINT_PATH, torch.device("cpu"))
    full = replay_games(model, games, observe=None)
    assert all(full.ended), "recorded games must replay to game over"
    invest = int(DecisionPhase.DPHASE_INVEST)
    observed = [
        obs for obs in full.observations
        if obs.phase != invest or obs.decision % CAPTURE_INVEST_STRIDE == 0
    ]
    assert {obs.phase for obs in observed} == {int(p) for p in DecisionPhase}
    assert all(int(obs.relations.max()) <= 1 for obs in observed)
    values = torch.stack([obs.values for obs in observed])
    assert values.std(dim=0).min() > 1e-3, "capture model values are insensitive to inputs"

    np.savez_compressed(
        EXPECTED_PATH,
        game_players=np.array([g.num_players for g in games], np.int8),
        game_seeds=np.array([g.seed for g in games], np.int64),
        game_acq_same_president=np.array([g.acq_same_president for g in games]),
        game_positive_income_closing=np.array([g.allow_positive_income_closing for g in games]),
        actions=np.concatenate([g.actions for g in games]),
        action_offsets=np.cumsum([0] + [len(g.actions) for g in games]),
        decision_phases=np.concatenate(full.phases).astype(np.int8),
        decision_legal_masks=np.concatenate(full.legal_masks).reshape(-1, int(UNIFIED_LOGIT_DIM)),
        obs_game=np.array([obs.game for obs in observed], np.int16),
        obs_decision=np.array([obs.decision for obs in observed], np.int16),
        obs_phase=np.array([obs.phase for obs in observed], np.int8),
        tokens=torch.stack([obs.tokens for obs in observed]).numpy(),
        relations=torch.stack([obs.relations for obs in observed]).numpy(),
        legal_masks=torch.stack([obs.legal_mask for obs in observed]).numpy(),
        logits=torch.stack([obs.logits for obs in observed]).numpy(),
        values=values.numpy(),
        reference_config_json=np.array(reference_config_json),
        reference_param_shapes=np.array(reference_shapes),
    )
    print(f"Captured {len(observed)} observations from {sum(map(len, full.phases))} decisions "
          f"in {len(games)} games -> {FIXTURE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", action="store_true", required=True)
    parser.add_argument("--reference-checkpoint", type=Path, default=None,
                        help="live v2 checkpoint to take config/shapes from (default: keep the fixture's)")
    capture(parser.parse_args().reference_checkpoint)
