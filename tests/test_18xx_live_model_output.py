import numpy as np

from core.actions import get_decision_phase_py
from core.state import GameState
from mcts.node import MCTSNode
from train.config import MCTSConfig
from train.profile_stats import SearchStats
from utils_18xx import live
from utils_18xx.action_parser import get_legal_actions
from utils_18xx.live import (
    _SearchEngine,
    _format_live_model_output,
    _nonnegative_int,
    _player_names_from_game_data,
    _resolve_live_eval_dtype,
    _resolve_live_search_batch_size,
)


def test_player_names_from_game_data_uses_18xx_order():
    game_data = {
        "players": [
            {"id": 11, "name": "rss-az-1"},
            {"id": 22, "name": "rss-az-2"},
            {"id": 33, "name": "rss-az-3"},
        ],
    }

    assert _player_names_from_game_data(game_data, 3) == [
        "rss-az-1",
        "rss-az-2",
        "rss-az-3",
    ]


def test_live_search_batch_size_defaults_to_checkpoint_config():
    class _Config:
        search_batch_size = 8

    assert _resolve_live_search_batch_size(_Config(), None) == 8
    assert _resolve_live_search_batch_size(_Config(), 3) == 3


def test_live_eval_dtype_defaults_to_checkpoint_config_and_allows_fp32_override():
    class _Config:
        eval_dtype = "bfloat16"

    assert _resolve_live_eval_dtype(_Config(), None) == "bfloat16"
    assert _resolve_live_eval_dtype(_Config(), "float16") == "float16"
    assert _resolve_live_eval_dtype(_Config(), "float32") is None


def test_nonnegative_int_rejects_negative_values():
    assert _nonnegative_int("0") == 0
    assert _nonnegative_int("4") == 4

    import argparse

    try:
        _nonnegative_int("-1")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("negative determinization count should be rejected")


def _search_root(actions, counts):
    root = MCTSNode(active_player_id=0, num_players=3)
    root.legal_actions = np.array(actions, dtype=np.int32)
    root.visit_counts = np.array(counts, dtype=np.int32)
    root.value_sums = np.zeros((len(actions), 3), dtype=np.float32)
    for idx, count in enumerate(counts):
        root.value_sums[idx] = np.array([count, 0, -count], dtype=np.float32)
    root.value_sum = root.value_sums.sum(axis=0)
    root.visit_count = int(root.visit_counts.sum())
    return root


def test_determinized_live_search_sums_root_visits(monkeypatch):
    state = GameState(3)
    state.initialize_game(3, seed=7)

    engine = _SearchEngine.__new__(_SearchEngine)
    engine.determinization_count = 2
    engine.model_output = False
    engine.num_simulations = 7
    engine._evaluator = object()
    engine._rng = np.random.default_rng(123)
    engine._state_pool = object()

    monkeypatch.setattr(
        _SearchEngine,
        "_mcts_config_for",
        lambda self, num_players: MCTSConfig(
            num_simulations=7,
            num_players=num_players,
            search_batch_size=1,
        ),
    )

    class _Deck:
        def __init__(self):
            self.calls = 0

        def determinize_remaining(self, source_state, rng):
            del rng
            self.calls += 1
            return source_state

    fake_deck = _Deck()
    roots = [
        _search_root([10, 20], [2, 5]),
        _search_root([10, 20], [8, 1]),
    ]
    calls = []

    def fake_run_search(
        root_state,
        evaluator,
        config,
        rng,
        *,
        state_pool=None,
        reuse_root=None,
        profile=None,
    ):
        del evaluator, rng, state_pool, profile
        assert root_state is state
        assert config.num_simulations == 7
        assert reuse_root is None
        calls.append(root_state)
        return roots[len(calls) - 1]

    monkeypatch.setattr(live, "DECK", fake_deck)
    monkeypatch.setattr(live, "run_search", fake_run_search)

    action_idx, root, _, search_stats = _SearchEngine._search(
        engine,
        state,
        3,
        reuse_root=object(),
    )

    assert search_stats is None
    assert fake_deck.calls == 2
    assert len(calls) == 2
    assert action_idx == 10
    assert root.legal_actions.tolist() == [10, 20]
    assert root.visit_counts.tolist() == [10, 6]


def test_live_model_output_relabels_analyzer_player_values():
    state = GameState(3)
    state.initialize_game(3, seed=7)
    action_idx = int(get_legal_actions(state)[0][0])

    root = MCTSNode(active_player_id=0, num_players=3)
    root.legal_actions = np.array([action_idx], dtype=np.int32)
    root.priors = np.array([1.0], dtype=np.float32)
    root.default_value = np.array([0.2, -0.1, -0.1], dtype=np.float32)
    root.visit_counts = np.array([3], dtype=np.int32)
    root.value_sums = np.array([[1.5, -0.3, -1.2]], dtype=np.float32)
    root.value_sum = np.array([1.5, -0.3, -1.2], dtype=np.float32)
    root.visit_count = 3

    rendered = _format_live_model_output(
        game_data={
            "id": 4,
            "players": [
                {"id": 11, "name": "rss-az-1"},
                {"id": 22, "name": "rss-az-2"},
                {"id": 33, "name": "rss-az-3"},
            ],
        },
        state=state,
        priors=np.array([1.0], dtype=np.float32),
        values=np.array([0.2, -0.1, -0.1], dtype=np.float32),
        action_ids=np.array([action_idx], dtype=np.int32),
        phase_id=get_decision_phase_py(state),
        num_players=3,
        root=root,
        action_idx=action_idx,
        mcts_config=MCTSConfig(
            num_simulations=3,
            num_players=3,
            search_batch_size=1,
            dirichlet_epsilon=0.0,
        ),
        search_stats=SearchStats(),
        elapsed_secs=0.12,
    )

    assert "### Live Decision: rss-az-1 [INVEST]" in rendered
    assert "NN Values: rss-az-1=+0.200" in rendered
    assert "rss-az-2=-0.100" in rendered
    assert "MCTS Visits" in rendered
    assert "A0GB Value: rss-az-1=+0.500" in rendered
    assert "**Action:" in rendered
    assert "P0" not in rendered
    assert "P1" not in rendered
    assert "P2" not in rendered


def test_live_model_output_can_skip_mcts_for_single_legal_action():
    state = GameState(3)
    state.initialize_game(3, seed=7)
    action_idx = int(get_legal_actions(state)[0][0])

    rendered = _format_live_model_output(
        game_data={
            "id": 4,
            "players": [
                {"id": 11, "name": "rss-az-1"},
                {"id": 22, "name": "rss-az-2"},
                {"id": 33, "name": "rss-az-3"},
            ],
        },
        state=state,
        priors=np.array([1.0], dtype=np.float32),
        values=np.array([0.2, -0.1, -0.1], dtype=np.float32),
        action_ids=np.array([action_idx], dtype=np.int32),
        phase_id=get_decision_phase_py(state),
        num_players=3,
        root=None,
        action_idx=action_idx,
    )

    assert "Search: skipped (single legal action)" in rendered
    assert "NN Values: rss-az-1=+0.200" in rendered
    assert "MCTS: skipped (single legal action)" in rendered
    assert "A0GB Value: skipped (single legal action)" in rendered
    assert "**Action:" in rendered
