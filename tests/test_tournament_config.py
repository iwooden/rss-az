from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from core.actions import enumerate_legal_actions_py
from core.data import MAX_ACTION_SIZE, GameConstants, GamePhases
from core.state import GameState
from entities.player import PLAYERS
from entities.turn import TURN
from mcts.evaluator import NNEvaluator, is_cross_president_acq_offer_state
from train import tournament
from train.config import MCTSConfig, TrainingConfig
from train.tournament import (
    EngineRules,
    ModelEntry,
    ModelPlayer,
    _resolve_game_rules,
    _resolve_tournament_num_players,
    _search_rules,
    _validate_tournament_num_players,
)

V2_CONFIG = TrainingConfig(num_players=0, min_players=3, max_players=5)
V3_CONFIG = TrainingConfig(
    num_players=0, min_players=3, max_players=5,
    v3_behavior=True, acq_same_president=False,
)


def _entry(path: str, config: TrainingConfig) -> ModelEntry:
    return ModelEntry(Path(path), 0, torch.nn.Identity(), config, path)


def test_tournament_num_players_defaults_to_effective_minimum() -> None:
    assert _resolve_tournament_num_players(TrainingConfig(num_players=4), None) == 4
    assert (
        _resolve_tournament_num_players(
            TrainingConfig(num_players=0, min_players=3, max_players=5),
            None,
        )
        == 3
    )


def test_tournament_num_players_accepts_configured_mixed_count() -> None:
    config = TrainingConfig(num_players=0, min_players=3, max_players=5)

    assert _resolve_tournament_num_players(config, 5) == 5


def test_tournament_num_players_rejects_out_of_range_count() -> None:
    config = TrainingConfig(num_players=0, min_players=3, max_players=5)

    with pytest.raises(ValueError, match="configured player range 3-5"):
        _resolve_tournament_num_players(config, 2)


def test_tournament_validates_all_checkpoints_support_selected_count() -> None:
    entries = [
        _entry("mixed.pt", TrainingConfig(num_players=0, min_players=3, max_players=5)),
        _entry("single.pt", TrainingConfig(num_players=4)),
    ]

    _validate_tournament_num_players(entries, 4)

    with pytest.raises(ValueError, match="single.pt supports player range 4-4"):
        _validate_tournament_num_players(entries, 3)


def test_play_game_reads_turn_fields_via_entity(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_search(
        state: object,
        evaluator: object,
        mcts_config: MCTSConfig,
        rng: np.random.Generator,
        *,
        state_pool: object | None = None,
    ) -> SimpleNamespace:
        del state, evaluator, mcts_config, rng, state_pool
        return SimpleNamespace(
            legal_actions=np.array([0], dtype=np.uint16),
            visit_counts=np.array([1], dtype=np.int32),
        )

    class FakeDriver:
        def apply_action(
            self,
            state: object,
            action: int,
            history: list[tuple[int, int]] | None = None,
        ) -> int:
            del state, action, history
            return tournament.STATUS_GAME_OVER

    monkeypatch.setattr(tournament, "run_search", fake_run_search)
    monkeypatch.setattr(tournament, "DRIVER", FakeDriver())

    rules = EngineRules(v3_behavior=False, acq_same_president=True)
    mcts_config = MCTSConfig(num_simulations=1, num_players=4)
    net_worths = tournament._play_game(
        players=[
            ModelPlayer(Mock(spec=NNEvaluator), mcts_config, rules) for _ in range(4)
        ],
        seat_to_model=[0, 1, 2, 3],
        num_players=4,
        max_players=5,
        game_rules=rules,
        game_seed=123,
        rng=np.random.default_rng(0),
        state_pool=object(),  # type: ignore[arg-type]
    )

    assert len(net_worths) == 4


def test_game_rules_default_to_most_permissive_checkpoint() -> None:
    v2 = _entry("v2.pt", V2_CONFIG)
    v3 = _entry("v3.pt", V3_CONFIG)

    assert _resolve_game_rules([v2, v2]) == EngineRules(False, True)
    assert _resolve_game_rules([v2, v3]) == EngineRules(True, False)
    assert _resolve_game_rules([v2, v3], v3_behavior=False, acq_same_president=True) == (
        EngineRules(False, True)
    )


def test_search_rules_stay_within_training_and_game_rules() -> None:
    v3_game = EngineRules(v3_behavior=True, acq_same_president=False)
    legacy_game = EngineRules(v3_behavior=False, acq_same_president=True)
    v2 = EngineRules.from_config(V2_CONFIG)
    v3 = EngineRules.from_config(V3_CONFIG)

    assert _search_rules(v3_game, v2) == legacy_game
    assert _search_rules(v3_game, v3) == v3_game
    assert _search_rules(legacy_game, v3) == legacy_game
    # Legacy rules lack v3's cross-president price floors, so a legacy model
    # trained with cross-president offers keeps the game's v3 behavior.
    assert _search_rules(v3_game, EngineRules(False, False)) == v3_game


def _legal_actions(state: GameState) -> list[int]:
    buf = np.zeros(MAX_ACTION_SIZE, dtype=np.uint16)
    return buf[:enumerate_legal_actions_py(state, buf)].tolist()


def _trade_history(state: GameState, num_players: int) -> list[int]:
    return [
        PLAYERS[p].get_share_buys(state, c) + PLAYERS[p].get_share_sells(state, c)
        for p in range(num_players)
        for c in range(int(GameConstants.NUM_CORPS))
    ]


def test_mixed_rule_games_search_each_seat_under_its_training_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Random v3-rules games with a v2 checkpoint in two of three seats."""
    num_players = 3
    game_rules = _resolve_game_rules(
        [_entry("v2.pt", V2_CONFIG), _entry("v3.pt", V3_CONFIG)],
    )
    counts = {"cross_offers": 0, "forced": 0, "hidden_trades": 0}
    real_search_state = tournament._search_state

    def checked_search_state(
        state: GameState, rules: EngineRules, n: int, max_players: int,
    ) -> GameState:
        before = state._array.copy()
        view = real_search_state(state, rules, n, max_players)
        np.testing.assert_array_equal(state._array, before)
        if view is state:
            return view
        view_legal = _legal_actions(view)
        assert set(view_legal) <= set(_legal_actions(state))
        counts["forced"] += len(view_legal) == 1
        if TURN.get_phase(state) in (GamePhases.PHASE_INVEST, GamePhases.PHASE_BID):
            assert _trade_history(view, n) == _trade_history(state, n)
        else:
            assert not any(_trade_history(view, n))
            counts["hidden_trades"] += any(_trade_history(state, n))
        return view

    def fake_run_search(
        state: GameState,
        evaluator: str,
        mcts_config: MCTSConfig,
        rng: np.random.Generator,
        *,
        state_pool: object | None = None,
    ) -> SimpleNamespace:
        del mcts_config, state_pool
        if evaluator == "v2":
            assert (state.v3_behavior, state.acq_same_president) == (False, True)
            counts["cross_offers"] += is_cross_president_acq_offer_state(state)
        else:
            assert (state.v3_behavior, state.acq_same_president) == (True, False)
        legal = np.array(_legal_actions(state), dtype=np.uint16)
        return SimpleNamespace(legal_actions=legal, visit_counts=rng.random(len(legal)))

    monkeypatch.setattr(tournament, "_search_state", checked_search_state)
    monkeypatch.setattr(tournament, "run_search", fake_run_search)
    mcts_config = MCTSConfig(num_simulations=1, num_players=num_players)
    players = [
        ModelPlayer("v2", mcts_config,
                    _search_rules(game_rules, EngineRules.from_config(V2_CONFIG))),
        ModelPlayer("v3", mcts_config,
                    _search_rules(game_rules, EngineRules.from_config(V3_CONFIG))),
    ]

    for seed in range(8):
        tournament._play_game(
            players, [0, 1, 0], num_players, 5, game_rules, seed,
            np.random.default_rng(seed), state_pool=None,  # type: ignore[arg-type]
        )

    assert counts["cross_offers"] > 0
    assert counts["forced"] > 0
    assert counts["hidden_trades"] > 0
