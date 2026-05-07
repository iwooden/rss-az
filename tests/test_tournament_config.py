from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train import tournament
from train.config import MCTSConfig, TrainingConfig
from train.tournament import (
    ModelEntry,
    _resolve_tournament_num_players,
    _validate_tournament_num_players,
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

    net_worths = tournament._play_game(
        evaluators=[object(), object(), object(), object()],
        seat_to_model=[0, 1, 2, 3],
        num_players=4,
        max_players=5,
        mcts_config=MCTSConfig(num_simulations=1, num_players=4),
        game_seed=123,
        rng=np.random.default_rng(0),
        state_pool=object(),  # type: ignore[arg-type]
    )

    assert len(net_worths) == 4
