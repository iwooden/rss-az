from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.actions import enumerate_legal_actions_py
from core.data import MAX_ACTION_SIZE, GameConstants, GamePhases
from core.state import GameState
from entities.player import PLAYERS
from entities.turn import TURN
from games import get_game, rss_tournament
from games.rss_live import is_cross_president_acq_offer_state
from games.rss_tournament import EngineRules, search_rules
from mcts.search import StatePool
from nn import create_model, get_model_input_spec
from train import tournament
from train.config import MCTSConfig, TrainingConfig
from train.tournament import (
    ModelEntry,
    ModelPlayer,
    _entry_mcts_config,
    _parse_checkpoint_spec,
    _resolve_game_rules,
    _resolve_tournament_num_players,
    _validate_tournament_num_players,
)

V2_CONFIG = TrainingConfig(num_players=0, min_players=3, max_players=5)
V3_CONFIG = TrainingConfig(
    num_players=0, min_players=3, max_players=5,
    v3_behavior=True, acq_same_president=False, max_acq_price_actions=8,
)


def _entry(path: str, config: TrainingConfig) -> ModelEntry:
    return ModelEntry(Path(path), 0, torch.nn.Identity(), config, path)


def test_checkpoint_spec_parses_optional_c_puct_override() -> None:
    assert _parse_checkpoint_spec(" cp.pt ") == (Path("cp.pt"), None)
    assert _parse_checkpoint_spec("dir/cp.pt@c_puct=1.2") == (Path("dir/cp.pt"), 1.2)
    with pytest.raises(ValueError, match="expected c_puct=VALUE"):
        _parse_checkpoint_spec("cp.pt@sims=10")
    with pytest.raises(ValueError, match="invalid c_puct value"):
        _parse_checkpoint_spec("cp.pt@c_puct=high")


def test_entry_c_puct_override_beats_checkpoint_setting() -> None:
    config = TrainingConfig(num_players=3, c_puct_final=1.7)
    overrides = {"num_simulations": 50}
    plain = _entry_mcts_config(_entry("cp.pt", config), 3, overrides)
    entry = ModelEntry(Path("cp.pt"), 0, torch.nn.Identity(), config, "cp", c_puct=1.2)
    tuned = _entry_mcts_config(entry, 3, overrides)
    assert (plain.c_puct, plain.num_simulations) == (1.7, 50)
    assert (tuned.c_puct, tuned.num_simulations) == (1.2, 50)
    assert overrides == {"num_simulations": 50}


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

    assert search_rules(v3_game, v2) == legacy_game
    # Each checkpoint searches under its own ACQ price cap.
    assert search_rules(v3_game, v3) == EngineRules(True, False, 8)
    assert search_rules(legacy_game, v3) == EngineRules(False, True, 8)
    # Legacy rules lack v3's cross-president price floors, so a legacy model
    # trained with cross-president offers keeps the game's v3 behavior.
    assert search_rules(v3_game, EngineRules(False, False)) == v3_game


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
    real_search_state = rss_tournament.search_state

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
        # V3 seats get a view only for their own ACQ price cap.
        if view.v3_behavior or TURN.get_phase(state) in (
            GamePhases.PHASE_INVEST, GamePhases.PHASE_BID,
        ):
            assert _trade_history(view, n) == _trade_history(state, n)
        else:
            assert not any(_trade_history(view, n))
            counts["hidden_trades"] += any(_trade_history(state, n))
        return view

    def fake_run_search(
        game: object,
        state: GameState,
        evaluator: str,
        mcts_config: MCTSConfig,
        rng: np.random.Generator,
        *,
        state_pool: object | None = None,
    ) -> SimpleNamespace:
        del game, mcts_config, state_pool
        rules = (state.v3_behavior, state.acq_same_president, state.max_acq_price_actions)
        if evaluator == "v2":
            assert rules == (False, True, 0)
            counts["cross_offers"] += is_cross_president_acq_offer_state(state)
        else:
            assert rules == (True, False, 8)
        legal = np.array(_legal_actions(state), dtype=np.uint16)
        return SimpleNamespace(legal_actions=legal, visit_counts=rng.random(len(legal)))

    monkeypatch.setattr(rss_tournament, "search_state", checked_search_state)
    monkeypatch.setattr(tournament, "run_search", fake_run_search)
    mcts_config = MCTSConfig(num_simulations=1, num_players=num_players)
    players = [
        ModelPlayer(name, mcts_config, game_rules.seat(config, name, num_players, 5)[1])
        for name, config in (("v2", V2_CONFIG), ("v3", V3_CONFIG))
    ]
    game = get_game(game_rules.game_config(V3_CONFIG))

    for seed in range(8):
        tournament._play_game(
            game, players, [0, 1, 0], num_players, 5, seed,
            np.random.default_rng(seed), state_pool=None,  # type: ignore[arg-type]
        )

    assert counts["cross_offers"] > 0
    assert counts["forced"] > 0
    assert counts["hidden_trades"] > 0


def test_each_checkpoint_reads_its_own_layout_in_a_shared_game() -> None:
    """A v2 and a v3 model play one game, each evaluator built from its own
    model's input spec and terminal settings, filling tokens in its layout."""
    configs = [
        dataclasses.replace(config, model_path=model_path, terminal_rank_tau=tau,
                            d_model=32, d_proj=8, num_heads=4, num_layers=1)
        for config, model_path, tau in (
            (V2_CONFIG, "nn/transformer-v2.py", 0.0), (V3_CONFIG, "nn/transformer-v3.py", 0.05),
        )
    ]
    entries = [
        ModelEntry(Path(f"{i}.pt"), 0, create_model(config), config, str(i))
        for i, config in enumerate(configs)
    ]
    game_rules = _resolve_game_rules(entries)
    filled: set[tuple[int, int]] = set()

    def spy(fill):
        def fill_tokens(states, tokens, *, max_players, layout_version):
            filled.add((layout_version, tokens.shape[-1]))
            fill(states, tokens, max_players=max_players, layout_version=layout_version)
        return fill_tokens

    game = get_game(game_rules.game_config(configs[0]))
    game = dataclasses.replace(
        game, fill_tokens=spy(game.fill_tokens), fill_tokens_batch=spy(game.fill_tokens_batch),
    )
    mcts_config = MCTSConfig(num_simulations=2, num_players=3, search_batch_size=2)
    players = tournament._build_players(
        game, entries, torch.device("cpu"), 3, 5, game_rules, [mcts_config] * 2, [0.25, 0.75],
    )

    for player, entry, weight in zip(players, entries, [0.25, 0.75], strict=True):
        evaluator = player.evaluator
        assert evaluator.model is entry.model
        assert evaluator.input_spec == get_model_input_spec(entry.config)
        assert (evaluator.terminal_rank_weight, evaluator.terminal_rank_tau) == (
            weight, entry.config.terminal_rank_tau,
        )
    scores = tournament._play_game(
        game, players, [0, 1, 0], 3, 5, 1, np.random.default_rng(1),
        StatePool(game, 2 * (mcts_config.num_simulations + 1), game.state_size(5)),
    )
    assert len(scores) == 3
    assert filled == {(2, game.token_dim(2)), (3, game.token_dim(3))}
    assert game.token_dim(2) != game.token_dim(3)
