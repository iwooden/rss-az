"""1889 tournaments (``train/tournament.py``; 1889/v1-design.md §8): games through the
``Game`` adapter on a seat-balanced schedule, seats counted from the player who acts
first, a pairwise score with its standard error, and checkpoints of one game and one
set of engine options; v0 and v1, on their two token layouts, in one game; and v1 and
v2, on layout 2's tokens and records and layout 3's packed vectors (1889/v2-design.md
§7.2)."""

from __future__ import annotations

import dataclasses
import itertools
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from g1889.entities.turn import TURN
from games import get_game
from nn import create_model
from train import tournament
from train.checkpoint import save_checkpoint
from train.config import MCTSConfig, TrainingConfig
from train.tournament import ModelEntry, ModelPlayer, SharedRules, _resolve_game_rules

CONFIG = TrainingConfig(
    game="1889", model_path="nn/transformer-1889.py", num_players=0, min_players=3, max_players=5,
    d_model=32, num_heads=4, num_layers=1, zero_sum_values=True,
)
GAME = get_game(CONFIG)
V1_CONFIG = dataclasses.replace(CONFIG, model_path="nn/transformer-1889-v1.py", terminal_rank_tau=0.05)
V2_CONFIG = dataclasses.replace(V1_CONFIG, model_path="nn/transformer-1889-v2.py")


@pytest.fixture(autouse=True)
def one_thread():
    """Tiny models search fastest on one thread."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("tournament")
    configs = {
        "a": CONFIG,
        "b": dataclasses.replace(CONFIG, terminal_rank_tau=0.05),
        "uncapped": dataclasses.replace(CONFIG, auction_pass_payout_cap=0),
        "v1": V1_CONFIG,
        "v2": V2_CONFIG,
    }
    for seed, (name, config) in enumerate(configs.items()):
        torch.manual_seed(seed)
        save_checkpoint(out / f"{name}.pt", seed, create_model(config), {}, config, {}, {})
    return {name: out / f"{name}.pt" for name in configs}


def _groups(pattern: str, text: str) -> tuple[str, ...]:
    match = re.search(pattern, text)
    assert match is not None, pattern
    return match.groups()


def _tournament(monkeypatch, capsys, *args: str) -> str:
    monkeypatch.setattr(sys, "argv", ["tournament", *args])
    tournament.main()
    return capsys.readouterr().out


def _even_arrangements(num_players: int) -> list[tuple[int, ...]]:
    """Both models in every game, as evenly split as the seats allow: at an odd
    count, each model is the majority in half the games."""
    return [s for s in itertools.product((0, 1), repeat=num_players) if abs(s.count(0) - s.count(1)) <= 1]


@pytest.mark.parametrize(("num_players", "workers"), [(3, 1), (4, 1), (5, 3)])
def test_two_models_play_every_even_arrangement_and_get_a_pairwise_score(
    checkpoints, monkeypatch, capsys, num_players, workers,
):
    out = _tournament(
        monkeypatch, capsys, f"{checkpoints['a']},{checkpoints['b']}", "--device", "cpu",
        "--simulations", "1", "--games-per-pair", "1", "--num-players", str(num_players),
        "--sample-moves", "10", "--workers", str(workers),
    )
    games = re.findall(r"Game \d+/\d+ \([\d.]+s\): (.*) → ", out)
    seatings = sorted(tuple(int(m) for m in re.findall(r"P\d=\[(\d)\]", g)) for g in games)
    assert seatings == _even_arrangements(num_players)
    assert f"All {len(seatings)} games completed" in out
    assert "Moves sampled from search visits: each game's first 10" in out
    assert "input layout 1" in out

    mean, se, played = _groups(r"\[0\] vs \[1\]: (\d\.\d+) ± (\d\.\d+)  \((\d+) games\)", out)
    assert 0 <= float(mean) <= 1 and float(se) > 0 and int(played) == len(seatings)
    per_seat = len(seatings) // 2
    for model in range(2):
        counts = _groups(rf"\[{model}\] transformer-1889 epoch {model} +([\d ]+)\n", out)[0].split()
        assert counts == [str(per_seat)] * num_players
    seat_scores = _groups(r"Seat score \(0\.5 = even\) +([\d. ]+)\n", out)[0].split()
    assert len(seat_scores) == num_players
    assert np.mean([float(s) for s in seat_scores]) == pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_v0_and_v1_share_a_game_each_on_its_own_layout(checkpoints, monkeypatch, capsys, num_players):
    """The evaluators of one game fill different layouts, under their own terminal
    values, and v1's trunk runs without the candidate rows."""
    fills: list[set[tuple[int, int]]] = []
    trunk_rows: list[set[int]] = []
    evaluators: list[tournament.NNEvaluator] = []

    class Spy(tournament.NNEvaluator):
        def __init__(self, game, model, *args, **kwargs):
            filled: set[tuple[int, int]] = set()
            rows: set[int] = set()

            def spy(fill):
                def wrapped(state, out, *, max_players, layout_version):
                    filled.add((layout_version, out.shape[-1]))
                    fill(state, out, max_players=max_players, layout_version=layout_version)
                return wrapped

            game = dataclasses.replace(
                game, fill_tokens=spy(game.fill_tokens), fill_tokens_batch=spy(game.fill_tokens_batch),
            )
            model.blocks[0].register_forward_hook(lambda _block, inputs, _output: rows.add(inputs[0].shape[1]))
            super().__init__(game, model, *args, **kwargs)
            fills.append(filled)
            trunk_rows.append(rows)
            evaluators.append(self)

    monkeypatch.setattr(tournament, "NNEvaluator", Spy)
    # Two of the round's arrangements, for time: every one seats both models.
    schedule = tournament._balanced_schedule
    monkeypatch.setattr(tournament, "_balanced_schedule", lambda *args: list(schedule(*args))[:2])
    # Leaf batches of 2, so the single-state and the batch fills both run.
    out = _tournament(
        monkeypatch, capsys, f"{checkpoints['a']},{checkpoints['v1']}", "--device", "cpu",
        "--simulations", "2", "--search-batch-size", "2", "--games-per-pair", "1",
        "--num-players", str(num_players), "--sample-moves", "10",
    )
    assert "All 2 games completed" in out
    assert "[0] transformer-1889 epoch 0:" in out and "input layout 1" in out
    assert "[1] transformer-1889-v1 epoch 3:" in out and "input layout 2" in out
    mean, played = _groups(r"\[0\] vs \[1\]: (\d\.\d+) ± \d\.\d+  \((\d+) games\)", out)
    assert 0 <= float(mean) <= 1 and int(played) == 2

    assert fills == [{(1, 108)}, {(2, 109)}]
    assert trunk_rows == [{95}, {80}]
    assert [(e.layout_version, e.token_dim, e.terminal_rank_tau) for e in evaluators] == [(1, 108, 0.0), (2, 109, 0.05)]


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_v1_and_v2_share_a_game_each_on_its_own_layout(checkpoints, monkeypatch, capsys, num_players):
    """v1's evaluator fills tokens and records and v2's packed vectors, in one
    game: each forward takes its own model's arguments, v2's two, and v2's trunk
    is its 45 rows."""
    fills: list[set[tuple[str, int, int]]] = []
    trunk_rows: list[set[int]] = []
    arguments: list[set[int]] = []
    evaluators: list[tournament.NNEvaluator] = []

    class Spy(tournament.NNEvaluator):
        def __init__(self, game, model, *args, **kwargs):
            filled: set[tuple[str, int, int]] = set()
            rows: set[int] = set()
            counts: set[int] = set()

            def spy(kind, fill):
                def wrapped(state, out, *, max_players, layout_version):
                    filled.add((kind, layout_version, out.shape[-1]))
                    fill(state, out, max_players=max_players, layout_version=layout_version)
                return wrapped

            game = dataclasses.replace(
                game, fill_tokens=spy("tokens", game.fill_tokens), fill_tokens_batch=spy("tokens", game.fill_tokens_batch),
                fill_inputs=spy("vectors", game.fill_inputs), fill_inputs_batch=spy("vectors", game.fill_inputs_batch),
            )
            model.blocks[0].register_forward_hook(lambda _block, inputs, _output: rows.add(inputs[0].shape[1]))
            model.register_forward_pre_hook(lambda _model, inputs: counts.add(len(inputs)))
            super().__init__(game, model, *args, **kwargs)
            fills.append(filled)
            trunk_rows.append(rows)
            arguments.append(counts)
            evaluators.append(self)

    monkeypatch.setattr(tournament, "NNEvaluator", Spy)
    schedule = tournament._balanced_schedule
    monkeypatch.setattr(tournament, "_balanced_schedule", lambda *args: list(schedule(*args))[:2])
    out = _tournament(
        monkeypatch, capsys, f"{checkpoints['v1']},{checkpoints['v2']}", "--device", "cpu",
        "--simulations", "2", "--search-batch-size", "2", "--games-per-pair", "1",
        "--num-players", str(num_players), "--sample-moves", "10",
    )
    assert "All 2 games completed" in out
    assert "[0] transformer-1889-v1 epoch 3:" in out and "input layout 2" in out
    assert "[1] transformer-1889-v2 epoch 4:" in out and "input layout 3" in out
    mean, played = _groups(r"\[0\] vs \[1\]: (\d\.\d+) ± \d\.\d+  \((\d+) games\)", out)
    assert 0 <= float(mean) <= 1 and int(played) == 2

    assert fills == [{("tokens", 2, 109)}, {("vectors", 3, 5108)}]
    assert trunk_rows == [{80}, {45}]
    assert arguments == [{3}, {2}]
    assert [(e.layout_version, e.input_spec.packed, e.terminal_rank_tau) for e in evaluators] == [
        (2, False, 0.05), (3, True, 0.05),
    ]


def test_worker_processes_play_the_games_the_seed_gives(checkpoints, monkeypatch):
    cpu = torch.device("cpu")
    entries = [
        ModelEntry(path, 0, *tournament._load_model(path, cpu)[:2], path.name)
        for path in (checkpoints["a"], checkpoints["b"])
    ]
    rules = _resolve_game_rules(entries)
    configs = [MCTSConfig(num_simulations=1, num_players=3, search_batch_size=1)] * 2
    results, _ = tournament.run_tournament(entries, cpu, 3, 5, configs, 1, 42, [0.75, 0.25], rules, 5, workers=2)
    assert [tuple(r.seat_to_model) for r in results] == _even_arrangements(3)

    # Any process plays a scheduled game the same way.
    monkeypatch.setattr(tournament, "_worker", {})
    tournament._init_worker([(e.path, e.c_puct) for e in entries], "cpu", 3, 5, configs, [0.75, 0.25], rules, 5)
    for game_num in (1, 6):
        result = results[game_num - 1]
        task = (game_num, tuple(result.seat_to_model), result.seed, 42)
        assert tournament._play_task(task)[:3] == (game_num, result.seed, result.scores)


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_seats_count_from_the_player_who_acts_first(monkeypatch, num_players):
    searched: list[tuple[str, int]] = []
    canonical: list[list[int]] = []

    def fake_run_search(game, state, evaluator, mcts_config, rng, *, state_pool=None):
        del game, mcts_config, state_pool
        searched.append((evaluator, TURN.active_player(state)))
        legal = np.empty(GAME.policy.max_action_size, dtype=np.uint16)
        n = GAME.enumerate_policy_actions(state, legal)
        return SimpleNamespace(legal_actions=legal[:n].copy(), visit_counts=rng.random(n))

    def final_scores(state):
        canonical.append(GAME.final_scores(state))
        return canonical[-1]

    monkeypatch.setattr(tournament, "run_search", fake_run_search)
    game = dataclasses.replace(GAME, final_scores=final_scores)
    config = MCTSConfig(num_simulations=1, num_players=num_players)
    players = [ModelPlayer(name, config) for name in ("a", "b")]
    seat_to_model = [1] + [0] * (num_players - 1)
    for seed in range(6):
        first = GAME.active_player(GAME.new_game(num_players, 5, seed))
        searched.clear()
        scores = tournament._play_game(
            game, players, seat_to_model, num_players, 5, seed, np.random.default_rng(seed), None,  # type: ignore[arg-type]
        )
        # Only seat 0, the first player, is "b": it makes the game's first decision.
        assert searched[0] == ("b", first)
        assert all((name == "b") == (player == first) for name, player in searched)
        assert scores == [canonical[-1][(seat + first) % num_players] for seat in range(num_players)]


def test_checkpoints_of_two_games_or_two_sets_of_engine_options_are_refused(
    checkpoints, monkeypatch, capsys,
):
    def entry(name: str, config: TrainingConfig) -> ModelEntry:
        return ModelEntry(Path(name), 0, torch.nn.Identity(), config, name)

    a, b = entry("a.pt", CONFIG), entry("b.pt", dataclasses.replace(CONFIG, terminal_rank_tau=0.05))
    rules = _resolve_game_rules([a, b])
    assert rules == SharedRules(tuple(
        (name, getattr(CONFIG, name))
        for name in ("private_sale_same_president", "train_sale_same_president", "auction_pass_payout_cap")
    ))
    assert rules.describe() == (
        "private_sale_same_president=True, train_sale_same_president=True, auction_pass_payout_cap=1"
    )
    with pytest.raises(ValueError, match="different games: a.pt 1889, rss.pt rss"):
        _resolve_game_rules([a, entry("rss.pt", TrainingConfig())])
    with pytest.raises(ValueError, match="RSS options"):
        _resolve_game_rules([a, b], acq_same_president=True)
    unlike = dataclasses.replace(CONFIG, train_sale_same_president=False, auction_pass_payout_cap=2)
    with pytest.raises(ValueError, match=(
        r"different 1889 engine options, which a tournament does not mix: "
        r"train_sale_same_president: a.pt True, c.pt False; auction_pass_payout_cap: a.pt 1, c.pt 2$"
    )):
        _resolve_game_rules([a, entry("c.pt", unlike)])

    with pytest.raises(SystemExit):
        _tournament(monkeypatch, capsys, f"{checkpoints['a']},{checkpoints['uncapped']}", "--device", "cpu")
    assert "auction_pass_payout_cap: a.pt 1, uncapped.pt 0" in capsys.readouterr().out
