"""A 1889 training run and its resume (1889/model-design.md §8 rows 8.3-8.4), on
the CPU at a small size.

- ``train.main.main`` plays an epoch at 3-5 players through ``NNEvaluator``, trains
  on mixed-player batches, and checkpoints; ``--resume latest`` then loads the
  checkpoint and the buffer and runs the next epoch. The stats hook's scalars and
  the per-phase policy metrics reach TensorBoard, each group's, for both epochs.
- A trainer rebuilt from a checkpoint and a saved buffer continues exactly as the
  original: the same batches, losses, and weights.

The local smoke run covers the eval server and the GPU
(``train_configs/1889-v0-smoke.json``).
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from games import get_game
from nn import create_model
from train import main as main_module
from train.checkpoint import load_checkpoint, save_checkpoint
from train.config import TrainingConfig
from train.replay_buffer import FLAG_FROM_POSITION, ReplayBuffer
from train.tb_reader import read_tb_scalars
from train.trainer import Trainer

from tests.g1889.phases.test_random_playouts import _choose

MAX_PLAYERS = 5
SMALL = {
    "d_model": 32,
    "num_layers": 1,
    "games_per_epoch": 3,
    "num_epochs": 1,
    "training_steps_per_epoch": 40,
    "num_simulations": 2,
    "search_batch_size": 2,
    "num_workers": 0,
    "buffer_capacity": 8192,
    "min_buffer_size": 64,
    "batch_size": 64,
    "profile": False,
}


def _smoke_config(tmp_path: Path) -> dict[str, object]:
    """The smoke config at a size for the CPU, writing under ``tmp_path``."""
    fields = json.loads(Path("train_configs/1889-v0-smoke.json").read_text())
    return {
        **fields,
        **SMALL,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "tensorboard_dir": str(tmp_path / "runs"),
    }


def _train(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--device", "cpu", "--no-compile", *args])
    main_module.main()


def test_a_run_checkpoints_and_resumes_with_its_scalars(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_smoke_config(tmp_path)))
    checkpoints = tmp_path / "checkpoints"
    device = torch.device("cpu")

    _train(monkeypatch, "--config", str(config_path))
    first = load_checkpoint(checkpoints / "checkpoint_epoch_0001.pt", device)
    assert first["epoch"] == 0
    first_rows = json.loads((checkpoints / "replay_buffer" / "metadata.json").read_text())["size"]
    assert first_rows == first["buffer_stats"]["size"] > 0
    capsys.readouterr()

    _train(monkeypatch, "--config", str(config_path), "--resume", "latest", "--num-epochs", "2")
    output = capsys.readouterr().out
    step = first["trainer_state"]["global_step"]
    assert isinstance(step, int)
    assert f"Resumed from epoch 0, step {step}" in output
    assert f"Loaded {first_rows:,} examples into replay buffer" in output
    assert "Restored RNG state" in output

    second = load_checkpoint(checkpoints / "checkpoint_epoch_0002.pt", device)
    assert second["epoch"] == 1
    second_step = second["trainer_state"]["global_step"]
    assert isinstance(second_step, int) and second_step > step
    assert any(
        not torch.equal(before, second["model_state_dict"][name])
        for name, before in first["model_state_dict"].items()
    )

    scalars = read_tb_scalars(str(tmp_path / "runs"))
    # The second epoch's rows joined the first's.
    new_rows = scalars["self_play_aggregate/total_examples"][1][1]
    assert [value for _, value in scalars["buffer/size"]] == [first_rows, first_rows + new_rows]

    for epoch_tags in (
        *(f"self_play_{group}/{name}" for group in ("aggregate", "3p", "4p", "5p") for name in (
            "end_bank", "end_bankruptcy", "ors_played", "track_pass_share", "withhold_share", "corps_floated",
            "corp_cash_end", "last_phase_D", "trains_bought_2", "trains_bought_D",
        )),
        "self_play_aggregate/policy/phase/track/decision_count",
        "self_play_aggregate/policy/phase/stock/target_kl_to_prior_mean",
        "epoch/policy_loss_track_avg",
        "epoch/policy_kl_track_avg",
        *(f"epoch/value_loss_{n}p_avg" for n in (3, 4, 5)),
    ):
        assert [step for step, _ in scalars[epoch_tags]] == [1, 2], epoch_tags
    # Each game ended one way.
    for group in ("aggregate", "3p", "4p", "5p"):
        for (_, bank), (_, bankrupt) in zip(
            scalars[f"self_play_{group}/end_bank"], scalars[f"self_play_{group}/end_bankruptcy"], strict=True,
        ):
            assert bank + bankrupt == pytest.approx(1.0)
    for name in ("total_loss_avg", "policy_loss_avg", "value_loss_avg"):
        assert all(np.isfinite(value) for _, value in scalars[f"epoch/{name}"])


def test_a_run_with_buffer_epochs_trains_on_its_newest_epochs_across_resumes(tmp_path, monkeypatch, capsys):
    """Two games an epoch and a window of two epochs, through a graceful shutdown
    mid-epoch: the buffer records each epoch's first game, the epoch replayed on
    resume keeps the game it had, each row carries its game and final net
    worths, and steps follow the share of the window held, not of the capacity."""
    fields = {
        **_smoke_config(tmp_path), "games_per_epoch": 2, "training_steps_per_epoch": 8,
        "buffer_epochs": 2, "buffer_capacity": 65_536,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(fields))
    buffer_dir = tmp_path / "checkpoints" / "replay_buffer"

    def record() -> tuple[list[list[int]], int]:
        metadata = json.loads((buffer_dir / "metadata.json").read_text())
        return metadata["epochs"], metadata["next_game_id"]

    _train(monkeypatch, "--config", str(config_path))
    assert record() == ([[0, 0]], 2)
    # q + Enter during the second epoch's first game: it is saved, untrained.
    stop = threading.Event()
    stop.set()
    with monkeypatch.context() as patch:
        patch.setattr(main_module, "_start_shutdown_listener", lambda: stop)
        _train(patch, "--config", str(config_path), "--resume", "latest", "--num-epochs", "3")
    assert "Trained: no" in capsys.readouterr().out
    assert record() == ([[0, 0], [1, 2]], 3)
    _train(monkeypatch, "--config", str(config_path), "--resume", "latest", "--num-epochs", "3")
    assert "Resumed from epoch 0" in capsys.readouterr().out
    assert record() == ([[0, 0], [1, 2], [2, 5]], 7)

    scalars = read_tb_scalars(str(tmp_path / "runs"))
    played = [int(value) for _, value in scalars["self_play_aggregate/total_examples"]]
    sizes = [int(value) for _, value in scalars["buffer/size"]]
    partial = sizes[1] - sum(played[:2])  # the game played before the shutdown
    assert partial > 0
    assert sizes == [played[0], played[0] + partial + played[1], partial + sum(played)]
    # The second epoch holds that game too; the third leaves the first behind.
    assert [value for _, value in scalars["buffer/window_size"]] == [sizes[0], sizes[1], sizes[2] - played[0]]
    assert [value for _, value in scalars["buffer/window_epochs"]] == [1, 2, 2]
    # By its share of the capacity, each epoch would earn one step.
    assert 8 * sizes[2] < 2 * 65_536
    assert [value for _, value in scalars["epoch/training_steps"]] == [4, 8, 8]

    buffer = _buffer(TrainingConfig.from_json(json.dumps(fields)))
    assert buffer.load(buffer_dir) == sizes[2]
    rows = buffer.rows_oldest_first()
    assert rows["game_ids"].tolist() == sorted(rows["game_ids"].tolist())
    assert set(rows["game_ids"].tolist()) == set(range(7))
    assert int((rows["game_ids"] < 2).sum()) == played[0] and int((rows["game_ids"] == 2).sum()) == partial
    assert (rows["flags"] == 0).all()
    for game in range(7):
        of_game = rows["game_ids"] == game
        players = int(rows["player_counts"][of_game][0])
        net_worths = rows["net_worths"][of_game]
        assert (net_worths == net_worths[0]).all()
        assert (net_worths[0, :players] >= 0).all() and net_worths[0, :players].max() > 0
        assert (net_worths[0, players:] == -1).all()


def test_a_run_set_by_a_replay_ratio_and_temperature_shares_resumes_where_it_was(tmp_path, monkeypatch, capsys):
    """One game a player count an epoch, through a graceful shutdown mid-epoch
    and its resume (1889/v1-design.md 2.4, 3.3, 3.4):

    - steps follow each epoch's rows, the replayed epoch's with the game it had;
    - every step trains at the epoch schedule's rate for its epoch and its
      share of the epoch's steps, on both optimizers, across the resumes;
    - the first epoch plays at the initial temperature, each later one in the
      windows its checkpoint's game lengths give, the replayed epoch in the
      windows the interrupted one had, and a game started from a position
      leaves its player count's length as it was;
    - a checkpoint saved without lengths resumes at the initial temperature.
    """
    start_pcts, end_pcts = [0.3, 0.3, 0.25], [0.6, 0.6, 0.75]
    fields = {
        **_smoke_config(tmp_path), "buffer_epochs": 2, "buffer_capacity": 65_536, "replay_ratio": 1.1,
        "training_steps_per_epoch": 5, "lr_decay_end_epoch": 4,
        "temp_anneal_start_pcts": start_pcts, "temp_anneal_end_pcts": end_pcts,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(fields))
    checkpoints = tmp_path / "checkpoints"
    device = torch.device("cpu")

    games: list[tuple[int, object]] = []  # each game's player count and windows
    rates: list[tuple[float, float | None]] = []  # each step's rates
    started: set[int] = set()  # indexes into games to mark as started from a position
    real_play_game, real_train_step = main_module.play_game, Trainer.train_step

    def play_game(evaluator, config, *args, **kwargs):
        record = real_play_game(evaluator, config, *args, **kwargs)
        if len(games) in started:
            record.flags |= FLAG_FROM_POSITION
        games.append((kwargs["num_players"], kwargs["epoch_config"].temp_anneal_windows))
        return record

    def train_step(trainer, *args, **kwargs):
        rates.append((trainer.lr, trainer.aux_lr))
        return real_train_step(trainer, *args, **kwargs)

    monkeypatch.setattr(main_module, "play_game", play_game)
    monkeypatch.setattr(Trainer, "train_step", train_step)

    def run(*args: str, stop: bool = False) -> str:
        event = threading.Event()
        if stop:
            event.set()
        with monkeypatch.context() as patch:
            patch.setattr(main_module, "_start_shutdown_listener", lambda: event)
            _train(patch, "--config", str(config_path), *args)
        return capsys.readouterr().out

    def windows(lengths: dict[int, float]) -> dict[int, tuple[int, int]]:
        return {
            n: (round(start * lengths[n]), round(end * lengths[n]))
            for n, start, end in zip((3, 4, 5), start_pcts, end_pcts)
        }

    run()  # epoch 0
    # q + Enter during epoch 1's first game: saved untrained, to be replayed.
    assert "Trained: no" in run("--resume", "latest", "--num-epochs", "3", stop=True)
    interrupted = load_checkpoint(checkpoints / "checkpoint_epoch_0001.pt", device)
    started.add(4 + [n for n, _ in games[:3]].index(4))  # epoch 1's 4-player game, replayed
    output = run("--resume", "latest", "--num-epochs", "3")  # epochs 1 and 2
    assert "Resumed from epoch 0" in output and "no game lengths" not in output

    scalars = read_tb_scalars(str(tmp_path / "runs"))
    # Each epoch's mean length by player count, over its games from the opening;
    # the started game is reported apart (self_play_started).
    length = {n: dict(scalars[f"self_play_{n}p/game_length_mean"]) for n in (3, 4, 5)}
    assert [sorted(length[n]) for n in (3, 4, 5)] == [[1, 2, 3], [1, 3], [1, 2, 3]]
    first = {n: length[n][1] for n in (3, 4, 5)}
    second = {3: length[3][2], 4: first[4], 5: length[5][2]}
    (started_epoch, started_length), = scalars["self_play_started_4p/game_length_mean"]
    assert started_epoch == 2 and started_length != first[4]  # the started game had a length of its own

    # Temperature windows, by game: 3 of epoch 0, 1 before the shutdown, 3 + 3.
    assert sorted(n for n, _ in games[:3]) == [3, 4, 5] and len(games) == 10
    assert [seen for _, seen in games] == [{}] * 3 + [windows(first)] * 4 + [windows(second)] * 3
    assert all(0 < start < end for start, end in windows(first).values())
    assert interrupted["epoch"] == 0 and interrupted.get("game_lengths") == first
    stored = [
        load_checkpoint(checkpoints / f"checkpoint_epoch_{epoch:04d}.pt", device).get("game_lengths")
        for epoch in (2, 3)
    ]
    assert stored == [second, {n: length[n][3] for n in (3, 4, 5)}]
    for n in (3, 4, 5):
        for edge, tag in enumerate(("start", "end")):
            assert scalars[f"schedule/temp_anneal_{tag}_{n}p"] == [
                (2, windows(first)[n][edge]), (3, windows(second)[n][edge]),
            ]

    # Steps: 1.1 draws a row, by the share of the 2 epochs held.
    played = [int(value) for _, value in scalars["self_play_aggregate/total_examples"]]
    played[1] += int(scalars["self_play_started/total_examples"][0][1])
    sizes = [int(value) for _, value in scalars["buffer/size"]]
    partial = sizes[1] - sum(played[:2])
    assert partial > 0
    steps = [round(1.1 * rows / 64 * share) for rows, share in (
        (played[0], 1 / 2), (partial + played[1], 1), (played[2], 1),
    )]
    assert [value for _, value in scalars["epoch/training_steps"]] == steps
    assert min(steps) > 5  # not training_steps_per_epoch, nor its share

    # Rates: a warmup of 1 epoch, then the cosine to epoch 4, in epochs.
    lr, lr_min = fields["learning_rate"], fields["lr_min"]
    assert isinstance(lr, float) and isinstance(lr_min, float)

    def rate(progress: float, step_share: float) -> float:
        if progress < 1:
            return lr * (progress + step_share)
        return lr_min + (lr - lr_min) * 0.5 * (1 + math.cos(math.pi * (progress - 1) / 3))

    expected = [rate(epoch + step / count, 1 / count) for epoch, count in enumerate(steps) for step in range(count)]
    assert [main for main, _ in rates] == pytest.approx(expected, rel=1e-9)
    assert all(aux == main for main, aux in rates)
    assert rates[steps[0]][0] == lr and rates[-1][0] < 0.6 * lr  # the resume's first step; the decay under way

    # A checkpoint from before the lengths: said, and played at temp_initial.
    path = checkpoints / "checkpoint_epoch_0003.pt"
    old = torch.load(path, weights_only=False)
    del old["game_lengths"]
    torch.save(old, path)
    output = run("--resume", "latest", "--num-epochs", "4", stop=True)
    assert "Checkpoint has no game lengths" in output
    assert games[10:] == [(games[10][0], {})]


def _buffer(config: TrainingConfig) -> ReplayBuffer:
    game = get_game(config)
    return ReplayBuffer(
        game, config.buffer_capacity, game.state_size(MAX_PLAYERS), MAX_PLAYERS,
        min_players=3, max_players=MAX_PLAYERS,
        priority_fraction=config.replay_priority_fraction, priority_exponent=config.replay_priority_exponent,
    )


def _random_rows(config: TrainingConfig) -> ReplayBuffer:
    """A buffer of random-playout rows at 3-5 players with priorities."""
    game = get_game(config)
    buffer = _buffer(config)
    scratch = np.empty(game.policy.max_action_size, dtype=np.uint16)
    lut = game.policy.build_action_lut().numpy()
    rng = np.random.default_rng(0)
    for num_players in (3, 4, 5):
        state = game.new_game(num_players, MAX_PLAYERS, num_players)
        while True:
            phase = game.decision_phase(state)
            count = game.enumerate_policy_actions(state, scratch)
            slots = lut[phase, scratch[:count]]
            mask = np.zeros((1, game.policy.unified_dim), np.uint8)
            target = np.zeros((1, game.policy.unified_dim), np.float32)
            mask[0, slots] = 1
            target[0, slots] = rng.dirichlet(np.ones(count)).astype(np.float32)
            value = rng.uniform(-1, 1, (1, num_players)).astype(np.float32)
            buffer.add_stacked(
                state._array[None].copy(), np.array([phase], np.int8), mask, target, value,
                num_players=num_players, priorities=rng.uniform(0.01, 1.0, 1).astype(np.float32),
            )
            if game.apply_action(state, _choose(state, rng, True)) == 1:
                break
    return buffer


def test_a_resumed_trainer_continues_exactly(tmp_path):
    config = TrainingConfig.from_json(json.dumps(_smoke_config(tmp_path)))
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = create_model(config)
    trainer = Trainer(model, config, device)
    buffer = _random_rows(config)
    rng = np.random.default_rng(1)
    for _ in range(3):
        trainer.train_step(buffer, config.batch_size, rng)

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, 0, model, trainer.state_dict(), config, {}, {"size": len(buffer), "capacity": config.buffer_capacity},
        rng_state=main_module._capture_rng_state(rng),
    )
    buffer.save(tmp_path / "replay_buffer")

    resumed = load_checkpoint(path, device)
    resumed_config = TrainingConfig.from_json(resumed["config_json"])
    resumed_model = create_model(resumed_config)
    resumed_model.load_state_dict(resumed["model_state_dict"])
    resumed_trainer = Trainer(resumed_model, resumed_config, device)
    resumed_trainer.load_state_dict(resumed["trainer_state"])
    resumed_buffer = _buffer(resumed_config)
    assert resumed_buffer.load(tmp_path / "replay_buffer") == len(buffer)
    resumed_rng = np.random.default_rng(99)
    assert "rng_state" in resumed
    main_module._restore_rng_state(resumed_rng, resumed["rng_state"])
    assert resumed_trainer.global_step == trainer.global_step and resumed_trainer.lr == trainer.lr

    for _ in range(3):
        losses = trainer.train_step(buffer, config.batch_size, rng)
        resumed_losses = resumed_trainer.train_step(resumed_buffer, config.batch_size, resumed_rng)
        assert resumed_losses == losses
    for name, weights in model.state_dict().items():
        assert torch.equal(weights, resumed_model.state_dict()[name]), name


def test_a_resume_takes_a_rescales_changes_and_keeps_the_rows(tmp_path, monkeypatch, capsys):
    """Section 7.1's epoch-20 rescale: every changed field reaches the resumed run,
    and a changed capacity keeps the newest rows. The file wins over the
    checkpoint's stored config, so a later resume keeps them only if it is edited."""
    fields = _smoke_config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(fields))
    checkpoints = tmp_path / "checkpoints"
    device = torch.device("cpu")
    _train(monkeypatch, "--config", str(config_path))
    first = load_checkpoint(checkpoints / "checkpoint_epoch_0001.pt", device)
    saved_rows = first["buffer_stats"]["size"]
    capacity = saved_rows // 2 + 100
    assert capacity < saved_rows

    new_windows = {
        "temp_anneal_starts": [111, 122, 133],
        "temp_anneal_ends": [222, 244, 266],
        "policy_target_temp_anneal_starts": [111, 122, 133],
        "policy_target_temp_anneal_ends": [222, 244, 266],
    }
    rescaled = {
        **fields, **new_windows,
        "buffer_capacity": capacity,
        "training_steps_per_epoch": 13,
        "num_epochs": 2,
        "min_buffer_size": 64,
    }
    config_path.write_text(json.dumps(rescaled))
    seen_windows = {}
    real_play_game = main_module.play_game

    def spy(evaluator, config, *args, **kwargs):
        seen_windows["action"] = config.temp_anneal_window(3)
        seen_windows["target"] = config.policy_target_temp_anneal_window(5)
        return real_play_game(evaluator, config, *args, **kwargs)

    monkeypatch.setattr(main_module, "play_game", spy)
    capsys.readouterr()
    _train(monkeypatch, "--config", str(config_path), "--resume", "latest")
    output = capsys.readouterr().out

    assert f"Config override: buffer_capacity = {capacity} (was {fields['buffer_capacity']})" in output
    assert f"Config override: training_steps_per_epoch = 13 (was {fields['training_steps_per_epoch']})" in output
    assert "Config override: temp_anneal_starts = [111, 122, 133]" in output
    assert f"keeping the newest {capacity:,} of {saved_rows:,} rows" in output
    assert f"Loaded {capacity:,} examples into replay buffer" in output
    assert seen_windows == {"action": (111, 222), "target": (133, 266)}

    second = load_checkpoint(checkpoints / "checkpoint_epoch_0002.pt", device)
    stored = TrainingConfig.from_json(second["config_json"])
    assert stored.buffer_capacity == capacity
    assert stored.training_steps_per_epoch == 13
    assert stored.temp_anneal_starts == [111, 122, 133]
    # The buffer is full, so the epoch ran the new steps in full.
    step, before = second["trainer_state"]["global_step"], first["trainer_state"]["global_step"]
    assert isinstance(step, int) and isinstance(before, int)
    assert step - before == 13
    assert second["buffer_stats"] == {"size": capacity, "capacity": capacity}
    # The LR schedule's span moved with the steps: the scheduler restored its step
    # and kept this run's lambda, so the rate is the new span's at that step.
    scheduler = second["trainer_state"]["scheduler"]
    assert isinstance(scheduler, dict)
    assert scheduler["last_epoch"] == step
    trainer = Trainer(create_model(stored), stored, device)
    span_rate = trainer.scheduler.lr_lambdas[0](step) * stored.learning_rate  # type: ignore[attr-defined]
    assert scheduler["_last_lr"][0] == pytest.approx(span_rate)
    old_span = dataclasses.replace(stored, training_steps_per_epoch=fields["training_steps_per_epoch"])
    old_trainer = Trainer(create_model(old_span), old_span, device)
    assert old_trainer.scheduler.lr_lambdas[0](step) * stored.learning_rate != pytest.approx(span_rate)  # type: ignore[attr-defined]
