"""Replay's game columns and its epoch window (1889/v1-design.md 3.1, 3.2).

- Each added game takes the next game id, with its flags and net worths; they
  stay with their rows through a ring wrap and a save and load, and a buffer
  saved before them loads them as unknown.
- With ``buffer_epochs`` N, sampling draws exactly as a buffer holding only the
  newest N epochs' rows would: wrapped or not, prioritized or not, reloaded or
  not. With 0 it draws what it drew before the window existed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.state import GameState, get_layout
from games import get_game
from nn.policy_layout import UNIFIED_LOGIT_DIM
from train.config import TrainingConfig
from train.replay_buffer import FLAG_FROM_POSITION, ReplayBuffer

RSS_GAME = get_game(TrainingConfig())
U_DIM = int(UNIFIED_LOGIT_DIM)
STATE_SIZE = get_layout(5).total_size
NEW_COLUMNS = ("game_ids", "flags", "net_worths")


def _buffer(capacity: int, **kwargs) -> ReplayBuffer:
    return ReplayBuffer(RSS_GAME, capacity, STATE_SIZE, 5, min_players=3, max_players=5, **kwargs)


class _Writer:
    """Adds games whose rows carry a running tag in value slot 0 and a priority
    that depends on it, and remembers each tag's game."""

    def __init__(self, buffer: ReplayBuffer) -> None:
        self.buffer = buffer
        self.tag = 0
        self.games = 0
        self.game_of: dict[int, int] = {}
        state = GameState(3, max_players=5)
        state.initialize_game(3, seed=0, max_players=5)
        self.state = state._array.copy()

    def add_game(self, rows: int, *, players: int = 3, flags: int = 0) -> list[int]:
        tags = list(range(self.tag, self.tag + rows))
        masks = np.zeros((rows, U_DIM), dtype=np.uint8)
        masks[:, :2] = 1
        targets = np.zeros((rows, U_DIM), dtype=np.float32)
        targets[:, 0] = 1.0
        values = np.zeros((rows, players), dtype=np.float32)
        values[:, 0] = tags
        self.buffer.add_stacked(
            np.repeat(self.state[None], rows, axis=0), np.zeros(rows, dtype=np.int8), masks, targets, values,
            num_players=players,
            priorities=np.array([(tag * 7 % 11) / 3 for tag in tags], dtype=np.float32),
            flags=flags,
            net_worths=_net_worths(self.games, players),
        )
        self.game_of.update(dict.fromkeys(tags, self.games))
        self.tag += rows
        self.games += 1
        return tags

    def add_epoch(self, epoch: int, game_rows: tuple[int, ...]) -> list[int]:
        self.buffer.begin_epoch(epoch)
        return [tag for rows in game_rows for tag in self.add_game(rows)]


def _net_worths(game: int, players: int) -> list[int]:
    return [1000 * game + seat for seat in range(players)]


def _tags(buffer: ReplayBuffer) -> list[int]:
    return buffer.rows_oldest_first()["value_targets"][:, 0].astype(int).tolist()


def _draws(buffer: ReplayBuffer, batches: int = 40, batch_size: int = 8, seed: int = 5):
    """Sampled tags and importance weights over ``batches`` batches."""
    rng = np.random.default_rng(seed)
    tags, weights = [], []
    for _ in range(batches):
        batch = buffer.sample(batch_size, rng)
        tags.append(batch["value_targets"][:, 0].numpy().astype(int))
        weights.append(batch["is_weights"].numpy())
    return np.concatenate(tags), np.concatenate(weights)


# ---------------------------------------------------------------------------
# Game ids, flags, net worths
# ---------------------------------------------------------------------------

def _assert_game_columns(buffer: ReplayBuffer, writer: _Writer, flagged: set[int]) -> None:
    """Every held row carries its own game's id, flags, and net worths."""
    rows = buffer.rows_oldest_first()
    assert len(rows["states"]) == len(buffer) > 0
    for tag, game_id, flags, net_worths, players in zip(
        _tags(buffer), rows["game_ids"], rows["flags"], rows["net_worths"], rows["player_counts"], strict=True,
    ):
        game = writer.game_of[tag]
        assert game_id == game
        assert flags == (FLAG_FROM_POSITION if game in flagged else 0)
        assert net_worths.tolist() == _net_worths(game, int(players)) + [-1] * (5 - int(players))
    assert (np.diff(rows["game_ids"]) >= 0).all()


def test_game_columns_stay_with_their_rows_through_a_wrap_and_a_save(tmp_path: Path) -> None:
    buffer = _buffer(20)
    writer = _Writer(buffer)
    # 51 rows: games 3 and 6 wrap inside themselves, over rows of the other flag.
    lengths = (4, 7, 3, 9, 5, 8, 6, 9)
    flagged = {1, 3, 5, 7}
    for game, rows in enumerate(lengths):
        writer.add_game(rows, players=3 + game % 3, flags=FLAG_FROM_POSITION if game in flagged else 0)
        _assert_game_columns(buffer, writer, flagged)
    assert _tags(buffer) == list(range(31, 51))

    buffer.save(tmp_path)
    for capacity in (20, 33, 11):  # as saved, grown, and shrunk to its newest rows
        loaded = _buffer(capacity)
        assert loaded.load(tmp_path) == min(capacity, 20)
        _assert_game_columns(loaded, writer, flagged)
        # The run's next game carries on from the saved count.
        writer.buffer = loaded
        writer.add_game(3)
        assert loaded.rows_oldest_first()["game_ids"][-3:].tolist() == [len(lengths)] * 3
        writer.tag, writer.games = 51, len(lengths)


def test_an_add_larger_than_the_buffer_keeps_its_newest_rows_columns() -> None:
    buffer = _buffer(6)
    writer = _Writer(buffer)
    writer.add_game(4)
    writer.add_game(9, flags=FLAG_FROM_POSITION)
    assert _tags(buffer) == list(range(7, 13))
    _assert_game_columns(buffer, writer, {1})


def _strip_to_old_format(directory: Path) -> None:
    """Leave ``directory`` as a buffer saved before the game columns and the
    epoch record: the same files and metadata, less the new ones."""
    for name in NEW_COLUMNS:
        (directory / f"{name}.npy").unlink()
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["next_game_id"], metadata["epochs"]
    metadata_path.write_text(json.dumps(metadata))


@pytest.mark.parametrize("capacity", [20, 12])
def test_a_buffer_saved_before_the_game_columns_loads_them_as_unknown(
    tmp_path: Path, capacity: int, capsys,
) -> None:
    saved = _buffer(20)
    writer = _Writer(saved)
    for epoch in range(4):
        writer.add_epoch(epoch, (5, 4))  # 36 rows: wrapped
    saved.save(tmp_path)
    before = saved.rows_oldest_first()
    _strip_to_old_format(tmp_path)

    loaded = _buffer(capacity)
    assert loaded.load(tmp_path) == capacity
    assert "no epoch record" not in capsys.readouterr().out  # buffer_epochs is off
    rows = loaded.rows_oldest_first()
    for name, array in before.items():
        if name not in NEW_COLUMNS:
            np.testing.assert_array_equal(rows[name], array[-capacity:], err_msg=name)
    assert (rows["game_ids"] == -1).all()
    assert (rows["flags"] == 0).all()
    assert (rows["net_worths"] == -1).all()
    assert loaded.epochs == []

    # Its rows count as one epoch, and the run's games number from 0 after them.
    windowed = _buffer(capacity, buffer_epochs=2)
    assert windowed.load(tmp_path) == capacity
    assert f"no epoch record: its {capacity:,} rows count as one epoch" in capsys.readouterr().out
    assert (windowed.window_size, windowed.window_epochs) == (capacity, 1)
    new = _Writer(windowed)
    new.tag = 100
    first = new.add_epoch(7, (3,))
    assert windowed.rows_oldest_first()["game_ids"].tolist() == [-1] * (capacity - 3) + [0] * 3
    assert (windowed.window_size, windowed.window_epochs) == (capacity, 2)
    second = new.add_epoch(8, (2, 2))
    assert (windowed.window_size, windowed.window_epochs) == (7, 2)
    assert set(_draws(windowed, batch_size=4)[0].tolist()) == set(first + second)


# ---------------------------------------------------------------------------
# The epoch window
# ---------------------------------------------------------------------------

EPOCH_GAMES = ((4, 6), (5, 3, 2), (7, 4), (3, 3, 5), (6, 2), (4, 5, 3))  # 10, 10, 11, 11, 8, 12 rows


def _reference(tags: list[int], fraction: float) -> ReplayBuffer:
    """A buffer holding exactly the rows ``tags``, unwrapped, sampling all of them."""
    reference = _buffer(len(tags), priority_fraction=fraction)
    writer = _Writer(reference)
    writer.tag = tags[0]
    writer.add_game(len(tags))
    assert _tags(reference) == tags
    return reference


def _assert_samples_as(buffer: ReplayBuffer, tags: list[int], fraction: float) -> None:
    """``buffer`` draws the rows and weights a buffer of only ``tags`` draws,
    and reaches every one of them."""
    reference = _reference(tags, fraction)
    assert buffer.window_size == len(tags)
    drawn, weights = _draws(buffer)
    expected, expected_weights = _draws(reference)
    np.testing.assert_array_equal(drawn, expected)
    np.testing.assert_array_equal(weights, expected_weights)
    assert set(drawn.tolist()) == set(tags)
    assert buffer.priority_stats() == pytest.approx(reference.priority_stats())
    with pytest.raises(ValueError, match="exceeds buffer size"):
        buffer.sample(len(tags) + 1, np.random.default_rng(0))


@pytest.mark.parametrize("fraction", [0.0, 0.5])
@pytest.mark.parametrize("capacity", [100, 40, 27])  # never full; wrapped; wrapped with the window across the seam
@pytest.mark.parametrize("buffer_epochs", [1, 2])
def test_a_window_of_n_epochs_samples_only_their_rows(
    tmp_path: Path, fraction: float, capacity: int, buffer_epochs: int,
) -> None:
    buffer = _buffer(capacity, priority_fraction=fraction, buffer_epochs=buffer_epochs)
    writer = _Writer(buffer)
    by_epoch: list[list[int]] = []
    seams = 0
    for epoch, game_rows in enumerate(EPOCH_GAMES):
        by_epoch.append(writer.add_epoch(epoch, game_rows))
        window = [tag for tags in by_epoch[-buffer_epochs:] for tag in tags]
        assert buffer.window_epochs == min(buffer_epochs, epoch + 1)
        _assert_samples_as(buffer, window, fraction)
        # The window's rows straddle the ring's end.
        seams += len(buffer) == capacity and 0 < buffer._index < len(window)

        buffer.save(tmp_path / str(epoch))
        loaded = _buffer(capacity, priority_fraction=fraction, buffer_epochs=buffer_epochs)
        assert loaded.load(tmp_path / str(epoch)) == len(buffer)
        _assert_samples_as(loaded, window, fraction)
        # At another capacity that still holds the window, too.
        resized = _buffer(capacity + 9, priority_fraction=fraction, buffer_epochs=buffer_epochs)
        assert resized.load(tmp_path / str(epoch)) == len(buffer)
        _assert_samples_as(resized, window, fraction)
    assert len(buffer) == min(capacity, writer.tag)
    assert seams > 0 or capacity != 27


def test_buffer_epochs_off_samples_the_whole_buffer() -> None:
    buffer = _buffer(40, priority_fraction=0.5)
    writer = _Writer(buffer)
    for epoch, game_rows in enumerate(EPOCH_GAMES):
        writer.add_epoch(epoch, game_rows)
    assert (buffer.window_size, len(buffer)) == (40, 40)
    assert set(_draws(buffer)[0].tolist()) == set(range(22, 62))


# The first draw's ring positions before the window existed (commit d6ef51b),
# from this wrapped ring at seed 7: uniform, then half prioritized.
RING_ADDS = (7, 13, 9, 11, 20, 6, 14)
UNIFORM_BEFORE = [33, 38, 24, 42, 2, 9, 0, 31, 41, 12, 34, 13, 40, 22, 45, 25]
PRIORITIZED_BEFORE = [37, 48, 11, 41, 46, 27, 30, 40, 40, 22, 15, 14, 13, 21, 24, 28]


@pytest.mark.parametrize(
    ("fraction", "before"), [(0.0, UNIFORM_BEFORE), (0.5, PRIORITIZED_BEFORE)],
)
@pytest.mark.parametrize("epochs_recorded", [False, True])
def test_buffer_epochs_off_draws_the_positions_it_drew_before(
    fraction: float, before: list[int], epochs_recorded: bool,
) -> None:
    buffer = ReplayBuffer(RSS_GAME, 50, get_layout(3).total_size, 3, priority_fraction=fraction)
    state = GameState(3)
    state.initialize_game(3, seed=0)
    priorities = np.random.default_rng(3)
    for epoch, n in enumerate(RING_ADDS):
        if epochs_recorded:
            buffer.begin_epoch(epoch)
        masks = np.zeros((n, U_DIM), dtype=np.uint8)
        masks[:, :2] = 1
        targets = np.zeros((n, U_DIM), dtype=np.float32)
        targets[:, 0] = 1
        buffer.add_stacked(
            np.repeat(state._array[None], n, axis=0), np.zeros(n, dtype=np.int8), masks, targets,
            np.zeros((n, 3), dtype=np.float32), priorities=priorities.random(n).astype(np.float32),
        )
    indices, _ = buffer._sample_indices(16, np.random.default_rng(7))
    assert indices.tolist() == before


def test_a_resume_that_replays_a_partial_epoch_keeps_it_one_epoch(tmp_path: Path) -> None:
    buffer = _buffer(100, buffer_epochs=1)
    writer = _Writer(buffer)
    writer.add_epoch(0, (4, 4))
    partial = writer.add_epoch(1, (3,))  # a graceful shutdown mid-epoch saves here
    buffer.save(tmp_path)

    resumed = _buffer(100, buffer_epochs=1)
    assert resumed.load(tmp_path) == 11
    assert resumed.epochs == [(0, 0), (1, 2)]
    writer.buffer = resumed
    replayed = writer.add_epoch(1, (5, 2))
    assert resumed.epochs == [(0, 0), (1, 2)]
    assert (resumed.window_size, resumed.window_epochs) == (10, 1)
    assert set(_draws(resumed)[0].tolist()) == set(partial + replayed)
    following = writer.add_epoch(2, (9,))
    assert resumed.epochs == [(0, 0), (1, 2), (2, 5)]
    assert set(_draws(resumed)[0].tolist()) == set(following)


def test_an_epoch_that_got_no_game_leaves_no_entry() -> None:
    buffer = _buffer(100, buffer_epochs=2)
    writer = _Writer(buffer)
    first = writer.add_epoch(0, (8,))
    buffer.begin_epoch(1)  # stopped before its first game finished
    assert (buffer.window_size, buffer.window_epochs) == (8, 1)
    # Resuming an older checkpoint, say, begins another epoch instead.
    second = writer.add_epoch(5, (8,))
    assert buffer.epochs == [(0, 0), (5, 1)]
    third = writer.add_epoch(6, (8,))
    assert buffer.window_epochs == 2
    assert set(_draws(buffer)[0].tolist()) == set(second + third)
    assert not set(first) & set(second + third)


def test_a_window_larger_than_the_capacity_warns_once_an_epoch(capsys) -> None:
    buffer = _buffer(20, buffer_epochs=2)
    writer = _Writer(buffer)
    writer.add_epoch(0, (5, 5))
    writer.add_epoch(1, (5, 5))  # full, and nothing lost yet
    writer.add_epoch(2, (5, 5))  # overwrites epoch 0, which has left the window
    assert "Warning" not in capsys.readouterr().out
    assert (buffer.window_size, buffer.window_epochs) == (20, 2)

    writer.add_epoch(3, (8, 8))  # 26 rows in epochs 2-3: the second game overwrites epoch 2's
    assert capsys.readouterr().out.count("newest 2 epochs' rows exceed buffer_capacity (20)") == 1
    assert (buffer.window_size, buffer.window_epochs) == (20, 2)
    writer.add_epoch(4, (9, 9, 9))  # 27 rows: epoch 3 has gone, and the epoch's first game too
    assert capsys.readouterr().out.count("Warning") == 1
    assert (buffer.window_size, buffer.window_epochs) == (20, 1)
    assert _tags(buffer) == list(range(writer.tag - 20, writer.tag))

    # With the window off, the ring overwrites silently, as ever.
    plain = _Writer(_buffer(20))
    for epoch in range(4):
        plain.add_epoch(epoch, (8, 8))
    assert "Warning" not in capsys.readouterr().out
