from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.attention_relations import NUM_ATTENTION_RELATIONS
from core.state import GameState, get_layout, get_turn_fields
from core.token_data import get_num_tokens
from games import get_game
from nn.policy_layout import UNIFIED_LOGIT_DIM
from train.config import TrainingConfig
from train.replay_buffer import ReplayBuffer

RSS_GAME = get_game(TrainingConfig())


def _targets(value_width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    legal_masks = np.zeros((1, int(UNIFIED_LOGIT_DIM)), dtype=np.uint8)
    legal_masks[0, 0] = 1
    policy_targets = np.zeros((1, int(UNIFIED_LOGIT_DIM)), dtype=np.float32)
    policy_targets[0, 0] = 1.0
    value_targets = np.arange(1, value_width + 1, dtype=np.float32)[None, :]
    return legal_masks, policy_targets, value_targets


def _add_state(buffer: ReplayBuffer, num_players: int, seed: int) -> None:
    state = GameState(num_players, max_players=5)
    state.initialize_game(num_players, seed=seed, max_players=5)
    legal_masks, policy_targets, value_targets = _targets(num_players)
    buffer.add_stacked(
        states=state._array[None, :].copy(),
        phase_ids=np.array([0], dtype=np.int8),
        legal_masks=legal_masks,
        policy_targets=policy_targets,
        value_targets=value_targets,
        num_players=num_players,
    )


def _mixed_buffer(capacity: int = 4) -> ReplayBuffer:
    return ReplayBuffer(
        RSS_GAME,
        capacity=capacity,
        state_size_int16=get_layout(5).total_size,
        num_players=5,
        min_players=3,
        max_players=5,
    )


def test_replay_buffer_samples_mixed_player_counts_at_max_width() -> None:
    buffer = _mixed_buffer()
    _add_state(buffer, num_players=3, seed=3)
    _add_state(buffer, num_players=5, seed=5)

    batch = buffer.sample(2, np.random.default_rng(0))
    counts = batch["player_counts"].numpy()

    assert set(counts.tolist()) == {3, 5}
    assert tuple(batch["states"].shape) == (2, get_layout(5).total_size)
    assert tuple(batch["value_targets"].shape) == (2, 5)
    assert tuple(batch["relations"].shape) == (
        2,
        int(NUM_ATTENTION_RELATIONS),
        get_num_tokens(5),
        get_num_tokens(5),
    )

    values_by_count = {
        int(count): batch["value_targets"][row].numpy()
        for row, count in enumerate(counts)
    }
    np.testing.assert_allclose(
        values_by_count[3],
        np.array([1.0, 2.0, 3.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        values_by_count[5],
        np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
    )

    turn = get_turn_fields()
    canonical_slot = get_layout(5).turn_offset + turn.num_players
    state_counts = batch["states"][:, canonical_slot].numpy()
    assert set(state_counts.tolist()) == {3, 5}


def test_replay_buffer_save_load_preserves_mixed_player_counts(tmp_path: Path) -> None:
    buffer = _mixed_buffer()
    _add_state(buffer, num_players=3, seed=13)
    _add_state(buffer, num_players=5, seed=15)
    buffer.save(tmp_path)

    loaded = _mixed_buffer()
    assert loaded.load(tmp_path) == 2

    batch = loaded.sample(2, np.random.default_rng(1))
    assert set(batch["player_counts"].numpy().tolist()) == {3, 5}
    assert tuple(batch["value_targets"].shape) == (2, 5)


def test_replay_buffer_load_skips_old_single_count_metadata(tmp_path: Path) -> None:
    state_size = get_layout(3).total_size
    np.save(tmp_path / "states.npy", np.zeros((1, state_size), dtype=np.int16))
    np.save(tmp_path / "phase_ids.npy", np.zeros(1, dtype=np.int8))
    np.save(
        tmp_path / "legal_masks.npy",
        np.zeros((1, int(UNIFIED_LOGIT_DIM)), dtype=np.uint8),
    )
    np.save(
        tmp_path / "policy_targets.npy",
        np.zeros((1, int(UNIFIED_LOGIT_DIM)), dtype=np.float32),
    )
    np.save(tmp_path / "value_targets.npy", np.zeros((1, 3), dtype=np.float32))
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "size": 1,
                "index": 1,
                "capacity": 2,
                "state_size": state_size,
                "num_players": 3,
                "unified_dim": int(UNIFIED_LOGIT_DIM),
            }
        )
    )

    buffer = ReplayBuffer(RSS_GAME, 2, state_size, 3)

    assert buffer.load(tmp_path) == 0
    assert len(buffer) == 0


def test_replay_buffer_load_skips_another_games_buffer(tmp_path: Path, capsys) -> None:
    buffer = _mixed_buffer()
    _add_state(buffer, num_players=3, seed=21)
    buffer.save(tmp_path)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["game"] == "rss"

    # Buffers saved before the field are RSS's.
    del metadata["game"]
    metadata_path.write_text(json.dumps(metadata))
    assert _mixed_buffer().load(tmp_path) == 1

    metadata["game"] = "1889"
    metadata_path.write_text(json.dumps(metadata))
    other = _mixed_buffer()
    assert other.load(tmp_path) == 0
    assert len(other) == 0
    assert "replay buffer is from game '1889', not 'rss', skipping load" in capsys.readouterr().out


def _tagged_rows(first: int, count: int) -> dict[str, np.ndarray]:
    """``count`` rows, each carrying its tag ``first + i`` in every array."""
    tags = np.arange(first, first + count)
    state_size = get_layout(5).total_size
    legal_masks = np.zeros((count, int(UNIFIED_LOGIT_DIM)), dtype=np.uint8)
    legal_masks[:, 0] = 1
    legal_masks[np.arange(count), tags % 7 + 1] = 1
    policy_targets = np.zeros((count, int(UNIFIED_LOGIT_DIM)), dtype=np.float32)
    policy_targets[:, 0] = tags / 1000.0
    value_targets = np.zeros((count, 5), dtype=np.float32)
    value_targets[:, 0] = tags
    return {
        "states": np.tile(tags[:, None].astype(np.int16), (1, state_size)),
        "phase_ids": (tags % 11).astype(np.int8),
        "legal_masks": legal_masks,
        "policy_targets": policy_targets,
        "value_targets": value_targets,
        "player_counts": (3 + tags % 3).astype(np.uint8),
        "priorities": (tags + 0.5).astype(np.float32),
        "game_ids": (tags // 3).astype(np.int32),
        "flags": (tags % 2).astype(np.uint8),
        "net_worths": (tags[:, None] * np.arange(1, 6)).astype(np.int32),
    }


def _tagged_buffer(capacity: int, first: int = 0, count: int = 0) -> ReplayBuffer:
    buffer = ReplayBuffer(
        RSS_GAME, capacity, get_layout(5).total_size, 5, min_players=3, max_players=5,
    )
    if count:
        _add_tagged(buffer, first, count)
    return buffer


def _add_tagged(buffer: ReplayBuffer, first: int, count: int) -> None:
    rows = _tagged_rows(first, count)
    buffer.add_stacked(
        rows["states"], rows["phase_ids"], rows["legal_masks"], rows["policy_targets"], rows["value_targets"],
        player_counts=rows["player_counts"], priorities=rows["priorities"],
        game_ids=rows["game_ids"], flags=rows["flags"], net_worths=rows["net_worths"],
    )


def _assert_rows(buffer: ReplayBuffer, tags: list[int]) -> None:
    """The buffer holds exactly these tagged rows, oldest first, in every array."""
    expected = _tagged_rows(tags[0], len(tags))
    assert tags == list(range(tags[0], tags[0] + len(tags)))
    assert len(buffer) == len(tags)
    actual = buffer.rows_oldest_first()
    assert actual.keys() == expected.keys()
    for name, array in expected.items():
        np.testing.assert_array_equal(actual[name], array, err_msg=name)


@pytest.mark.parametrize(
    ("saved_capacity", "written", "new_capacity", "kept"),
    [
        (8, 11, 20, list(range(3, 11))),  # wrapped, grown: the whole ring
        (8, 11, 5, list(range(6, 11))),  # wrapped, shrunk: its newest
        (8, 19, 20, list(range(11, 19))),  # wrapped, its index back past the start
        (8, 16, 5, list(range(11, 16))),  # exactly full: its index is 0
        (8, 5, 12, list(range(5))),  # partial, grown
        (8, 5, 3, list(range(2, 5))),  # partial, shrunk
        (8, 5, 5, list(range(5))),  # partial, shrunk to its rows
        (8, 11, 8, list(range(3, 11))),  # unchanged
    ],
)
def test_replay_buffer_load_keeps_newest_rows_across_capacities(
    tmp_path: Path, saved_capacity: int, written: int, new_capacity: int, kept: list[int],
) -> None:
    saved = _tagged_buffer(saved_capacity)
    # Written in two adds, so a wrapped ring wraps inside an add.
    _add_tagged(saved, 0, written // 2)
    _add_tagged(saved, written // 2, written - written // 2)
    saved.save(tmp_path)

    loaded = _tagged_buffer(new_capacity)
    assert loaded.load(tmp_path) == len(kept)
    _assert_rows(loaded, kept)

    # New rows carry on from the newest, overwriting the oldest once full.
    _add_tagged(loaded, written, 3)
    tags = [*kept, written, written + 1, written + 2][-new_capacity:]
    _assert_rows(loaded, tags)

    # What it saves now loads back the same, at its capacity and at another.
    loaded.save(tmp_path / "again")
    again = _tagged_buffer(new_capacity)
    assert again.load(tmp_path / "again") == len(tags)
    _assert_rows(again, tags)
    smaller = _tagged_buffer(2)
    assert smaller.load(tmp_path / "again") == 2
    _assert_rows(smaller, tags[-2:])


def test_replay_buffer_load_at_new_capacity_samples_by_its_kept_priorities(tmp_path: Path) -> None:
    _tagged_buffer(8, 0, 11).save(tmp_path)
    loaded = ReplayBuffer(
        RSS_GAME, 5, get_layout(5).total_size, 5, min_players=3, max_players=5,
        priority_fraction=0.5,
    )
    assert loaded.load(tmp_path) == 5
    indices, weights = loaded._sample_indices(4, np.random.default_rng(3))
    assert indices.max() < 5
    assert weights is not None
    # The sampler's weights are the kept rows', not the rows of a stale cache.
    assert loaded.priority_stats()["priority_kl_mean"] == pytest.approx(np.mean(np.arange(6, 11) + 0.5))


def test_replay_buffer_load_at_new_capacity_without_saved_priorities(tmp_path: Path) -> None:
    _tagged_buffer(8, 0, 11).save(tmp_path)
    (tmp_path / "priorities.npy").unlink()
    loaded = _tagged_buffer(5)
    assert loaded.load(tmp_path) == 5
    rows = loaded.rows_oldest_first()
    assert np.isnan(rows["priorities"]).all()
    np.testing.assert_array_equal(rows["states"][:, 0], np.arange(6, 11))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_size", 1), ("num_players", 3), ("min_players", 4), ("max_players", 4),
        ("unified_dim", 1), ("size", 9), ("index", 8), ("capacity", 0),
    ],
)
def test_replay_buffer_load_skips_any_other_mismatch_at_a_new_capacity(
    tmp_path: Path, field: str, value: int, capsys,
) -> None:
    _tagged_buffer(8, 0, 11).save(tmp_path)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))
    loaded = _tagged_buffer(5)
    assert loaded.load(tmp_path) == 0
    assert len(loaded) == 0
    assert "mismatch" in capsys.readouterr().out
