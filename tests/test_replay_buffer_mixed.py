from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.attention_relations import NUM_ATTENTION_RELATIONS
from core.state import GameState, get_layout, get_turn_fields
from core.token_data import get_num_tokens
from nn.transformer import UNIFIED_LOGIT_DIM
from train.replay_buffer import ReplayBuffer


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

    buffer = ReplayBuffer(2, state_size, 3)

    assert buffer.load(tmp_path) == 0
    assert len(buffer) == 0
