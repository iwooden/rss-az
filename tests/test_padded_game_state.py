import numpy as np
import pytest

from core.attention_relations import NUM_ATTENTION_RELATIONS
from core.state import (
    GameState,
    get_layout,
    get_storage_player_capacity,
    get_turn_fields,
)
from core.token_data import get_num_tokens
from entities.player import PLAYERS
from entities.turn import TURN
from nn.transformer import UNIFIED_LOGIT_DIM
from train.replay_buffer import ReplayBuffer


def _padded_player_tail(
    state: GameState, num_players: int, max_players: int,
) -> np.ndarray:
    layout = get_layout(max_players)
    start = layout.players_offset + num_players * layout.player_size
    stop = layout.players_offset + max_players * layout.player_size
    return state._array[start:stop]


def test_construct_3p_game_with_5p_storage_keeps_padded_players_zeroed() -> None:
    state = GameState(3, max_players=5)
    state.initialize_game(3, seed=42)

    assert state.max_players == 5
    assert TURN.get_num_players(state) == 3
    assert state._array.shape == (get_layout(5).total_size,)
    np.testing.assert_array_equal(_padded_player_tail(state, 3, 5), 0)

    assert [PLAYERS[i].get_cash(state) for i in range(3)] == [30, 30, 30]
    assert [PLAYERS[i].get_turn_order(state) for i in range(3)] == [0, 1, 2]


def test_from_array_and_from_buffer_accept_padded_storage_when_max_players_is_given() -> None:
    source = GameState(3, max_players=5)
    source.initialize_game(3, seed=42)
    array = source._array.copy()

    copied = GameState.from_array(array, 3, max_players=5)
    assert copied.max_players == 5
    assert TURN.get_num_players(copied) == 3
    np.testing.assert_array_equal(copied._array, array)

    wrapped = GameState.from_buffer(array, 3, max_players=5)
    assert wrapped.max_players == 5
    assert TURN.get_num_players(wrapped) == 3
    wrapped._array[0] = 123
    assert array[0] == 123


def test_exact_size_from_array_default_still_rejects_padded_arrays() -> None:
    state = GameState(3, max_players=5)
    state.initialize_game(3, seed=42)

    with pytest.raises(AssertionError, match="length"):
        GameState.from_array(state._array, 3)

    exact = GameState(3)
    exact.initialize_game(3, seed=42)
    copied = GameState.from_array(exact._array, 3)
    assert copied.max_players == 3
    assert copied._array.shape == (get_layout(3).total_size,)


def test_rebind_accepts_same_capacity_padded_rows_with_different_actual_counts() -> None:
    row3 = GameState(3, max_players=5)
    row3.initialize_game(3, seed=42)
    row4 = GameState(4, max_players=5)
    row4.initialize_game(4, seed=43)

    scratch = GameState.from_buffer(row3._array, 3, max_players=5)
    scratch.rebind(row4._array, 4, max_players=5)

    assert scratch.max_players == 5
    assert TURN.get_num_players(scratch) == 4
    np.testing.assert_array_equal(_padded_player_tail(scratch, 4, 5), 0)


def test_from_array_rejects_canonical_actual_player_mismatch() -> None:
    state = GameState(3, max_players=5)
    state.initialize_game(3, seed=42)
    array = state._array.copy()
    turn = get_turn_fields()
    layout = get_layout(5)
    array[layout.turn_offset + turn.num_players] = 4

    with pytest.raises(AssertionError, match="canonical num_players"):
        GameState.from_array(array, 3, max_players=5)


@pytest.mark.parametrize("delta", [-1, 1])
def test_from_array_rejects_wrong_length_for_max_players(delta: int) -> None:
    state = GameState(3, max_players=5)
    state.initialize_game(3, seed=42)
    size = get_layout(5).total_size + delta
    bad = np.zeros(size, dtype=np.int16)

    with pytest.raises(AssertionError, match="length"):
        GameState.from_array(bad, 3, max_players=5)


def test_rebind_rejects_wrong_length_and_canonical_mismatch() -> None:
    state = GameState(3, max_players=5)
    state.initialize_game(3, seed=42)
    scratch = GameState.from_buffer(state._array, 3, max_players=5)

    with pytest.raises(AssertionError, match="length"):
        scratch.rebind(state._array[:-1], 3, max_players=5)

    bad = state._array.copy()
    turn = get_turn_fields()
    layout = get_layout(5)
    bad[layout.turn_offset + turn.num_players] = 4
    with pytest.raises(AssertionError, match="canonical num_players"):
        scratch.rebind(bad, 3, max_players=5)


def test_storage_player_capacity_is_derived_from_layout_width() -> None:
    assert get_storage_player_capacity(get_layout(3).total_size) == 3
    assert get_storage_player_capacity(get_layout(5).total_size) == 5

    with pytest.raises(AssertionError, match="aligned"):
        get_storage_player_capacity(get_layout(5).total_size - 1)


def test_replay_buffer_samples_padded_state_rows_and_relation_scratch() -> None:
    num_players = 3
    max_players = 5
    state = GameState(num_players, max_players=max_players)
    state.initialize_game(num_players, seed=42)
    buffer = ReplayBuffer(
        capacity=2,
        state_size_int16=get_layout(max_players).total_size,
        num_players=num_players,
    )
    buffer.add_stacked(
        states=state._array.reshape(1, -1).copy(),
        phase_ids=np.array([0], dtype=np.int8),
        legal_masks=np.zeros((1, int(UNIFIED_LOGIT_DIM)), dtype=np.uint8),
        policy_targets=np.zeros((1, int(UNIFIED_LOGIT_DIM)), dtype=np.float32),
        value_targets=np.zeros((1, num_players), dtype=np.float32),
    )

    sample = buffer.sample(1, np.random.default_rng(1))

    assert tuple(sample["states"].shape) == (1, get_layout(max_players).total_size)
    assert tuple(sample["relations"].shape) == (
        1,
        int(NUM_ATTENTION_RELATIONS),
        get_num_tokens(num_players),
        get_num_tokens(num_players),
    )
    turn = get_turn_fields()
    canonical_slot = get_layout(max_players).turn_offset + turn.num_players
    assert int(sample["states"][0, canonical_slot].item()) == num_players
