"""g1889.state: the row layout (1889/engine-design.md §3), padded rows, and
initialize_game's new-game state at 2-6 players."""

import numpy as np
import pytest

import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.bank import BANK
from g1889.entities.corp import CORPS
from g1889.entities.hexmap import MAP
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.state import (
    GameState, get_corp_fields, get_global_fields, get_hex_fields, get_layout, get_player_fields,
    get_private_fields, get_storage_player_capacity, get_train_fields,
)

PLAYER_COUNTS = range(D.MIN_PLAYERS, D.MAX_PLAYERS + 1)
NUM_PLAYERS_SLOT = get_layout(D.MAX_PLAYERS).global_offset + get_global_fields().num_players


def new_game(num_players, seed=1, max_players=0, **kwargs):
    state = GameState(num_players, max_players=max_players)
    state.initialize_game(num_players, seed=seed, **kwargs)
    return state


def total_cash(state):
    n = TURN.num_players(state)
    return (BANK.cash(state) + sum(PLAYERS[i].cash(state) for i in range(n))
            + sum(corp.cash(state) for corp in CORPS))


# Layout


def test_block_sizes_and_total_width():
    layout = get_layout(5)
    sizes = {
        'global': layout.corps_offset - layout.global_offset,
        'corps': layout.privates_offset - layout.corps_offset,
        'privates': layout.hexes_offset - layout.privates_offset,
        'hexes': layout.supply_offset - layout.hexes_offset,
        'supply': layout.trains_offset - layout.supply_offset,
        'trains': layout.graph_offset - layout.trains_offset,
        'graph': layout.players_offset - layout.graph_offset,
    }
    assert layout.global_offset == 0
    assert sizes == {'global': 40, 'corps': 7 * 19, 'privates': 7 * 2, 'hexes': 52 * 3, 'supply': 40,
                     'trains': 12, 'graph': 7 * 52}
    assert (get_global_fields().size, get_corp_fields().size, get_private_fields().size,
            get_hex_fields().size, get_train_fields().size, get_player_fields().size) == (40, 19, 2, 3, 12, 18)
    assert layout.players_offset == sum(sizes.values()) == 759


@pytest.mark.parametrize('capacity', PLAYER_COUNTS)
def test_players_last_so_only_the_width_varies(capacity):
    layout = get_layout(capacity)
    assert layout.total_size == 759 + 18 * capacity
    assert layout._replace(total_size=0, max_players=0) == get_layout(2)._replace(total_size=0, max_players=0)
    assert get_storage_player_capacity(layout.total_size) == capacity
    assert GameState(D.MIN_PLAYERS, max_players=capacity)._array.shape == (layout.total_size,)


def test_field_offsets_tile_each_block():
    for fields in (get_global_fields(), get_corp_fields(), get_private_fields(), get_hex_fields(),
                   get_train_fields(), get_player_fields()):
        offsets = fields._asdict()
        size = offsets.pop('size')
        starts = sorted(offsets.values())
        assert starts[0] == 0 and len(set(starts)) == len(starts) and starts[-1] < size, fields


@pytest.mark.parametrize('width', [758, 759 + 18 * 1, 759 + 18 * 7, 759 + 18 * 3 + 1])
def test_capacity_rejects_bad_widths(width):
    with pytest.raises(AssertionError):
        get_storage_player_capacity(width)


# Rows


def test_padded_rows_rebind_across_capacities():
    rows = {cap: new_game(3, seed=11, max_players=cap) for cap in (3, 5, 6)}
    reference = rows[3]._array
    for cap, state in rows.items():
        np.testing.assert_array_equal(state._array[:reference.shape[0]], reference)
        np.testing.assert_array_equal(state._array[reference.shape[0]:], 0)
    scratch = GameState(4, max_players=4)
    for cap in (6, 3, 5):
        scratch.rebind(rows[cap]._array, 3, max_players=cap)
        assert scratch.max_players == cap and TURN.num_players(scratch) == 3
        assert [PLAYERS[i].cash(scratch) for i in range(3)] == [420] * 3
        BANK.transfer(scratch, 0, 1, 5)
        assert PLAYERS[1].cash(rows[cap]) == 425  # writes go to the bound row
        BANK.transfer(scratch, 1, 0, 5)


def test_rebind_validates_and_keeps_flags():
    state = new_game(4, max_players=5)
    scratch = GameState(3, private_sale_same_president=True, auction_pass_payout_cap=1)
    scratch.rebind(state._array, 4, max_players=5)
    assert scratch.private_sale_same_president and scratch.auction_pass_payout_cap == 1
    with pytest.raises(AssertionError):
        scratch.rebind(state._array, 4)  # exact width expected
    with pytest.raises(AssertionError):
        scratch.rebind(state._array, 3, max_players=5)  # stored player count differs
    with pytest.raises(AssertionError):
        scratch.rebind(state._array.astype(np.int32), 4, max_players=5)


def test_from_array_copies_and_from_buffer_wraps():
    state = new_game(3, max_players=5)
    row = state._array.copy()
    copied = GameState.from_array(row, 3, max_players=5)
    wrapped = GameState.from_buffer(row, 3, max_players=5)
    np.testing.assert_array_equal(copied._array, row)
    assert copied._array is not row and np.shares_memory(wrapped._array, row)
    BANK.transfer(wrapped, -1, 0, 10)
    assert PLAYERS[0].cash(GameState.from_buffer(row, 3, max_players=5)) == 430
    assert PLAYERS[0].cash(copied) == 420
    with pytest.raises(AssertionError):
        GameState.from_array(row, 3)


def test_initialize_reuses_a_wrapped_buffer_of_the_same_width():
    row = np.zeros(get_layout(5).total_size, dtype=np.int16)
    row[NUM_PLAYERS_SLOT] = 4
    state = GameState.from_buffer(row, 4, max_players=5)
    state.initialize_game(4, seed=3)
    assert state.max_players == 5 and np.shares_memory(state._array, row)
    assert PLAYERS[3].cash(GameState.from_buffer(row, 4, max_players=5)) == 420
    state.initialize_game(4, seed=3, max_players=4)  # a new width needs a new buffer
    assert state._array.shape == (get_layout(4).total_size,)


def test_runtime_flags_default_off_and_validate():
    state = GameState(4)
    assert (state.private_sale_same_president, state.train_sale_same_president,
            state.auction_pass_payout_cap) == (False, False, 0)
    state.auction_pass_payout_cap = 2
    assert state.auction_pass_payout_cap == 2
    with pytest.raises(AssertionError):
        state.auction_pass_payout_cap = -1
    assert not GameState.from_array(new_game(4)._array, 4).private_sale_same_president


# Initial state


@pytest.mark.parametrize('num_players', PLAYER_COUNTS)
def test_initial_state(num_players):
    state = new_game(num_players, max_players=D.MAX_PLAYERS)
    assert total_cash(state) == D.BANK_CASH
    assert [PLAYERS[i].cash(state) for i in range(num_players)] == [D.START_CASH[num_players]] * num_players
    assert BANK.cash(state) == D.BANK_CASH - num_players * D.START_CASH[num_players] and not BANK.broken(state)

    in_play = D.PRIVATES_IN_PLAY[num_players]
    assert [p.owner(state) for p in PRIVATES] == [
        K.OWNER_UNSOLD if in_play >> i & 1 else K.OWNER_NOT_IN_PLAY for i in range(D.NUM_PRIVATES)]
    assert [p.flags(state) for p in PRIVATES] == [0] * D.NUM_PRIVATES

    for corp in CORPS:
        assert (corp.par_index(state), corp.floated(state), corp.president(state)) == (-1, False, -1)
        assert (corp.ipo_shares(state), corp.pool_shares(state), corp.cash(state)) == (10, 0, 0)
        assert corp.tokens_left(state) == D.CORP_TOKENS[corp.corp_id]
        assert (MARKET.row(state, corp.corp_id), MARKET.price(state, corp.corp_id)) == (-1, 0)
        assert all(TRAINS.corp_trains(state, corp.corp_id, t) == 0 for t in range(D.NUM_TRAIN_TYPES))
        assert all(PLAYERS[i].shares(state, corp.corp_id) == 0 for i in range(num_players))

    assert [MAP.supply(state, t) for t in range(D.NUM_SUPPLY_TILES)] == D.TILE_COUNT[:D.NUM_SUPPLY_TILES]
    assert [TRAINS.depot(state, t) for t in range(D.NUM_TRAIN_TYPES)] == D.TRAIN_COUNT
    assert [TRAINS.pool(state, t) for t in range(D.NUM_TRAIN_TYPES)] == [0] * D.NUM_TRAIN_TYPES
    for h in range(D.NUM_HEXES):
        assert (MAP.tile(state, h), MAP.rotation(state, h), MAP.tokens(state, h)) == (D.NUM_SUPPLY_TILES + h, 0, 0)

    assert (TURN.round(state), TURN.step(state)) == (K.ROUND_AUCTION, K.STEP_AUCTION)
    assert (PHASE.current(state), TURN.turn(state), TURN.or_count(state)) == (D.GAME_PHASE_2, 1, 1)
    priority = TURN.priority_player(state)
    assert 0 <= priority < num_players
    assert TURN.active_player(state) == TURN.auction_turn(state) == priority
    assert TURN.auction_private(state) == TURN.pending_private(state) == TURN.return_step(state) == -1
    assert MAP.conn_dirty(state) == 0 and not any(corp.route_dirty(state) for corp in CORPS)

    layout = get_layout(D.MAX_PLAYERS)
    np.testing.assert_array_equal(state._array[layout.players_offset + num_players * layout.player_size:], 0)


@pytest.mark.parametrize('num_players', PLAYER_COUNTS)
def test_initialize_is_deterministic(num_players):
    first = new_game(num_players, seed=1889)
    np.testing.assert_array_equal(first._array, new_game(num_players, seed=1889)._array)
    # Only the priority player depends on the seed, and every seat gets it for some seed.
    priorities = {TURN.priority_player(new_game(num_players, seed=s)) for s in range(64)}
    assert priorities == set(range(num_players))


def test_explicit_priority_player_overrides_the_seed():
    for seed in range(8):
        assert TURN.priority_player(new_game(4, seed=seed, priority_player=0)) == 0
    assert TURN.active_player(new_game(4, seed=1, priority_player=2)) == 2
    with pytest.raises(AssertionError):
        new_game(4, priority_player=4)


def test_initialize_keeps_capacity_and_resets_the_row():
    state = new_game(3, max_players=5)
    BANK.transfer(state, 0, -1, 100)
    state.initialize_game(4, seed=1)
    assert state.max_players == 5 and TURN.num_players(state) == 4
    assert total_cash(state) == D.BANK_CASH and PLAYERS[0].cash(state) == 420
    with pytest.raises(AssertionError):
        state.initialize_game(6, seed=1, max_players=5)
    with pytest.raises(AssertionError):
        GameState(1)
