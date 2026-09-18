from __future__ import annotations

import numpy as np
import pytest

from core.attention_relations import (
    ATTENTION_RELATION_COORD_WIDTH,
    ATTENTION_RELATION_SCALES,
    MAX_ATTENTION_RELATION_EDGES,
    NUM_ATTENTION_RELATIONS,
    AttentionRelation,
)
from core.data import PY_SHARE_DIVISOR
from core.relations import (
    get_num_attention_relations,
    get_relation_coord_data,
    get_relation_coord_data_batch,
    get_relation_data,
    get_relation_data_batch,
)
from core.state import GameState, get_layout
from core.token_data import get_num_tokens, get_token_dim
from entities.corp import CORPS
from entities.company import COMPANIES, CompanyLocation
from entities.deck import DECK
from entities.player import PLAYERS
from nn.policy_layout import UNIFIED_LOGIT_DIM
from nn.model_contract import ModelInputSpec
from train.eval_server import RemoteEvaluator, SharedEvalBuffers
from train.replay_buffer import ReplayBuffer


NUM_PLAYERS = 3
COMPANY_TOKEN_START = 1
CORP_TOKEN_START = 46
PLAYER_TOKEN_START = 54


def _materialize_relation_coords_np(
    coords: np.ndarray,
    *,
    num_tokens: int,
) -> np.ndarray:
    dense = np.zeros(
        (NUM_ATTENTION_RELATIONS, num_tokens, num_tokens),
        dtype=np.uint8,
    )
    for relation_id, query_tok, key_tok, value in coords:
        if value == 0:
            continue
        dense[int(relation_id), int(query_tok), int(key_tok)] = value
    return dense


def _state_with_corp_company(corp_id: int, company_id: int) -> GameState:
    state = GameState(NUM_PLAYERS)
    state.initialize_game(NUM_PLAYERS, seed=100 + corp_id * 10 + company_id)
    DECK.set_company_location(
        state,
        company_id,
        int(CompanyLocation.LOC_PLAYER),
        0,
    )
    CORPS[corp_id].float_corp(
        state,
        player_id=0,
        company_id=company_id,
        market_index=10,
    )
    return state


def _state_with_player_company(player_id: int, company_id: int) -> GameState:
    state = GameState(NUM_PLAYERS)
    state.initialize_game(NUM_PLAYERS, seed=200 + player_id * 10 + company_id)
    DECK.set_company_location(
        state,
        company_id,
        int(CompanyLocation.LOC_PLAYER),
        player_id,
    )
    return state


def _state_with_player_company_for_players(
    num_players: int,
    player_id: int,
    company_id: int,
    *,
    max_players: int = 0,
) -> GameState:
    state = GameState(num_players, max_players=max_players)
    state.initialize_game(num_players, seed=400 + num_players * 100 + company_id)
    DECK.set_company_location(
        state,
        company_id,
        int(CompanyLocation.LOC_PLAYER),
        player_id,
    )
    return state


def _state_with_fi_company(company_id: int) -> GameState:
    state = GameState(NUM_PLAYERS)
    state.initialize_game(NUM_PLAYERS, seed=300 + company_id)
    DECK.set_company_location(
        state,
        company_id,
        int(CompanyLocation.LOC_FI),
    )
    return state


def test_relation_count_matches_cython_source() -> None:
    assert NUM_ATTENTION_RELATIONS == get_num_attention_relations()


def test_share_normalization_matches_engine_maximum():
    assert PY_SHARE_DIVISOR == max(corp.get_total_shares() for corp in CORPS) == 7
    for relation in AttentionRelation:
        expected = 1 / PY_SHARE_DIVISOR if relation in (
            AttentionRelation.PLAYER_CORP_SHARE_COUNT,
            AttentionRelation.CORP_PLAYER_SHARE_COUNT,
        ) else 1
        assert ATTENTION_RELATION_SCALES[relation] == expected


def _state_with_share_quantity(num_players, shares):
    state = GameState(num_players, max_players=5)
    state.initialize_game(num_players, seed=321, max_players=5)
    DECK.set_company_location(state, 0, int(CompanyLocation.LOC_PLAYER), 0)
    CORPS[0].float_corp(state, player_id=0, company_id=0, market_index=10)
    PLAYERS[0].set_shares(state, 0, 0)
    CORPS[0].set_unissued_shares(state, 0)
    CORPS[0].set_issued_shares(state, 7)
    CORPS[0].set_bank_shares(state, 7)
    PLAYERS[num_players - 1].set_shares(state, 0, shares)
    return state


@pytest.mark.parametrize("num_players", [3, 4, 5])
@pytest.mark.parametrize("shares", [0, 1, 4, 7])
def test_raw_share_quantities_survive_dense_and_sparse_extraction(num_players, shares):
    state = _state_with_share_quantity(num_players, shares)
    n = get_num_tokens(5)
    dense = np.full((NUM_ATTENTION_RELATIONS, n, n), 99, dtype=np.uint8)
    coords = np.full((MAX_ATTENTION_RELATION_EDGES, ATTENTION_RELATION_COORD_WIDTH), 99, dtype=np.uint8)
    get_relation_data(state, dense, max_players=5)
    count = get_relation_coord_data(state, coords, max_players=5)
    player, corp = PLAYER_TOKEN_START + num_players - 1, CORP_TOKEN_START
    assert dense[AttentionRelation.PLAYER_CORP_SHARE_COUNT, player, corp] == shares
    assert dense[AttentionRelation.CORP_PLAYER_SHARE_COUNT, corp, player] == shares
    assert dense[AttentionRelation.PLAYER_OWNS_CORP_SHARES, player, corp] == bool(shares)
    assert dense[AttentionRelation.CORP_HAS_PLAYER_SHAREHOLDER, corp, player] == bool(shares)
    assert not dense[:, PLAYER_TOKEN_START + num_players:].any()
    assert not dense[:, :, PLAYER_TOKEN_START + num_players:].any()
    assert count == np.count_nonzero(dense)
    assert not coords[count:].any()
    np.testing.assert_array_equal(dense, _materialize_relation_coords_np(coords, num_tokens=n))

    # Reusing worker buffers after selling out must clear both quantities and presence.
    PLAYERS[num_players - 1].set_shares(state, 0, 0)
    get_relation_data(state, dense, max_players=5)
    get_relation_coord_data(state, coords, max_players=5)
    assert not dense[AttentionRelation.PLAYER_CORP_SHARE_COUNT].any()
    assert not dense[AttentionRelation.CORP_PLAYER_SHARE_COUNT].any()
    np.testing.assert_array_equal(dense, _materialize_relation_coords_np(coords, num_tokens=n))


def test_share_counts_round_trip_through_worker_ipc_and_replay(monkeypatch):
    states = [_state_with_share_quantity(n, shares) for n, shares in ((3, 4), (5, 7))]
    spec = ModelInputSpec(
        model_type="transformer", num_players=5, value_dim=5,
        policy_dim=int(UNIFIED_LOGIT_DIM), num_tokens=get_num_tokens(5),
        token_dim=get_token_dim(3), layout_version=3,
    )
    shared = SharedEvalBuffers(num_workers=2, batch_size=1, num_players=5, input_spec=spec)
    shared.init_bitmap([(0, 1), (1, 1)])
    mask = np.zeros((2, int(UNIFIED_LOGIT_DIM)), dtype=np.uint8)
    mask[:, 0] = 1
    # Each MCTS worker serves one player count; the server combines workers.
    for worker, state in enumerate(states):
        evaluator = RemoteEvaluator(5, shared, worker_idx=worker)
        monkeypatch.setattr(evaluator, "_request_eval", lambda n: None)
        evaluator.evaluate_leaves([state._array], mask[worker:worker + 1])
    wire = np.stack([shared.get_input_relation_coords_np(worker)[0] for worker in range(2)])
    replay = ReplayBuffer(capacity=2, state_size_int16=get_layout(5).total_size,
                          num_players=5, min_players=3, max_players=5)
    replay.add_stacked(
        states=np.stack([s._array for s in states]), phase_ids=np.zeros(2, dtype=np.int8),
        legal_masks=mask, policy_targets=mask.astype(np.float32),
        value_targets=np.zeros((2, 5), dtype=np.float32),
        player_counts=np.array([3, 5], dtype=np.uint8),
    )
    batch = replay.sample(2, np.random.default_rng(0))
    for row, n in enumerate(batch["player_counts"].tolist()):
        wire_row = 0 if n == 3 else 1
        materialized = _materialize_relation_coords_np(wire[wire_row], num_tokens=get_num_tokens(5))
        assert materialized[AttentionRelation.PLAYER_CORP_SHARE_COUNT,
                            PLAYER_TOKEN_START + n - 1, CORP_TOKEN_START] == (4 if n == 3 else 7)
        np.testing.assert_array_equal(batch["relations"][row].numpy(), materialized)


def test_sparse_capacity_covers_full_portfolios_and_shareholders():
    state = GameState(5)
    state.initialize_game(5, seed=456)
    for corp_id, corp in enumerate(CORPS):
        DECK.set_company_location(state, corp_id, int(CompanyLocation.LOC_PLAYER), 0)
        corp.float_corp(state, player_id=0, company_id=corp_id, market_index=corp_id + 5)
        total = corp.get_total_shares()
        corp.set_bank_shares(state, total - PLAYERS[0].get_shares(state, corp_id))
        corp.set_issued_shares(state, total)
        corp.set_unissued_shares(state, 0)
        for player_id in range(min(5, total)):
            PLAYERS[player_id].set_shares(state, corp_id, 1)
    for company_id in range(8, 36):
        DECK.set_company_location(state, company_id, int(CompanyLocation.LOC_CORP), company_id % 8)
    coords = np.zeros((MAX_ATTENTION_RELATION_EDGES, ATTENTION_RELATION_COORD_WIDTH), dtype=np.uint8)
    n = get_num_tokens(5)
    dense = np.zeros((NUM_ATTENTION_RELATIONS, n, n), dtype=np.uint8)
    count = get_relation_coord_data(state, coords)
    get_relation_data(state, dense)
    assert count == 2 * 36 + 4 * 38 + 2 * 8 == 240
    assert count <= MAX_ATTENTION_RELATION_EDGES
    np.testing.assert_array_equal(dense, _materialize_relation_coords_np(coords, num_tokens=n))


def test_get_relation_data_marks_corp_company_ownership_directions() -> None:
    corp_id = 2
    company_id = 5
    state = _state_with_corp_company(corp_id, company_id)
    num_tokens = get_num_tokens(NUM_PLAYERS)
    relations = np.full(
        (NUM_ATTENTION_RELATIONS, num_tokens, num_tokens),
        7,
        dtype=np.uint8,
    )

    get_relation_data(state, relations)

    owns_relation_id = int(AttentionRelation.CORP_OWNS_COMPANY)
    owned_by_relation_id = int(AttentionRelation.COMPANY_OWNED_BY_CORP)
    corp_tok = CORP_TOKEN_START + corp_id
    company_tok = COMPANY_TOKEN_START + company_id
    assert relations[owns_relation_id, corp_tok, company_tok] == 1
    assert relations[owns_relation_id, company_tok, corp_tok] == 0
    assert relations[owned_by_relation_id, company_tok, corp_tok] == 1
    assert relations[owned_by_relation_id, corp_tok, company_tok] == 0
    assert int(relations[owns_relation_id].sum()) == 1
    assert int(relations[owned_by_relation_id].sum()) == 1


def test_get_relation_data_treats_acq_pile_as_corp_company_ownership() -> None:
    corp_id = 2
    acq_company_id = 6
    state = _state_with_corp_company(corp_id, company_id=5)
    DECK.set_company_location(
        state,
        acq_company_id,
        int(CompanyLocation.LOC_PLAYER),
        0,
    )
    COMPANIES[acq_company_id].transfer_to_corp_acquisition(state, corp_id)
    num_tokens = get_num_tokens(NUM_PLAYERS)
    relations = np.zeros(
        (NUM_ATTENTION_RELATIONS, num_tokens, num_tokens),
        dtype=np.uint8,
    )

    get_relation_data(state, relations)

    owns_relation_id = int(AttentionRelation.CORP_OWNS_COMPANY)
    owned_by_relation_id = int(AttentionRelation.COMPANY_OWNED_BY_CORP)
    corp_tok = CORP_TOKEN_START + corp_id
    company_tok = COMPANY_TOKEN_START + acq_company_id
    assert relations[owns_relation_id, corp_tok, company_tok] == 1
    assert relations[owned_by_relation_id, company_tok, corp_tok] == 1


def test_get_relation_data_marks_player_company_ownership_directions() -> None:
    player_id = 1
    company_id = 7
    state = _state_with_player_company(player_id, company_id)
    num_tokens = get_num_tokens(NUM_PLAYERS)
    relations = np.zeros(
        (NUM_ATTENTION_RELATIONS, num_tokens, num_tokens),
        dtype=np.uint8,
    )

    get_relation_data(state, relations)

    owns_relation_id = int(AttentionRelation.PLAYER_OWNS_COMPANY)
    owned_by_relation_id = int(AttentionRelation.COMPANY_OWNED_BY_PLAYER)
    player_tok = 54 + player_id
    company_tok = COMPANY_TOKEN_START + company_id
    assert relations[owns_relation_id, player_tok, company_tok] == 1
    assert relations[owns_relation_id, company_tok, player_tok] == 0
    assert relations[owned_by_relation_id, company_tok, player_tok] == 1
    assert relations[owned_by_relation_id, player_tok, company_tok] == 0
    assert int(relations.sum()) == 2


def test_get_relation_data_marks_fi_company_ownership_directions() -> None:
    company_id = 9
    state = _state_with_fi_company(company_id)
    num_tokens = get_num_tokens(NUM_PLAYERS)
    relations = np.zeros(
        (NUM_ATTENTION_RELATIONS, num_tokens, num_tokens),
        dtype=np.uint8,
    )

    get_relation_data(state, relations)

    owns_relation_id = int(AttentionRelation.FI_OWNS_COMPANY)
    owned_by_relation_id = int(AttentionRelation.COMPANY_OWNED_BY_FI)
    fi_tok = 37
    company_tok = COMPANY_TOKEN_START + company_id
    assert relations[owns_relation_id, fi_tok, company_tok] == 1
    assert relations[owns_relation_id, company_tok, fi_tok] == 0
    assert relations[owned_by_relation_id, company_tok, fi_tok] == 1
    assert relations[owned_by_relation_id, fi_tok, company_tok] == 0
    assert int(relations.sum()) == 2


def test_get_relation_data_marks_player_corp_shareholder_directions() -> None:
    corp_id = 4
    player_id = 2
    company_id = 10
    state = _state_with_corp_company(corp_id, company_id)
    num_tokens = get_num_tokens(NUM_PLAYERS)
    relations = np.zeros(
        (NUM_ATTENTION_RELATIONS, num_tokens, num_tokens),
        dtype=np.uint8,
    )

    get_relation_data(state, relations)

    owns_relation_id = int(AttentionRelation.PLAYER_OWNS_CORP_SHARES)
    shareholder_relation_id = int(AttentionRelation.CORP_HAS_PLAYER_SHAREHOLDER)
    player_tok = 54 + player_id
    corp_tok = CORP_TOKEN_START + corp_id
    assert relations[owns_relation_id, player_tok, corp_tok] == 0
    assert relations[shareholder_relation_id, corp_tok, player_tok] == 0

    PLAYERS[player_id].set_shares(state, corp_id, 1)
    get_relation_data(state, relations)

    assert relations[owns_relation_id, player_tok, corp_tok] == 1
    assert relations[owns_relation_id, corp_tok, player_tok] == 0
    assert relations[shareholder_relation_id, corp_tok, player_tok] == 1
    assert relations[shareholder_relation_id, player_tok, corp_tok] == 0


def test_get_relation_data_marks_player_corp_president_directions() -> None:
    corp_id = 5
    player_id = 0
    company_id = 11
    state = _state_with_corp_company(corp_id, company_id)
    num_tokens = get_num_tokens(NUM_PLAYERS)
    relations = np.zeros(
        (NUM_ATTENTION_RELATIONS, num_tokens, num_tokens),
        dtype=np.uint8,
    )

    get_relation_data(state, relations)

    president_relation_id = int(AttentionRelation.PLAYER_PRESIDENT_OF_CORP)
    corp_president_relation_id = int(AttentionRelation.CORP_PRESIDENT_PLAYER)
    player_tok = 54 + player_id
    corp_tok = CORP_TOKEN_START + corp_id
    assert relations[president_relation_id, player_tok, corp_tok] == 1
    assert relations[president_relation_id, corp_tok, player_tok] == 0
    assert relations[corp_president_relation_id, corp_tok, player_tok] == 1
    assert relations[corp_president_relation_id, player_tok, corp_tok] == 0


def test_get_relation_data_batch_marks_each_row_independently() -> None:
    state_a = _state_with_corp_company(corp_id=0, company_id=3)
    state_b = _state_with_corp_company(corp_id=1, company_id=4)
    num_tokens = get_num_tokens(NUM_PLAYERS)
    relations = np.full(
        (2, NUM_ATTENTION_RELATIONS, num_tokens, num_tokens),
        7,
        dtype=np.uint8,
    )

    get_relation_data_batch(
        [state_a._array, state_b._array],
        NUM_PLAYERS,
        relations,
    )

    owns_relation_id = int(AttentionRelation.CORP_OWNS_COMPANY)
    owned_by_relation_id = int(AttentionRelation.COMPANY_OWNED_BY_CORP)
    assert (
        relations[
            0,
            owns_relation_id,
            CORP_TOKEN_START + 0,
            COMPANY_TOKEN_START + 3,
        ]
        == 1
    )
    assert (
        relations[
            1,
            owns_relation_id,
            CORP_TOKEN_START + 1,
            COMPANY_TOKEN_START + 4,
        ]
        == 1
    )
    assert (
        relations[
            0,
            owned_by_relation_id,
            COMPANY_TOKEN_START + 3,
            CORP_TOKEN_START + 0,
        ]
        == 1
    )
    assert (
        relations[
            1,
            owned_by_relation_id,
            COMPANY_TOKEN_START + 4,
            CORP_TOKEN_START + 1,
        ]
        == 1
    )
    assert int(relations[0, owns_relation_id].sum()) == 1
    assert int(relations[0, owned_by_relation_id].sum()) == 1
    assert int(relations[1, owns_relation_id].sum()) == 1
    assert int(relations[1, owned_by_relation_id].sum()) == 1


def test_get_relation_coord_data_matches_dense_relation_planes() -> None:
    state = _state_with_corp_company(corp_id=4, company_id=12)
    PLAYERS[2].set_shares(state, 4, 1)
    num_tokens = get_num_tokens(NUM_PLAYERS)
    dense = np.zeros(
        (NUM_ATTENTION_RELATIONS, num_tokens, num_tokens),
        dtype=np.uint8,
    )
    coords = np.full(
        (MAX_ATTENTION_RELATION_EDGES, ATTENTION_RELATION_COORD_WIDTH),
        7,
        dtype=np.uint8,
    )

    get_relation_data(state, dense)
    count = get_relation_coord_data(state, coords)
    materialized = _materialize_relation_coords_np(coords, num_tokens=num_tokens)

    assert count == np.count_nonzero(dense)
    np.testing.assert_array_equal(materialized, dense)
    assert int(coords[count:].sum()) == 0


def test_get_relation_coord_data_batch_matches_dense_relation_planes() -> None:
    states = [
        _state_with_corp_company(corp_id=0, company_id=3),
        _state_with_player_company(player_id=2, company_id=14),
    ]
    PLAYERS[1].set_shares(states[0], 0, 1)
    num_tokens = get_num_tokens(NUM_PLAYERS)
    dense = np.zeros(
        (2, NUM_ATTENTION_RELATIONS, num_tokens, num_tokens),
        dtype=np.uint8,
    )
    coords = np.full(
        (2, MAX_ATTENTION_RELATION_EDGES, ATTENTION_RELATION_COORD_WIDTH),
        7,
        dtype=np.uint8,
    )

    get_relation_data_batch([s._array for s in states], NUM_PLAYERS, dense)
    get_relation_coord_data_batch([s._array for s in states], NUM_PLAYERS, coords)

    for i in range(2):
        materialized = _materialize_relation_coords_np(
            coords[i],
            num_tokens=num_tokens,
        )
        np.testing.assert_array_equal(materialized, dense[i])


def test_get_relation_data_with_max_players_pads_relation_planes() -> None:
    num_players = 3
    max_players = 5
    player_id = 2
    company_id = 15
    state = _state_with_player_company_for_players(
        num_players,
        player_id,
        company_id,
        max_players=max_players,
    )
    num_tokens = get_num_tokens(max_players)
    relations = np.full(
        (NUM_ATTENTION_RELATIONS, num_tokens, num_tokens),
        7,
        dtype=np.uint8,
    )

    get_relation_data(state, relations, max_players=max_players)

    owns_relation_id = int(AttentionRelation.PLAYER_OWNS_COMPANY)
    owned_by_relation_id = int(AttentionRelation.COMPANY_OWNED_BY_PLAYER)
    player_tok = PLAYER_TOKEN_START + player_id
    company_tok = COMPANY_TOKEN_START + company_id
    padded_start = PLAYER_TOKEN_START + num_players
    assert relations.shape == (
        NUM_ATTENTION_RELATIONS,
        get_num_tokens(max_players),
        get_num_tokens(max_players),
    )
    assert relations[owns_relation_id, player_tok, company_tok] == 1
    assert relations[owned_by_relation_id, company_tok, player_tok] == 1
    assert int(relations[:, padded_start:, :].sum()) == 0
    assert int(relations[:, :, padded_start:].sum()) == 0


def test_get_relation_coord_data_with_max_players_omits_padded_player_tokens() -> None:
    num_players = 3
    max_players = 5
    state = _state_with_player_company_for_players(
        num_players,
        player_id=2,
        company_id=16,
        max_players=max_players,
    )
    dense = np.zeros(
        (
            NUM_ATTENTION_RELATIONS,
            get_num_tokens(max_players),
            get_num_tokens(max_players),
        ),
        dtype=np.uint8,
    )
    coords = np.full(
        (MAX_ATTENTION_RELATION_EDGES, ATTENTION_RELATION_COORD_WIDTH),
        7,
        dtype=np.uint8,
    )

    get_relation_data(state, dense, max_players=max_players)
    count = get_relation_coord_data(state, coords, max_players=max_players)
    materialized = _materialize_relation_coords_np(
        coords,
        num_tokens=get_num_tokens(max_players),
    )

    assert count == np.count_nonzero(dense)
    assert count > 0
    assert int(coords[count:].sum()) == 0
    assert np.all(coords[:count, 1:3] < PLAYER_TOKEN_START + num_players)
    np.testing.assert_array_equal(materialized, dense)


def test_relation_batch_handles_mixed_player_counts_with_max_players() -> None:
    max_players = 5
    states = [
        _state_with_player_company_for_players(
            num_players,
            player_id=num_players - 1,
            company_id=17 + i,
            max_players=max_players,
        )
        for i, num_players in enumerate((3, 4, 5))
    ]
    num_tokens = get_num_tokens(max_players)
    dense = np.zeros(
        (len(states), NUM_ATTENTION_RELATIONS, num_tokens, num_tokens),
        dtype=np.uint8,
    )
    coords = np.full(
        (
            len(states),
            MAX_ATTENTION_RELATION_EDGES,
            ATTENTION_RELATION_COORD_WIDTH,
        ),
        7,
        dtype=np.uint8,
    )

    get_relation_data_batch([s._array for s in states], dense, max_players=max_players)
    get_relation_coord_data_batch(
        [s._array for s in states],
        coords,
        max_players=max_players,
    )

    owns_relation_id = int(AttentionRelation.PLAYER_OWNS_COMPANY)
    owned_by_relation_id = int(AttentionRelation.COMPANY_OWNED_BY_PLAYER)
    for i, num_players in enumerate((3, 4, 5)):
        player_tok = PLAYER_TOKEN_START + num_players - 1
        company_tok = COMPANY_TOKEN_START + 17 + i
        assert dense[i, owns_relation_id, player_tok, company_tok] == 1
        assert dense[i, owned_by_relation_id, company_tok, player_tok] == 1

        padded_start = PLAYER_TOKEN_START + num_players
        assert int(dense[i, :, padded_start:, :].sum()) == 0
        assert int(dense[i, :, :, padded_start:].sum()) == 0

        materialized = _materialize_relation_coords_np(
            coords[i],
            num_tokens=num_tokens,
        )
        np.testing.assert_array_equal(materialized, dense[i])


def test_remote_evaluator_evaluate_leaves_populates_relation_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corp_id = 3
    company_id = 8
    state = _state_with_corp_company(corp_id, company_id)
    shared_bufs = SharedEvalBuffers(num_workers=1, batch_size=2, num_players=NUM_PLAYERS)
    shared_bufs.init_bitmap([(0, 1)])
    evaluator = RemoteEvaluator(NUM_PLAYERS, shared_bufs, worker_idx=0)
    monkeypatch.setattr(evaluator, "_request_eval", lambda n: None)

    legal_mask = np.zeros((1, int(UNIFIED_LOGIT_DIM)), dtype=np.uint8)
    legal_mask[0, 0] = 1

    evaluator.evaluate_leaves([state._array], legal_mask)

    relation_coords = shared_bufs.get_input_relation_coords_np(0)
    relations = _materialize_relation_coords_np(
        relation_coords[0],
        num_tokens=get_num_tokens(NUM_PLAYERS),
    )
    owns_relation_id = int(AttentionRelation.CORP_OWNS_COMPANY)
    owned_by_relation_id = int(AttentionRelation.COMPANY_OWNED_BY_CORP)
    assert (
        relations[
            owns_relation_id,
            CORP_TOKEN_START + corp_id,
            COMPANY_TOKEN_START + company_id,
        ]
        == 1
    )
    assert (
        relations[
            owned_by_relation_id,
            COMPANY_TOKEN_START + company_id,
            CORP_TOKEN_START + corp_id,
        ]
        == 1
    )
    assert int(relations[owns_relation_id].sum()) == 1
    assert int(relations[owned_by_relation_id].sum()) == 1


def test_remote_evaluator_slices_max_width_values_to_actual_players(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    max_players = 5
    num_players = 3
    state = GameState(num_players, max_players=max_players)
    state.initialize_game(num_players, seed=777, max_players=max_players)
    shared_bufs = SharedEvalBuffers(
        num_workers=1,
        batch_size=2,
        num_players=max_players,
    )
    shared_bufs.init_bitmap([(0, 1)])
    evaluator = RemoteEvaluator(max_players, shared_bufs, worker_idx=0)
    monkeypatch.setattr(evaluator, "_request_eval", lambda n: None)

    shared_bufs.get_output_values_np(0)[0] = np.arange(
        max_players, dtype=np.float32,
    )
    legal_mask = np.zeros((1, int(UNIFIED_LOGIT_DIM)), dtype=np.uint8)
    legal_mask[0, 0] = 1

    _priors, values = evaluator.evaluate_leaves([state._array], legal_mask)

    assert values.shape == (1, num_players)
    np.testing.assert_array_equal(
        values[0],
        np.arange(num_players, dtype=np.float32),
    )


def _one_row_training_targets() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phase_ids = np.array([0], dtype=np.int8)
    legal_masks = np.zeros((1, int(UNIFIED_LOGIT_DIM)), dtype=np.uint8)
    legal_masks[0, 0] = 1
    policy_targets = np.zeros((1, int(UNIFIED_LOGIT_DIM)), dtype=np.float32)
    policy_targets[0, 0] = 1.0
    value_targets = np.zeros((1, NUM_PLAYERS), dtype=np.float32)
    return phase_ids, legal_masks, policy_targets, value_targets


def test_replay_buffer_sample_materializes_relation_planes() -> None:
    corp_id = 2
    company_id = 5
    state = _state_with_corp_company(corp_id, company_id)
    buffer = ReplayBuffer(
        capacity=1,
        state_size_int16=get_layout(NUM_PLAYERS).total_size,
        num_players=NUM_PLAYERS,
    )
    phase_ids, legal_masks, policy_targets, value_targets = _one_row_training_targets()
    buffer.add_stacked(
        states=state._array[None, :],
        phase_ids=phase_ids,
        legal_masks=legal_masks,
        policy_targets=policy_targets,
        value_targets=value_targets,
    )

    batch = buffer.sample(1, np.random.default_rng(0))
    relations = batch["relations"].numpy()

    owns_relation_id = int(AttentionRelation.CORP_OWNS_COMPANY)
    owned_by_relation_id = int(AttentionRelation.COMPANY_OWNED_BY_CORP)
    assert relations.dtype == np.uint8
    assert relations.shape == (
        1,
        NUM_ATTENTION_RELATIONS,
        get_num_tokens(NUM_PLAYERS),
        get_num_tokens(NUM_PLAYERS),
    )
    assert (
        relations[
            0,
            owns_relation_id,
            CORP_TOKEN_START + corp_id,
            COMPANY_TOKEN_START + company_id,
        ]
        == 1
    )
    assert (
        relations[
            0,
            owned_by_relation_id,
            COMPANY_TOKEN_START + company_id,
            CORP_TOKEN_START + corp_id,
        ]
        == 1
    )


def test_replay_buffer_sample_into_fills_relation_scratch() -> None:
    corp_id = 4
    company_id = 10
    state = _state_with_corp_company(corp_id, company_id)
    buffer = ReplayBuffer(
        capacity=1,
        state_size_int16=get_layout(NUM_PLAYERS).total_size,
        num_players=NUM_PLAYERS,
    )
    phase_ids, legal_masks, policy_targets, value_targets = _one_row_training_targets()
    buffer.add_stacked(
        states=state._array[None, :],
        phase_ids=phase_ids,
        legal_masks=legal_masks,
        policy_targets=policy_targets,
        value_targets=value_targets,
    )

    states_out = np.empty((1, get_layout(NUM_PLAYERS).total_size), dtype=np.int16)
    phase_ids_out = np.empty(1, dtype=np.int64)
    legal_masks_out = np.empty((1, int(UNIFIED_LOGIT_DIM)), dtype=np.uint8)
    policy_targets_out = np.empty((1, int(UNIFIED_LOGIT_DIM)), dtype=np.float32)
    value_targets_out = np.empty((1, NUM_PLAYERS), dtype=np.float32)
    relations_out = np.full(
        (
            1,
            NUM_ATTENTION_RELATIONS,
            get_num_tokens(NUM_PLAYERS),
            get_num_tokens(NUM_PLAYERS),
        ),
        7,
        dtype=np.uint8,
    )

    buffer.sample_into(
        1,
        np.random.default_rng(0),
        states_out,
        phase_ids_out,
        legal_masks_out,
        policy_targets_out,
        value_targets_out,
        relations_out=relations_out,
    )

    owns_relation_id = int(AttentionRelation.CORP_OWNS_COMPANY)
    owned_by_relation_id = int(AttentionRelation.COMPANY_OWNED_BY_CORP)
    assert (
        relations_out[
            0,
            owns_relation_id,
            CORP_TOKEN_START + corp_id,
            COMPANY_TOKEN_START + company_id,
        ]
        == 1
    )
    assert (
        relations_out[
            0,
            owned_by_relation_id,
            COMPANY_TOKEN_START + company_id,
            CORP_TOKEN_START + corp_id,
        ]
        == 1
    )
    assert 7 not in relations_out
