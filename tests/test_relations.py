from __future__ import annotations

import numpy as np
import pytest

from core.attention_relations import NUM_ATTENTION_RELATIONS, AttentionRelation
from core.relations import (
    get_num_attention_relations,
    get_relation_data,
    get_relation_data_batch,
)
from core.state import GameState
from core.token_data import get_num_tokens
from entities.corp import CORPS
from entities.company import COMPANIES, CompanyLocation
from entities.deck import DECK
from entities.player import PLAYERS
from nn.transformer import UNIFIED_LOGIT_DIM
from train.eval_server import RemoteEvaluator, SharedEvalBuffers


NUM_PLAYERS = 3
COMPANY_TOKEN_START = 1
CORP_TOKEN_START = 46


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

    relations = shared_bufs.get_input_relations_np(0)
    owns_relation_id = int(AttentionRelation.CORP_OWNS_COMPANY)
    owned_by_relation_id = int(AttentionRelation.COMPANY_OWNED_BY_CORP)
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
    assert int(relations[0, owns_relation_id].sum()) == 1
    assert int(relations[0, owned_by_relation_id].sum()) == 1
