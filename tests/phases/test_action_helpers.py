from core.actions import ACTION_AUCTION_PY as ACTION_AUCTION

from tests.phases.conftest import (
    get_legal_actions,
    find_legal_action_with_info,
    find_all_legal_actions_with_info,
)


def test_find_legal_action_with_info_returns_matching_pair(game_state):
    expected_action_id, expected_info = next(
        (action_id, info)
        for action_id, info in get_legal_actions(game_state)
        if info.action_type == ACTION_AUCTION
    )

    action_id, info = find_legal_action_with_info(
        game_state,
        action_type=ACTION_AUCTION,
        company_id=expected_info.company_id,
    )

    assert action_id == expected_action_id
    assert info == expected_info


def test_find_all_legal_actions_with_info_returns_matching_pairs(game_state):
    expected = [
        (action_id, info)
        for action_id, info in get_legal_actions(game_state)
        if info.action_type == ACTION_AUCTION
    ]

    actual = find_all_legal_actions_with_info(
        game_state,
        action_type=ACTION_AUCTION,
    )

    assert actual == expected
