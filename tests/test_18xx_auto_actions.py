import json
from pathlib import Path

from utils_18xx.auto_actions import (
    attach_expected_auto_actions,
    strip_post_action_metadata,
)


def test_strip_post_action_metadata_removes_server_generated_fields():
    action = {
        "type": "pass",
        "entity": 3,
        "entity_type": "player",
        "id": 11,
        "user": 3,
        "created_at": 1778526611,
        "auto_actions": [{
            "type": "pass",
            "entity": 2,
            "entity_type": "player",
            "created_at": 1778526611,
        }],
    }

    assert strip_post_action_metadata(action) == {
        "type": "pass",
        "entity": 3,
        "entity_type": "player",
        "auto_actions": [{
            "type": "pass",
            "entity": 2,
            "entity_type": "player",
        }],
    }


def test_attach_expected_auto_actions_matches_18xx_fixture():
    fixture = Path("tests/games_18xx/data/222662.json")
    game_data = json.loads(fixture.read_text())
    actions = game_data["actions"]

    for idx, action in enumerate(actions):
        if action.get("auto_actions"):
            game_before_action = dict(game_data)
            game_before_action["actions"] = actions[:idx]
            post_action = strip_post_action_metadata(action)
            post_action.pop("auto_actions", None)

            enriched = attach_expected_auto_actions(game_before_action, post_action)

            assert enriched["auto_actions"] == [
                strip_post_action_metadata(auto_action)
                for auto_action in action["auto_actions"]
            ]
            return

    raise AssertionError("Expected fixture to contain an action with auto_actions")
