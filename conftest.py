# Root conftest.py — ensures the repo root is on sys.path so pytest
# can import the in-place Cython extensions (core, entities, phases, etc.).

from pathlib import Path

import pytest

_SUBMODULE = Path(__file__).parent / "submodules" / "18xx" / "lib"
# Node id prefixes of the tests that read the 18xx.games submodule, through Ruby,
# or the gitignored replay fixtures it makes: a cloud box has neither.
_NEEDS_18XX = (
    "tests/g1889/games/",
    "tests/g1889/replay/",
    "tests/g1889/test_data_vs_ruby.py",
    "tests/g1889/test_action_slots.py::test_ruby_keeps_private_0_first_while_it_is_unsold",
    "tests/games_18xx/test_replay_harness.py::test_replay_regression",
    "tests/test_18xx_analyze_replay.py::test_analyze_replay_outputs",
    "tests/test_18xx_auto_actions.py::test_attach_expected_auto_actions_matches_18xx_fixture",
    "tests/test_18xx_game_feed.py::test_bus_replay_matches_18xx_server_fields",
    "tests/test_18xx_live_composer.py::test_planned_acq_pass_into_closing_accepts_any_18xx_closing_actor",
    "tests/test_18xx_live_composer.py::test_planned_issue_pass_expects_18xx_to_wait_on_unparable_ipo",
    "tests/test_18xx_live_composer.py::test_bot_posts_explicit_18xx_pass_for_unparable_ipo",
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _SUBMODULE.is_dir():
        return
    skip = pytest.mark.skip(reason="needs submodules/18xx or its replay fixtures")
    for item in items:
        if item.nodeid.startswith(_NEEDS_18XX):
            item.add_marker(skip)
