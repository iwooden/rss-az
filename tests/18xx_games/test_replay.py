"""Pytest integration for 18xx game replay validation.

Replays real human game data from 18xx.games through our Cython engine and
compares state at phase boundaries against reference snapshots extracted
by the Ruby state extractor.
"""

import importlib
import os

import pytest

_harness = importlib.import_module("tests.18xx_games.replay_harness")
ReplayHarness = _harness.ReplayHarness
format_mismatches = _harness.format_mismatches

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@pytest.mark.parametrize("game_id", [224885])
def test_replay_game(game_id):
    """Replay an 18xx game and verify state matches at phase boundaries."""
    game_json = os.path.join(DATA_DIR, f"{game_id}.json")
    states_json = os.path.join(DATA_DIR, f"{game_id}_states.json")

    assert os.path.exists(game_json), f"Game JSON not found: {game_json}"
    assert os.path.exists(states_json), (
        f"States JSON not found: {states_json}. "
        "Run: ruby tests/18xx_games/extract_states.rb "
        f"tests/18xx_games/data/{game_id}.json > {states_json}"
    )

    harness = ReplayHarness(
        game_json_path=game_json,
        states_json_path=states_json,
        verbose=True,
    )
    mismatches = harness.run()
    assert mismatches == [], (
        f"State mismatches found ({len(mismatches)}):\n"
        f"{format_mismatches(mismatches)}"
    )
