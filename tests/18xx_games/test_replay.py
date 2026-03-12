"""Pytest integration for 18xx game replay validation.

Replays real human game data from 18xx.games through our Cython engine and
compares state at phase boundaries against reference snapshots extracted
by the Ruby state extractor (run inline via subprocess).
"""

import importlib
import os

import pytest

_harness = importlib.import_module("tests.18xx_games.replay_harness")
ReplayHarness = _harness.ReplayHarness
extract_ref_states = _harness.extract_ref_states
format_mismatches = _harness.format_mismatches

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@pytest.mark.parametrize("game_id", [224885, 226358, 239448, 244645])
def test_replay_game(game_id):
    """Replay an 18xx game and verify state matches at phase boundaries."""
    game_json = os.path.join(DATA_DIR, f"{game_id}.json")
    assert os.path.exists(game_json), f"Game JSON not found: {game_json}"

    ref_states = extract_ref_states(game_json)

    harness = ReplayHarness(
        game_json_path=game_json,
        ref_states=ref_states,
        verbose=True,
    )
    mismatches = harness.run()
    assert mismatches == [], (
        f"State mismatches found ({len(mismatches)}):\n"
        f"{format_mismatches(mismatches)}"
    )
