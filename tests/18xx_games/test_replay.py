"""Pytest integration for 18xx game replay validation.

Replays real human game data from 18xx.games through our Cython engine and
compares state at phase boundaries against reference snapshots extracted
by the Ruby state extractor.

Game JSON files are discovered dynamically from the data/ directory.
Reference states are pre-extracted to {game_id}_extract.json files by
the Ruby extractor (runs once per session if any are missing or stale).
"""

import glob
import importlib
import os

import pytest

_harness = importlib.import_module("tests.18xx_games.replay_harness")
ReplayHarness = _harness.ReplayHarness
ensure_extracts = _harness.ensure_extracts
load_ref_states = _harness.load_ref_states
format_mismatches = _harness.format_mismatches

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _discover_game_ids():
    """Find all game IDs from JSON files in the data directory."""
    pattern = os.path.join(DATA_DIR, "*.json")
    return sorted(
        int(os.path.basename(f).removesuffix(".json"))
        for f in glob.glob(pattern)
        if not f.endswith("_extract.json")
    )


game_ids = _discover_game_ids()

# Games with known 18xx.games engine bugs — skip with explanation.
SKIP_GAMES = {
    # 18xx bug: receivership corps only buy ONE company from FI per turn.
    # Rules say "Repeat until cannot afford more" (RULES.md Phase 3).
    # DA (receivership, cash=47) buys OL (high=20) but not HE (high=18)
    # despite having 27 cash remaining.  Our engine correctly repeats.
    # Tracked: rss-az-xwzk (submit PR to 18xx.games repo)
    243592,
    # Current vendored 18xx engine does let OS buy from FI at company face
    # value, but its "always considered highest share price" special does not
    # appear to be applied consistently during receivership FI auto-buy
    # ordering. These replays diverge when OS loses priority to another
    # receivership corp before the later ACQ state catches up.
    # Tracked: rss-az-76nd
    210560,
    210896,
}


@pytest.fixture(scope="session", autouse=True)
def _extract_ref_states():
    """Ensure all reference state extracts exist and are up to date."""
    ensure_extracts(DATA_DIR)


@pytest.mark.parametrize("game_id", game_ids, ids=[str(g) for g in game_ids])
def test_replay_game(game_id):
    """Replay an 18xx game and verify state matches at phase boundaries."""
    if game_id in SKIP_GAMES:
        pytest.skip(f"Known 18xx.games engine bug (game {game_id})")

    game_json = os.path.join(DATA_DIR, f"{game_id}.json")

    ref_states = load_ref_states(game_json)

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
