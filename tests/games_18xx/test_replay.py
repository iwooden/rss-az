"""Pytest integration for 18xx replay validation."""

from __future__ import annotations

import glob
import os

import pytest

from tests.games_18xx.replay_harness import (
    ReplayHarness,
    ensure_extracts,
    format_mismatches,
    load_ref_states,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SMOKE_GAME_IDS = [205409, 227377, 202494, 203129]
RUN_FULL_REPLAY = os.environ.get("RSS_AZ_REPLAY_FULL") == "1"


def _discover_game_ids() -> list[int]:
    pattern = os.path.join(DATA_DIR, "*.json")
    discovered = sorted(
        int(os.path.basename(path).removesuffix(".json"))
        for path in glob.glob(pattern)
        if not path.endswith("_extract.json")
    )
    if RUN_FULL_REPLAY:
        return discovered
    smoke_available = [game_id for game_id in SMOKE_GAME_IDS if game_id in discovered]
    return smoke_available or discovered


game_ids = _discover_game_ids()

# Confirmed replay divergences. Revalidate before broadening the suite.
SKIP_GAMES = {
    243592,
    210560,
    210896,
    234064,  # Diverges from repo rules on long ACQ receivership/preemption ordering.
}


@pytest.fixture(scope="session", autouse=True)
def _extract_ref_states():
    ensure_extracts(DATA_DIR)


@pytest.mark.parametrize("game_id", game_ids, ids=[str(game_id) for game_id in game_ids])
def test_replay_game(game_id: int):
    if game_id in SKIP_GAMES:
        pytest.skip(f"Known replay divergence (game {game_id})")

    game_json = os.path.join(DATA_DIR, f"{game_id}.json")
    ref_states = load_ref_states(game_json)
    harness = ReplayHarness(
        game_json_path=game_json,
        ref_states=ref_states,
        verbose=not RUN_FULL_REPLAY,
    )
    mismatches = harness.run()
    assert mismatches == [], (
        f"State mismatches found ({len(mismatches)}):\n"
        f"{format_mismatches(mismatches)}"
    )
