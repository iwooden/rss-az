#!/usr/bin/env python3
"""Replay a game up to a specific action and dump engine state.

Usage:
    python tests/games_18xx/debug/replay_to_action.py <game_id> <stop_before_action_id>
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Share the current phase-local replay and state display implementation.
from tests.games_18xx.debug.inspect_replay_point import main


if __name__ == "__main__":
    main()
