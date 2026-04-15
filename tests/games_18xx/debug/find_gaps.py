#!/usr/bin/env python3
"""Find gaps in committed actions where undone actions may contain needed phase transitions.

Detects places where consecutive committed extract snapshots have different
rounds with undone actions between them. These gaps often cause harness
failures when the engine needs auto-applied actions (like IPO passes) that
were undone in the 18xx action stream.

Usage:
    python find_gaps.py <game_id>
    python find_gaps.py 213447
"""
import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    parser = argparse.ArgumentParser(
        description="Find phase-transition gaps in committed action stream")
    parser.add_argument("game_id", type=int)
    args = parser.parse_args()

    game_path = DATA_DIR / f"{args.game_id}.json"
    extract_path = DATA_DIR / f"{args.game_id}_extract.json"
    if not extract_path.exists():
        print(f"Extract not found: {extract_path}", file=sys.stderr)
        sys.exit(1)

    extract = json.loads(extract_path.read_text())
    committed = set(extract[0].get("committed_action_ids", []))

    # Load raw actions for gap details
    raw_actions = []
    if game_path.exists():
        game_data = json.loads(game_path.read_text())
        raw_actions = game_data.get("actions", [])

    # Build action ID -> raw action lookup
    raw_by_id = {a.get("id", -1): a for a in raw_actions}

    # Find gaps between consecutive extract snapshots
    found = False
    for i in range(1, len(extract)):
        prev = extract[i - 1]
        curr = extract[i]

        prev_aid = prev["action_id"]
        curr_aid = curr["action_id"]
        prev_round = prev["round"]
        curr_round = curr["round"]

        # Check if there's a round change with undone actions in between
        if prev_round == curr_round:
            continue
        if curr_aid - prev_aid <= 1:
            continue

        # Count undone actions in the gap
        undone = []
        for aid in range(prev_aid + 1, curr_aid):
            if aid not in committed and aid in raw_by_id:
                a = raw_by_id[aid]
                atype = a.get("type", "?")
                entity = a.get("entity", "")
                undone.append(f"    aid={aid} type={atype} entity={entity}")

        if undone:
            found = True
            print(f"Gap: aid {prev_aid} ({prev_round}) -> {curr_aid} ({curr_round})"
                  f"  [{len(undone)} undone actions]")
            for line in undone:
                print(line)
            print()

    if not found:
        print(f"No phase-transition gaps found in game {args.game_id}")


if __name__ == "__main__":
    main()
