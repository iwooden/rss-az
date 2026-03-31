#!/usr/bin/env python3
"""Show all ACQ outcomes annotated by the Ruby extractor.

Usage:
    python show_acq_outcomes.py <game_id>
    python show_acq_outcomes.py <game_id> --cross-only
"""
import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    parser = argparse.ArgumentParser(description="Show ACQ outcomes across a game")
    parser.add_argument("game_id", type=int)
    parser.add_argument("--cross-only", action="store_true",
                        help="Only show cross-president transfers")
    args = parser.parse_args()

    extract_path = DATA_DIR / f"{args.game_id}_extract.json"
    if not extract_path.exists():
        print(f"Extract not found: {extract_path}", file=sys.stderr)
        sys.exit(1)

    snapshots = json.loads(extract_path.read_text())

    found = False
    for s in snapshots:
        outcomes = s.get("acq_outcomes")
        if not outcomes:
            continue

        for o in outcomes:
            if args.cross_only and not o.get("cross_president", False):
                continue

            cross = " [CROSS-PRESIDENT]" if o.get("cross_president") else ""
            seller_type = o.get("seller_type", "?")
            seller = o.get("seller", "?")
            seller_id = o.get("seller_id")
            seller_str = f"{seller_type}:{seller}"
            if seller_id is not None:
                seller_str += f"(id={seller_id})"

            print(f"aid={s['action_id']:>5}  {o['company']:<6}  "
                  f"{seller_str} -> {o['buyer']}  price={o['price']}{cross}")
            found = True

    if not found:
        label = "cross-president ACQ outcomes" if args.cross_only else "ACQ outcomes"
        print(f"No {label} found in game {args.game_id}")


if __name__ == "__main__":
    main()
