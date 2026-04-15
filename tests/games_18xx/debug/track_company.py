#!/usr/bin/env python3
"""Track a company's ownership across a game's extract snapshots.

Usage:
    python track_company.py <game_id> <company_name> [--range START-END]

Examples:
    python track_company.py 210560 SJ
    python track_company.py 210560 RENFE --range 200-300
"""
import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def find_owner(snapshot, company_name):
    """Return (owner_type, owner_name, extra_info) for a company."""
    for c in snapshot.get("corporations", []):
        if company_name in c.get("companies", []):
            return "corp", c["name"], f"price={c['price']} cash={c['cash']}"
    for p in snapshot.get("players", []):
        if company_name in p.get("companies", []):
            return "player", p["name"], f"cash={p['cash']}"
    fi = snapshot.get("foreign_investor", {})
    if company_name in fi.get("companies", []):
        return "fi", "FI", f"cash={fi.get('cash')}"
    if company_name in snapshot.get("offering", []):
        return "offering", "offering", ""
    return "other", "deck/removed", ""


def main():
    parser = argparse.ArgumentParser(description="Track company ownership across snapshots")
    parser.add_argument("game_id", type=int)
    parser.add_argument("company", type=str, help="Company name (e.g. SJ, RENFE)")
    parser.add_argument("--range", type=str, default=None,
                        help="Action ID range START-END (e.g. 200-300)")
    parser.add_argument("--transitions-only", action="store_true",
                        help="Only show when ownership changes")
    args = parser.parse_args()

    extract_path = DATA_DIR / f"{args.game_id}_extract.json"
    if not extract_path.exists():
        print(f"Extract not found: {extract_path}", file=sys.stderr)
        sys.exit(1)

    snapshots = json.loads(extract_path.read_text())

    # Parse range
    aid_min, aid_max = 0, float("inf")
    if args.range:
        parts = args.range.split("-")
        aid_min = int(parts[0])
        aid_max = int(parts[1]) if len(parts) > 1 else float("inf")

    prev_key = None
    for s in snapshots:
        aid = s["action_id"]
        if aid < aid_min or aid > aid_max:
            continue

        owner_type, owner_name, extra = find_owner(s, args.company)
        key = (owner_type, owner_name)

        if args.transitions_only and key == prev_key:
            continue

        forced = " [FORCED]" if s.get("forced") else ""
        extra_str = f" ({extra})" if extra else ""
        print(f"aid={aid:>5}  round={s['round']:<4}  type={s['action_type']:<16}  "
              f"{args.company} -> {owner_type}:{owner_name}{extra_str}{forced}")
        prev_key = key


if __name__ == "__main__":
    main()
