#!/usr/bin/env python3
"""Dump detailed state at one or more action IDs from a game extract.

Usage:
    python dump_state.py <game_id> <action_ids...>
    python dump_state.py 210560 282 291
    python dump_state.py 210560 --range 280-290
"""
import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def dump_snapshot(s):
    """Print a detailed snapshot."""
    aid = s["action_id"]
    forced = " [FORCED]" if s.get("forced") else ""
    print(f"=== Action {aid}  round={s['round']}  type={s['action_type']}{forced} ===")

    if s.get("active_player") is not None:
        print(f"  Active player: id={s['active_player']}")
    if s.get("active_corp") is not None:
        print(f"  Active corp: {s['active_corp']}")

    # Players
    for p in s.get("players", []):
        shares_str = ", ".join(f"{k}:{v}" for k, v in sorted(p.get("shares", {}).items()))
        cos_str = ", ".join(p.get("companies", []))
        print(f"  Player {p['name']} (id={p['id']}): "
              f"cash={p['cash']} value={p['value']} "
              f"shares=[{shares_str}] companies=[{cos_str}]")

    # Corporations
    for c in s.get("corporations", []):
        if not c["floated"]:
            continue
        cos_str = ", ".join(c.get("companies", []))
        pres = c.get("president", "Market")
        print(f"  Corp {c['name']}: price={c['price']} cash={c['cash']} "
              f"bank_shares={c.get('shares_in_market', 0)} president={pres} "
              f"companies=[{cos_str}]")

    # Non-floated corps (briefly)
    non_floated = [c["name"] for c in s.get("corporations", []) if not c["floated"]]
    if non_floated:
        print(f"  Non-floated: {', '.join(non_floated)}")

    # FI
    fi = s.get("foreign_investor", {})
    fi_cos = ", ".join(fi.get("companies", []))
    print(f"  FI: cash={fi.get('cash', 0)} companies=[{fi_cos}]")

    # Offering / Deck
    offering = ", ".join(s.get("offering", []))
    print(f"  Offering: [{offering}]")
    print(f"  Deck: {s.get('deck_size', '?')} cards, CoO level={s.get('cost_level', '?')}")

    # ACQ outcomes
    if s.get("acq_outcomes"):
        print(f"  ACQ outcomes:")
        for o in s["acq_outcomes"]:
            cross = " [CROSS]" if o.get("cross_president") else ""
            print(f"    {o['company']}: {o.get('seller_type','?')}:{o.get('seller','?')} "
                  f"-> {o['buyer']} @ {o['price']}{cross}")

    # Adjusted income (CLO)
    if s.get("adjusted_income") is not None:
        print(f"  adjusted_income={s['adjusted_income']}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Dump state at specific action IDs")
    parser.add_argument("game_id", type=int)
    parser.add_argument("action_ids", nargs="*", type=int,
                        help="Action IDs to dump")
    parser.add_argument("--range", type=str, default=None,
                        help="Action ID range START-END")
    args = parser.parse_args()

    extract_path = DATA_DIR / f"{args.game_id}_extract.json"
    if not extract_path.exists():
        print(f"Extract not found: {extract_path}", file=sys.stderr)
        sys.exit(1)

    snapshots = json.loads(extract_path.read_text())

    # Collect target action IDs
    targets = set(args.action_ids or [])
    if args.range:
        parts = args.range.split("-")
        lo = int(parts[0])
        hi = int(parts[1]) if len(parts) > 1 else lo
        for s in snapshots:
            if lo <= s["action_id"] <= hi:
                targets.add(s["action_id"])

    if not targets:
        parser.error("Specify action IDs or --range")

    for s in snapshots:
        if s["action_id"] in targets:
            dump_snapshot(s)


if __name__ == "__main__":
    main()
