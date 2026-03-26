#!/usr/bin/env python3
"""Diff two extract snapshots to see what changed between them.

Useful for understanding what happened during automated phases (WRAP_UP,
INCOME, etc.) where there are no intermediate snapshots.

Usage:
    python diff_states.py <game_id> <action_id_before> <action_id_after>
    python diff_states.py 210560 280 282
"""
import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def diff_players(before, after):
    """Diff player state between two snapshots."""
    diffs = []
    p_before = {p["id"]: p for p in before.get("players", [])}
    p_after = {p["id"]: p for p in after.get("players", [])}

    for pid, pb in p_before.items():
        pa = p_after.get(pid)
        if pa is None:
            continue
        name = pb["name"]
        if pb["cash"] != pa["cash"]:
            diffs.append(f"  player[{name}].cash: {pb['cash']} -> {pa['cash']} (delta={pa['cash']-pb['cash']})")
        if pb["value"] != pa["value"]:
            diffs.append(f"  player[{name}].value: {pb['value']} -> {pa['value']} (delta={pa['value']-pb['value']})")
        if sorted(pb.get("companies", [])) != sorted(pa.get("companies", [])):
            diffs.append(f"  player[{name}].companies: {sorted(pb.get('companies',[]))} -> {sorted(pa.get('companies',[]))}")
        if pb.get("shares", {}) != pa.get("shares", {}):
            diffs.append(f"  player[{name}].shares: {pb.get('shares',{})} -> {pa.get('shares',{})}")
    return diffs


def diff_corps(before, after):
    """Diff corporation state between two snapshots."""
    diffs = []
    c_before = {c["name"]: c for c in before.get("corporations", [])}
    c_after = {c["name"]: c for c in after.get("corporations", [])}

    for name, cb in c_before.items():
        ca = c_after.get(name)
        if ca is None:
            continue

        if cb["floated"] != ca["floated"]:
            diffs.append(f"  corp[{name}].floated: {cb['floated']} -> {ca['floated']}")
            continue

        if not cb["floated"]:
            continue

        if cb["price"] != ca["price"]:
            diffs.append(f"  corp[{name}].price: {cb['price']} -> {ca['price']} (delta={ca['price']-cb['price'] if ca['price'] and cb['price'] else '?'})")
        if cb["cash"] != ca["cash"]:
            diffs.append(f"  corp[{name}].cash: {cb['cash']} -> {ca['cash']} (delta={ca['cash']-cb['cash']})")
        if sorted(cb.get("companies", [])) != sorted(ca.get("companies", [])):
            removed = set(cb.get("companies", [])) - set(ca.get("companies", []))
            added = set(ca.get("companies", [])) - set(cb.get("companies", []))
            parts = []
            if removed:
                parts.append(f"removed={sorted(removed)}")
            if added:
                parts.append(f"added={sorted(added)}")
            diffs.append(f"  corp[{name}].companies: {', '.join(parts)}")
        if cb.get("shares_in_market", 0) != ca.get("shares_in_market", 0):
            diffs.append(f"  corp[{name}].bank_shares: {cb.get('shares_in_market',0)} -> {ca.get('shares_in_market',0)}")
        if cb.get("president") != ca.get("president"):
            diffs.append(f"  corp[{name}].president: {cb.get('president')} -> {ca.get('president')}")
    return diffs


def diff_fi(before, after):
    """Diff FI state."""
    diffs = []
    fb = before.get("foreign_investor", {})
    fa = after.get("foreign_investor", {})

    if fb.get("cash", 0) != fa.get("cash", 0):
        diffs.append(f"  fi.cash: {fb.get('cash',0)} -> {fa.get('cash',0)} (delta={fa.get('cash',0)-fb.get('cash',0)})")
    if sorted(fb.get("companies", [])) != sorted(fa.get("companies", [])):
        removed = set(fb.get("companies", [])) - set(fa.get("companies", []))
        added = set(fa.get("companies", [])) - set(fb.get("companies", []))
        parts = []
        if removed:
            parts.append(f"removed={sorted(removed)}")
        if added:
            parts.append(f"added={sorted(added)}")
        diffs.append(f"  fi.companies: {', '.join(parts)}")
    return diffs


def diff_other(before, after):
    """Diff offering, deck, CoO."""
    diffs = []
    if sorted(before.get("offering", [])) != sorted(after.get("offering", [])):
        diffs.append(f"  offering: {sorted(before.get('offering',[]))} -> {sorted(after.get('offering',[]))}")
    if before.get("deck_size") != after.get("deck_size"):
        diffs.append(f"  deck_size: {before.get('deck_size')} -> {after.get('deck_size')}")
    if before.get("cost_level") != after.get("cost_level"):
        diffs.append(f"  cost_level: {before.get('cost_level')} -> {after.get('cost_level')}")
    return diffs


def main():
    parser = argparse.ArgumentParser(description="Diff two extract snapshots")
    parser.add_argument("game_id", type=int)
    parser.add_argument("before_id", type=int, help="Action ID of 'before' snapshot")
    parser.add_argument("after_id", type=int, help="Action ID of 'after' snapshot")
    args = parser.parse_args()

    extract_path = DATA_DIR / f"{args.game_id}_extract.json"
    if not extract_path.exists():
        print(f"Extract not found: {extract_path}", file=sys.stderr)
        sys.exit(1)

    snapshots = json.loads(extract_path.read_text())
    ref_by_action = {s["action_id"]: s for s in snapshots}

    if args.before_id not in ref_by_action:
        print(f"Action {args.before_id} not found in extract", file=sys.stderr)
        sys.exit(1)
    if args.after_id not in ref_by_action:
        print(f"Action {args.after_id} not found in extract", file=sys.stderr)
        sys.exit(1)

    before = ref_by_action[args.before_id]
    after = ref_by_action[args.after_id]

    print(f"Diff: action {args.before_id} (round={before['round']}, type={before['action_type']})")
    print(f"   -> action {args.after_id} (round={after['round']}, type={after['action_type']})")
    print()

    all_diffs = (
        diff_players(before, after)
        + diff_corps(before, after)
        + diff_fi(before, after)
        + diff_other(before, after)
    )

    if all_diffs:
        for d in all_diffs:
            print(d)
    else:
        print("  (no differences)")


if __name__ == "__main__":
    main()
