#!/usr/bin/env python3
"""Show all extract snapshots for a specific round type within an action range.

Useful for seeing everything that happened during an ACQ, CLO, DIV, etc. round.

Usage:
    python show_round.py <game_id> <round_name> [--range START-END]
    python show_round.py 210560 ACQ
    python show_round.py 210560 ACQ --range 200-300
    python show_round.py 210560 CLO --range 260-290
"""
import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    parser = argparse.ArgumentParser(description="Show snapshots for a round type")
    parser.add_argument("game_id", type=int)
    parser.add_argument("round_name", type=str,
                        help="Round name: INV, ACQ, CLO, DIV, ISS, IPO, etc.")
    parser.add_argument("--range", type=str, default=None,
                        help="Action ID range START-END")
    parser.add_argument("--brief", action="store_true",
                        help="One-line-per-snapshot summary")
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

    target = args.round_name.upper()
    found = False
    for s in snapshots:
        aid = s["action_id"]
        if aid < aid_min or aid > aid_max:
            continue
        if s["round"] != target:
            continue

        found = True
        forced = " [FORCED]" if s.get("forced") else ""

        if args.brief:
            extras = []
            if s.get("acq_outcomes"):
                cos = [o["company"] for o in s["acq_outcomes"]]
                extras.append(f"acq_outcomes=[{','.join(cos)}]")
            if s.get("adjusted_income") is not None:
                extras.append(f"adj_income={s['adjusted_income']}")
            extra = "  " + "  ".join(extras) if extras else ""
            print(f"aid={aid:>5}  type={s['action_type']:<16}{forced}{extra}")
        else:
            print(f"=== aid={aid}  round={s['round']}  type={s['action_type']}{forced} ===")

            # Show corps
            for c in s.get("corporations", []):
                if c["floated"]:
                    cos = ", ".join(c.get("companies", []))
                    print(f"  {c['name']}: price={c['price']} cash={c['cash']} "
                          f"president={c.get('president', 'Market')} [{cos}]")

            # FI
            fi = s.get("foreign_investor", {})
            fi_cos = ", ".join(fi.get("companies", []))
            print(f"  FI: cash={fi.get('cash', 0)} [{fi_cos}]")

            # ACQ outcomes
            if s.get("acq_outcomes"):
                for o in s["acq_outcomes"]:
                    cross = " CROSS" if o.get("cross_president") else ""
                    print(f"  -> {o['company']}: {o.get('seller_type','?')}:{o.get('seller','?')} "
                          f"-> {o['buyer']} @ {o['price']}{cross}")

            if s.get("adjusted_income") is not None:
                print(f"  adjusted_income={s['adjusted_income']}")
            print()

    if not found:
        print(f"No {target} snapshots found in game {args.game_id}"
              + (f" (range {args.range})" if args.range else ""))


if __name__ == "__main__":
    main()
