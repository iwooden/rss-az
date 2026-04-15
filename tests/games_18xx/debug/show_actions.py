#!/usr/bin/env python3
"""Show raw 18xx actions from a game JSON, with committed/undone status.

Usage:
    python show_actions.py <game_id> --range START-END
    python show_actions.py 210560 --range 278-290
    python show_actions.py 210560 --range 278-290 --committed-only
"""
import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def format_action(a, committed_ids):
    """Format a single action for display."""
    aid = a.get("id", -1)
    atype = a.get("type", "?")
    entity = a.get("entity", "")
    etype = a.get("entity_type", "")
    committed = aid in committed_ids if committed_ids is not None else None

    parts = [f"aid={aid:<5}  type={atype:<22}  entity={etype}:{entity}"]

    # Action-specific fields
    if atype == "bid":
        parts.append(f"  company={a.get('company')} price={a.get('price')}")
    elif atype == "buy_shares":
        parts.append(f"  shares={a.get('shares')}")
    elif atype == "sell_shares":
        parts.append(f"  shares={a.get('shares')}")
    elif atype == "par":
        parts.append(f"  corp={a.get('corporation')} price={a.get('share_price')} "
                      f"company={a.get('entity')}")
    elif atype == "offer":
        parts.append(f"  company={a.get('company')} corp={a.get('corporation')} "
                      f"price={a.get('price')}")
    elif atype == "respond":
        parts.append(f"  corp={a.get('corporation')} accept={a.get('accept')}")
    elif atype == "sell_company":
        parts.append(f"  company={a.get('company')}")
    elif atype == "dividend":
        parts.append(f"  amount={a.get('amount')}")
    elif atype == "undo":
        parts.append(f"  action_id={a.get('action_id')}")

    if committed is not None:
        status = "committed" if committed else "UNDONE"
        parts.append(f"  [{status}]")

    # Auto-actions summary
    autos = a.get("auto_actions", [])
    if autos:
        parts.append(f"  auto_actions={len(autos)}")

    return "".join(parts)


def format_auto_actions(a):
    """Format auto_actions detail lines (for --expand-autos)."""
    lines = []
    for aa in a.get("auto_actions", []):
        aa_type = aa.get("type", "?")
        aa_entity = aa.get("entity", "")
        aa_etype = aa.get("entity_type", "")
        lines.append(f"       -> auto: type={aa_type}  entity={aa_etype}:{aa_entity}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Show raw game actions")
    parser.add_argument("game_id", type=int)
    parser.add_argument("--range", type=str, required=True,
                        help="Action ID range START-END")
    parser.add_argument("--committed-only", action="store_true",
                        help="Only show committed actions")
    parser.add_argument("--expand-autos", action="store_true",
                        help="Show auto_action details on separate lines")
    args = parser.parse_args()

    game_path = DATA_DIR / f"{args.game_id}.json"
    extract_path = DATA_DIR / f"{args.game_id}_extract.json"
    if not game_path.exists():
        print(f"Game JSON not found: {game_path}", file=sys.stderr)
        sys.exit(1)

    game_data = json.loads(game_path.read_text())

    # Load committed IDs from extract if available
    committed_ids = None
    if extract_path.exists():
        extract_data = json.loads(extract_path.read_text())
        committed_ids = set(extract_data[0].get("committed_action_ids", []))

    # Parse range
    parts = args.range.split("-")
    aid_min = int(parts[0])
    aid_max = int(parts[1]) if len(parts) > 1 else aid_min

    for a in game_data.get("actions", []):
        aid = a.get("id", -1)
        if aid < aid_min or aid > aid_max:
            continue
        if args.committed_only and committed_ids and aid not in committed_ids:
            continue
        print(format_action(a, committed_ids))
        if args.expand_autos:
            for line in format_auto_actions(a):
                print(line)


if __name__ == "__main__":
    main()
