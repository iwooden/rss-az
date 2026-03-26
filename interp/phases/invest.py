"""Invest phase analysis: auction pricing, share trades, per-turn activity.

Reconstructs auction narratives from sequential INVEST/BID states and analyzes
share trading patterns.

Analyses:
1. **Auction pricing**: Per-company opening bid offset, bid rounds, final price
2. **Share trades**: Buys/sells per-corp, president vs non-president breakdown
3. **Per-turn activity**: Share trades per corp over game turns

Usage:
    .venv/bin/python -m interp.phases.invest --load-data interp/data/states.npz
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.actions import decode_action_py
from core.data import (
    COMPANY_NAMES,
    CORP_NAMES,
    PY_COMPANY_PRICE_DIVISOR,
    PY_COMPANY_STAR_DIVISOR,
    get_company_face_value,
)
from core.state import get_layout, get_player_fields, get_turn_fields
from interp.html import BAR_CSS, HIST_BAR_CSS, JS_MAKE_BAR, STAT_BOX_CSS, html_page, open_file
from interp.utils import (
    InterpDataset,
    collect_states,
    forward_batched,
    load_model,
)

_TIER_NAMES = {1: "Red(1)", 2: "Org(2)", 3: "Yel(3)", 4: "Grn(4)", 5: "Blu(5)"}
_COMPANY_STARS = [1] * 6 + [2] * 8 + [3] * 8 + [4] * 7 + [5] * 7
SLOT_STRIDE = 5  # stars, low, face, high, income per auction slot


# ---------------------------------------------------------------------------
# Auction reconstruction
# ---------------------------------------------------------------------------


@dataclass
class AuctionRecord:
    """One completed auction (from INVEST open through BID rounds to resolution)."""

    company_id: int
    company_stars: int
    face_value: int
    slot: int
    open_offset: int  # bid_offset over face value at auction open
    open_price: int  # face_value + open_offset
    bid_rounds: int  # number of BID phase states that followed
    final_price: int  # auction_price at last BID state (or open_price if 0 rounds)
    price_over_face: int  # final_price - face_value
    game_idx: int
    game_third: int  # 0=early, 1=mid, 2=late


@dataclass
class ShareTrade:
    """One buy or sell action."""

    corp_id: int
    is_buy: bool  # True=buy, False=sell
    is_president: bool  # was the active player president of this corp?
    game_idx: int
    game_third: int


def reconstruct_invest_activity(
    dataset: InterpDataset,
    model: Any,
    device: torch.device,
    num_players: int,
    batch_size: int = 256,
) -> tuple[list[AuctionRecord], list[ShareTrade], dict[str, Any], np.ndarray]:
    """Process sequential states to reconstruct auctions and share trades.

    Returns (auctions, trades, invest_stats, logits).
    """
    layout = get_layout(num_players)
    tf = get_turn_fields(num_players)
    pf = get_player_fields(num_players)
    t = layout.turn_offset

    # Forward pass to get argmax actions
    logits, _ = forward_batched(model, device, dataset.states, batch_size)

    phases = dataset.phases
    n = len(phases)

    # Compute game boundaries and thirds
    game_starts: list[int] = [0]
    for i in range(1, n):
        if phases[i] == 0 and phases[i - 1] != 0:
            game_starts.append(i)
    game_starts.append(n)

    game_idx_arr = np.zeros(n, dtype=int)
    game_third_arr = np.zeros(n, dtype=int)
    for gi in range(len(game_starts) - 1):
        start, end = game_starts[gi], game_starts[gi + 1]
        game_idx_arr[start:end] = gi
        game_len = end - start
        if game_len >= 3:
            t1 = start + game_len // 3
            t2 = start + 2 * game_len // 3
            game_third_arr[start:t1] = 0
            game_third_arr[t1:t2] = 1
            game_third_arr[t2:end] = 2

    auctions: list[AuctionRecord] = []
    trades: list[ShareTrade] = []

    # Action stats
    invest_action_counts: dict[str, int] = defaultdict(int)

    # Track pending auction
    pending_auction: dict[str, Any] | None = None

    for i in range(n):
        state = dataset.states[i]
        mask = dataset.legal_masks[i]
        phase = int(phases[i])

        # Get argmax action
        masked_logits = logits[i].copy()
        masked_logits[mask <= 0] = -1e9
        argmax = int(np.argmax(masked_logits))
        _, atype, slot, corp_id, amount = decode_action_py(argmax, num_players)

        gi = int(game_idx_arr[i])
        gt = int(game_third_arr[i])

        if phase == 0:  # INVEST
            if atype == 0:  # pass
                invest_action_counts["pass"] += 1
            elif atype == 1:  # auction
                invest_action_counts["auction"] += 1
                # Resolve any pending auction first
                if pending_auction is not None:
                    auctions.append(AuctionRecord(**pending_auction))
                    pending_auction = None

                # Determine company from slot
                slot_off = layout.auction_slot_info_offset + slot * SLOT_STRIDE
                face_norm = state[slot_off + 2]  # face value
                stars_norm = state[slot_off + 0]  # stars
                face_val = int(face_norm * PY_COMPANY_PRICE_DIVISOR + 0.5)
                stars_val = max(1, int(stars_norm * PY_COMPANY_STAR_DIVISOR + 0.5))

                # Find company_id from face value
                co_id = -1
                for c in range(36):
                    if get_company_face_value(c) == face_val:
                        # Verify stars match
                        if _COMPANY_STARS[c] == stars_val:
                            co_id = c
                            break
                if co_id == -1:
                    # Fallback: just match face value
                    for c in range(36):
                        if get_company_face_value(c) == face_val:
                            co_id = c
                            break

                open_price = face_val + amount
                pending_auction = {
                    "company_id": co_id,
                    "company_stars": stars_val,
                    "face_value": face_val,
                    "slot": slot,
                    "open_offset": amount,
                    "open_price": open_price,
                    "bid_rounds": 0,
                    "final_price": open_price,
                    "price_over_face": amount,
                    "game_idx": gi,
                    "game_third": gt,
                }

            elif atype == 2:  # buy_share
                invest_action_counts["buy"] += 1
                # Check if active player is president
                is_pres = bool(
                    state[layout.players_offset + pf.is_president + corp_id] > 0.5
                )
                trades.append(ShareTrade(
                    corp_id=corp_id, is_buy=True, is_president=is_pres,
                    game_idx=gi, game_third=gt,
                ))

            elif atype == 3:  # sell_share
                invest_action_counts["sell"] += 1
                is_pres = bool(
                    state[layout.players_offset + pf.is_president + corp_id] > 0.5
                )
                trades.append(ShareTrade(
                    corp_id=corp_id, is_buy=False, is_president=is_pres,
                    game_idx=gi, game_third=gt,
                ))

        elif phase == 1:  # BID
            if pending_auction is not None:
                pending_auction["bid_rounds"] += 1
                # Update final price from state's auction_price
                price_norm = state[t + tf.auction_price]
                price = int(price_norm * PY_COMPANY_PRICE_DIVISOR + 0.5)
                if price > 0:
                    pending_auction["final_price"] = price
                    pending_auction["price_over_face"] = price - pending_auction["face_value"]

        else:
            # Non-INVEST/BID phase: finalize any pending auction
            if pending_auction is not None:
                auctions.append(AuctionRecord(**pending_auction))
                pending_auction = None

    # Finalize last pending auction
    if pending_auction is not None:
        auctions.append(AuctionRecord(**pending_auction))

    invest_stats = {
        "total_invest_states": int(np.sum(phases == 0)),
        "total_bid_states": int(np.sum(phases == 1)),
        "action_counts": dict(invest_action_counts),
    }

    return auctions, trades, invest_stats, logits


# ---------------------------------------------------------------------------
# Per-turn activity (uses turn_numbers from dataset)
# ---------------------------------------------------------------------------


def compute_per_turn_activity(
    dataset: InterpDataset,
    logits: np.ndarray,
    num_players: int,
) -> list[dict[str, Any]]:
    """Compute per-game-turn share buy/sell activity from INVEST states.

    Uses ``dataset.turn_numbers`` (recorded from the game engine during
    state collection) to group INVEST states by turn.  For each turn,
    decodes the argmax action and tallies buys/sells per corp.
    """
    phases = dataset.phases
    turn_numbers = dataset.turn_numbers
    n = len(phases)

    # Aggregate buys/sells per turn per corp
    turn_buys: dict[int, list[int]] = {}
    turn_sells: dict[int, list[int]] = {}

    for i in range(n):
        if int(phases[i]) != 0:  # only INVEST states
            continue
        turn = int(turn_numbers[i])
        if turn <= 0:
            continue

        mask = dataset.legal_masks[i]
        masked = logits[i].copy()
        masked[mask <= 0] = -1e9
        argmax = int(np.argmax(masked))
        _, atype, _, corp_id, _ = decode_action_py(argmax, num_players)

        if atype == 2:  # buy_share
            if turn not in turn_buys:
                turn_buys[turn] = [0] * 8
            turn_buys[turn][corp_id] += 1
        elif atype == 3:  # sell_share
            if turn not in turn_sells:
                turn_sells[turn] = [0] * 8
            turn_sells[turn][corp_id] += 1

    max_turn = max(max(turn_buys, default=0), max(turn_sells, default=0))
    turn_activity: list[dict[str, Any]] = []
    for turn in range(1, min(max_turn + 1, 30)):
        cb = turn_buys.get(turn, [0] * 8)
        cs = turn_sells.get(turn, [0] * 8)
        total_b = sum(cb)
        total_s = sum(cs)
        if total_b + total_s == 0:
            continue
        turn_activity.append({
            "turn": turn,
            "buys": total_b,
            "sells": total_s,
            "net": total_b - total_s,
            "corp_buys": cb,
            "corp_sells": cs,
        })

    return turn_activity


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass
class InvestAnalysis:
    """Aggregated invest phase analysis results."""

    invest_stats: dict[str, Any]

    # Auction analysis
    total_auctions: int
    auctions_by_tier: dict[int, dict[str, Any]]
    auctions_by_company: list[dict[str, Any]]
    auctions_by_stage: dict[str, dict[str, Any]]
    open_offset_histogram: list[int]  # count per offset 0-14

    # Share trade analysis
    total_buys: int
    total_sells: int
    trades_by_corp: list[dict[str, Any]]
    trades_by_stage: dict[str, dict[str, Any]]

    # Per-turn share activity
    turn_activity: list[dict[str, Any]]


def run_analysis(
    auctions: list[AuctionRecord],
    trades: list[ShareTrade],
    invest_stats: dict[str, Any],
    turn_activity: list[dict[str, Any]],
) -> InvestAnalysis:
    # --- Auction analysis ---
    total_auctions = len(auctions)

    # Open offset histogram
    open_hist = [0] * 15
    for a in auctions:
        if 0 <= a.open_offset < 15:
            open_hist[a.open_offset] += 1

    # By tier
    auctions_by_tier: dict[int, dict[str, Any]] = {}
    for stars in sorted(set(a.company_stars for a in auctions)):
        tier = [a for a in auctions if a.company_stars == stars]
        auctions_by_tier[stars] = {
            "count": len(tier),
            "mean_open_offset": float(np.mean([a.open_offset for a in tier])),
            "mean_bid_rounds": float(np.mean([a.bid_rounds for a in tier])),
            "mean_final_over_face": float(np.mean([a.price_over_face for a in tier])),
            "median_final_over_face": float(np.median([a.price_over_face for a in tier])),
            "pct_at_face": sum(1 for a in tier if a.price_over_face == 0) / len(tier),
            "mean_face_value": float(np.mean([a.face_value for a in tier])),
        }

    # By company (detailed)
    company_stats: dict[int, list[AuctionRecord]] = defaultdict(list)
    for a in auctions:
        company_stats[a.company_id].append(a)

    auctions_by_company: list[dict[str, Any]] = []
    for co_id in sorted(company_stats.keys()):
        recs = company_stats[co_id]
        fv = recs[0].face_value
        stars = recs[0].company_stars
        auctions_by_company.append({
            "company_id": co_id,
            "face_value": fv,
            "stars": stars,
            "tier": _TIER_NAMES.get(stars, str(stars)),
            "count": len(recs),
            "mean_open_offset": float(np.mean([r.open_offset for r in recs])),
            "mean_bid_rounds": float(np.mean([r.bid_rounds for r in recs])),
            "mean_final_price": float(np.mean([r.final_price for r in recs])),
            "mean_price_over_face": float(np.mean([r.price_over_face for r in recs])),
            "pct_at_face": sum(1 for r in recs if r.price_over_face == 0) / len(recs),
        })

    # By game stage
    auctions_by_stage: dict[str, dict[str, Any]] = {}
    stage_names = {0: "early", 1: "mid", 2: "late"}
    for stage, name in stage_names.items():
        sg = [a for a in auctions if a.game_third == stage]
        if not sg:
            continue
        auctions_by_stage[name] = {
            "count": len(sg),
            "mean_open_offset": float(np.mean([a.open_offset for a in sg])),
            "mean_bid_rounds": float(np.mean([a.bid_rounds for a in sg])),
            "mean_final_over_face": float(np.mean([a.price_over_face for a in sg])),
        }

    # --- Share trade analysis ---
    buys = [t for t in trades if t.is_buy]
    sells = [t for t in trades if not t.is_buy]

    trades_by_corp: list[dict[str, Any]] = []
    for corp_id in range(8):
        cb = [t for t in buys if t.corp_id == corp_id]
        cs = [t for t in sells if t.corp_id == corp_id]
        cb_pres = sum(1 for t in cb if t.is_president)
        cs_pres = sum(1 for t in cs if t.is_president)
        trades_by_corp.append({
            "corp_id": corp_id,
            "name": CORP_NAMES[corp_id],
            "buys": len(cb),
            "sells": len(cs),
            "net": len(cb) - len(cs),
            "buys_as_pres": cb_pres,
            "buys_as_non_pres": len(cb) - cb_pres,
            "sells_as_pres": cs_pres,
            "sells_as_non_pres": len(cs) - cs_pres,
        })

    trades_by_stage: dict[str, dict[str, Any]] = {}
    for stage, name in stage_names.items():
        sb = [t for t in buys if t.game_third == stage]
        ss = [t for t in sells if t.game_third == stage]
        if not sb and not ss:
            continue
        trades_by_stage[name] = {
            "buys": len(sb),
            "sells": len(ss),
            "net": len(sb) - len(ss),
        }

    return InvestAnalysis(
        invest_stats=invest_stats,
        total_auctions=total_auctions,
        auctions_by_tier=auctions_by_tier,
        auctions_by_company=auctions_by_company,
        auctions_by_stage=auctions_by_stage,
        open_offset_histogram=open_hist,
        total_buys=len(buys),
        total_sells=len(sells),
        trades_by_corp=trades_by_corp,
        trades_by_stage=trades_by_stage,
        turn_activity=turn_activity,
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_report(a: InvestAnalysis) -> None:
    print("\n" + "=" * 90)
    print(f"  INVEST PHASE ANALYSIS")
    print("=" * 90)

    s = a.invest_stats
    print(f"\n  States: {s['total_invest_states']} INVEST + {s['total_bid_states']} BID")
    ac = s["action_counts"]
    total_actions = sum(ac.values())
    print(f"  Actions: " + ", ".join(f"{k}={v} ({v/total_actions:.1%})" for k, v in sorted(ac.items())))

    # --- Auctions ---
    print(f"\n  1. AUCTIONS ({a.total_auctions} total)")

    if a.total_auctions > 0:
        print(f"\n  Opening bid offset histogram:")
        max_c = max(a.open_offset_histogram)
        for off, c in enumerate(a.open_offset_histogram):
            if c > 0:
                bar = "#" * (c * 40 // max(max_c, 1))
                print(f"    face+{off:>2d}: {c:>4d}  {bar}")

        print(f"\n  By company tier:")
        print(f"  {'Tier':<8s} {'Count':>6s} {'AvgOpen':>8s} {'AvgRnds':>8s} {'AvgOver':>8s} {'MedOver':>8s} {'%@Face':>7s}")
        print(f"  {'-' * 8} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 7}")
        for stars in sorted(a.auctions_by_tier.keys()):
            ts = a.auctions_by_tier[stars]
            name = _TIER_NAMES.get(stars, str(stars))
            print(
                f"  {name:<8s} {ts['count']:>6d} {ts['mean_open_offset']:>8.1f} "
                f"{ts['mean_bid_rounds']:>8.1f} {ts['mean_final_over_face']:>8.1f} "
                f"{ts['median_final_over_face']:>8.1f} {ts['pct_at_face']:>6.1%}"
            )

        print(f"\n  Per-company auction detail:")
        print(f"  {'Company':<12s} {'Face':>5s} {'Tier':<6s} {'#Auc':>5s} {'AvgOpen':>8s} {'Rnds':>5s} {'AvgFinal':>9s} {'Over':>5s} {'%@Face':>7s}")
        print(f"  {'-' * 12} {'-' * 5} {'-' * 6} {'-' * 5} {'-' * 8} {'-' * 5} {'-' * 9} {'-' * 5} {'-' * 7}")
        for co in a.auctions_by_company:
            cid = co['company_id']
            co_label = f"{COMPANY_NAMES[cid]} ({cid})"
            print(
                f"  {co_label:<12s} {co['face_value']:>5d} {co['tier']:<6s} "
                f"{co['count']:>5d} {co['mean_open_offset']:>8.1f} {co['mean_bid_rounds']:>5.1f} "
                f"{co['mean_final_price']:>9.1f} {co['mean_price_over_face']:>+5.1f} "
                f"{co['pct_at_face']:>6.1%}"
            )

        print(f"\n  By game stage:")
        for name in ["early", "mid", "late"]:
            if name not in a.auctions_by_stage:
                continue
            ss = a.auctions_by_stage[name]
            print(
                f"    {name:<6s}: n={ss['count']}, open_offset={ss['mean_open_offset']:.1f}, "
                f"rounds={ss['mean_bid_rounds']:.1f}, over_face={ss['mean_final_over_face']:.1f}"
            )

    # --- Share trades ---
    print(f"\n  2. SHARE TRADES ({a.total_buys} buys, {a.total_sells} sells, net={a.total_buys - a.total_sells:+d})")

    print(f"\n  Per-corp breakdown:")
    print(f"  {'Corp':<4s} {'Buys':>5s} {'Sells':>6s} {'Net':>5s} {'BuyPres':>8s} {'BuyNon':>7s} {'SellPres':>9s} {'SellNon':>8s}")
    print(f"  {'-' * 4} {'-' * 5} {'-' * 6} {'-' * 5} {'-' * 8} {'-' * 7} {'-' * 9} {'-' * 8}")
    for tc in a.trades_by_corp:
        if tc["buys"] + tc["sells"] == 0:
            continue
        print(
            f"  {tc['name']:<4s} {tc['buys']:>5d} {tc['sells']:>6d} {tc['net']:>+5d} "
            f"{tc['buys_as_pres']:>8d} {tc['buys_as_non_pres']:>7d} "
            f"{tc['sells_as_pres']:>9d} {tc['sells_as_non_pres']:>8d}"
        )

    print(f"\n  By game stage:")
    for name in ["early", "mid", "late"]:
        if name not in a.trades_by_stage:
            continue
        ss = a.trades_by_stage[name]
        print(f"    {name:<6s}: buys={ss['buys']}, sells={ss['sells']}, net={ss['net']:+d}")

    # --- Per-turn ---
    if a.turn_activity:
        print(f"\n  3. PER-INVEST-TURN ACTIVITY")
        print(f"  {'Turn':>5s} {'Buys':>5s} {'Sells':>6s} {'Net':>5s}  Active corps")
        print(f"  {'-' * 5} {'-' * 5} {'-' * 6} {'-' * 5}  {'-' * 30}")
        for ta in a.turn_activity[:20]:
            active = []
            for c in range(8):
                b, s = ta["corp_buys"][c], ta["corp_sells"][c]
                if b > 0 or s > 0:
                    active.append(f"{CORP_NAMES[c]}:{b:+d}/{-s:+d}" if s > 0 else f"{CORP_NAMES[c]}:+{b}")
            print(
                f"  {ta['turn']:>5d} {ta['buys']:>5d} {ta['sells']:>6d} {ta['net']:>+5d}  "
                + ", ".join(active)
            )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def format_html_report(
    a: InvestAnalysis,
    epoch: int,
    num_states: int,
    num_games: int,
) -> str:
    s = a.invest_stats

    stats_json = json.dumps({
        "invest_states": s["total_invest_states"],
        "bid_states": s["total_bid_states"],
        "action_counts": s["action_counts"],
        "total_auctions": a.total_auctions,
        "total_buys": a.total_buys,
        "total_sells": a.total_sells,
    })

    open_hist_json = json.dumps(a.open_offset_histogram)

    tier_json = json.dumps([
        {"tier": _TIER_NAMES.get(s, str(s)), "stars": s, **v}
        for s, v in sorted(a.auctions_by_tier.items())
    ])

    company_json = json.dumps(a.auctions_by_company)

    stage_auc_json = json.dumps([
        {"stage": name, **a.auctions_by_stage[name]}
        for name in ["early", "mid", "late"] if name in a.auctions_by_stage
    ])

    corp_json = json.dumps(a.trades_by_corp)

    stage_trade_json = json.dumps([
        {"stage": name, **a.trades_by_stage[name]}
        for name in ["early", "mid", "late"] if name in a.trades_by_stage
    ])

    turn_json = json.dumps(a.turn_activity[:20])
    corp_names_json = json.dumps(CORP_NAMES)
    company_names_json = json.dumps(list(COMPANY_NAMES))

    body = (
        '<div id="overview-stats"></div>\n'
        "\n"
        "<h2>1. Auction Pricing</h2>\n"
        "<h3>Opening Bid Offset Histogram</h3>\n"
        '<div id="open-hist"></div>\n'
        "<h3>By Company Tier</h3>\n"
        '<table id="tbl-tier"></table>\n'
        "<h3>Per-Company Detail</h3>\n"
        '<table id="tbl-company"></table>\n'
        "<h3>By Game Stage</h3>\n"
        '<table id="tbl-auc-stage"></table>\n'
        "\n"
        "<h2>2. Share Trades</h2>\n"
        "<h3>Per-Corp Breakdown</h3>\n"
        '<table id="tbl-corp"></table>\n'
        "<h3>By Game Stage</h3>\n"
        '<table id="tbl-trade-stage"></table>\n'
        "\n"
        "<h2>3. Per-Turn Activity</h2>\n"
        '<p style="color:#888;font-size:0.85rem">Share buys/sells per invest round, aggregated across all games.</p>\n'
        '<table id="tbl-turns"></table>'
    )

    data_js = (
        f"const stats = {stats_json};\n"
        f"const openHist = {open_hist_json};\n"
        f"const tiers = {tier_json};\n"
        f"const companies = {company_json};\n"
        f"const stageAuc = {stage_auc_json};\n"
        f"const corps = {corp_json};\n"
        f"const stageTrade = {stage_trade_json};\n"
        f"const turns = {turn_json};\n"
        f"const corpNames = {corp_names_json};\n"
        f"const companyNames = {company_names_json};"
    )

    report_js = (
        '// --- Overview ---\n'
        '(function() {\n'
        '  const div = document.getElementById("overview-stats");\n'
        '  const ac = stats.action_counts;\n'
        '  const total = Object.values(ac).reduce((a,b) => a+b, 0);\n'
        '  div.innerHTML =\n'
        '    \'<div class="stat-box"><div class="stat-label">INVEST States</div><div class="stat-value">\' + stats.invest_states + \'</div></div>\' +\n'
        '    \'<div class="stat-box"><div class="stat-label">BID States</div><div class="stat-value">\' + stats.bid_states + \'</div></div>\' +\n'
        '    \'<div class="stat-box"><div class="stat-label">Auctions</div><div class="stat-value">\' + stats.total_auctions + \'</div></div>\' +\n'
        '    \'<div class="stat-box"><div class="stat-label">Share Buys</div><div class="stat-value">\' + stats.total_buys + \'</div></div>\' +\n'
        '    \'<div class="stat-box"><div class="stat-label">Share Sells</div><div class="stat-value">\' + stats.total_sells + \'</div></div>\';\n'
        '})();\n'
        '\n'
        '// --- Open histogram ---\n'
        '(function() {\n'
        '  const div = document.getElementById("open-hist");\n'
        '  const maxC = Math.max(...openHist);\n'
        '  if (maxC === 0) { div.innerHTML = \'<p style="color:#888">No auctions</p>\'; return; }\n'
        '  let html = \'<table style="width:auto"><tr><th>face+N</th><th>Count</th><th style="min-width:300px"></th></tr>\';\n'
        '  for (let i = 0; i < openHist.length; i++) {\n'
        '    if (openHist[i] > 0) {\n'
        '      const w = openHist[i] / maxC * 280;\n'
        '      html += \'<tr><td>face+\' + i + \'</td><td>\' + openHist[i] + \'</td>\' +\n'
        '        \'<td style="text-align:left"><span class="hist-bar" style="width:\' + w + \'px"></span></td></tr>\';\n'
        '    }\n'
        '  }\n'
        '  html += \'</table>\';\n'
        '  div.innerHTML = html;\n'
        '})();\n'
        '\n'
        '// --- Tier table ---\n'
        '(function() {\n'
        '  const tbl = document.getElementById("tbl-tier");\n'
        '  let html = \'<tr><th>Tier</th><th>Count</th><th>Avg Open</th><th>Avg Rounds</th><th>Avg Over Face</th><th>Med Over Face</th><th>% @ Face</th></tr>\';\n'
        '  for (const t of tiers) {\n'
        '    html += \'<tr><td>\' + t.tier + \'</td>\' +\n'
        '      \'<td>\' + t.count + \'</td>\' +\n'
        '      \'<td>\' + t.mean_open_offset.toFixed(1) + \'</td>\' +\n'
        '      \'<td>\' + t.mean_bid_rounds.toFixed(1) + \'</td>\' +\n'
        '      \'<td>\' + t.mean_final_over_face.toFixed(1) + \'</td>\' +\n'
        '      \'<td>\' + t.median_final_over_face.toFixed(1) + \'</td>\' +\n'
        '      \'<td>\' + (t.pct_at_face * 100).toFixed(1) + \'%</td></tr>\';\n'
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();\n'
        '\n'
        '// --- Company table ---\n'
        '(function() {\n'
        '  const tbl = document.getElementById("tbl-company");\n'
        '  let html = \'<tr><th>Company</th><th>Face</th><th>Tier</th><th>#Auc</th><th>Avg Open</th><th>Rounds</th><th>Avg Final</th><th>Over Face</th><th>% @ Face</th></tr>\';\n'
        '  for (const co of companies) {\n'
        '    const coName = companyNames[co.company_id] || co.company_id;\n'
        '    html += \'<tr><td>\' + coName + \' (\' + co.company_id + \')</td>\' +\n'
        '      \'<td>\' + co.face_value + \'</td>\' +\n'
        '      \'<td>\' + co.tier + \'</td>\' +\n'
        '      \'<td>\' + co.count + \'</td>\' +\n'
        '      \'<td>\' + co.mean_open_offset.toFixed(1) + \'</td>\' +\n'
        '      \'<td>\' + co.mean_bid_rounds.toFixed(1) + \'</td>\' +\n'
        '      \'<td>\' + co.mean_final_price.toFixed(1) + \'</td>\' +\n'
        "      '<td>' + (co.mean_price_over_face >= 0 ? '+' : '') + co.mean_price_over_face.toFixed(1) + '</td>' +\n"
        '      \'<td>\' + (co.pct_at_face * 100).toFixed(1) + \'%</td></tr>\';\n'
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();\n'
        '\n'
        '// --- Auction stage table ---\n'
        '(function() {\n'
        '  const tbl = document.getElementById("tbl-auc-stage");\n'
        '  let html = \'<tr><th>Stage</th><th>Count</th><th>Avg Open Offset</th><th>Avg Rounds</th><th>Avg Over Face</th></tr>\';\n'
        '  for (const s of stageAuc) {\n'
        '    html += \'<tr><td>\' + s.stage + \'</td>\' +\n'
        '      \'<td>\' + s.count + \'</td>\' +\n'
        '      \'<td>\' + s.mean_open_offset.toFixed(1) + \'</td>\' +\n'
        '      \'<td>\' + s.mean_bid_rounds.toFixed(1) + \'</td>\' +\n'
        '      \'<td>\' + s.mean_final_over_face.toFixed(1) + \'</td></tr>\';\n'
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();\n'
        '\n'
        '// --- Corp trade table ---\n'
        '(function() {\n'
        '  const tbl = document.getElementById("tbl-corp");\n'
        '  const maxTrade = Math.max(...corps.map(c => Math.max(c.buys, c.sells)));\n'
        "  let html = '<tr><th>Corp</th><th>Buys</th><th></th><th>Sells</th><th></th><th>Net</th><th>Buy(Pres)</th><th>Buy(Non)</th><th>Sell(Pres)</th><th>Sell(Non)</th></tr>';\n"
        '  for (const c of corps) {\n'
        '    if (c.buys + c.sells === 0) continue;\n'
        "    const netCls = c.net > 0 ? 'positive' : c.net < 0 ? 'negative' : '';\n"
        "    html += '<tr><td>' + c.name + '</td>' +\n"
        "      '<td>' + c.buys + '</td><td>' + makeBar(c.buys, maxTrade, 'bar-green') + '</td>' +\n"
        "      '<td>' + c.sells + '</td><td>' + makeBar(c.sells, maxTrade, 'bar-red') + '</td>' +\n"
        '      \'<td class="\' + netCls + \'">\' + (c.net > 0 ? \'+\' : \'\') + c.net + \'</td>\' +\n'
        "      '<td>' + c.buys_as_pres + '</td><td>' + c.buys_as_non_pres + '</td>' +\n"
        "      '<td>' + c.sells_as_pres + '</td><td>' + c.sells_as_non_pres + '</td></tr>';\n"
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();\n'
        '\n'
        '// --- Trade stage table ---\n'
        '(function() {\n'
        '  const tbl = document.getElementById("tbl-trade-stage");\n'
        "  let html = '<tr><th>Stage</th><th>Buys</th><th>Sells</th><th>Net</th></tr>';\n"
        '  for (const s of stageTrade) {\n'
        "    const netCls = s.net > 0 ? 'positive' : s.net < 0 ? 'negative' : '';\n"
        "    html += '<tr><td>' + s.stage + '</td>' +\n"
        "      '<td>' + s.buys + '</td><td>' + s.sells + '</td>' +\n"
        '      \'<td class="\' + netCls + \'">\' + (s.net > 0 ? \'+\' : \'\') + s.net + \'</td></tr>\';\n'
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();\n'
        '\n'
        '// --- Per-turn activity ---\n'
        '(function() {\n'
        '  const tbl = document.getElementById("tbl-turns");\n'
        '  // Find global max for bar scaling\n'
        '  let maxVal = 1;\n'
        '  for (const t of turns) {\n'
        '    for (let c = 0; c < 8; c++) {\n'
        '      maxVal = Math.max(maxVal, t.corp_buys[c], t.corp_sells[c]);\n'
        '    }\n'
        '  }\n'
        "  let html = '<tr><th>Turn</th><th>Buys</th><th>Sells</th><th>Net</th>';\n"
        "  for (const cn of corpNames) html += '<th style=\"min-width:80px\">' + cn + '</th>';\n"
        "  html += '</tr>';\n"
        '  for (const t of turns) {\n'
        "    const netCls = t.net > 0 ? 'positive' : t.net < 0 ? 'negative' : '';\n"
        "    html += '<tr><td>' + t.turn + '</td>' +\n"
        "      '<td>' + t.buys + '</td><td>' + t.sells + '</td>' +\n"
        '      \'<td class="\' + netCls + \'">\' + (t.net > 0 ? \'+\' : \'\') + t.net + \'</td>\';\n'
        '    for (let c = 0; c < 8; c++) {\n'
        '      const b = t.corp_buys[c];\n'
        '      const s = t.corp_sells[c];\n'
        '      if (b === 0 && s === 0) { html += \'<td style="color:#555">-</td>\'; }\n'
        '      else {\n'
        '        const bw = b / maxVal * 60;\n'
        '        const sw = s / maxVal * 60;\n'
        '        let cell = \'<td style="text-align:left;padding:2px 4px">\';\n'
        '        if (b > 0) cell += \'<div style="display:inline-block;width:\' + bw + \'px;height:10px;background:#4ecca3;border-radius:1px;vertical-align:middle" title="Buy \' + b + \'"></div> \';\n'
        '        if (s > 0) cell += \'<div style="display:inline-block;width:\' + sw + \'px;height:10px;background:#e94560;border-radius:1px;vertical-align:middle" title="Sell \' + s + \'"></div>\';\n'
        "        if (b > 0 || s > 0) cell += '<br><span style=\"font-size:0.7rem;color:#888\">' + (b > 0 ? '+' + b : '') + (b > 0 && s > 0 ? '/' : '') + (s > 0 ? '-' + s : '') + '</span>';\n"
        "        cell += '</td>';\n"
        '        html += cell;\n'
        '      }\n'
        '    }\n'
        "    html += '</tr>';\n"
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();'
    )

    extra_css = (
        BAR_CSS + "\n"
        + STAT_BOX_CSS + "\n"
        + HIST_BAR_CSS + "\n"
        + ".positive { color: #4ecca3; }\n"
        + ".negative { color: #e94560; }"
    )

    return html_page(
        f"Invest Phase Analysis \u2014 Epoch {epoch}",
        meta=f"{num_states:,} total states from {num_games} games.",
        body=body,
        script=data_js + "\n\n" + JS_MAKE_BAR + "\n\n" + report_js,
        extra_css=extra_css,
        max_width=1200,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Invest phase analysis (auctions, share trades, per-turn activity)"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--load-data", type=str, default=None)
    parser.add_argument("--save-data", type=str, default=None)
    parser.add_argument(
        "--no-open", action="store_true",
        help="Don't open the HTML report in a browser",
    )
    args = parser.parse_args()

    model, config, device, epoch = load_model(
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    if args.load_data:
        print(f"\nLoading data from {args.load_data}")
        dataset = InterpDataset.load(args.load_data)
        print(f"Loaded {dataset.num_states} states from {dataset.num_games} games")
    else:
        print(f"\nCollecting states from {args.num_games} games...")
        dataset = collect_states(
            model, config, device,
            num_games=args.num_games, seed=args.seed,
            checkpoint_path=str(args.checkpoint or f"latest (epoch {epoch})"),
        )
        if args.save_data:
            dataset.save(args.save_data)

    print(f"\nReconstructing invest activity...")
    auctions, trades, invest_stats, logits = reconstruct_invest_activity(
        dataset, model, device, config.num_players, batch_size=args.batch_size,
    )
    print(f"Found {len(auctions)} auctions, {len(trades)} share trades")

    turn_activity = compute_per_turn_activity(dataset, logits, config.num_players)
    print(f"Per-turn activity: {len(turn_activity)} turns with trades")

    analysis = run_analysis(auctions, trades, invest_stats, turn_activity)
    print_report(analysis)

    # HTML report
    html_path = Path("interp/data") / f"invest_phase_epoch{epoch}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html = format_html_report(analysis, epoch, dataset.num_states, dataset.num_games)
    html_path.write_text(html)
    print(f"\nHTML report written to {html_path}")

    if not args.no_open:
        open_file(html_path)


if __name__ == "__main__":
    main()
