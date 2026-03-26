"""Acquisition phase analysis: price action distribution, value sensitivity, uncertainty.

Analyzes how the model evaluates acquisition offers:
- Action distribution: does it use the full price range or just low/high?
- Price selection patterns by company tier, game stage, and offer type
- Uncertainty analysis: which acquisition states are hardest?
- Value head behavior during acquisitions

Usage:
    .venv/bin/python -m interp.phases.acquisition --load-data interp/data/states.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.actions import decode_action_py
from core.data import PY_COMPANY_PRICE_DIVISOR
from core.state import get_corp_fields, get_layout, get_turn_fields
from interp.html import BAR_CSS, HIST_BAR_CSS, JS_MAKE_BAR, STAT_BOX_CSS, html_page, open_file
from interp.utils import (
    InterpDataset,
    batch_masked_softmax,
    collect_states,
    forward_batched,
    load_model,
)

# Company tiers by star rating
_TIER_NAMES = {1: "Red(1)", 2: "Org(2)", 3: "Yel(3)", 4: "Grn(4)", 5: "Blu(5)"}
_COMPANY_STARS = (
    [1] * 6 + [2] * 8 + [3] * 8 + [4] * 7 + [5] * 7
)  # 36 companies


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


@dataclass
class AcqState:
    """Extracted acquisition state for one decision point."""

    state_idx: int
    company_id: int
    company_stars: int
    low_price: int
    high_price: int
    face_value: int
    price_span: int
    is_fi_offer: bool
    corp_id: int  # -1 if unknown
    corp_cash_norm: float
    corp_share_price_norm: float
    corp_stars_norm: float
    company_income_norm: float
    # Policy
    argmax_action_type: int  # 0=pass, 6=acq_price, 7=fi_buy
    argmax_price_offset: int  # -1 if not acq_price
    argmax_actual_price: int  # -1 if not acq_price
    top1_prob: float
    top2_prob: float
    entropy: float
    # Price distribution
    prob_pass: float
    prob_fi_buy: float
    prob_price_low: float  # offset 0 (= low_price)
    prob_price_high: float  # offset = span (= high_price)
    prob_price_mid: float  # everything between
    # Value
    value_p0: float
    value_spread: float
    # Game progress proxy
    game_third: int  # 0=early, 1=mid, 2=late


def extract_acq_states(
    model: Any,
    device: torch.device,
    dataset: InterpDataset,
    num_players: int,
    batch_size: int = 256,
) -> list[AcqState]:
    """Extract detailed acquisition state information."""
    layout = get_layout(num_players)
    tf = get_turn_fields(num_players)
    cf = get_corp_fields()
    t = layout.turn_offset

    # ACQ phase = 3
    acq_mask = dataset.phases == 3
    acq_indices = np.where(acq_mask)[0]

    if len(acq_indices) == 0:
        return []

    # Forward pass on all states
    logits, values = forward_batched(model, device, dataset.states, batch_size)

    # Compute game thirds (position within each game)
    n = len(dataset.phases)
    game_thirds = np.zeros(n, dtype=int)
    game_breaks: list[int] = [0]
    for i in range(1, n):
        if dataset.phases[i] == 0 and dataset.phases[i - 1] != 0:
            game_breaks.append(i)
    game_breaks.append(n)
    for gi in range(len(game_breaks) - 1):
        start, end = game_breaks[gi], game_breaks[gi + 1]
        game_len = end - start
        if game_len < 3:
            continue
        t1 = start + game_len // 3
        t2 = start + 2 * game_len // 3
        game_thirds[start:t1] = 0
        game_thirds[t1:t2] = 1
        game_thirds[t2:end] = 2

    results: list[AcqState] = []

    for idx in acq_indices:
        state = dataset.states[idx]
        mask = dataset.legal_masks[idx]

        # Extract company info from turn state
        company_stars_norm = state[t + tf.active_company_stars]
        company_stars = max(1, int(company_stars_norm * 5 + 0.5))
        low_price_norm = state[t + tf.active_company_low_price]
        high_price_norm = state[t + tf.active_company_high_price]
        face_value_norm = state[t + tf.active_company_face_value]
        company_income_norm = state[t + tf.active_company_income]

        low_price = int(low_price_norm * PY_COMPANY_PRICE_DIVISOR + 0.5)
        high_price = int(high_price_norm * PY_COMPANY_PRICE_DIVISOR + 0.5)
        face_value = int(face_value_norm * PY_COMPANY_PRICE_DIVISOR + 0.5)
        price_span = high_price - low_price

        is_fi_offer = bool(state[t + tf.acq_is_fi_offer] > 0.5)

        # Corp info
        corp_share_price_norm = state[t + tf.active_corp_share_price]
        corp_stars_norm = state[t + tf.active_corp_stars]

        # Find which corp is active to get corp cash and corp_id
        corp_cash_norm = 0.0
        active_corp_id = -1
        for c in range(8):
            if state[t + tf.active_corp + c] > 0.5:
                active_corp_id = c
                corp_off = layout.corps_offset + c * layout.corp_stride
                corp_cash_norm = state[corp_off + cf.cash]
                break

        # Find company_id from one-hot
        company_oh = state[t + tf.active_company: t + tf.active_company + 36]
        company_id = int(np.argmax(company_oh))

        # Policy analysis
        masked_logits = logits[idx].copy()
        masked_logits[mask <= 0] = -1e9
        probs = batch_masked_softmax(
            logits[idx:idx + 1], mask[np.newaxis]
        )[0]

        argmax = int(np.argmax(masked_logits))
        _, atype, _, _, amount = decode_action_py(argmax, num_players)

        if atype == 6:  # acq_price
            argmax_price_offset = amount
            argmax_actual_price = low_price + amount
        else:
            argmax_price_offset = -1
            argmax_actual_price = -1

        sorted_probs = np.sort(probs)
        top1 = float(sorted_probs[-1])
        top2 = float(sorted_probs[-2])
        ent = float(-np.sum(probs[probs > 1e-10] * np.log(probs[probs > 1e-10])))

        # Price distribution: sum probs for each category
        # ACQ actions: 77-127 = price offsets 0-50, 128 = fi_buy, pass is elsewhere
        # We need to find the pass action
        prob_pass = 0.0
        prob_fi_buy = 0.0
        prob_prices = np.zeros(51)
        for a in range(len(probs)):
            if probs[a] < 1e-10:
                continue
            _, at, _, _, am = decode_action_py(a, num_players)
            if at == 0:  # pass
                prob_pass += probs[a]
            elif at == 7:  # fi_buy
                prob_fi_buy += probs[a]
            elif at == 6:  # acq_price
                if 0 <= am < 51:
                    prob_prices[am] += probs[a]

        # Categorize price probs
        prob_price_low = float(prob_prices[0]) if price_span > 0 else 0.0
        prob_price_high = float(prob_prices[price_span]) if price_span > 0 else float(prob_prices[0])
        prob_price_mid = float(prob_prices[1:price_span].sum()) if price_span > 1 else 0.0

        results.append(AcqState(
            state_idx=int(idx),
            company_id=company_id,
            company_stars=company_stars,
            low_price=low_price,
            high_price=high_price,
            face_value=face_value,
            price_span=price_span,
            is_fi_offer=is_fi_offer,
            corp_id=active_corp_id,
            corp_cash_norm=float(corp_cash_norm),
            corp_share_price_norm=float(corp_share_price_norm),
            corp_stars_norm=float(corp_stars_norm),
            company_income_norm=float(company_income_norm),
            argmax_action_type=atype,
            argmax_price_offset=argmax_price_offset,
            argmax_actual_price=argmax_actual_price,
            top1_prob=top1,
            top2_prob=top2,
            entropy=ent,
            prob_pass=float(prob_pass),
            prob_fi_buy=float(prob_fi_buy),
            prob_price_low=prob_price_low,
            prob_price_high=prob_price_high,
            prob_price_mid=prob_price_mid,
            value_p0=float(values[idx, 0]),
            value_spread=float(np.std(values[idx])),
            game_third=int(game_thirds[idx]),
        ))

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass
class AcqAnalysis:
    """Aggregated acquisition analysis results."""

    total_states: int

    # Action distribution
    count_pass: int
    count_fi_buy: int
    count_price: int
    pct_pass: float
    pct_fi_buy: float
    pct_price: float

    # Price offset distribution (for acq_price actions only)
    count_offset_low: int  # offset=0
    count_offset_high: int  # offset=span
    count_offset_mid: int  # everything between
    pct_offset_low: float
    pct_offset_high: float
    pct_offset_mid: float

    # Mean probability mass on pass/fi_buy/low/high/mid (all states)
    mean_prob_pass: float
    mean_prob_fi_buy: float
    mean_prob_price_low: float
    mean_prob_price_high: float
    mean_prob_price_mid: float

    # By company tier
    tier_stats: dict[int, dict[str, Any]]

    # By game stage
    stage_stats: dict[str, dict[str, Any]]

    # By FI offer vs corp offer
    fi_stats: dict[str, Any]
    corp_stats: dict[str, Any]

    # FI offer detail (OS vs non-OS)
    fi_detail: dict[str, Any]

    # Uncertainty analysis
    mean_entropy: float
    high_uncertainty_count: int  # entropy > 1.0
    uncertain_examples: list[dict[str, Any]]

    # Price offset histogram (for acq_price actions)
    offset_histogram: list[int]  # count per offset 0-50


def run_analysis(states: list[AcqState]) -> AcqAnalysis:
    """Compute aggregated statistics from extracted ACQ states."""
    total = len(states)
    if total == 0:
        raise ValueError("No acquisition states found")

    # Action distribution
    count_pass = sum(1 for s in states if s.argmax_action_type == 0)
    count_fi = sum(1 for s in states if s.argmax_action_type == 7)
    count_price = sum(1 for s in states if s.argmax_action_type == 6)

    price_states = [s for s in states if s.argmax_action_type == 6]
    count_low = sum(1 for s in price_states if s.argmax_price_offset == 0)
    count_high = sum(1 for s in price_states if s.argmax_price_offset == s.price_span)
    count_mid = len(price_states) - count_low - count_high

    # Probability mass
    mean_prob_pass = np.mean([s.prob_pass for s in states])
    mean_prob_fi = np.mean([s.prob_fi_buy for s in states])
    mean_prob_low = np.mean([s.prob_price_low for s in states])
    mean_prob_high = np.mean([s.prob_price_high for s in states])
    mean_prob_mid = np.mean([s.prob_price_mid for s in states])

    # Offset histogram
    offset_hist = [0] * 51
    for s in price_states:
        if 0 <= s.argmax_price_offset < 51:
            offset_hist[s.argmax_price_offset] += 1

    # By tier
    tier_stats: dict[int, dict[str, Any]] = {}
    for stars in sorted(set(s.company_stars for s in states)):
        tier = [s for s in states if s.company_stars == stars]
        t_price = [s for s in tier if s.argmax_action_type == 6]
        t_low = sum(1 for s in t_price if s.argmax_price_offset == 0)
        t_high = sum(1 for s in t_price if s.argmax_price_offset == s.price_span)
        tier_stats[stars] = {
            "count": len(tier),
            "pct_pass": sum(1 for s in tier if s.argmax_action_type == 0) / len(tier),
            "pct_price": len(t_price) / len(tier) if tier else 0,
            "pct_fi_buy": sum(1 for s in tier if s.argmax_action_type == 7) / len(tier),
            "pct_low": t_low / len(t_price) if t_price else 0,
            "pct_high": t_high / len(t_price) if t_price else 0,
            "mean_entropy": float(np.mean([s.entropy for s in tier])),
            "mean_span": float(np.mean([s.price_span for s in tier])),
        }

    # By game stage
    stage_names = {0: "early", 1: "mid", 2: "late"}
    stage_stats: dict[str, dict[str, Any]] = {}
    for stage, name in stage_names.items():
        sg = [s for s in states if s.game_third == stage]
        if not sg:
            continue
        sg_price = [s for s in sg if s.argmax_action_type == 6]
        sg_low = sum(1 for s in sg_price if s.argmax_price_offset == 0)
        sg_high = sum(1 for s in sg_price if s.argmax_price_offset == s.price_span)
        stage_stats[name] = {
            "count": len(sg),
            "pct_pass": sum(1 for s in sg if s.argmax_action_type == 0) / len(sg),
            "pct_price": len(sg_price) / len(sg),
            "pct_low": sg_low / len(sg_price) if sg_price else 0,
            "pct_high": sg_high / len(sg_price) if sg_price else 0,
            "mean_entropy": float(np.mean([s.entropy for s in sg])),
            "mean_value_p0": float(np.mean([s.value_p0 for s in sg])),
        }

    # FI vs corp offers
    fi_offers = [s for s in states if s.is_fi_offer]
    corp_offers = [s for s in states if not s.is_fi_offer]

    def offer_stats(ss: list[AcqState]) -> dict[str, Any]:
        if not ss:
            return {"count": 0}
        sp = [s for s in ss if s.argmax_action_type == 6]
        return {
            "count": len(ss),
            "pct_pass": sum(1 for s in ss if s.argmax_action_type == 0) / len(ss),
            "pct_price": len(sp) / len(ss),
            "pct_fi_buy": sum(1 for s in ss if s.argmax_action_type == 7) / len(ss),
            "mean_entropy": float(np.mean([s.entropy for s in ss])),
        }

    # FI offer detail: OS (corp_id=2) vs non-OS
    # OS has special ability: buys from FI at face_value instead of high_price
    OS_CORP_ID = 2
    fi_os = [s for s in fi_offers if s.corp_id == OS_CORP_ID]
    fi_non_os = [s for s in fi_offers if s.corp_id != OS_CORP_ID]

    def fi_detail_stats(ss: list[AcqState], label: str) -> dict[str, Any]:
        if not ss:
            return {"label": label, "count": 0}
        n_pass = sum(1 for s in ss if s.argmax_action_type == 0)
        n_fi_buy = sum(1 for s in ss if s.argmax_action_type == 7)
        return {
            "label": label,
            "count": len(ss),
            "pct_pass": n_pass / len(ss),
            "pct_fi_buy": n_fi_buy / len(ss),
            "mean_prob_pass": float(np.mean([s.prob_pass for s in ss])),
            "mean_prob_fi_buy": float(np.mean([s.prob_fi_buy for s in ss])),
            "mean_entropy": float(np.mean([s.entropy for s in ss])),
            "mean_value_p0": float(np.mean([s.value_p0 for s in ss])),
            "tier_breakdown": {
                stars: {
                    "count": sum(1 for s in ss if s.company_stars == stars),
                    "pct_fi_buy": (
                        sum(1 for s in ss if s.company_stars == stars and s.argmax_action_type == 7)
                        / max(1, sum(1 for s in ss if s.company_stars == stars))
                    ),
                }
                for stars in sorted(set(s.company_stars for s in ss))
            },
        }

    fi_detail: dict[str, Any] = {
        "all": fi_detail_stats(fi_offers, "All FI offers"),
        "os": fi_detail_stats(fi_os, "OS (face value)"),
        "non_os": fi_detail_stats(fi_non_os, "Other corps (high price)"),
    }

    # Uncertainty
    mean_entropy = float(np.mean([s.entropy for s in states]))
    uncertain = sorted(states, key=lambda s: -s.entropy)
    high_unc = [s for s in states if s.entropy > 1.0]

    uncertain_examples = []
    for s in uncertain[:10]:
        uncertain_examples.append({
            "company_id": s.company_id,
            "stars": s.company_stars,
            "span": s.price_span,
            "is_fi": s.is_fi_offer,
            "entropy": s.entropy,
            "top1_prob": s.top1_prob,
            "argmax": (
                "pass" if s.argmax_action_type == 0
                else "fi_buy" if s.argmax_action_type == 7
                else f"price@{s.argmax_price_offset}"
            ),
            "prob_pass": s.prob_pass,
            "prob_low": s.prob_price_low,
            "prob_high": s.prob_price_high,
            "prob_mid": s.prob_price_mid,
        })

    return AcqAnalysis(
        total_states=total,
        count_pass=count_pass,
        count_fi_buy=count_fi,
        count_price=count_price,
        pct_pass=count_pass / total,
        pct_fi_buy=count_fi / total,
        pct_price=count_price / total,
        count_offset_low=count_low,
        count_offset_high=count_high,
        count_offset_mid=count_mid,
        pct_offset_low=count_low / count_price if count_price else 0,
        pct_offset_high=count_high / count_price if count_price else 0,
        pct_offset_mid=count_mid / count_price if count_price else 0,
        mean_prob_pass=float(mean_prob_pass),
        mean_prob_fi_buy=float(mean_prob_fi),
        mean_prob_price_low=float(mean_prob_low),
        mean_prob_price_high=float(mean_prob_high),
        mean_prob_price_mid=float(mean_prob_mid),
        tier_stats=tier_stats,
        stage_stats=stage_stats,
        fi_stats=offer_stats(fi_offers),
        corp_stats=offer_stats(corp_offers),
        fi_detail=fi_detail,
        mean_entropy=mean_entropy,
        high_uncertainty_count=len(high_unc),
        uncertain_examples=uncertain_examples,
        offset_histogram=offset_hist,
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_report(analysis: AcqAnalysis) -> None:
    a = analysis

    print("\n" + "=" * 90)
    print(f"  ACQUISITION PHASE ANALYSIS ({a.total_states} states)")
    print("=" * 90)

    print(f"\n  1. ACTION DISTRIBUTION (argmax)")
    print(f"  {'Action':<16s} {'Count':>6s} {'%':>8s}")
    print(f"  {'-' * 16} {'-' * 6} {'-' * 8}")
    print(f"  {'pass':<16s} {a.count_pass:>6d} {a.pct_pass:>7.1%}")
    print(f"  {'acq_price':<16s} {a.count_price:>6d} {a.pct_price:>7.1%}")
    print(f"  {'acq_fi_buy':<16s} {a.count_fi_buy:>6d} {a.pct_fi_buy:>7.1%}")

    if a.count_price > 0:
        print(f"\n  Price offset distribution (acq_price actions only):")
        print(f"  {'Category':<16s} {'Count':>6s} {'%':>8s}")
        print(f"  {'-' * 16} {'-' * 6} {'-' * 8}")
        print(f"  {'offset=0 (low)':<16s} {a.count_offset_low:>6d} {a.pct_offset_low:>7.1%}")
        print(f"  {'offset=max(high)':<16s} {a.count_offset_high:>6d} {a.pct_offset_high:>7.1%}")
        print(f"  {'mid offsets':<16s} {a.count_offset_mid:>6d} {a.pct_offset_mid:>7.1%}")

    print(f"\n  Mean probability mass (all ACQ states):")
    print(f"    pass:      {a.mean_prob_pass:.3f}")
    print(f"    fi_buy:    {a.mean_prob_fi_buy:.3f}")
    print(f"    price_low: {a.mean_prob_price_low:.3f}")
    print(f"    price_high:{a.mean_prob_price_high:.3f}")
    print(f"    price_mid: {a.mean_prob_price_mid:.3f}")

    if a.count_price > 0:
        print(f"\n  2. PRICE OFFSET HISTOGRAM (acq_price argmax)")
        max_count = max(a.offset_histogram)
        for off in range(51):
            c = a.offset_histogram[off]
            if c > 0:
                bar = "#" * (c * 40 // max(max_count, 1))
                print(f"    offset {off:>2d}: {c:>4d}  {bar}")

    print(f"\n  3. BY COMPANY TIER")
    print(f"  {'Tier':<8s} {'Count':>6s} {'%Pass':>7s} {'%Price':>7s} {'%FI':>7s} {'%Low':>7s} {'%High':>7s} {'Entropy':>8s} {'Span':>6s}")
    print(f"  {'-' * 8} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 6}")
    for stars in sorted(a.tier_stats.keys()):
        ts = a.tier_stats[stars]
        name = _TIER_NAMES.get(stars, str(stars))
        print(
            f"  {name:<8s} {ts['count']:>6d} {ts['pct_pass']:>6.1%} {ts['pct_price']:>6.1%} "
            f"{ts['pct_fi_buy']:>6.1%} {ts['pct_low']:>6.1%} {ts['pct_high']:>6.1%} "
            f"{ts['mean_entropy']:>8.3f} {ts['mean_span']:>6.1f}"
        )

    print(f"\n  4. BY GAME STAGE")
    print(f"  {'Stage':<8s} {'Count':>6s} {'%Pass':>7s} {'%Price':>7s} {'%Low':>7s} {'%High':>7s} {'Entropy':>8s} {'V(p0)':>8s}")
    print(f"  {'-' * 8} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 8}")
    for name in ["early", "mid", "late"]:
        if name not in a.stage_stats:
            continue
        ss = a.stage_stats[name]
        print(
            f"  {name:<8s} {ss['count']:>6d} {ss['pct_pass']:>6.1%} {ss['pct_price']:>6.1%} "
            f"{ss['pct_low']:>6.1%} {ss['pct_high']:>6.1%} "
            f"{ss['mean_entropy']:>8.3f} {ss['mean_value_p0']:>8.4f}"
        )

    print(f"\n  5. FI vs CORP OFFERS")
    print(f"  {'Type':<8s} {'Count':>6s} {'%Pass':>7s} {'%Price':>7s} {'%FI_buy':>8s} {'Entropy':>8s}")
    print(f"  {'-' * 8} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 8}")
    for name, stats in [("FI", a.fi_stats), ("Corp", a.corp_stats)]:
        if stats["count"] == 0:
            continue
        print(
            f"  {name:<8s} {stats['count']:>6d} {stats['pct_pass']:>6.1%} "
            f"{stats['pct_price']:>6.1%} {stats.get('pct_fi_buy', 0):>7.1%} "
            f"{stats['mean_entropy']:>8.3f}"
        )

    print(f"\n  5b. FI OFFER DETAIL (OS buys at face value, others at high price)")
    for key in ["all", "os", "non_os"]:
        fd = a.fi_detail[key]
        if fd["count"] == 0:
            print(f"    {fd['label']}: no states")
            continue
        print(
            f"    {fd['label']} (n={fd['count']}): "
            f"pass={fd['pct_pass']:.1%}, fi_buy={fd['pct_fi_buy']:.1%}, "
            f"P(pass)={fd['mean_prob_pass']:.3f}, P(fi_buy)={fd['mean_prob_fi_buy']:.3f}, "
            f"entropy={fd['mean_entropy']:.3f}, V(p0)={fd['mean_value_p0']:.4f}"
        )
        # Tier breakdown
        for stars, tb in sorted(fd.get("tier_breakdown", {}).items()):
            if tb["count"] > 0:
                tier_name = _TIER_NAMES.get(stars, str(stars))
                print(f"      {tier_name}: n={tb['count']}, fi_buy={tb['pct_fi_buy']:.1%}")

    print(f"\n  6. UNCERTAINTY")
    print(f"  Mean entropy: {a.mean_entropy:.3f}")
    print(f"  High uncertainty states (entropy > 1.0): {a.high_uncertainty_count}")
    if a.uncertain_examples:
        print(f"\n  Top-10 most uncertain ACQ decisions:")
        print(f"  {'Stars':>5s} {'Span':>5s} {'FI':>3s} {'Entropy':>8s} {'Top1':>6s} {'Argmax':<12s} {'P(pass)':>8s} {'P(low)':>7s} {'P(high)':>7s} {'P(mid)':>7s}")
        for ex in a.uncertain_examples:
            print(
                f"  {ex['stars']:>5d} {ex['span']:>5d} {'Y' if ex['is_fi'] else 'N':>3s} "
                f"{ex['entropy']:>8.3f} {ex['top1_prob']:>5.1%} {ex['argmax']:<12s} "
                f"{ex['prob_pass']:>7.3f} {ex['prob_low']:>7.3f} {ex['prob_high']:>7.3f} "
                f"{ex['prob_mid']:>7.3f}"
            )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def format_html_report(
    analysis: AcqAnalysis,
    epoch: int,
    num_states: int,
    num_games: int,
) -> str:
    a = analysis

    action_dist_json = json.dumps({
        "pass": a.count_pass, "acq_price": a.count_price, "fi_buy": a.count_fi_buy,
        "total": a.total_states,
    })

    offset_json = json.dumps({
        "low": a.count_offset_low, "high": a.count_offset_high, "mid": a.count_offset_mid,
        "total_price": a.count_price,
    })

    prob_mass_json = json.dumps({
        "pass": a.mean_prob_pass, "fi_buy": a.mean_prob_fi_buy,
        "price_low": a.mean_prob_price_low, "price_high": a.mean_prob_price_high,
        "price_mid": a.mean_prob_price_mid,
    })

    histogram_json = json.dumps(a.offset_histogram)

    tier_json = json.dumps([
        {"tier": _TIER_NAMES.get(s, str(s)), "stars": s, **v}
        for s, v in sorted(a.tier_stats.items())
    ])

    stage_json = json.dumps([
        {"stage": name, **a.stage_stats[name]}
        for name in ["early", "mid", "late"] if name in a.stage_stats
    ])

    offer_json = json.dumps({"fi": a.fi_stats, "corp": a.corp_stats})

    fi_detail_json = json.dumps(a.fi_detail)

    uncertain_json = json.dumps(a.uncertain_examples)

    body = (
        '<h2>1. Action Distribution</h2>\n'
        '<div id="action-stats"></div>\n'
        '<table id="tbl-action-dist"></table>\n'
        '<h3>Price Offset Breakdown (acq_price only)</h3>\n'
        '<table id="tbl-offset-dist"></table>\n'
        '<h3>Mean Probability Mass (all ACQ states)</h3>\n'
        '<table id="tbl-prob-mass"></table>\n'
        '\n'
        '<h2>2. Price Offset Histogram</h2>\n'
        '<p style="color:#888;font-size:0.85rem">Distribution of argmax price offsets for acq_price actions. Offset 0 = low price, offset = span = high price.</p>\n'
        '<div id="histogram"></div>\n'
        '\n'
        '<h2>3. By Company Tier</h2>\n'
        '<table id="tbl-tier"></table>\n'
        '\n'
        '<h2>4. By Game Stage</h2>\n'
        '<table id="tbl-stage"></table>\n'
        '\n'
        '<h2>5. FI vs Corp Offers</h2>\n'
        '<table id="tbl-offers"></table>\n'
        '\n'
        '<h3>FI Offer Detail (OS buys at face value, others at high price)</h3>\n'
        '<table id="tbl-fi-detail"></table>\n'
        '\n'
        '<h2>6. Most Uncertain Decisions</h2>\n'
        '<table id="tbl-uncertain"></table>'
    )

    data_js = (
        f"const actionDist = {action_dist_json};\n"
        f"const offsetDist = {offset_json};\n"
        f"const probMass = {prob_mass_json};\n"
        f"const histogram = {histogram_json};\n"
        f"const tiers = {tier_json};\n"
        f"const stages = {stage_json};\n"
        f"const offers = {offer_json};\n"
        f"const fiDetail = {fi_detail_json};\n"
        f"const uncertain = {uncertain_json};"
    )

    report_js = (
        JS_MAKE_BAR + "\n\n"
        "function pct(n, d) { return d > 0 ? (n / d * 100).toFixed(1) + '%' : '\u2014'; }\n"
        "\n"
        "// --- 1. Action Distribution ---\n"
        "(function() {\n"
        '  const div = document.getElementById("action-stats");\n'
        "  div.innerHTML =\n"
        '    \'<div class="stat-box"><div class="stat-label">Total ACQ States</div><div class="stat-value">\' + actionDist.total + \'</div></div>\' +\n'
        '    \'<div class="stat-box"><div class="stat-label">Pass</div><div class="stat-value">\' + pct(actionDist.pass, actionDist.total) + \'</div></div>\' +\n'
        '    \'<div class="stat-box"><div class="stat-label">Price</div><div class="stat-value">\' + pct(actionDist.acq_price, actionDist.total) + \'</div></div>\' +\n'
        '    \'<div class="stat-box"><div class="stat-label">FI Buy</div><div class="stat-value">\' + pct(actionDist.fi_buy, actionDist.total) + \'</div></div>\';\n'
        "\n"
        '  const tbl = document.getElementById("tbl-action-dist");\n'
        "  const maxC = Math.max(actionDist.pass, actionDist.acq_price, actionDist.fi_buy);\n"
        "  tbl.innerHTML =\n"
        "    '<tr><th>Action</th><th>Count</th><th>%</th><th></th></tr>' +\n"
        "    '<tr><td>pass</td><td>' + actionDist.pass + '</td><td>' + pct(actionDist.pass, actionDist.total) + '</td><td>' + makeBar(actionDist.pass, maxC, 'bar-blue') + '</td></tr>' +\n"
        "    '<tr><td>acq_price</td><td>' + actionDist.acq_price + '</td><td>' + pct(actionDist.acq_price, actionDist.total) + '</td><td>' + makeBar(actionDist.acq_price, maxC, 'bar-green') + '</td></tr>' +\n"
        "    '<tr><td>fi_buy</td><td>' + actionDist.fi_buy + '</td><td>' + pct(actionDist.fi_buy, actionDist.total) + '</td><td>' + makeBar(actionDist.fi_buy, maxC, 'bar-orange') + '</td></tr>';\n"
        "\n"
        "  // Offset dist\n"
        '  const tbl2 = document.getElementById("tbl-offset-dist");\n'
        "  const tp = offsetDist.total_price;\n"
        "  const maxO = Math.max(offsetDist.low, offsetDist.high, offsetDist.mid);\n"
        "  tbl2.innerHTML =\n"
        "    '<tr><th>Category</th><th>Count</th><th>%</th><th></th></tr>' +\n"
        "    '<tr><td>offset=0 (low price)</td><td>' + offsetDist.low + '</td><td>' + pct(offsetDist.low, tp) + '</td><td>' + makeBar(offsetDist.low, maxO, 'bar-green') + '</td></tr>' +\n"
        "    '<tr><td>offset=max (high price)</td><td>' + offsetDist.high + '</td><td>' + pct(offsetDist.high, tp) + '</td><td>' + makeBar(offsetDist.high, maxO, 'bar-orange') + '</td></tr>' +\n"
        "    '<tr><td>intermediate offsets</td><td>' + offsetDist.mid + '</td><td>' + pct(offsetDist.mid, tp) + '</td><td>' + makeBar(offsetDist.mid, maxO, 'bar-blue') + '</td></tr>';\n"
        "\n"
        "  // Prob mass\n"
        '  const tbl3 = document.getElementById("tbl-prob-mass");\n'
        "  tbl3.innerHTML =\n"
        "    '<tr><th>Category</th><th>Mean Prob</th><th></th></tr>' +\n"
        "    '<tr><td>pass</td><td>' + probMass.pass.toFixed(3) + '</td><td>' + makeBar(probMass.pass, 1.0, 'bar-blue') + '</td></tr>' +\n"
        "    '<tr><td>price (low)</td><td>' + probMass.price_low.toFixed(3) + '</td><td>' + makeBar(probMass.price_low, 1.0, 'bar-green') + '</td></tr>' +\n"
        "    '<tr><td>price (high)</td><td>' + probMass.price_high.toFixed(3) + '</td><td>' + makeBar(probMass.price_high, 1.0, 'bar-orange') + '</td></tr>' +\n"
        "    '<tr><td>price (mid)</td><td>' + probMass.price_mid.toFixed(3) + '</td><td>' + makeBar(probMass.price_mid, 1.0, 'bar-blue') + '</td></tr>' +\n"
        "    '<tr><td>fi_buy</td><td>' + probMass.fi_buy.toFixed(3) + '</td><td>' + makeBar(probMass.fi_buy, 1.0, 'bar-orange') + '</td></tr>';\n"
        "})();\n"
        "\n"
        "// --- 2. Histogram ---\n"
        "(function() {\n"
        '  const div = document.getElementById("histogram");\n'
        "  const maxC = Math.max(...histogram);\n"
        '  if (maxC === 0) { div.innerHTML = \'<p style="color:#888">No acq_price actions</p>\'; return; }\n'
        "  let html = '<table style=\"width:auto\"><tr><th>Offset</th><th>Count</th><th style=\"min-width:300px\"></th></tr>';\n"
        "  for (let i = 0; i < histogram.length; i++) {\n"
        "    if (histogram[i] > 0) {\n"
        "      const w = histogram[i] / maxC * 280;\n"
        "      html += '<tr><td>' + i + '</td><td>' + histogram[i] + '</td>' +\n"
        '        \'<td style="text-align:left"><span class="hist-bar" style="width:\' + w + \'px"></span></td></tr>\';\n'
        "    }\n"
        "  }\n"
        "  html += '</table>';\n"
        "  div.innerHTML = html;\n"
        "})();\n"
        "\n"
        "// --- 3. Tiers ---\n"
        "(function() {\n"
        '  const tbl = document.getElementById("tbl-tier");\n'
        "  let html = '<tr><th>Tier</th><th>Count</th><th>%Pass</th><th>%Price</th><th>%FI</th><th>%Low</th><th>%High</th><th>Entropy</th><th>Span</th></tr>';\n"
        "  for (const t of tiers) {\n"
        "    html += '<tr><td>' + t.tier + '</td>' +\n"
        "      '<td>' + t.count + '</td>' +\n"
        "      '<td>' + (t.pct_pass * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + (t.pct_price * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + (t.pct_fi_buy * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + (t.pct_low * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + (t.pct_high * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + t.mean_entropy.toFixed(3) + '</td>' +\n"
        "      '<td>' + t.mean_span.toFixed(1) + '</td></tr>';\n"
        "  }\n"
        "  tbl.innerHTML = html;\n"
        "})();\n"
        "\n"
        "// --- 4. Stages ---\n"
        "(function() {\n"
        '  const tbl = document.getElementById("tbl-stage");\n'
        "  let html = '<tr><th>Stage</th><th>Count</th><th>%Pass</th><th>%Price</th><th>%Low</th><th>%High</th><th>Entropy</th><th>V(p0)</th></tr>';\n"
        "  for (const s of stages) {\n"
        "    html += '<tr><td>' + s.stage + '</td>' +\n"
        "      '<td>' + s.count + '</td>' +\n"
        "      '<td>' + (s.pct_pass * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + (s.pct_price * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + (s.pct_low * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + (s.pct_high * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + s.mean_entropy.toFixed(3) + '</td>' +\n"
        "      '<td>' + s.mean_value_p0.toFixed(4) + '</td></tr>';\n"
        "  }\n"
        "  tbl.innerHTML = html;\n"
        "})();\n"
        "\n"
        "// --- 5. Offers ---\n"
        "(function() {\n"
        '  const tbl = document.getElementById("tbl-offers");\n'
        "  let html = '<tr><th>Type</th><th>Count</th><th>%Pass</th><th>%Price</th><th>%FI Buy</th><th>Entropy</th></tr>';\n"
        "  for (const [name, stats] of [['FI Offer', offers.fi], ['Corp Offer', offers.corp]]) {\n"
        "    if (stats.count === 0) continue;\n"
        "    html += '<tr><td>' + name + '</td>' +\n"
        "      '<td>' + stats.count + '</td>' +\n"
        "      '<td>' + (stats.pct_pass * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + ((stats.pct_price || 0) * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + ((stats.pct_fi_buy || 0) * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + stats.mean_entropy.toFixed(3) + '</td></tr>';\n"
        "  }\n"
        "  tbl.innerHTML = html;\n"
        "})();\n"
        "\n"
        "// --- 5b. FI Detail ---\n"
        "(function() {\n"
        '  const tbl = document.getElementById("tbl-fi-detail");\n'
        "  let html = '<tr><th>Category</th><th>Count</th><th>%Pass</th><th>%FI Buy</th><th>P(pass)</th><th>P(fi_buy)</th><th>Entropy</th><th>V(p0)</th></tr>';\n"
        "  for (const key of ['all', 'os', 'non_os']) {\n"
        "    const d = fiDetail[key];\n"
        "    if (!d || d.count === 0) continue;\n"
        "    html += '<tr><td>' + d.label + '</td>' +\n"
        "      '<td>' + d.count + '</td>' +\n"
        "      '<td>' + (d.pct_pass * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + (d.pct_fi_buy * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + d.mean_prob_pass.toFixed(3) + '</td>' +\n"
        "      '<td>' + d.mean_prob_fi_buy.toFixed(3) + '</td>' +\n"
        "      '<td>' + d.mean_entropy.toFixed(3) + '</td>' +\n"
        "      '<td>' + d.mean_value_p0.toFixed(4) + '</td></tr>';\n"
        "  }\n"
        "  tbl.innerHTML = html;\n"
        "})();\n"
        "\n"
        "// --- 6. Uncertainty ---\n"
        "(function() {\n"
        '  const tbl = document.getElementById("tbl-uncertain");\n'
        "  let html = '<tr><th>Stars</th><th>Span</th><th>FI</th><th>Entropy</th><th>Top1%</th><th>Argmax</th><th>P(pass)</th><th>P(low)</th><th>P(high)</th><th>P(mid)</th></tr>';\n"
        "  for (const ex of uncertain) {\n"
        "    html += '<tr><td>' + ex.stars + '</td>' +\n"
        "      '<td>' + ex.span + '</td>' +\n"
        "      '<td>' + (ex.is_fi ? 'Y' : 'N') + '</td>' +\n"
        "      '<td>' + ex.entropy.toFixed(3) + '</td>' +\n"
        "      '<td>' + (ex.top1_prob * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + ex.argmax + '</td>' +\n"
        "      '<td>' + ex.prob_pass.toFixed(3) + '</td>' +\n"
        "      '<td>' + ex.prob_low.toFixed(3) + '</td>' +\n"
        "      '<td>' + ex.prob_high.toFixed(3) + '</td>' +\n"
        "      '<td>' + ex.prob_mid.toFixed(3) + '</td></tr>';\n"
        "  }\n"
        "  tbl.innerHTML = html;\n"
        "})();"
    )

    meta = f"{a.total_states} ACQ states from {num_states:,} total states ({num_games} games)."
    extra_css = BAR_CSS + "\n" + STAT_BOX_CSS + "\n" + HIST_BAR_CSS

    return html_page(
        f"Acquisition Phase Analysis \u2014 Epoch {epoch}",
        meta=meta,
        body=body,
        script=data_js + "\n\n" + report_js,
        extra_css=extra_css,
        max_width=1100,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquisition phase analysis (price distribution, uncertainty, game stage)"
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

    print(f"\nExtracting acquisition states...")
    acq_states = extract_acq_states(
        model, device, dataset, config.num_players, batch_size=args.batch_size,
    )
    print(f"Found {len(acq_states)} ACQ states")

    if not acq_states:
        print("No acquisition states found!")
        return

    analysis = run_analysis(acq_states)
    print_report(analysis)

    # HTML report
    html_path = Path("interp/data") / f"acq_phase_epoch{epoch}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html = format_html_report(analysis, epoch, dataset.num_states, dataset.num_games)
    html_path.write_text(html)
    print(f"\nHTML report written to {html_path}")

    if not args.no_open:
        open_file(html_path)


if __name__ == "__main__":
    main()
