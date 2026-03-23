"""Human-readable game state and action trace rendering.

Usage:
    python tests/debug_trace.py [--players N] [--seed S] [--verbose] [--output FILE]

Also available as a setup.py command:
    python setup.py trace_game [--num-players=N] [--seed=S] [--verbose] [--output=FILE]
"""

import argparse
import numpy as np

from core.actions import (
    decode_action_py,
    get_valid_action_mask,
    ACTION_PASS_PY as ACTION_PASS,
    ACTION_AUCTION_PY as ACTION_AUCTION,
    ACTION_BUY_SHARE_PY as ACTION_BUY_SHARE,
    ACTION_SELL_SHARE_PY as ACTION_SELL_SHARE,
    ACTION_LEAVE_AUCTION_PY as ACTION_LEAVE_AUCTION,
    ACTION_RAISE_BID_PY as ACTION_RAISE_BID,
    ACTION_ACQ_PRICE_PY as ACTION_ACQ_PRICE,
    ACTION_ACQ_FI_BUY_PY as ACTION_ACQ_FI_BUY,
    ACTION_CLOSE_PY as ACTION_CLOSE,
    ACTION_DIVIDEND_PY as ACTION_DIVIDEND,
    ACTION_ISSUE_PY as ACTION_ISSUE,
    ACTION_IPO_PY as ACTION_IPO,
    ACTION_PAR_PY as ACTION_PAR,
)
from core.data import (
    COMPANY_NAMES,
    CORP_NAMES,
    GamePhases,
    get_company_face_value,
    get_company_income,
    get_company_stars,
    get_par_price,
)
from core.driver import DRIVER, STATUS_GAME_OVER_PY as STATUS_GAME_OVER
from core.state import GameState
from entities.company import COMPANIES
from entities.corp import CORPS
from entities.deck import DECK
from entities.fi import FI
from entities.player import PLAYERS
from entities.turn import TURN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE_NAMES = {
    0: "INVEST",
    1: "BID_IN_AUCTION",
    2: "WRAP_UP",
    3: "ACQUISITION",
    4: "CLOSING",
    5: "INCOME",
    6: "DIVIDENDS",
    7: "END_CARD",
    8: "ISSUE_SHARES",
    9: "IPO",
    10: "PAR",
    11: "GAME_OVER",
}

SENTINEL_NAMES = {
    -100: "AUTO:WRAP_UP",
    -101: "AUTO:ACQUISITION",
    -102: "AUTO:CLOSING",
    -103: "AUTO:INCOME",
    -105: "AUTO:END_CARD",
}

NUM_COMPANIES = 36
NUM_CORPS = 8


# ---------------------------------------------------------------------------
# Action formatting
# ---------------------------------------------------------------------------

def _auction_companies(state):
    """Return list of (slot, company_id) for companies available for auction."""
    result = []
    for cid in range(NUM_COMPANIES):
        if COMPANIES[cid].is_for_auction(state):
            result.append((len(result), cid))
    return result


def format_action(action_idx, num_players, state=None):
    """Decode an action index into a human-readable string."""
    if action_idx < 0:
        return SENTINEL_NAMES.get(action_idx, f"SENTINEL({action_idx})")

    phase, atype, slot, corp_id, amount = decode_action_py(action_idx, num_players)

    if atype == ACTION_PASS:
        phase_name = PHASE_NAMES.get(phase, str(phase))
        return f"PASS ({phase_name})"

    if atype == ACTION_AUCTION:
        company_name = "?"
        face = "?"
        if state is not None:
            auction = _auction_companies(state)
            if slot < len(auction):
                cid = auction[slot][1]
                company_name = COMPANY_NAMES[cid]
                face = get_company_face_value(cid)
        bid = f"${face}+{amount}" if amount > 0 else f"${face}"
        return f"AUCTION slot {slot} ({company_name}, bid {bid})"

    if atype == ACTION_BUY_SHARE:
        return f"BUY {CORP_NAMES[corp_id]} share"

    if atype == ACTION_SELL_SHARE:
        return f"SELL {CORP_NAMES[corp_id]} share"

    if atype == ACTION_LEAVE_AUCTION:
        return "LEAVE AUCTION"

    if atype == ACTION_RAISE_BID:
        return f"RAISE BID +{amount + 1}"

    if atype == ACTION_ACQ_PRICE:
        return f"ACQUIRE at price offset {amount}"

    if atype == ACTION_ACQ_FI_BUY:
        return "ACQUIRE from FI"

    if atype == ACTION_CLOSE:
        return "CLOSE company"

    if atype == ACTION_DIVIDEND:
        return f"DIVIDEND ${amount}"

    if atype == ACTION_ISSUE:
        return "ISSUE shares"

    if atype == ACTION_IPO:
        return f"IPO → select {CORP_NAMES[corp_id]}"

    if atype == ACTION_PAR:
        par = get_par_price(slot)
        return f"PAR → ${par}"

    return f"UNKNOWN(idx={action_idx})"


# ---------------------------------------------------------------------------
# State formatting
# ---------------------------------------------------------------------------

def format_state_compact(state):
    """One-line state summary."""
    np_ = state.get_num_players()
    phase = PHASE_NAMES.get(state.get_phase(), str(state.get_phase()))
    active = state.get_active_player()
    turn = TURN.get_turn_number(state)
    cash = ", ".join(f"P{i}=${PLAYERS[i].get_cash(state)}" for i in range(np_))
    return f"Turn {turn} | {phase} | Player {active} | {cash}"


def format_phase_context(state) -> str:
    """Return a one-line phase-specific context string, or empty string."""
    phase_idx = state.get_phase()
    if phase_idx == GamePhases.PHASE_BID_IN_AUCTION:
        ac = TURN.get_auction_company(state)
        ap = TURN.get_auction_price(state)
        hb = TURN.get_auction_high_bidder(state)
        st = TURN.get_auction_starter(state)
        ac_name = COMPANY_NAMES[ac] if 0 <= ac < NUM_COMPANIES else "?"
        return (f"**Auction**: {ac_name} current bid=${ap} "
                f"high bidder=P{hb} starter=P{st}")
    elif phase_idx == GamePhases.PHASE_ACQUISITION:
        acorp = TURN.get_acq_active_corp(state)
        atgt = TURN.get_acq_target_company(state)
        fi_offer = TURN.is_acq_fi_offer(state)
        corp_name = CORP_NAMES[acorp] if 0 <= acorp < NUM_CORPS else "?"
        tgt_name = COMPANY_NAMES[atgt] if 0 <= atgt < NUM_COMPANIES else "?"
        return (f"**Acquisition Offer**: {corp_name} → {tgt_name}"
                f"{' (from FI)' if fi_offer else ''}")
    elif phase_idx == GamePhases.PHASE_CLOSING:
        cc = TURN.get_closing_company(state)
        if cc >= 0:
            return f"**Closing Offer**: {COMPANY_NAMES[cc]}"
    elif phase_idx == GamePhases.PHASE_DIVIDENDS:
        dc = TURN.get_dividend_corp(state)
        if dc >= 0:
            return f"**Dividends**: {CORP_NAMES[dc]}"
    elif phase_idx == GamePhases.PHASE_ISSUE_SHARES:
        ic = TURN.get_issue_corp(state)
        if ic >= 0:
            return f"**Issue**: {CORP_NAMES[ic]}"
    elif phase_idx == GamePhases.PHASE_IPO:
        ic = TURN.get_ipo_company(state)
        if ic >= 0:
            return f"**IPO**: {COMPANY_NAMES[ic]}"
    elif phase_idx == GamePhases.PHASE_PAR:
        ic = TURN.get_ipo_company(state)
        pc = TURN.get_par_corp(state)
        if ic >= 0 and pc >= 0:
            return f"**PAR**: {COMPANY_NAMES[ic]} → {CORP_NAMES[pc]}"
    return ""


def format_state_full(state):
    """Multi-line visible-state dump."""
    np_ = state.get_num_players()
    lines = []

    # Header
    phase = PHASE_NAMES.get(state.get_phase(), str(state.get_phase()))
    lines.append(f"Phase: {phase}  |  Turn: {TURN.get_turn_number(state)}  |  "
                 f"CoO Level: {TURN.get_coo_level(state)}  |  "
                 f"Active Player: {state.get_active_player()}  |  "
                 f"End Card: {'YES' if TURN.is_end_card_flipped(state) else 'no'}")
    lines.append("")

    # Players
    lines.append("**Players**")
    for pid in range(np_):
        p = PLAYERS[pid]
        cash = p.get_cash(state)
        nw = p.get_net_worth(state)
        order = p.get_turn_order(state)
        income = p.get_income(state)

        # Owned companies
        owned = [COMPANY_NAMES[cid] for cid in range(NUM_COMPANIES)
                 if COMPANIES[cid].is_owned_by_player(state, pid)]

        # Shares in active corps
        shares = []
        for cid in range(NUM_CORPS):
            if CORPS[cid].is_active(state):
                s = p.get_shares(state, cid)
                if s > 0:
                    pres = " (pres)" if CORPS[cid].get_president_id(state) == pid else ""
                    shares.append(f"{CORP_NAMES[cid]}={s}{pres}")

        line = f"  P{pid}: ${cash} (NW ${nw}) order={order} income=${income}"
        if owned:
            line += f"  companies=[{', '.join(owned)}]"
        if shares:
            line += f"  shares=[{', '.join(shares)}]"
        lines.append(line)
    lines.append("")

    # Foreign Investor
    fi_cash = FI.get_cash(state)
    fi_income = FI.calculate_income(state)
    fi_companies = [COMPANY_NAMES[cid] for cid in range(NUM_COMPANIES)
                    if COMPANIES[cid].is_owned_by_fi(state)]
    fi_line = f"**FI**: ${fi_cash} income=${fi_income}"
    if fi_companies:
        fi_line += f"  companies=[{', '.join(fi_companies)}]"
    lines.append(fi_line)
    lines.append("")

    # Auction row
    auction = _auction_companies(state)
    if auction:
        items = []
        for _, cid in auction:
            name = COMPANY_NAMES[cid]
            fv = get_company_face_value(cid)
            stars = get_company_stars(cid)
            inc = get_company_income(cid)
            items.append(f"{name} (fv=${fv}, {stars}★, inc=${inc})")
        lines.append(f"**Auction Row** [{len(auction)}]: {', '.join(items)}")
    else:
        lines.append("**Auction Row**: (empty)")
    lines.append("")

    # Corporations
    active_corps = [cid for cid in range(NUM_CORPS) if CORPS[cid].is_active(state)]
    if active_corps:
        lines.append("**Corporations**")
        for cid in active_corps:
            c = CORPS[cid]
            name = CORP_NAMES[cid]
            cash = c.get_cash(state)
            price = c.get_share_price(state)
            pidx = c.get_price_index(state)
            bank = c.get_bank_shares(state)
            unissued = c.get_unissued_shares(state)
            issued = c.get_issued_shares(state)
            income = c.get_income(state)
            stars = c.get_stars(state)
            pres = c.get_president_id(state)
            recv = c.is_in_receivership(state)

            owned = [COMPANY_NAMES[co] for co in range(NUM_COMPANIES)
                     if c.owns_company(state, co)]

            line = (f"  {name}: ${cash} price=${price}(idx {pidx}) "
                    f"shares=bank:{bank}/unissued:{unissued}/issued:{issued} "
                    f"income=${income} stars={stars}")
            if recv:
                line += " RECEIVERSHIP"
            else:
                line += f" pres=P{pres}"
            if owned:
                line += f"  companies=[{', '.join(owned)}]"
            lines.append(line)
        lines.append("")

    # Deck (hidden, but unique info)
    lines.append(f"**Deck**: {DECK.get_remaining_count(state)} remaining")
    lines.append("")

    # Phase-specific context
    ctx = format_phase_context(state)
    if ctx:
        lines.append(ctx)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Game trace
# ---------------------------------------------------------------------------

def trace_random_game(num_players, seed, verbose=False):
    """Play a random game and return a markdown trace string."""
    rng = np.random.default_rng(seed)
    state = GameState(num_players=num_players)
    state.initialize_game(seed=seed)

    lines = []
    lines.append(f"# Game Trace: {num_players} players, seed={seed}")
    lines.append("")
    lines.append(format_state_full(state))
    lines.append("")
    lines.append("---")
    lines.append("")

    step = 0
    prev_phase = state.get_phase()
    prev_turn = TURN.get_turn_number(state)

    while state.get_phase() != GamePhases.PHASE_GAME_OVER:
        mask = get_valid_action_mask(state)
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            lines.append(f"**ERROR**: No valid actions at step {step}")
            break

        action = int(rng.choice(valid))
        action_str = format_action(action, num_players, state)
        player = state.get_active_player()
        cur_phase = PHASE_NAMES.get(state.get_phase(), str(state.get_phase()))

        history = []
        status = DRIVER.apply_action(state, action, history=history)

        # Format the player action
        lines.append(f"**Step {step}** P{player} [{cur_phase}]: {action_str}")

        # Show auto-applied actions
        if len(history) > 1:
            for _, aid in history[1:]:
                auto_str = format_action(aid, num_players)
                lines.append(f"  ↳ auto: {auto_str}")

        step += 1
        new_phase = state.get_phase()
        new_turn = TURN.get_turn_number(state)

        # Full state dump on phase or turn change (or always if verbose)
        if verbose or new_phase != prev_phase or new_turn != prev_turn:
            if new_turn != prev_turn:
                lines.append("")
                lines.append(f"--- Turn {new_turn} ---")
            lines.append("")
            lines.append(format_state_full(state))
            lines.append("")

        prev_phase = new_phase
        prev_turn = new_turn

        if status == STATUS_GAME_OVER:
            break

    # Game over summary
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Game Over")
    lines.append("")
    lines.append(f"Completed in {step} player actions")
    lines.append("")
    for pid in range(num_players):
        nw = PLAYERS[pid].get_net_worth(state)
        lines.append(f"  P{pid}: net worth ${nw}")

    net_worths = [PLAYERS[pid].get_net_worth(state) for pid in range(num_players)]
    winner = max(range(num_players), key=lambda i: net_worths[i])
    lines.append("")
    lines.append(f"**Winner: P{winner} (${net_worths[winner]})**")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trace a random game")
    parser.add_argument("--players", type=int, default=3, help="Number of players (2-6)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Full state dump every step")
    parser.add_argument("--output", type=str, default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    result = trace_random_game(args.players, args.seed, args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
            f.write("\n")
        print(f"Trace written to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
