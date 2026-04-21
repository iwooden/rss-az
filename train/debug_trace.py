"""Human-readable state and action rendering for analysis traces.

This restores the old debug-trace style output used by the pre-refactor
``train.analyze_game`` while adapting it to the refactored phase-local action
space and the current ACQ_OFFER / merged IPO flow.
"""

from __future__ import annotations

from core.actions import (
    ACTION_ACQ_FI_BUY_PY as ACTION_ACQ_FI_BUY,
    ACTION_ACQ_OFFER_ACCEPT_PY as ACTION_ACQ_OFFER_ACCEPT,
    ACTION_ACQ_PRICE_PY as ACTION_ACQ_PRICE,
    ACTION_ACQ_SELECT_COMPANY_PY as ACTION_ACQ_SELECT_COMPANY,
    ACTION_ACQ_SELECT_CORP_PY as ACTION_ACQ_SELECT_CORP,
    ACTION_AUCTION_PY as ACTION_AUCTION,
    ACTION_BUY_SHARE_PY as ACTION_BUY_SHARE,
    ACTION_CLOSE_PY as ACTION_CLOSE,
    ACTION_DIVIDEND_PY as ACTION_DIVIDEND,
    ACTION_IPO_PY as ACTION_IPO,
    ACTION_ISSUE_PY as ACTION_ISSUE,
    ACTION_PAR_PY as ACTION_PAR,
    ACTION_PASS_PY as ACTION_PASS,
    ACTION_RAISE_PY as ACTION_RAISE,
    ACTION_SELL_SHARE_PY as ACTION_SELL_SHARE,
    decode_action_py,
)
from core.data import ALL_PAR_PRICES, COMPANY_NAMES, CORP_NAMES, CorpIndices, GamePhases
from core.state import GameState
from entities.company import COMPANIES, CompanyLocation
from entities.corp import CORPS
from entities.deck import DECK
from entities.fi import FI
from entities.player import PLAYERS
from entities.turn import TURN

NUM_COMPANIES = 36
NUM_CORPS = 8

# Engine-phase display names used in headers and automated-phase history entries.
PHASE_NAMES = {
    GamePhases.PHASE_INVEST: "INVEST",
    GamePhases.PHASE_BID: "BID_IN_AUCTION",
    GamePhases.PHASE_WRAP_UP: "WRAP_UP",
    GamePhases.PHASE_ACQ_SELECT_CORP: "ACQ_SELECT_CORP",
    GamePhases.PHASE_ACQ_OFFER: "ACQ_OFFER",
    GamePhases.PHASE_CLOSING: "CLOSING",
    GamePhases.PHASE_INCOME: "INCOME",
    GamePhases.PHASE_DIVIDENDS: "DIVIDENDS",
    GamePhases.PHASE_END_CARD: "END_CARD",
    GamePhases.PHASE_ISSUE_SHARES: "ISSUE_SHARES",
    GamePhases.PHASE_IPO: "IPO",
    GamePhases.PHASE_GAME_OVER: "GAME_OVER",
    GamePhases.PHASE_PAR: "PAR",
    GamePhases.PHASE_ACQ_SELECT_COMPANY: "ACQ_SELECT_COMPANY",
    GamePhases.PHASE_ACQ_SELECT_PRICE: "ACQ_SELECT_PRICE",
}

# Decision-phase display names used when rendering pass-class actions.
DECISION_PHASE_NAMES = {
    0: "INVEST",
    1: "BID_IN_AUCTION",
    2: "ACQ_SELECT_CORP",
    3: "ACQ_OFFER",
    4: "CLOSING",
    5: "DIVIDENDS",
    6: "ISSUE_SHARES",
    7: "IPO",
    8: "PAR",
    9: "ACQ_SELECT_COMPANY",
    10: "ACQ_SELECT_PRICE",
}


def _auction_companies(state: GameState) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for cid in range(NUM_COMPANIES):
        if COMPANIES[cid].is_for_auction(state):
            result.append((len(result), cid))
    return result


def _format_auction_company(company_id: int) -> str:
    face = COMPANIES[company_id].get_face_value()
    return f"${face}"


def _corp_total_stars(state: GameState, corp_id: int) -> int:
    total = CORPS[corp_id].get_company_stars(state) + CORPS[corp_id].get_cash_stars(state)
    if corp_id == int(CorpIndices.CORP_SI):
        total += 2
    return total


def _corp_company_names(state: GameState, corp_id: int) -> list[str]:
    names: list[str] = []
    for company_id in range(NUM_COMPANIES):
        company = COMPANIES[company_id]
        loc = company.get_location(state)
        owner = company.get_owner_id(state)
        if owner != corp_id:
            continue
        if loc == int(CompanyLocation.LOC_CORP):
            names.append(COMPANY_NAMES[company_id])
        elif loc == int(CompanyLocation.LOC_CORP_ACQ):
            names.append(f"{COMPANY_NAMES[company_id]}*")
    return names


def format_action(phase_id: int, action_id: int, state: GameState | None = None) -> str:
    """Decode a phase-local action into old-style human-readable text.

    ``phase_id == -1`` is the driver-history sentinel for an automated engine
    phase, where ``action_id`` holds the engine phase enum.
    """
    if phase_id < 0:
        return f"AUTO:{PHASE_NAMES.get(action_id, str(action_id))}"

    info = decode_action_py(phase_id, action_id)
    at = info.action_type

    if at == ACTION_PASS:
        return f"PASS ({DECISION_PHASE_NAMES.get(phase_id, str(phase_id))})"

    if at == ACTION_AUCTION:
        face = _format_auction_company(info.company_id)
        if state is None:
            return f"AUCTION {COMPANY_NAMES[info.company_id]} (face {face})"
        slot = next(
            (slot_id for slot_id, company_id in _auction_companies(state) if company_id == info.company_id),
            "?",
        )
        return (
            f"AUCTION slot {slot} "
            f"({COMPANY_NAMES[info.company_id]}, face {face})"
        )

    if at == ACTION_BUY_SHARE:
        return f"BUY {CORP_NAMES[info.corp_id]} share"

    if at == ACTION_SELL_SHARE:
        return f"SELL {CORP_NAMES[info.corp_id]} share"

    if at == ACTION_RAISE:
        if state is not None:
            active_company = TURN.get_active_company(state)
            if 0 <= active_company < NUM_COMPANIES:
                face = COMPANIES[active_company].get_face_value()
                return f"BID ${face + info.amount}"
        return f"BID face+{info.amount}"

    if at == ACTION_ACQ_PRICE:
        if state is not None:
            corp_id = TURN.get_active_corp(state)
            company_id = TURN.get_active_company(state)
            if corp_id >= 0 and company_id >= 0:
                price = COMPANIES[company_id].get_low_price() + info.amount
                return f"ACQUIRE {COMPANY_NAMES[company_id]} with {CORP_NAMES[corp_id]} @ ${price}"
        return f"ACQUIRE at low+{info.amount}"

    if at == ACTION_ACQ_FI_BUY:
        if state is not None:
            corp_id = TURN.get_active_corp(state)
            company_id = TURN.get_active_company(state)
            if corp_id >= 0 and company_id >= 0:
                price = (
                    COMPANIES[company_id].get_face_value()
                    if corp_id == int(CorpIndices.CORP_OS)
                    else COMPANIES[company_id].get_high_price()
                )
                return f"ACQUIRE {COMPANY_NAMES[company_id]} from FI with {CORP_NAMES[corp_id]} @ ${price}"
        return "ACQUIRE from FI"

    if at == ACTION_ACQ_OFFER_ACCEPT:
        if state is not None:
            offered_corp = TURN.get_active_corp(state)
            company_id = TURN.get_active_company(state)
            price = TURN.get_acq_offer_price(state)
            if offered_corp >= 0 and company_id >= 0:
                return f"ACCEPT {CORP_NAMES[offered_corp]} → {COMPANY_NAMES[company_id]} @ ${price}"
        return "ACCEPT OFFER"

    if at == ACTION_CLOSE:
        return f"CLOSE {COMPANY_NAMES[info.company_id]}"

    if at == ACTION_DIVIDEND:
        return f"DIVIDEND ${info.amount}"

    if at == ACTION_ISSUE:
        if state is not None:
            corp_id = TURN.get_active_corp(state)
            if corp_id >= 0:
                return f"ISSUE {CORP_NAMES[corp_id]} shares"
        return "ISSUE shares"

    if at == ACTION_IPO:
        if state is not None:
            company_id = TURN.get_active_company(state)
            if company_id >= 0:
                return f"IPO {COMPANY_NAMES[company_id]} → float {CORP_NAMES[info.corp_id]}"
        return f"IPO → float {CORP_NAMES[info.corp_id]}"

    if at == ACTION_PAR:
        par = ALL_PAR_PRICES[info.amount]
        if state is not None:
            corp_id = TURN.get_active_corp(state)
            company_id = TURN.get_active_company(state)
            if corp_id >= 0 and company_id >= 0:
                return f"PAR {CORP_NAMES[corp_id]} @${par} (IPO {COMPANY_NAMES[company_id]})"
            if corp_id >= 0:
                return f"PAR {CORP_NAMES[corp_id]} @${par}"
        return f"PAR @${par}"

    if at == ACTION_ACQ_SELECT_CORP:
        return f"ACQ select {CORP_NAMES[info.corp_id]}"

    if at == ACTION_ACQ_SELECT_COMPANY:
        if state is not None:
            corp_id = TURN.get_active_corp(state)
            if corp_id >= 0:
                return f"ACQ target {COMPANY_NAMES[info.company_id]} (with {CORP_NAMES[corp_id]})"
        return f"ACQ target {COMPANY_NAMES[info.company_id]}"

    return f"UNKNOWN(phase={phase_id}, action={action_id}, type={at})"


def format_phase_context(state: GameState) -> str:
    """Return a one-line phase-specific context string, or empty string."""
    phase = TURN.get_phase(state)

    if phase == GamePhases.PHASE_BID:
        company_id = TURN.get_active_company(state)
        company = COMPANY_NAMES[company_id] if company_id >= 0 else "?"
        return (
            f"**Auction**: {company} current bid=${TURN.get_auction_price(state)} "
            f"high bidder=P{TURN.get_auction_high_bidder(state)} "
            f"starter=P{TURN.get_auction_starter(state)}"
        )

    if phase == GamePhases.PHASE_ACQ_SELECT_CORP:
        active_player = TURN.get_active_player(state)
        corps = [
            f"{CORP_NAMES[cid]}(${CORPS[cid].get_cash(state)})"
            for cid in range(NUM_CORPS)
            if CORPS[cid].is_active(state)
            and not CORPS[cid].is_in_receivership(state)
            and CORPS[cid].get_president_id(state) == active_player
        ]
        if corps:
            return f"**Acquisition — Select Corp**: P{active_player} may buy with {', '.join(corps)}"
        return f"**Acquisition — Select Corp**: P{active_player}"

    if phase == GamePhases.PHASE_ACQ_SELECT_COMPANY:
        active_player = TURN.get_active_player(state)
        corp_id = TURN.get_active_corp(state)
        if corp_id >= 0:
            return (
                f"**Acquisition — Select Company**: P{active_player} buying with "
                f"{CORP_NAMES[corp_id]} (${CORPS[corp_id].get_cash(state)})"
            )
        return f"**Acquisition — Select Company**: P{active_player}"

    if phase == GamePhases.PHASE_ACQ_SELECT_PRICE:
        active_player = TURN.get_active_player(state)
        corp_id = TURN.get_active_corp(state)
        company_id = TURN.get_active_company(state)
        if corp_id >= 0 and company_id >= 0:
            low = COMPANIES[company_id].get_low_price()
            high = COMPANIES[company_id].get_high_price()
            return (
                f"**Acquisition — Select Price**: P{active_player} "
                f"{CORP_NAMES[corp_id]} -> {COMPANY_NAMES[company_id]} "
                f"(price range ${low}-${high})"
            )
        return f"**Acquisition — Select Price**: P{active_player}"

    if phase == GamePhases.PHASE_ACQ_OFFER:
        offered_corp = TURN.get_active_corp(state)
        company_id = TURN.get_active_company(state)
        original_corp = TURN.get_acq_offer_corp(state)
        price = TURN.get_acq_offer_price(state)
        if offered_corp >= 0 and company_id >= 0:
            if original_corp >= 0 and original_corp != offered_corp:
                return (
                    f"**Acquisition Offer**: {CORP_NAMES[offered_corp]} may preempt "
                    f"{COMPANY_NAMES[company_id]} from FI for ${price} "
                    f"(original buyer: {CORP_NAMES[original_corp]})"
                )
            return (
                f"**Acquisition Offer**: approve {CORP_NAMES[offered_corp]} → "
                f"{COMPANY_NAMES[company_id]} for ${price}"
            )

    if phase == GamePhases.PHASE_CLOSING:
        active_player = TURN.get_active_player(state)
        closable: list[str] = []
        for company_id in range(NUM_COMPANIES):
            company = COMPANIES[company_id]
            loc = company.get_location(state)
            owner = company.get_owner_id(state)
            if loc == int(CompanyLocation.LOC_PLAYER) and owner == active_player:
                closable.append(COMPANY_NAMES[company_id])
                continue
            if loc != int(CompanyLocation.LOC_CORP):
                continue
            corp_id = owner
            if not CORPS[corp_id].is_active(state):
                continue
            if CORPS[corp_id].is_in_receivership(state):
                continue
            if CORPS[corp_id].get_president_id(state) != active_player:
                continue
            if CORPS[corp_id].count_companies(state, include_acquisition=False) <= 1:
                continue
            closable.append(f"{COMPANY_NAMES[company_id]} ({CORP_NAMES[corp_id]})")
        if closable:
            return f"**Closing**: P{active_player} may close {', '.join(closable)}"
        return f"**Closing**: P{active_player}"

    if phase == GamePhases.PHASE_DIVIDENDS:
        corp_id = TURN.get_active_corp(state)
        if corp_id >= 0:
            return f"**Dividends**: {CORP_NAMES[corp_id]}"

    if phase == GamePhases.PHASE_ISSUE_SHARES:
        corp_id = TURN.get_active_corp(state)
        if corp_id >= 0:
            return f"**Issue**: {CORP_NAMES[corp_id]}"

    if phase == GamePhases.PHASE_IPO:
        company_id = TURN.get_active_company(state)
        if company_id >= 0:
            return f"**IPO**: {COMPANY_NAMES[company_id]}"

    if phase == GamePhases.PHASE_PAR:
        company_id = TURN.get_active_company(state)
        corp_id = TURN.get_active_corp(state)
        if company_id >= 0 and corp_id >= 0:
            return f"**PAR**: {COMPANY_NAMES[company_id]} -> {CORP_NAMES[corp_id]}"

    return ""


def format_state_full(state: GameState) -> str:
    """Multi-line visible-state dump in the old debug-trace style."""
    num_players = TURN.get_num_players(state)
    phase = PHASE_NAMES.get(TURN.get_phase(state), str(TURN.get_phase(state)))
    lines: list[str] = []

    lines.append(
        f"Phase: {phase}  |  Turn: {TURN.get_turn_number(state)}  |  "
        f"CoO Level: {TURN.get_coo_level(state)}  |  "
        f"Active Player: {TURN.get_active_player(state)}  |  "
        f"End Card: {'YES' if TURN.is_end_card_flipped(state) else 'no'}"
    )
    lines.append("")

    lines.append("**Players**")
    for pid in range(num_players):
        player = PLAYERS[pid]
        owned = [
            COMPANY_NAMES[cid]
            for cid in range(NUM_COMPANIES)
            if COMPANIES[cid].is_owned_by_player(state, pid)
        ]
        shares: list[str] = []
        for corp_id in range(NUM_CORPS):
            if not CORPS[corp_id].is_active(state):
                continue
            held = player.get_shares(state, corp_id)
            if held <= 0:
                continue
            pres = " (pres)" if CORPS[corp_id].get_president_id(state) == pid else ""
            shares.append(f"{CORP_NAMES[corp_id]}={held}{pres}")
        line = (
            f"  P{pid}: ${player.get_cash(state)} (NW ${player.get_net_worth(state)}) "
            f"order={player.get_turn_order(state)} income=${player.get_income(state)}"
        )
        if owned:
            line += f"  companies=[{', '.join(owned)}]"
        if shares:
            line += f"  shares=[{', '.join(shares)}]"
        lines.append(line)
    lines.append("")

    fi_companies = [
        COMPANY_NAMES[cid]
        for cid in range(NUM_COMPANIES)
        if COMPANIES[cid].is_owned_by_fi(state)
    ]
    fi_line = f"**FI**: ${FI.get_cash(state)} income=${FI.get_income(state)}"
    if fi_companies:
        fi_line += f"  companies=[{', '.join(fi_companies)}]"
    lines.append(fi_line)
    lines.append("")

    auction = _auction_companies(state)
    if auction:
        items = []
        for _, company_id in auction:
            company = COMPANIES[company_id]
            items.append(
                f"{COMPANY_NAMES[company_id]} "
                f"(fv=${company.get_face_value()}, {company.get_stars()}★, inc=${company.get_base_income()})"
            )
        lines.append(f"**Auction Row** [{len(auction)}]: {', '.join(items)}")
    else:
        lines.append("**Auction Row**: (empty)")
    lines.append("")

    active_corps = [corp_id for corp_id in range(NUM_CORPS) if CORPS[corp_id].is_active(state)]
    if active_corps:
        lines.append("**Corporations**")
        for corp_id in active_corps:
            corp = CORPS[corp_id]
            line = (
                f"  {CORP_NAMES[corp_id]}: ${corp.get_cash(state)} "
                f"price=${corp.get_share_price(state)}(idx {corp.get_price_index(state)}) "
                f"shares=bank:{corp.get_bank_shares(state)}/"
                f"unissued:{corp.get_unissued_shares(state)}/issued:{corp.get_issued_shares(state)} "
                f"income=${corp.get_income(state)} stars={_corp_total_stars(state, corp_id)}"
            )
            if corp.is_in_receivership(state):
                line += " RECEIVERSHIP"
            else:
                line += f" pres=P{corp.get_president_id(state)}"
            owned = _corp_company_names(state, corp_id)
            if owned:
                line += f"  companies=[{', '.join(owned)}]"
            lines.append(line)
        lines.append("")

    lines.append(f"**Deck**: {DECK.get_remaining_count(state)} remaining")
    lines.append("")

    ctx = format_phase_context(state)
    if ctx:
        lines.append(ctx)

    return "\n".join(lines)
