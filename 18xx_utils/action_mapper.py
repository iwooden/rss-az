"""Reverse action mapping: engine action_idx → 18xx intent dict.

Converts our Cython engine's integer action indices into simplified intent
dicts that the 18xx.games frontend can translate into proper Engine::Action
objects.
"""

from __future__ import annotations

from core.actions import (
    decode_action_py,
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
    get_company_low_price,
    get_market_index,
    get_par_price,
)
from core.state import GameState
from entities.company import COMPANIES
from entities.turn import TURN

NUM_COMPANIES = 36


def _auction_companies(state: GameState) -> list[tuple[int, int]]:
    """Return list of (slot, company_id) for companies available for auction."""
    result = []
    for cid in range(NUM_COMPANIES):
        if COMPANIES[cid].is_for_auction(state):
            result.append((len(result), cid))
    return result


def engine_action_to_18xx(
    action_idx: int,
    state: GameState,
    num_players: int = 3,
) -> dict:
    """Convert an engine action index to an 18xx intent dict.

    The intent dict is a simplified representation that the 18xx frontend
    translates into proper Engine::Action Ruby objects.

    Args:
        action_idx: Engine action index from MCTS.
        state: Current GameState (needed for auction slot lookups, etc.).
        num_players: Number of players.

    Returns:
        Intent dict with 'type' key and phase-specific fields.
    """
    phase, atype, slot, corp_id, amount = decode_action_py(action_idx, num_players)

    # INVEST phase
    if phase == GamePhases.PHASE_INVEST:
        if atype == ACTION_PASS:
            return {"type": "pass"}
        if atype == ACTION_AUCTION:
            auction = _auction_companies(state)
            if slot >= len(auction):
                raise ValueError(f"Auction slot {slot} out of range ({len(auction)} companies)")
            cid = auction[slot][1]
            company_name = COMPANY_NAMES[cid]
            face = get_company_face_value(cid)
            price = face + amount
            return {"type": "bid", "company": company_name, "price": price}
        if atype == ACTION_BUY_SHARE:
            return {"type": "buy_shares", "corporation": CORP_NAMES[corp_id]}
        if atype == ACTION_SELL_SHARE:
            return {"type": "sell_shares", "corporation": CORP_NAMES[corp_id]}

    # BID phase
    if phase == GamePhases.PHASE_BID_IN_AUCTION:
        if atype == ACTION_LEAVE_AUCTION:
            return {"type": "pass"}
        if atype == ACTION_RAISE_BID:
            company_id = TURN.get_auction_company(state)
            face = get_company_face_value(company_id)
            price = face + amount + 1
            return {"type": "bid", "company": COMPANY_NAMES[company_id], "price": price}

    # IPO phase (returns ipo_select; caller combines with PAR)
    if phase == GamePhases.PHASE_IPO:
        if atype == ACTION_PASS:
            return {"type": "pass"}
        if atype == ACTION_IPO:
            return {"type": "ipo_select", "corporation": CORP_NAMES[corp_id]}

    # PAR phase
    if phase == GamePhases.PHASE_PAR:
        par_price = get_par_price(slot)
        col = get_market_index(par_price)
        return {
            "type": "par_price",
            "share_price": f"{par_price},0,{col}",
            "par_price": par_price,
        }

    # DIVIDENDS phase
    if phase == GamePhases.PHASE_DIVIDENDS:
        return {"type": "dividend", "amount": amount}

    # ISSUE phase
    if phase == GamePhases.PHASE_ISSUE_SHARES:
        if atype == ACTION_PASS:
            return {"type": "pass"}
        if atype == ACTION_ISSUE:
            return {"type": "issue"}

    # ACQUISITION phase
    if phase == GamePhases.PHASE_ACQUISITION:
        if atype == ACTION_ACQ_PRICE:
            acq_company = TURN.get_acq_target_company(state)
            low = get_company_low_price(acq_company)
            price = low + amount
            buyer_corp = TURN.get_acq_active_corp(state)
            return {
                "type": "acquire",
                "corporation": CORP_NAMES[buyer_corp],
                "company": COMPANY_NAMES[acq_company],
                "price": price,
            }
        if atype == ACTION_ACQ_FI_BUY:
            acq_company = TURN.get_acq_target_company(state)
            buyer_corp = TURN.get_acq_active_corp(state)
            return {
                "type": "acquire_fi",
                "corporation": CORP_NAMES[buyer_corp],
                "company": COMPANY_NAMES[acq_company],
            }
        if atype == ACTION_PASS:
            return {"type": "acq_pass"}

    # CLOSING phase
    if phase == GamePhases.PHASE_CLOSING:
        if atype == ACTION_CLOSE:
            closing_company = TURN.get_closing_company(state)
            return {
                "type": "close",
                "company": COMPANY_NAMES[closing_company],
            }
        if atype == ACTION_PASS:
            return {"type": "close_pass"}

    raise ValueError(
        f"Unknown action: idx={action_idx}, phase={phase}, type={atype}, "
        f"slot={slot}, corp_id={corp_id}, amount={amount}"
    )
