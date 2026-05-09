"""Reverse action mapping: engine phase-local action id -> 18xx intent dict."""

from __future__ import annotations

from core.actions import (
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
    get_decision_phase_py,
)
from core.data import ALL_PAR_PRICES, CORP_NAMES, CorpIndices, GamePhases
from core.state import GameState
from entities.company import COMPANIES, CompanyLocation
from entities.market import MARKET
from entities.turn import TURN

LOC_FI = CompanyLocation.LOC_FI


def _fi_purchase_price(corp_id: int, company_id: int) -> int:
    if corp_id == int(CorpIndices.CORP_OS):
        return COMPANIES[company_id].get_face_value()
    return COMPANIES[company_id].get_high_price()


def _par_share_price(par_index: int) -> str:
    par_price = int(ALL_PAR_PRICES[par_index])
    market_index = MARKET.get_index_for_price(par_price)
    if market_index < 0:
        raise ValueError(f"Par price {par_price} is not on the market")
    return f"{par_price},0,{market_index}"


def engine_action_to_18xx(
    action_idx: int,
    state: GameState,
    num_players: int = 3,
) -> dict:
    """Convert a current engine action id into a live 18xx action intent.

    ``action_idx`` is a phase-local id from ``enumerate_legal_actions_py``.
    Some split engine decisions are internal to a single 18xx action:
    ACQ corp/company selection and IPO/PAR composition. Those return
    ``select_*`` / ``par_price`` intents for the caller to combine before
    posting to the 18xx API.
    """
    del num_players

    phase_id = get_decision_phase_py(state)
    if phase_id < 0:
        raise ValueError("Cannot map an action while the engine is in an auto phase")

    info = decode_action_py(phase_id, action_idx)
    phase = TURN.get_phase(state)
    atype = info.action_type

    if phase == GamePhases.PHASE_INVEST:
        if atype == ACTION_PASS:
            return {"type": "pass"}
        if atype == ACTION_AUCTION:
            company_id = info.company_id
            return {
                "type": "bid",
                "company": COMPANIES[company_id].name,
                "price": COMPANIES[company_id].get_face_value(),
            }
        if atype == ACTION_BUY_SHARE:
            return {"type": "buy_shares", "corporation": CORP_NAMES[info.corp_id]}
        if atype == ACTION_SELL_SHARE:
            return {"type": "sell_shares", "corporation": CORP_NAMES[info.corp_id]}

    if phase == GamePhases.PHASE_BID:
        if atype == ACTION_PASS:
            return {"type": "pass"}
        if atype == ACTION_RAISE:
            company_id = TURN.get_active_company(state)
            price = COMPANIES[company_id].get_face_value() + info.amount
            return {
                "type": "bid",
                "company": COMPANIES[company_id].name,
                "price": price,
            }

    if phase == GamePhases.PHASE_IPO:
        if atype == ACTION_PASS:
            return {"type": "pass"}
        if atype == ACTION_IPO:
            return {"type": "ipo_select", "corporation": CORP_NAMES[info.corp_id]}

    if phase == GamePhases.PHASE_PAR:
        if atype == ACTION_PAR:
            par_price = int(ALL_PAR_PRICES[info.amount])
            return {
                "type": "par_price",
                "share_price": _par_share_price(info.amount),
                "par_price": par_price,
            }

    if phase == GamePhases.PHASE_DIVIDENDS:
        if atype == ACTION_DIVIDEND:
            return {"type": "dividend", "amount": info.amount}

    if phase == GamePhases.PHASE_ISSUE_SHARES:
        if atype == ACTION_PASS:
            return {"type": "pass"}
        if atype == ACTION_ISSUE:
            return {"type": "issue"}

    if phase == GamePhases.PHASE_ACQ_SELECT_CORP:
        if atype == ACTION_PASS:
            return {"type": "pass"}
        if atype == ACTION_ACQ_SELECT_CORP:
            return {"type": "select_corp", "corporation": CORP_NAMES[info.corp_id]}

    if phase == GamePhases.PHASE_ACQ_SELECT_COMPANY:
        if atype == ACTION_ACQ_SELECT_COMPANY:
            corp_id = TURN.get_active_corp(state)
            company_id = info.company_id
            if COMPANIES[company_id].get_location(state) == LOC_FI:
                return {
                    "type": "offer",
                    "corporation": CORP_NAMES[corp_id],
                    "company": COMPANIES[company_id].name,
                    "price": _fi_purchase_price(corp_id, company_id),
                }
            return {
                "type": "select_company",
                "company": COMPANIES[company_id].name,
            }

    if phase == GamePhases.PHASE_ACQ_SELECT_PRICE:
        if atype == ACTION_ACQ_PRICE:
            corp_id = TURN.get_active_corp(state)
            company_id = TURN.get_active_company(state)
            return {
                "type": "offer",
                "corporation": CORP_NAMES[corp_id],
                "company": COMPANIES[company_id].name,
                "price": COMPANIES[company_id].get_low_price() + info.amount,
            }

    if phase == GamePhases.PHASE_ACQ_OFFER:
        company_id = TURN.get_active_company(state)
        return {
            "type": "respond",
            "corporation": CORP_NAMES[TURN.get_acq_offer_corp(state)],
            "company": COMPANIES[company_id].name,
            "accept": "true" if atype == ACTION_ACQ_OFFER_ACCEPT else "false",
        }

    if phase == GamePhases.PHASE_CLOSING:
        if atype == ACTION_CLOSE:
            return {"type": "close", "company": COMPANIES[info.company_id].name}
        if atype == ACTION_PASS:
            return {"type": "pass"}

    raise ValueError(
        f"Unknown action: idx={action_idx}, phase={phase}, "
        f"decision_phase={phase_id}, type={atype}, info={info}"
    )
