"""Parser for 18xx game JSON actions -> live engine action ids.

This module intentionally maps against the current sparse legal-action surface.
It does not maintain a parallel copy of action-layout arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    ActionInfoTuple,
    decode_action_py,
    enumerate_legal_actions_py,
    get_decision_phase_py,
)
from core.data import (
    ALL_PAR_PRICES,
    COMPANY_NAME_TO_ID,
    CORP_NAME_TO_ID,
    GamePhases,
    MAX_ACTION_SIZE,
)
from core.state import GameState
from entities.company import COMPANIES, CompanyLocation
from entities.deck import DECK
from entities.turn import TURN

LOC_AUCTION = CompanyLocation.LOC_AUCTION
LOC_REVEALED = CompanyLocation.LOC_REVEALED
LOC_FI = CompanyLocation.LOC_FI

# Phase constants shared by replay helpers.
PHASE_INVEST = GamePhases.PHASE_INVEST
PHASE_BID = GamePhases.PHASE_BID
PHASE_WRAP_UP = GamePhases.PHASE_WRAP_UP
PHASE_ACQ_SELECT_CORP = GamePhases.PHASE_ACQ_SELECT_CORP
PHASE_ACQ_OFFER = GamePhases.PHASE_ACQ_OFFER
PHASE_CLOSING = GamePhases.PHASE_CLOSING
PHASE_INCOME = GamePhases.PHASE_INCOME
PHASE_DIVIDENDS = GamePhases.PHASE_DIVIDENDS
PHASE_END_CARD = GamePhases.PHASE_END_CARD
PHASE_ISSUE = GamePhases.PHASE_ISSUE_SHARES
PHASE_IPO = GamePhases.PHASE_IPO
PHASE_PAR = GamePhases.PHASE_PAR
PHASE_ACQ_SELECT_COMPANY = GamePhases.PHASE_ACQ_SELECT_COMPANY
PHASE_ACQ_SELECT_PRICE = GamePhases.PHASE_ACQ_SELECT_PRICE
PHASE_GAME_OVER = GamePhases.PHASE_GAME_OVER

# Legacy compatibility alias for older helpers that still import PHASE_ACQ.
# Do not treat this as the full post-refactor acquisition phase group.
PHASE_ACQ = PHASE_ACQ_SELECT_CORP


@dataclass(frozen=True)
class ActionLayout:
    """Compatibility wrapper retained for callers that still construct one.

    The replay path no longer uses numeric layout offsets. Only ``num_players`` is
    retained because older callers thread a layout object through the API.
    """

    num_players: int


INVEST_ACTIONS = {"bid", "buy_shares", "sell_shares", "pass"}
BID_ACTIONS = {"bid", "pass"}
IPO_ACTIONS = {"par", "pass"}
DIVIDEND_ACTIONS = {"dividend"}
ISSUE_ACTIONS = {"sell_shares", "pass"}

SKIP_ACTIONS = {
    "end_game",
    "message",
    "program_close_pass",
    "program_disable",
    "program_share_pass",
    "redo",
    "undo",
}


def filter_actions(actions: list, committed_ids: set | None = None) -> list:
    """Remove meta and undone actions from a raw 18xx action list."""
    result = []
    for action in actions:
        atype = action.get("type", "")
        action_id = action.get("id")

        if atype in SKIP_ACTIONS:
            if committed_ids is not None and action_id is not None and action_id not in committed_ids:
                continue
            for auto in action.get("auto_actions", []):
                result.append(auto)
            continue

        if committed_ids is not None and action_id is not None and action_id not in committed_ids:
            continue

        result.append(action)

    return result


def flatten_auto_actions(actions: list) -> list:
    """Expand nested ``auto_actions`` into the main action stream."""
    result = []
    for action in actions:
        result.append(action)
        for auto in action.get("auto_actions", []):
            annotated = dict(auto)
            annotated["_auto_parent_type"] = action.get("type")
            annotated["_auto_parent_id"] = action.get("id")
            result.append(annotated)
    return result


class AutoPassTracker:
    """Tracks 18xx ``program_*`` auto-pass flags by player id."""

    def __init__(self, player_ids: list):
        self.player_ids = list(player_ids)
        self.share_pass: dict = {}
        self.close_pass: dict = {}

    def process_action(self, action: dict):
        atype = action.get("type", "")
        entity = action.get("entity")

        if atype == "program_share_pass":
            self.share_pass[entity] = {
                "unconditional": action.get("unconditional", False),
                "indefinite": action.get("indefinite", False),
            }
        elif atype == "program_close_pass":
            self.close_pass[entity] = True
        elif atype == "program_disable":
            original = action.get("original_type", "")
            if original == "program_share_pass":
                self.share_pass.pop(entity, None)
            elif original == "program_close_pass":
                self.close_pass.pop(entity, None)

    def should_auto_pass_invest(self, player_id) -> bool:
        return player_id in self.share_pass

    def should_auto_pass_closing(self, player_id) -> bool:
        return player_id in self.close_pass


def entity_to_player_index(players_json: list, entity_id) -> int:
    """Return the 0-based index of the player with the given entity id."""
    try:
        target = int(entity_id)
    except (TypeError, ValueError):
        target = entity_id

    for idx, player in enumerate(players_json):
        pid = player.get("id")
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            pass
        if pid == target:
            return idx

    raise ValueError(f"Player with entity_id={entity_id!r} not found in players list")


def get_legal_actions(state: GameState) -> list[tuple[int, ActionInfoTuple]]:
    """Enumerate and decode all currently legal actions."""
    phase_id = get_decision_phase_py(state)
    if phase_id < 0:
        return []

    buf = np.zeros(MAX_ACTION_SIZE, dtype=np.uint16)
    count = int(enumerate_legal_actions_py(state, buf))
    return [
        (int(buf[i]), decode_action_py(phase_id, int(buf[i])))
        for i in range(count)
    ]


def find_legal_actions(
    state: GameState,
    *,
    action_type: int | None = None,
    corp_id: int | None = None,
    company_id: int | None = None,
    amount: int | None = None,
) -> list[tuple[int, ActionInfoTuple]]:
    """Return all legal actions matching decoded action fields."""
    matches = []
    for action_id, info in get_legal_actions(state):
        if action_type is not None and info.action_type != action_type:
            continue
        if corp_id is not None and info.corp_id != corp_id:
            continue
        if company_id is not None and info.company_id != company_id:
            continue
        if amount is not None and info.amount != amount:
            continue
        matches.append((action_id, info))
    return matches


def find_legal_action(
    state: GameState,
    *,
    action_type: int | None = None,
    corp_id: int | None = None,
    company_id: int | None = None,
    amount: int | None = None,
) -> int:
    """Return the unique legal action matching decoded action fields."""
    matches = find_legal_actions(
        state,
        action_type=action_type,
        corp_id=corp_id,
        company_id=company_id,
        amount=amount,
    )
    if len(matches) == 1:
        return matches[0][0]

    available = [
        f"aid={aid} type={info.action_type} corp={info.corp_id} company={info.company_id} amount={info.amount}"
        for aid, info in get_legal_actions(state)
    ]
    raise ValueError(
        "Expected exactly one legal action matching "
        f"action_type={action_type}, corp_id={corp_id}, company_id={company_id}, amount={amount}; "
        f"found {len(matches)}. Available: {available}"
    )


def _parse_share_corp_id(action: dict) -> int:
    share_name = action["shares"][0]
    corp_name = share_name.split("_")[0]
    return CORP_NAME_TO_ID[corp_name]


def _parse_par_index(action: dict) -> int:
    price_str = action["share_price"]
    par_price = int(str(price_str).split(",")[0])
    try:
        return ALL_PAR_PRICES.index(par_price)
    except ValueError as exc:
        raise ValueError(f"Par price {par_price} is not in ALL_PAR_PRICES") from exc


def _parse_boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def map_invest_action(state: GameState, action: dict, layout: ActionLayout | None = None) -> int:
    atype = action["type"]

    if atype == "pass":
        return find_legal_action(state, action_type=ACTION_PASS)

    if atype == "bid":
        company_id = COMPANY_NAME_TO_ID[action["company"]]
        return find_legal_action(
            state,
            action_type=ACTION_AUCTION,
            company_id=company_id,
        )

    if atype == "buy_shares":
        corp_id = _parse_share_corp_id(action)
        return find_legal_action(state, action_type=ACTION_BUY_SHARE, corp_id=corp_id)

    if atype == "sell_shares":
        corp_id = _parse_share_corp_id(action)
        return find_legal_action(state, action_type=ACTION_SELL_SHARE, corp_id=corp_id)

    raise ValueError(f"Unrecognised INVEST action type: {atype!r}")


def map_bid_action(state: GameState, action: dict, layout: ActionLayout | None = None) -> int:
    atype = action["type"]

    if atype == "pass":
        return find_legal_action(state, action_type=ACTION_PASS)

    if atype == "bid":
        company_id = COMPANY_NAME_TO_ID[action["company"]]
        price = int(action["price"])
        amount = price - COMPANIES[company_id].get_face_value()
        return find_legal_action(
            state,
            action_type=ACTION_RAISE,
            amount=amount,
        )

    raise ValueError(f"Unrecognised BID action type: {atype!r}")


def map_ipo_action(state: GameState, action: dict, layout: ActionLayout | None = None) -> int:
    atype = action["type"]

    if atype == "pass":
        return find_legal_action(state, action_type=ACTION_PASS)

    if atype == "par":
        corp_id = CORP_NAME_TO_ID[action["corporation"]]
        return find_legal_action(
            state,
            action_type=ACTION_IPO,
            corp_id=corp_id,
        )

    raise ValueError(f"Unrecognised IPO action type: {atype!r}")


def map_par_action(state: GameState, action: dict, layout: ActionLayout | None = None) -> int:
    if action["type"] != "par":
        raise ValueError(f"Unrecognised PAR action type: {action['type']!r}")
    return find_legal_action(
        state,
        action_type=ACTION_PAR,
        amount=_parse_par_index(action),
    )


def _action_matches_active_corp(state: GameState, action: dict) -> bool:
    corp_id = CORP_NAME_TO_ID.get(action.get("entity"))
    return corp_id is not None and TURN.get_active_corp(state) == corp_id


def map_dividend_action(state: GameState, action: dict, layout: ActionLayout | None = None) -> int | None:
    if action["type"] != "dividend":
        raise ValueError(f"Unrecognised DIVIDENDS action type: {action['type']!r}")
    if not _action_matches_active_corp(state, action):
        return None
    return find_legal_action(
        state,
        action_type=ACTION_DIVIDEND,
        amount=int(action["amount"]),
    )


def map_issue_action(state: GameState, action: dict, layout: ActionLayout | None = None) -> int | None:
    if not _action_matches_active_corp(state, action):
        return None

    atype = action["type"]
    if atype == "sell_shares":
        return find_legal_action(state, action_type=ACTION_ISSUE)
    if atype == "pass":
        return find_legal_action(state, action_type=ACTION_PASS)
    raise ValueError(f"Unrecognised ISSUE action type: {atype!r}")


def map_acquisition_action(
    state: GameState,
    action: dict,
    phase: int,
    layout: ActionLayout | None = None,
) -> int:
    if action["type"] != "offer":
        raise ValueError(f"Unrecognised ACQUISITION action type: {action['type']!r}")

    company_id = COMPANY_NAME_TO_ID[action["company"]]
    corp_id = CORP_NAME_TO_ID[action["corporation"]]

    if phase == PHASE_ACQ_SELECT_CORP:
        return find_legal_action(
            state,
            action_type=ACTION_ACQ_SELECT_CORP,
            corp_id=corp_id,
        )

    if phase == PHASE_ACQ_SELECT_COMPANY:
        return find_legal_action(
            state,
            action_type=ACTION_ACQ_SELECT_COMPANY,
            company_id=company_id,
        )

    if phase != PHASE_ACQ_SELECT_PRICE:
        raise ValueError(f"Offer cannot be mapped in phase {phase}")

    if TURN.get_active_corp(state) != corp_id:
        raise ValueError("Offer corporation does not match active acquisition corp")
    if TURN.get_active_company(state) != company_id:
        raise ValueError("Offer company does not match active acquisition company")

    if COMPANIES[company_id].get_location(state) == LOC_FI:
        raise ValueError("FI target should execute during ACQ_SELECT_COMPANY, not ACQ_SELECT_PRICE")

    amount = int(action["price"]) - COMPANIES[company_id].get_low_price()
    return find_legal_action(
        state,
        action_type=ACTION_ACQ_PRICE,
        amount=amount,
    )


def map_acq_offer_action(state: GameState, action: dict, layout: ActionLayout | None = None) -> int:
    if action["type"] != "respond":
        raise ValueError(f"Unrecognised ACQ_OFFER action type: {action['type']!r}")
    if _parse_boolish(action.get("accept")):
        return find_legal_action(state, action_type=ACTION_ACQ_OFFER_ACCEPT)
    return find_legal_action(state, action_type=ACTION_PASS)


def map_closing_action(state: GameState, action: dict, layout: ActionLayout | None = None) -> int:
    atype = action["type"]
    if atype == "pass":
        return find_legal_action(state, action_type=ACTION_PASS)
    if atype in {"sell_company", "close"}:
        company_name = action.get("company") or action.get("entity")
        company_id = COMPANY_NAME_TO_ID[company_name]
        return find_legal_action(
            state,
            action_type=ACTION_CLOSE,
            company_id=company_id,
        )
    raise ValueError(f"Unrecognised CLOSING action type: {atype!r}")


def override_deck_and_offering(
    state: GameState,
    deck_order_names: list[str],
    offering_names: list[str],
) -> None:
    """Override deck/offering to match an 18xx game's initial state.

    ``DECK.set_order`` rewrites semantic membership for included cards, but it
    intentionally preserves non-deck companies that are already sitting in
    auction/revealed locations. Fresh initialized states can therefore leak
    seed-opening companies that are excluded for this player count unless we
    clear them first.
    """
    for cid in range(36):
        if COMPANIES[cid].get_location(state) in (LOC_AUCTION, LOC_REVEALED):
            COMPANIES[cid].exclude_from_game(state)

    remaining_ids = [COMPANY_NAME_TO_ID[name] for name in reversed(deck_order_names)]
    offering_ids = [COMPANY_NAME_TO_ID[name] for name in reversed(offering_names)]
    DECK.set_order(state, remaining_ids + offering_ids)

    for _ in range(len(offering_names)):
        cid = DECK.draw(state)
        COMPANIES[cid].move_to_auction(state)


def map_action(
    state: GameState,
    action: dict,
    phase: int,
    layout: ActionLayout | None,
) -> int | None:
    """Map a single 18xx action to a current engine action id."""
    atype = action.get("type", "")
    entity_type = action.get("entity_type", "")

    if phase == PHASE_INVEST:
        if atype in INVEST_ACTIONS and entity_type == "player":
            return map_invest_action(state, action, layout)
        return None

    if phase == PHASE_BID:
        if atype in BID_ACTIONS:
            return map_bid_action(state, action, layout)
        return None

    if phase == PHASE_IPO:
        if atype in IPO_ACTIONS:
            return map_ipo_action(state, action, layout)
        return None

    if phase == PHASE_PAR:
        if atype == "par":
            return map_par_action(state, action, layout)
        return None

    if phase == PHASE_DIVIDENDS:
        if atype in DIVIDEND_ACTIONS and entity_type == "corporation":
            return map_dividend_action(state, action, layout)
        return None

    if phase == PHASE_ISSUE:
        if atype in ISSUE_ACTIONS and entity_type == "corporation":
            return map_issue_action(state, action, layout)
        return None

    if phase in (PHASE_ACQ_SELECT_CORP, PHASE_ACQ_SELECT_COMPANY, PHASE_ACQ_SELECT_PRICE):
        if atype == "offer":
            return map_acquisition_action(state, action, phase, layout)
        if atype == "pass" and phase == PHASE_ACQ_SELECT_CORP:
            return find_legal_action(state, action_type=ACTION_PASS)
        return None

    if phase == PHASE_ACQ_OFFER:
        if atype == "respond":
            return map_acq_offer_action(state, action, layout)
        return None

    if phase == PHASE_CLOSING:
        if atype in {"sell_company", "close", "pass"}:
            return map_closing_action(state, action, layout)
        return None

    if phase in (PHASE_WRAP_UP, PHASE_INCOME, PHASE_END_CARD, PHASE_GAME_OVER):
        return None

    return None
