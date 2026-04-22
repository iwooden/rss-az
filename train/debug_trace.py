"""Human-readable state and action rendering for analysis traces.

This restores the old debug-trace style output used by the pre-refactor
``train.analyze_game`` while adapting it to the refactored phase-local action
space and the current ACQ_OFFER / merged IPO flow.
"""

from __future__ import annotations

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
    decode_action_py,
)
from core.data import (
    ALL_PAR_PRICES,
    COMPANY_NAMES,
    CORP_NAMES,
    CorpIndices,
    GamePhases,
    PY_CASH_DIVISOR,
    PY_COMPANY_INCOME_DIVISOR,
    PY_COMPANY_PRICE_DIVISOR,
    PY_COMPANY_STAR_DIVISOR,
    PY_CORP_STAR_DIVISOR,
    PY_ENTITY_INCOME_DIVISOR,
    PY_IMPACT_DIVISOR,
    PY_NET_WORTH_DIVISOR,
    PY_SHARE_DIVISOR,
    PY_SHARE_PRICE_DIVISOR,
)
from core.state import GameState
from core.token_data import TokenDataSize, get_num_tokens, get_token_data, get_token_widths
from entities.company import COMPANIES, CompanyLocation
from entities.corp import CORPS
from entities.deck import DECK
from entities.fi import FI
from entities.player import PLAYERS
from entities.turn import TURN

NUM_COMPANIES = 36
NUM_CORPS = 8
NUM_STATIC_TOKENS = 1 + NUM_COMPANIES
PY_PRICE_RANGE_DIVISOR = 51.0
FLOAT_SHARES_MAX = 4.0
CARDS_REMAINING_DIVISOR = float(NUM_COMPANIES)
CONSECUTIVE_PASSES_DIVISOR = 5.0
AUCTION_OFFSET_DIVISOR = 15.0
ACQ_OFFSET_DIVISOR = 51.0

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


def _round_values(values: np.ndarray, scale: float = 1.0) -> list[int]:
    return [int(round(float(value) * scale)) for value in values]


def _token_labels(num_players: int) -> list[str]:
    labels = ["market_prices"]
    labels.extend(f"company[{company_id}]" for company_id in range(NUM_COMPANIES))
    labels.extend([
        "market_availability",
        "company_loc_removed",
        "company_loc_auction",
        "company_loc_revealed",
        "company_loc_corp_acq",
        "company_adj_income",
        "fi",
        "active_player",
        "active_corp",
        "active_company",
        "phase",
        "num_players",
        "game_progress",
        "invest",
        "auction",
        "dividend",
        "issue",
        "par",
        "acq_offer",
        "acq_price",
    ])
    labels.extend(f"corp[{corp_id}]" for corp_id in range(NUM_CORPS))
    labels.extend(f"player[{player_id}]" for player_id in range(num_players))
    return labels


def _denormalize_token_values(token_label: str, row: np.ndarray) -> list[int]:
    if token_label == "market_prices":
        return _round_values(row[:27], PY_SHARE_PRICE_DIVISOR)
    if token_label.startswith("company["):
        return (
            _round_values(row[:36])
            + _round_values(row[36:39], PY_COMPANY_PRICE_DIVISOR)
            + _round_values(row[39:40], PY_PRICE_RANGE_DIVISOR)
            + _round_values(row[40:41], PY_COMPANY_INCOME_DIVISOR)
            + _round_values(row[41:42], PY_COMPANY_STAR_DIVISOR)
            + _round_values(row[42:78], PY_COMPANY_INCOME_DIVISOR)
        )
    if token_label == "market_availability":
        return _round_values(row[:27])
    if token_label.startswith("company_loc_"):
        return _round_values(row[:36])
    if token_label == "company_adj_income":
        return _round_values(row[:36], PY_COMPANY_INCOME_DIVISOR)
    if token_label == "fi":
        return (
            _round_values(row[:1], PY_CASH_DIVISOR)
            + _round_values(row[1:2], PY_ENTITY_INCOME_DIVISOR)
            + _round_values(row[2:38])
        )
    if token_label == "active_player":
        return _round_values(row[:5])
    if token_label == "active_corp":
        return _round_values(row[:8])
    if token_label == "active_company":
        return _round_values(row[:36])
    if token_label == "phase":
        return _round_values(row[:11])
    if token_label == "num_players":
        return _round_values(row[:3])
    if token_label == "game_progress":
        return (
            _round_values(row[:8])
            + _round_values(row[8:9], CARDS_REMAINING_DIVISOR)
        )
    if token_label == "invest":
        return (
            _round_values(row[:1], CONSECUTIVE_PASSES_DIVISOR)
            + _round_values(row[1:17], PY_IMPACT_DIVISOR)
        )
    if token_label == "auction":
        return (
            _round_values(row[:1], AUCTION_OFFSET_DIVISOR)
            + _round_values(row[1:2], PY_COMPANY_PRICE_DIVISOR)
            + _round_values(row[2:13])
        )
    if token_label == "dividend":
        return _round_values(row[:26], PY_IMPACT_DIVISOR) + _round_values(row[26:34])
    if token_label == "issue":
        return _round_values(row[:1], PY_IMPACT_DIVISOR) + _round_values(row[1:9])
    if token_label == "par":
        return (
            _round_values(row[:14], PY_CASH_DIVISOR)
            + _round_values(row[14:28], PY_CASH_DIVISOR)
            + _round_values(row[28:42], FLOAT_SHARES_MAX)
            + _round_values(row[42:50])
        )
    if token_label == "acq_offer":
        return (
            _round_values(row[:1], ACQ_OFFSET_DIVISOR)
            + _round_values(row[1:2], PY_COMPANY_PRICE_DIVISOR)
            + _round_values(row[2:11])
        )
    if token_label == "acq_price":
        return (
            _round_values(row[:1], PY_PRICE_RANGE_DIVISOR)
            + _round_values(row[1:2])
            + _round_values(row[2:3], PY_ENTITY_INCOME_DIVISOR)
        )
    if token_label.startswith("corp["):
        return (
            _round_values(row[:11])
            + _round_values(row[11:14], PY_SHARE_DIVISOR)
            + _round_values(row[14:41])
            + _round_values(row[41:42], PY_SHARE_PRICE_DIVISOR)
            + _round_values(row[42:43], PY_IMPACT_DIVISOR)
            + _round_values(row[43:45], PY_CASH_DIVISOR)
            + _round_values(row[45:46], PY_ENTITY_INCOME_DIVISOR)
            + _round_values(row[46:47], PY_CORP_STAR_DIVISOR)
            + _round_values(row[47:51], PY_ENTITY_INCOME_DIVISOR)
            + _round_values(row[51:56])
            + _round_values(row[56:92])
        )
    if token_label.startswith("player["):
        return (
            _round_values(row[:11])
            + _round_values(row[11:12], PY_CASH_DIVISOR)
            + _round_values(row[12:14], PY_NET_WORTH_DIVISOR)
            + _round_values(row[14:15], PY_ENTITY_INCOME_DIVISOR)
            + _round_values(row[15:23], PY_SHARE_DIVISOR)
            + _round_values(row[23:24])
            + _round_values(row[24:40], PY_SHARE_DIVISOR)
            + _round_values(row[40:84])
        )
    raise ValueError(f"unknown token label: {token_label}")


def _nonzero_indices(values: list[int]) -> list[int]:
    return [idx for idx, value in enumerate(values) if value != 0]


def _nonzero_map(values: list[int]) -> dict[int, int]:
    return {idx: value for idx, value in enumerate(values) if value != 0}


def _one_hot_index(values: list[int]) -> int | None:
    indices = _nonzero_indices(values)
    return indices[0] if len(indices) == 1 else None


def _summarize_token_values(token_label: str, values: list[int]) -> str:
    if token_label == "market_prices":
        return str(values)
    if token_label.startswith("company["):
        synergies = _nonzero_map(values[42:78])
        return (
            f"id={_one_hot_index(values[:36])} low={values[36]} face={values[37]} "
            f"high={values[38]} range={values[39]} income={values[40]} "
            f"stars={values[41]} synergies={synergies}"
        )
    if token_label == "market_availability":
        return f"open={_nonzero_indices(values)}"
    if token_label.startswith("company_loc_"):
        return f"companies={_nonzero_indices(values)}"
    if token_label == "company_adj_income":
        return str(values)
    if token_label == "fi":
        return f"cash={values[0]} income={values[1]} companies={_nonzero_indices(values[2:38])}"
    if token_label == "active_player":
        return f"player={_one_hot_index(values[:5])}"
    if token_label == "active_corp":
        return f"corp={_one_hot_index(values[:8])}"
    if token_label == "active_company":
        return f"company={_one_hot_index(values[:36])}"
    if token_label == "phase":
        phase_idx = _one_hot_index(values[:11])
        return f"phase={DECISION_PHASE_NAMES.get(phase_idx, phase_idx)}"
    if token_label == "num_players":
        player_idx = _one_hot_index(values[:3])
        return f"num_players={None if player_idx is None else player_idx + 3}"
    if token_label == "game_progress":
        coo = _one_hot_index(values[:7])
        return (
            f"coo={None if coo is None else coo + 1} "
            f"end_card={values[7]} cards_remaining={values[8]}"
        )
    if token_label == "invest":
        return (
            f"passes={values[0]} buy_impacts={_nonzero_map(values[1:9])} "
            f"sell_impacts={_nonzero_map(values[9:17])}"
        )
    if token_label == "auction":
        return (
            f"min_bid_offset={values[0]} min_bid={values[1]} first_bid={values[2]} "
            f"high_bidder={_one_hot_index(values[3:8])} starter={_one_hot_index(values[8:13])}"
        )
    if token_label == "dividend":
        return f"impacts={_nonzero_map(values[:26])} remaining={_nonzero_indices(values[26:34])}"
    if token_label == "issue":
        return f"impact={values[0]} remaining={_nonzero_indices(values[1:9])}"
    if token_label == "par":
        return (
            f"player_cash={_nonzero_map(values[:14])} corp_cash={_nonzero_map(values[14:28])} "
            f"issued={_nonzero_map(values[28:42])} remaining={_nonzero_indices(values[42:50])}"
        )
    if token_label == "acq_offer":
        return (
            f"offer_offset={values[0]} offer_price={values[1]} offer_corp={_one_hot_index(values[2:10])} "
            f"fi_company={values[10]}"
        )
    if token_label == "acq_price":
        return f"max_offset={values[0]} fi_flag={values[1]} total_synergies={values[2]}"
    if token_label.startswith("corp["):
        return (
            f"id={_one_hot_index(values[:8])} active={values[8]} recv={values[9]} passed_acq={values[10]} "
            f"unissued={values[11]} issued={values[12]} bank={values[13]} "
            f"price_idx={_one_hot_index(values[14:41])} share_price={values[41]} pending_move={values[42]} "
            f"cash={values[43]} acq_proceeds={values[44]} income={values[45]} stars={values[46]} "
            f"raw_revenue={values[47]} synergy={values[48]} coo_cost={values[49]} ability={values[50]} "
            f"president={_one_hot_index(values[51:56])} companies={_nonzero_indices(values[56:92])}"
        )
    if token_label.startswith("player["):
        return (
            f"id={_one_hot_index(values[:5])} order={_one_hot_index(values[5:10])} passed={values[10]} "
            f"cash={values[11]} net_worth={values[12]} liquidity={values[13]} income={values[14]} "
            f"shares={_nonzero_map(values[15:23])} round_trip={values[23]} "
            f"buys={_nonzero_map(values[24:32])} sells={_nonzero_map(values[32:40])} "
            f"presidencies={_nonzero_indices(values[40:48])} companies={_nonzero_indices(values[48:84])}"
        )
    return str(values)


def format_token_dump(state: GameState, *, skip_static_tokens: bool = False) -> str:
    num_players = TURN.get_num_players(state)
    if not (3 <= num_players <= 5):
        return f"token dump unavailable for num_players={num_players}"

    num_tokens = get_num_tokens(num_players)
    token_dim = int(TokenDataSize.TOKEN_DIM)
    buffer = np.zeros((num_tokens, token_dim), dtype=np.float32)
    get_token_data(state, buffer)
    widths = get_token_widths(num_players)
    labels = _token_labels(num_players)

    start_index = NUM_STATIC_TOKENS if skip_static_tokens else 0
    lines = ["idx | token | width | values", "--- | --- | ---: | ---"]
    for token_index in range(start_index, len(labels)):
        label = labels[token_index]
        width_int = int(widths[token_index])
        values = _denormalize_token_values(label, buffer[token_index, :width_int])
        lines.append(f"{token_index:02d} | {label} | {width_int} | {_summarize_token_values(label, values)}")
    return "\n".join(lines)


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
