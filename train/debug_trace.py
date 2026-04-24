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
NUM_MARKET_SPACES = 27
NUM_DECISION_PHASES = 11
NUM_COO_LEVELS = 7
NUM_PLAYER_SLOTS = 5  # one-hot padded to 5
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
    labels = ["market_info"]
    labels.extend(f"company[{company_id}]" for company_id in range(NUM_COMPANIES))
    labels.extend([
        "fi",
        "global_info",
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


def _field_names(prefix: str, count: int) -> list[str]:
    return [f"{prefix}[{idx}]" for idx in range(count)]


def _token_field_labels(token_label: str) -> list[str]:
    if token_label == "market_info":
        return (
            ["attn_mask"]
            + _field_names("market_price", NUM_MARKET_SPACES)
            + _field_names("market_available", NUM_MARKET_SPACES)
        )
    if token_label.startswith("company["):
        return (
            ["attn_mask", "is_selected"]
            + ["low_price", "face_value", "high_price", "low_high_diff", "base_income", "stars"]
            + ["adj_income"]
            + ["at_removed", "at_auction", "at_revealed", "at_corp_acq"]
            + ["acq_select_synergy_delta"]
            + _field_names("owner_corp", NUM_CORPS)
            + _field_names("owner_player", NUM_PLAYER_SLOTS)
            + ["owner_fi"]
        )
    if token_label == "fi":
        return ["attn_mask", "cash", "income"] + _field_names("owned_company", NUM_COMPANIES)
    if token_label == "global_info":
        return (
            ["attn_mask"]
            + _field_names("phase", NUM_DECISION_PHASES)
            + _field_names("coo_level", NUM_COO_LEVELS)
            + ["end_card_flipped", "cards_remaining"]
            + _field_names("num_players", 3)
        )
    if token_label == "invest":
        return ["attn_mask", "consecutive_passes"]
    if token_label == "auction":
        return ["attn_mask", "min_bid_index", "min_bid_value", "is_first_bid"]
    if token_label == "dividend":
        return ["attn_mask"] + _field_names("dividend_impact", 26)
    if token_label == "issue":
        return ["attn_mask", "issue_impact"]
    if token_label == "par":
        return (
            ["attn_mask"]
            + _field_names("player_cash_required", 14)
            + _field_names("resulting_corp_cash", 14)
            + _field_names("resulting_issued_shares", 14)
        )
    if token_label == "acq_offer":
        return ["attn_mask", "offer_price_index", "offer_price", "fi_company"]
    if token_label == "acq_price":
        return ["attn_mask", "max_offset", "fi_flag", "total_synergies"]
    if token_label.startswith("corp["):
        return (
            [
                "attn_mask",
                "is_selected",
                "active",
                "in_receivership",
                "passed_acq_offer",
                "unissued_shares",
                "issued_shares",
                "bank_shares",
            ]
            + _field_names("price_index", NUM_MARKET_SPACES)
            + [
                "share_price",
                "pending_price_move",
                "cash",
                "acq_proceeds",
                "income",
                "stars",
                "raw_revenue",
                "synergy_income",
                "coo_cost",
                "ability_income",
                "acq_offer_corp",
                "dividend_remaining",
                "issue_remaining",
                "ipo_remaining",
                "buy_impact",
                "sell_impact",
            ]
            + _field_names("president_id", NUM_PLAYER_SLOTS)
            + _field_names("owned_company", NUM_COMPANIES)
        )
    if token_label.startswith("player["):
        return (
            ["attn_mask", "is_selected"]
            + _field_names("turn_order", NUM_PLAYER_SLOTS)
            + ["has_passed", "cash", "net_worth", "liquidity", "income"]
            + ["auction_high_bidder", "auction_starter", "round_trips"]
            + _field_names("owned_share", NUM_CORPS)
            + _field_names("owned_company", NUM_COMPANIES)
        )
    raise ValueError(f"unknown token label: {token_label}")


def _extract_token_buffer(state: GameState) -> tuple[np.ndarray, np.ndarray, list[str]]:
    num_players = TURN.get_num_players(state)
    if not (3 <= num_players <= 5):
        raise ValueError(f"token dump unavailable for num_players={num_players}")

    num_tokens = get_num_tokens(num_players)
    token_dim = int(TokenDataSize.TOKEN_DIM)
    buffer = np.zeros((num_tokens, token_dim), dtype=np.float32)
    get_token_data(state, buffer)
    widths = np.asarray(get_token_widths(num_players), dtype=np.int32)
    labels = _token_labels(num_players)
    return buffer, widths, labels


def _denormalize_token_values(token_label: str, row: np.ndarray) -> list[int]:
    if token_label == "market_info":
        return (
            _round_values(row[:1])
            + _round_values(row[1:28], PY_SHARE_PRICE_DIVISOR)
            + _round_values(row[28:55])
        )
    if token_label.startswith("company["):
        return (
            _round_values(row[:2])
            + _round_values(row[2:5], PY_COMPANY_PRICE_DIVISOR)
            + _round_values(row[5:6], PY_PRICE_RANGE_DIVISOR)
            + _round_values(row[6:7], PY_COMPANY_INCOME_DIVISOR)
            + _round_values(row[7:8], PY_COMPANY_STAR_DIVISOR)
            + _round_values(row[8:9], PY_COMPANY_INCOME_DIVISOR)
            + _round_values(row[9:13])
            + _round_values(row[13:14], PY_ENTITY_INCOME_DIVISOR)
            + _round_values(row[14:28])
        )
    if token_label == "fi":
        return (
            _round_values(row[:1])
            + _round_values(row[1:2], PY_CASH_DIVISOR)
            + _round_values(row[2:3], PY_ENTITY_INCOME_DIVISOR)
            + _round_values(row[3:39])
        )
    if token_label == "global_info":
        return (
            _round_values(row[:20])
            + _round_values(row[20:21], CARDS_REMAINING_DIVISOR)
            + _round_values(row[21:24])
        )
    if token_label == "invest":
        return (
            _round_values(row[:1])
            + _round_values(row[1:2], CONSECUTIVE_PASSES_DIVISOR)
        )
    if token_label == "auction":
        return (
            _round_values(row[:1])
            + _round_values(row[1:2], AUCTION_OFFSET_DIVISOR)
            + _round_values(row[2:3], PY_COMPANY_PRICE_DIVISOR)
            + _round_values(row[3:4])
        )
    if token_label == "dividend":
        return _round_values(row[:1]) + _round_values(row[1:27], PY_IMPACT_DIVISOR)
    if token_label == "issue":
        return _round_values(row[:1]) + _round_values(row[1:2], PY_IMPACT_DIVISOR)
    if token_label == "par":
        return (
            _round_values(row[:1])
            + _round_values(row[1:15], PY_CASH_DIVISOR)
            + _round_values(row[15:29], PY_CASH_DIVISOR)
            + _round_values(row[29:43], FLOAT_SHARES_MAX)
        )
    if token_label == "acq_offer":
        return (
            _round_values(row[:1])
            + _round_values(row[1:2], ACQ_OFFSET_DIVISOR)
            + _round_values(row[2:3], PY_COMPANY_PRICE_DIVISOR)
            + _round_values(row[3:4])
        )
    if token_label == "acq_price":
        return (
            _round_values(row[:1])
            + _round_values(row[1:2], PY_PRICE_RANGE_DIVISOR)
            + _round_values(row[2:3])
            + _round_values(row[3:4], PY_ENTITY_INCOME_DIVISOR)
        )
    if token_label.startswith("corp["):
        return (
            _round_values(row[:5])
            + _round_values(row[5:8], PY_SHARE_DIVISOR)
            + _round_values(row[8:35])
            + _round_values(row[35:36], PY_SHARE_PRICE_DIVISOR)
            + _round_values(row[36:37], PY_IMPACT_DIVISOR)
            + _round_values(row[37:39], PY_CASH_DIVISOR)
            + _round_values(row[39:40], PY_ENTITY_INCOME_DIVISOR)
            + _round_values(row[40:41], PY_CORP_STAR_DIVISOR)
            + _round_values(row[41:45], PY_ENTITY_INCOME_DIVISOR)
            + _round_values(row[45:49])
            + _round_values(row[49:51], PY_IMPACT_DIVISOR)
            + _round_values(row[51:56])
            + _round_values(row[56:92])
        )
    if token_label.startswith("player["):
        return (
            _round_values(row[:8])
            + _round_values(row[8:9], PY_CASH_DIVISOR)
            + _round_values(row[9:11], PY_NET_WORTH_DIVISOR)
            + _round_values(row[11:12], PY_ENTITY_INCOME_DIVISOR)
            + _round_values(row[12:15])
            + _round_values(row[15:23], PY_SHARE_DIVISOR)
            + _round_values(row[23:59])
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
    if token_label == "market_info":
        return f"attn={values[0]} prices={values[1:28]} open={_nonzero_indices(values[28:55])}"
    if token_label.startswith("company["):
        owner_corp = _one_hot_index(values[14:22])
        owner_player = _one_hot_index(values[22:27])
        return (
            f"attn={values[0]} low={values[2]} face={values[3]} "
            f"high={values[4]} range={values[5]} base_inc={values[6]} "
            f"stars={values[7]} adj_inc={values[8]} selected={values[1]} "
            f"at=[rem={values[9]},auc={values[10]},rev={values[11]},corp_acq={values[12]}] "
            f"acq_synergy={values[13]} owner_corp={owner_corp} "
            f"owner_player={owner_player} owner_fi={values[27]}"
        )
    if token_label == "fi":
        return f"attn={values[0]} cash={values[1]} income={values[2]} companies={_nonzero_indices(values[3:39])}"
    if token_label == "global_info":
        phase_idx = _one_hot_index(values[1:12])
        coo = _one_hot_index(values[12:19])
        np_idx = _one_hot_index(values[21:24])
        phase_str = DECISION_PHASE_NAMES.get(phase_idx, str(phase_idx)) if phase_idx is not None else "None"
        return (
            f"attn={values[0]} phase={phase_str} "
            f"coo={None if coo is None else coo + 1} "
            f"end_card={values[19]} cards_remaining={values[20]} "
            f"num_players={None if np_idx is None else np_idx + 3}"
        )
    if token_label == "invest":
        return f"attn={values[0]} passes={values[1]}"
    if token_label == "auction":
        return (
            f"attn={values[0]} min_bid_offset={values[1]} "
            f"min_bid={values[2]} first_bid={values[3]}"
        )
    if token_label == "dividend":
        return f"attn={values[0]} impacts={_nonzero_map(values[1:27])}"
    if token_label == "issue":
        return f"attn={values[0]} impact={values[1]}"
    if token_label == "par":
        return (
            f"attn={values[0]} player_cash={_nonzero_map(values[1:15])} "
            f"corp_cash={_nonzero_map(values[15:29])} issued={_nonzero_map(values[29:43])}"
        )
    if token_label == "acq_offer":
        return (
            f"attn={values[0]} offer_offset={values[1]} "
            f"offer_price={values[2]} fi_company={values[3]}"
        )
    if token_label == "acq_price":
        return f"attn={values[0]} max_offset={values[1]} fi_flag={values[2]} total_synergies={values[3]}"
    if token_label.startswith("corp["):
        return (
            f"attn={values[0]} selected={values[1]} active={values[2]} recv={values[3]} "
            f"passed_acq={values[4]} unissued={values[5]} issued={values[6]} bank={values[7]} "
            f"price_idx={_one_hot_index(values[8:35])} share_price={values[35]} pending_move={values[36]} "
            f"cash={values[37]} acq_proceeds={values[38]} income={values[39]} stars={values[40]} "
            f"raw_revenue={values[41]} synergy={values[42]} coo_cost={values[43]} ability={values[44]} "
            f"offer_corp={values[45]} div_rem={values[46]} issue_rem={values[47]} ipo_rem={values[48]} "
            f"buy_impact={values[49]} sell_impact={values[50]} "
            f"president={_one_hot_index(values[51:56])} companies={_nonzero_indices(values[56:92])} "
        )
    if token_label.startswith("player["):
        return (
            f"attn={values[0]} selected={values[1]} order={_one_hot_index(values[2:7])} "
            f"passed={values[7]} cash={values[8]} net_worth={values[9]} "
            f"liquidity={values[10]} income={values[11]} auc_high={values[12]} "
            f"auc_starter={values[13]} round_trip={values[14]} "
            f"shares={_nonzero_map(values[15:23])} companies={_nonzero_indices(values[23:59])}"
        )
    return str(values)


def format_token_dump(state: GameState) -> str:
    try:
        buffer, widths, labels = _extract_token_buffer(state)
    except ValueError as exc:
        return str(exc)
    return format_token_dump_from_buffer(buffer, widths, labels)


def format_token_dump_from_buffer(
    buffer: np.ndarray,
    widths: np.ndarray,
    labels: list[str],
) -> str:
    lines = ["idx | token | width | values", "--- | --- | ---: | ---"]
    for token_index in range(len(labels)):
        label = labels[token_index]
        width_int = int(widths[token_index])
        values = _denormalize_token_values(label, buffer[token_index, :width_int])
        lines.append(f"{token_index:02d} | {label} | {width_int} | {_summarize_token_values(label, values)}")
    return "\n".join(lines)


class TokenNormalizationAccumulator:
    THRESHOLDS = (1.00, 1.10, 1.25)

    def __init__(self, num_players: int):
        if not (3 <= num_players <= 5):
            raise ValueError(f"token normalization unavailable for num_players={num_players}")
        self.labels = _token_labels(num_players)
        self.widths = np.asarray(get_token_widths(num_players), dtype=np.int32)
        token_dim = int(TokenDataSize.TOKEN_DIM)
        self._mins = np.full((len(self.labels), token_dim), np.inf, dtype=np.float64)
        self._maxs = np.full((len(self.labels), token_dim), -np.inf, dtype=np.float64)
        self._sums = np.zeros((len(self.labels), token_dim), dtype=np.float64)
        self.samples = 0

    def add_state(self, state: GameState) -> np.ndarray:
        buffer, widths, labels = _extract_token_buffer(state)
        assert labels == self.labels
        if not np.array_equal(widths, self.widths):
            raise ValueError("token width mismatch while accumulating normalization stats")

        for token_index, width in enumerate(self.widths):
            width_int = int(width)
            row = buffer[token_index, :width_int].astype(np.float64, copy=False)
            self._mins[token_index, :width_int] = np.minimum(self._mins[token_index, :width_int], row)
            self._maxs[token_index, :width_int] = np.maximum(self._maxs[token_index, :width_int], row)
            self._sums[token_index, :width_int] += row

        self.samples += 1
        return buffer

    def _iter_field_stats(self):
        for token_index in range(len(self.labels)):
            token_label = self.labels[token_index]
            field_labels = _token_field_labels(token_label)
            width_int = int(self.widths[token_index])
            if len(field_labels) != width_int:
                raise ValueError(
                    f"field label width mismatch for {token_label}: {len(field_labels)} != {width_int}"
                )
            for field_index in range(width_int):
                min_val = float(self._mins[token_index, field_index])
                max_val = float(self._maxs[token_index, field_index])
                avg = float(self._sums[token_index, field_index] / self.samples)
                yield token_label, field_labels[field_index], min_val, max_val, avg

    def format_report(self) -> str:
        if self.samples == 0:
            return "token normalization report unavailable (no token states captured)"

        field_stats = list(self._iter_field_stats())
        lines = [f"Captured {self.samples} decision-state token dumps.", "", "### Threshold Summary", ""]
        lines.extend([
            "threshold | fields_exceeding | worst_abs | worst_field",
            "---: | ---: | ---: | ---",
        ])
        for threshold in self.THRESHOLDS:
            offenders: list[tuple[float, str, str]] = []
            for token_label, field_label, min_val, max_val, _avg in field_stats:
                peak_abs = max(abs(min_val), abs(max_val))
                if peak_abs > threshold:
                    offenders.append((peak_abs, token_label, field_label))
            if offenders:
                peak_abs, token_label, field_label = max(offenders, key=lambda item: item[0])
                worst_field = f"{token_label} | {field_label}"
            else:
                peak_abs = 0.0
                worst_field = "-"
            lines.append(f"> {threshold:.2f} | {len(offenders)} | {peak_abs:+.4f} | {worst_field}")

        lines.extend(["", "token | field | min | max | avg", "--- | --- | ---: | ---: | ---:"])
        for token_label, field_label, min_val, max_val, avg in field_stats:
            lines.append(
                f"{token_label} | {field_label} | {min_val:+.4f} | {max_val:+.4f} | {avg:+.4f}"
            )
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
