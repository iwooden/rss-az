"""
Utilities for replaying 18xx.games Rolling Stock Stars games through our Cython engine.

Provides:
- JSON parsing and action filtering
- Name-to-ID mappings for companies/corps/players
- Action translation from 18xx format to our engine's action indices
"""

import json
import re
from dataclasses import dataclass, field
from typing import Iterator, Optional
from pathlib import Path

from data import (
    COMPANY_NAMES, COMPANY_NAME_TO_ID,
    CORP_NAMES, CORP_NAME_TO_ID,
    py_get_company_face_value, py_get_company_low_price, py_get_par_price
)
from actions import get_action_layout


# =============================================================================
# CONSTANTS
# =============================================================================

# Phase constants (matching state.pyx)
PHASE_INVEST = 0
PHASE_BID_IN_AUCTION = 1
PHASE_WRAP_UP = 2
PHASE_ACQUISITION = 3
PHASE_CLOSING = 4
PHASE_INCOME = 5
PHASE_DIVIDENDS = 6
PHASE_END_CARD = 7
PHASE_ISSUE_SHARES = 8
PHASE_IPO = 9
PHASE_GAME_OVER = 10

# Phase name mapping for net worth table
PHASE_ABBREV = {
    PHASE_INVEST: "INV",
    PHASE_WRAP_UP: "WRP",
    PHASE_ACQUISITION: "ACQ",
    PHASE_CLOSING: "CLO",
    PHASE_INCOME: "INC",
    PHASE_DIVIDENDS: "DIV",
    PHASE_END_CARD: "END",
    PHASE_ISSUE_SHARES: "ISS",
    PHASE_IPO: "IPO",
    PHASE_GAME_OVER: "END",
}

# Action types to skip during replay (not actual game moves)
SKIP_ACTION_TYPES = {"undo", "redo", "program_share_pass"}

# Number of par prices
NUM_PAR_PRICES = 14
ALL_PAR_PRICES = [py_get_par_price(i) for i in range(NUM_PAR_PRICES)]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Player18xx:
    """Player info from 18xx.games JSON."""
    id: int
    name: str
    seat: int  # 0-indexed position in players array


@dataclass
class ParsedAction:
    """Parsed action from 18xx.games JSON."""
    action_type: str           # Original 18xx type: bid, pass, buy_shares, etc.
    entity_id: int | str       # Player ID or entity name
    entity_type: str           # "player", "company", "corporation"
    company: Optional[str] = None
    corporation: Optional[str] = None
    price: Optional[int] = None
    shares: list = field(default_factory=list)
    amount: Optional[int] = None      # For dividends
    accept: Optional[bool] = None     # For respond actions
    auto_actions: list = field(default_factory=list)  # Nested auto-pass actions
    original_id: int = 0              # Action ID from JSON for debugging
    share_price_str: Optional[str] = None  # For par actions: "price,row,col"


# =============================================================================
# PARSER
# =============================================================================

class Game18xxParser:
    """
    Parse and iterate through 18xx.games JSON.

    Filters out undo/redo actions and yields ParsedAction objects.
    """

    def __init__(self, json_path: str, md_path: Optional[str] = None):
        """
        Initialize parser with game data.

        Args:
            json_path: Path to 18xx.games JSON file
            md_path: Optional path to markdown file with game log
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.md_path = md_path
        self._players: list[Player18xx] = []
        self._player_id_to_seat: dict[int, int] = {}
        self._parse_players()

        # Cache deck order from md file if available
        self._deck_order: Optional[list[str]] = None
        self._initial_auction: Optional[list[str]] = None

    def _parse_players(self):
        """Parse player list from JSON."""
        for seat, p in enumerate(self.data["players"]):
            player = Player18xx(id=p["id"], name=p["name"], seat=seat)
            self._players.append(player)
            self._player_id_to_seat[p["id"]] = seat

    @property
    def num_players(self) -> int:
        return len(self._players)

    @property
    def players(self) -> list[Player18xx]:
        return self._players

    def get_player_seat(self, player_id: int) -> int:
        """Get seat index (0-based) for player ID."""
        return self._player_id_to_seat[player_id]

    def get_final_result(self) -> dict[int, int]:
        """Get final net worth by player ID."""
        return {int(k): v for k, v in self.data["result"].items()}

    def get_initial_auction_companies(self) -> list[str]:
        """
        Get initial auction companies from markdown file.

        Returns list of company names in auction at game start.
        """
        if self._initial_auction is not None:
            return self._initial_auction

        if self.md_path is None:
            raise ValueError("No markdown path provided")

        with open(self.md_path, 'r') as f:
            content = f.read()

        # Parse "Initial companies available for auction:" line
        match = re.search(r'Initial companies available for auction:\s*\n([A-Z, ]+)', content)
        if match:
            companies = [c.strip() for c in match.group(1).split(',')]
            self._initial_auction = companies
            return companies

        raise ValueError("Could not find initial auction companies in markdown")

    def get_deck_order(self) -> list[str]:
        """
        Get deck order from markdown game log.

        Parses "X revealed from deck" messages to determine deck order.
        Returns list of company names in reveal order (first revealed = first in list).
        """
        if self._deck_order is not None:
            return self._deck_order

        if self.md_path is None:
            raise ValueError("No markdown path provided")

        with open(self.md_path, 'r') as f:
            content = f.read()

        # Find all "X revealed from deck" patterns
        reveals = re.findall(r'(\w+) revealed from deck', content)
        self._deck_order = reveals
        return reveals

    def iterate_actions(self) -> Iterator[ParsedAction]:
        """
        Iterate through game actions, properly handling undo/redo.

        Yields ParsedAction objects in game order.
        Also yields nested auto_actions from pass actions.

        Undo/redo handling: When we see "undo", we remove the last effective
        action. When we see "redo", we re-add the last undone action. This
        ensures we only yield actions that are part of the final game state.
        """
        # First pass: build list of effective actions with undo/redo applied
        effective_actions: list[dict] = []
        undo_stack: list[dict] = []

        for action in self.data["actions"]:
            action_type = action["type"]

            if action_type == "undo":
                # Pop last effective action onto undo stack
                if effective_actions:
                    undone = effective_actions.pop()
                    undo_stack.append(undone)
            elif action_type == "redo":
                # Re-add from undo stack
                if undo_stack:
                    redone = undo_stack.pop()
                    effective_actions.append(redone)
            elif action_type == "program_share_pass":
                # Skip auto-pass configuration actions
                continue
            else:
                # Normal action - add to effective list, clear undo stack
                effective_actions.append(action)
                undo_stack.clear()

        # Second pass: yield parsed actions
        for action in effective_actions:
            parsed = self._parse_action(action)
            yield parsed

            # Yield auto_actions if present (nested pass actions)
            if parsed.auto_actions:
                for auto in parsed.auto_actions:
                    if auto.get("type") not in SKIP_ACTION_TYPES:
                        yield self._parse_action(auto)

    def _parse_action(self, action: dict) -> ParsedAction:
        """Parse a single action dict into ParsedAction."""
        action_type = action["type"]
        entity = action.get("entity")
        entity_type = action.get("entity_type", "player")

        # Handle accept field (comes as string "true"/"false")
        accept = None
        if "accept" in action:
            accept = action["accept"].lower() == "true"

        return ParsedAction(
            action_type=action_type,
            entity_id=entity,
            entity_type=entity_type,
            company=action.get("company"),
            corporation=action.get("corporation"),
            price=action.get("price"),
            shares=action.get("shares", []),
            amount=action.get("amount"),
            accept=accept,
            auto_actions=action.get("auto_actions", []),
            original_id=action.get("id", 0),
            share_price_str=action.get("share_price"),  # For par actions
        )


# =============================================================================
# NET WORTH TABLE PARSER
# =============================================================================

def parse_net_worth_table(md_path: str) -> dict[tuple[str, int], dict[int, int]]:
    """
    Parse net worth table from markdown file.

    Returns dict mapping (phase_abbrev, turn) -> {seat: net_worth}

    The table format is:
        	chadamir	reveler	d_choo	CardboardBits
        INV 14	$206	$151	$149	$106
        ...
    """
    with open(md_path, 'r') as f:
        content = f.read()

    # Find the net worth table section
    match = re.search(r'# Net worth totals.*?\n\t(.+)\n((?:[A-Z]{3} \d+\t.+\n)+)', content)
    if not match:
        raise ValueError("Could not find net worth table in markdown")

    # Parse header to get player order
    header = match.group(1)
    player_names = [n.strip() for n in header.split('\t')]

    # Parse player names from JSON to get seat mapping
    # We need to map names -> seats based on JSON player order
    # For this game: CardboardBits=0, chadamir=1, d_choo=2, reveler=3
    # But we need to read from JSON to be generic

    # Parse data rows
    result = {}
    data_lines = match.group(2).strip().split('\n')

    for line in data_lines:
        parts = line.split('\t')
        phase_turn = parts[0]  # e.g., "INV 14"
        values = parts[1:]

        # Parse phase and turn
        phase_abbrev, turn_str = phase_turn.split()
        turn = int(turn_str)

        # Parse net worth values (remove $ and convert to int)
        net_worths = {}
        for i, val in enumerate(values):
            nw = int(val.replace('$', ''))
            # The column order in the table matches how we need to map to seats
            # We'll map by name when we have the parser
            net_worths[player_names[i]] = nw

        result[(phase_abbrev, turn)] = net_worths

    return result


# =============================================================================
# ACTION TRANSLATOR
# =============================================================================

class ActionTranslator:
    """
    Translate 18xx.games actions to our engine's action indices.

    Handles the mapping between 18xx's flexible action system and our
    ordered/optimized action space.
    """

    def __init__(self, num_players: int):
        """
        Initialize translator.

        Args:
            num_players: Number of players in game
        """
        self.num_players = num_players
        self.layout = get_action_layout(num_players)

        # Build face value lookup for auction slot mapping
        self._company_face_values = {
            cid: py_get_company_face_value(cid)
            for cid in range(len(COMPANY_NAMES))
        }

    def get_auction_slot(self, company_name: str, available_company_ids: list[int]) -> int:
        """
        Get auction slot for a company based on face value ordering.

        Our engine orders auction slots by ascending face value.
        Slot 0 = lowest face value available company, etc.

        Args:
            company_name: Name of company being bid on
            available_company_ids: List of company IDs currently available for auction

        Returns:
            Slot index (0-based)
        """
        company_id = COMPANY_NAME_TO_ID[company_name]

        # Sort available companies by face value
        sorted_available = sorted(
            available_company_ids,
            key=lambda cid: self._company_face_values[cid]
        )

        return sorted_available.index(company_id)

    def translate_start_auction(self, company_name: str, bid_price: int,
                                 available_company_ids: list[int]) -> int:
        """
        Translate start auction action to action index.

        Args:
            company_name: Company to auction
            bid_price: Opening bid amount
            available_company_ids: Currently available companies

        Returns:
            Action index
        """
        company_id = COMPANY_NAME_TO_ID[company_name]
        face_value = py_get_company_face_value(company_id)
        bid_offset = bid_price - face_value

        slot = self.get_auction_slot(company_name, available_company_ids)

        # Action index = auction_base + slot * 20 + bid_offset
        return self.layout['auction_base'] + slot * 20 + bid_offset

    def translate_raise_bid(self, new_price: int, auction_company_id: int) -> int:
        """
        Translate raise bid action.

        Args:
            new_price: New bid amount
            auction_company_id: Company being auctioned

        Returns:
            Action index
        """
        face_value = py_get_company_face_value(auction_company_id)
        # Raise bid offset is relative to face+1
        # bid_offset 0 = face+1, bid_offset 1 = face+2, etc.
        # So bid_offset = new_price - face_value - 1
        bid_offset = new_price - face_value - 1

        return self.layout['raise_bid_base'] + bid_offset

    def translate_leave_auction(self) -> int:
        """Translate leave auction action."""
        return self.layout['leave_auction']

    def translate_pass_invest(self) -> int:
        """Translate pass in invest phase."""
        return self.layout['pass_invest']

    def translate_buy_share(self, corp_name: str) -> int:
        """Translate buy share action."""
        corp_id = CORP_NAME_TO_ID[corp_name]
        return self.layout['buy_share_base'] + corp_id

    def translate_sell_share(self, corp_name: str) -> int:
        """Translate sell share action."""
        corp_id = CORP_NAME_TO_ID[corp_name]
        return self.layout['sell_share_base'] + corp_id

    def translate_dividend(self, amount: int) -> int:
        """Translate dividend action."""
        return self.layout['dividend_base'] + amount

    def translate_issue(self) -> int:
        """Translate issue share action."""
        return self.layout['issue_action']

    def translate_issue_pass(self) -> int:
        """Translate pass issue share."""
        return self.layout['issue_pass']

    def translate_ipo(self, corp_name: str, par_price: int, star_tier: int) -> int:
        """
        Translate IPO action.

        Args:
            corp_name: Target corporation name
            par_price: Par price chosen
            star_tier: Star tier of the converting company (1-5)

        Returns:
            Action index
        """
        corp_id = CORP_NAME_TO_ID[corp_name]

        # Find par_slot - the index among valid par prices for this star tier
        par_slot = self._get_par_slot(par_price, star_tier)

        return self.layout['ipo_base'] + corp_id * 8 + par_slot

    def _get_par_slot(self, par_price: int, star_tier: int) -> int:
        """
        Get par slot index for a given par price and star tier.

        Par slots are ordered among valid par prices for the star tier.
        """
        # Valid par price ranges by star tier (from data.pyx PAR_PRICE_VALID)
        valid_ranges = {
            1: [10, 11, 12, 13, 14],        # Reds
            2: [10, 11, 12, 13, 14, 16, 18, 20],  # Oranges
            3: [16, 18, 20, 22, 24, 27],    # Yellows
            4: [22, 24, 27, 30, 33, 37],    # Greens
            5: [30, 33, 37],                # Blues
        }

        valid_prices = valid_ranges.get(star_tier, [])
        if par_price in valid_prices:
            return valid_prices.index(par_price)

        raise ValueError(f"Invalid par price {par_price} for star tier {star_tier}")

    def translate_ipo_pass(self) -> int:
        """Translate pass IPO action."""
        return self.layout['ipo_pass']

    def translate_close(self) -> int:
        """Translate close company action."""
        return self.layout['close_action']

    def translate_close_pass(self) -> int:
        """Translate pass closing action."""
        return self.layout['close_pass']

    def translate_acquisition_price(self, price: int, low_price: int) -> int:
        """
        Translate acquisition at specific price.

        Args:
            price: Acquisition price
            low_price: Company's low price

        Returns:
            Action index
        """
        price_offset = price - low_price
        return self.layout['acq_price_base'] + price_offset

    def translate_acquisition_pass(self) -> int:
        """Translate pass acquisition action."""
        return self.layout['acq_pass']

    def translate_acquisition_fi_high(self) -> int:
        """Translate FI acquisition at high price."""
        return self.layout['acq_fi_high']

    def translate_acquisition_fi_face(self) -> int:
        """Translate FI acquisition at face value (OS only)."""
        return self.layout['acq_fi_face']


# =============================================================================
# ACQUISITION STATE MACHINE
# =============================================================================

@dataclass
class PendingAcquisition:
    """An acquisition offer from 18xx that hasn't been matched yet."""
    company_name: str
    corp_name: str
    price: int
    from_fi: bool = False  # True if from Foreign Investor


class AcquisitionMatcher:
    """
    Match 18xx acquisitions to our engine's sequential offering.

    18xx allows players to propose acquisitions in any order.
    Our engine presents potential acquisitions one at a time in a fixed order.
    This class buffers 18xx acquisitions and matches them as our engine offers.
    """

    def __init__(self):
        self.pending: list[PendingAcquisition] = []
        self.rejected: set[tuple[str, str]] = set()  # (company, corp) pairs rejected

    def add_offer(self, company: str, corp: str, price: int, from_fi: bool = False):
        """Add an acquisition offer from 18xx action stream."""
        self.pending.append(PendingAcquisition(
            company_name=company,
            corp_name=corp,
            price=price,
            from_fi=from_fi,
        ))

    def add_rejection(self, company: str, corp: str):
        """Record that an acquisition was rejected."""
        self.rejected.add((company, corp))

    def find_match(self, offered_company: str, offered_corp: str) -> Optional[PendingAcquisition]:
        """
        Find a pending acquisition matching what the engine is offering.

        Returns the matching PendingAcquisition or None if should pass.
        """
        for i, pending in enumerate(self.pending):
            if pending.company_name == offered_company and pending.corp_name == offered_corp:
                return self.pending.pop(i)

        # Check if this was explicitly rejected
        if (offered_company, offered_corp) in self.rejected:
            self.rejected.discard((offered_company, offered_corp))
            return None

        # No match found - pass on this acquisition
        return None

    def has_pending(self) -> bool:
        """Check if there are still pending acquisitions."""
        return len(self.pending) > 0

    def has_pending_offers(self) -> bool:
        """Alias for has_pending() - check if there are buffered offers."""
        return self.has_pending()

    def get_offer_price(self, company: str, corp: str) -> Optional[int]:
        """
        Get the price for a buffered offer matching company/corp names.

        Does NOT remove the offer - just looks up the price.

        Returns:
            Price if found, None otherwise
        """
        for pending in self.pending:
            if pending.company_name == company and pending.corp_name == corp:
                return pending.price
        return None

    def get_matching_offer(self, target_company_id: int, target_corp_id: int) -> Optional[PendingAcquisition]:
        """
        Find and remove a pending offer matching engine's current acquisition.

        Args:
            target_company_id: Company ID the engine is offering to acquire
            target_corp_id: Corporation ID the engine wants to acquire for

        Returns:
            Matching PendingAcquisition if found (removed from buffer), None otherwise
        """
        # Convert IDs to names for matching
        target_company_name = None
        target_corp_name = None

        for name, cid in COMPANY_NAME_TO_ID.items():
            if cid == target_company_id:
                target_company_name = name
                break

        for name, cid in CORP_NAME_TO_ID.items():
            if cid == target_corp_id:
                target_corp_name = name
                break

        if target_company_name is None or target_corp_name is None:
            return None

        for i, pending in enumerate(self.pending):
            if pending.company_name == target_company_name and pending.corp_name == target_corp_name:
                return self.pending.pop(i)

        return None

    def clear(self):
        """Clear all pending state (call at phase end)."""
        self.pending.clear()
        self.rejected.clear()


# =============================================================================
# CLOSING STATE MACHINE
# =============================================================================

@dataclass
class PendingClose:
    """A company close action from 18xx that hasn't been matched yet."""
    company_name: str
    corp_name: Optional[str] = None  # Corp that owns it, if any


class ClosingMatcher:
    """
    Match 18xx close actions to our engine's sequential offering.

    Similar to AcquisitionMatcher - 18xx allows arbitrary order,
    our engine presents one at a time.
    """

    def __init__(self):
        self.pending: list[PendingClose] = []

    def add_close(self, company: str, corp: Optional[str] = None):
        """Add a close action from 18xx."""
        self.pending.append(PendingClose(company_name=company, corp_name=corp))

    def should_close(self, offered_company: str) -> bool:
        """Check if we should close the offered company."""
        for i, pending in enumerate(self.pending):
            if pending.company_name == offered_company:
                self.pending.pop(i)
                return True
        return False

    def has_pending(self) -> bool:
        return len(self.pending) > 0

    def clear(self):
        self.pending.clear()
