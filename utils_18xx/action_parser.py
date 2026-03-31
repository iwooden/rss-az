"""Parser for 18xx game JSON actions -> our engine action indices.

Converts the action stream from 18xx.games JSON exports into the integer
action indices used by our Cython game engine. Handles undo/redo processing,
auto-action flattening, program_* auto-pass tracking, and per-phase mapping.

Also provides shared utilities for synchronizing engine state with 18xx games:
deck/offering override and simple-phase action dispatch.
"""

from __future__ import annotations

from core.data import (
    COMPANY_NAME_TO_ID,
    CORP_NAME_TO_ID,
    COMPANY_NAMES,
    GamePhases,
    get_company_face_value,
    get_par_price,
)
from core.actions import get_action_layout
from core.state import GameState
from entities.company import COMPANIES, CompanyLocation
from entities.deck import DECK

LOC_AUCTION = CompanyLocation.LOC_AUCTION
LOC_REVEALED = CompanyLocation.LOC_REVEALED

# Phase constants — shared by game_session, replay_harness, etc.
PHASE_INVEST = GamePhases.PHASE_INVEST
PHASE_BID = GamePhases.PHASE_BID_IN_AUCTION
PHASE_WRAP_UP = GamePhases.PHASE_WRAP_UP
PHASE_ACQ = GamePhases.PHASE_ACQUISITION
PHASE_CLOSING = GamePhases.PHASE_CLOSING
PHASE_INCOME = GamePhases.PHASE_INCOME
PHASE_DIVIDENDS = GamePhases.PHASE_DIVIDENDS
PHASE_END_CARD = GamePhases.PHASE_END_CARD
PHASE_ISSUE = GamePhases.PHASE_ISSUE_SHARES
PHASE_IPO = GamePhases.PHASE_IPO
PHASE_PAR = GamePhases.PHASE_PAR
PHASE_GAME_OVER = GamePhases.PHASE_GAME_OVER


# =============================================================================
# ACTION LAYOUT WRAPPER
# =============================================================================

class ActionLayout:
    """
    Wraps the layout dict returned by get_action_layout() to provide
    attribute-style access (layout.auction_base etc.).

    Created with ActionLayout(num_players).
    """

    def __init__(self, num_players: int):
        self._d = get_action_layout(num_players)
        self.num_players = num_players

    def __getattr__(self, name: str):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(f"ActionLayout has no attribute '{name}'")


# =============================================================================
# CONSTANTS
# =============================================================================

# 18xx action type sets (used to identify which engine phase an action belongs to)
INVEST_ACTIONS = {'bid', 'buy_shares', 'sell_shares', 'pass'}
BID_ACTIONS = {'bid', 'pass'}       # When in our BID_IN_AUCTION phase
IPO_ACTIONS = {'par', 'pass'}
DIVIDEND_ACTIONS = {'dividend'}
ISSUE_ACTIONS = {'sell_shares', 'pass'}  # sell_shares by corp entity = issue

# Action types that carry no game-state meaning and should be dropped entirely.
# Undo/redo are resolved by the Ruby extractor (snapshots are truncated/replayed),
# so the Python side never sees them.  program_* are auto-pass convenience
# features; message is chat.
SKIP_ACTIONS = {
    'program_share_pass',
    'program_close_pass',
    'program_disable',
    'message',
    'undo',
    'redo',
}

AUCTION_CAP = 15  # Maximum bid offset above face value (per slot)


# =============================================================================
# 1. FILTER ACTIONS
# =============================================================================

def filter_actions(actions: list, committed_ids: set | None = None) -> list:
    """
    Remove meta and undone actions from a raw 18xx action list.

    When *committed_ids* is provided (from the Ruby extractor's initial
    record), only actions whose id is in that set are kept — this cleanly
    handles all undo/redo without reimplementing the logic in Python.
    Actions in SKIP_ACTIONS (program_*, message) are also removed, but
    their auto_actions are preserved (they represent real pass actions
    that count toward consecutive-pass tracking).

    Returns the cleaned list in original order.
    """
    result = []
    for action in actions:
        atype = action.get('type', '')
        action_id = action.get('id')

        # Handle SKIP_ACTIONS (program_*, message): the action itself is
        # always dropped, but its auto_actions are preserved ONLY if the
        # action is committed.  Undone program_* actions must not leak
        # auto_action passes into the stream — that would advance the
        # active player and desync the replay.
        if atype in SKIP_ACTIONS:
            if committed_ids is not None and action_id is not None:
                if action_id not in committed_ids:
                    continue  # Undone — skip entirely including auto_actions
            for auto in action.get('auto_actions', []):
                result.append(auto)
            continue

        # Drop undone actions (not in the committed set from the extractor)
        if committed_ids is not None and action_id is not None:
            if action_id not in committed_ids:
                continue

        result.append(action)

    return result


# =============================================================================
# 2. FLATTEN AUTO-ACTIONS
# =============================================================================

def flatten_auto_actions(actions: list) -> list:
    """
    Expand 'auto_actions' sub-lists into the main action stream.

    For each action that has an 'auto_actions' key, the auto-actions are
    inserted immediately after the parent action in the returned list.
    The parent action itself is also retained.

    Returns the flattened list.
    """
    result = []
    for action in actions:
        result.append(action)
        for auto in action.get('auto_actions', []):
            result.append(auto)
    return result


# =============================================================================
# 3. AUTO-PASS TRACKER
# =============================================================================

class AutoPassTracker:
    """
    Tracks program_* auto-pass state for each player.

    18xx.games supports programmatic auto-pass rules (program_share_pass and
    program_close_pass). This class maintains the active state so that callers
    can query whether a given player should be auto-passed in a given phase.
    """

    def __init__(self, player_ids: list):
        """
        Args:
            player_ids: List of 18xx player IDs (ints) in play order.
        """
        self.player_ids = list(player_ids)
        # player_id -> dict with 'unconditional' and 'indefinite' keys
        self.share_pass: dict = {}
        # player_id -> True (simple flag)
        self.close_pass: dict = {}

    def process_action(self, action: dict):
        """
        Update auto-pass state based on a program_* action.

        Handles:
        - program_share_pass: activate share-pass for entity
        - program_close_pass: activate close-pass for entity
        - program_disable: clear the flag identified by 'original_type'
        """
        atype = action.get('type', '')
        entity = action.get('entity')

        if atype == 'program_share_pass':
            self.share_pass[entity] = {
                'unconditional': action.get('unconditional', False),
                'indefinite': action.get('indefinite', False),
            }

        elif atype == 'program_close_pass':
            self.close_pass[entity] = True

        elif atype == 'program_disable':
            original = action.get('original_type', '')
            if original == 'program_share_pass':
                self.share_pass.pop(entity, None)
            elif original == 'program_close_pass':
                self.close_pass.pop(entity, None)

    def should_auto_pass_invest(self, player_id) -> bool:
        """Return True if the player currently has an active share-pass program."""
        return player_id in self.share_pass

    def should_auto_pass_closing(self, player_id) -> bool:
        """Return True if the player currently has an active close-pass program."""
        return player_id in self.close_pass


# =============================================================================
# 4. PLAYER LOOKUP
# =============================================================================

def entity_to_player_index(players_json: list, entity_id) -> int:
    """
    Return the 0-based index of the player with the given entity_id.

    Args:
        players_json: The 'players' array from the 18xx game JSON. Each element
                      is a dict with at least an 'id' key.
        entity_id: The player ID to look up. May be int or str (18xx sometimes
                   serialises IDs as strings).

    Returns:
        0-based player index.

    Raises:
        ValueError: If no player with the given entity_id is found.
    """
    # Normalise to int for comparison (18xx IDs are always numeric)
    try:
        target = int(entity_id)
    except (TypeError, ValueError):
        target = entity_id

    for idx, player in enumerate(players_json):
        pid = player.get('id')
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            pass
        if pid == target:
            return idx

    raise ValueError(f"Player with entity_id={entity_id!r} not found in players list")


# =============================================================================
# 5. AUCTION SLOT LOOKUP
# =============================================================================

def find_auction_slot(state, company_id: int) -> int:
    """
    Return the auction slot index (0-based) for a company currently in auction.

    Slot ordering matches our engine: companies are assigned slots in ascending
    company_id order. Slot 0 = the lowest-id company currently in auction.

    Args:
        state: Current GameState (needed to read company locations).
        company_id: Target company's ID (0-35).

    Returns:
        Slot index (0-based).

    Raises:
        ValueError: If company_id is not currently in an auction slot.
    """
    slot = 0
    for cid in range(36):  # NUM_COMPANIES = 36
        loc = COMPANIES[cid].get_location(state)
        if loc == LOC_AUCTION:
            if cid == company_id:
                return slot
            slot += 1
    raise ValueError(
        f"Company {COMPANY_NAMES[company_id]} (id={company_id}) is not in auction"
    )


# =============================================================================
# 6-10. PER-PHASE ACTION MAPPERS
# =============================================================================

def map_invest_action(
    state,
    action: dict,
    layout: ActionLayout,
) -> int:
    """
    Map an 18xx INVEST-phase action to our engine's action index.

    Supported action types:
    - 'bid'         -> auction slot action
    - 'buy_shares'  -> buy share for a corp
    - 'sell_shares' -> sell share of a corp (player entity only)
    - 'pass'        -> invest pass (action 0)

    Args:
        state: Current GameState (used to find auction slots).
        action: 18xx action dict.
        layout: ActionLayout for the current num_players.
    Returns:
        Engine action index.

    Raises:
        KeyError: If required action fields are missing.
        ValueError: If the action cannot be mapped.
    """
    atype = action['type']

    if atype == 'pass':
        return layout.pass_invest

    if atype == 'bid':
        company_name = action['company']
        company_id = COMPANY_NAME_TO_ID[company_name]
        face_value = get_company_face_value(company_id)
        bid_price = int(action['price'])
        bid_offset = bid_price - face_value
        slot = find_auction_slot(state, company_id)
        return layout.auction_base + slot * AUCTION_CAP + bid_offset

    if atype == 'buy_shares':
        share_name = action['shares'][0]          # e.g. "DA_1"
        corp_name = share_name.split('_')[0]      # e.g. "DA"
        corp_id = CORP_NAME_TO_ID[corp_name]
        return layout.buy_share_base + corp_id

    if atype == 'sell_shares':
        share_name = action['shares'][0]
        corp_name = share_name.split('_')[0]
        corp_id = CORP_NAME_TO_ID[corp_name]
        return layout.sell_share_base + corp_id

    raise ValueError(f"Unrecognised invest action type: {atype!r}")


def map_bid_action(action: dict, layout: ActionLayout) -> int:
    """
    Map an 18xx BID_IN_AUCTION-phase action to our engine's action index.

    Supported action types:
    - 'pass' -> leave auction
    - 'bid'  -> raise bid (new_bid = face_value + bid_offset + 1)

    The auction company is identified from the action's 'company' field.

    Args:
        action: 18xx action dict.
        layout: ActionLayout for the current num_players.

    Returns:
        Engine action index.

    Raises:
        ValueError: If the action type is not recognised.
    """
    atype = action['type']

    if atype == 'pass':
        return layout.leave_auction

    if atype == 'bid':
        # Identify the company being auctioned from the action itself
        company_name = action['company']
        company_id = COMPANY_NAME_TO_ID[company_name]
        face_value = get_company_face_value(company_id)
        new_bid = int(action['price'])
        # bid_offset: raise_bid_base + offset, where offset = new_bid - face_value - 1
        bid_offset = new_bid - face_value - 1
        return layout.raise_bid_base + bid_offset

    raise ValueError(f"Unrecognised bid action type: {atype!r}")


def map_ipo_action(action: dict, layout: ActionLayout) -> list[int]:
    """
    Map an 18xx IPO-phase action to our engine's action indices.

    The IPO+PAR two-phase flow means a 'par' action maps to TWO engine
    actions: corp selection (IPO phase) + par price (PAR phase).
    A 'pass' maps to a single action.

    Supported action types:
    - 'pass' -> [IPO pass]
    - 'par'  -> [IPO select corp, PAR select price]

    The 'par' action format:
    - action['entity']:      company name being IPO'd (e.g. 'MHE')
    - action['corporation']: target corp name (e.g. 'DA')
    - action['share_price']: "price,row,col" string (e.g. '10,0,6')

    Args:
        action: 18xx action dict.
        layout: ActionLayout for the current num_players.

    Returns:
        List of engine action indices (1 for pass, 2 for par).

    Raises:
        ValueError: If the par price is not valid for the company's star tier,
                    or if the action type is not recognised.
    """
    atype = action['type']

    if atype == 'pass':
        return [layout.ipo_pass]

    if atype == 'par':
        # Parse share price from "price,row,col" format
        price_str = action['share_price']
        par_price = int(price_str.split(',')[0])

        corp_name = action['corporation']
        corp_id = CORP_NAME_TO_ID[corp_name]

        # Find par_index by matching the price against ALL_PAR_PRICES
        par_index = None
        for idx in range(14):  # NUM_PAR_PRICES = 14
            if get_par_price(idx) == par_price:
                par_index = idx
                break

        if par_index is None:
            raise ValueError(
                f"Par price {par_price} is not a valid par price"
            )

        return [
            layout.ipo_base + corp_id,      # IPO: select corp
            layout.par_base + par_index,    # PAR: select par price
        ]

    raise ValueError(f"Unrecognised IPO action type: {atype!r}")


def map_dividend_action(action: dict, layout: ActionLayout) -> int:
    """
    Map an 18xx DIVIDENDS-phase action to our engine's action index.

    The 18xx 'dividend' action has an 'amount' field giving the per-share
    dividend. Our engine encodes dividends as layout.dividend_base + amount.

    Args:
        action: 18xx action dict with 'amount' key.
        layout: ActionLayout for the current num_players.

    Returns:
        Engine action index.
    """
    amount = int(action['amount'])
    return layout.dividend_base + amount


def map_issue_action(action: dict, layout: ActionLayout) -> int:
    """
    Map an 18xx ISSUE_SHARES-phase action to our engine's action index.

    In 18xx, a corporation issuing a share is represented as a 'sell_shares'
    action where the entity_type is 'corporation'. A pass in this phase is a
    plain 'pass' action.

    Supported action types:
    - 'sell_shares' (by corp entity) -> issue share
    - 'pass'                         -> issue pass

    Args:
        action: 18xx action dict.
        layout: ActionLayout for the current num_players.

    Returns:
        Engine action index.

    Raises:
        ValueError: If the action type is not recognised.
    """
    atype = action['type']

    if atype == 'sell_shares':
        # Corporation issuing a share appears as sell_shares with entity_type='corporation'
        return layout.issue_action

    if atype == 'pass':
        return layout.issue_pass

    raise ValueError(f"Unrecognised issue action type: {atype!r}")


# =============================================================================
# 11. SHARED GAME SETUP
# =============================================================================

def override_deck_and_offering(
    state: GameState,
    deck_order_names: list[str],
    offering_names: list[str],
) -> None:
    """Override deck/offering to match an 18xx game's initial state.

    After initialize_game(), the engine has a valid state but with the wrong
    deck order and offering (from the random seed). This patches the state to
    match the 18xx reference by:
    1. Clearing stale auction/revealed company locations from the seed init
    2. Setting the correct deck order
    3. Drawing and auctioning the correct offering companies
    """
    for cid in range(36):
        loc = COMPANIES[cid].get_location(state)
        if loc == LOC_AUCTION:
            state.set_company_for_auction(cid, False)
            COMPANIES[cid].exclude_from_game(state)
        elif loc == LOC_REVEALED:
            COMPANIES[cid].exclude_from_game(state)

    remaining_ids = [COMPANY_NAME_TO_ID[n] for n in reversed(deck_order_names)]
    offering_ids = [COMPANY_NAME_TO_ID[n] for n in reversed(offering_names)]
    full_deck = remaining_ids + offering_ids
    DECK.set_order(state, full_deck)

    for _ in range(len(offering_names)):
        cid = DECK.draw(state)
        COMPANIES[cid].move_to_auction(state)


# =============================================================================
# 12. SHARED ACTION DISPATCH
# =============================================================================

def map_action(
    state: GameState,
    action: dict,
    phase: int,
    layout: ActionLayout,
) -> int | list[int] | None:
    """Map a single 18xx action to engine action index(es).

    Dispatches to the per-phase mappers for simple phases (INVEST, BID, IPO,
    DIVIDENDS, ISSUE). Returns None for automated phases (WRAP_UP, INCOME,
    END_CARD, ACQ, CLOSING) and for actions that don't match the current phase.
    """
    atype = action.get("type", "")
    entity_type = action.get("entity_type", "")

    if phase == PHASE_INVEST:
        if atype in ("bid", "buy_shares", "sell_shares", "pass"):
            if entity_type != "player":
                return None
            return map_invest_action(state, action, layout)
        return None

    if phase == PHASE_BID:
        if atype in ("bid", "pass"):
            return map_bid_action(action, layout)
        return None

    if phase == PHASE_IPO:
        if atype in ("par", "pass"):
            if entity_type != "company":
                return None
            return map_ipo_action(action, layout)
        return None

    if phase == PHASE_DIVIDENDS:
        if atype == "dividend":
            if entity_type != "corporation":
                return None
            return map_dividend_action(action, layout)
        return None

    if phase == PHASE_ISSUE:
        if atype in ("sell_shares", "pass"):
            if entity_type != "corporation":
                return None
            return map_issue_action(action, layout)
        return None

    # Automated phases — no player actions needed
    if phase in (PHASE_WRAP_UP, PHASE_INCOME, PHASE_END_CARD,
                 PHASE_ACQ, PHASE_CLOSING):
        return None

    return None
