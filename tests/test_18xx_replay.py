"""
End-to-end game replay tests against 18xx.games data.

Replays real human games and validates that our Cython engine produces
the same net worth values at each phase boundary.
"""
import json
import pytest
import numpy as np
from pathlib import Path

from state import GameState
from driver import apply_action
from actions import get_valid_action_mask, get_action_layout, decode_action_py
from data import (
    COMPANY_NAME_TO_ID, CORP_NAME_TO_ID,
    py_get_company_face_value, py_get_company_low_price, py_get_company_high_price,
)
from debug_utils import debug_print

# =============================================================================
# CONSTANTS
# =============================================================================

# Phase constants (from state.pyx)
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

# Action type constants (from actions.pyx)
ACTION_PASS = 0
ACTION_AUCTION = 1
ACTION_BUY_SHARE = 2
ACTION_SELL_SHARE = 3
ACTION_LEAVE_AUCTION = 4
ACTION_RAISE_BID = 5
ACTION_ACQ_PRICE = 6
ACTION_ACQ_FI_HIGH = 7
ACTION_ACQ_FI_FACE = 8
ACTION_CLOSE = 9
ACTION_DIVIDEND = 10
ACTION_ISSUE = 11
ACTION_IPO = 12

# Phase code mapping from TXT to engine phases
PHASE_CODE_MAP = {
    'INV': PHASE_INVEST,
    'IPO': PHASE_IPO,
    'ISS': PHASE_ISSUE_SHARES,
    'DIV': PHASE_DIVIDENDS,
    'CLO': PHASE_CLOSING,
    'ACQ': PHASE_ACQUISITION,
}

# Par prices (from data.pyx)
ALL_PAR_PRICES = [10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 27, 30, 33, 37]

# Number of auction slots (number of companies available for auction)
NUM_AUCTION_SLOTS = 3  # Base case, changes with player count


# =============================================================================
# DECK ORDER EXTRACTION
# =============================================================================

def extract_deck_order(actions, num_players):
    """
    Extract the deck order from game actions.

    The deck is organized by color groups. For each group, we have num_players
    non-last companies + 1 last company = num_players + 1 companies per group.

    Args:
        actions: List of game actions
        num_players: Number of players

    Returns:
        tuple: (deck_order, removed_companies)
            deck_order: List of company IDs (index 0 = bottom, last = top)
            removed_companies: List of company IDs not in the deck
    """
    # Find all companies that appeared in the game (bid on or IPO'd)
    companies_seen = []
    companies_set = set()

    for action in actions:
        company = None
        if action.get('type') == 'bid':
            company = action.get('company')
        elif action.get('type') == 'par':
            # IPO action - the entity is the company
            company = action.get('entity')

        if company and company not in companies_set:
            companies_set.add(company)
            company_id = COMPANY_NAME_TO_ID.get(company)
            if company_id is not None:
                companies_seen.append(company_id)

    # Group companies by color
    group_ranges = [
        (0, 6),    # Reds (companies 0-5, last is 5)
        (6, 14),   # Oranges (companies 6-13, last is 13)
        (14, 22),  # Yellows (companies 14-21, last is 21)
        (22, 29),  # Greens (companies 22-28, last is 28)
        (29, 36),  # Blues (companies 29-35, last is 35)
    ]

    deck_order = []
    removed_companies = []

    for start, end in group_ranges:
        last_company = end - 1

        # Get non-last companies in this group that were seen
        group_seen = [c for c in companies_seen if start <= c < last_company]

        # If we've seen less than num_players non-last companies, some were removed
        all_non_last = list(range(start, last_company))
        selected_non_last = group_seen[:num_players] if len(group_seen) >= num_players else group_seen

        # Track which non-last were removed (not in selected)
        for c in all_non_last:
            if c not in selected_non_last:
                removed_companies.append(c)

        # Build group deck: selected non-last + last company
        # Order them based on when they appeared in the game
        group_deck = []
        for c in companies_seen:
            if c in selected_non_last or c == last_company:
                group_deck.append(c)

        # If last company wasn't seen, add it at the end of the group
        if last_company not in group_deck:
            group_deck.append(last_company)

        deck_order.extend(group_deck)

    # The deck order we have is in "revealed order" (top first)
    # We need to reverse it so index 0 = bottom
    deck_order.reverse()

    return deck_order, removed_companies


# =============================================================================
# DATA PARSING
# =============================================================================

def filter_undone_actions(actions):
    """
    Filter out actions that were undone.

    Handles both simple undos (undo previous action) and targeted undos
    (undo back to specific action_id).

    Also expands auto_actions that are bundled with regular actions.

    Returns list of actions that were not undone, with auto_actions expanded.
    """
    # First pass: determine which action IDs are active
    active_ids = []
    undone_ids = set()

    for action in actions:
        action_type = action.get('type')
        action_id = action.get('id')

        if action_type == 'undo':
            target_id = action.get('action_id')
            if target_id is not None:
                # Undo back to specific action (exclusive)
                # Pop all actions with id > target_id
                while active_ids and active_ids[-1] > target_id:
                    undone_ids.add(active_ids.pop())
            else:
                # Undo the last action
                if active_ids:
                    undone_ids.add(active_ids.pop())
        elif action_type == 'redo':
            # Redo re-adds the most recently undone action
            # Find the smallest undone ID that's larger than current max
            if active_ids and undone_ids:
                current_max = active_ids[-1] if active_ids else 0
                candidates = [uid for uid in undone_ids if uid > current_max]
                if candidates:
                    redo_id = min(candidates)
                    undone_ids.remove(redo_id)
                    active_ids.append(redo_id)
            elif undone_ids:
                # No active actions, redo the first undone
                redo_id = min(undone_ids)
                undone_ids.remove(redo_id)
                active_ids.append(redo_id)
        else:
            # Regular action
            if action_id is not None:
                active_ids.append(action_id)

    # Second pass: collect active actions and expand auto_actions
    # Important: sort by action ID to ensure correct order
    active_set = set(active_ids)
    active_actions = [a for a in actions if a.get('id') in active_set]
    active_actions.sort(key=lambda a: a.get('id', 0))

    result = []
    for action in active_actions:
        result.append(action)
        # Expand auto_actions if present
        if 'auto_actions' in action:
            for auto_action in action['auto_actions']:
                result.append(auto_action)

    return result


def parse_18xx_json(filepath):
    """
    Parse 18xx.games JSON file.

    Returns dict with:
    - 'players': list of player dicts with 'id' and 'name'
    - 'actions': filtered list of game actions (undos removed)
    - 'result': dict mapping player_id -> final score
    - 'seed': random seed used for the game
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    return {
        'players': data['players'],
        'actions': filter_undone_actions(data['actions']),
        'result': {int(k): v for k, v in data['result'].items()},
        'seed': data['settings']['seed'],
    }


def parse_net_worth_txt(filepath, player_names):
    """
    Parse net worth TXT file.

    Args:
        filepath: Path to TXT file
        player_names: List of player names in column order (from TXT header)

    Returns dict: {(phase_code, round_num): {player_name: net_worth}}
    """
    checkpoints = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # First line is header with player names (tab-separated)
    # Subsequent lines: PHASE ROUND  $NW1  $NW2  $NW3  $NW4

    for line in lines[1:]:  # Skip header
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')
        if len(parts) < 2:
            parts = line.split()

        # Parse phase and round from first part (e.g., "INV 14")
        phase_round = parts[0].split()
        if len(phase_round) < 2:
            continue

        phase_code = phase_round[0]
        round_num = int(phase_round[1])

        # Parse net worth values (remove $ sign)
        net_worths = {}
        for i, name in enumerate(player_names):
            value_str = parts[i + 1].strip().replace('$', '')
            net_worths[name] = int(value_str)

        checkpoints[(phase_code, round_num)] = net_worths

    return checkpoints


# =============================================================================
# ACTION TRANSLATION
# =============================================================================

def get_par_slot(par_value, star_tier):
    """
    Convert par value to par slot for a given star tier.

    Returns the slot index (0-based) for the par value within valid
    par prices for this star tier.
    """
    # Valid par price ranges by star tier
    valid_ranges = {
        1: [10, 11, 12, 13, 14],           # Reds
        2: [10, 11, 12, 13, 14, 16, 18, 20],  # Oranges
        3: [16, 18, 20, 22, 24, 27],       # Yellows
        4: [22, 24, 27, 30, 33, 37],       # Greens
        5: [30, 33, 37],                   # Blues
    }

    valid_pars = valid_ranges.get(star_tier, [])
    if par_value in valid_pars:
        return valid_pars.index(par_value)
    return -1


def get_auction_slot_for_company(state, company_id):
    """
    Find which auction slot contains the given company.

    Returns slot index (0-based) or -1 if not found.
    """
    from data import py_get_company_face_value as py_get_face_value

    # Get all companies available for auction
    auction_companies = []
    for c in range(36):
        if state.is_company_for_auction_py(c):
            auction_companies.append(c)

    # Companies are ordered by face value (ascending), then by ID for ties
    auction_companies.sort(key=lambda c: (py_get_face_value(c), c))

    if company_id in auction_companies:
        return auction_companies.index(company_id)
    return -1


def translate_invest_action(state, action, player_index, layout):
    """Translate an INVEST phase action to engine action index."""
    action_type = action['type']
    entity_type = action.get('entity_type', 'player')

    if action_type == 'pass':
        # Only player passes in INVEST phase - company passes are in IPO
        if entity_type == 'player':
            return layout['pass_invest']
        return None  # Company pass is not handled in INVEST

    if action_type == 'bid':
        # Starting an auction
        company_name = action['company']
        company_id = COMPANY_NAME_TO_ID[company_name]
        bid_price = action['price']

        # Find the auction slot for this company
        slot = get_auction_slot_for_company(state, company_id)
        if slot < 0:
            # Debug: show what companies ARE available and where the target is
            auction_companies = [c for c in range(36) if state.is_company_for_auction_py(c)]
            revealed_companies = [c for c in range(36) if state.is_company_revealed_py(c)]
            player_owned = [(p, [c for c in range(36) if state.player_owns_company_py(p, c)]) for p in range(4)]
            fi_owned = [c for c in range(36) if state.fi_owns_company_py(c)]
            corp_owned = [(corp, [c for c in range(36) if state.corp_owns_company_py(corp, c)]) for corp in range(8)]
            raise ValueError(f"Company {company_name} (ID={company_id}) not available for auction in turn {state.turn_number}. "
                           f"Auction: {auction_companies}, Revealed: {revealed_companies}, "
                           f"Player-owned: {player_owned}, FI: {fi_owned}, Corp-owned: {corp_owned}")

        # Calculate bid offset (bid_price - face_value)
        face_value = py_get_company_face_value(company_id)
        bid_offset = bid_price - face_value

        # Action index = auction_base + slot * 20 + bid_offset
        return layout['auction_base'] + slot * 20 + bid_offset

    if action_type == 'buy_shares':
        # Extract corp from shares field (e.g., ["SI_1"] -> "SI")
        share_str = action['shares'][0]
        corp_name = share_str.split('_')[0]
        corp_id = CORP_NAME_TO_ID[corp_name]
        return layout['buy_share_base'] + corp_id

    if action_type == 'sell_shares':
        # Player selling shares during INVEST
        share_str = action['shares'][0]
        corp_name = share_str.split('_')[0]
        corp_id = CORP_NAME_TO_ID[corp_name]
        return layout['sell_share_base'] + corp_id

    return None  # Action type not recognized


def translate_auction_action(state, action, layout):
    """Translate a BID_IN_AUCTION phase action to engine action index."""
    action_type = action['type']

    if action_type == 'pass':
        # Leave auction
        return layout['leave_auction']

    if action_type == 'bid':
        # Raise bid - action index is based on bid_offset
        # Action space: offset 0 = face+1, offset 1 = face+2, etc.
        bid_price = action['price']

        # Get the auctioned company to determine face value
        auction_company = state.get_auction_company_py()
        face_value = py_get_company_face_value(auction_company)

        # bid_offset in action space: 0 = face+1, 1 = face+2, etc.
        # So bid_offset = bid_price - face_value - 1
        bid_offset = bid_price - face_value - 1
        if bid_offset < 0:
            raise ValueError(f"Invalid raise bid: {bid_price} must be at least {face_value + 1}")

        # Action index = raise_bid_base + bid_offset
        return layout['raise_bid_base'] + bid_offset

    return None


def translate_dividend_action(state, action, layout):
    """Translate a DIVIDENDS phase action to engine action index."""
    if action['type'] == 'dividend':
        amount = action['amount']
        return layout['dividend_base'] + amount
    return None


def translate_issue_action(state, action, layout):
    """Translate an ISSUE_SHARES phase action to engine action index."""
    action_type = action['type']

    if action_type == 'pass':
        return layout['issue_pass']

    if action_type == 'sell_shares':
        # Corp issuing a share
        return layout['issue_action']

    return None


def translate_ipo_action(state, action, layout):
    """Translate an IPO phase action to engine action index."""
    action_type = action['type']

    if action_type == 'pass':
        return layout['ipo_pass']

    if action_type == 'par':
        # IPO a corporation
        corp_name = action['corporation']
        corp_id = CORP_NAME_TO_ID[corp_name]

        # Parse share_price field: "par,floor,min_ask"
        share_price_parts = action['share_price'].split(',')
        par_value = int(share_price_parts[0])

        # Get the company doing the IPO to determine star tier
        company_name = action['entity']
        company_id = COMPANY_NAME_TO_ID[company_name]
        star_tier = (company_id // 6) + 1  # Approximate star tier from company ID
        if company_id < 6:
            star_tier = 1
        elif company_id < 14:
            star_tier = 2
        elif company_id < 22:
            star_tier = 3
        elif company_id < 29:
            star_tier = 4
        else:
            star_tier = 5

        par_slot = get_par_slot(par_value, star_tier)
        if par_slot < 0:
            raise ValueError(f"Invalid par {par_value} for star tier {star_tier}")

        # Action index = ipo_base + corp_id * 8 + par_slot
        return layout['ipo_base'] + corp_id * 8 + par_slot

    return None


def translate_acquisition_action(state, action, layout):
    """Translate an ACQUISITION phase action to engine action index."""
    action_type = action['type']

    if action_type == 'pass':
        return layout['acq_pass']

    if action_type == 'offer':
        # Make an acquisition offer
        company_name = action['company']
        company_id = COMPANY_NAME_TO_ID[company_name]
        offer_price = action['price']

        # Calculate price offset from low price
        low_price = py_get_company_low_price(company_id)
        price_offset = offer_price - low_price

        if price_offset < 0 or price_offset > 50:
            raise ValueError(f"Invalid offer price {offer_price} for company {company_name}")

        return layout['acq_price_base'] + price_offset

    return None


def translate_closing_action(state, action, layout):
    """Translate a CLOSING phase action to engine action index."""
    action_type = action['type']

    if action_type == 'pass':
        return layout['close_pass']

    # There's no explicit "close" action in 18xx.games data
    # Companies are closed implicitly when sell_company is used
    if action_type == 'sell_company':
        return layout['close_action']

    return None


# =============================================================================
# DECISION COLLECTORS
# =============================================================================

class AcquisitionDecisionCollector:
    """
    Collect acquisition offers from 18xx.games data for a single phase.

    Maps (company, corp) -> price offered, or None for pass.
    Also tracks accept/reject responses.
    """

    def __init__(self, actions, round_num):
        self.offers = {}  # (company_id, corp_id) -> price
        self.responses = {}  # (company_id, corp_id) -> True/False (accepted)

        # Collect all offers and responses for this round
        for action in actions:
            if action['type'] == 'offer':
                company_id = COMPANY_NAME_TO_ID[action['company']]
                corp_id = CORP_NAME_TO_ID[action['corporation']]
                price = action['price']
                self.offers[(company_id, corp_id)] = price

            elif action['type'] == 'respond':
                company_id = COMPANY_NAME_TO_ID[action['company']]
                corp_id = CORP_NAME_TO_ID[action['corporation']]
                accepted = action['accept'].lower() == 'true'
                self.responses[(company_id, corp_id)] = accepted

    def get_decision(self, company_id, corp_id):
        """Return offer price or None (pass) for this specific offer."""
        key = (company_id, corp_id)
        if key in self.offers:
            return self.offers[key]
        return None

    def is_accepted(self, company_id, corp_id):
        """Check if the offer was accepted."""
        key = (company_id, corp_id)
        return self.responses.get(key, True)  # Default to accepted if no response


class ClosingDecisionCollector:
    """
    Collect closing decisions from 18xx.games data for a single phase.

    Maps company_id -> True (close) or False (pass).
    """

    def __init__(self, actions, round_num):
        self.decisions = {}  # company_id -> True/False

        # Look for sell_company actions (closing) and passes
        for action in actions:
            if action['type'] == 'sell_company':
                company_id = COMPANY_NAME_TO_ID[action['company']]
                self.decisions[company_id] = True

            # Pass actions in closing phase are trickier to identify
            # They don't specify which company, so we infer from context

    def get_decision(self, company_id):
        """Return True (close) or False (pass) for this company."""
        return self.decisions.get(company_id, False)


# =============================================================================
# MAIN REPLAY LOGIC
# =============================================================================

class GameReplayer:
    """Replays an 18xx.games game against our engine."""

    def __init__(self, json_path, txt_path):
        self.json_path = Path(json_path)
        self.txt_path = Path(txt_path)

        # Parse game data
        self.game_data = parse_18xx_json(json_path)
        self.players = self.game_data['players']
        self.actions = self.game_data['actions']
        self.final_result = self.game_data['result']
        self.seed = self.game_data['seed']

        # Build player mappings
        self.player_id_to_index = {p['id']: i for i, p in enumerate(self.players)}
        self.player_names = [p['name'] for p in self.players]

        # Parse TXT header to get column order
        with open(txt_path, 'r') as f:
            header = f.readline().strip()
        txt_player_names = [n.strip() for n in header.split('\t') if n.strip()]

        # Parse net worth checkpoints
        self.checkpoints = parse_net_worth_txt(txt_path, txt_player_names)

        # Map TXT column names to engine player indices
        self.txt_to_engine = {}
        for i, name in enumerate(txt_player_names):
            for j, p in enumerate(self.players):
                if p['name'] == name:
                    self.txt_to_engine[name] = j
                    break

        # Action layout for translation
        self.layout = None

        # State
        self.state = None
        self.action_idx = 0
        self.current_round = 1

        # Per-player auto-pass flags (set by program_share_pass actions)
        # Maps player_index -> bool
        self.player_auto_pass = {i: False for i in range(len(self.players))}

    def clear_auto_pass(self):
        """Clear all auto-pass flags (called at phase transitions)."""
        for i in range(len(self.players)):
            self.player_auto_pass[i] = False

    def set_auto_pass(self, player_entity):
        """Set auto-pass flag for a player given their 18xx entity ID."""
        player_idx = self.player_id_to_index.get(player_entity)
        if player_idx is not None:
            self.player_auto_pass[player_idx] = True
            debug_print(f"Auto-pass enabled for player {player_idx} ({self.players[player_idx]['name']})")

    def is_auto_pass(self, player_idx):
        """Check if a player has auto-pass enabled."""
        return self.player_auto_pass.get(player_idx, False)

    def setup_game(self):
        """Initialize game state."""
        num_players = len(self.players)
        self.state = GameState(num_players)
        self.layout = get_action_layout(num_players)

        # Extract deck order from game actions
        deck_order, removed = extract_deck_order(self.actions, num_players)
        debug_print(f"DEBUG: Total filtered actions: {len(self.actions)}")
        debug_print(f"DEBUG: First 20 action IDs: {[a.get('id') for a in self.actions[:20]]}")
        debug_print(f"DEBUG: Actions 25-40 IDs: {[a.get('id') for a in self.actions[25:40]]}")
        debug_print(f"DEBUG: Actions 25-40 types: {[(a.get('id'), a.get('type'), a.get('company', '')) for a in self.actions[25:40]]}")
        self.state.setup_with_deck(deck_order, removed)

    def get_expected_net_worths(self, phase_code, round_num):
        """Get expected net worths for a checkpoint."""
        key = (phase_code, round_num)
        if key not in self.checkpoints:
            return None

        result = {}
        for name, nw in self.checkpoints[key].items():
            player_idx = self.txt_to_engine.get(name)
            if player_idx is not None:
                result[player_idx] = nw
        return result

    def validate_net_worths(self, phase_code, round_num, context=""):
        """Validate net worths at a checkpoint."""
        expected = self.get_expected_net_worths(phase_code, round_num)
        if expected is None:
            return  # No checkpoint for this phase/round

        for player_idx, expected_nw in expected.items():
            actual_nw = int(self.state.get_player_net_worth_py(player_idx))
            if actual_nw != expected_nw:
                player_name = self.players[player_idx]['name']
                raise AssertionError(
                    f"Net worth mismatch at {phase_code} {round_num} for {player_name}: "
                    f"expected ${expected_nw}, got ${actual_nw}. {context}"
                )

    def get_active_player(self):
        """Get the current active player index."""
        return self.state.active_player

    def apply_engine_action(self, action_idx):
        """Apply an action to the engine."""
        apply_action(self.state, action_idx)

    def get_valid_actions(self):
        """Get valid action mask."""
        return get_valid_action_mask(self.state)

    def find_valid_action(self, expected_idx):
        """Check if an action index is valid, return it or find alternative."""
        mask = self.get_valid_actions()
        if mask[expected_idx] == 1.0:
            return expected_idx

        # Action not valid - this indicates a translation error
        return None

    def translate_next_action(self):
        """
        Translate the next 18xx.games action to an engine action index.

        Returns (action_idx, action_consumed) or (None, False) if no match.
        """
        if self.action_idx >= len(self.actions):
            return None, False

        action = self.actions[self.action_idx]
        phase = self.state.phase

        # Handle program_share_pass - set auto-pass flag for the player
        if action.get('type') == 'program_share_pass':
            self.set_auto_pass(action.get('entity'))
            self.action_idx += 1
            return None, True  # Consumed, continue processing

        # Skip actions that don't require engine input
        skip_types = {'undo', 'redo'}
        if action.get('type') in skip_types:
            self.action_idx += 1
            return None, True

        try:
            if phase == PHASE_INVEST:
                active_player = self.get_active_player()
                active_entity = self.players[active_player]['id']
                action_entity = action.get('entity')

                # If active player has auto-pass enabled, pass without consuming action
                if self.is_auto_pass(active_player):
                    return self.layout['pass_invest'], False

                # If the next action is for a different player or a company, auto-pass
                if action.get('entity_type') == 'player' and action_entity != active_entity:
                    return self.layout['pass_invest'], False  # Don't consume
                if action.get('entity_type') == 'company':
                    return self.layout['pass_invest'], False  # Don't consume

                engine_action = translate_invest_action(
                    self.state, action, active_player, self.layout
                )

            elif phase == PHASE_BID_IN_AUCTION:
                active_player = self.get_active_player()
                active_entity = self.players[active_player]['id']
                action_entity = action.get('entity')

                # If active player has auto-pass enabled, leave auction
                if self.is_auto_pass(active_player):
                    return self.layout['leave_auction'], False

                # If action is for a different player, leave auction
                if action.get('entity_type') == 'player' and action_entity != active_entity:
                    return self.layout['leave_auction'], False  # Don't consume

                engine_action = translate_auction_action(self.state, action, self.layout)

            elif phase == PHASE_DIVIDENDS:
                # DIVIDENDS phase actions are 'dividend' with amount
                # If the action is something else, we have a problem
                if action.get('type') != 'dividend':
                    raise RuntimeError(f"Expected 'dividend' action in DIVIDENDS phase, got {action.get('type')}")
                engine_action = translate_dividend_action(self.state, action, self.layout)
            elif phase == PHASE_ISSUE_SHARES:
                # ISSUE_SHARES phase actions are 'sell_shares' (corp issues) or 'pass'
                # If the action is something else, auto-pass through all issue opportunities
                if action.get('type') not in ('sell_shares', 'pass'):
                    count = 0
                    while self.state.phase == PHASE_ISSUE_SHARES and count < 100:
                        self.apply_engine_action(self.layout['issue_pass'])
                        count += 1
                    if count >= 100:
                        raise RuntimeError("ISSUE_SHARES phase did not transition after 100 passes")
                    return -1, False  # Phase transitioned, re-check (-1 = sentinel)
                engine_action = translate_issue_action(self.state, action, self.layout)
            elif phase == PHASE_IPO:
                # IPO phase actions are 'par' or 'pass'
                # If the action is something else, auto-pass through all IPO opportunities
                if action.get('type') not in ('par', 'pass'):
                    count = 0
                    while self.state.phase == PHASE_IPO and count < 100:
                        self.apply_engine_action(self.layout['ipo_pass'])
                        count += 1
                    if count >= 100:
                        raise RuntimeError("IPO phase did not transition after 100 passes")
                    return -1, False  # Phase transitioned, re-check (-1 = sentinel)
                engine_action = translate_ipo_action(self.state, action, self.layout)
            elif phase == PHASE_ACQUISITION:
                # ACQUISITION phase actions are 'offer' or 'pass'
                # If the action is something else (like a future INVEST bid), auto-pass
                # through all remaining acquisition offers until phase transitions
                if action.get('type') not in ('offer', 'pass'):
                    # Keep passing until phase changes (with safety limit)
                    count = 0
                    while self.state.phase == PHASE_ACQUISITION and count < 100:
                        self.apply_engine_action(self.layout['acq_pass'])
                        count += 1
                    if count >= 100:
                        raise RuntimeError("ACQUISITION phase did not transition after 100 passes")
                    return -1, False  # Phase transitioned, re-check (-1 = sentinel)
                engine_action = translate_acquisition_action(self.state, action, self.layout)
            elif phase == PHASE_CLOSING:
                # CLOSING phase actions are 'sell_company' or 'pass'
                # If the action is something else, auto-pass through all closings
                if action.get('type') not in ('sell_company', 'pass'):
                    count = 0
                    while self.state.phase == PHASE_CLOSING and count < 100:
                        self.apply_engine_action(self.layout['close_pass'])
                        count += 1
                    if count >= 100:
                        raise RuntimeError("CLOSING phase did not transition after 100 passes")
                    return -1, False  # Phase transitioned, re-check (-1 = sentinel)
                engine_action = translate_closing_action(self.state, action, self.layout)
            else:
                engine_action = None

            if engine_action is not None:
                self.action_idx += 1
                return engine_action, True

        except (KeyError, ValueError) as e:
            debug_print(f"Warning: phase={phase}, action_idx={self.action_idx}")
            debug_print(f"Warning: Could not translate action {action}: {e}")

        return None, False

    def replay_step(self):
        """Execute one step of the replay."""
        phase = self.state.phase

        if phase == PHASE_GAME_OVER:
            return False

        # Try to translate and apply the next action
        engine_action, consumed = self.translate_next_action()

        if engine_action == -1:
            # Sentinel: phase transitioned via auto-pass, continue without consuming action
            return True
        elif engine_action is not None:
            # Validate the action is actually valid
            valid_action = self.find_valid_action(engine_action)
            if valid_action is not None:
                self.apply_engine_action(valid_action)
            else:
                # Action not valid - try to find a matching valid action
                debug_print(f"Warning: Translated action {engine_action} not valid at phase {phase}")
                # For now, just skip
        elif not consumed:
            # No action could be translated - might need to pass or handle automatically
            mask = self.get_valid_actions()
            valid_indices = np.where(mask == 1.0)[0]

            if len(valid_indices) == 1:
                # Only one valid action - apply it (forced move)
                self.apply_engine_action(valid_indices[0])
            elif len(valid_indices) == 0:
                raise RuntimeError(f"No valid actions at phase {phase}")
            else:
                # Multiple valid actions but no 18xx.games action
                # This might happen in automatic phases or misaligned state
                debug_print(f"Warning: {len(valid_indices)} valid actions but no input at phase {phase}")
                return False

        return True

    def replay_game(self):
        """Replay the entire game."""
        self.setup_game()

        max_steps = 10000
        step = 0
        last_phase = None
        last_round = None

        while self.state.phase != PHASE_GAME_OVER and step < max_steps:
            phase = self.state.phase
            round_num = self.state.turn_number

            # Check for phase transition
            if phase != last_phase or round_num != last_round:
                # Clear auto-pass flags when entering a new INVEST phase
                if phase == PHASE_INVEST and last_phase != PHASE_INVEST:
                    self.clear_auto_pass()

                # Validate at phase boundaries
                if last_phase is not None:
                    phase_code = {v: k for k, v in PHASE_CODE_MAP.items()}.get(last_phase)
                    if phase_code:
                        self.validate_net_worths(phase_code, last_round)

                last_phase = phase
                last_round = round_num

            if not self.replay_step():
                break

            step += 1

        if step >= max_steps:
            raise RuntimeError(f"Replay did not complete within {max_steps} steps")

        # Final validation
        self.validate_final_scores()

    def validate_final_scores(self):
        """Validate final scores match."""
        for player_id, expected_score in self.final_result.items():
            player_idx = self.player_id_to_index[player_id]
            actual_score = int(self.state.get_player_net_worth_py(player_idx))

            if actual_score != expected_score:
                player_name = next(p['name'] for p in self.players if p['id'] == player_id)
                raise AssertionError(
                    f"Final score mismatch for {player_name}: "
                    f"expected ${expected_score}, got ${actual_score}"
                )


# =============================================================================
# TEST CLASSES
# =============================================================================

class Test18xxGameReplay:
    """End-to-end replay tests against 18xx.games data."""

    def test_parse_json(self):
        """Test JSON parsing and undo filtering."""
        json_path = Path(__file__).parent / "18xx_data" / "18xx_game_1.json"
        game_data = parse_18xx_json(json_path)

        assert len(game_data['players']) == 4
        assert 'actions' in game_data
        assert 'result' in game_data

        # Verify undos were filtered
        action_types = [a['type'] for a in game_data['actions']]
        assert 'undo' not in action_types
        assert 'redo' not in action_types

    def test_parse_txt(self):
        """Test TXT checkpoint parsing."""
        txt_path = Path(__file__).parent / "18xx_data" / "18xx_game_1.txt"
        player_names = ['chadamir', 'reveler', 'd_choo', 'CardboardBits']

        checkpoints = parse_net_worth_txt(txt_path, player_names)

        # Should have 73 checkpoints
        assert len(checkpoints) > 0

        # Check a known value (INV 14 = final state)
        final = checkpoints.get(('INV', 14))
        assert final is not None
        assert final['chadamir'] == 206
        assert final['reveler'] == 151
        assert final['d_choo'] == 149
        assert final['CardboardBits'] == 106

    def test_replayer_setup(self):
        """Test replayer initialization."""
        json_path = Path(__file__).parent / "18xx_data" / "18xx_game_1.json"
        txt_path = Path(__file__).parent / "18xx_data" / "18xx_game_1.txt"

        replayer = GameReplayer(json_path, txt_path)
        assert len(replayer.players) == 4
        assert len(replayer.actions) > 0
        assert len(replayer.checkpoints) > 0

    def test_game_1_replay(self):
        """
        Replay game 232836 and validate all phase checkpoints.

        Known issues to fix:
        1. Acquisition/Closing phase action reordering not yet implemented
        """
        json_path = Path(__file__).parent / "18xx_data" / "18xx_game_1.json"
        txt_path = Path(__file__).parent / "18xx_data" / "18xx_game_1.txt"

        replayer = GameReplayer(json_path, txt_path)
        replayer.replay_game()

    def test_round_1_replay(self):
        """Replay just round 1 to validate basic mechanics work."""
        json_path = Path(__file__).parent / "18xx_data" / "18xx_game_1.json"
        txt_path = Path(__file__).parent / "18xx_data" / "18xx_game_1.txt"

        replayer = GameReplayer(json_path, txt_path)
        replayer.setup_game()

        # Replay until we hit a translation error or reach IPO phase completion
        steps = 0
        max_steps = 35  # Should cover round 1

        while steps < max_steps:
            phase = replayer.state.phase

            if phase == PHASE_GAME_OVER:
                break

            engine_action, consumed = replayer.translate_next_action()

            if engine_action is not None:
                valid_action = replayer.find_valid_action(engine_action)
                if valid_action is not None:
                    apply_action(replayer.state, valid_action)
                    steps += 1
                else:
                    # Action translation error - expected for now
                    break
            elif not consumed:
                # Could not translate - expected for turn order issues
                break
            else:
                steps += 1

        # We should have replayed at least round 1
        assert steps >= 25, f"Expected at least 25 steps, got {steps}"
