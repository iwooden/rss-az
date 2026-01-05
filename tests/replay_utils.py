"""
Utility functions for 18xx.games replay testing.

Simplifies the replay logic by handling phase transitions cleanly.
"""

from state import GameState
from driver import apply_action
from actions import get_valid_action_mask, get_action_layout
from data import COMPANY_NAME_TO_ID, CORP_NAME_TO_ID, py_get_company_face_value
from conftest import debug_print

# Phase constants
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

PHASE_NAMES = ['INVEST', 'BID_IN_AUCTION', 'WRAP_UP', 'ACQUISITION', 'CLOSING',
               'INCOME', 'DIVIDENDS', 'END_CARD', 'ISSUE_SHARES', 'IPO', 'GAME_OVER']


def get_auction_slot(state, company_id):
    """Find the auction slot for a company."""
    auction_companies = []
    for c in range(36):
        if state.is_company_for_auction_py(c):
            auction_companies.append(c)
    auction_companies.sort(key=lambda c: (py_get_company_face_value(c), c))
    return auction_companies.index(company_id) if company_id in auction_companies else -1


def auto_pass_phase(state, layout, phase, max_iterations=100):
    """Auto-pass through a phase until it transitions. Returns True if phase changed."""
    pass_actions = {
        PHASE_INVEST: layout['pass_invest'],
        PHASE_BID_IN_AUCTION: layout['leave_auction'],
        PHASE_ACQUISITION: layout['acq_pass'],
        PHASE_CLOSING: layout['close_pass'],
        PHASE_ISSUE_SHARES: layout['issue_pass'],
        PHASE_IPO: layout['ipo_pass'],
    }

    if phase not in pass_actions:
        return False

    count = 0
    while state.phase == phase and count < max_iterations:
        apply_action(state, pass_actions[phase])
        count += 1

    return count > 0


def handle_automatic_phases(state, layout):
    """Handle phases that run automatically (WRAP_UP, INCOME, etc.)."""
    from phases.wrapup import handle_wrap_up

    while state.phase in [PHASE_WRAP_UP, PHASE_ACQUISITION, PHASE_CLOSING,
                          PHASE_INCOME, PHASE_DIVIDENDS, PHASE_END_CARD,
                          PHASE_ISSUE_SHARES]:
        if state.phase == PHASE_WRAP_UP:
            handle_wrap_up(state)
        elif state.phase == PHASE_ACQUISITION:
            auto_pass_phase(state, layout, PHASE_ACQUISITION)
        elif state.phase == PHASE_CLOSING:
            auto_pass_phase(state, layout, PHASE_CLOSING)
        elif state.phase == PHASE_ISSUE_SHARES:
            auto_pass_phase(state, layout, PHASE_ISSUE_SHARES)
        else:
            # INCOME, DIVIDENDS, END_CARD are truly automatic
            break


class SimpleReplayer:
    """
    Simplified game replayer that processes actions without complex entity matching.

    Key insight: Instead of trying to match entities, we pass players until
    the engine state matches what we need for the action.
    """

    def __init__(self, players, layout):
        self.players = players
        self.layout = layout
        self.entity_to_player = {p['id']: i for i, p in enumerate(players)}
        self.player_names = [p['name'] for p in players]

    def pass_until_player(self, state, target_player, max_passes=10):
        """Pass in INVEST phase until target_player is active."""
        passes = 0
        while state.active_player != target_player and passes < max_passes:
            if state.phase != PHASE_INVEST:
                return False  # Phase changed
            apply_action(state, self.layout['pass_invest'])
            passes += 1
        return state.active_player == target_player

    def leave_until_player(self, state, target_player, max_leaves=10):
        """Leave auction until target_player is active."""
        leaves = 0
        while state.active_player != target_player and leaves < max_leaves:
            if state.phase != PHASE_BID_IN_AUCTION:
                return False  # Auction ended
            apply_action(state, self.layout['leave_auction'])
            leaves += 1
        return state.active_player == target_player

    def apply_invest_action(self, state, action):
        """Apply an INVEST phase action."""
        action_type = action['type']
        entity = action.get('entity')
        player_idx = self.entity_to_player.get(entity)

        if player_idx is None:
            return False

        # Pass until we reach the right player
        if state.active_player != player_idx:
            if not self.pass_until_player(state, player_idx):
                return False

        if action_type == 'pass':
            apply_action(state, self.layout['pass_invest'])
        elif action_type == 'bid':
            company_name = action['company']
            company_id = COMPANY_NAME_TO_ID[company_name]
            face = py_get_company_face_value(company_id)
            bid_offset = action['price'] - face
            slot = get_auction_slot(state, company_id)
            if slot < 0:
                return False
            apply_action(state, self.layout['auction_base'] + slot * 20 + bid_offset)
        elif action_type == 'buy_shares':
            share_str = action['shares'][0]
            corp_name = share_str.split('_')[0]
            corp_id = CORP_NAME_TO_ID[corp_name]
            apply_action(state, self.layout['buy_share_base'] + corp_id)
        elif action_type == 'sell_shares':
            share_str = action['shares'][0]
            corp_name = share_str.split('_')[0]
            corp_id = CORP_NAME_TO_ID[corp_name]
            apply_action(state, self.layout['sell_share_base'] + corp_id)
        else:
            return False

        return True

    def apply_auction_action(self, state, action):
        """Apply a BID_IN_AUCTION phase action."""
        action_type = action['type']
        entity = action.get('entity')
        player_idx = self.entity_to_player.get(entity)

        if player_idx is None:
            return False

        # Leave auction for players until we reach the right one
        if state.active_player != player_idx:
            if not self.leave_until_player(state, player_idx):
                return False

        if action_type == 'pass':
            apply_action(state, self.layout['leave_auction'])
        elif action_type == 'bid':
            company_id = state.get_auction_company_py()
            face = py_get_company_face_value(company_id)
            bid_offset = action['price'] - face - 1  # Raise bid uses offset-1
            apply_action(state, self.layout['raise_bid_base'] + bid_offset)
        else:
            return False

        return True

    def apply_ipo_action(self, state, action):
        """Apply an IPO phase action."""
        action_type = action['type']

        if action_type == 'pass':
            apply_action(state, self.layout['ipo_pass'])
        elif action_type == 'par':
            corp_name = action['corporation']
            corp_id = CORP_NAME_TO_ID[corp_name]
            par_value = int(action['share_price'].split(',')[0])
            # Simplified par slot lookup
            valid_pars = [10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 27, 30, 33, 37]
            if par_value in valid_pars:
                par_slot = valid_pars.index(par_value)
                apply_action(state, self.layout['ipo_base'] + corp_id * 8 + par_slot)
            else:
                return False
        else:
            return False

        return True

    def apply_action(self, state, action):
        """Apply a single action, handling phase transitions as needed."""
        action_type = action.get('type')
        entity_type = action.get('entity_type')
        phase = state.phase

        # Handle automatic phases first
        handle_automatic_phases(state, self.layout)
        phase = state.phase

        # Route to appropriate handler based on action type and phase
        if entity_type == 'player':
            if phase == PHASE_INVEST:
                if action_type in ('pass', 'bid', 'buy_shares', 'sell_shares'):
                    return self.apply_invest_action(state, action)
                else:
                    # Action doesn't match phase - need to transition
                    auto_pass_phase(state, self.layout, PHASE_INVEST)
                    handle_automatic_phases(state, self.layout)
                    return self.apply_action(state, action)  # Retry

            elif phase == PHASE_BID_IN_AUCTION:
                if action_type in ('pass', 'bid'):
                    return self.apply_auction_action(state, action)
                else:
                    # Player action but not auction-related (e.g., buy_shares)
                    # This means the auction should auto-resolve
                    # All remaining players leave until auction ends
                    auto_pass_phase(state, self.layout, PHASE_BID_IN_AUCTION)
                    handle_automatic_phases(state, self.layout)
                    return self.apply_action(state, action)  # Retry in new phase

        elif entity_type == 'company':
            if phase == PHASE_IPO:
                return self.apply_ipo_action(state, action)
            else:
                # Need to transition to IPO phase
                auto_pass_phase(state, self.layout, phase)
                handle_automatic_phases(state, self.layout)
                if state.phase == PHASE_IPO:
                    return self.apply_ipo_action(state, action)

        return False


def debug_state(state, msg=""):
    """Print debug info about current state (only when debug mode enabled)."""
    debug_print(f"{msg}")
    debug_print(f"  Phase: {PHASE_NAMES[state.phase]}, Turn: {state.turn_number}")
    debug_print(f"  Active player: {state.active_player}")
    debug_print(f"  Cash: " + ", ".join([f"P{i}=${state.get_player_cash_py(i)}" for i in range(4)]))
