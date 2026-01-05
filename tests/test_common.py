"""Shared test utilities, constants, and StateBuilder for phase tests."""

import pytest
from cython_core.state import GameState


# =============================================================================
# CONSTANTS
# =============================================================================

# Corporation IDs
CORP_JS = 0   # Junkyard Scrappers
CORP_S = 1    # Synergistic
CORP_OS = 2   # Overseas Trading
CORP_SM = 3   # Stock Masters
CORP_PR = 4   # Priority Mail
CORP_DA = 5   # Delta Airlines
CORP_VM = 6   # VMware
CORP_SI = 7   # Stars, Inc.

NUM_CORPS = 8
NUM_COMPANIES = 36
NUM_MARKET_SPACES = 27

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

# Market prices by index
MARKET_PRICES = [
    0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16,
    18, 20, 22, 24, 27, 30, 33, 37, 41, 45,
    50, 55, 61, 68, 75
]

# Par price indices (subset of market indices)
PAR_PRICE_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
NUM_PAR_PRICES = 14

# Normalization divisors
CASH_DIVISOR = 200.0
SHARE_DIVISOR = 7.0
STAR_DIVISOR = 20.0


# =============================================================================
# STATE BUILDER
# =============================================================================

class StateBuilder:
    """
    Helper to set up game state for testing.

    Provides accessor methods for manipulating the numpy array representation
    of game state. All methods return self for chaining.

    State layout for 3 players:
    - [0-10]      Phase one-hot (11 values)
    - [11-17]     COO level one-hot (7 values)
    - [18-91]     Player 0 data (74 values)
    - [92-165]    Player 1 data (74 values)
    - [166-239]   Player 2 data (74 values)
    - [240-276]   Foreign Investor data (37 values)
    - [277-312]   Auction companies (36 values)
    - [313-348]   Revealed companies (36 values)
    - [349-384]   Removed companies (36 values)
    - [385-420]   Company adjusted incomes (36 values, normalized by 10)
    - [421-447]   Market availability (27 values)
    - [448-1319]  Corporations (8 * 109 = 872 values)
    - [1320+]     Turn state (variable)

    Player layout (74 values):
    - [0]       cash (normalized by 200)
    - [1]       net_worth (normalized by 200)
    - [2-4]     turn_order one-hot (3 values for 3 players)
    - [5]       is_auction_high_bidder flag
    - [6-41]    owned_companies (36 values)
    - [42-49]   owned_shares (8 values, normalized by 7)
    - [50-57]   is_president (8 values)
    - [58-65]   share_sells (8 values)
    - [66-73]   share_buys (8 values)

    Corp layout (109 values):
    - [0]       active
    - [1]       cash (normalized by 200)
    - [2]       unissued_shares (normalized by 7)
    - [3]       issued_shares (normalized by 7)
    - [4]       bank_shares (normalized by 7)
    - [5]       income (normalized by 200)
    - [6]       stars (normalized by 20)
    - [7]       share_price (normalized by 200)
    - [8]       acquisition_proceeds (normalized by 200)
    - [9]       in_receivership
    - [10-36]   price_index one-hot (27 values)
    - [37-72]   owned_companies (36 values)
    - [73-108]  acquisition_companies (36 values)
    """

    # Class constants for layout (3 players)
    PHASE_OFFSET = 0
    COO_OFFSET = 11  # After phase one-hot (11 phases now, was 12)
    PLAYERS_OFFSET = 18  # After phase[11] + coo[7]
    PLAYER_STRIDE = 74

    # Player field offsets (within player block)
    P_CASH = 0
    P_NET_WORTH = 1
    P_TURN_ORDER = 2
    P_AUCTION_HIGH_BIDDER = 5
    P_OWNED_COMPANIES = 6
    P_OWNED_SHARES = 42
    P_IS_PRESIDENT = 50

    # Corp field offsets (within corp block)
    C_ACTIVE = 0
    C_CASH = 1
    C_UNISSUED_SHARES = 2
    C_ISSUED_SHARES = 3
    C_BANK_SHARES = 4
    C_INCOME = 5
    C_STARS = 6
    C_SHARE_PRICE = 7
    C_ACQUISITION_PROCEEDS = 8
    C_IN_RECEIVERSHIP = 9
    C_PRICE_INDEX = 10
    C_OWNED_COMPANIES = 37
    C_ACQUISITION_COMPANIES = 73

    CORP_STRIDE = 109  # 10 fields + 27 price_index + 36 owned + 36 acquisition

    def __init__(self, state: GameState, num_players: int = 3):
        self.state = state
        self._arr = state.as_numpy()
        self._num_players = num_players

        # Compute dynamic offsets based on player count
        self._players_offset = self.PLAYERS_OFFSET
        self._player_stride = self.PLAYER_STRIDE
        self._fi_offset = self._players_offset + num_players * self._player_stride
        self._auction_companies_offset = self._fi_offset + 37
        self._revealed_companies_offset = self._auction_companies_offset + 36
        self._removed_companies_offset = self._revealed_companies_offset + 36
        self._company_incomes_offset = self._removed_companies_offset + 36
        self._market_offset = self._company_incomes_offset + 36
        self._corps_offset = self._market_offset + 27
        self._corp_stride = self.CORP_STRIDE
        self._turn_offset = self._corps_offset + NUM_CORPS * self._corp_stride

    # =========================================================================
    # PLAYER ACCESSORS
    # =========================================================================

    def _player_offset(self, player_id: int) -> int:
        """Get offset to start of player data."""
        return self._players_offset + player_id * self._player_stride

    def set_player_cash(self, player_id: int, cash: int):
        """Set player's cash."""
        self._arr[self._player_offset(player_id) + self.P_CASH] = cash / CASH_DIVISOR
        return self

    def get_player_cash(self, player_id: int) -> int:
        """Get player's cash."""
        return int(self._arr[self._player_offset(player_id) + self.P_CASH] * CASH_DIVISOR + 0.5)

    def set_player_net_worth(self, player_id: int, net_worth: int):
        """Set player's net worth."""
        self._arr[self._player_offset(player_id) + self.P_NET_WORTH] = net_worth / CASH_DIVISOR
        return self

    def get_player_net_worth(self, player_id: int) -> int:
        """Get player's net worth."""
        return int(self._arr[self._player_offset(player_id) + self.P_NET_WORTH] * CASH_DIVISOR + 0.5)

    def set_player_turn_order(self, player_id: int, position: int):
        """Set player's turn order position (one-hot)."""
        offset = self._player_offset(player_id) + self.P_TURN_ORDER
        for i in range(self._num_players):
            self._arr[offset + i] = 0.0
        self._arr[offset + position] = 1.0
        return self

    def get_player_turn_order(self, player_id: int) -> int:
        """Get player's turn order position."""
        offset = self._player_offset(player_id) + self.P_TURN_ORDER
        for i in range(self._num_players):
            if self._arr[offset + i] == 1.0:
                return i
        return -1

    def set_player_owns_company(self, player_id: int, company_id: int, owns: bool = True):
        """Set whether player owns a company."""
        offset = self._player_offset(player_id) + self.P_OWNED_COMPANIES + company_id
        self._arr[offset] = 1.0 if owns else 0.0
        return self

    def player_owns_company(self, player_id: int, company_id: int) -> bool:
        """Check if player owns a company."""
        offset = self._player_offset(player_id) + self.P_OWNED_COMPANIES + company_id
        return self._arr[offset] == 1.0

    def set_player_shares(self, player_id: int, corp_id: int, shares: int):
        """Set player's shares of a corp."""
        offset = self._player_offset(player_id) + self.P_OWNED_SHARES + corp_id
        self._arr[offset] = shares / SHARE_DIVISOR
        return self

    def get_player_shares(self, player_id: int, corp_id: int) -> int:
        """Get player's shares of a corp."""
        offset = self._player_offset(player_id) + self.P_OWNED_SHARES + corp_id
        return int(self._arr[offset] * SHARE_DIVISOR + 0.5)

    def set_player_president(self, player_id: int, corp_id: int, is_pres: bool = True):
        """Set whether player is president of a corp."""
        offset = self._player_offset(player_id) + self.P_IS_PRESIDENT + corp_id
        self._arr[offset] = 1.0 if is_pres else 0.0
        return self

    def is_player_president(self, player_id: int, corp_id: int) -> bool:
        """Check if player is president of a corp."""
        offset = self._player_offset(player_id) + self.P_IS_PRESIDENT + corp_id
        return self._arr[offset] == 1.0

    # =========================================================================
    # FOREIGN INVESTOR ACCESSORS
    # =========================================================================

    def set_fi_cash(self, cash: int):
        """Set FI's cash."""
        self._arr[self._fi_offset] = cash / CASH_DIVISOR
        return self

    def get_fi_cash(self) -> int:
        """Get FI's cash."""
        return int(self._arr[self._fi_offset] * CASH_DIVISOR + 0.5)

    def set_fi_owns_company(self, company_id: int, owns: bool = True):
        """Set whether FI owns a company."""
        self._arr[self._fi_offset + 1 + company_id] = 1.0 if owns else 0.0
        return self

    def fi_owns_company(self, company_id: int) -> bool:
        """Check if FI owns a company."""
        return self._arr[self._fi_offset + 1 + company_id] == 1.0

    # =========================================================================
    # CORP ACCESSORS
    # =========================================================================

    def _corp_offset(self, corp_id: int) -> int:
        """Get offset to start of corp data."""
        return self._corps_offset + corp_id * self._corp_stride

    def set_corp_active(self, corp_id: int, active: bool = True):
        """Set whether corp is active."""
        self._arr[self._corp_offset(corp_id) + self.C_ACTIVE] = 1.0 if active else 0.0
        return self

    def is_corp_active(self, corp_id: int) -> bool:
        """Check if corp is active."""
        return self._arr[self._corp_offset(corp_id) + self.C_ACTIVE] == 1.0

    def set_corp_cash(self, corp_id: int, cash: int):
        """Set corp's cash."""
        self._arr[self._corp_offset(corp_id) + self.C_CASH] = cash / CASH_DIVISOR
        return self

    def get_corp_cash(self, corp_id: int) -> int:
        """Get corp's cash."""
        return int(self._arr[self._corp_offset(corp_id) + self.C_CASH] * CASH_DIVISOR + 0.5)

    def set_corp_unissued_shares(self, corp_id: int, shares: int):
        """Set corp's unissued shares."""
        self._arr[self._corp_offset(corp_id) + self.C_UNISSUED_SHARES] = shares / SHARE_DIVISOR
        return self

    def get_corp_unissued_shares(self, corp_id: int) -> int:
        """Get corp's unissued shares."""
        return int(self._arr[self._corp_offset(corp_id) + self.C_UNISSUED_SHARES] * SHARE_DIVISOR + 0.5)

    def set_corp_issued_shares(self, corp_id: int, shares: int):
        """Set corp's issued shares."""
        self._arr[self._corp_offset(corp_id) + self.C_ISSUED_SHARES] = shares / SHARE_DIVISOR
        return self

    def get_corp_issued_shares(self, corp_id: int) -> int:
        """Get corp's issued shares."""
        return int(self._arr[self._corp_offset(corp_id) + self.C_ISSUED_SHARES] * SHARE_DIVISOR + 0.5)

    def set_corp_bank_shares(self, corp_id: int, shares: int):
        """Set corp's bank shares."""
        self._arr[self._corp_offset(corp_id) + self.C_BANK_SHARES] = shares / SHARE_DIVISOR
        return self

    def get_corp_bank_shares(self, corp_id: int) -> int:
        """Get corp's bank shares."""
        return int(self._arr[self._corp_offset(corp_id) + self.C_BANK_SHARES] * SHARE_DIVISOR + 0.5)

    def set_corp_in_receivership(self, corp_id: int, in_recv: bool = True):
        """Set whether corp is in receivership."""
        self._arr[self._corp_offset(corp_id) + self.C_IN_RECEIVERSHIP] = 1.0 if in_recv else 0.0
        return self

    def is_corp_in_receivership(self, corp_id: int) -> bool:
        """Check if corp is in receivership."""
        return self._arr[self._corp_offset(corp_id) + self.C_IN_RECEIVERSHIP] == 1.0

    def set_corp_price_index(self, corp_id: int, index: int):
        """Set corp's market price index (updates both one-hot and compact storage)."""
        self.state.set_corp_price_index_py(corp_id, index)
        # Mark market space as taken
        if index >= 0:
            self._arr[self._market_offset + index] = 0.0
        return self

    def get_corp_price_index(self, corp_id: int) -> int:
        """Get corp's market price index (from one-hot visible state)."""
        offset = self._corp_offset(corp_id) + self.C_PRICE_INDEX
        for i in range(NUM_MARKET_SPACES):
            if self._arr[offset + i] == 1.0:
                return i
        return -1

    def get_corp_share_price(self, corp_id: int) -> int:
        """Get corp's share price."""
        return int(self._arr[self._corp_offset(corp_id) + self.C_SHARE_PRICE] * CASH_DIVISOR + 0.5)

    def set_corp_owns_company(self, corp_id: int, company_id: int, owns: bool = True):
        """Set whether corp owns a company."""
        offset = self._corp_offset(corp_id) + self.C_OWNED_COMPANIES + company_id
        self._arr[offset] = 1.0 if owns else 0.0
        return self

    def corp_owns_company(self, corp_id: int, company_id: int) -> bool:
        """Check if corp owns a company."""
        offset = self._corp_offset(corp_id) + self.C_OWNED_COMPANIES + company_id
        return self._arr[offset] == 1.0

    def get_corp_acquisition_proceeds(self, corp_id: int) -> int:
        """Get corp's pending acquisition proceeds."""
        return int(self._arr[self._corp_offset(corp_id) + self.C_ACQUISITION_PROCEEDS] * CASH_DIVISOR + 0.5)

    def corp_has_acquisition_company(self, corp_id: int, company_id: int) -> bool:
        """Check if corp has company in acquisition pile."""
        offset = self._corp_offset(corp_id) + self.C_ACQUISITION_COMPANIES + company_id
        return self._arr[offset] == 1.0

    # =========================================================================
    # MARKET ACCESSORS
    # =========================================================================

    def set_market_available(self, index: int, available: bool = True):
        """Set market space availability."""
        self._arr[self._market_offset + index] = 1.0 if available else 0.0
        return self

    def is_market_available(self, index: int) -> bool:
        """Check if market space is available."""
        return self._arr[self._market_offset + index] == 1.0

    def init_all_market_available(self):
        """Initialize all market spaces as available."""
        for i in range(NUM_MARKET_SPACES):
            self._arr[self._market_offset + i] = 1.0
        return self

    # =========================================================================
    # COMPANY ACCESSORS
    # =========================================================================

    def set_company_for_auction(self, company_id: int, available: bool = True):
        """Set whether company is available for auction."""
        self._arr[self._auction_companies_offset + company_id] = 1.0 if available else 0.0
        return self

    def has_company_for_auction(self, company_id: int) -> bool:
        """Check if company is available for auction."""
        return self._arr[self._auction_companies_offset + company_id] == 1.0

    def set_company_revealed(self, company_id: int, revealed: bool = True):
        """Set whether company is revealed."""
        self._arr[self._revealed_companies_offset + company_id] = 1.0 if revealed else 0.0
        return self

    def is_company_revealed(self, company_id: int) -> bool:
        """Check if company is revealed."""
        return self._arr[self._revealed_companies_offset + company_id] == 1.0

    def set_company_removed(self, company_id: int, removed: bool = True):
        """Set whether company has been removed from game."""
        self._arr[self._removed_companies_offset + company_id] = 1.0 if removed else 0.0
        return self

    def is_company_removed(self, company_id: int) -> bool:
        """Check if company has been removed from game."""
        return self._arr[self._removed_companies_offset + company_id] == 1.0

    # =========================================================================
    # DECK ACCESSORS
    # =========================================================================

    def setup_deck(self, company_ids: list):
        """Set up deck with companies (first in list is top).

        Deck is stored as: deck_order[deck_top] is the top card.
        When drawing, we take deck_order[deck_top] and decrement deck_top.
        So first in list goes at highest index.
        """
        hidden_offset = self.state.visible_size
        deck_top_offset = hidden_offset + 2  # After active_player, num_players
        deck_order_offset = hidden_offset + 3

        if company_ids:
            self._arr[deck_top_offset] = len(company_ids) - 1
            # Reverse so first in list is at top (highest index)
            for i, cid in enumerate(reversed(company_ids)):
                self._arr[deck_order_offset + i] = cid
        else:
            self._arr[deck_top_offset] = -1
        return self

    # =========================================================================
    # TURN STATE ACCESSORS
    # =========================================================================

    def _turn_issue_remaining_offset(self) -> int:
        """Get offset to issue_remaining in turn state."""
        # turn_number(1) + end_card_flipped(1) + consecutive_passes(1) = 3
        # auction: company(36) + price(1) + high_bidder(num_players) + starter(num_players) + passed(num_players)
        auction_size = 36 + 1 + self._num_players * 3
        # dividends: corp(8) + impact(26) + remaining(8) = 42
        dividends_size = 42
        # issue: corp(8) + remaining(8)
        return 3 + auction_size + dividends_size + NUM_CORPS

    def get_turn_issue_remaining(self, corp_id: int) -> float:
        """Get issue_remaining flag for a corp from turn state."""
        offset = self._turn_offset + self._turn_issue_remaining_offset() + corp_id
        return self._arr[offset]

    def set_turn_issue_remaining(self, corp_id: int, value: float):
        """Set issue_remaining flag for a corp in turn state."""
        offset = self._turn_offset + self._turn_issue_remaining_offset() + corp_id
        self._arr[offset] = value
        return self

    def _turn_ipo_remaining_offset(self) -> int:
        """Get offset to ipo_remaining in turn state."""
        # After issue: corp(8) + remaining(8) = 16
        return self._turn_issue_remaining_offset() + NUM_CORPS + NUM_COMPANIES

    def get_turn_ipo_remaining(self, company_id: int) -> float:
        """Get ipo_remaining flag for a company from turn state."""
        # After issue_remaining(8) comes ipo_company(36), then ipo_remaining(36)
        offset = self._turn_offset + self._turn_issue_remaining_offset() + NUM_CORPS + NUM_COMPANIES + company_id
        return self._arr[offset]

    def set_turn_ipo_remaining(self, company_id: int, value: float):
        """Set ipo_remaining flag for a company in turn state."""
        # After issue_remaining(8) comes ipo_company(36), then ipo_remaining(36)
        offset = self._turn_offset + self._turn_issue_remaining_offset() + NUM_CORPS + NUM_COMPANIES + company_id
        self._arr[offset] = value
        return self
