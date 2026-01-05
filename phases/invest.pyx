# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Invest phase implementation.

The Invest phase (Phase 1) allows players to:
- Buy one share of an active corporation
- Sell one share of a corporation they own
- Start an auction for a company
- Pass

The phase ends when all players pass consecutively.

During an auction (BID_IN_AUCTION sub-phase):
- Players can raise the bid or leave the auction
- When one player remains, they win and pay their bid
- Turn passes to player after the auction STARTER (not winner)
"""

cimport cython
from libc.string cimport memset

from state cimport GameState, PHASE_INVEST, PHASE_BID_IN_AUCTION, PHASE_WRAP_UP
from data cimport (
    NUM_COMPANIES, NUM_CORPS, NUM_MARKET_SPACES,
    get_company_face_value, get_company_high_price, get_market_price,
)

# Import shared helpers
from helpers.player cimport (
    PlayerOffsets, get_player_offsets,
    get_player_cash, set_player_cash, add_player_cash,
    get_player_shares, set_player_shares,
    player_owns_company, set_player_owns_company,
    get_share_buys, increment_share_buys,
    get_share_sells, increment_share_sells,
    get_roundtrips, clear_roundtrip_tracking
)
from helpers.corp cimport (
    CorpOffsets, get_corp_offsets,
    is_corp_active, get_corp_bank_shares, set_corp_bank_shares,
    get_corp_share_price, get_corp_price_index, set_corp_price_index,
    is_corp_in_receivership, set_corp_in_receivership, handle_corp_bankruptcy
)
from helpers.player cimport update_all_player_net_worths
from helpers.market cimport (
    is_market_space_available, set_market_space_available,
    find_next_higher_price_index, find_next_lower_price_index
)

# Constants
DEF MAX_ROUNDTRIPS = 2
DEF AUCTION_CAP = 20


# =============================================================================
# COMPANY LOCATION HELPERS (invest-specific)
# =============================================================================

cdef inline bint is_company_for_auction(float* auction_companies, int company_id) noexcept nogil:
    """Check if company is available for auction."""
    return auction_companies[company_id] == 1.0


cdef inline void remove_company_from_auction(float* auction_companies, int company_id) noexcept nogil:
    """Remove company from auction pool."""
    auction_companies[company_id] = 0.0


# =============================================================================
# INVEST PHASE ACTIONS
# =============================================================================

cdef void invest_buy_share(
    float* player, PlayerOffsets* po,
    float* corp, CorpOffsets* co,
    float* market,
    float* hidden_price_indices,
    int corp_id
) noexcept nogil:
    """
    Buy one share of a corporation.

    - Corp price moves up one space
    - Player pays the NEW (higher) price
    - Corp's bank_shares decreases
    - Player's owned_shares increases
    """
    cdef int current_index = get_corp_price_index(corp, co)

    # Find next higher available price
    cdef int new_index = find_next_higher_price_index(market, current_index)
    cdef int new_price = get_market_price(new_index)

    # Update market availability
    # Don't release index 0 (inactive) or 26 (multiple corps can be at $75)
    if current_index > 0 and current_index < NUM_MARKET_SPACES - 1:
        set_market_space_available(market, current_index, True)
    # Don't take index 26 ($75) - multiple corps can be there
    if new_index < NUM_MARKET_SPACES - 1:
        set_market_space_available(market, new_index, False)

    # Update corp price
    set_corp_price_index(corp, co, new_index, hidden_price_indices, corp_id)

    # Transfer share
    cdef int bank_shares = get_corp_bank_shares(corp, co)
    set_corp_bank_shares(corp, co, bank_shares - 1)

    cdef int player_shares = get_player_shares(player, po, corp_id)
    set_player_shares(player, po, corp_id, player_shares + 1)

    # Player pays new price
    add_player_cash(player, po, -new_price)

    # Track for round-trip limit
    increment_share_buys(player, po, corp_id)


cdef void invest_sell_share(
    float* player, PlayerOffsets* po,
    float* corp, CorpOffsets* co,
    float* market,
    float* hidden_price_indices,
    int corp_id
) noexcept nogil:
    """
    Sell one share of a corporation.

    - Corp price moves down one space
    - Player receives the NEW (lower) price
    - Corp's bank_shares increases
    - Player's owned_shares decreases
    """
    cdef int current_index = get_corp_price_index(corp, co)

    # Find next lower available price
    cdef int new_index = find_next_lower_price_index(market, current_index)
    cdef int new_price = get_market_price(new_index)

    # Update market availability
    # Don't release index 0 (inactive) or 26 (multiple corps can be at $75)
    if current_index > 0 and current_index < NUM_MARKET_SPACES - 1:
        set_market_space_available(market, current_index, True)
    # Don't take index 0 (bankruptcy) or 26 ($75, multiple allowed)
    if new_index > 0 and new_index < NUM_MARKET_SPACES - 1:
        set_market_space_available(market, new_index, False)

    # Update corp price
    set_corp_price_index(corp, co, new_index, hidden_price_indices, corp_id)

    # Transfer share
    cdef int bank_shares = get_corp_bank_shares(corp, co)
    set_corp_bank_shares(corp, co, bank_shares + 1)

    cdef int player_shares = get_player_shares(player, po, corp_id)
    set_player_shares(player, po, corp_id, player_shares - 1)

    # Player receives new price
    add_player_cash(player, po, new_price)

    # Track for round-trip limit
    increment_share_sells(player, po, corp_id)


# =============================================================================
# VALID ACTION CHECKING
# =============================================================================

cdef bint can_buy_share(
    float* player, PlayerOffsets* po,
    float* corp, CorpOffsets* co,
    float* market,
    int corp_id
) noexcept nogil:
    """Check if player can buy a share of corp."""
    # Corp must be active
    if not is_corp_active(corp, co):
        return False

    # Corp must have shares available
    if get_corp_bank_shares(corp, co) <= 0:
        return False

    # Check round-trip limit
    if get_roundtrips(player, po, corp_id) >= MAX_ROUNDTRIPS:
        return False

    # Player must afford the buy price (next higher)
    cdef int current_index = get_corp_price_index(corp, co)
    cdef int buy_index = find_next_higher_price_index(market, current_index)
    cdef int buy_price = get_market_price(buy_index)

    if get_player_cash(player, po) < buy_price:
        return False

    return True


cdef bint can_sell_share(
    float* player, PlayerOffsets* po,
    float* corp, CorpOffsets* co,
    int corp_id
) noexcept nogil:
    """Check if player can sell a share of corp."""
    # Player must own at least one share
    if get_player_shares(player, po, corp_id) <= 0:
        return False

    # Check round-trip limit
    if get_roundtrips(player, po, corp_id) >= MAX_ROUNDTRIPS:
        return False

    return True


cdef bint can_start_auction(
    float* player, PlayerOffsets* po,
    float* auction_companies,
    int company_id,
    int bid_offset
) noexcept nogil:
    """Check if player can start auction for company at bid."""
    # Company must be available
    if not is_company_for_auction(auction_companies, company_id):
        return False

    # Bid must be within cap
    if bid_offset < 0 or bid_offset >= AUCTION_CAP:
        return False

    # Player must afford the bid
    cdef int face_value = get_company_face_value(company_id)
    cdef int bid = face_value + bid_offset

    if get_player_cash(player, po) < bid:
        return False

    return True


cdef bint can_raise_bid(
    float* player, PlayerOffsets* po,
    int company_id,
    int current_min_bid,
    int bid_offset
) noexcept nogil:
    """Check if player can raise to bid_offset during auction."""
    cdef int face_value = get_company_face_value(company_id)
    cdef int bid = face_value + bid_offset

    # Must beat current minimum bid
    if bid < current_min_bid:
        return False

    # Must be within cap
    if bid_offset >= AUCTION_CAP:
        return False

    # Player must afford it
    if get_player_cash(player, po) < bid:
        return False

    return True


# =============================================================================
# INVEST PHASE CLASS
# =============================================================================

cdef class InvestPhase:
    """
    Invest phase handler.

    Provides high-level methods that operate on GameState.
    Caches offset computations for efficiency.
    """
    # Attributes declared in invest.pxd

    def __cinit__(self, int num_players):
        self._num_players = num_players
        self._po = get_player_offsets(num_players)
        self._co = get_corp_offsets()

    # =========================================================================
    # POINTER EXTRACTION HELPERS
    # =========================================================================

    cdef float* _get_player(self, GameState state, int player_id) noexcept nogil:
        """Get pointer to player data."""
        return state._player_ptr(player_id)

    cdef float* _get_corp(self, GameState state, int corp_id) noexcept nogil:
        """Get pointer to corp data."""
        return state._corp_ptr(corp_id)

    cdef float* _get_market(self, GameState state) noexcept nogil:
        """Get pointer to market data."""
        return state._market_ptr()

    cdef float* _get_auction_companies(self, GameState state) noexcept nogil:
        """Get pointer to companies available for auction."""
        return state._data + state._layout.auction_companies_offset

    cdef int _get_active_player(self, GameState state) noexcept nogil:
        """Get active player ID."""
        return state._get_active_player()

    # =========================================================================
    # BUY SHARE
    # =========================================================================

    cpdef bint can_do_buy_share(self, GameState state, int corp_id):
        """Check if active player can buy a share of corp."""
        cdef int player_id = self._get_active_player(state)
        cdef float* player = self._get_player(state, player_id)
        cdef float* corp = self._get_corp(state, corp_id)
        cdef float* market = self._get_market(state)

        return can_buy_share(player, &self._po, corp, &self._co, market, corp_id)

    cpdef void do_buy_share(self, GameState state, int corp_id):
        """Execute buy share action for active player."""
        cdef int player_id = self._get_active_player(state)
        cdef float* player = self._get_player(state, player_id)
        cdef float* corp = self._get_corp(state, corp_id)
        cdef float* market = self._get_market(state)

        invest_buy_share(player, &self._po, corp, &self._co, market, state._hidden_price_indices_ptr(), corp_id)

        # Clear consecutive passes (action taken)
        state.clear_consecutive_passes()

        # Update presidency if needed
        self._update_presidency(state, corp_id)

        # Advance to next player
        state.advance_active_player()

    # =========================================================================
    # SELL SHARE
    # =========================================================================

    cpdef bint can_do_sell_share(self, GameState state, int corp_id):
        """Check if active player can sell a share of corp."""
        cdef int player_id = self._get_active_player(state)
        cdef float* player = self._get_player(state, player_id)
        cdef float* corp = self._get_corp(state, corp_id)

        return can_sell_share(player, &self._po, corp, &self._co, corp_id)

    cpdef void do_sell_share(self, GameState state, int corp_id):
        """Execute sell share action for active player."""
        cdef int player_id = self._get_active_player(state)
        cdef float* player = self._get_player(state, player_id)
        cdef float* corp = self._get_corp(state, corp_id)
        cdef float* market = self._get_market(state)
        cdef int old_price_index = get_corp_price_index(corp, &self._co)

        invest_sell_share(player, &self._po, corp, &self._co, market, state._hidden_price_indices_ptr(), corp_id)

        # Check for bankruptcy (price fell to index 0 = $0)
        cdef int new_price_index = get_corp_price_index(corp, &self._co)
        if new_price_index == 0:
            handle_corp_bankruptcy(state, corp_id, old_price_index, self._num_players)
            update_all_player_net_worths(state, self._num_players)
            # Clear consecutive passes (action taken)
            state.clear_consecutive_passes()
            # Advance to next player
            state.advance_active_player()
            return

        # Clear consecutive passes (action taken)
        state.clear_consecutive_passes()

        # Update presidency if needed
        self._update_presidency(state, corp_id)

        # Advance to next player
        state.advance_active_player()

    # =========================================================================
    # START AUCTION
    # =========================================================================

    cpdef bint can_do_start_auction(self, GameState state, int company_id, int bid_offset):
        """Check if active player can start auction for company at bid."""
        cdef int player_id = self._get_active_player(state)
        cdef float* player = self._get_player(state, player_id)
        cdef float* auction_companies = self._get_auction_companies(state)

        return can_start_auction(player, &self._po, auction_companies, company_id, bid_offset)

    cpdef void do_start_auction(self, GameState state, int company_id, int bid_offset):
        """Start an auction for company at bid_offset over face value."""
        cdef int player_id = self._get_active_player(state)
        cdef int face_value = get_company_face_value(company_id)
        cdef int bid = face_value + bid_offset

        # Set auction state
        state.set_auction_company(company_id)
        state.set_auction_price(bid)
        state.set_auction_high_bidder(player_id)
        state.set_auction_starter(player_id)
        state.init_auction_passed()  # All players in auction initially

        # Switch to auction phase
        state.set_phase(PHASE_BID_IN_AUCTION)

        # Next player gets first chance to bid
        state.advance_active_player()

    # =========================================================================
    # AUCTION BIDDING
    # =========================================================================

    cpdef bint can_do_raise_bid(self, GameState state, int bid_offset):
        """Check if active player can raise bid during auction."""
        cdef int player_id = self._get_active_player(state)
        cdef float* player = self._get_player(state, player_id)

        cdef int company_id = state.get_auction_company()
        if company_id < 0:
            return False

        # Player must still be in auction
        if state.get_auction_passed(player_id):
            return False

        # Current minimum is one above current high bid
        cdef int current_bid = state.get_auction_price()
        cdef int min_bid = current_bid + 1

        return can_raise_bid(player, &self._po, company_id, min_bid, bid_offset)

    cpdef void do_raise_bid(self, GameState state, int bid_offset):
        """Raise the bid during auction."""
        cdef int player_id = self._get_active_player(state)
        cdef int company_id = state.get_auction_company()
        cdef int face_value = get_company_face_value(company_id)
        cdef int new_bid = face_value + bid_offset

        # Update auction state
        state.set_auction_price(new_bid)
        state.set_auction_high_bidder(player_id)

        # Advance to next player still in auction
        self._advance_auction_player(state)

    cpdef void do_leave_auction(self, GameState state):
        """Leave the current auction."""
        cdef int player_id = self._get_active_player(state)

        # Mark player as passed in auction
        state.set_auction_passed(player_id, True)

        # Check if auction is resolved (only one player left)
        if self._check_auction_resolved(state):
            self._resolve_auction(state)
        else:
            # Advance to next player still in auction
            self._advance_auction_player(state)

    # =========================================================================
    # PASS
    # =========================================================================

    cpdef void do_pass(self, GameState state):
        """Pass during invest phase."""
        cdef int next_player
        cdef float* player

        state.increment_consecutive_passes()

        # Check if all players passed
        if state.get_consecutive_passes() >= self._num_players:
            self._end_invest_phase(state)
        else:
            # Clear round-trip tracking for next player
            next_player = (self._get_active_player(state) + 1) % self._num_players
            player = self._get_player(state, next_player)
            clear_roundtrip_tracking(player, &self._po)

            state.advance_active_player()

    # =========================================================================
    # VALID ACTIONS MASK
    # =========================================================================

    cpdef dict get_valid_actions(self, GameState state):
        """
        Get mask of valid actions in current state.

        Returns a dict with action categories and their valid indices.
        """
        cdef int phase = state.get_phase()

        if phase == PHASE_INVEST:
            return self._get_invest_valid_actions(state)
        elif phase == PHASE_BID_IN_AUCTION:
            return self._get_auction_valid_actions(state)
        else:
            return {}

    cdef dict _get_invest_valid_actions(self, GameState state):
        """Get valid actions during invest phase."""
        cdef dict result = {
            'buy': [],
            'sell': [],
            'auction': [],
            'pass': True
        }
        cdef int i, j
        cdef int player_id = self._get_active_player(state)
        cdef float* player = self._get_player(state, player_id)
        cdef float* market = self._get_market(state)
        cdef float* auction_companies = self._get_auction_companies(state)

        # Check each corp for buy/sell
        for i in range(NUM_CORPS):
            corp = self._get_corp(state, i)
            if can_buy_share(player, &self._po, corp, &self._co, market, i):
                result['buy'].append(i)
            if can_sell_share(player, &self._po, corp, &self._co, i):
                result['sell'].append(i)

        # Check each company + bid offset for auction
        for i in range(NUM_COMPANIES):
            if is_company_for_auction(auction_companies, i):
                for j in range(AUCTION_CAP):
                    if can_start_auction(player, &self._po, auction_companies, i, j):
                        result['auction'].append((i, j))

        return result

    cdef dict _get_auction_valid_actions(self, GameState state):
        """Get valid actions during auction phase."""
        cdef dict result = {
            'raise': [],
            'leave': True
        }
        cdef int j
        cdef int player_id = self._get_active_player(state)
        cdef float* player = self._get_player(state, player_id)

        cdef int company_id = state.get_auction_company()
        cdef int current_bid = state.get_auction_price()
        cdef int min_bid = current_bid + 1

        # Check each bid offset
        for j in range(AUCTION_CAP):
            if can_raise_bid(player, &self._po, company_id, min_bid, j):
                result['raise'].append(j)

        return result

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    cdef void _update_presidency(self, GameState state, int corp_id) noexcept:
        """Update presidency after share change."""
        cdef int i
        cdef int max_shares = 0
        cdef int president = -1
        cdef float* player
        cdef float* corp
        cdef int shares

        # Find player with most shares
        for i in range(self._num_players):
            player = self._get_player(state, i)
            shares = get_player_shares(player, &self._po, corp_id)
            if shares > max_shares:
                max_shares = shares
                president = i

        # Update presidency flags
        for i in range(self._num_players):
            player = self._get_player(state, i)
            if i == president and max_shares > 0:
                player[self._po.is_president + corp_id] = 1.0
            else:
                player[self._po.is_president + corp_id] = 0.0

        # If no player has any shares, corp enters receivership
        corp = self._get_corp(state, corp_id)
        if max_shares == 0:
            set_corp_in_receivership(corp, &self._co, True)
        else:
            set_corp_in_receivership(corp, &self._co, False)

    cdef void _advance_auction_player(self, GameState state) noexcept:
        """Advance to next player still in auction (using turn order)."""
        cdef int current = self._get_active_player(state)
        cdef int current_position = state.get_player_turn_order(current)
        cdef int next_position = current_position
        cdef int next_player
        cdef int checked = 0

        while checked < self._num_players:
            next_position = (next_position + 1) % self._num_players
            next_player = state.get_player_at_turn_order(next_position)
            checked += 1
            if not state.get_auction_passed(next_player):
                break

        state._set_active_player(next_player)

    cdef bint _check_auction_resolved(self, GameState state) noexcept nogil:
        """Check if only one player remains in auction."""
        cdef int remaining = 0
        cdef int i

        for i in range(self._num_players):
            if not state.get_auction_passed(i):
                remaining += 1
                if remaining > 1:
                    return False

        return remaining == 1

    cdef void _resolve_auction(self, GameState state) noexcept:
        """Resolve auction - winner gets company, pays bid."""
        cdef int winner = state.get_auction_high_bidder()
        cdef int company_id = state.get_auction_company()
        cdef int bid = state.get_auction_price()
        cdef int starter = state.get_auction_starter()

        # Give company to winner
        cdef float* player = self._get_player(state, winner)
        set_player_owns_company(player, &self._po, company_id, True)

        # Deduct payment
        add_player_cash(player, &self._po, -bid)

        # Remove company from auction pool
        cdef float* auction_companies = self._get_auction_companies(state)
        remove_company_from_auction(auction_companies, company_id)

        # Draw new company to revealed pile (becomes available in Wrap-up phase)
        state.draw_company_to_revealed()

        # Clear auction state
        state.clear_auction_state()

        # Return to invest phase
        state.set_phase(PHASE_INVEST)

        # Clear consecutive passes (action was taken)
        state.clear_consecutive_passes()

        # Next player is after the auction STARTER (not winner) in turn order
        cdef int starter_position = state.get_player_turn_order(starter)
        cdef int next_position = (starter_position + 1) % self._num_players
        cdef int next_player = state.get_player_at_turn_order(next_position)
        state._set_active_player(next_player)

    cdef void _draw_company_to_auction(self, GameState state) noexcept:
        """Draw top company from deck to auction pool."""
        cdef int deck_top = state._get_deck_top()
        if deck_top < 0:
            return  # Deck empty

        cdef int company_id = state._get_deck_company(deck_top)
        if company_id < 0:
            return  # Invalid

        # Add to auction pool
        cdef float* auction_companies = self._get_auction_companies(state)
        auction_companies[company_id] = 1.0

        # Update deck top
        state._set_deck_top(deck_top - 1)

    cdef void _end_invest_phase(self, GameState state) noexcept:
        """Transition to Wrap Up phase."""
        state.set_phase(PHASE_WRAP_UP)
        state.clear_consecutive_passes()


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

# Cached phase handler per player count
cdef dict _phase_handlers = {}

def get_phase_handler(int num_players):
    """Get or create InvestPhase handler for player count."""
    if num_players not in _phase_handlers:
        _phase_handlers[num_players] = InvestPhase(num_players)
    return _phase_handlers[num_players]
