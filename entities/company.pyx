# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Company entity implementation.

Provides clean getter/setter access to company state and efficient transfer
operations. Each Company instance tracks where it exists in the state vector
for O(1) location queries and atomic transfers.
"""

from libc.math cimport lround
from core.state cimport GameState, StateLayout, PlayerFieldOffsets, CorpFieldOffsets
from core.data cimport (
    GameConstants, CASH_DIVISOR,
    get_company_face_value, get_company_low_price, get_company_high_price,
    get_company_stars, get_company_income, get_company_synergy,
    is_last_in_group as data_is_last_in_group
)

# Use constants from GameConstants (imported above)
from core.data import COMPANY_NAMES
from entities import deck as deck_module
from entities import corp as corp_module


# =============================================================================
# LOW-LEVEL FUNCTIONS (for nogil performance)
# =============================================================================

cdef inline int get_auction_company_for_slot(GameState state, int slot) noexcept nogil:
    """
    Return company_id for the Nth auction slot (by company_id order), or -1.

    Auction slots are ordered by company_id. Slot 0 maps to the lowest
    company_id that is available for auction, slot 1 to the next lowest, etc.

    Args:
        state: Game state to query
        slot: Slot index (0 to MAX_AUCTION_SLOTS-1)

    Returns:
        Company ID for the slot, or -1 if slot index is out of range
    """
    cdef int count = 0
    cdef int company_id
    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if state._is_company_for_auction(company_id):
            if count == slot:
                return company_id
            count += 1
    return -1


# =============================================================================
# COMPANY CLASS
# =============================================================================

cdef class Company:
    """
    Entity handle for accessing company state.

    Companies are instantiated once at module load with their company_id.
    Offsets are computed on first access to a GameState via initialize().
    All methods take GameState as first argument for stateless operation.

    Company location is stored in hidden state for O(1) access across multiple
    GameState instances (essential for MCTS). The hidden state stores both
    location type (CompanyLocation enum) and owner_id (player_id or corp_id).
    When transferring, the old visible flag is cleared and the new visible
    flag and hidden state are updated atomically.
    """

    def __cinit__(self, int company_id, str name):
        self.company_id = company_id
        self.name = name
        self._num_players = 0
        self._hidden_location_offset = -1
        self._hidden_owner_id_offset = -1

    cpdef void initialize(self, GameState state):
        """
        Initialize offsets from state layout. Call once when starting a new game.

        This must be called before using any other methods on this Company instance.
        """
        cdef StateLayout layout = state._layout
        cdef PlayerFieldOffsets player_fields = state._player_fields
        cdef CorpFieldOffsets corp_fields = state._corp_fields

        self._num_players = state._num_players

        # Direct offsets for this company's flags
        self._auction_offset = layout.auction_companies_offset + self.company_id
        self._revealed_offset = layout.revealed_companies_offset + self.company_id
        self._removed_offset = layout.removed_companies_offset + self.company_id
        self._fi_offset = layout.fi_offset + 1 + self.company_id  # +1 for fi_cash
        self._income_offset = layout.company_incomes_offset + self.company_id

        # Player ownership requires computing per-player offsets
        self._players_offset = layout.players_offset
        self._player_stride = layout.player_stride
        self._player_companies_field = player_fields.owned_companies

        # Corp ownership requires computing per-corp offsets
        self._corps_offset = layout.corps_offset
        self._corp_stride = layout.corp_stride
        self._corp_companies_field = corp_fields.owned_companies
        self._corp_acq_field = corp_fields.acquisition_companies

        # Hidden state offsets for O(1) location access
        self._hidden_location_offset = layout.hidden_company_locations_offset + self.company_id
        self._hidden_owner_id_offset = layout.hidden_company_owner_ids_offset + self.company_id

    # =========================================================================
    # HIDDEN STATE LOCATION ACCESS
    # =========================================================================

    cdef int _get_hidden_location(self, GameState state) noexcept nogil:
        """Get company location from hidden state. O(1) access."""
        return <int>state._data[self._hidden_location_offset]

    cdef int _get_hidden_owner_id(self, GameState state) noexcept nogil:
        """Get company owner ID from hidden state. O(1) access."""
        return <int>state._data[self._hidden_owner_id_offset]

    cdef void _set_hidden_location(self, GameState state, int location, int owner_id) noexcept nogil:
        """Set company location and owner in hidden state."""
        state._data[self._hidden_location_offset] = <float>location
        state._data[self._hidden_owner_id_offset] = <float>owner_id

    # =========================================================================
    # LOCATION QUERIES
    # =========================================================================

    cpdef int get_location(self, GameState state):
        """Get current location type. Returns CompanyLocation enum value."""
        return self._get_hidden_location(state)

    cpdef int get_owner_id(self, GameState state):
        """Get owner ID (player or corp) if applicable, -1 otherwise."""
        return self._get_hidden_owner_id(state)

    cpdef bint is_in_deck(self, GameState state):
        """Check if company is in the draw deck."""
        return self._get_hidden_location(state) == LOC_DECK

    cpdef bint is_for_auction(self, GameState state):
        """Check if company is available for auction."""
        return state._data[self._auction_offset] == 1.0

    cpdef bint is_revealed(self, GameState state):
        """Check if company was revealed this turn (drawn but not auctionable)."""
        return state._data[self._revealed_offset] == 1.0

    cpdef bint is_owned_by_player(self, GameState state, int player_id):
        """Check if company is owned by specific player."""
        if player_id < 0 or player_id >= self._num_players:
            return False
        return state._data[self._players_offset + player_id * self._player_stride + self._player_companies_field + self.company_id] == 1.0

    cpdef bint is_owned_by_fi(self, GameState state):
        """Check if company is owned by Foreign Investor."""
        return state._data[self._fi_offset] == 1.0

    cpdef bint is_owned_by_corp(self, GameState state, int corp_id):
        """Check if company is owned by specific corporation."""
        if corp_id < 0 or corp_id >= GameConstants.NUM_CORPS:
            return False
        return state._data[self._corps_offset + corp_id * self._corp_stride + self._corp_companies_field + self.company_id] == 1.0

    cpdef bint is_in_corp_acquisition(self, GameState state, int corp_id):
        """Check if company is in specific corporation's acquisition pile."""
        if corp_id < 0 or corp_id >= GameConstants.NUM_CORPS:
            return False
        return state._data[self._corps_offset + corp_id * self._corp_stride + self._corp_acq_field + self.company_id] == 1.0

    cpdef bint is_removed(self, GameState state):
        """Check if company has been removed from the game."""
        return state._data[self._removed_offset] == 1.0

    # =========================================================================
    # TRANSFER OPERATIONS
    # =========================================================================

    cdef void _clear_visible_flag(self, GameState state) noexcept nogil:
        """Clear company's current visible state flag based on hidden location."""
        cdef int location = self._get_hidden_location(state)
        cdef int owner_id = self._get_hidden_owner_id(state)

        if location == LOC_AUCTION:
            state._data[self._auction_offset] = 0.0
        elif location == LOC_REVEALED:
            state._data[self._revealed_offset] = 0.0
        elif location == LOC_REMOVED:
            state._data[self._removed_offset] = 0.0
        elif location == LOC_FI:
            state._data[self._fi_offset] = 0.0
        elif location == LOC_PLAYER:
            state._data[self._players_offset + owner_id * self._player_stride + self._player_companies_field + self.company_id] = 0.0
        elif location == LOC_CORP:
            state._data[self._corps_offset + owner_id * self._corp_stride + self._corp_companies_field + self.company_id] = 0.0
        elif location == LOC_CORP_ACQ:
            state._data[self._corps_offset + owner_id * self._corp_stride + self._corp_acq_field + self.company_id] = 0.0
        # LOC_DECK has no flag to clear

    cdef void _remove_from_deck_if_needed(self, GameState state):
        """If company is currently in the deck, remove it from the deck order array."""
        if self._get_hidden_location(state) == LOC_DECK:
            deck_module.DECK.remove(state, self.company_id)

    cpdef void transfer_to_player(self, GameState state, int player_id):
        """Transfer company to player ownership."""
        if player_id < 0 or player_id >= self._num_players:
            return
        cdef int old_loc = self._get_hidden_location(state)
        cdef int old_owner = self._get_hidden_owner_id(state)
        self._remove_from_deck_if_needed(state)
        self._clear_visible_flag(state)
        state._data[self._players_offset + player_id * self._player_stride + self._player_companies_field + self.company_id] = 1.0
        self._set_hidden_location(state, LOC_PLAYER, player_id)
        if old_loc == LOC_CORP and corp_module.CORPS[old_owner].is_active(state):
            corp_module.CORPS[old_owner].recalculate_stars(state)
            corp_module.CORPS[old_owner].calculate_income(state)

    cpdef void transfer_to_fi(self, GameState state):
        """Transfer company to Foreign Investor ownership."""
        cdef int old_loc = self._get_hidden_location(state)
        cdef int old_owner = self._get_hidden_owner_id(state)
        self._remove_from_deck_if_needed(state)
        self._clear_visible_flag(state)
        state._data[self._fi_offset] = 1.0
        self._set_hidden_location(state, LOC_FI, -1)
        if old_loc == LOC_CORP and corp_module.CORPS[old_owner].is_active(state):
            corp_module.CORPS[old_owner].recalculate_stars(state)
            corp_module.CORPS[old_owner].calculate_income(state)

    cpdef void transfer_to_corp(self, GameState state, int corp_id):
        """Transfer company to corporation ownership."""
        if corp_id < 0 or corp_id >= GameConstants.NUM_CORPS:
            return
        cdef int old_loc = self._get_hidden_location(state)
        cdef int old_owner = self._get_hidden_owner_id(state)
        self._remove_from_deck_if_needed(state)
        self._clear_visible_flag(state)
        state._data[self._corps_offset + corp_id * self._corp_stride + self._corp_companies_field + self.company_id] = 1.0
        self._set_hidden_location(state, LOC_CORP, corp_id)
        if old_loc == LOC_CORP and old_owner != corp_id and corp_module.CORPS[old_owner].is_active(state):
            corp_module.CORPS[old_owner].recalculate_stars(state)
            corp_module.CORPS[old_owner].calculate_income(state)
        if corp_module.CORPS[corp_id].is_active(state):
            corp_module.CORPS[corp_id].recalculate_stars(state)
            corp_module.CORPS[corp_id].calculate_income(state)

    cpdef void transfer_to_corp_acquisition(self, GameState state, int corp_id):
        """Transfer company to corporation's acquisition pile."""
        if corp_id < 0 or corp_id >= GameConstants.NUM_CORPS:
            return
        cdef int old_loc = self._get_hidden_location(state)
        cdef int old_owner = self._get_hidden_owner_id(state)
        self._remove_from_deck_if_needed(state)
        self._clear_visible_flag(state)
        state._data[self._corps_offset + corp_id * self._corp_stride + self._corp_acq_field + self.company_id] = 1.0
        self._set_hidden_location(state, LOC_CORP_ACQ, corp_id)
        # Acq zone companies aren't counted in stars or income, but old corp owner lost a company
        if old_loc == LOC_CORP and corp_module.CORPS[old_owner].is_active(state):
            corp_module.CORPS[old_owner].recalculate_stars(state)
            corp_module.CORPS[old_owner].calculate_income(state)

    cpdef void move_to_auction(self, GameState state):
        """Make company available for auction."""
        self._remove_from_deck_if_needed(state)
        self._clear_visible_flag(state)
        state._data[self._auction_offset] = 1.0
        self._set_hidden_location(state, LOC_AUCTION, -1)

    cpdef void mark_revealed(self, GameState state):
        """Mark company as revealed this turn (drawn but not auctionable)."""
        self._remove_from_deck_if_needed(state)
        self._clear_visible_flag(state)
        state._data[self._revealed_offset] = 1.0
        self._set_hidden_location(state, LOC_REVEALED, -1)

    cpdef void remove_from_game(self, GameState state):
        """Remove company from the game (closed)."""
        cdef int old_loc = self._get_hidden_location(state)
        cdef int old_owner = self._get_hidden_owner_id(state)
        self._clear_visible_flag(state)
        state._data[self._removed_offset] = 1.0
        self._set_hidden_location(state, LOC_REMOVED, -1)
        if old_loc == LOC_CORP and corp_module.CORPS[old_owner].is_active(state):
            corp_module.CORPS[old_owner].recalculate_stars(state)
            corp_module.CORPS[old_owner].calculate_income(state)

    cpdef void exclude_from_game(self, GameState state):
        """Mark company as excluded during game init (hidden state only).

        Used for companies not included in the deck for this player count.
        Only updates the hidden location — visible state is left untouched
        so the NN cannot infer which companies were excluded (and therefore
        which are in the hidden deck).
        """
        self._set_hidden_location(state, LOC_REMOVED, -1)

    # =========================================================================
    # STATIC COMPANY DATA
    # =========================================================================

    cpdef int get_face_value(self):
        """Get company's face value."""
        return get_company_face_value(self.company_id)

    cpdef int get_low_price(self):
        """Get company's low acquisition price."""
        return get_company_low_price(self.company_id)

    cpdef int get_high_price(self):
        """Get company's high acquisition price."""
        return get_company_high_price(self.company_id)

    cpdef int get_stars(self):
        """Get company's star rating."""
        return get_company_stars(self.company_id)

    cpdef int get_base_income(self):
        """Get company's base income (before cost of ownership)."""
        return get_company_income(self.company_id)

    cpdef bint is_last_in_group(self):
        """Check if company is last in its color group (triggers CoO increase)."""
        return data_is_last_in_group(self.company_id)

    cpdef int get_synergy_with(self, int other_company_id):
        """Get synergy bonus when this company is paired with another."""
        if other_company_id < 0 or other_company_id >= GameConstants.NUM_COMPANIES:
            return 0
        return get_company_synergy(self.company_id, other_company_id)

    # =========================================================================
    # DYNAMIC DATA FROM STATE
    # =========================================================================

    cpdef int get_adjusted_income(self, GameState state):
        """Get company's adjusted income (after cost of ownership).

        Uses lround for proper rounding of negative values (high CoO can make income negative).
        """
        return <int>lround(state._data[self._income_offset] * CASH_DIVISOR)

    cpdef void set_adjusted_income(self, GameState state, int income):
        """Set company's adjusted income."""
        state._data[self._income_offset] = <float>income / CASH_DIVISOR


# =============================================================================
# GLOBAL COMPANY INSTANCES
# =============================================================================

# Initialize companies and store in both list (by ID) and dict (by name)
COMPANIES = [Company(i, name) for i, name in enumerate(COMPANY_NAMES)]
COMPANIES_BY_NAME = {name: COMPANIES[i] for i, name in enumerate(COMPANY_NAMES)}
