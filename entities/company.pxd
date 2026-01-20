# cython: language_level=3
"""
Company entity declarations.

Companies can exist in exactly one location at a time. The Company handle
tracks where it currently exists in the state vector and provides efficient
transfer operations that update both the old and new locations atomically.
"""

from core.state cimport GameState


# Location type enum
cdef enum CompanyLocation:
    LOC_UNKNOWN = -1    # Not yet initialized or invalid
    LOC_DECK = 0        # In the draw deck (not visible in any flag)
    LOC_AUCTION = 1     # Available for auction
    LOC_REVEALED = 2    # Drawn this turn but not auctionable
    LOC_PLAYER = 3      # Owned by a player
    LOC_FI = 4          # Owned by Foreign Investor
    LOC_CORP = 5        # Owned by a corporation
    LOC_CORP_ACQ = 6    # In a corporation's acquisition pile
    LOC_REMOVED = 7     # Closed/removed from game


cdef class Company:
    cdef readonly int company_id
    cdef readonly str name

    # Cached offsets into state vector
    cdef int _auction_offset          # companies_for_auction[company_id]
    cdef int _revealed_offset         # companies_revealed[company_id]
    cdef int _removed_offset          # companies_removed[company_id]
    cdef int _fi_offset               # fi_companies[company_id]
    cdef int _income_offset           # company_incomes[company_id]

    # For player/corp ownership, we need base offsets + strides
    cdef int _players_offset          # Start of players section
    cdef int _player_stride           # Size of each player's data
    cdef int _player_companies_field  # Offset to owned_companies within player stride

    cdef int _corps_offset            # Start of corps section
    cdef int _corp_stride             # Size of each corp's data
    cdef int _corp_companies_field    # Offset to owned_companies within corp stride
    cdef int _corp_acq_field          # Offset to acquisition_companies within corp stride

    cdef int _num_players

    # Tracking current location (cached for O(1) access)
    cdef CompanyLocation _location
    cdef int _owner_id                # Player or corp ID when location is LOC_PLAYER/LOC_CORP/LOC_CORP_ACQ

    # Initialization
    cpdef void initialize(self, GameState state)

    # Location queries
    cpdef int get_location(self, GameState state)
    cpdef int get_owner_id(self, GameState state)
    cpdef bint is_in_deck(self, GameState state)
    cpdef bint is_for_auction(self, GameState state)
    cpdef bint is_revealed(self, GameState state)
    cpdef bint is_owned_by_player(self, GameState state, int player_id)
    cpdef bint is_owned_by_fi(self, GameState state)
    cpdef bint is_owned_by_corp(self, GameState state, int corp_id)
    cpdef bint is_in_corp_acquisition(self, GameState state, int corp_id)
    cpdef bint is_removed(self, GameState state)

    # Internal helper to scan for current location
    cdef void _scan_location(self, GameState state)

    # Transfer operations (zero old location, set new location, update cache)
    cpdef void transfer_to_player(self, GameState state, int player_id)
    cpdef void transfer_to_fi(self, GameState state)
    cpdef void transfer_to_corp(self, GameState state, int corp_id)
    cpdef void transfer_to_corp_acquisition(self, GameState state, int corp_id)
    cpdef void move_to_auction(self, GameState state)
    cpdef void set_revealed(self, GameState state, bint revealed)
    cpdef void remove_from_game(self, GameState state)
    cpdef void clear_location(self, GameState state)  # Remove from current location only

    # Static company data (from data.pyx)
    cpdef int get_face_value(self)
    cpdef int get_low_price(self)
    cpdef int get_high_price(self)
    cpdef int get_stars(self)
    cpdef int get_base_income(self)
    cpdef bint is_last_in_group(self)
    cpdef int get_synergy_with(self, int other_company_id)

    # Dynamic data from state
    cpdef int get_adjusted_income(self, GameState state)
    cpdef void set_adjusted_income(self, GameState state, int income)
