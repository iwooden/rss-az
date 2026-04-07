# cython: language_level=3
"""
Company entity declarations.

Each company exists in exactly one location at any time. The Company handle
caches the offsets it needs into the compact GameState array and provides
atomic transfer operations that update both the location enum and the owner
id together.
"""

from core.state cimport GameState


# Location type enum (cpdef for Python access in tests)
#
# LOC_DECK = 0 is the zero-init default — a freshly allocated int16 state
# starts with every company "in the deck". LOC_EXCLUDED is the explicit
# sentinel for companies that are not part of the deck for the current
# player count; it must be set explicitly during deck setup so the engine
# can distinguish "in the live deck" from "removed from the game before it
# started".
cpdef enum CompanyLocation:
    LOC_DECK = 0        # In the draw deck (zero-init default)
    LOC_AUCTION = 1     # Available for auction
    LOC_REVEALED = 2    # Drawn this turn but not yet auctionable
    LOC_PLAYER = 3      # Owned by a player (owner_id = player_id)
    LOC_FI = 4          # Owned by Foreign Investor (owner_id = -1)
    LOC_CORP = 5        # Owned by a corporation (owner_id = corp_id)
    LOC_CORP_ACQ = 6    # In a corporation's acquisition pile (owner_id = corp_id)
    LOC_REMOVED = 7     # Closed during the game
    LOC_EXCLUDED = 8    # Excluded from the deck at game setup (player count)


cdef class Company:
    cdef readonly int company_id
    cdef readonly str name

    # Cached absolute offsets into the compact state array. Companies are
    # tracked entirely through company_locations and company_owner_ids —
    # there are no longer any per-section ownership flags to update.
    cdef int _location_offset         # company_locations[company_id]
    cdef int _owner_id_offset         # company_owner_ids[company_id]
    cdef int _income_offset           # company_incomes[company_id]

    cdef int _num_players

    # Initialization
    cpdef void initialize(self, GameState state)

    # Location queries
    cpdef int get_location(self, GameState state)
    cpdef int get_owner_id(self, GameState state)
    cpdef bint is_in_deck(self, GameState state)
    cpdef bint is_excluded(self, GameState state)
    cpdef bint is_for_auction(self, GameState state)
    cpdef bint is_revealed(self, GameState state)
    cpdef bint is_owned_by_player(self, GameState state, int player_id)
    cpdef bint is_owned_by_fi(self, GameState state)
    cpdef bint is_owned_by_corp(self, GameState state, int corp_id)
    cpdef bint is_in_corp_acquisition(self, GameState state, int corp_id)
    cpdef bint is_removed(self, GameState state)
    cpdef bint is_acquired(self, GameState state)

    # Internal helpers
    cdef int _get_location(self, GameState state) noexcept nogil
    cdef int _get_owner_id(self, GameState state) noexcept nogil
    cdef void _set_location(self, GameState state, int location, int owner_id) noexcept nogil
    cdef void _remove_from_deck_if_needed(self, GameState state)
    cdef void _recalc_after_change(self, GameState state, int location, int owner_id)
    cdef void _move(self, GameState state, int new_loc, int new_owner)

    # Transfer operations (update location + owner, recalc downstream entities)
    cpdef void transfer_to_player(self, GameState state, int player_id)
    cpdef void transfer_to_fi(self, GameState state)
    cpdef void transfer_to_corp(self, GameState state, int corp_id)
    cpdef void transfer_to_corp_acquisition(self, GameState state, int corp_id)
    cpdef void move_to_auction(self, GameState state)
    cpdef void mark_revealed(self, GameState state)
    cpdef void remove_from_game(self, GameState state)
    cpdef void exclude_from_game(self, GameState state)

    # Static company data (read directly from core.data arrays)
    cpdef int get_face_value(self)
    cpdef int get_low_price(self)
    cpdef int get_high_price(self)
    cpdef int get_stars(self)
    cpdef int get_base_income(self)
    cpdef bint is_last_in_group(self)
    cpdef int get_synergy_with(self, int other_company_id)

    # Dynamic data from state (raw int16, no normalization)
    cpdef int get_adjusted_income(self, GameState state)
    cpdef void set_adjusted_income(self, GameState state, int income)
