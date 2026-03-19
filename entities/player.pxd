# cython: language_level=3
"""
Player entity declarations.

Provides the Player class for accessing player state in the game vector.
The class uses cached offsets for efficient nogil access in performance-critical
code paths (e.g., action mask generation in actions.pyx).
"""

from core.state cimport GameState


# =============================================================================
# NET WORTH UPDATE
# =============================================================================

cdef void update_all_player_net_worths(GameState state, int num_players) noexcept


# =============================================================================
# PLAYER CLASS
# =============================================================================

cdef class Player:
    """
    Entity handle for accessing player state.
    Uses cached offsets for efficient nogil access in performance-critical paths.
    """
    cdef readonly int player_id
    cdef int _base_offset
    cdef int _num_players
    cdef bint _initialized

    # Cached field offsets (visible state)
    cdef int _cash_offset
    cdef int _net_worth_offset
    cdef int _turn_order_offset
    cdef int _owned_companies_offset
    cdef int _owned_shares_offset
    cdef int _is_president_offset
    cdef int _round_trips_offset
    cdef int _acquisition_proceeds_offset
    cdef int _income_offset
    # Cached field offsets (hidden state — share buy/sell tracking)
    cdef int _hidden_share_buys_offset
    cdef int _hidden_share_sells_offset

    # Initialization
    cpdef void initialize(self, GameState state)

    # Nogil methods (use cached offsets for performance)
    cdef inline int _get_cash_nogil(self, float* data) noexcept nogil
    cdef inline int _get_shares_nogil(self, float* data, int corp_id) noexcept nogil
    cdef inline int _get_share_buys_nogil(self, float* data, int corp_id) noexcept nogil
    cdef inline int _get_share_sells_nogil(self, float* data, int corp_id) noexcept nogil
    cdef inline bint _owns_company_nogil(self, float* data, int company_id) noexcept nogil

    # Cash
    cpdef int get_cash(self, GameState state)
    cpdef void set_cash(self, GameState state, int cash)
    cpdef void add_cash(self, GameState state, int amount)

    # Net worth
    cpdef int get_net_worth(self, GameState state)
    cpdef void set_net_worth(self, GameState state, int net_worth)
    cpdef int calculate_net_worth(self, GameState state)
    cpdef void update_net_worth(self, GameState state)

    # Turn order
    cpdef int get_turn_order(self, GameState state)
    cpdef void set_turn_order(self, GameState state, int order)

    # Company ownership (use Company.transfer_to_player() to set)
    cpdef bint owns_company(self, GameState state, int company_id)

    # Corporation shares
    cpdef int get_shares(self, GameState state, int corp_id)
    cpdef void set_shares(self, GameState state, int corp_id, int shares)

    # President status (read-only - presidency is derived from share ownership)
    cpdef bint is_president_of(self, GameState state, int corp_id)

    # Round-trip tracking (hidden: buys/sells, visible: round_trips)
    cpdef int get_share_buys(self, GameState state, int corp_id)
    cpdef void increment_share_buys(self, GameState state, int corp_id)
    cpdef int get_share_sells(self, GameState state, int corp_id)
    cpdef void increment_share_sells(self, GameState state, int corp_id)
    cdef void _update_visible_roundtrips(self, GameState state, int corp_id)
    cpdef int get_roundtrips(self, GameState state, int corp_id)
    cpdef void clear_roundtrip_tracking(self, GameState state)

    # Acquisition proceeds
    cpdef int get_acquisition_proceeds(self, GameState state)
    cpdef void set_acquisition_proceeds(self, GameState state, int proceeds)
    cpdef void add_acquisition_proceeds(self, GameState state, int amount)
    cpdef void clear_acquisition_proceeds(self, GameState state)

    # Income
    cpdef int get_income(self, GameState state)
    cpdef void set_income(self, GameState state, int income)
    cpdef void calculate_income(self, GameState state)
