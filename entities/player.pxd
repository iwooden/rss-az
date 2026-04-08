# cython: language_level=3
"""
Player entity declarations.

The Player handle is fully stateless: every read/write derives its slot
inline from the module-level ``LAYOUT`` and ``PLAYER_FIELDS`` constants
on ``core.state``. Player handles only carry their own ``player_id``;
the singletons can be reused with any GameState at any player count
(up to MAX_PLAYERS). Player-id bounds checks call
``TURN._get_num_players(state)``, which reads the canonical slot inside
the turn block, so the singleton instance can be reused with any GameState.

All values are raw int16 — no normalization, no one-hot encoding (turn
order is now a single integer slot, not a per-player one-hot). Company
ownership is read from the shared ``company_locations`` /
``company_owner_ids`` arrays via ``LOC_PLAYER``; there is no per-player
owned_companies bitmap any more. Per-turn share buy/sell tracking and
the auction-passed flag live inside the player block, so
``_player_ptr(i)`` reaches everything for player ``i`` in one pointer
hop.
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
    cdef readonly int player_id

    # Low-level (nogil) accessors used by hot paths inside the engine.
    cdef int _get_cash(self, GameState state) noexcept nogil
    cdef int _get_shares(self, GameState state, int corp_id) noexcept nogil
    cdef int _get_share_buys(self, GameState state, int corp_id) noexcept nogil
    cdef int _get_share_sells(self, GameState state, int corp_id) noexcept nogil
    cdef bint _owns_company(self, GameState state, int company_id) noexcept nogil
    cdef inline int _slot(self, int field) noexcept nogil

    # Cash
    cpdef int get_cash(self, GameState state)
    cpdef void set_cash(self, GameState state, int cash)
    cpdef void add_cash(self, GameState state, int amount)

    # Net worth
    cpdef int get_net_worth(self, GameState state)
    cpdef void set_net_worth(self, GameState state, int net_worth)
    cpdef int calculate_net_worth(self, GameState state)
    cpdef void update_net_worth(self, GameState state)

    # Liquidity
    cpdef int get_liquidity(self, GameState state)
    cpdef void set_liquidity(self, GameState state, int liquidity)
    cpdef int calculate_liquidity(self, GameState state)

    # Turn order (single integer, not one-hot)
    cpdef int get_turn_order(self, GameState state)
    cpdef void set_turn_order(self, GameState state, int order)

    # Company ownership (use Company.transfer_to_player() to set)
    cpdef bint owns_company(self, GameState state, int company_id)

    # Corporation shares
    cpdef int get_shares(self, GameState state, int corp_id)
    cpdef void set_shares(self, GameState state, int corp_id, int shares)

    # President status (read-only — presidency is derived from share ownership)
    cpdef bint is_president_of(self, GameState state, int corp_id)

    # Round-trip tracking (per-corp buy/sell counts inside the player block)
    cpdef int get_share_buys(self, GameState state, int corp_id)
    cpdef void increment_share_buys(self, GameState state, int corp_id)
    cpdef int get_share_sells(self, GameState state, int corp_id)
    cpdef void increment_share_sells(self, GameState state, int corp_id)
    cdef void _update_roundtrips(self, GameState state)
    cpdef int get_roundtrips(self, GameState state, int corp_id)
    cpdef void clear_roundtrip_tracking(self, GameState state)

    # Income
    cpdef int get_income(self, GameState state)
    cpdef void set_income(self, GameState state, int income)
    cpdef void calculate_income(self, GameState state)

    # Auction-passed flag (per-player; lives inside the player block)
    cpdef bint has_passed_auction(self, GameState state)
    cpdef void set_passed_auction(self, GameState state, bint passed)
