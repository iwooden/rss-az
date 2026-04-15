# cython: language_level=3
"""
Turn state entity declarations.

Turn state in the compact GameState lives entirely inside the turn block:
metadata / active context (active player, active corp/company, phase,
coo_level, turn_number), per-turn tracking such as end_card_flipped,
consecutive_passes, cards_remaining, auction state, and the
dividend/issue/IPO "remaining" flag arrays.

All values are raw int16 — no one-hot encoding, no normalization.

The TurnState handle is fully stateless: there is no per-instance offset
cache and no initialize() step. Each access derives its slot inline from
the module-level ``LAYOUT`` and ``TURN_OFFSETS`` constants on
``core.state``. Player-id bounds checks call ``self._get_num_players(state)``
(which reads the canonical slot inside the turn block) so the singleton
instance can be reused with any GameState.
"""

from core.state cimport GameState


cdef class TurnState:
    # Low-level (nogil) accessors used by hot paths inside the engine.
    cdef int _get_phase(self, GameState state) noexcept nogil
    cdef int _get_coo_level(self, GameState state) noexcept nogil
    cdef int _get_active_player(self, GameState state) noexcept nogil
    cdef void _set_active_player(self, GameState state, int player_id) noexcept nogil
    cdef int _get_active_corp(self, GameState state) noexcept nogil
    cdef int _get_active_company(self, GameState state) noexcept nogil
    cdef int _get_auction_price(self, GameState state) noexcept nogil
    cdef int _get_num_players(self, GameState state) noexcept nogil

    # Active selection / player context (lives in the turn block)
    cpdef int get_active_player(self, GameState state)
    cpdef void set_active_player(self, GameState state, int player_id)
    cpdef int get_active_corp(self, GameState state)
    cpdef void set_active_corp(self, GameState state, int corp_id)
    cpdef void clear_active_corp(self, GameState state)
    cpdef int get_active_company(self, GameState state)
    cpdef void set_active_company(self, GameState state, int company_id)
    cpdef void clear_active_company(self, GameState state)

    # Number of players (state-level metadata; lives in the turn block)
    cpdef int get_num_players(self, GameState state)

    # Phase
    cpdef int get_phase(self, GameState state)
    cpdef void set_phase(self, GameState state, int phase)

    # Cost of ownership level (1-7 in game terms)
    cpdef int get_coo_level(self, GameState state)
    cpdef void set_coo_level(self, GameState state, int level)
    cdef void _update_all_company_incomes(self, GameState state, int coo_level)

    # Turn number
    cpdef int get_turn_number(self, GameState state)
    cpdef void set_turn_number(self, GameState state, int turn)

    # End card flipped
    cpdef bint is_end_card_flipped(self, GameState state)
    cpdef void set_end_card_flipped(self, GameState state, bint flipped)

    # Consecutive passes (INVEST phase)
    cpdef int get_consecutive_passes(self, GameState state)
    cpdef void set_consecutive_passes(self, GameState state, int passes)
    cpdef void increment_consecutive_passes(self, GameState state)
    cpdef void clear_consecutive_passes(self, GameState state)

    # Cards remaining (deck mirror)
    cpdef int get_cards_remaining(self, GameState state)
    cpdef void set_cards_remaining(self, GameState state, int count)

    # Auction state
    cpdef int get_auction_price(self, GameState state)
    cpdef void set_auction_price(self, GameState state, int price)

    cpdef int get_auction_high_bidder(self, GameState state)
    cpdef void set_auction_high_bidder(self, GameState state, int player_id)
    cpdef void clear_auction_high_bidder(self, GameState state)

    cpdef int get_auction_starter(self, GameState state)
    cpdef void set_auction_starter(self, GameState state, int player_id)
    cpdef void clear_auction_starter(self, GameState state)

    # ACQ_OFFER context
    cpdef int get_acq_offer_price(self, GameState state)
    cpdef void set_acq_offer_price(self, GameState state, int price)
    cpdef void clear_acq_offer_price(self, GameState state)
    cpdef int get_acq_offer_corp(self, GameState state)
    cpdef void set_acq_offer_corp(self, GameState state, int corp_id)
    cpdef void clear_acq_offer_corp(self, GameState state)
    cpdef void enter_acq_offer(
        self,
        GameState state,
        int offered_corp,
        int company_id,
        int price,
        int original_corp,
        int deciding_player,
    )
    cpdef void clear_acquisition_context(self, GameState state)

    # Per-player passed-flag bulk reset
    cpdef void clear_passed_flags(self, GameState state)

    # Phase-remaining flag arrays
    cpdef bint is_dividend_remaining(self, GameState state, int corp_id)
    cpdef void set_dividend_remaining(self, GameState state, int corp_id, bint remaining)

    cpdef bint is_issue_remaining(self, GameState state, int corp_id)
    cpdef void set_issue_remaining(self, GameState state, int corp_id, bint remaining)

    cpdef bint is_ipo_remaining(self, GameState state, int company_id)
    cpdef void set_ipo_remaining(self, GameState state, int company_id, bint remaining)

    # Turn order navigation
    cpdef int find_player_at_position(self, GameState state, int position)
    cpdef void advance_to_next_bidder(self, GameState state)
    cpdef void set_active_player_after(self, GameState state, int player_id)
