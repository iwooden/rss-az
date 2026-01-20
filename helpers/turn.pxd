# cython: language_level=3
"""
Turn state offset helper declarations.

Provides structs and functions for computing offsets into the turn state
portion of the game state tensor. Each phase has its own view of turn state.
"""

from core.data cimport NUM_COMPANIES, NUM_CORPS, MAX_DIVIDEND

# =============================================================================
# TURN STATE OFFSET STRUCTS
# =============================================================================

cdef struct AuctionTurnOffsets:
    int auction_company      # One-hot: which company is being auctioned (36)
    int auction_price        # Normalized current bid price (1)
    int high_bidder          # One-hot: which player is high bidder (num_players)
    int auction_starter      # One-hot: which player started the auction (num_players)
    int auction_passed       # Binary: which players have left auction (num_players)


cdef struct DividendTurnOffsets:
    int dividend_corp        # One-hot: which corp is paying dividends (8)
    int dividend_impact      # Impact scores for each dividend level (26)
    int dividend_remaining   # Binary: which corps still need to pay (8)


cdef struct IssueTurnOffsets:
    int issue_corp           # One-hot: which corp is issuing (8)
    int issue_remaining      # Binary/ternary: which corps can still issue (8)
    int ipo_company          # One-hot: which company is being IPO'd (36)
    int ipo_remaining        # Binary: which companies can still IPO (36)


# =============================================================================
# PHASE-SPECIFIC OFFSET GETTERS
# =============================================================================

cdef AuctionTurnOffsets get_auction_turn_offsets(int num_players) noexcept nogil
cdef DividendTurnOffsets get_dividend_turn_offsets(int num_players) noexcept nogil
cdef IssueTurnOffsets get_issue_turn_offsets(int num_players) noexcept nogil
