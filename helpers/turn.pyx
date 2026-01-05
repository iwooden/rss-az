# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Turn state offset helper functions.

Computes offsets into the turn state portion of the game state tensor.
Turn state is laid out sequentially with the following sections:

Base (always present):
- turn_number (1)
- end_card_flipped (1)
- consecutive_passes (1)

Auction (invest/bid phases):
- auction_company (NUM_COMPANIES) - one-hot
- auction_price (1)
- high_bidder (num_players) - one-hot
- auction_starter (num_players) - one-hot
- auction_passed (num_players) - binary flags

Dividends (dividend phase):
- dividend_corp (NUM_CORPS) - one-hot
- dividend_impact (MAX_DIVIDEND) - impact scores
- dividend_remaining (NUM_CORPS) - binary flags

Issue/IPO (issue and IPO phases):
- issue_corp (NUM_CORPS) - one-hot
- issue_remaining (NUM_CORPS) - ternary (-1/0/1)
- ipo_company (NUM_COMPANIES) - one-hot
- ipo_remaining (NUM_COMPANIES) - ternary (-1/0/1)
"""

from cython_core.data cimport NUM_COMPANIES, NUM_CORPS, MAX_DIVIDEND


# =============================================================================
# BASE OFFSETS
# =============================================================================

cdef inline int get_turn_number_offset() noexcept nogil:
    """Offset of turn_number field (always 0)."""
    return 0


cdef inline int get_end_card_flipped_offset() noexcept nogil:
    """Offset of end_card_flipped field (always 1)."""
    return 1


cdef inline int get_consecutive_passes_offset() noexcept nogil:
    """Offset of consecutive_passes field (always 2)."""
    return 2


# =============================================================================
# AUCTION OFFSETS
# =============================================================================

cdef AuctionTurnOffsets get_auction_turn_offsets(int num_players) noexcept nogil:
    """
    Get offsets for auction-related turn state fields.

    Layout after base fields (offset 3):
    - auction_company (NUM_COMPANIES)
    - auction_price (1)
    - high_bidder (num_players)
    - auction_starter (num_players)
    - auction_passed (num_players)
    """
    cdef AuctionTurnOffsets ato
    cdef int offset = 3  # After base fields

    ato.auction_company = offset
    offset += NUM_COMPANIES

    ato.auction_price = offset
    offset += 1

    ato.high_bidder = offset
    offset += num_players

    ato.auction_starter = offset
    offset += num_players

    ato.auction_passed = offset
    # offset += num_players

    return ato


# =============================================================================
# DIVIDEND OFFSETS
# =============================================================================

cdef DividendTurnOffsets get_dividend_turn_offsets(int num_players) noexcept nogil:
    """
    Get offsets for dividend-related turn state fields.

    Layout after auction fields:
    - dividend_corp (NUM_CORPS)
    - dividend_impact (MAX_DIVIDEND)
    - dividend_remaining (NUM_CORPS)
    """
    cdef DividendTurnOffsets dto
    cdef int offset = 3  # After base fields

    # Skip auction fields
    offset += NUM_COMPANIES + 1 + num_players * 3

    dto.dividend_corp = offset
    offset += NUM_CORPS

    dto.dividend_impact = offset
    offset += MAX_DIVIDEND

    dto.dividend_remaining = offset
    # offset += NUM_CORPS

    return dto


# =============================================================================
# ISSUE/IPO OFFSETS
# =============================================================================

cdef IssueTurnOffsets get_issue_turn_offsets(int num_players) noexcept nogil:
    """
    Get offsets for issue and IPO-related turn state fields.

    Layout after dividend fields:
    - issue_corp (NUM_CORPS)
    - issue_remaining (NUM_CORPS)
    - ipo_company (NUM_COMPANIES)
    - ipo_remaining (NUM_COMPANIES)
    """
    cdef IssueTurnOffsets ito
    cdef int offset = 3  # After base fields

    # Skip auction fields
    offset += NUM_COMPANIES + 1 + num_players * 3

    # Skip dividend fields
    # dividends: corp(8) + impact(26) + remaining(8)
    offset += NUM_CORPS + MAX_DIVIDEND + NUM_CORPS

    ito.issue_corp = offset
    offset += NUM_CORPS

    ito.issue_remaining = offset
    offset += NUM_CORPS

    ito.ipo_company = offset
    offset += NUM_COMPANIES

    ito.ipo_remaining = offset
    # offset += NUM_COMPANIES

    return ito
