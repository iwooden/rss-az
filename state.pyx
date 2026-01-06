# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Game state implementation.

The state is a single contiguous float array organized as:
  [VISIBLE STATE (for NN)][HIDDEN STATE (truncated before NN)]

All game logic operates directly on the array - no Python object overhead.
The visible state is presented to the NN with player rotation (active player first).
"""

cimport cython
from libc.string cimport memcpy, memset
cimport numpy as cnp
import numpy as np

from data cimport get_adjusted_company_income
from data cimport (
    get_adjusted_company_income,
    GameConstants,
    GamePhases,
    CASH_DIVISOR,
    SHARE_DIVISOR,
    STAR_DIVISOR,
    MARKET_PRICES
)

cnp.import_array()

# =============================================================================
# STATE LAYOUT STRUCTURE
# =============================================================================
#
# VISIBLE STATE (fed to NN, with player rotation):
# ------------------------------------------------
# phase[12]                    - one-hot phase
# coo[7]                       - one-hot cost of ownership level
#
# PLAYERS (repeated num_players times):
#   cash[1]                    - normalized
#   net_worth[1]               - normalized
#   turn_order[num_players]    - one-hot position
#   is_auction_high_bidder[1]  - auction flag
#   owned_companies[36]        - flags
#   owned_shares[8]            - normalized per corp
#   is_president[8]            - flags
#   share_buys[8]              - round-trip tracking
#   share_sells[8]             - round-trip tracking
#
# FOREIGN INVESTOR:
#   fi_cash[1]                 - normalized
#   fi_companies[36]           - flags
#
# COMPANY LOCATIONS:
#   companies_for_auction[36]  - flags
#   companies_revealed[36]     - flags (unavailable this turn)
#   companies_removed[36]      - flags (out of game)
#
# MARKET:
#   market_available[27]       - flags (1=available, 0=taken)
#
# CORPS (repeated 8 times):
#   active[1]                  - flag
#   cash[1]                    - normalized
#   unissued_shares[1]         - normalized
#   issued_shares[1]           - normalized
#   bank_shares[1]             - normalized
#   income[1]                  - normalized (derived)
#   stars[1]                   - normalized (derived)
#   share_price[1]             - normalized
#   acquisition_proceeds[1]    - normalized
#   in_receivership[1]         - flag
#   price_index[27]            - one-hot market position
#   owned_companies[36]        - flags
#   acquisition_companies[36]  - flags (pending this phase)
#
# TURN STATE:
#   turn_number[1]             - normalized
#   end_card_flipped[1]        - flag
#   consecutive_passes[1]      - normalized (passes / num_players)
#
#   # Auction (BID_IN_AUCTION phase, -1 otherwise)
#   auction_company[36]        - one-hot
#   auction_price[1]           - normalized
#   auction_high_bidder[num_players] - one-hot
#   auction_starter[num_players]     - one-hot
#   auction_passed[num_players]      - flags
#
#
#   # Dividends (DIVIDENDS phase, -1 otherwise)
#   dividend_corp[8]           - one-hot current
#   dividend_impact[26]        - price impact per level
#   dividend_remaining[8]      - flags corps left
#
#   # Issue (ISSUE_SHARES phase, -1 otherwise)
#   issue_corp[8]              - one-hot current
#   issue_remaining[8]         - flags corps left
#
#   # IPO (IPO phase, -1 otherwise)
#   ipo_company[36]            - one-hot current
#   ipo_remaining[36]          - flags companies left
#
#   # Acquisition (ACQUISITION phase, -1 otherwise)
#   acq_active_corp[8]         - one-hot buying corp
#   acq_target_company[36]     - one-hot target company
#   acq_is_fi_offer[1]         - flag (1=FI target, 0=player/corp target)
#
# STATIC COMPANY DATA (constant, for NN reference):
#   For each company[36]:
#     stars[1]                 - normalized
#     low_price[1]             - normalized
#     face_value[1]            - normalized
#     high_price[1]            - normalized
#     synergies[36]            - flags
#
# HIDDEN STATE (truncated before NN):
# ------------------------------------
# active_player[1]             - canonical player index
# num_players[1]               - player count (for variable games)
# deck_top[1]                  - index of top card (-1 if empty)
# deck_order[36]               - company IDs in order (-1 for empty slots)
#
# =============================================================================


# =============================================================================
# LAYOUT COMPUTATION
# =============================================================================

cdef StateLayout compute_layout(int num_players) noexcept nogil:
    """Compute complete state layout for given player count."""
    cdef StateLayout layout
    cdef int offset = 0

    # Phase one-hot
    layout.phase_size = GameConstants.NUM_PHASES
    layout.phase_offset = offset
    offset += layout.phase_size

    # Cost of ownership one-hot
    layout.coo_size = GameConstants.NUM_COO_LEVELS
    layout.coo_offset = offset
    offset += layout.coo_size

    # Players
    layout.player_stride = (
        1 +                 # cash
        1 +                 # net_worth
        num_players +       # turn_order one-hot
        GameConstants.NUM_COMPANIES +     # owned_companies
        GameConstants.NUM_CORPS +         # owned_shares
        GameConstants.NUM_CORPS +         # is_president
        GameConstants.NUM_CORPS +         # share_buys
        GameConstants.NUM_CORPS           # share_sells
    )
    layout.players_offset = offset
    layout.players_size = layout.player_stride * num_players
    offset += layout.players_size

    # Foreign investor
    layout.fi_size = 1 + GameConstants.NUM_COMPANIES # cash, owned companies
    layout.fi_offset = offset
    offset += layout.fi_size

    # Company locations (3 arrays)
    layout.companies_size = GameConstants.NUM_COMPANIES * 3
    layout.auction_companies_offset = offset
    offset += GameConstants.NUM_COMPANIES
    layout.revealed_companies_offset = offset
    offset += GameConstants.NUM_COMPANIES
    layout.removed_companies_offset = offset
    offset += GameConstants.NUM_COMPANIES

    # Company adjusted incomes (dynamic based on CoO level)
    layout.company_incomes_size = GameConstants.NUM_COMPANIES
    layout.company_incomes_offset = offset
    offset += GameConstants.NUM_COMPANIES

    # Market availability
    layout.market_size = GameConstants.NUM_MARKET_SPACES
    layout.market_offset = offset
    offset += layout.market_size

    # Corporations
    layout.corp_stride = (
        1 +                 # active
        1 +                 # cash
        1 +                 # unissued_shares
        1 +                 # issued_shares
        1 +                 # bank_shares
        1 +                 # income
        1 +                 # stars
        1 +                 # share_price
        1 +                 # acquisition_proceeds
        1 +                 # in_receivership
        GameConstants.NUM_MARKET_SPACES + # price_index one-hot
        GameConstants.NUM_COMPANIES +     # owned_companies
        GameConstants.NUM_COMPANIES       # acquisition_companies
    )
    layout.corps_offset = offset
    layout.corps_size = layout.corp_stride * GameConstants.NUM_CORPS
    offset += layout.corps_size

    # Turn state
    layout.turn_size = (
        1 +                 # turn_number
        1 +                 # end_card_flipped
        1 +                 # consecutive_passes (for INVEST phase)
        # Auction
        GameConstants.NUM_COMPANIES +     # auction_company
        1 +                 # auction_price
        num_players +       # auction_high_bidder
        num_players +       # auction_starter
        num_players +       # auction_passed
        # Dividends
        GameConstants.NUM_CORPS +         # dividend_corp
        GameConstants.MAX_DIVIDEND +      # dividend_impact
        GameConstants.NUM_CORPS +         # dividend_remaining
        # Issue
        GameConstants.NUM_CORPS +         # issue_corp
        GameConstants.NUM_CORPS +         # issue_remaining
        # IPO
        GameConstants.NUM_COMPANIES +     # ipo_company
        GameConstants.NUM_COMPANIES +     # ipo_remaining
        # Acquisition offers
        GameConstants.NUM_CORPS +         # acq_active_corp
        GameConstants.NUM_COMPANIES +     # acq_target_company
        1                   # acq_is_fi_offer
    )
    layout.turn_offset = offset
    offset += layout.turn_size

    # Static company data
    layout.static_size = GameConstants.NUM_COMPANIES * (4 + GameConstants.NUM_COMPANIES)  # stars, low, face, high, synergies
    layout.static_offset = offset
    offset += layout.static_size

    layout.visible_size = offset

    # Hidden state layout:
    # [0] active_player
    # [1] num_players
    # [2] deck_top
    # [3..38] deck_order (36 slots)
    # [39] phase (compact)
    # [40] coo_level (compact)
    # [41] auction_company (compact)
    # [42] auction_high_bidder (compact)
    # [43] auction_starter (compact)
    # [44..51] corp_price_indices (8 slots)
    layout.hidden_active_player_offset = offset
    offset += 1
    layout.hidden_num_players_offset = offset
    offset += 1
    layout.hidden_deck_top_offset = offset
    offset += 1
    layout.hidden_deck_order_offset = offset
    offset += GameConstants.MAX_DECK_SIZE
    layout.hidden_phase_offset = offset
    offset += 1
    layout.hidden_coo_level_offset = offset
    offset += 1
    layout.hidden_auction_company_offset = offset
    offset += 1
    layout.hidden_auction_high_bidder_offset = offset
    offset += 1
    layout.hidden_auction_starter_offset = offset
    offset += 1
    layout.hidden_corp_price_indices_offset = offset
    offset += GameConstants.NUM_CORPS
    layout.hidden_acq_corps_done_offset = offset
    offset += GameConstants.NUM_CORPS
    layout.hidden_size = offset - layout.visible_size

    layout.total_size = layout.visible_size + layout.hidden_size

    return layout


# =============================================================================
# TURN STATE SUB-OFFSETS
# =============================================================================

cdef TurnStateOffsets compute_turn_offsets(int num_players) noexcept nogil:
    """Compute sub-offsets within turn state section."""
    cdef TurnStateOffsets t
    cdef int offset = 0

    t.turn_number = offset
    offset += 1
    t.end_card_flipped = offset
    offset += 1
    t.consecutive_passes = offset
    offset += 1

    # Auction
    t.auction_company = offset
    offset += GameConstants.NUM_COMPANIES
    t.auction_price = offset
    offset += 1
    t.auction_high_bidder = offset
    offset += num_players
    t.auction_starter = offset
    offset += num_players
    t.auction_passed = offset
    offset += num_players

    # Dividends
    t.dividend_corp = offset
    offset += GameConstants.NUM_CORPS
    t.dividend_impact = offset
    offset += GameConstants.MAX_DIVIDEND
    t.dividend_remaining = offset
    offset += GameConstants.NUM_CORPS

    # Issue
    t.issue_corp = offset
    offset += GameConstants.NUM_CORPS
    t.issue_remaining = offset
    offset += GameConstants.NUM_CORPS

    # IPO
    t.ipo_company = offset
    offset += GameConstants.NUM_COMPANIES
    t.ipo_remaining = offset
    offset += GameConstants.NUM_COMPANIES

    # Acquisition offers
    t.acq_active_corp = offset
    offset += GameConstants.NUM_CORPS
    t.acq_target_company = offset
    offset += GameConstants.NUM_COMPANIES
    t.acq_is_fi_offer = offset
    offset += 1

    # Closing phase
    t.closing_company = offset
    offset += GameConstants.NUM_COMPANIES

    return t


# =============================================================================
# PLAYER FIELD OFFSETS (within player stride)
# =============================================================================

cdef PlayerFieldOffsets compute_player_field_offsets(int num_players) noexcept nogil:
    """Compute field offsets within a player's data block."""
    cdef PlayerFieldOffsets p
    cdef int offset = 0

    p.cash = offset
    offset += 1
    p.net_worth = offset
    offset += 1
    p.turn_order = offset
    offset += num_players
    p.is_auction_high_bidder = offset
    offset += 1
    p.owned_companies = offset
    offset += GameConstants.NUM_COMPANIES
    p.owned_shares = offset
    offset += GameConstants.NUM_CORPS
    p.is_president = offset
    offset += GameConstants.NUM_CORPS
    p.share_buys = offset
    offset += GameConstants.NUM_CORPS
    p.share_sells = offset

    return p


# =============================================================================
# CORP FIELD OFFSETS (within corp stride)
# =============================================================================

cdef CorpFieldOffsets compute_corp_field_offsets() noexcept nogil:
    """Compute field offsets within a corp's data block."""
    cdef CorpFieldOffsets c
    cdef int offset = 0

    c.active = offset
    offset += 1
    c.cash = offset
    offset += 1
    c.unissued_shares = offset
    offset += 1
    c.issued_shares = offset
    offset += 1
    c.bank_shares = offset
    offset += 1
    c.income = offset
    offset += 1
    c.stars = offset
    offset += 1
    c.share_price = offset
    offset += 1
    c.acquisition_proceeds = offset
    offset += 1
    c.in_receivership = offset
    offset += 1
    c.price_index = offset
    offset += GameConstants.NUM_MARKET_SPACES
    c.owned_companies = offset
    offset += GameConstants.NUM_COMPANIES
    c.acquisition_companies = offset

    return c