# cython: language_level=3
"""
Declaration file for action vector.

Defines the action space layout, action decoding, and mask generation for
the NN output layer.
"""

from state cimport GameState

# =============================================================================
# CONSTANTS
# =============================================================================

cdef enum:
    # Action space dimensions
    AUCTION_CAP = 20         # Max bid offset over face value
    MAX_PAR_SLOTS = 8        # Max valid par prices per star tier
    ACQ_PRICE_RANGE = 51     # 0-50 price offset
    MAX_DIVIDEND = 26        # 0-25 dividend amount
    NUM_CORPS = 8
    NUM_COMPANIES = 36
    NUM_PAR_PRICES = 14


# =============================================================================
# ACTION TYPE ENUM
# =============================================================================

cdef enum ActionType:
    ACTION_PASS = 0
    ACTION_AUCTION = 1        # Start auction: (slot, bid_offset)
    ACTION_BUY_SHARE = 2      # Buy share: corp_id
    ACTION_SELL_SHARE = 3     # Sell share: corp_id
    ACTION_LEAVE_AUCTION = 4  # Leave auction
    ACTION_RAISE_BID = 5      # Raise bid: bid_offset
    ACTION_ACQ_PRICE = 6      # Acquire at price: price_offset
    ACTION_ACQ_FI_HIGH = 7    # FI buy at high price
    ACTION_ACQ_FI_FACE = 8    # FI buy at face value (OS only)
    ACTION_CLOSE = 9          # Close current company
    ACTION_DIVIDEND = 10      # Pay dividend: amount
    ACTION_ISSUE = 11         # Issue share
    ACTION_IPO = 12           # IPO: (corp_id, par_slot)


# =============================================================================
# ACTION LAYOUT STRUCT
# =============================================================================

cdef struct ActionLayout:
    int total_size
    # Phase boundaries (start indices)
    int invest_start          # 0
    int bid_start             # 137
    int acquisition_start     # 157
    int closing_start         # 211
    int dividends_start       # 213
    int issue_start           # 239
    int ipo_start             # 241
    # INVEST sub-offsets
    int pass_invest           # 0
    int auction_base          # 1 (slot * AUCTION_CAP + bid_offset)
    int buy_share_base        # 121 (+corp_id)
    int sell_share_base       # 129 (+corp_id)
    # BID sub-offsets
    int leave_auction         # 137
    int raise_bid_base        # 138 (+bid_offset, 0-18 = new bid face+1 to face+19)
    # ACQUISITION sub-offsets
    int acq_price_base        # 157 (+price_offset 0-50)
    int acq_fi_high           # 208
    int acq_fi_face           # 209
    int acq_pass              # 210
    # CLOSING sub-offsets
    int close_action          # 211
    int close_pass            # 212
    # DIVIDENDS sub-offsets
    int dividend_base         # 213 (+amount 0-25)
    # ISSUE sub-offsets
    int issue_pass            # 239
    int issue_action          # 240
    # IPO sub-offsets
    int ipo_pass              # 241
    int ipo_base              # 242 (corp_id * MAX_PAR_SLOTS + par_slot)


# =============================================================================
# ACTION INFO STRUCT
# =============================================================================

cdef struct ActionInfo:
    int phase           # PHASE_* constant
    int action_type     # ActionType enum
    int slot            # auction_slot, par_slot (needs mapping to actual ID)
    int corp_id         # -1 if not applicable
    int amount          # price_offset, bid_offset, dividend amount


# =============================================================================
# FUNCTION DECLARATIONS
# =============================================================================

# Layout computation (dynamic based on player count)
cdef ActionLayout compute_action_layout(int num_players) noexcept nogil

# Total action count for a given player count
cdef int get_total_actions_for_players(int num_players) noexcept nogil

# Action decoding
cdef ActionInfo decode_action(ActionLayout* layout, int action_idx) noexcept nogil
