# cython: language_level=3
"""
Declaration file for action vector.

Defines the action space layout, action decoding, and mask generation for
the NN output layer.
"""

from core.state cimport GameState

# =============================================================================
# CONSTANTS
# =============================================================================

cdef enum:
    # Action space dimensions (game constants like NUM_CORPS are in data.pxd)
    AUCTION_CAP = 15         # Max bid offset over face value
    MAX_PAR_SLOTS = 8        # Max valid par prices per star tier
    ACQ_PRICE_RANGE = 51     # 0-50 price offset


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
    int bid_start             # 107
    int acquisition_start     # 122
    int closing_start         # 176
    int dividends_start       # 178
    int issue_start           # 204
    int ipo_start             # 206
    # INVEST sub-offsets
    int pass_invest           # 0
    int auction_base          # 1 (slot * AUCTION_CAP + bid_offset)
    int buy_share_base        # 91 (+corp_id)
    int sell_share_base       # 99 (+corp_id)
    # BID sub-offsets
    int leave_auction         # 107
    int raise_bid_base        # 108 (+bid_offset, 0-13 = new bid face+1 to face+14)
    # ACQUISITION sub-offsets
    int acq_price_base        # 122 (+price_offset 0-50)
    int acq_fi_high           # 173
    int acq_fi_face           # 174
    int acq_pass              # 175
    # CLOSING sub-offsets
    int close_action          # 176
    int close_pass            # 177
    # DIVIDENDS sub-offsets
    int dividend_base         # 178 (+amount 0-25)
    # ISSUE sub-offsets
    int issue_pass            # 204
    int issue_action          # 205
    # IPO sub-offsets
    int ipo_pass              # 206
    int ipo_base              # 207 (corp_id * MAX_PAR_SLOTS + par_slot)


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

# Forced action check (returns single valid action if only one exists)
cpdef tuple get_forced_action(GameState state)

# Pre-allocated mask buffer (max size for 6 players = 271)
cdef float _mask_buffer[271]

# Internal mask helpers (no numpy allocation)
cdef void _fill_action_mask(GameState state)
cdef bint _is_action_valid_in_buffer(int action_idx, int total_actions) noexcept nogil
