"""Phase-local action ids, decoding, and legality for the 1889 engine.

Ids follow the named blocks in ``g1889/constants.pxd`` (engine-design.md §5).
There are three id sets:

- engine-legal ids, from ``enumerate_legal_actions`` in a fixed order;
- policy ids, from ``enumerate_policy_actions``: the legal ids minus the
  state's runtime-flag restrictions and the permanent policy filters (§5.1,
  §7). Search expands only these;
- exact amounts, ``*_EXACT + amount`` in BID, TRAIN_PRICE, and
  PRIVATE_PRICE: legal by range check, never enumerated, and at or above the
  phase size, so they have no unified policy slot.

``is_terminal`` and ``final_scores`` are the game's end and its scores (§9).
"""

cimport numpy as cnp
from libc.stdint cimport uint16_t

from g1889.state cimport GameState


cpdef enum ActionType:
    ACTION_PASS = 0                 # each phase's pass, done, or drop slot
    ACTION_BUY_TOPMOST = 1          # AUCTION: the topmost private at its price
    ACTION_SELECT_BID = 2           # AUCTION: bid on ``private`` (BID follows)
    ACTION_PRICE_POINT = 3          # BID, TRAIN_PRICE, PRIVATE_PRICE: grid point ``count``
    ACTION_EXACT_PRICE = 4          # the same phases: ``amount`` yen
    ACTION_BUY_IPO = 5              # STOCK: a share of ``corp`` from the IPO
    ACTION_BUY_POOL = 6             # STOCK: a share of ``corp`` from the pool
    ACTION_SELECT_PAR = 7           # STOCK: par ``corp`` (PAR follows)
    ACTION_SELECT_SALE = 8          # STOCK, BUY_TRAIN: sell ``corp``'s shares (SELL_AMOUNT, EMERGENCY_SELL_AMOUNT)
    ACTION_SELL = 9                 # SELL_AMOUNT, EMERGENCY_SELL_AMOUNT: ``count`` shares of the pending corp
    ACTION_DR_EXCHANGE = 10         # DR for an IR share
    ACTION_MF_PORT = 11             # use MF (PORT follows)
    ACTION_PAR = 12                 # PAR: par index ``count`` at ``amount`` yen
    ACTION_SELECT_HEX = 13          # TRACK: lay on ``hex`` (TILE follows)
    ACTION_LAY_CANDIDATE = 14       # TILE, ER_BONUS: lay candidate ``count``
    ACTION_PLACE_TOKEN = 15         # TOKEN: token ``hex``
    ACTION_BUY_PRIVATE = 16         # buy ``private`` (PRIVATE_PRICE follows)
    ACTION_WITHHOLD = 17            # DIVIDEND
    ACTION_PAYOUT = 18              # DIVIDEND
    ACTION_DISCARD = 19             # DISCARD: a ``train_type``
    ACTION_BUY_DEPOT_TRAIN = 20     # BUY_TRAIN: a train from the depot (SELECT_TRAIN follows)
    ACTION_BUY_POOL_TRAIN = 21      # BUY_TRAIN: a train from the pool (SELECT_TRAIN follows)
    ACTION_BUY_CORP_TRAIN = 22      # BUY_TRAIN: a train from ``corp`` (SELECT_TRAIN, then TRAIN_PRICE)
    ACTION_TRAIN_TYPE = 23          # SELECT_TRAIN: a ``train_type`` from the pending source
    ACTION_TRADE_IN = 24            # SELECT_TRAIN: trade ``train_type`` (4, 5, or 6) and ¥800 for a D
    ACTION_BANKRUPT = 25            # BUY_TRAIN
    ACTION_PORT_LAY = 26            # PORT: the port tile on ``hex`` at rotation ``count``


# A tile lay at the chosen hex: a supply tile at a canonical rotation.
cdef struct LayCandidate:
    int tile
    int rotation


# Fields that do not apply are -1.
cdef struct ActionInfo:
    int phase
    int type
    int corp
    int private
    int hex
    int train_type
    int count                       # shares sold, an index (point, par, candidate), or a rotation
    int amount                      # yen


cdef int encode_action(ActionInfo info) except -1 nogil
cdef ActionInfo decode_action(int phase, int action_id) noexcept nogil

cdef int get_decision_phase(GameState state) noexcept nogil
cdef bint is_terminal(GameState state) noexcept nogil
cdef int final_scores(GameState state, int* scores) noexcept nogil
cdef int enumerate_legal_actions(GameState state, uint16_t* action_ids) except -1 nogil
cdef int enumerate_policy_actions(GameState state, uint16_t* action_ids) except -1 nogil
cdef bint is_legal_action(GameState state, int action_id) except -1 nogil
cdef int price_amount(GameState state, const ActionInfo* info) noexcept nogil
cdef bint wide_amount_range(GameState state, int phase) except -1 nogil

# Track and token legality (§5.4): lay candidates sorted by (tile, rotation).
cdef int lay_candidates(GameState state, int corp, int hex, LayCandidate* buf) noexcept nogil
cdef int enumerate_lay_candidates(GameState state, int hex, LayCandidate* buf) noexcept nogil
cdef bint can_lay(GameState state, int corp, int hex) noexcept nogil
cdef bint can_token(GameState state, int corp, int hex) noexcept nogil
cdef bint must_buy_train(GameState state, int corp) noexcept nogil

# Emergency money (§6.2): a must-buy the treasury cannot fund.
cdef bint in_emergency(GameState state, int corp) noexcept nogil
cdef bint can_go_bankrupt(GameState state, int corp) noexcept nogil

# Privates (§5.1, §5.3): purchases by the operating corp, DR's exchange, MF's
# port tile (PORT's ids), and ER's lays on C4 sorted like TILE's.
cdef bint can_buy_private(GameState state, int corp, int private) noexcept nogil
cdef bint can_exchange_dr(GameState state, int player) noexcept nogil
cdef int port_lays(GameState state, uint16_t* ids) noexcept nogil
cdef int er_candidates(GameState state, LayCandidate* buf) noexcept nogil

cpdef bint has_exact_amounts(int phase) noexcept nogil
cpdef object decode_action_py(int phase, int action_id)
cpdef int encode_action_py(object info) except -1
cpdef int get_decision_phase_py(GameState state)
cpdef int enumerate_legal_actions_py(GameState state, cnp.ndarray action_ids) except -1
cpdef int enumerate_policy_actions_py(GameState state, cnp.ndarray action_ids) except -1
cpdef bint is_legal_action_py(GameState state, int action_id) except -1
cpdef int price_amount_py(GameState state, int action_id) except -1
cpdef list lay_candidates_py(GameState state, int hex, int corp=*)
cpdef list lay_hexes_py(GameState state, int corp=*)
cpdef list token_hexes_py(GameState state, int corp=*)
cpdef bint must_buy_train_py(GameState state, int corp=*)
cpdef bint in_emergency_py(GameState state, int corp=*)
cpdef bint can_go_bankrupt_py(GameState state, int corp=*)
cpdef bint is_terminal_py(GameState state)
cpdef list final_scores_py(GameState state)
cpdef list purchasable_privates_py(GameState state, int corp=*)
cpdef bint can_exchange_dr_py(GameState state, int player)
cpdef list port_lays_py(GameState state)
cpdef list er_candidates_py(GameState state)
