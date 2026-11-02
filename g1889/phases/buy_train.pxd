"""BUY_TRAIN, SELECT_TRAIN, TRAIN_PRICE, EMERGENCY_SELL_AMOUNT, and DISCARD
(1889/engine-design.md §5.2-5.3, §6.1-6.2), with emergency money and
bankruptcy. The handlers assume legal actions and change the state only
through entity handles."""

from g1889.actions cimport ActionInfo
from g1889.state cimport GameState

cdef void apply_buy_train(GameState state, const ActionInfo* info) except *
cdef void apply_select_train(GameState state, const ActionInfo* info) except *
cdef void apply_train_price(GameState state, const ActionInfo* info) except *
cdef void apply_emergency_sell_amount(GameState state, const ActionInfo* info) except *
cdef void apply_discard(GameState state, const ActionInfo* info) except *
