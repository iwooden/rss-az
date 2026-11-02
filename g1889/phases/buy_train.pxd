"""BUY_TRAIN, TRAIN_PRICE, and DISCARD (1889/engine-design.md §5.2-5.3, §6.1).
The handlers assume legal actions and change the state only through entity
handles."""

from g1889.actions cimport ActionInfo
from g1889.state cimport GameState

cdef void apply_buy_train(GameState state, const ActionInfo* info) except *
cdef void apply_train_price(GameState state, const ActionInfo* info) except *
cdef void apply_discard(GameState state, const ActionInfo* info) except *
