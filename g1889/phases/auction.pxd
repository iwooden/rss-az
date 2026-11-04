"""Private auction handlers: AUCTION and BID (1889/engine-design.md §5.2,
§6.1). They assume legal actions and change the state only through
entity handles."""

from g1889.actions cimport ActionInfo
from g1889.state cimport GameState

cdef void select_bid(GameState state, int private) except *
cdef void apply_auction(GameState state, const ActionInfo* info) except *
cdef void apply_bid(GameState state, const ActionInfo* info) except *
