"""DIVIDEND: the operating corp pays out or withholds its revenue
(1889/engine-design.md §5.2, §6.1). The handler assumes a legal action and
changes the state only through entity handles."""

from g1889.actions cimport ActionInfo
from g1889.state cimport GameState

cdef void apply_dividend(GameState state, const ActionInfo* info) except *
