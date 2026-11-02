"""Stock round handlers: STOCK, PAR, SELL_AMOUNT, and the automated SR_END
step (1889/engine-design.md §5.2-5.3, §6.1). They assume legal actions and
change the state only through entity handles."""

from g1889.actions cimport ActionInfo
from g1889.state cimport GameState

cdef void start_stock_round(GameState state) except *
cdef void apply_stock(GameState state, const ActionInfo* info) except *
cdef void apply_par(GameState state, const ActionInfo* info) except *
cdef void apply_sell_amount(GameState state, const ActionInfo* info) except *
cdef void run_sr_end(GameState state) except *
