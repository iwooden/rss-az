"""Operating round: the automated OR_START, TURN_START, ROUTE, TURN_END, and
OR_END steps, and the TRACK, TILE, TOKEN, and BUY_PRIVATE handlers
(1889/engine-design.md §5.2-5.3, §6.1); ``dividend.pyx`` and
``buy_train.pyx`` handle the turn's DIVIDEND, BUY_TRAIN, TRAIN_PRICE, and
DISCARD. They assume legal actions and change the state only through entity
handles."""

from g1889.actions cimport ActionInfo
from g1889.state cimport GameState

cdef void run_or_start(GameState state) except *
cdef void run_turn_start(GameState state) except *
cdef void run_route(GameState state) except *
cdef void run_turn_end(GameState state) except *
cdef void run_or_end(GameState state) except *
cdef void apply_track(GameState state, const ActionInfo* info) except *
cdef void apply_tile(GameState state, const ActionInfo* info) except *
cdef void apply_token(GameState state, const ActionInfo* info) except *
cdef void apply_buy_private(GameState state, const ActionInfo* info) except *
