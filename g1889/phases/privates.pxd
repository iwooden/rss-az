"""Privates: the interjection slots (buy a private, DR's exchange, MF's port
tile), BUY_PRIVATE_PRE, BUY_PRIVATE_POST, and the PRIVATE_PRICE, ER_BONUS, and
PORT sub-phases (1889/engine-design.md §5.1-5.3). The handlers assume legal
actions and change the state only through entity handles."""

from g1889.actions cimport ActionInfo
from g1889.state cimport GameState

cdef void apply_interjection(GameState state, const ActionInfo* info) except *
cdef void apply_buy_private_pre(GameState state, const ActionInfo* info) except *
cdef void apply_buy_private_post(GameState state, const ActionInfo* info) except *
cdef void apply_private_price(GameState state, const ActionInfo* info) except *
cdef void apply_er_bonus(GameState state, const ActionInfo* info) except *
cdef void apply_port(GameState state, const ActionInfo* info) except *
