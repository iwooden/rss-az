"""DIVIDEND handler; see dividend.pxd.

Ruby's ``Step::Dividend`` (``step/dividend.rb``) with 1889's options, payout
and withhold (rules §8.7). The bank pays all of it, and a payment that takes
it below zero breaks it (``BANK.transfer``); the order Ruby pays in cannot
change the result. Ruby also marks the corp's trains as run
(``train.operated``), which never binds in 1889 (engine-design.md §8).
"""

from g1889.actions cimport ACTION_PAYOUT, ACTION_WITHHOLD, ActionInfo
from g1889.constants cimport OWNER_CORP, OWNER_PLAYER, STEP_BUY_TRAIN
from g1889.entities.bank cimport BANK_ACCOUNT, Bank
from g1889.entities.corp cimport SHARES_PER_CORP, Corporation, corp_pool_shares
from g1889.entities.market cimport Market
from g1889.entities.player cimport player_shares
from g1889.entities.turn cimport TurnState, turn_num_players, turn_operating_corp, turn_route_revenue
from g1889.state cimport GameState

from g1889.entities import bank as _bank, corp as _corp, market as _market, turn as _turn

cdef Bank BANK = _bank.BANK
cdef list CORPS = _corp.CORPS
cdef Market MARKET = _market.MARKET
cdef TurnState TURN = _turn.TURN


cdef void _pay(GameState state, int account, int shares, int revenue) except *:
    """The bank pays ``shares`` their part of ``revenue``, rounded up
    (``dividends_for_entity``; 1889 revenues are multiples of ¥10)."""
    cdef int amount = (shares * revenue + <int>SHARES_PER_CORP - 1) // <int>SHARES_PER_CORP
    if amount:
        BANK.transfer(state, <int>BANK_ACCOUNT, account, amount)


cdef void apply_dividend(GameState state, const ActionInfo* info) except *:
    """``process_dividend``. A payout pays each share a tenth of the revenue:
    players for the shares they hold (the President's Certificate counts
    2), the corp for its shares in the Open Market (``holder_for_corporation``
    is the share pool under full capitalization), and nobody for those in
    the IPO; the price moves right. A withhold puts the revenue in the
    treasury, and the price moves left; Ruby skips the step at zero revenue
    with the same withhold (``share_price_change``). Either way the revenue
    is the corp's last revenue, and BUY_TRAIN follows."""
    cdef int corp = turn_operating_corp(state)
    cdef int revenue = turn_route_revenue(state)
    cdef int i
    if info.type == ACTION_PAYOUT:
        for i in range(turn_num_players(state)):
            _pay(state, <int>OWNER_PLAYER + i, player_shares(state, i, corp), revenue)
        _pay(state, <int>OWNER_CORP + corp, corp_pool_shares(state, corp), revenue)
        MARKET.move_right(state, corp)
    else:
        assert info.type == ACTION_WITHHOLD, f"apply_dividend: action type {info.type}"
        if revenue:
            BANK.transfer(state, <int>BANK_ACCOUNT, <int>OWNER_CORP + corp, revenue)
        MARKET.move_left(state, corp)
    (<Corporation>CORPS[corp]).set_last_revenue(state, revenue)
    TURN.set_route_revenue(state, 0)
    TURN.set_step(state, STEP_BUY_TRAIN)
