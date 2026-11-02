"""TRAINS entity; see trains.pxd.

Ruby's ``Depot`` keeps every train object and moves it between owners
(``depot.rb``, ``Game#buy_train``, ``Game#rust``); the engine counts trains
by type in the depot, the pool (Ruby's ``discarded``), and each corp. Every
change to a corp's count goes through ``set_corp_trains``, which makes the
corp's route cache stale. A rusted train leaves play, so the counts of a
rank that has rusted no longer add up to ``TRAIN_COUNT``.
"""

from libc.stdint cimport int16_t

from g1889.constants cimport OWNER_CORP
from g1889.data.tables cimport (
    D_TRAIN, NUM_CORPS, NUM_TRAIN_TYPES, PHASE_TRAIN_LIMIT, TRAIN_AVAILABLE_ON, TRAIN_PRICE, TRAIN_RUSTS_ON,
    TRAIN_TRADE_IN,
)
from g1889.entities.bank cimport BANK_ACCOUNT, Bank
from g1889.entities.corp cimport invalidate_route
from g1889.state cimport CORP_FIELDS, GLOBALS, LAYOUT, TRAIN_FIELDS, GameState

from g1889.entities import bank as _bank

cdef Bank BANK = _bank.BANK


cdef inline int _corp_slot(int corp, int train_type) noexcept nogil:
    return LAYOUT.corps_offset + corp * CORP_FIELDS.size + CORP_FIELDS.trains + train_type


cdef inline int _phase(GameState state) noexcept nogil:
    """The game phase, read raw: PHASE owns it and cimports this module."""
    return <int>state._data[LAYOUT.global_offset + GLOBALS.game_phase]


cdef int depot_count(GameState state, int train_type) noexcept nogil:
    """Copies left in the depot; -1 is unlimited (D)."""
    return <int>state._data[LAYOUT.trains_offset + TRAIN_FIELDS.depot + train_type]


cdef int pool_count(GameState state, int train_type) noexcept nogil:
    return <int>state._data[LAYOUT.trains_offset + TRAIN_FIELDS.pool + train_type]


cdef int corp_train_count(GameState state, int corp, int train_type) noexcept nogil:
    return <int>state._data[_corp_slot(corp, train_type)]


cdef int corp_train_total(GameState state, int corp) noexcept nogil:
    cdef int t, total = 0
    for t in range(<int>NUM_TRAIN_TYPES):
        total += corp_train_count(state, corp, t)
    return total


cdef int trains_depot_types(GameState state) noexcept nogil:
    """The train types the depot sells now, as a type mask (``Depot#depot_trains``
    without the pool): the lowest rank left, and any later rank whose
    ``available_on`` phase has come (D from phase 6)."""
    cdef int phase = _phase(state)
    cdef int mask = 0
    cdef int t
    for t in range(<int>NUM_TRAIN_TYPES):
        if depot_count(state, t) == 0:
            continue
        if mask == 0 or 0 <= TRAIN_AVAILABLE_ON[t] <= phase:
            mask |= 1 << t
    return mask


cdef int trains_cheapest_type(GameState state) noexcept nogil:
    """The type of the cheapest train in the depot or the pool
    (``Depot#min_depot_train``, whose ``depot_trains`` include the pool), -1
    with none. Ruby lists the depot's trains first, so when the depot sells
    that type too, its train is the one (``trains_cheapest_in_pool``)."""
    cdef int types = trains_depot_types(state)
    cdef int cheapest = -1
    cdef int t
    for t in range(<int>NUM_TRAIN_TYPES):
        if (types >> t & 1 or pool_count(state, t)) and (cheapest < 0 or TRAIN_PRICE[t] < TRAIN_PRICE[cheapest]):
            cheapest = t
    return cheapest


cdef bint trains_cheapest_in_pool(GameState state) noexcept nogil:
    """Whether ``Depot#min_depot_train`` is the pool's: the depot does not sell its type."""
    cdef int t = trains_cheapest_type(state)
    return t >= 0 and not (trains_depot_types(state) >> t & 1)


cdef int trains_cheapest_price(GameState state) noexcept nogil:
    """The cheapest train's price (``Depot#min_depot_price``), 0 with none."""
    cdef int t = trains_cheapest_type(state)
    return TRAIN_PRICE[t] if t >= 0 else 0


cdef int trains_limit(GameState state) noexcept nogil:
    """The phase's train limit per corp (``train_limit``)."""
    return PHASE_TRAIN_LIMIT[_phase(state)]


cdef int trains_first_crowded(GameState state) noexcept nogil:
    """The first corp over the limit in corp order, which discards next
    (``crowded_corps``, ``DiscardTrain#active_entities``), or -1."""
    cdef int limit = trains_limit(state)
    cdef int c
    for c in range(<int>NUM_CORPS):
        if corp_train_total(state, c) > limit:
            return c
    return -1


cdef int trade_in_price(int train_type) noexcept nogil:
    """What a D costs with a ``train_type`` traded in (``Train#price(exchange)``):
    ¥800 for a 4, 5, or 6."""
    return TRAIN_PRICE[<int>D_TRAIN] - TRAIN_TRADE_IN[<int>D_TRAIN][train_type]


cdef inline int _check(int train_type, int count, int lowest) except -1:
    assert 0 <= train_type < <int>NUM_TRAIN_TYPES and count >= lowest, f"{count} trains of type {train_type}"
    return 0


cdef class Trains:
    """Stateless handle over the trains block and the corps' trains."""

    cpdef int depot(self, GameState state, int train_type):
        return depot_count(state, train_type)

    cpdef void set_depot(self, GameState state, int train_type, int count):
        _check(train_type, count, -1)
        state._data[LAYOUT.trains_offset + TRAIN_FIELDS.depot + train_type] = <int16_t>count

    cpdef int pool(self, GameState state, int train_type):
        return pool_count(state, train_type)

    cpdef void set_pool(self, GameState state, int train_type, int count):
        _check(train_type, count, 0)
        state._data[LAYOUT.trains_offset + TRAIN_FIELDS.pool + train_type] = <int16_t>count

    cpdef int corp_trains(self, GameState state, int corp, int train_type):
        return corp_train_count(state, corp, train_type)

    cpdef void set_corp_trains(self, GameState state, int corp, int train_type, int count):
        """Set a corp's count of one train type; the corp's route cache goes stale."""
        assert 0 <= corp < <int>NUM_CORPS, f"corp {corp} out of range"
        _check(train_type, count, 0)
        state._data[_corp_slot(corp, train_type)] = <int16_t>count
        invalidate_route(state, corp)

    cpdef int corp_total(self, GameState state, int corp):
        assert 0 <= corp < <int>NUM_CORPS, f"corp {corp} out of range"
        return corp_train_total(state, corp)

    cpdef int depot_types(self, GameState state):
        return trains_depot_types(state)

    cpdef int cheapest_price(self, GameState state):
        return trains_cheapest_price(state)

    cpdef int cheapest_type(self, GameState state):
        return trains_cheapest_type(state)

    cpdef bint cheapest_in_pool(self, GameState state):
        return trains_cheapest_in_pool(state)

    cpdef int limit(self, GameState state):
        return trains_limit(state)

    cpdef bint crowded(self, GameState state, int corp):
        """Whether the corp holds more trains than the phase allows."""
        return self.corp_total(state, corp) > trains_limit(state)

    cpdef int first_crowded(self, GameState state):
        return trains_first_crowded(state)

    # Moves. The buyer pays the bank for a depot or pool train and the seller
    # for a corp's (``Game#buy_train``); the handlers apply the phase change.

    cdef void _give(self, GameState state, int corp, int train_type) except *:
        self.set_corp_trains(state, corp, train_type, corp_train_count(state, corp, train_type) + 1)

    cpdef void buy_from_depot(self, GameState state, int corp, int train_type):
        """The corp buys a depot train at its list price; the unlimited D stays at -1."""
        cdef int left = depot_count(state, train_type)
        assert trains_depot_types(state) >> train_type & 1, f"the depot does not sell type {train_type}"
        BANK.transfer(state, <int>OWNER_CORP + corp, <int>BANK_ACCOUNT, TRAIN_PRICE[train_type])
        if left > 0:
            self.set_depot(state, train_type, left - 1)
        self._give(state, corp, train_type)

    cpdef void buy_from_pool(self, GameState state, int corp, int train_type):
        """The corp buys a pool train at its list price."""
        assert pool_count(state, train_type) > 0, f"no type {train_type} in the pool"
        BANK.transfer(state, <int>OWNER_CORP + corp, <int>BANK_ACCOUNT, TRAIN_PRICE[train_type])
        self.set_pool(state, train_type, pool_count(state, train_type) - 1)
        self._give(state, corp, train_type)

    cpdef void buy_from_corp(self, GameState state, int corp, int seller, int train_type, int price):
        """The corp buys one of ``seller``'s trains for ``price``, paid to the seller."""
        cdef int held = corp_train_count(state, seller, train_type)
        assert seller != corp and held > 0 and price > 0, f"corp {corp} cannot buy {seller}'s type {train_type}"
        BANK.transfer(state, <int>OWNER_CORP + corp, <int>OWNER_CORP + seller, price)
        self.set_corp_trains(state, seller, train_type, held - 1)
        self._give(state, corp, train_type)

    cpdef void trade_in_for_d(self, GameState state, int corp, int train_type):
        """The corp trades one of its 4s, 5s, or 6s and ¥800 for a D: the old
        train goes to the pool first (``Depot#reclaim_train``), where a 4
        then rusts as phase D begins."""
        assert trade_in_price(train_type) < TRAIN_PRICE[<int>D_TRAIN], f"type {train_type} has no trade-in"
        assert trains_depot_types(state) >> <int>D_TRAIN & 1, "the depot does not sell D yet"
        self.discard(state, corp, train_type)
        BANK.transfer(state, <int>OWNER_CORP + corp, <int>BANK_ACCOUNT, trade_in_price(train_type))
        self._give(state, corp, <int>D_TRAIN)

    cpdef void discard(self, GameState state, int corp, int train_type):
        """One of the corp's trains goes to the pool without compensation."""
        cdef int held = corp_train_count(state, corp, train_type)
        assert held > 0, f"corp {corp} has no type {train_type}"
        self.set_corp_trains(state, corp, train_type, held - 1)
        self.set_pool(state, train_type, pool_count(state, train_type) + 1)

    cpdef void rust(self, GameState state, int train_type):
        """Every train of each rank that rusts on ``train_type`` leaves play:
        the corps', the pool's, and any left in the depot (``rust_trains!``)."""
        cdef int r, c
        _check(train_type, 0, 0)
        for r in range(<int>NUM_TRAIN_TYPES):
            if TRAIN_RUSTS_ON[r] != train_type:
                continue
            for c in range(<int>NUM_CORPS):
                if corp_train_count(state, c, r):
                    self.set_corp_trains(state, c, r, 0)
            self.set_pool(state, r, 0)
            if depot_count(state, r) > 0:
                self.set_depot(state, r, 0)


TRAINS = Trains()
