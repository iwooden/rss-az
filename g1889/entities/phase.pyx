"""PHASE entity; see phase.pxd.

A phase's tile colors, train limit, offboard values, private-purchase window,
and ORs per set are table lookups by ``game_phase`` (the ORs per set take
effect when the next SR ends, ``TURN.or_count``), as are UTF's ¥50 revenue
(``private_revenue``) and D's availability (``trains_depot_types``).
``advance`` applies the rest: rusting and the privates' closing.
"""

from libc.stdint cimport int16_t

from g1889.constants cimport OWNER_CLOSED, OWNER_CORP, OWNER_NOT_IN_PLAY, OWNER_PLAYER
from g1889.data.tables cimport (
    CLOSE_PRIVATES_PHASE, NUM_GAME_PHASES, NUM_PRIVATES, NUM_TRAIN_TYPES, PHASE_ON_TRAIN, PRIVATE_UTF,
    TRAIN_RUSTS_ON,
)
from g1889.entities.corp cimport invalidate_all_routes
from g1889.entities.private cimport private_close, private_owner
from g1889.entities.trains cimport trains_rust
from g1889.state cimport GLOBALS, LAYOUT, GameState


cdef int game_phase(GameState state) noexcept nogil:
    return <int>state._data[LAYOUT.global_offset + GLOBALS.game_phase]


cdef int next_rust_phase(int phase) noexcept nogil:
    """The first game phase after ``phase`` whose start rusts a rank (4 for
    the 2s, 6 for the 3s, D for the 4s), or -1 after the last."""
    cdef int later, t
    for later in range(phase + 1, <int>NUM_GAME_PHASES):
        for t in range(<int>NUM_TRAIN_TYPES):
            if TRAIN_RUSTS_ON[t] == PHASE_ON_TRAIN[later]:
                return later
    return -1


cdef void phase_advance(GameState state) noexcept nogil:
    """Start the next phase as ``Phase#buying_train!`` does: ``next!``,
    then ``rust_trains!``, then the first 5-train's ``close_companies``.
    Every train of a rank that rusts on the new phase's train leaves play
    (2s at phase 4, 3s at 6, 4s at D), and at phase 5 every private
    closes but a player-owned UTF (``event_close_companies!``: its
    ``close`` ability, ``on_phase: 'never'``, binds only for a player
    owner). Ruby runs ``rust_trains!`` on every purchase, but a rank's
    trains are gone after its first. Revenues and train limits change, so
    routes go stale. The caller checks that a next phase exists."""
    cdef int phase = game_phase(state) + 1
    cdef int p, owner
    state._data[LAYOUT.global_offset + GLOBALS.game_phase] = <int16_t>phase
    invalidate_all_routes(state)
    trains_rust(state, PHASE_ON_TRAIN[phase])
    if phase != <int>CLOSE_PRIVATES_PHASE:
        return
    for p in range(<int>NUM_PRIVATES):
        owner = private_owner(state, p)
        if (owner == <int>OWNER_NOT_IN_PLAY or owner == <int>OWNER_CLOSED
                or p == <int>PRIVATE_UTF and <int>OWNER_PLAYER <= owner < <int>OWNER_CORP):
            continue
        private_close(state, p)


cdef bint phase_on_train_bought(GameState state, int train_type) noexcept nogil:
    """After any purchase (``Phase#buying_train!``): the first train of the
    next phase's rank starts that phase. Only a depot train or a D
    trade-in can be one. Returns whether the phase advanced."""
    cdef int phase = game_phase(state)
    if phase + 1 < <int>NUM_GAME_PHASES and PHASE_ON_TRAIN[phase + 1] == train_type:
        phase_advance(state)
        return True
    return False


cdef class Phase:
    """Stateless handle over the game phase."""

    cpdef int current(self, GameState state):
        return game_phase(state)

    cpdef void set_current(self, GameState state, int phase):
        """Set the phase; revenues and train limits change, so routes go stale."""
        assert 0 <= phase < <int>NUM_GAME_PHASES, f"game phase {phase} out of range"
        state._data[LAYOUT.global_offset + GLOBALS.game_phase] = <int16_t>phase
        invalidate_all_routes(state)

    cpdef void advance(self, GameState state):
        """Start the next phase (``phase_advance``)."""
        assert game_phase(state) + 1 < <int>NUM_GAME_PHASES, "no phase after the last"
        phase_advance(state)

    cpdef bint on_train_bought(self, GameState state, int train_type):
        """After any purchase: start the next phase if ``train_type`` is its
        rank (``phase_on_train_bought``). Returns whether the phase advanced."""
        return phase_on_train_bought(state, train_type)


PHASE = Phase()
