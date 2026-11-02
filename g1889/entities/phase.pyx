"""PHASE entity; see phase.pxd."""

from libc.stdint cimport int16_t

from g1889.data.tables cimport NUM_GAME_PHASES
from g1889.entities.corp cimport invalidate_all_routes
from g1889.state cimport GLOBALS, LAYOUT, GameState


cdef int game_phase(GameState state) noexcept nogil:
    return <int>state._data[LAYOUT.global_offset + GLOBALS.game_phase]


cdef class Phase:
    """Stateless handle over the game phase."""

    cpdef int current(self, GameState state):
        return game_phase(state)

    cpdef void set_current(self, GameState state, int phase):
        """Set the phase; revenues and train limits change, so routes go stale."""
        assert 0 <= phase < <int>NUM_GAME_PHASES, f"game phase {phase} out of range"
        state._data[LAYOUT.global_offset + GLOBALS.game_phase] = <int16_t>phase
        invalidate_all_routes(state)


PHASE = Phase()
