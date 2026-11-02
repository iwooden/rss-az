"""PRIVATES entity; see private.pxd."""

from libc.stdint cimport int16_t

from g1889.constants cimport OWNER_CORP, OWNER_NOT_IN_PLAY, OWNER_PLAYER, OWNER_UNSOLD
from g1889.data.tables cimport (
    NUM_CORPS, PRIVATE_REVENUE, PRIVATE_UTF, UTF_LATE_PHASE, UTF_LATE_REVENUE,
)
from g1889.state cimport GLOBALS, LAYOUT, PRIVATE_FIELDS, GameState

from g1889.data.tables import PrivateId


cdef inline int _slot(int private, int field) noexcept nogil:
    return LAYOUT.privates_offset + private * PRIVATE_FIELDS.size + field


cdef int private_owner(GameState state, int private) noexcept nogil:
    return <int>state._data[_slot(private, PRIVATE_FIELDS.owner)]


cdef int private_revenue(GameState state, int private) noexcept nogil:
    """Revenue paid to the owner; a player-owned UTF pays ¥50 from phase 5."""
    cdef int owner = private_owner(state, private)
    if (private == <int>PRIVATE_UTF and owner >= <int>OWNER_PLAYER and owner < <int>OWNER_CORP
            and state._data[LAYOUT.global_offset + GLOBALS.game_phase] >= <int>UTF_LATE_PHASE):
        return <int>UTF_LATE_REVENUE
    return PRIVATE_REVENUE[private]


cdef class Private:
    """Stateless handle over one private's block."""

    def __cinit__(self, int private_id, str sym):
        self.private_id = private_id
        self.sym = sym

    cpdef int owner(self, GameState state):
        return private_owner(state, self.private_id)

    cpdef void set_owner(self, GameState state, int owner):
        """Set the owner code: not in play, closed, unsold, a player, or a corp."""
        cdef int num_players = <int>state._data[LAYOUT.global_offset + GLOBALS.num_players]
        assert (<int>OWNER_NOT_IN_PLAY <= owner <= <int>OWNER_UNSOLD
                or <int>OWNER_PLAYER <= owner < <int>OWNER_PLAYER + num_players
                or <int>OWNER_CORP <= owner < <int>OWNER_CORP + <int>NUM_CORPS), f"owner code {owner}"
        state._data[_slot(self.private_id, PRIVATE_FIELDS.owner)] = <int16_t>owner

    cpdef int flags(self, GameState state):
        return <int>state._data[_slot(self.private_id, PRIVATE_FIELDS.flags)]

    cpdef void set_flags(self, GameState state, int flags):
        state._data[_slot(self.private_id, PRIVATE_FIELDS.flags)] = <int16_t>flags

    cpdef int revenue(self, GameState state):
        return private_revenue(state, self.private_id)


PRIVATES = [Private(p.value, p.name.removeprefix('PRIVATE_')) for p in PrivateId]
PRIVATES_BY_SYM = {private.sym: private for private in PRIVATES}
