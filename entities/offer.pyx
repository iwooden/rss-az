# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
OfferBuffer implementation.

Two module-level singletons (ACQ_OFFERS, CLOSE_OFFERS) are initialized once
per GameState via initialize(), caching the three hidden-state offsets for
their respective buffers.
"""

from core.state cimport GameState


cdef class OfferBuffer:

    def __cinit__(self, bint is_close_buffer):
        self._is_close_buffer = is_close_buffer
        self._count_offset = 0
        self._index_offset = 0
        self._buffer_offset = 0

    cpdef void initialize(self, GameState state):
        """Cache buffer offsets from state layout. Call once per game."""
        if self._is_close_buffer:
            self._count_offset = state._layout.hidden_close_offer_count_offset
            self._index_offset = state._layout.hidden_close_offer_index_offset
            self._buffer_offset = state._layout.hidden_close_offer_buffer_offset
        else:
            self._count_offset = state._layout.hidden_offer_count_offset
            self._index_offset = state._layout.hidden_offer_index_offset
            self._buffer_offset = state._layout.hidden_offer_buffer_offset

    # =========================================================================
    # WRITE OPERATIONS
    # =========================================================================

    cdef void reset(self, float* data) noexcept nogil:
        """Reset offer buffer count and index to 0."""
        data[self._count_offset] = 0.0
        data[self._index_offset] = 0.0

    cdef void set_count(self, float* data, int count) noexcept nogil:
        """Write offer count."""
        data[self._count_offset] = <float>count

    cdef void set_index(self, float* data, int index) noexcept nogil:
        """Write offer index."""
        data[self._index_offset] = <float>index

    cdef void advance(self, float* data) noexcept nogil:
        """Increment offer index by 1."""
        cdef int index = <int>data[self._index_offset]
        data[self._index_offset] = <float>(index + 1)

    cdef void append(self, float* data, int position,
                     int field0, int field1, int field2) noexcept nogil:
        """Write a 3-tuple at the given position in the buffer."""
        cdef int base = self._buffer_offset + (position * 3)
        data[base] = <float>field0
        data[base + 1] = <float>field1
        data[base + 2] = <float>field2

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    cdef int get_count(self, float* data) noexcept nogil:
        """Read offer count."""
        return <int>data[self._count_offset]

    cdef int get_index(self, float* data) noexcept nogil:
        """Read current offer index."""
        return <int>data[self._index_offset]

    cdef int offer_base(self, int index) noexcept nogil:
        """Compute base offset for offer at given index."""
        return self._buffer_offset + (index * 3)


# Module-level singletons
ACQ_OFFERS = OfferBuffer(False)
CLOSE_OFFERS = OfferBuffer(True)
