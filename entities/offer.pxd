# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
OfferBuffer: Cached-offset access to stride-3 offer buffers in hidden state.

Both acquisition and closing phases store offers as (field0, field1, field2)
triples in hidden state with a count/index pair. This class caches the three
layout offsets so call sites don't need to pass them repeatedly.

Acquisition buffer: (owner_type, buyer_corp_id, company_id)
Closing buffer:     (owner_type, owner_id, company_id)
"""

from core.state cimport GameState


cdef class OfferBuffer:
    cdef bint _is_close_buffer
    cdef int _count_offset
    cdef int _index_offset
    cdef int _buffer_offset

    cpdef void initialize(self, GameState state)

    # Write operations
    cdef void reset(self, float* data) noexcept nogil
    cdef void set_count(self, float* data, int count) noexcept nogil
    cdef void set_index(self, float* data, int index) noexcept nogil
    cdef void advance(self, float* data) noexcept nogil
    cdef void append(self, float* data, int position, int field0, int field1, int field2) noexcept nogil

    # Read operations
    cdef int get_count(self, float* data) noexcept nogil
    cdef int get_index(self, float* data) noexcept nogil
    cdef int offer_base(self, int index) noexcept nogil
