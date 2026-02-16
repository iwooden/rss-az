# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Shared inline helpers for stride-3 offer buffer writes.

Both acquisition and closing phases store offers as (field0, field1, field2)
triples in hidden state with a count/index pair. These helpers centralize
the write-side operations (reset, append, advance, set) that encode the
stride-3 layout. Reads are left inline at call sites.

Acquisition buffer: (owner_type, buyer_corp_id, company_id)
Closing buffer:     (owner_type, owner_id, company_id)
"""


cdef inline void offer_buf_reset(float* data, int count_offset, int index_offset) noexcept nogil:
    """Reset offer buffer count and index to 0."""
    data[count_offset] = 0.0
    data[index_offset] = 0.0


cdef inline void offer_buf_set_count(float* data, int count_offset, int count) noexcept nogil:
    """Write offer count."""
    data[count_offset] = <float>count


cdef inline void offer_buf_set_index(float* data, int index_offset, int index) noexcept nogil:
    """Write offer index."""
    data[index_offset] = <float>index


cdef inline void offer_buf_advance(float* data, int index_offset) noexcept nogil:
    """Increment offer index by 1."""
    cdef int index = <int>data[index_offset]
    data[index_offset] = <float>(index + 1)


cdef inline void offer_buf_append(float* data, int buffer_offset, int position,
                                   int field0, int field1, int field2) noexcept nogil:
    """Write a 3-tuple at the given position in the buffer."""
    cdef int base = buffer_offset + (position * 3)
    data[base] = <float>field0
    data[base + 1] = <float>field1
    data[base + 2] = <float>field2
