# cython: language_level=3
"""
One-hot encoding helper functions for game state arrays.

This module provides reusable inline functions for one-hot encoding operations
across game state entities (turns, corporations, etc). All functions operate on
raw float* pointers for maximum flexibility and zero-overhead inlining.

Used to eliminate ~300 LOC of duplicated one-hot encoding logic across turn.pyx,
corp.pyx, and state.pyx.

Note: Inline functions must have their full definition in .pxd files for proper
inlining across modules (similar to C inline functions in headers).
"""


cdef inline void set_one_hot(float* data, int offset, int size, int value) noexcept nogil:
    """
    Set one-hot encoding: clear all slots, then set value position to 1.0.

    Args:
        data: Pointer to float array
        offset: Starting position in array
        size: Number of slots in one-hot encoding
        value: Index to set to 1.0 (0-indexed, must be in [0, size))

    If value is out of bounds, all slots are cleared (no value set to 1.0).
    """
    cdef int i
    for i in range(size):
        data[offset + i] = 0.0
    if 0 <= value < size:
        data[offset + value] = 1.0


cdef inline int get_one_hot_index(float* data, int offset, int size) noexcept nogil:
    """
    Get index of 1.0 in one-hot encoding via O(n) scan.

    NOTE: For one-hot values with hidden compact mirrors, use the compact value
    directly for O(1) access. This function is only appropriate for permutation
    vectors (like turn_order) where each element is independently one-hot encoded
    and no single compact mirror exists.

    Args:
        data: Pointer to float array
        offset: Starting position in array
        size: Number of slots in one-hot encoding

    Returns:
        Index (0-based) of slot containing 1.0, or -1 if not found.
    """
    cdef int i
    for i in range(size):
        if data[offset + i] == 1.0:
            return i
    return -1


cdef inline void clear_one_hot(float* data, int offset, int size) noexcept nogil:
    """
    Clear one-hot encoding (set all slots to 0.0).

    Args:
        data: Pointer to float array
        offset: Starting position in array
        size: Number of slots in one-hot encoding
    """
    cdef int i
    for i in range(size):
        data[offset + i] = 0.0


cdef inline void set_one_hot_with_compact(
    float* data, int one_hot_offset, int size, int compact_offset, int value
) noexcept nogil:
    """
    Set one-hot encoding and update hidden compact storage atomically.

    This is the canonical way to update one-hot encoded state values that have
    hidden compact mirrors. Using this function ensures visible and hidden state
    stay synchronized and enforces the invariant that all one-hot values must
    have a compact mirror.

    Args:
        data: Pointer to float array
        one_hot_offset: Starting position of one-hot encoding in array
        size: Number of slots in one-hot encoding
        compact_offset: Position of hidden compact value in array
        value: Index to set (0-indexed, must be in [0, size))

    If value is out of bounds, one-hot is cleared and compact is set to -1.0.
    """
    set_one_hot(data, one_hot_offset, size, value)
    if 0 <= value < size:
        data[compact_offset] = <float>value
    else:
        data[compact_offset] = -1.0


cdef inline void clear_one_hot_with_compact(
    float* data, int one_hot_offset, int size, int compact_offset
) noexcept nogil:
    """
    Clear one-hot encoding and set hidden compact storage to -1.0.

    Args:
        data: Pointer to float array
        one_hot_offset: Starting position of one-hot encoding in array
        size: Number of slots in one-hot encoding
        compact_offset: Position of hidden compact value in array
    """
    clear_one_hot(data, one_hot_offset, size)
    data[compact_offset] = -1.0
