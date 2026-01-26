# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
One-hot encoding helper functions for game state arrays.

This module provides reusable inline functions for one-hot encoding operations
across game state entities (turns, corporations, etc). All functions operate on
raw float* pointers for maximum flexibility and zero-overhead inlining.

Used to eliminate ~300 LOC of duplicated one-hot encoding logic across turn.pyx,
corp.pyx, and state.pyx.
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
    Get index of 1.0 in one-hot encoding.

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
