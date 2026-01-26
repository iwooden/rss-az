# cython: language_level=3
"""
One-hot encoding helper functions for game state arrays.

This module provides reusable inline functions for one-hot encoding operations
across game state entities (turns, corporations, etc). All functions operate on
raw float* pointers for maximum flexibility and zero-overhead inlining.

Used to eliminate ~300 LOC of duplicated one-hot encoding logic across turn.pyx,
corp.pyx, and state.pyx.
"""

cdef inline void set_one_hot(float* data, int offset, int size, int value) noexcept nogil

cdef inline int get_one_hot_index(float* data, int offset, int size) noexcept nogil

cdef inline void clear_one_hot(float* data, int offset, int size) noexcept nogil
