# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
One-hot encoding helper functions for game state arrays.

Note: The inline function implementations are in encoding.pxd to enable
proper cross-module inlining. This file exists only to satisfy the Cython
build system's requirement for a .pyx file to accompany the .pxd.

See encoding.pxd for the actual implementations of:
- set_one_hot()
- get_one_hot_index()
- clear_one_hot()
"""

# No implementation needed here - all inline functions are defined in encoding.pxd
