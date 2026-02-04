# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
One-hot encoding helper functions for game state arrays.

Note: The inline function implementations are in encoding.pxd to enable
proper cross-module inlining. This file exists only to satisfy the Cython
build system's requirement for a .pyx file to accompany the .pxd.

See encoding.pxd for the actual implementations of:
- set_one_hot()
- get_one_hot_index() - for permutation vectors only (O(n) scan)
- clear_one_hot()
- set_one_hot_with_compact() - canonical setter for mirrored one-hot values
- clear_one_hot_with_compact() - canonical clearer for mirrored one-hot values
"""

# No implementation needed here - all inline functions are defined in encoding.pxd
