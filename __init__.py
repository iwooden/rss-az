# RSS Cython Core - High-performance game engine for Rolling Stock Stars.
#
# This package provides a Cython implementation of the game engine optimized for
# AlphaZero-style self-play training. The key design principle is that the game
# state is a contiguous float array that can be passed directly to PyTorch
# without serialization overhead.

from .state import (
    GameState,
    get_market_index,
    get_market_price,
    get_state_size,
    get_visible_size,
    PHASE_NAMES,
)
from .data import (
    COMPANY_NAMES,
    COMPANY_NAME_TO_ID,
    CORP_NAMES,
    CORP_NAME_TO_ID,
)

__all__ = [
    'GameState',
    'get_market_index',
    'get_market_price',
    'get_state_size',
    'get_visible_size',
    'PHASE_NAMES',
    'COMPANY_NAMES',
    'COMPANY_NAME_TO_ID',
    'CORP_NAMES',
    'CORP_NAME_TO_ID',
]
