# cython: language_level=3
"""
Entity module - provides entity handles for game state access.

Entities are lightweight wrappers that provide clean getter/setter access
to the game state vector. They are instantiated once at module load and
reused across all game states.
"""

from entities import player as _player_module
from entities import corp as _corp_module

# Re-export at package level
Player = _player_module.Player
PLAYERS = _player_module.PLAYERS

Corporation = _corp_module.Corporation
CORPS = _corp_module.CORPS

__all__ = ['Player', 'PLAYERS', 'Corporation', 'CORPS']
