# cython: language_level=3
"""
Entity module - provides entity handles for game state access.

Entities are lightweight wrappers that provide clean getter/setter access
to the game state vector. They are instantiated once at module load and
reused across all game states.
"""

from entities import player as _player_module
from entities import corp as _corp_module
from entities import fi as _fi_module
from entities import market as _market_module
from entities import turn as _turn_module
from entities import company as _company_module
from entities import deck as _deck_module

# Re-export at package level
Player = _player_module.Player
PLAYERS = _player_module.PLAYERS

Corporation = _corp_module.Corporation
CORPS = _corp_module.CORPS

ForeignInvestor = _fi_module.ForeignInvestor
FI = _fi_module.FI

Market = _market_module.Market
MARKET = _market_module.MARKET

TurnState = _turn_module.TurnState
TURN = _turn_module.TURN

Company = _company_module.Company
COMPANIES = _company_module.COMPANIES
COMPANIES_BY_NAME = _company_module.COMPANIES_BY_NAME

Deck = _deck_module.Deck
DECK = _deck_module.DECK

__all__ = [
    'Player', 'PLAYERS',
    'Corporation', 'CORPS',
    'ForeignInvestor', 'FI',
    'Market', 'MARKET',
    'TurnState', 'TURN',
    'Company', 'COMPANIES', 'COMPANIES_BY_NAME',
    'Deck', 'DECK',
]
