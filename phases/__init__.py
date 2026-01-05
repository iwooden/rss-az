# Phase implementations for the game engine.
# Each phase operates directly on the GameState array.

from .invest import InvestPhase, get_phase_handler as get_invest_handler
from .wrapup import WrapUpPhase, get_phase_handler as get_wrapup_handler
from .acquisition import AcquisitionPhase, get_phase_handler as get_acquisition_handler
from .closing import ClosingPhase
from .income import IncomePhase
from .dividends import DividendsPhase, get_phase_handler as get_dividends_handler
from .endcard import EndCardPhase, get_phase_handler as get_endcard_handler
from .issue import IssuePhase, get_phase_handler as get_issue_handler
from .ipo import IPOPhase, get_phase_handler as get_ipo_handler

__all__ = [
    'InvestPhase',
    'WrapUpPhase',
    'AcquisitionPhase',
    'ClosingPhase',
    'IncomePhase',
    'DividendsPhase',
    'EndCardPhase',
    'get_invest_handler',
    'get_wrapup_handler',
    'get_acquisition_handler',
    'get_dividends_handler',
    'get_endcard_handler',
    'IssuePhase',
    'get_issue_handler',
    'IPOPhase',
    'get_ipo_handler',
]
