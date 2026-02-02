# cython: language_level=3
"""Phase handlers package declarations."""
from phases.invest cimport apply_invest_action
from phases.bid cimport apply_bid_action
from phases.wrap_up cimport apply_wrap_up
from phases.acquisition cimport apply_acquisition_action
from phases.closing cimport apply_closing_action, apply_closing_auto
from phases.income cimport apply_income
from phases.temp_end_turn cimport apply_temp_end_turn
