"""Meta-tests for ``assert_token_data_invariants``.

These tests inject known corruption into the post-extraction token buffer
and verify the invariant helper raises. They protect the helper itself
from silent decay — e.g., someone adds a new token slot without a
matching assertion, or renames an offset and the check still passes
vacuously because the old slot stays zero. Each case targets one
assertion block in ``conftest.assert_token_data_invariants``.

Strategy: monkey-patch ``conftest.get_token_data`` with a wrapper that
applies ``mutation(buf)`` after the real extractor fills the buffer. All
cases run against a fresh 3p INVEST state (seed=42), whose layout is:

    active_player=0, active_corp=-1, active_company=-1,
    phase=INVEST (dp=0), coo_level=1, end_card=False, cards_remaining=17,
    LOC_AUCTION={1,2,5}, LOC_REVEALED={}, LOC_CORP_ACQ={}, LOC_REMOVED={},
    all 8 corps inactive.

Token positions below match the conftest layout banner.
"""
import pytest

from core.actions import (
    ACTION_ACQ_SELECT_COMPANY_PY as ACTION_ACQ_SELECT_COMPANY,
    ACTION_ACQ_SELECT_CORP_PY as ACTION_ACQ_SELECT_CORP,
    ACTION_PASS_PY as ACTION_PASS,
)
from core.data import GamePhases, GameConstants
from core.state import GameState
from entities.company import COMPANIES
from entities.corp import CORPS
from entities.turn import TURN
from phases.acq_select_corp import setup_acquisition_phase_py
from tests.phases import conftest as phase_conftest
from tests.phases.conftest import assert_token_data_invariants


NUM_CORPS = int(GameConstants.NUM_CORPS)
NUM_COMPANIES = int(GameConstants.NUM_COMPANIES)

# Token positions for a 3p buffer.
MARKET_INFO_TOK         = 0
COMPANY_BASE_TOK        = 1       # companies 0..35 at positions 1..36
FI_TOK                  = 37
GLOBAL_INFO_TOK         = 38
INVEST_TOK              = 39
AUCTION_TOK             = 40
DIVIDEND_TOK            = 41
ISSUE_TOK               = 42
PAR_TOK                 = 43
ACQ_SELECT_COMPANY_TOK  = 44
ACQ_OFFER_TOK           = 45
ACQ_PRICE_INFO_TOK      = 46
CORP_BASE_TOK           = 47
PLAYER_BASE_TOK         = 55


def _invest_state():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    assert TURN.get_phase(state) == GamePhases.PHASE_INVEST
    assert TURN.get_active_player(state) == 0
    assert TURN.get_active_corp(state) == -1
    assert TURN.get_active_company(state) == -1
    return state


def _active_corp_invest_state():
    state = _invest_state()
    company_id = phase_conftest.float_corp_for_test(
        state, corp_id=0, player_id=0, par_index=10,
    )
    assert TURN.get_phase(state) == GamePhases.PHASE_INVEST
    assert CORPS[0].is_active(state)
    assert CORPS[0].count_companies(state, include_acquisition=True) >= 1
    assert company_id >= 0
    return state


def _select_price_state():
    state = _invest_state()
    private_co = phase_conftest.draw_to_player(state, 0)
    phase_conftest.float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
    CORPS[0].set_cash(state, COMPANIES[private_co].get_low_price() + 2)

    setup_acquisition_phase_py(state)
    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_CORP)

    target_president = CORPS[0].get_president_id(state)
    for _ in range(16):
        if TURN.get_active_player(state) == target_president:
            break
        assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_CORP)
        pass_id = phase_conftest.find_legal_action(state, action_type=ACTION_PASS)
        phase_conftest.apply_and_verify(state, pass_id)
    else:
        assert False, "_select_price_state: never reached corp 0 president"

    aid = phase_conftest.find_legal_action(
        state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=0,
    )
    phase_conftest.apply_and_verify(state, aid)
    if TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_COMPANY):
        aid = phase_conftest.find_legal_action(
            state, action_type=ACTION_ACQ_SELECT_COMPANY, company_id=private_co,
        )
        phase_conftest.apply_and_verify(state, aid)

    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_PRICE)
    assert TURN.get_active_corp(state) == 0
    assert TURN.get_active_company(state) == private_co
    return state


def _patch(monkeypatch, mutation):
    """Patch conftest.get_token_data to apply ``mutation(buf)`` after fill."""
    original = phase_conftest.get_token_data

    def patched(state, buf):
        original(state, buf)
        mutation(buf)

    monkeypatch.setattr(phase_conftest, "get_token_data", patched)


# =============================================================================
# MUTATION FUNCTIONS
# =============================================================================
# Each function mutates a buffer slice that, for the 3p seed=42 INVEST
# fixture, produces a value the helper's assertions reject. Named rather
# than lambdas so pytest IDs come out readable.

# MarketInfo token layout: 27 slot prices (0..26) + 27 availability (27..53).
_MI_AVAIL_BASE = 27

# ---- MarketInfo: slot prices ---------------------------------------------

def corrupt_market_slot_price(buf):
    buf[MARKET_INFO_TOK, 5] = 0.0                  # real value ≈ 9/75

# ---- MarketInfo: availability --------------------------------------------

def corrupt_market_avail_boundary(buf):
    buf[MARKET_INFO_TOK, _MI_AVAIL_BASE + 0] = 0.0  # slot 0 ($0) must be 1.0

def corrupt_market_info_tail(buf):
    buf[MARKET_INFO_TOK, _MI_AVAIL_BASE + 27] = 1.0  # tail past 54 slots

# ---- Company token: static data ------------------------------------------

def corrupt_company_face_value(buf):
    buf[COMPANY_BASE_TOK + 5, 2] = 0.0             # FACE_VALUE offset

def corrupt_company_low_high_diff(buf):
    buf[COMPANY_BASE_TOK + 5, 4] = 0.0             # LOW_HIGH_DIFF offset

def corrupt_company_base_income(buf):
    buf[COMPANY_BASE_TOK + 5, 5] = 0.0             # BASE_INCOME offset

def corrupt_company_stars(buf):
    buf[COMPANY_BASE_TOK + 5, 6] = 0.0             # STARS offset

# ---- Company token: dynamic (adj_income / is_selected / at_* / owner_*) --

def corrupt_company_adj_income(buf):
    buf[COMPANY_BASE_TOK + 5, 7] = 9.99            # ADJ_INCOME mismatch

def corrupt_company_is_selected(buf):
    buf[COMPANY_BASE_TOK + 0, 0] = 1.0             # active_company=-1 so must be 0

def corrupt_company_at_auction_wrong(buf):
    buf[COMPANY_BASE_TOK + 5, 9] = 0.0             # company 5 IS at AUCTION

def corrupt_company_at_removed_false(buf):
    buf[COMPANY_BASE_TOK + 0, 8] = 1.0             # company 0 is NOT at REMOVED

def corrupt_company_at_revealed_false(buf):
    buf[COMPANY_BASE_TOK + 0, 10] = 1.0            # nothing at REVEALED initially

def corrupt_company_owner_corp_false(buf):
    buf[COMPANY_BASE_TOK + 0, 12] = 1.0            # no corp owns anything initially

def corrupt_company_owner_player_false(buf):
    buf[COMPANY_BASE_TOK + 0, 20] = 1.0            # no player owns anything initially

def corrupt_company_owner_fi_false(buf):
    buf[COMPANY_BASE_TOK + 0, 25] = 1.0            # FI owns nothing initially

def corrupt_company_tail(buf):
    buf[COMPANY_BASE_TOK + 0, 26] = 1.0            # padding past TW_COMPANY must stay 0

# ---- FI ------------------------------------------------------------------

def corrupt_fi_cash(buf):
    buf[FI_TOK, 0] = 0.0                           # FI cash = 4 initially

def corrupt_fi_income(buf):
    buf[FI_TOK, 1] = 0.0                           # FI income = 5 initially

def corrupt_fi_owned(buf):
    buf[FI_TOK, 2] = 1.0                           # FI owns nothing initially

def corrupt_fi_tail(buf):
    buf[FI_TOK, 2 + NUM_COMPANIES] = 1.0           # tail past bitmap

# ---- GlobalInfo (phase + CoO + end_card + cards_remaining + num_players) -

def corrupt_global_phase_onehot(buf):
    buf[GLOBAL_INFO_TOK, 0] = 0.0                  # INVEST dp=0 bit set

def corrupt_global_coo_onehot(buf):
    buf[GLOBAL_INFO_TOK, 11] = 0.0                 # CoO level 1 → slot 0 of COO region

def corrupt_global_end_card(buf):
    buf[GLOBAL_INFO_TOK, 18] = 1.0                 # end_card unflipped initially

def corrupt_global_cards_remaining(buf):
    buf[GLOBAL_INFO_TOK, 19] = 0.0                 # cards_remaining = 17 != 0

def corrupt_global_num_players(buf):
    buf[GLOBAL_INFO_TOK, 20] = 0.0                 # 3p → slot 20 set

def corrupt_global_info_tail(buf):
    buf[GLOBAL_INFO_TOK, 23] = 1.0                 # tail past 23 slots

# ---- Invest token (in-phase) --------------------------------------------

def corrupt_invest_passes(buf):
    buf[INVEST_TOK, 0] = 1.0                       # consecutive_passes=0

def corrupt_invest_tail(buf):
    buf[INVEST_TOK, 17] = 1.0                      # tail past 17 slots

# ---- Phase-specific tokens that must be zero outside their phase --------

def corrupt_auction_out_of_phase(buf):
    buf[AUCTION_TOK, 0] = 1.0

def corrupt_dividend_out_of_phase(buf):
    buf[DIVIDEND_TOK, 0] = 1.0

def corrupt_issue_out_of_phase(buf):
    buf[ISSUE_TOK, 0] = 1.0

def corrupt_par_out_of_phase(buf):
    buf[PAR_TOK, 0] = 1.0

def corrupt_acq_select_company_out_of_phase(buf):
    buf[ACQ_SELECT_COMPANY_TOK, 0] = 1.0

def corrupt_acq_offer_out_of_phase(buf):
    buf[ACQ_OFFER_TOK, 0] = 1.0

def corrupt_acq_price_info_out_of_phase(buf):
    buf[ACQ_PRICE_INFO_TOK, 0] = 1.0

# ---- Corp token (inactive) ----------------------------------------------

def corrupt_corp_inactive_active_flag(buf):
    buf[CORP_BASE_TOK + 0, 1] = 1.0                # OFF_ACTIVE; corp 0 inactive

def corrupt_corp_inactive_price_idx(buf):
    buf[CORP_BASE_TOK + 0, 7] = 1.0                # inactive → price_idx zeros

def corrupt_corp_is_selected(buf):
    buf[CORP_BASE_TOK + 0, 0] = 1.0                # active_corp=-1 so must be 0

# The corp token pins TOKEN_DIM (TW_CORP=85 == TOKEN_DIM=85), so there is
# no zero-padded tail past TW_CORP to corrupt. The conftest tail check
# runs vacuously on the empty slice.

# ---- Player token --------------------------------------------------------

def corrupt_player_cash(buf):
    buf[PLAYER_BASE_TOK + 0, 7] = 0.0              # OFF_CASH; player 0 cash=30

def corrupt_player_presidency(buf):
    buf[PLAYER_BASE_TOK + 0, 36] = 1.0             # OFF_PRESIDENCIES: none held

def corrupt_player_is_selected_wrong(buf):
    buf[PLAYER_BASE_TOK + 1, 0] = 1.0              # active_player=0 so player 1 must be 0

def corrupt_player_tail(buf):
    buf[PLAYER_BASE_TOK + 0, 80] = 1.0             # padding past TW_PLAYER must stay 0


def corrupt_inactive_corp_companies(buf):
    buf[CORP_BASE_TOK + 0, 49] = 1.0               # inactive corp company bitmap must be 0


def corrupt_active_corp_pending_move(buf):
    buf[CORP_BASE_TOK + 0, 35] = 0.0               # active corp pending move


def corrupt_active_corp_raw_revenue(buf):
    buf[CORP_BASE_TOK + 0, 40] = 0.0               # active corp raw revenue


def corrupt_active_corp_synergy(buf):
    buf[CORP_BASE_TOK + 0, 41] = 1.0               # active corp synergy income


def corrupt_active_corp_coo_cost(buf):
    buf[CORP_BASE_TOK + 0, 42] = 1.0               # active corp CoO cost


def corrupt_active_corp_ability(buf):
    buf[CORP_BASE_TOK + 0, 43] = 1.0               # active corp ability income


def corrupt_acq_price_info_max_offset(buf):
    buf[ACQ_PRICE_INFO_TOK, 0] = 0.0


def corrupt_acq_price_info_fi_flag(buf):
    buf[ACQ_PRICE_INFO_TOK, 1] = 1.0


def corrupt_acq_price_info_total_synergies(buf):
    buf[ACQ_PRICE_INFO_TOK, 2] = 0.0


# Each case: (mutation_fn, expected-error-substring).
# The match argument to pytest.raises is re.search on the assertion text;
# substrings here are chosen to uniquely identify the block being checked.
CASES = [
    (corrupt_market_slot_price,               "price slot 5"),
    (corrupt_market_avail_boundary,           r"slot 0 \(\$0\) must always be available"),
    (corrupt_market_info_tail,                "tail beyond availability flags"),
    (corrupt_company_face_value,              "face_value"),
    (corrupt_company_low_high_diff,           "low_high_diff"),
    (corrupt_company_base_income,             "base_income"),
    (corrupt_company_stars,                   ": stars"),
    (corrupt_company_adj_income,              "adjusted_income"),
    (corrupt_company_is_selected,             "is_selected"),
    (corrupt_company_at_auction_wrong,        "at_auction"),
    (corrupt_company_at_removed_false,        "at_removed"),
    (corrupt_company_at_revealed_false,       "at_revealed"),
    (corrupt_company_owner_corp_false,        "owner_corp"),
    (corrupt_company_owner_player_false,      "owner_player"),
    (corrupt_company_owner_fi_false,          "owner_fi"),
    (corrupt_company_tail,                    "tail beyond TW_COMPANY features"),
    (corrupt_fi_cash,                         r"FI token: cash"),
    (corrupt_fi_income,                       r"FI token: income"),
    (corrupt_fi_owned,                        r"FI token: owned"),
    (corrupt_fi_tail,                         "tail beyond owned bitmap"),
    (corrupt_global_phase_onehot,             "Phase token"),
    (corrupt_global_coo_onehot,               "CoO one-hot"),
    (corrupt_global_end_card,                 "end_card flag"),
    (corrupt_global_cards_remaining,          "cards_remaining"),
    (corrupt_global_num_players,              "num_players one-hot"),
    (corrupt_global_info_tail,                "tail beyond global_info features"),
    (corrupt_invest_passes,                   "consecutive_passes"),
    (corrupt_invest_tail,                     r"Invest token: tail"),
    (corrupt_auction_out_of_phase,            r"Auction token.*all-zero outside PHASE_BID"),
    (corrupt_dividend_out_of_phase,           r"Dividend token.*all-zero outside PHASE_DIVIDENDS"),
    (corrupt_issue_out_of_phase,              r"Issue token.*all-zero outside PHASE_ISSUE_SHARES"),
    (corrupt_par_out_of_phase,                r"Par/IPO token.*all-zero outside PHASE_IPO"),
    (corrupt_acq_select_company_out_of_phase, r"AcqSelectCompany token.*all-zero outside PHASE_ACQ_SELECT_COMPANY"),
    (corrupt_acq_offer_out_of_phase,          r"Acq-offer token.*all-zero outside PHASE_ACQ_OFFER"),
    (corrupt_acq_price_info_out_of_phase,     r"AcqPriceInfo token"),
    (corrupt_corp_inactive_active_flag,       ": active flag"),
    (corrupt_corp_inactive_price_idx,         "inactive corp price_idx"),
    (corrupt_corp_is_selected,                "is_selected"),
    (corrupt_player_cash,                     r"player token p=0.*: cash"),
    (corrupt_player_presidency,               "presidency"),
    (corrupt_player_is_selected_wrong,        "is_selected"),
    (corrupt_player_tail,                     "tail beyond TW_PLAYER features"),
    (corrupt_inactive_corp_companies,         "inactive corp owned_company region must be zero"),
]


ACTIVE_CORP_CASES = [
    (corrupt_active_corp_pending_move,     "pending_price_move"),
    (corrupt_active_corp_raw_revenue,      "raw_revenue"),
    (corrupt_active_corp_synergy,          "synergy_income"),
    (corrupt_active_corp_coo_cost,         "coo_cost"),
    (corrupt_active_corp_ability,          "ability_income"),
]


ACQ_PRICE_INFO_CASES = [
    (corrupt_acq_price_info_max_offset,      "max_offset"),
    (corrupt_acq_price_info_fi_flag,         "fi_flag"),
    (corrupt_acq_price_info_total_synergies, "total_synergies"),
]


@pytest.mark.parametrize(
    "mutation,msg_pattern",
    CASES,
    ids=[fn.__name__ for fn, _ in CASES],
)
def test_invariants_catch_corruption(monkeypatch, mutation, msg_pattern):
    """Each mutation must make ``assert_token_data_invariants`` raise."""
    state = _invest_state()
    _patch(monkeypatch, mutation)
    with pytest.raises(AssertionError, match=msg_pattern):
        assert_token_data_invariants(state)


def test_baseline_passes_without_corruption():
    """Sanity: the fixture itself is a clean state — no mutation, no error.

    Guards against the meta-test becoming vacuous if the fixture drifts
    into something the invariants already reject.
    """
    state = _invest_state()
    assert_token_data_invariants(state)


@pytest.mark.parametrize(
    "mutation,msg_pattern",
    ACTIVE_CORP_CASES,
    ids=[fn.__name__ for fn, _ in ACTIVE_CORP_CASES],
)
def test_invariants_catch_active_corp_corruption(monkeypatch, mutation, msg_pattern):
    state = _active_corp_invest_state()
    _patch(monkeypatch, mutation)
    with pytest.raises(AssertionError, match=msg_pattern):
        assert_token_data_invariants(state)


@pytest.mark.parametrize(
    "mutation,msg_pattern",
    ACQ_PRICE_INFO_CASES,
    ids=[fn.__name__ for fn, _ in ACQ_PRICE_INFO_CASES],
)
def test_invariants_catch_acq_price_info_corruption(monkeypatch, mutation, msg_pattern):
    state = _select_price_state()
    _patch(monkeypatch, mutation)
    with pytest.raises(AssertionError, match=msg_pattern):
        assert_token_data_invariants(state)
