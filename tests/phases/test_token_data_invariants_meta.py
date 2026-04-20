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

from core.data import GamePhases, GameConstants
from core.state import GameState
from entities.turn import TURN
from tests.phases import conftest as phase_conftest
from tests.phases.conftest import assert_token_data_invariants


NUM_CORPS = int(GameConstants.NUM_CORPS)
NUM_COMPANIES = int(GameConstants.NUM_COMPANIES)

# Token positions for a 3p buffer.
MARKET_SLOT_PRICES_TOK = 0
COMPANY_BASE_TOK       = 1       # companies 0..35 at positions 1..36
MARKET_AVAIL_TOK       = 37
LOC_REMOVED_TOK        = 38
LOC_AUCTION_TOK        = 39
LOC_REVEALED_TOK       = 40
LOC_CORP_ACQ_TOK       = 41
COMPANY_ADJ_INCOME_TOK = 42
FI_TOK                 = 43
ACTIVE_PLAYER_TOK      = 44
ACTIVE_CORP_TOK        = 45
ACTIVE_COMPANY_TOK     = 46
PHASE_TOK              = 47
NUM_PLAYERS_TOK        = 48
GAME_PROGRESS_TOK      = 49
INVEST_TOK             = 50
AUCTION_TOK            = 51
DIVIDEND_TOK           = 52
ISSUE_TOK              = 53
PAR_TOK                = 54
ACQ_OFFER_TOK          = 55
ACQ_PRICE_INFO_TOK     = 56
CORP_BASE_TOK          = 57
PLAYER_BASE_TOK        = 65


def _invest_state():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    assert TURN.get_phase(state) == GamePhases.PHASE_INVEST
    assert TURN.get_active_player(state) == 0
    assert TURN.get_active_corp(state) == -1
    assert TURN.get_active_company(state) == -1
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

# ---- Static data: market slot prices -------------------------------------

def corrupt_market_slot_price(buf):
    buf[MARKET_SLOT_PRICES_TOK, 5] = 0.0           # real value ≈ 9/75

def corrupt_market_slot_prices_tail(buf):
    buf[MARKET_SLOT_PRICES_TOK, 27] = 0.5          # tail must stay 0

# ---- Company token (static game-setup data) ------------------------------

def corrupt_company_id_onehot(buf):
    buf[COMPANY_BASE_TOK + 0, 0] = 0.0             # drop id bit

def corrupt_company_face_value(buf):
    buf[COMPANY_BASE_TOK + 5, 37] = 0.0            # FACE_VALUE offset

def corrupt_company_low_high_diff(buf):
    buf[COMPANY_BASE_TOK + 5, 39] = 0.0            # LOW_HIGH_DIFF offset

def corrupt_company_base_income(buf):
    buf[COMPANY_BASE_TOK + 5, 40] = 0.0            # BASE_INCOME offset

def corrupt_company_stars(buf):
    buf[COMPANY_BASE_TOK + 5, 41] = 0.0            # STARS offset

# ---- Market availability -------------------------------------------------

def corrupt_market_avail_boundary(buf):
    buf[MARKET_AVAIL_TOK, 0] = 0.0                 # slot 0 ($0) must be 1.0

def corrupt_market_avail_tail(buf):
    buf[MARKET_AVAIL_TOK, 27] = 1.0                # tail must stay 0

# ---- Company-location bitmaps (REMOVED, AUCTION, REVEALED, CORP_ACQ) -----

def corrupt_loc_removed_bit(buf):
    buf[LOC_REMOVED_TOK, 0] = 1.0                  # company 0 is not at REMOVED

def corrupt_loc_auction_bit(buf):
    buf[LOC_AUCTION_TOK, 5] = 0.0                  # company 5 IS at AUCTION

def corrupt_loc_revealed_bit(buf):
    buf[LOC_REVEALED_TOK, 0] = 1.0                 # nothing at REVEALED initially

def corrupt_loc_corp_acq_tail(buf):
    buf[LOC_CORP_ACQ_TOK, 36] = 1.0                # tail must stay 0

# ---- Company adjusted income --------------------------------------------

def corrupt_company_adj_income(buf):
    buf[COMPANY_ADJ_INCOME_TOK, 5] = 9.99          # per-company value mismatch

def corrupt_company_adj_income_tail(buf):
    buf[COMPANY_ADJ_INCOME_TOK, 36] = 1.0

# ---- FI ------------------------------------------------------------------

def corrupt_fi_cash(buf):
    buf[FI_TOK, 0] = 0.0                           # FI cash = 4 initially

def corrupt_fi_income(buf):
    buf[FI_TOK, 1] = 0.0                           # FI income = 5 initially

def corrupt_fi_owned(buf):
    buf[FI_TOK, 2] = 1.0                           # FI owns nothing initially

def corrupt_fi_tail(buf):
    buf[FI_TOK, 2 + NUM_COMPANIES] = 1.0           # tail past bitmap

# ---- Active-entity one-hots ---------------------------------------------

def corrupt_active_player_onehot(buf):
    buf[ACTIVE_PLAYER_TOK, 0] = 0.0                # active_player=0 → bit 0 set

def corrupt_active_player_tail(buf):
    buf[ACTIVE_PLAYER_TOK, 5] = 1.0                # padding slot

def corrupt_active_corp_unset(buf):
    buf[ACTIVE_CORP_TOK, 0] = 1.0                  # active_corp=-1 → all zero

def corrupt_active_company_unset(buf):
    buf[ACTIVE_COMPANY_TOK, 0] = 1.0               # active_company=-1 → all zero

# ---- Phase / num_players / game_progress --------------------------------

def corrupt_phase_onehot(buf):
    buf[PHASE_TOK, 0] = 0.0                        # INVEST dp=0 bit set

def corrupt_phase_tail(buf):
    buf[PHASE_TOK, 11] = 1.0                       # tail past decision phases

def corrupt_num_players(buf):
    buf[NUM_PLAYERS_TOK, 0] = 0.0                  # 3p → slot 0 set

def corrupt_num_players_tail(buf):
    buf[NUM_PLAYERS_TOK, 3] = 1.0

def corrupt_coo_onehot(buf):
    buf[GAME_PROGRESS_TOK, 0] = 0.0                # CoO level 1 → slot 0 set

def corrupt_end_card(buf):
    buf[GAME_PROGRESS_TOK, 7] = 1.0                # end_card unflipped initially

def corrupt_cards_remaining(buf):
    buf[GAME_PROGRESS_TOK, 8] = 0.0                # cards_remaining = 17 != 0

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

def corrupt_acq_offer_out_of_phase(buf):
    buf[ACQ_OFFER_TOK, 0] = 1.0

def corrupt_acq_price_info_out_of_phase(buf):
    buf[ACQ_PRICE_INFO_TOK, 0] = 1.0

# ---- Corp token (inactive) ----------------------------------------------

def corrupt_corp_id_onehot(buf):
    buf[CORP_BASE_TOK + 0, 0] = 0.0                # corp 0 id bit

def corrupt_corp_inactive_active_flag(buf):
    buf[CORP_BASE_TOK + 0, 8] = 1.0                # OFF_ACTIVE; corp 0 inactive

def corrupt_corp_inactive_price_idx(buf):
    buf[CORP_BASE_TOK + 0, 14] = 1.0               # inactive → price_idx zeros

# ---- Player token --------------------------------------------------------

def corrupt_player_id_onehot(buf):
    buf[PLAYER_BASE_TOK + 0, 0] = 0.0              # player 0 id bit

def corrupt_player_cash(buf):
    buf[PLAYER_BASE_TOK + 0, 11] = 0.0             # OFF_CASH; player 0 cash=30

def corrupt_player_presidency(buf):
    buf[PLAYER_BASE_TOK + 0, 40] = 1.0             # OFF_PRESIDENCIES: none held


# Each case: (mutation_fn, expected-error-substring).
# The match argument to pytest.raises is re.search on the assertion text;
# substrings here are chosen to uniquely identify the block being checked.
CASES = [
    (corrupt_market_slot_price,            "price slot 5"),
    (corrupt_market_slot_prices_tail,      "tail beyond price slots"),
    (corrupt_company_id_onehot,            "company_id one-hot"),
    (corrupt_company_face_value,           "face_value"),
    (corrupt_company_low_high_diff,        "low_high_diff"),
    (corrupt_company_base_income,          "base_income"),
    (corrupt_company_stars,                ": stars"),
    (corrupt_market_avail_boundary,        r"slot 0 \(\$0\) must always be available"),
    (corrupt_market_avail_tail,            "tail beyond availability flags"),
    (corrupt_loc_removed_bit,              r"CompanyLocation\[REMOVED\]"),
    (corrupt_loc_auction_bit,              r"CompanyLocation\[AUCTION\]"),
    (corrupt_loc_revealed_bit,             r"CompanyLocation\[REVEALED\]"),
    (corrupt_loc_corp_acq_tail,            "tail beyond company bitmap"),
    (corrupt_company_adj_income,           "adjusted_income"),
    (corrupt_company_adj_income_tail,      "tail beyond per-company income"),
    (corrupt_fi_cash,                      r"FI token: cash"),
    (corrupt_fi_income,                    r"FI token: income"),
    (corrupt_fi_owned,                     r"FI token: owned"),
    (corrupt_fi_tail,                      "tail beyond owned bitmap"),
    (corrupt_active_player_onehot,         "ActivePlayer token"),
    (corrupt_active_player_tail,           "ActivePlayer token: tail"),
    (corrupt_active_corp_unset,            "ActiveCorp token: must be all-zero"),
    (corrupt_active_company_unset,         "ActiveCompany token: must be all-zero"),
    (corrupt_phase_onehot,                 "Phase token"),
    (corrupt_phase_tail,                   "tail beyond phase one-hot"),
    (corrupt_num_players,                  "num_players one-hot"),
    (corrupt_num_players_tail,             "tail beyond num_players one-hot"),
    (corrupt_coo_onehot,                   "CoO one-hot"),
    (corrupt_end_card,                     "end_card flag"),
    (corrupt_cards_remaining,              "cards_remaining"),
    (corrupt_invest_passes,                "consecutive_passes"),
    (corrupt_invest_tail,                  r"Invest token: tail"),
    (corrupt_auction_out_of_phase,         r"Auction token.*all-zero outside PHASE_BID"),
    (corrupt_dividend_out_of_phase,        r"Dividend token.*all-zero outside PHASE_DIVIDENDS"),
    (corrupt_issue_out_of_phase,           r"Issue token.*all-zero outside PHASE_ISSUE_SHARES"),
    (corrupt_par_out_of_phase,             r"Par/IPO token.*all-zero outside PHASE_IPO"),
    (corrupt_acq_offer_out_of_phase,       r"Acq-offer token.*all-zero outside PHASE_ACQ_OFFER"),
    (corrupt_acq_price_info_out_of_phase,  r"AcqPriceInfo token"),
    (corrupt_corp_id_onehot,               "corp_id one-hot"),
    (corrupt_corp_inactive_active_flag,    ": active flag"),
    (corrupt_corp_inactive_price_idx,      "inactive corp price_idx"),
    (corrupt_player_id_onehot,             "player_id one-hot"),
    (corrupt_player_cash,                  r"player token p=0.*: cash"),
    (corrupt_player_presidency,            "presidency"),
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
