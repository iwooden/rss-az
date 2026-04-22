"""
Static game data for Rolling Stock Stars.

Pure data module: company / corp / market arrays, normalization constants used
by token extraction, and the enums shared across the engine. There are no
accessor functions here — entity handles own all reads/writes against
GameState, and computational helpers (synergy aggregation, cost of ownership,
par-price logic, etc.) live in the modules that use them. Other Cython modules
cimport the underlying arrays directly when they need static game data.

Companies are indexed 0-35, sorted by face_value (matching Python's
all_companies_sorted). Corps are indexed 0-7 in fixed order:
JS, S, OS, SM, PR, DA, VM, SI.
"""

cimport cython
from libc.stdint cimport uint8_t, int8_t

# =============================================================================
# NORMALIZATION CONSTANTS
# =============================================================================
#
# These divisors are applied during token extraction (get_token_data) to bring
# raw int16 game-state values into roughly the [-1, 1] range expected by the
# transformer's input projections. They are NOT applied during state storage —
# the state array always holds raw integers.
#
# The cdef constants themselves live in data.pxd as ``cdef extern from *``
# C #defines so every cimporting module gets them as compile-time literals
# that the C compiler can fold directly into multiplies/divisions. Only
# the Python-visible mirrors live here, for tests and notebooks that want
# to inspect the divisors without dropping into Cython.

PY_CASH_DIVISOR = CASH_DIVISOR
PY_NET_WORTH_DIVISOR = NET_WORTH_DIVISOR
PY_COMPANY_PRICE_DIVISOR = COMPANY_PRICE_DIVISOR
PY_SHARE_PRICE_DIVISOR = SHARE_PRICE_DIVISOR
PY_COMPANY_INCOME_DIVISOR = COMPANY_INCOME_DIVISOR
PY_COMPANY_SYNERGY_DIVISOR = COMPANY_SYNERGY_DIVISOR
PY_ENTITY_INCOME_DIVISOR = ENTITY_INCOME_DIVISOR
PY_SHARE_DIVISOR = SHARE_DIVISOR
PY_COMPANY_STAR_DIVISOR = COMPANY_STAR_DIVISOR
PY_CORP_STAR_DIVISOR = CORP_STAR_DIVISOR
PY_MAX_ROUNDTRIPS = MAX_ROUNDTRIPS
PY_IMPACT_DIVISOR = IMPACT_DIVISOR

# =============================================================================
# ENGINE → DECISION PHASE MAPPING
# =============================================================================
#
# The engine's 15 ``GamePhases`` fold down to 11 ``DecisionPhase`` slots for
# the transformer. Automated/terminal engine phases (``WRAP_UP``, ``INCOME``,
# ``END_CARD``, ``GAME_OVER``) map to -1 and are fast-forwarded by the
# driver. ``ENGINE_TO_DECISION_PHASE`` is the canonical lookup both for the
# nogil Cython path in ``core/actions.pyx::get_decision_phase`` (which
# cimports the C array) and for Python callers (which import the list
# mirror below).

cdef void _init_engine_to_decision_phase() noexcept nogil:
    cdef int i
    for i in range(15):
        ENGINE_TO_DECISION_PHASE[i] = -1
    ENGINE_TO_DECISION_PHASE[<int>GamePhases.PHASE_INVEST] = <int>DecisionPhase.DPHASE_INVEST
    ENGINE_TO_DECISION_PHASE[<int>GamePhases.PHASE_BID] = <int>DecisionPhase.DPHASE_BID
    ENGINE_TO_DECISION_PHASE[<int>GamePhases.PHASE_ACQ_SELECT_CORP] = <int>DecisionPhase.DPHASE_ACQ_SELECT_CORP
    ENGINE_TO_DECISION_PHASE[<int>GamePhases.PHASE_ACQ_SELECT_COMPANY] = <int>DecisionPhase.DPHASE_ACQ_SELECT_COMPANY
    ENGINE_TO_DECISION_PHASE[<int>GamePhases.PHASE_ACQ_SELECT_PRICE] = <int>DecisionPhase.DPHASE_ACQ_SELECT_PRICE
    ENGINE_TO_DECISION_PHASE[<int>GamePhases.PHASE_ACQ_OFFER] = <int>DecisionPhase.DPHASE_ACQ_OFFER
    ENGINE_TO_DECISION_PHASE[<int>GamePhases.PHASE_CLOSING] = <int>DecisionPhase.DPHASE_CLOSING
    ENGINE_TO_DECISION_PHASE[<int>GamePhases.PHASE_DIVIDENDS] = <int>DecisionPhase.DPHASE_DIVIDENDS
    ENGINE_TO_DECISION_PHASE[<int>GamePhases.PHASE_ISSUE_SHARES] = <int>DecisionPhase.DPHASE_ISSUE
    ENGINE_TO_DECISION_PHASE[<int>GamePhases.PHASE_IPO] = <int>DecisionPhase.DPHASE_IPO
    ENGINE_TO_DECISION_PHASE[<int>GamePhases.PHASE_PAR] = <int>DecisionPhase.DPHASE_PAR


_init_engine_to_decision_phase()


cdef list _engine_to_decision_phase_pylist():
    """Build a plain Python list mirror of the C lookup table."""
    cdef int i
    cdef list out = []
    for i in range(15):
        out.append(ENGINE_TO_DECISION_PHASE[i])
    return out


# Python-level mirror of the Cython lookup table. Same ``globals()`` trick
# as ``PHASE_ACTION_SIZES`` — a bare ``ENGINE_TO_DECISION_PHASE = [...]``
# assignment would be shadowed by the cimported ``cdef int[15]`` array.
globals()["ENGINE_TO_DECISION_PHASE"] = _engine_to_decision_phase_pylist()


# =============================================================================
# POLICY ACTION-SPACE SIZES (Python-accessible)
# =============================================================================
#
# The per-phase sizes live in the ``ActionSize`` ``cpdef enum`` in data.pxd.
# That makes them cimportable as compile-time constants from Cython code,
# but it does **not** put them in this module's Python namespace — only the
# ``ActionSize`` class itself is visible, and members with the same int
# value (e.g. ``ACTION_SIZE_ISSUE`` / ``ACTION_SIZE_ACQ_OFFER`` both == 2)
# alias in ``repr`` which is ugly. So we inject plain-int mirrors into
# ``globals()`` at import so that Python consumers
# (``nn/transformer.py``, trainer, replay) can do a straightforward
# ``from core.data import PHASE_ACTION_SIZES, MAX_ACTION_SIZE``. Cython
# callers still ``cimport`` the enum members directly.
#
# Using ``globals()[...] = ...`` rather than a bare assignment because
# ``MAX_ACTION_SIZE = ...`` at module scope shadows the cpdef enum member
# in the Cython compile pass and silently drops the assignment.
_py_phase_action_sizes = [
    int(ActionSize.ACTION_SIZE_INVEST),
    int(ActionSize.ACTION_SIZE_BID),
    int(ActionSize.ACTION_SIZE_ACQ_SELECT_CORP),
    int(ActionSize.ACTION_SIZE_ACQ_OFFER),
    int(ActionSize.ACTION_SIZE_CLOSING),
    int(ActionSize.ACTION_SIZE_DIVIDENDS),
    int(ActionSize.ACTION_SIZE_ISSUE),
    int(ActionSize.ACTION_SIZE_IPO),
    int(ActionSize.ACTION_SIZE_PAR),
    int(ActionSize.ACTION_SIZE_ACQ_SELECT_COMPANY),
    int(ActionSize.ACTION_SIZE_ACQ_SELECT_PRICE),
]
# Catch the "added a DecisionPhase enum entry but forgot to extend the
# Python mirror" failure mode at import time — downstream consumers index
# this list by phase id, so a short list hands back stale sizes silently.
assert len(_py_phase_action_sizes) == int(GameConstants.NUM_DECISION_PHASES), (
    f"_py_phase_action_sizes has {len(_py_phase_action_sizes)} entries but "
    f"NUM_DECISION_PHASES = {int(GameConstants.NUM_DECISION_PHASES)}"
)
assert max(_py_phase_action_sizes) == int(ActionSize.MAX_ACTION_SIZE), (
    f"max(_py_phase_action_sizes) = {max(_py_phase_action_sizes)} but "
    f"MAX_ACTION_SIZE = {int(ActionSize.MAX_ACTION_SIZE)}"
)
globals()["PHASE_ACTION_SIZES"] = _py_phase_action_sizes
globals()["MAX_ACTION_SIZE"] = int(ActionSize.MAX_ACTION_SIZE)
globals()["AUCTION_CAP"] = int(GameConstants.AUCTION_CAP)

# =============================================================================
# COMPANY DATA
# =============================================================================

# Company IDs as strings (for debugging/display only)
COMPANY_NAMES = [
    "BME", "BSE", "KME", "AKE", "BPM", "MHE",  # Reds (stars=1)
    "WT", "BY", "BD", "HE", "OL", "SX", "MS", "PR",  # Oranges (stars=2)
    "DSB", "KK", "NS", "SBB", "B", "PKP", "SNCF", "DR",  # Yellows (stars=3)
    "SZD", "SJ", "FS", "RENFE", "BR", "BSR", "E",  # Greens (stars=4)
    "HH", "HA", "HR", "MAD", "FRA", "LHR", "CDG",  # Blues (stars=5)
]

# Map company name to index
COMPANY_NAME_TO_ID = {name: i for i, name in enumerate(COMPANY_NAMES)}

# Face values (sorted order, so this is ascending)
cdef int[36] COMPANY_FACE_VALUE = [
    1, 2, 5, 6, 7, 8,           # Reds
    11, 12, 13, 14, 15, 16, 17, 19,  # Oranges
    20, 21, 22, 23, 24, 25, 26, 29,  # Yellows
    30, 31, 32, 33, 34, 36, 43,      # Greens
    45, 46, 47, 50, 56, 58, 60,      # Blues
]

# Low prices (face_value / 2, rounded up for some)
cdef int[36] COMPANY_LOW_PRICE = [
    1, 1, 3, 3, 4, 4,           # Reds
    6, 6, 7, 7, 8, 8, 9, 10,    # Oranges
    10, 11, 11, 12, 12, 13, 13, 15,  # Yellows
    15, 16, 16, 17, 17, 18, 22,      # Greens
    23, 23, 24, 25, 28, 29, 30,      # Blues
]

# High prices
cdef int[36] COMPANY_HIGH_PRICE = [
    2, 3, 7, 8, 9, 10,          # Reds
    14, 16, 17, 18, 20, 21, 22, 25,  # Oranges
    26, 28, 29, 30, 32, 33, 34, 38,  # Yellows
    40, 41, 42, 44, 45, 48, 57,      # Greens
    60, 61, 62, 66, 74, 77, 80,      # Blues
]

# Star ratings (1-5)
cdef int[36] COMPANY_STARS = [
    1, 1, 1, 1, 1, 1,           # Reds
    2, 2, 2, 2, 2, 2, 2, 2,     # Oranges
    3, 3, 3, 3, 3, 3, 3, 3,     # Yellows
    4, 4, 4, 4, 4, 4, 4,        # Greens
    5, 5, 5, 5, 5, 5, 5,        # Blues
]

# Base income
cdef int[36] COMPANY_INCOME = [
    1, 1, 2, 2, 2, 2,           # Reds (1-2)
    3, 3, 3, 3, 3, 3, 3, 3,     # Oranges (3)
    5, 5, 5, 5, 5, 5, 5, 5,     # Yellows (5)
    7, 7, 7, 7, 7, 7, 7,        # Greens (7)
    10, 10, 10, 10, 10, 10, 10, # Blues (10)
]

# Last in group flag (triggers CoO increase when revealed)
cdef uint8_t[36] COMPANY_LAST_IN_GROUP = [
    0, 0, 0, 0, 0, 1,           # MHE is last red
    0, 0, 0, 0, 0, 0, 0, 1,     # PR is last orange
    0, 0, 0, 0, 0, 0, 0, 1,     # DR is last yellow
    0, 0, 0, 0, 0, 0, 1,        # E is last green
    0, 0, 0, 0, 0, 0, 1,        # CDG is last blue
]

# =============================================================================
# SYNERGY DATA
# =============================================================================

# Synergy matrix: SYNERGY[i][j] = bonus that company i gets when paired with company j
# Most entries are 0, so this is sparse in practice. Module-scope cdef arrays
# are zero-initialized by the C runtime, so no explicit clear step is needed.
# Using int8 since max synergy bonus is 16.
cdef int8_t[36][36] COMPANY_SYNERGY

# Populated from Python at module init since the data is complex
def _set_synergy(int company_id, int target_id, int bonus):
    """Set synergy bonus that company_id gets when paired with target_id."""
    COMPANY_SYNERGY[company_id][target_id] = bonus

# =============================================================================
# CORPORATION DATA
# =============================================================================

# Corp IDs as strings
CORP_NAMES = ["JS", "S", "OS", "SM", "PR", "DA", "VM", "SI"]

# Map corp name to index
CORP_NAME_TO_ID = {name: i for i, name in enumerate(CORP_NAMES)}

# Share counts per corp
cdef int[8] CORP_SHARE_COUNT = [7, 7, 6, 6, 5, 5, 4, 4]

# =============================================================================
# MARKET DATA
# =============================================================================

# All 27 market spaces (index 0 = bankruptcy)
cdef int[27] MARKET_PRICES = [
    0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16,
    18, 20, 22, 24, 27, 30, 33, 37, 41, 45,
    50, 55, 61, 68, 75
]

# Reverse lookup: price -> index (for common prices)
# Returns -1 if not a valid market price
cdef int[76] PRICE_TO_MARKET_INDEX

cdef void _init_price_lookup() noexcept nogil:
    cdef int i
    for i in range(76):
        PRICE_TO_MARKET_INDEX[i] = -1
    for i in range(27):
        PRICE_TO_MARKET_INDEX[MARKET_PRICES[i]] = i

# All unique par prices (sorted)
cdef int[14] ALL_PAR_PRICES = [
    10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 27, 30, 33, 37
]

# Par price validity by star tier
# PAR_PRICE_VALID[star-1][par_price_index] = 1 if valid for that tier
cdef uint8_t[5][14] PAR_PRICE_VALID = [
    # Star 1 (reds): 10-14
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Star 2 (oranges): 10-20
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    # Star 3 (yellows): 16-27
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    # Star 4 (greens): 22-37
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    # Star 5 (blues): 30-37
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
]

# Python-visible mirrors of the par-price tables. Same ``globals()`` trick
# as ``PHASE_ACTION_SIZES`` / ``ENGINE_TO_DECISION_PHASE`` — a bare module-
# level assignment would be shadowed by the cimported ``cdef`` array.
# Cython callers still cimport the raw arrays above.
globals()["ALL_PAR_PRICES"] = tuple(ALL_PAR_PRICES[i] for i in range(14))
globals()["PAR_PRICE_VALID"] = tuple(
    tuple(bool(PAR_PRICE_VALID[star][par]) for par in range(14))
    for star in range(5)
)

# =============================================================================
# COST OF OWNERSHIP TABLE
# =============================================================================

# cost_of_ownership[coo_level-1][star_tier-1] = cost
# CoO levels are 1-7, star tiers are 1-5
cdef int[7][5] COST_OF_OWNERSHIP = [
    [0, 0, 0, 0, 0],  # Level 1 - Initial
    [0, 0, 0, 0, 0],  # Level 2 - Orange on top of deck
    [0, 0, 0, 0, 0],  # Level 3 - Yellow on top of deck
    [2, 0, 0, 0, 0],  # Level 4: reds cost 2 - Green on top of deck
    [4, 4, 0, 0, 0],  # Level 5: reds/oranges cost 4 - Blue on top of deck
    [7, 7, 7, 0, 0],  # Level 6: reds/oranges/yellows cost 7 - End card face-up
    [10, 10, 10, 10, 0],  # Level 7: all but blues cost 10 - End card flipped
]

# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def _init_module():
    """Initialize static data. Called once at module import."""
    _init_price_lookup()
    _populate_synergies()

def _populate_synergies():
    """Populate the synergy matrix from the game data."""
    # This maps the complex synergy structure to our flat matrix
    # Format: (company_name, {bonus: [target_names]})
    synergy_data = {
        # Blues
        "CDG": {16: ["MAD", "FRA", "LHR"], 8: ["E"], 4: ["SBB", "SNCF"]},
        "MAD": {8: ["RENFE"]},
        "LHR": {8: ["BR", "E"], 16: ["FRA"]},
        "HH": {4: ["DSB", "PKP", "DR"], 8: ["BSR"]},
        "HR": {4: ["NS", "B", "DR"], 8: ["E"]},
        "HA": {4: ["NS", "B", "SNCF"], 8: ["E"]},
        "FRA": {4: ["KK", "SBB", "PKP", "DR"]},
        # Greens
        "E": {4: ["NS", "B", "SNCF"], 8: ["BR"]},
        "RENFE": {4: ["SNCF"]},
        "BSR": {4: ["DSB", "PKP", "DR"], 8: ["SJ"]},
        "FS": {4: ["KK", "SBB", "SNCF"]},
        "SZD": {4: ["PKP"]},
        # Yellows
        "DR": {2: ["WT", "BY", "BD", "HE", "OL", "SX", "MS", "PR"],
               4: ["DSB", "KK", "NS", "SBB", "B", "PKP", "SNCF"]},
        "DSB": {2: ["OL", "MS", "PR"]},
        "B": {2: ["PR"], 4: ["NS"]},
        "SNCF": {2: ["BD"], 4: ["SBB", "B"]},
        "PKP": {2: ["SX", "MS", "PR"], 4: ["KK"]},
        "KK": {2: ["BY", "SX"]},
        "SBB": {2: ["WT", "BD"], 4: ["KK"]},
        "NS": {2: ["OL", "PR"]},
        # Oranges
        "PR": {1: ["BME", "BSE", "KME", "AKE", "BPM", "MHE"], 2: ["HE", "OL", "SX", "MS"]},
        "SX": {1: ["BSE", "BPM", "MHE"], 2: ["BY"]},
        "BY": {2: ["WT"]},
        "HE": {1: ["BME", "KME"], 2: ["BY", "BD"]},
        "MS": {1: ["BSE", "AKE", "BPM", "MHE"], 2: ["OL", "SX"]},
        "BD": {1: ["BME"], 2: ["WT"]},
        "OL": {1: ["KME", "AKE", "MHE"]},
        # Reds
        "MHE": {1: ["KME", "AKE", "BPM"]},
        "BPM": {1: ["BSE", "AKE"]},
        "KME": {1: ["BME"]},
    }

    for company_name, syns in synergy_data.items():
        company_id = COMPANY_NAME_TO_ID[company_name]
        for bonus, targets in syns.items():
            for target_name in targets:
                target_id = COMPANY_NAME_TO_ID[target_name]
                _set_synergy(company_id, target_id, bonus)

# Auto-initialize on import
_init_module()


# =============================================================================
# PYTHON-VISIBLE MIRRORS OF PAR-PRICE TABLES
# =============================================================================
#
# ``ALL_PAR_PRICES`` is ``cdef int[14]`` and ``PAR_PRICE_VALID`` is
# ``cdef uint8_t[5][14]``; neither is visible to Python by default.
# Tests (e.g. token-data invariants) and notebooks that want to iterate
# over valid par indices per star tier read the mirrors below. Defined
# after ``_init_module()`` so the ``cdef`` arrays above are fully
# populated before we snapshot them.

cdef list _all_par_prices_pylist():
    cdef int i
    cdef list out = []
    for i in range(14):
        out.append(ALL_PAR_PRICES[i])
    return out


cdef list _par_price_valid_pylist():
    cdef int tier, idx
    cdef list out = []
    cdef list row
    for tier in range(5):
        row = []
        for idx in range(14):
            row.append(<int>PAR_PRICE_VALID[tier][idx])
        out.append(row)
    return out


globals()["PY_ALL_PAR_PRICES"] = _all_par_prices_pylist()
globals()["PY_PAR_PRICE_VALID"] = _par_price_valid_pylist()
