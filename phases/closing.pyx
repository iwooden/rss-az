# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""CLOSING phase handler implementation."""

from core.state cimport GameState
from core.data cimport (
    GameConstants,
    get_cost_of_ownership, get_company_income, get_company_stars, get_company_face_value
)
from entities import turn as turn_module
from entities import company as company_module
from entities import corp as corp_module
from entities import fi as fi_module


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

cdef void _close_company(GameState state, int company_id, int owner_type, int owner_id) noexcept:
    """
    Close a company and handle cleanup.

    Steps:
    1. Clear ownership from previous owner (FI or corp)
    2. Apply Junkyard Scrappers bonus (2x printed income to JS cash)
    3. Remove company from game

    Args:
        state: Game state
        company_id: Company to close (0-35)
        owner_type: LOC_FI (4) or LOC_CORP (5)
        owner_id: Corp ID if owner_type is LOC_CORP, -1 otherwise
    """
    cdef int printed_income = get_company_income(company_id)

    # Clear ownership before removal
    if owner_type == 4:  # LOC_FI
        fi_module.FI.set_owns_company(state, company_id, False)
    elif owner_type == 5:  # LOC_CORP
        corp_module.CORPS[owner_id].set_owns_company(state, company_id, False)

    # Junkyard Scrappers (corp_id 0) bonus: 2x printed income
    if corp_module.CORPS[0].is_active(state):
        corp_module.CORPS[0].add_cash(state, printed_income * 2)

    # Remove company from game
    company_module.COMPANIES[company_id].remove_from_game(state)


cdef void _process_fi_auto_close(GameState state) noexcept:
    """
    Execute FI auto-close logic.

    FI closes companies with negative adjusted income (income - CoO < 0).
    Only applies to companies owned by FI.
    """
    cdef int coo_level = turn_module.TURN.get_coo_level(state)
    cdef int company_id
    cdef int base_income, stars, coo_value, adjusted_income
    cdef int num_to_close = 0
    cdef int[36] companies_to_close  # Track which companies to close

    # First pass: identify companies to close
    for company_id in range(GameConstants.NUM_COMPANIES):
        if fi_module.FI.owns_company(state, company_id):
            base_income = get_company_income(company_id)
            stars = get_company_stars(company_id)
            coo_value = get_cost_of_ownership(coo_level, stars)
            adjusted_income = base_income - coo_value

            # Close if NEGATIVE (not zero)
            if adjusted_income < 0:
                companies_to_close[num_to_close] = company_id
                num_to_close += 1

    # Second pass: close identified companies
    for company_id in range(num_to_close):
        _close_company(state, companies_to_close[company_id], 4, -1)  # LOC_FI = 4


cdef void _process_receivership_auto_close(GameState state) noexcept:
    """
    Execute receivership corporation auto-close logic.

    Receivership corps close companies based on CoO thresholds:
    - Red (1 star): close if CoO >= $4
    - Orange (2 stars): close if CoO >= $7
    - Yellow/Green/Blue (3-5 stars): NEVER auto-close

    Protection: Highest face value company in each receivership corp is protected.
    Vintage Machinery (corp 6): Apply $10 CoO reduction before threshold check.
    """
    cdef int corp_id, company_id, stars, coo_value
    cdef int coo_level = turn_module.TURN.get_coo_level(state)
    cdef bint is_vm
    cdef int protected_company, max_face_value, face_value
    cdef int num_to_close
    cdef int[36] companies_to_close

    for corp_id in range(GameConstants.NUM_CORPS):
        # Skip inactive corps and non-receivership corps
        if not corp_module.CORPS[corp_id].is_active(state):
            continue
        if not corp_module.CORPS[corp_id].is_in_receivership(state):
            continue

        # Check if this is Vintage Machinery (corp_id 6)
        is_vm = (corp_id == 6)

        # Find highest face value company (protected)
        protected_company = -1
        max_face_value = -1
        for company_id in range(GameConstants.NUM_COMPANIES):
            if corp_module.CORPS[corp_id].owns_company(state, company_id):
                face_value = get_company_face_value(company_id)
                if face_value > max_face_value:
                    max_face_value = face_value
                    protected_company = company_id

        # Identify companies to close
        num_to_close = 0
        for company_id in range(GameConstants.NUM_COMPANIES):
            # Skip if not owned by this corp
            if not corp_module.CORPS[corp_id].owns_company(state, company_id):
                continue

            # Skip protected company
            if company_id == protected_company:
                continue

            # Get CoO value
            stars = get_company_stars(company_id)
            coo_value = get_cost_of_ownership(coo_level, stars)

            # Apply Vintage Machinery reduction
            if is_vm:
                coo_value = max(0, coo_value - 10)

            # Check thresholds
            if stars == 1 and coo_value >= 4:  # Red
                companies_to_close[num_to_close] = company_id
                num_to_close += 1
            elif stars == 2 and coo_value >= 7:  # Orange
                companies_to_close[num_to_close] = company_id
                num_to_close += 1
            # Yellow (3), Green (4), Blue (5): never auto-close

        # Close identified companies
        for company_id in range(num_to_close):
            _close_company(state, companies_to_close[company_id], 5, corp_id)  # LOC_CORP = 5


# =============================================================================
# MAIN PHASE HANDLER
# =============================================================================

cdef int apply_closing_auto(GameState state) noexcept:
    """
    Execute auto-close logic for FI and receivership corps.

    This is a deterministic non-player phase with 0 actions.
    Steps:
    1. FI closes companies with negative adjusted income
    2. Receivership corps close red/orange companies above CoO thresholds
    3. Junkyard Scrappers receives 2x printed income for each closure

    Returns: 0 always (deterministic, no failure modes)
    """
    _process_fi_auto_close(state)
    _process_receivership_auto_close(state)
    return 0


def apply_closing_auto_py(GameState state):
    """Python wrapper for testing."""
    return apply_closing_auto(state)
