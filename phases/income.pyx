# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Income phase implementation.

This phase is fully automatic - no player actions.

Flow:
1. FI collects income (adjusted company incomes + $5)
2. Players collect income (adjusted company incomes)
3. Corps collect income (adjusted incomes + synergies + special abilities)
4. Handle corp bankruptcies
5. Transition to Dividends
"""

cimport cython
from cython_core.state cimport GameState, PHASE_INCOME, PHASE_DIVIDENDS, NUM_COMPANIES, NUM_CORPS
from cython_core.data cimport (
    get_adjusted_company_income, get_company_income, get_company_synergy,
    get_cost_of_ownership, get_company_stars, get_corp_share_count,
    CORP_JS, CORP_S, CORP_OS, CORP_SM, CORP_PR, CORP_DA, CORP_VM, CORP_SI
)

# =============================================================================
# CONSTANTS (phase-specific only; NUM_COMPANIES/NUM_CORPS from state.pxd)
# =============================================================================

DEF FI_BASE_INCOME = 5  # Foreign investor always gets $5


# =============================================================================
# INCOME PHASE CLASS
# =============================================================================

cdef class IncomePhase:
    """
    Handles the Income phase.

    This is fully automatic - calculates and applies income for all entities,
    handles bankruptcies, and transitions to Dividends.
    """

    def __cinit__(self, int num_players):
        self._num_players = num_players

    cpdef void handle_income_phase(self, GameState state):
        """
        Main entry point - process entire income phase.
        """
        cdef int corp_id, player_id
        cdef int fi_income, player_income, corp_income, share_count

        if state.get_phase() != PHASE_INCOME:
            return

        # 1. Foreign investor income
        fi_income = self.calculate_fi_income(state)
        state.add_fi_cash(fi_income)

        # 2. Player income
        for player_id in range(self._num_players):
            player_income = self.calculate_player_income(state, player_id)
            state.add_player_cash(player_id, player_income)

        # 3. Corp income (with bankruptcy handling)
        for corp_id in range(NUM_CORPS):
            if not state.is_corp_active(corp_id):
                continue

            corp_income = self.calculate_corp_income(state, corp_id)
            state.add_corp_cash(corp_id, corp_income)

            # Check for bankruptcy
            if state.get_corp_cash(corp_id) < 0:
                # Reset unissued shares to full count before bankruptcy
                share_count = get_corp_share_count(corp_id)
                state.set_corp_unissued_shares(corp_id, share_count)
                state.bankrupt_corp(corp_id)

        # 4. Transition to Dividends
        state.set_phase(PHASE_DIVIDENDS)

    cpdef int calculate_fi_income(self, GameState state):
        """Calculate foreign investor's total income."""
        cdef int total = FI_BASE_INCOME
        cdef int company_id
        cdef int coo_level = state.get_coo_level()

        for company_id in range(NUM_COMPANIES):
            if state.fi_owns_company(company_id):
                total += get_adjusted_company_income(company_id, coo_level)

        return total

    cpdef int calculate_player_income(self, GameState state, int player_id):
        """Calculate player's total income from private companies."""
        cdef int total = 0
        cdef int company_id
        cdef int coo_level = state.get_coo_level()

        for company_id in range(NUM_COMPANIES):
            if state.player_owns_company(player_id, company_id):
                total += get_adjusted_company_income(company_id, coo_level)

        return total

    cpdef int calculate_corp_income(self, GameState state, int corp_id):
        """
        Calculate corporation's total income.

        Includes:
        - Adjusted company incomes
        - Synergy bonuses
        - Special abilities
        """
        cdef int coo_level = state.get_coo_level()
        cdef int company_id, other_id
        cdef int total = 0
        cdef int company_count = 0
        cdef int max_printed_income = 0
        cdef int total_coo = 0
        cdef int synergy_marker_count = 0
        cdef int printed_income, stars, synergy_bonus, vm_bonus
        cdef list owned_companies = []

        # Collect owned companies and calculate base income
        for company_id in range(NUM_COMPANIES):
            if state.corp_owns_company(corp_id, company_id):
                owned_companies.append(company_id)
                company_count += 1

                # Add adjusted income
                total += get_adjusted_company_income(company_id, coo_level)

                # Track max printed income (for DA ability)
                printed_income = get_company_income(company_id)
                if printed_income > max_printed_income:
                    max_printed_income = printed_income

                # Track total CoO (for VM ability)
                stars = get_company_stars(company_id)
                total_coo += get_cost_of_ownership(coo_level, stars)

        # Calculate synergies
        synergy_marker_count = self.calculate_corp_synergies(state, corp_id)

        # Add synergy income to total
        # Note: synergy income is tracked in the synergy matrix as bonus amount
        for company_id in owned_companies:
            for other_id in owned_companies:
                if company_id != other_id:
                    synergy_bonus = get_company_synergy(company_id, other_id)
                    if synergy_bonus > 0:
                        total += synergy_bonus

        # Special abilities
        # PR (Prussian Railway): +1 per company owned
        if corp_id == CORP_PR:
            total += company_count

        # DA (Doppler AG): +max printed income of owned companies
        if corp_id == CORP_DA:
            total += max_printed_income

        # VM (Vintage Machinery): reduces CoO by up to 10
        # This means we add back min(10, total_coo) to income
        if corp_id == CORP_VM:
            vm_bonus = total_coo
            if vm_bonus > 10:
                vm_bonus = 10
            total += vm_bonus

        # S (Synergistic): +1 per 2 synergy markers
        if corp_id == CORP_S:
            total += synergy_marker_count // 2

        return total

    cpdef int calculate_corp_synergies(self, GameState state, int corp_id):
        """
        Count synergy markers for a corporation.

        Returns total number of synergy connections (each pair counted once).
        """
        cdef int count = 0
        cdef int company_id, other_id, i, j
        cdef list owned_companies = []

        # Collect owned companies
        for company_id in range(NUM_COMPANIES):
            if state.corp_owns_company(corp_id, company_id):
                owned_companies.append(company_id)

        # Count synergy pairs (each pair once)
        for i in range(len(owned_companies)):
            company_id = owned_companies[i]
            for j in range(i + 1, len(owned_companies)):
                other_id = owned_companies[j]
                # Check if there's a synergy between these companies
                # (either direction counts as one connection)
                if get_company_synergy(company_id, other_id) > 0:
                    count += 1
                elif get_company_synergy(other_id, company_id) > 0:
                    count += 1

        return count
