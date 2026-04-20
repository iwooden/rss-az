"""Tests for the ACQ_SELECT_COMPANY phase (middle leg of three-step Acquisition).

Entered only via ``apply_acq_select_corp_action`` from PHASE_ACQ_SELECT_CORP.
Covers per-company legality (affordability, seller retains-one gate, cross-
president gate under both ``acq_same_president`` modes), action id encoding
(``action_id == company_id``), no-pass invariant, and the SELECT_COMPANY →
SELECT_PRICE transition (active_company set, phase advances).

Pass / corp-select concerns live in ``test_acq_select_corp.py``; price
selection + execution live in ``test_acq_select_price.py``.
"""
from core.actions import (
    ACTION_PASS_PY as ACTION_PASS,
    ACTION_ACQ_SELECT_CORP_PY as ACTION_ACQ_SELECT_CORP,
    ACTION_ACQ_SELECT_COMPANY_PY as ACTION_ACQ_SELECT_COMPANY,
)
from core.data import GamePhases
from core.state import GameState
from entities.turn import TURN
from entities.corp import CORPS
from entities.company import COMPANIES, CompanyLocation
from phases.acq_select_corp import setup_acquisition_phase_py

from tests.phases.conftest import (
    apply_and_verify,
    get_legal_actions,
    find_legal_action,
    find_all_legal_actions,
    float_corp_for_test,
    draw_to_player,
    draw_to_fi,
    draw_to_corp,
)


# =============================================================================
# HELPERS
# =============================================================================

def _enter_select_company(state, corp_id):
    """Seed state so the active decision is PHASE_ACQ_SELECT_COMPANY for ``corp_id``.

    Routes through PHASE_ACQ_SELECT_CORP and applies a corp-select action so
    the SELECT_COMPANY handler sees the same active_corp seeding production
    code would produce. Passes intermediate players until the target corp's
    president is active, then corp-selects. Callers must stage enough legal
    targets (≥2) so the driver auto-chain does not immediately auto-pick a
    forced company.
    """
    setup_acquisition_phase_py(state)
    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_CORP), (
        f"_enter_select_company: expected PHASE_ACQ_SELECT_CORP after setup, "
        f"got phase={TURN.get_phase(state)}"
    )

    target_president = CORPS[corp_id].get_president_id(state)
    assert target_president >= 0, (
        f"_enter_select_company: corp {corp_id} has no president (inactive or receivership?)"
    )
    # Pass through earlier players until the target corp's president is active.
    for _ in range(16):
        if TURN.get_active_player(state) == target_president:
            break
        assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_CORP), (
            f"_enter_select_company: fell out of PHASE_ACQ_SELECT_CORP while "
            f"skipping to president {target_president}; phase={TURN.get_phase(state)}"
        )
        pass_id = find_legal_action(state, action_type=ACTION_PASS)
        apply_and_verify(state, pass_id)
    else:
        assert False, (
            f"_enter_select_company: never reached president {target_president}"
        )

    aid = find_legal_action(
        state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=corp_id,
    )
    apply_and_verify(state, aid)
    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_COMPANY), (
        f"_enter_select_company: expected PHASE_ACQ_SELECT_COMPANY after "
        f"corp-select, got phase={TURN.get_phase(state)}. The driver "
        f"auto-chains past SELECT_COMPANY when only one target is legal — "
        f"add more eligible companies to observe a real decision."
    )
    assert TURN.get_active_corp(state) == corp_id
    assert TURN.get_active_company(state) == -1


# =============================================================================
# ENUMERATION
# =============================================================================

class TestEnumeration:
    """SELECT_COMPANY legality: per-target affordability × ownership gate. No pass."""

    def test_no_pass_action(self, game_state):
        """SELECT_COMPANY has no pass — corp-select already committed."""
        draw_to_player(game_state, 0)
        draw_to_player(game_state, 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        _enter_select_company(game_state, corp_id=0)

        actions = get_legal_actions(game_state)
        assert all(info.action_type == ACTION_ACQ_SELECT_COMPANY for _, info in actions)
        assert not any(info.action_type == ACTION_PASS for _, info in actions)

    def test_action_id_encoding(self, game_state):
        """SELECT_COMPANY action id equals the ``company_id`` (no pass offset)."""
        co1 = draw_to_player(game_state, 0)
        draw_to_player(game_state, 0)  # second target
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        _enter_select_company(game_state, corp_id=0)

        aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_COMPANY, company_id=co1,
        )
        assert aid == co1

    def test_fi_target_enumerated(self, game_state):
        """LOC_FI companies the corp can afford are legal SELECT_COMPANY targets."""
        fi_co = draw_to_fi(game_state)
        draw_to_fi(game_state)  # second FI target so SELECT_COMPANY doesn't auto-chain
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        _enter_select_company(game_state, corp_id=0)

        co_ids = {info.company_id for _, info in get_legal_actions(game_state)}
        assert fi_co in co_ids

    def test_player_target_enumerated(self, game_state):
        """LOC_PLAYER companies the corp can afford are legal targets."""
        co1 = draw_to_player(game_state, 0)
        draw_to_player(game_state, 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        _enter_select_company(game_state, corp_id=0)

        co_ids = {info.company_id for _, info in get_legal_actions(game_state)}
        assert co1 in co_ids

    def test_corp_target_enumerated_with_retention_gate(self, game_state):
        """Corp-owned target is legal when seller retains ≥1 company after sale."""
        # Seller (corp 0) has 2 companies → can sell one, retains one.
        seller_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        draw_to_corp(game_state, 0)

        # Buyer (corp 1, same president)
        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)
        CORPS[1].set_cash(game_state, 500)

        _enter_select_company(game_state, corp_id=1)

        co_ids = {info.company_id for _, info in get_legal_actions(game_state)}
        assert seller_co in co_ids

    def test_seller_with_one_company_not_enumerated(self, game_state):
        """Corp with exactly 1 company cannot sell it — target excluded."""
        # Seller corp 0 has 1 company (floated only), not sellable.
        seller_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        # Two player-owned targets so SELECT_COMPANY has ≥2 legal actions.
        draw_to_player(game_state, 0)
        draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)
        CORPS[1].set_cash(game_state, 500)

        _enter_select_company(game_state, corp_id=1)

        co_ids = {info.company_id for _, info in get_legal_actions(game_state)}
        assert seller_co not in co_ids

    def test_own_company_not_enumerated(self, game_state):
        """Corp cannot buy a company it already owns."""
        owned_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        draw_to_corp(game_state, 0)  # seller retains ≥1 after any sale
        # Two player-owned targets so SELECT_COMPANY is a real decision.
        draw_to_player(game_state, 0)
        draw_to_player(game_state, 0)
        CORPS[0].set_cash(game_state, 500)

        _enter_select_company(game_state, corp_id=0)

        co_ids = {info.company_id for _, info in get_legal_actions(game_state)}
        assert owned_co not in co_ids

    def test_unaffordable_target_excluded(self, game_state):
        """Company whose low price exceeds corp cash is not a legal target."""
        # Three companies so that even if one is priced out, ≥2 remain legal.
        co_a = draw_to_player(game_state, 0)
        co_b = draw_to_player(game_state, 0)
        co_c = draw_to_player(game_state, 0)
        by_price = sorted([co_a, co_b, co_c], key=lambda c: COMPANIES[c].get_low_price())
        cheap_co, mid_co, expensive_co = by_price

        # Only proceed when the seed actually produced a price split.
        if (COMPANIES[expensive_co].get_low_price()
                <= COMPANIES[mid_co].get_low_price()):
            return

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, COMPANIES[mid_co].get_low_price())

        _enter_select_company(game_state, corp_id=0)

        co_ids = {info.company_id for _, info in get_legal_actions(game_state)}
        assert cheap_co in co_ids
        assert mid_co in co_ids
        assert expensive_co not in co_ids

    def test_cross_president_corp_to_corp_excluded_in_same_pres_mode(self, game_state):
        """Cross-president corp-to-corp is not enumerated when flag is True (default)."""
        seller_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        draw_to_corp(game_state, 0)

        # Buyer presided by a different player (player 1)
        float_corp_for_test(game_state, corp_id=1, player_id=1, par_index=12)
        CORPS[1].set_cash(game_state, 500)
        # Two local targets so SELECT_COMPANY is a real decision.
        draw_to_player(game_state, 1)
        draw_to_player(game_state, 1)

        _enter_select_company(game_state, corp_id=1)

        co_ids = {info.company_id for _, info in get_legal_actions(game_state)}
        assert seller_co not in co_ids

    def test_cross_president_corp_to_corp_included_with_flag_false(self):
        """Cross-president corp-to-corp IS enumerated when acq_same_president=False."""
        state = GameState(3, acq_same_president=False)
        state.initialize_game(3, seed=42)

        seller_co = float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        draw_to_corp(state, 0)

        float_corp_for_test(state, corp_id=1, player_id=1, par_index=12)
        CORPS[1].set_cash(state, 500)
        # Local target so SELECT_COMPANY has ≥2 legal actions (local + cross-pres).
        draw_to_player(state, 1)

        _enter_select_company(state, corp_id=1)

        co_ids = {info.company_id for _, info in get_legal_actions(state)}
        assert seller_co in co_ids

    def test_cross_president_player_target_excluded_in_same_pres_mode(self, game_state):
        """Cross-president corp-to-player is excluded in default mode."""
        foreign_private = draw_to_player(game_state, 1)
        # Two local privates so SELECT_COMPANY is a real decision.
        local_a = draw_to_player(game_state, 0)
        local_b = draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        _enter_select_company(game_state, corp_id=0)

        co_ids = {info.company_id for _, info in get_legal_actions(game_state)}
        assert local_a in co_ids
        assert local_b in co_ids
        assert foreign_private not in co_ids

    def test_receivership_seller_excluded(self, game_state):
        """Receivership corps cannot be the seller — targets are excluded."""
        from tests.phases.conftest import setup_receivership_corp

        # Buyer
        float_corp_for_test(game_state, corp_id=0, company_id=10, player_id=0, par_index=12)
        CORPS[0].set_cash(game_state, 500)
        # Two local targets so SELECT_COMPANY stays a real decision.
        draw_to_player(game_state, 0)
        draw_to_player(game_state, 0)

        # Receivership seller with ≥2 companies (would otherwise be sellable).
        recv_co1 = 11
        recv_co2 = 12
        setup_receivership_corp(game_state, corp_id=1, company_ids=[recv_co1, recv_co2])

        _enter_select_company(game_state, corp_id=0)

        co_ids = {info.company_id for _, info in get_legal_actions(game_state)}
        assert recv_co1 not in co_ids
        assert recv_co2 not in co_ids


# =============================================================================
# TRANSITIONS
# =============================================================================

class TestTransitions:
    """SELECT_COMPANY → SELECT_PRICE: active_company set, phase advances."""

    def test_company_select_sets_active_company(self, game_state):
        """Applying SELECT_COMPANY seeds active_company with the chosen id."""
        co1 = draw_to_player(game_state, 0)
        draw_to_player(game_state, 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        # Cash covers low but leaves multiple offsets so SELECT_PRICE is a real decision.
        CORPS[0].set_cash(
            game_state, COMPANIES[co1].get_low_price() + 10,
        )

        _enter_select_company(game_state, corp_id=0)

        aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_COMPANY, company_id=co1,
        )
        apply_and_verify(game_state, aid)

        assert TURN.get_active_company(game_state) == co1

    def test_company_select_switches_to_select_price(self, game_state):
        """Applying SELECT_COMPANY transitions to PHASE_ACQ_SELECT_PRICE."""
        co1 = draw_to_player(game_state, 0)
        draw_to_player(game_state, 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(
            game_state, COMPANIES[co1].get_low_price() + 10,
        )

        _enter_select_company(game_state, corp_id=0)

        aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_COMPANY, company_id=co1,
        )
        apply_and_verify(game_state, aid)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_SELECT_PRICE)

    def test_company_select_preserves_active_corp(self, game_state):
        """active_corp remains stamped across the SELECT_COMPANY → SELECT_PRICE handoff."""
        co1 = draw_to_player(game_state, 0)
        draw_to_player(game_state, 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(
            game_state, COMPANIES[co1].get_low_price() + 10,
        )

        _enter_select_company(game_state, corp_id=0)

        aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_COMPANY, company_id=co1,
        )
        apply_and_verify(game_state, aid)

        assert TURN.get_active_corp(game_state) == 0


# =============================================================================
# BOUNDARY: FI TARGET AUTO-CHAINS
# =============================================================================

class TestFiTargetAutoChain:
    """FI targets collapse SELECT_PRICE to a single action — observe end state."""

    def test_single_fi_target_auto_chains_through_price(self, game_state):
        """When only one legal target (FI) exists, SELECT_COMPANY auto-chains to execution."""
        fi_co = draw_to_fi(game_state)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        setup_acquisition_phase_py(game_state)

        aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=0,
        )
        apply_and_verify(game_state, aid)

        # SELECT_COMPANY had 1 legal action; SELECT_PRICE for FI had 1. Driver
        # auto-chained through both and executed the buy.
        loc = COMPANIES[fi_co].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[fi_co].get_owner_id(game_state) == 0


# =============================================================================
# EMPTY ENUMERATION GUARD
# =============================================================================

class TestGatingInvariant:
    """SELECT_CORP's corp legality hoist must match SELECT_COMPANY's filter."""

    def test_select_corp_hides_corp_with_no_reachable_target(self, game_state):
        """Corp with cash but no reachable targets is not offered in SELECT_CORP.

        If this regresses, SELECT_COMPANY hits the 0-legal-actions assertion.
        """
        # Corp with cash. Only available company is its own (can't buy it).
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        setup_acquisition_phase_py(game_state)

        # Only PASS remains — SELECT_CORP hoisted out the empty-target corp.
        corps_offered = [
            info for _, info in find_all_legal_actions(
                game_state, action_type=ACTION_ACQ_SELECT_CORP,
            )
        ]
        assert corps_offered == []
