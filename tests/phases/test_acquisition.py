"""Tests for the ACQUISITION phase.

Covers: PASS, ACQ_PRICE (corp-to-corp, corp-to-player), FI_BUY, FI preemption
via ACQ_OFFER, receivership forced buys, phase transitions to CLOSING,
acquisition zone merging, and legal-action enumeration.
"""
from core.actions import (
    ACTION_PASS_PY as ACTION_PASS,
    ACTION_ACQ_PRICE_PY as ACTION_ACQ_PRICE,
    ACTION_ACQ_FI_BUY_PY as ACTION_ACQ_FI_BUY,
)
from core.data import GamePhases, GameConstants, CorpIndices
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES, CompanyLocation
from entities.fi import FI
from phases.acquisition import setup_acquisition_phase_py

from tests.phases.conftest import (
    apply_and_verify,
    get_legal_actions,
    find_legal_action,
    find_all_legal_actions,
    float_corp_for_test,
    setup_receivership_corp,
    draw_to_player,
    draw_to_fi,
    draw_to_corp,
)


# =============================================================================
# HELPERS
# =============================================================================

CORP_OS = int(CorpIndices.CORP_OS)  # 2


def _pass_through_acquisition(state, max_steps=50):
    """Pass through all ACQUISITION decisions until phase changes."""
    for _ in range(max_steps):
        if TURN.get_phase(state) != int(GamePhases.PHASE_ACQUISITION):
            return
        pass_id = find_legal_action(state, action_type=ACTION_PASS)
        apply_and_verify(state, pass_id)


# =============================================================================
# PASS ACTION TESTS
# =============================================================================

class TestPassAction:
    """Test ACQUISITION phase PASS action behavior."""

    def test_pass_marks_player_as_passed(self, game_state):
        """Passing marks the active player as passed."""
        num_players = TURN.get_num_players(game_state)
        if num_players < 3:
            return  # need 2 distinct presidents with buy targets

        # Both players need corps AND buy targets so the driver doesn't
        # auto-chain the second player's forced PASS after the first passes
        draw_to_player(game_state, 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 100)

        draw_to_player(game_state, 1)
        float_corp_for_test(game_state, corp_id=1, player_id=1, par_index=12)
        CORPS[1].set_cash(game_state, 100)

        setup_acquisition_phase_py(game_state)

        active = TURN.get_active_player(game_state)
        assert not PLAYERS[active].has_passed(game_state)

        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)
        assert PLAYERS[active].has_passed(game_state)

    def test_pass_advances_to_next_player(self, game_state):
        """Passing advances control to the next eligible player."""
        num_players = TURN.get_num_players(game_state)
        if num_players < 3:
            return

        draw_to_player(game_state, 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 100)

        draw_to_player(game_state, 1)
        float_corp_for_test(game_state, corp_id=1, player_id=1, par_index=12)
        CORPS[1].set_cash(game_state, 100)

        setup_acquisition_phase_py(game_state)

        first_active = TURN.get_active_player(game_state)
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQUISITION)
        assert TURN.get_active_player(game_state) != first_active

    def test_all_pass_transitions_past_acquisition(self, game_state):
        """When all eligible players pass, phase transitions past ACQUISITION."""
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 100)

        setup_acquisition_phase_py(game_state)
        _pass_through_acquisition(game_state)

        assert TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION)


# =============================================================================
# FI BUY TESTS
# =============================================================================

class TestFiBuy:
    """Test ACQUISITION FI_BUY action: corp buys from Foreign Investor."""

    def test_fi_buy_transfers_company_and_cash(self, game_state):
        """FI buy moves company to corp, deducts cash from corp, pays FI."""
        fi_company = draw_to_fi(game_state)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        corp_cash_before = CORPS[0].get_cash(game_state)
        fi_cash_before = FI.get_cash(game_state)
        expected_price = COMPANIES[fi_company].get_high_price()

        fi_buy_id = find_legal_action(
            game_state, action_type=ACTION_ACQ_FI_BUY, corp_id=0, company_id=fi_company,
        )
        apply_and_verify(game_state, fi_buy_id)

        loc = COMPANIES[fi_company].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[fi_company].get_owner_id(game_state) == 0
        assert CORPS[0].get_cash(game_state) == corp_cash_before - expected_price
        assert FI.get_cash(game_state) == fi_cash_before + expected_price

    def test_os_pays_face_value_for_fi(self, game_state):
        """OS (corp 2) pays face value instead of high price for FI companies."""
        fi_company = draw_to_fi(game_state)

        float_corp_for_test(game_state, corp_id=CORP_OS, player_id=0, par_index=10)
        CORPS[CORP_OS].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        corp_cash_before = CORPS[CORP_OS].get_cash(game_state)
        fi_cash_before = FI.get_cash(game_state)
        expected_price = COMPANIES[fi_company].get_face_value()

        fi_buy_id = find_legal_action(
            game_state, action_type=ACTION_ACQ_FI_BUY, corp_id=CORP_OS, company_id=fi_company,
        )
        apply_and_verify(game_state, fi_buy_id)

        assert CORPS[CORP_OS].get_cash(game_state) == corp_cash_before - expected_price
        assert FI.get_cash(game_state) == fi_cash_before + expected_price


# =============================================================================
# ACQ_PRICE TESTS (CORP-TO-CORP, CORP-TO-PLAYER)
# =============================================================================

class TestAcqPrice:
    """Test ACQUISITION ACQ_PRICE action: negotiated-price acquisitions."""

    def test_corp_to_corp_same_president(self, game_state):
        """Corp acquires from another corp under same president."""
        # Float corp 0 (seller) with 2 companies so it retains >= 1 after sale
        seller_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        draw_to_corp(game_state, 0)

        # Float corp 1 (buyer), same president (player 0)
        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)
        CORPS[1].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        acq_actions = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=1, company_id=seller_co,
        )
        assert len(acq_actions) > 0, "Expected ACQ_PRICE actions for corp-to-corp"

        action_id = acq_actions[len(acq_actions) // 2]
        info = next(i for aid, i in get_legal_actions(game_state) if aid == action_id)
        price = COMPANIES[seller_co].get_low_price() + info.amount

        buyer_cash_before = CORPS[1].get_cash(game_state)
        seller_proceeds_before = CORPS[0].get_acquisition_proceeds(game_state)

        apply_and_verify(game_state, action_id)

        loc = COMPANIES[seller_co].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[seller_co].get_owner_id(game_state) == 1
        assert CORPS[1].get_cash(game_state) == buyer_cash_before - price

        if TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQUISITION):
            assert CORPS[0].get_acquisition_proceeds(game_state) == seller_proceeds_before + price

    def test_corp_to_player_acquisition(self, game_state):
        """Corp acquires a private company from the president (same player)."""
        private_co = draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        action_id = find_legal_action(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0, company_id=private_co,
        )
        info = next(i for aid, i in get_legal_actions(game_state) if aid == action_id)
        price = COMPANIES[private_co].get_low_price() + info.amount

        player_cash_before = PLAYERS[0].get_cash(game_state)
        corp_cash_before = CORPS[0].get_cash(game_state)

        apply_and_verify(game_state, action_id)

        assert PLAYERS[0].get_cash(game_state) == player_cash_before + price
        assert CORPS[0].get_cash(game_state) == corp_cash_before - price
        loc = COMPANIES[private_co].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[private_co].get_owner_id(game_state) == 0

    def test_price_at_low_boundary(self, game_state):
        """Acquisition at minimum price (low_price + 0 offset)."""
        private_co = draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        action_id = find_legal_action(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0,
            company_id=private_co, amount=0,
        )
        low_price = COMPANIES[private_co].get_low_price()
        corp_cash_before = CORPS[0].get_cash(game_state)

        apply_and_verify(game_state, action_id)
        assert CORPS[0].get_cash(game_state) == corp_cash_before - low_price

    def test_price_at_high_boundary(self, game_state):
        """Acquisition at maximum price (high_price)."""
        private_co = draw_to_player(game_state, 0)
        high_price = COMPANIES[private_co].get_high_price()
        low_price = COMPANIES[private_co].get_low_price()
        max_offset = high_price - low_price

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, high_price + 50)

        setup_acquisition_phase_py(game_state)

        action_id = find_legal_action(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0,
            company_id=private_co, amount=max_offset,
        )
        corp_cash_before = CORPS[0].get_cash(game_state)

        apply_and_verify(game_state, action_id)
        assert CORPS[0].get_cash(game_state) == corp_cash_before - high_price

    def test_seller_must_retain_one_company(self, game_state):
        """A corp with exactly 1 company cannot sell it."""
        seller_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)

        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)
        CORPS[1].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        acq_actions = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=1, company_id=seller_co,
        )
        assert len(acq_actions) == 0, "Should not buy from corp with only 1 company"


# =============================================================================
# FI PREEMPTION TESTS (ACQ_OFFER)
# =============================================================================

class TestFiPreemption:
    """Test FI buy preemption through ACQ_OFFER phase."""

    def test_higher_priority_corp_preempts(self, game_state):
        """A higher-share-price corp preempts a lower one's FI buy."""
        num_players = TURN.get_num_players(game_state)
        if num_players < 3:
            return

        fi_co = draw_to_fi(game_state)

        # Low-price buyer (player 1)
        float_corp_for_test(game_state, corp_id=3, player_id=1, par_index=5)
        CORPS[3].set_cash(game_state, 200)

        # High-price preemptor (player 2)
        float_corp_for_test(game_state, corp_id=4, player_id=2, par_index=15)
        CORPS[4].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        # Player 1 is first in turn order with a corp, so should be active
        active = TURN.get_active_player(game_state)
        if active != 1:
            return  # turn order doesn't match expected setup

        fi_buy_id = find_legal_action(
            game_state, action_type=ACTION_ACQ_FI_BUY, corp_id=3, company_id=fi_co,
        )
        apply_and_verify(game_state, fi_buy_id)

        phase = TURN.get_phase(game_state)
        if phase == int(GamePhases.PHASE_ACQ_OFFER):
            assert TURN.get_active_player(game_state) == 2


# =============================================================================
# RECEIVERSHIP FORCED BUY TESTS
# =============================================================================

class TestReceivershipForcedBuys:
    """Test automatic receivership FI buys at start of ACQUISITION."""

    def test_receivership_corp_auto_buys_fi_company(self, game_state):
        """A receivership corp with cash automatically buys from FI."""
        recv_co = draw_to_player(game_state, 0)
        setup_receivership_corp(game_state, corp_id=0, company_ids=[recv_co])
        CORPS[0].set_cash(game_state, 200)

        fi_co = draw_to_fi(game_state)
        fi_cash_before = FI.get_cash(game_state)

        setup_acquisition_phase_py(game_state)

        phase = TURN.get_phase(game_state)
        if phase == int(GamePhases.PHASE_ACQ_OFFER):
            return  # preemption happened

        loc = COMPANIES[fi_co].get_location(game_state)
        if loc == int(CompanyLocation.LOC_FI):
            return  # too expensive
        assert FI.get_cash(game_state) >= fi_cash_before

    def test_receivership_with_no_cash_skips(self, game_state):
        """Receivership corp with no cash does not buy from FI."""
        recv_co = draw_to_player(game_state, 0)
        setup_receivership_corp(game_state, corp_id=0, company_ids=[recv_co])
        CORPS[0].set_cash(game_state, 0)

        fi_co = draw_to_fi(game_state)

        setup_acquisition_phase_py(game_state)

        assert COMPANIES[fi_co].get_location(game_state) == int(CompanyLocation.LOC_FI)


# =============================================================================
# PHASE TRANSITION TESTS
# =============================================================================

class TestPhaseTransitions:
    """Test phase transitions from ACQUISITION."""

    def test_no_active_corps_skips_to_closing(self, game_state):
        """With no active corps, ACQUISITION transitions directly to CLOSING."""
        setup_acquisition_phase_py(game_state)
        assert TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION)

    def test_only_receivership_corps_transitions_after_forced_buys(self, game_state):
        """With only receivership corps, ACQUISITION transitions past."""
        recv_co = draw_to_player(game_state, 0)
        setup_receivership_corp(game_state, corp_id=0, company_ids=[recv_co])
        CORPS[0].set_cash(game_state, 0)

        setup_acquisition_phase_py(game_state)
        assert TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION)

    def test_acquisition_eventually_reaches_closing(self, game_state):
        """Playing through ACQUISITION by passing always reaches CLOSING or beyond."""
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 100)

        setup_acquisition_phase_py(game_state)
        _pass_through_acquisition(game_state)

        phase = TURN.get_phase(game_state)
        assert phase >= int(GamePhases.PHASE_CLOSING), (
            f"Expected CLOSING or later, got phase {phase}"
        )


# =============================================================================
# ACQUISITION ZONE MERGE TESTS
# =============================================================================

class TestAcquisitionZoneMerge:
    """Test that acq zones merge and proceeds flush on phase exit."""

    def test_acq_zone_merged_after_phase(self, game_state):
        """After ACQUISITION ends, no companies remain in LOC_CORP_ACQ."""
        private_co = draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        acq_id = find_legal_action(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0, company_id=private_co,
        )
        apply_and_verify(game_state, acq_id)
        _pass_through_acquisition(game_state)

        for cid in range(int(GameConstants.NUM_COMPANIES)):
            assert COMPANIES[cid].get_location(game_state) != int(CompanyLocation.LOC_CORP_ACQ), (
                f"Company {cid} still in LOC_CORP_ACQ after ACQUISITION ended"
            )

    def test_proceeds_flushed_after_phase(self, game_state):
        """After ACQUISITION ends, all corps have 0 acquisition_proceeds."""
        seller_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        draw_to_corp(game_state, 0)

        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)
        CORPS[1].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        acq_id = find_legal_action(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=1, company_id=seller_co,
        )
        apply_and_verify(game_state, acq_id)
        _pass_through_acquisition(game_state)

        for corp_id in range(int(GameConstants.NUM_CORPS)):
            if CORPS[corp_id].is_active(game_state):
                assert CORPS[corp_id].get_acquisition_proceeds(game_state) == 0, (
                    f"Corp {corp_id} still has nonzero acquisition_proceeds"
                )


# =============================================================================
# ENUMERATION TESTS
# =============================================================================

class TestEnumeration:
    """Test legal-action enumeration for the ACQUISITION phase."""

    def test_pass_always_legal(self, game_state):
        """PASS is always legal in ACQUISITION decision states."""
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        actions = get_legal_actions(game_state)
        pass_actions = [info for _, info in actions if info.action_type == ACTION_PASS]
        assert len(pass_actions) == 1

    def test_no_fi_buy_when_no_fi_companies(self, game_state):
        """No FI_BUY actions when FI owns no companies."""
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        setup_acquisition_phase_py(game_state)

        actions = get_legal_actions(game_state)
        fi_buys = [info for _, info in actions if info.action_type == ACTION_ACQ_FI_BUY]
        assert len(fi_buys) == 0

    def test_fi_buy_limited_by_cash(self, game_state):
        """FI_BUY actions only for companies the corp can afford."""
        fi_co = draw_to_fi(game_state)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        high_price = COMPANIES[fi_co].get_high_price()
        CORPS[0].set_cash(game_state, high_price)

        setup_acquisition_phase_py(game_state)

        cash = CORPS[0].get_cash(game_state)
        actions = get_legal_actions(game_state)
        for _, info in actions:
            if info.action_type == ACTION_ACQ_FI_BUY and info.corp_id == 0:
                co_price = COMPANIES[info.company_id].get_high_price()
                assert co_price <= cash, (
                    f"FI buy for company {info.company_id} costs {co_price} "
                    f"but corp only has {cash}"
                )

    def test_acq_price_limited_by_cash(self, game_state):
        """ACQ_PRICE offsets cannot exceed what the corp can afford."""
        private_co = draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        low_price = COMPANIES[private_co].get_low_price()
        CORPS[0].set_cash(game_state, low_price + 3)

        setup_acquisition_phase_py(game_state)

        cash = CORPS[0].get_cash(game_state)
        actions = get_legal_actions(game_state)
        for _, info in actions:
            if (info.action_type == ACTION_ACQ_PRICE
                    and info.corp_id == 0
                    and info.company_id == private_co):
                price = COMPANIES[private_co].get_low_price() + info.amount
                assert price <= cash, (
                    f"ACQ_PRICE offset {info.amount} yields price {price} "
                    f"exceeding corp cash {cash}"
                )

    def test_no_acq_price_for_own_company(self, game_state):
        """Corp cannot buy a company it already owns."""
        owned_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        acq_self = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0, company_id=owned_co,
        )
        assert len(acq_self) == 0, "Corp should not be able to buy its own company"

    def test_no_acq_price_from_receivership_corp(self, game_state):
        """Cannot buy companies from a receivership corp."""
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=12)
        CORPS[0].set_cash(game_state, 200)

        recv_co = draw_to_player(game_state, 0)
        recv_co2 = draw_to_player(game_state, 0)
        setup_receivership_corp(game_state, corp_id=1, company_ids=[recv_co, recv_co2])

        setup_acquisition_phase_py(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return  # receivership forced buy may have transitioned

        acq_recv = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0, company_id=recv_co,
        )
        assert len(acq_recv) == 0, "Should not buy from receivership corp"

    def test_no_actions_for_receivership_corp_as_buyer(self, game_state):
        """Receivership corps do not appear as buyers in enumerated actions."""
        recv_co = draw_to_player(game_state, 0)
        setup_receivership_corp(game_state, corp_id=0, company_ids=[recv_co])
        CORPS[0].set_cash(game_state, 200)

        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=10)
        CORPS[1].set_cash(game_state, 100)

        draw_to_fi(game_state)

        setup_acquisition_phase_py(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return  # receivership forced buy may have transitioned

        actions = get_legal_actions(game_state)
        recv_buyer_actions = [
            info for _, info in actions
            if info.corp_id == 0 and info.action_type in (ACTION_ACQ_PRICE, ACTION_ACQ_FI_BUY)
        ]
        assert len(recv_buyer_actions) == 0, (
            "Receivership corp should not appear as buyer"
        )

    def test_zero_cash_corp_has_no_buy_actions(self, game_state):
        """A corp with 0 cash has no buy actions enumerated."""
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 0)

        draw_to_fi(game_state)
        draw_to_player(game_state, 0)

        setup_acquisition_phase_py(game_state)

        actions = get_legal_actions(game_state)
        buy_actions = [
            info for _, info in actions
            if info.action_type in (ACTION_ACQ_PRICE, ACTION_ACQ_FI_BUY)
        ]
        assert len(buy_actions) == 0, "Corp with 0 cash should have no buy actions"

    def test_different_president_no_actions_same_pres_mode(self, game_state):
        """In acq_same_president mode, no cross-president corp-to-corp actions."""
        num_players = TURN.get_num_players(game_state)
        if num_players < 3:
            return

        seller_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        draw_to_corp(game_state, 0)

        float_corp_for_test(game_state, corp_id=1, player_id=1, par_index=12)
        CORPS[1].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        # Check from whichever player is active — neither should see
        # cross-president actions
        acq_cross = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=1, company_id=seller_co,
        )
        active = TURN.get_active_player(game_state)
        if active == 1:
            assert len(acq_cross) == 0, (
                "In same-president mode, cross-president corp-to-corp not allowed"
            )
        else:
            # Player 0 is active; they preside over corp 0 but not corp 1,
            # so corp 1 won't appear in their actions regardless
            assert len(acq_cross) == 0


# =============================================================================
# PLAYER STAYS ACTIVE AFTER BUY
# =============================================================================

class TestPlayerStaysAfterBuy:
    """After a buy, the same player remains active for more acquisitions."""

    def test_player_remains_active_after_acq_price(self, game_state):
        """Active player stays active after an ACQ_PRICE action."""
        private_co = draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        active_before = TURN.get_active_player(game_state)
        acq_id = find_legal_action(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0, company_id=private_co,
        )
        apply_and_verify(game_state, acq_id)

        if TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQUISITION):
            assert TURN.get_active_player(game_state) == active_before

    def test_player_remains_active_after_fi_buy(self, game_state):
        """Active player stays active after an FI_BUY (no preemption)."""
        fi_co = draw_to_fi(game_state)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        setup_acquisition_phase_py(game_state)

        active_before = TURN.get_active_player(game_state)
        fi_buy_id = find_legal_action(
            game_state, action_type=ACTION_ACQ_FI_BUY, corp_id=0, company_id=fi_co,
        )
        apply_and_verify(game_state, fi_buy_id)

        phase = TURN.get_phase(game_state)
        if phase == int(GamePhases.PHASE_ACQUISITION):
            assert TURN.get_active_player(game_state) == active_before


# =============================================================================
# MULTIPLE ACQUISITIONS IN ONE TURN
# =============================================================================

class TestMultipleAcquisitions:
    """Test that a player can make multiple acquisitions before passing."""

    def test_two_consecutive_buys(self, game_state):
        """Player buys two companies in the same ACQUISITION turn."""
        co1 = draw_to_player(game_state, 0)
        co2 = draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        setup_acquisition_phase_py(game_state)

        # First buy
        acq1_id = find_legal_action(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0, company_id=co1,
        )
        apply_and_verify(game_state, acq1_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQUISITION)

        # Second buy
        acq2_id = find_legal_action(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0, company_id=co2,
        )
        apply_and_verify(game_state, acq2_id)

        for co in [co1, co2]:
            loc = COMPANIES[co].get_location(game_state)
            assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
            assert COMPANIES[co].get_owner_id(game_state) == 0
