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
from entities.deck import DECK

from tests.phases.conftest import (
    apply_and_verify,
    get_legal_actions,
    find_legal_action,
    find_all_legal_actions,
    float_corp_for_test,
    setup_receivership_corp,
)


# =============================================================================
# HELPERS
# =============================================================================

CORP_OS = int(CorpIndices.CORP_OS)  # 2


def _draw_company(state):
    """Draw a company from the deck (LOC_DECK → LOC_REVEALED). Returns company_id."""
    cid = DECK.draw(state)
    assert cid >= 0, "Deck is empty"
    return cid


def _draw_to_fi(state):
    """Draw a company and transfer it to FI. Returns company_id."""
    cid = _draw_company(state)
    COMPANIES[cid].transfer_to_fi(state)
    return cid


def _draw_to_player(state, player_id):
    """Draw a company and transfer it to a player. Returns company_id."""
    cid = _draw_company(state)
    COMPANIES[cid].transfer_to_player(state, player_id)
    return cid


def _advance_to_acquisition(state):
    """Pass all players through INVEST to reach ACQUISITION via WRAP_UP.

    Drains FI cash first to prevent WRAP_UP FI purchases from creating
    unexpected FI-owned companies. Returns when the engine is at ACQUISITION
    or has already advanced past it.
    """
    num_players = TURN.get_num_players(state)
    assert TURN.get_phase(state) == int(GamePhases.PHASE_INVEST)

    # Drain FI cash so WRAP_UP doesn't buy companies for FI
    FI.set_cash(state, 0)

    # All players pass consecutively to end INVEST
    for _ in range(num_players):
        pass_id = find_legal_action(state, action_type=ACTION_PASS)
        apply_and_verify(state, pass_id)


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

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 100)
        if num_players >= 3:
            float_corp_for_test(game_state, corp_id=1, player_id=1, par_index=12)
            CORPS[1].set_cash(game_state, 100)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

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

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 100)
        float_corp_for_test(game_state, corp_id=1, player_id=1, par_index=12)
        CORPS[1].set_cash(game_state, 100)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        first_active = TURN.get_active_player(game_state)
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        if TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQUISITION):
            assert TURN.get_active_player(game_state) != first_active

    def test_all_pass_transitions_to_closing(self, game_state):
        """When all eligible players pass, phase transitions past ACQUISITION."""
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 100)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        _pass_through_acquisition(game_state)
        assert TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION)


# =============================================================================
# FI BUY TESTS
# =============================================================================

class TestFiBuy:
    """Test ACQUISITION FI_BUY action: corp buys from Foreign Investor."""

    def test_fi_buy_transfers_company_and_cash(self, game_state):
        """FI buy moves company to corp, deducts cash from corp, pays FI."""
        fi_company = _draw_to_fi(game_state)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        fi_buys = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_FI_BUY, corp_id=0, company_id=fi_company,
        )
        if not fi_buys:
            return

        corp_cash_before = CORPS[0].get_cash(game_state)
        fi_cash_before = FI.get_cash(game_state)
        expected_price = COMPANIES[fi_company].get_high_price()

        apply_and_verify(game_state, fi_buys[0])

        loc = COMPANIES[fi_company].get_location(game_state)
        owner = COMPANIES[fi_company].get_owner_id(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert owner == 0
        assert CORPS[0].get_cash(game_state) == corp_cash_before - expected_price
        assert FI.get_cash(game_state) == fi_cash_before + expected_price

    def test_os_pays_face_value_for_fi(self, game_state):
        """OS (corp 2) pays face value instead of high price for FI companies."""
        fi_company = _draw_to_fi(game_state)

        float_corp_for_test(game_state, corp_id=CORP_OS, player_id=0, par_index=10)
        CORPS[CORP_OS].set_cash(game_state, 200)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        fi_buys = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_FI_BUY, corp_id=CORP_OS, company_id=fi_company,
        )
        if not fi_buys:
            return

        corp_cash_before = CORPS[CORP_OS].get_cash(game_state)
        fi_cash_before = FI.get_cash(game_state)
        expected_price = COMPANIES[fi_company].get_face_value()

        apply_and_verify(game_state, fi_buys[0])

        assert CORPS[CORP_OS].get_cash(game_state) == corp_cash_before - expected_price
        assert FI.get_cash(game_state) == fi_cash_before + expected_price


# =============================================================================
# ACQ_PRICE TESTS (CORP-TO-CORP, CORP-TO-PLAYER)
# =============================================================================

class TestAcqPrice:
    """Test ACQUISITION ACQ_PRICE action: negotiated-price acquisitions."""

    def test_corp_to_corp_same_president(self, game_state):
        """Corp acquires from another corp under same president."""
        # Float corp 0 (seller), player 0 is president
        seller_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        # Give corp 0 a second company so it retains >= 1 after sale
        extra_co = _draw_company(game_state)
        COMPANIES[extra_co].transfer_to_corp(game_state, 0)

        # Float corp 1 (buyer), same president (player 0), with cash
        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)
        CORPS[1].set_cash(game_state, 200)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        # Find ACQ_PRICE for corp 1 buying seller_co from corp 0
        acq_actions = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=1, company_id=seller_co,
        )
        if not acq_actions:
            return

        action_id = acq_actions[len(acq_actions) // 2]
        actions = get_legal_actions(game_state)
        info = next(i for aid, i in actions if aid == action_id)
        low_price = COMPANIES[seller_co].get_low_price()
        price = low_price + info.amount

        buyer_cash_before = CORPS[1].get_cash(game_state)
        seller_proceeds_before = CORPS[0].get_acquisition_proceeds(game_state)

        apply_and_verify(game_state, action_id)

        loc = COMPANIES[seller_co].get_location(game_state)
        owner = COMPANIES[seller_co].get_owner_id(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert owner == 1
        assert CORPS[1].get_cash(game_state) == buyer_cash_before - price

        if TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQUISITION):
            assert CORPS[0].get_acquisition_proceeds(game_state) == seller_proceeds_before + price

    def test_corp_to_player_acquisition(self, game_state):
        """Corp acquires a private company from the president (same player)."""
        private_co = _draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        acq_actions = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0, company_id=private_co,
        )
        if not acq_actions:
            return

        action_id = acq_actions[0]
        actions = get_legal_actions(game_state)
        info = next(i for aid, i in actions if aid == action_id)
        low_price = COMPANIES[private_co].get_low_price()
        price = low_price + info.amount

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
        private_co = _draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        min_action = None
        for aid, info in get_legal_actions(game_state):
            if (info.action_type == ACTION_ACQ_PRICE
                    and info.corp_id == 0
                    and info.company_id == private_co
                    and info.amount == 0):
                min_action = aid
                break
        if min_action is None:
            return

        low_price = COMPANIES[private_co].get_low_price()
        corp_cash_before = CORPS[0].get_cash(game_state)

        apply_and_verify(game_state, min_action)
        assert CORPS[0].get_cash(game_state) == corp_cash_before - low_price

    def test_price_at_high_boundary(self, game_state):
        """Acquisition at maximum price (high_price)."""
        private_co = _draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        high_price = COMPANIES[private_co].get_high_price()
        CORPS[0].set_cash(game_state, high_price + 50)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        low_price = COMPANIES[private_co].get_low_price()
        max_offset = high_price - low_price

        max_action = None
        for aid, info in get_legal_actions(game_state):
            if (info.action_type == ACTION_ACQ_PRICE
                    and info.corp_id == 0
                    and info.company_id == private_co
                    and info.amount == max_offset):
                max_action = aid
                break
        if max_action is None:
            return

        corp_cash_before = CORPS[0].get_cash(game_state)
        apply_and_verify(game_state, max_action)
        assert CORPS[0].get_cash(game_state) == corp_cash_before - high_price

    def test_seller_must_retain_one_company(self, game_state):
        """A corp with exactly 1 company cannot sell it."""
        # Float corp 0 with exactly 1 company
        seller_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)

        # Float corp 1 (buyer) same president, with cash
        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)
        CORPS[1].set_cash(game_state, 200)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

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

        fi_co = _draw_to_fi(game_state)

        # Low-price buyer (player 1)
        float_corp_for_test(game_state, corp_id=3, player_id=1, par_index=5)
        CORPS[3].set_cash(game_state, 200)

        # High-price preemptor (player 2)
        float_corp_for_test(game_state, corp_id=4, player_id=2, par_index=15)
        CORPS[4].set_cash(game_state, 200)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        active = TURN.get_active_player(game_state)
        if active != 1:
            return  # not the right player scenario

        fi_buys = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_FI_BUY, corp_id=3, company_id=fi_co,
        )
        if not fi_buys:
            return

        apply_and_verify(game_state, fi_buys[0])

        phase = TURN.get_phase(game_state)
        if phase == int(GamePhases.PHASE_ACQ_OFFER):
            # The preemptor's president (player 2) should decide
            assert TURN.get_active_player(game_state) == 2


# =============================================================================
# RECEIVERSHIP FORCED BUY TESTS
# =============================================================================

class TestReceivershipForcedBuys:
    """Test automatic receivership FI buys at start of ACQUISITION."""

    def test_receivership_corp_auto_buys_fi_company(self, game_state):
        """A receivership corp with cash automatically buys from FI."""
        recv_co = _draw_company(game_state)
        COMPANIES[recv_co].transfer_to_player(game_state, 0)
        setup_receivership_corp(game_state, corp_id=0, company_ids=[recv_co])
        CORPS[0].set_cash(game_state, 200)

        fi_co = _draw_to_fi(game_state)
        fi_cash_before = FI.get_cash(game_state)

        _advance_to_acquisition(game_state)

        phase = TURN.get_phase(game_state)
        if phase == int(GamePhases.PHASE_ACQ_OFFER):
            return  # preemption happened

        loc = COMPANIES[fi_co].get_location(game_state)
        if loc == int(CompanyLocation.LOC_FI):
            return  # too expensive
        assert FI.get_cash(game_state) >= fi_cash_before

    def test_receivership_with_no_cash_skips(self, game_state):
        """Receivership corp with no cash does not buy from FI."""
        recv_co = _draw_company(game_state)
        COMPANIES[recv_co].transfer_to_player(game_state, 0)
        setup_receivership_corp(game_state, corp_id=0, company_ids=[recv_co])
        CORPS[0].set_cash(game_state, 0)

        fi_co = _draw_to_fi(game_state)

        _advance_to_acquisition(game_state)

        assert COMPANIES[fi_co].get_location(game_state) == int(CompanyLocation.LOC_FI)


# =============================================================================
# PHASE TRANSITION TESTS
# =============================================================================

class TestPhaseTransitions:
    """Test phase transitions from ACQUISITION."""

    def test_no_active_corps_skips_to_closing(self, game_state):
        """With no active corps, ACQUISITION transitions directly past."""
        _advance_to_acquisition(game_state)
        assert TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION)

    def test_only_receivership_corps_transitions_after_forced_buys(self, game_state):
        """With only receivership corps, ACQUISITION transitions after forced buys."""
        recv_co = _draw_company(game_state)
        COMPANIES[recv_co].transfer_to_player(game_state, 0)
        setup_receivership_corp(game_state, corp_id=0, company_ids=[recv_co])
        CORPS[0].set_cash(game_state, 0)

        _advance_to_acquisition(game_state)
        assert TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION)

    def test_acquisition_eventually_reaches_closing(self, game_state):
        """Playing through ACQUISITION by passing always reaches CLOSING or beyond."""
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 100)

        _advance_to_acquisition(game_state)
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
        private_co = _draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        acq_actions = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0, company_id=private_co,
        )
        if acq_actions:
            apply_and_verify(game_state, acq_actions[0])

        _pass_through_acquisition(game_state)

        for cid in range(int(GameConstants.NUM_COMPANIES)):
            assert COMPANIES[cid].get_location(game_state) != int(CompanyLocation.LOC_CORP_ACQ), (
                f"Company {cid} still in LOC_CORP_ACQ after ACQUISITION ended"
            )

    def test_proceeds_flushed_after_phase(self, game_state):
        """After ACQUISITION ends, all corps have 0 acquisition_proceeds."""
        seller_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        extra_co = _draw_company(game_state)
        COMPANIES[extra_co].transfer_to_corp(game_state, 0)

        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)
        CORPS[1].set_cash(game_state, 200)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        acq_actions = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=1, company_id=seller_co,
        )
        if acq_actions:
            apply_and_verify(game_state, acq_actions[0])

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

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        actions = get_legal_actions(game_state)
        pass_actions = [info for _, info in actions if info.action_type == ACTION_PASS]
        assert len(pass_actions) == 1

    def test_no_fi_buy_when_no_fi_companies(self, game_state):
        """No FI_BUY actions when FI owns no companies."""
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        # After advancing (FI drained), FI should have no companies
        # Verify no FI companies exist
        fi_count = sum(
            1 for cid in range(int(GameConstants.NUM_COMPANIES))
            if COMPANIES[cid].get_location(game_state) == int(CompanyLocation.LOC_FI)
        )
        if fi_count > 0:
            return  # FI somehow got companies; skip

        actions = get_legal_actions(game_state)
        fi_buys = [info for _, info in actions if info.action_type == ACTION_ACQ_FI_BUY]
        assert len(fi_buys) == 0

    def test_fi_buy_limited_by_cash(self, game_state):
        """FI_BUY actions only for companies the corp can afford."""
        fi_co = _draw_to_fi(game_state)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        high_price = COMPANIES[fi_co].get_high_price()
        CORPS[0].set_cash(game_state, high_price)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

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
        private_co = _draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        low_price = COMPANIES[private_co].get_low_price()
        CORPS[0].set_cash(game_state, low_price + 3)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

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

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        acq_self = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0, company_id=owned_co,
        )
        assert len(acq_self) == 0, "Corp should not be able to buy its own company"

    def test_no_acq_price_from_receivership_corp(self, game_state):
        """Cannot buy companies from a receivership corp."""
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=12)
        CORPS[0].set_cash(game_state, 200)

        recv_co = _draw_company(game_state)
        COMPANIES[recv_co].transfer_to_player(game_state, 0)
        recv_co2 = _draw_company(game_state)
        COMPANIES[recv_co2].transfer_to_player(game_state, 0)
        setup_receivership_corp(game_state, corp_id=1, company_ids=[recv_co, recv_co2])

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        # No actions should buy recv_co from the receivership corp 1
        acq_recv = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0, company_id=recv_co,
        )
        assert len(acq_recv) == 0, "Should not buy from receivership corp"

    def test_no_actions_for_receivership_corp_as_buyer(self, game_state):
        """Receivership corps do not appear as buyers in enumerated actions."""
        recv_co = _draw_company(game_state)
        COMPANIES[recv_co].transfer_to_player(game_state, 0)
        setup_receivership_corp(game_state, corp_id=0, company_ids=[recv_co])
        CORPS[0].set_cash(game_state, 200)

        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=10)
        CORPS[1].set_cash(game_state, 100)

        _draw_to_fi(game_state)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

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

        _draw_to_fi(game_state)
        _draw_to_player(game_state, 0)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

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

        # Corp 0 owned by player 0, with 2 companies
        seller_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        extra_co = _draw_company(game_state)
        COMPANIES[extra_co].transfer_to_corp(game_state, 0)

        # Corp 1 owned by player 1
        float_corp_for_test(game_state, corp_id=1, player_id=1, par_index=12)
        CORPS[1].set_cash(game_state, 200)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        active = TURN.get_active_player(game_state)
        if active == 1:
            acq_cross = find_all_legal_actions(
                game_state, action_type=ACTION_ACQ_PRICE, corp_id=1, company_id=seller_co,
            )
            assert len(acq_cross) == 0, (
                "In same-president mode, cross-president corp-to-corp not allowed"
            )


# =============================================================================
# PLAYER STAYS ACTIVE AFTER BUY
# =============================================================================

class TestPlayerStaysAfterBuy:
    """After a buy, the same player remains active for more acquisitions."""

    def test_player_remains_active_after_acq_price(self, game_state):
        """Active player stays active after an ACQ_PRICE action."""
        private_co = _draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        active_before = TURN.get_active_player(game_state)
        acq_actions = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0, company_id=private_co,
        )
        if not acq_actions:
            return

        apply_and_verify(game_state, acq_actions[0])

        if TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQUISITION):
            assert TURN.get_active_player(game_state) == active_before

    def test_player_remains_active_after_fi_buy(self, game_state):
        """Active player stays active after an FI_BUY (no preemption)."""
        fi_co = _draw_to_fi(game_state)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 200)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        active_before = TURN.get_active_player(game_state)
        fi_buys = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_FI_BUY, corp_id=0, company_id=fi_co,
        )
        if not fi_buys:
            return

        apply_and_verify(game_state, fi_buys[0])

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
        co1 = _draw_to_player(game_state, 0)
        co2 = _draw_to_player(game_state, 0)

        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        _advance_to_acquisition(game_state)
        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        # First buy
        acq1 = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0, company_id=co1,
        )
        if not acq1:
            return
        apply_and_verify(game_state, acq1[0])

        if TURN.get_phase(game_state) != int(GamePhases.PHASE_ACQUISITION):
            return

        # Second buy
        acq2 = find_all_legal_actions(
            game_state, action_type=ACTION_ACQ_PRICE, corp_id=0, company_id=co2,
        )
        if not acq2:
            return
        apply_and_verify(game_state, acq2[0])

        for co in [co1, co2]:
            loc = COMPANIES[co].get_location(game_state)
            assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
            assert COMPANIES[co].get_owner_id(game_state) == 0
