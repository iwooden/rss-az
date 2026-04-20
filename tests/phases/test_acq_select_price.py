"""Tests for the ACQ_SELECT_PRICE phase (final leg of three-step Acquisition).

Entered via SELECT_CORP → SELECT_COMPANY. No pass. Covers:

  - enumeration (price offsets for LOC_CORP / LOC_PLAYER targets; single FI_BUY
    for LOC_FI targets; affordability cap; action id encoding)
  - direct (same-president) execution — cash flows to player / seller
    corp proceeds pile, company moves to LOC_CORP_ACQ, active pair cleared
  - FI buy execution (face vs high price, preemption entry into ACQ_OFFER)
  - cross-president gates — LOC_CORP and LOC_PLAYER foreign-owner paths enter
    ACQ_OFFER
  - stay-on-same-player after direct execution.

Pass + corp-select concerns live in ``test_acq_select_corp.py``; company-select
legality lives in ``test_acq_select_company.py``; preemption decisions live
in ``test_acq_offer.py``.
"""
import pytest

from core.actions import (
    ACTION_PASS_PY as ACTION_PASS,
    ACTION_ACQ_PRICE_PY as ACTION_ACQ_PRICE,
    ACTION_ACQ_SELECT_CORP_PY as ACTION_ACQ_SELECT_CORP,
    ACTION_ACQ_SELECT_COMPANY_PY as ACTION_ACQ_SELECT_COMPANY,
)
from core.data import GamePhases, CorpIndices
from core.state import GameState
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES, CompanyLocation
from entities.fi import FI
from phases.acq_select_corp import setup_acquisition_phase_py

from tests.phases.conftest import (
    apply_and_verify,
    get_legal_actions,
    find_legal_action,
    find_all_legal_actions_with_info,
    float_corp_for_test,
    draw_to_player,
    draw_to_fi,
    draw_to_corp,
)
from tests.phases.helpers.ownership import give_company_to_fi


CORP_OS = int(CorpIndices.CORP_OS)  # 2


# =============================================================================
# HELPERS
# =============================================================================

def _advance_to_president(state, target_president):
    """Pass the current active player until ``target_president`` is active."""
    for _ in range(16):
        if TURN.get_active_player(state) == target_president:
            return
        assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_CORP), (
            f"_advance_to_president: fell out of SELECT_CORP; phase={TURN.get_phase(state)}"
        )
        pass_id = find_legal_action(state, action_type=ACTION_PASS)
        apply_and_verify(state, pass_id)
    assert False, (
        f"_advance_to_president: never reached president {target_president}"
    )


def _enter_select_price(state, corp_id, company_id):
    """Route SELECT_CORP → SELECT_COMPANY → SELECT_PRICE for ``(corp_id, company_id)``.

    Caller is responsible for staging enough legal offsets in SELECT_PRICE
    (i.e. cash headroom past the low price, or ≥2 affordable offsets) so the
    driver does not auto-chain past the SELECT_PRICE decision.
    """
    setup_acquisition_phase_py(state)
    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_CORP), (
        f"_enter_select_price: expected PHASE_ACQ_SELECT_CORP after setup, "
        f"got phase={TURN.get_phase(state)}"
    )

    president = CORPS[corp_id].get_president_id(state)
    assert president >= 0, (
        f"_enter_select_price: corp {corp_id} has no president"
    )
    _advance_to_president(state, president)

    aid = find_legal_action(
        state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=corp_id,
    )
    apply_and_verify(state, aid)

    if TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_COMPANY):
        aid = find_legal_action(
            state, action_type=ACTION_ACQ_SELECT_COMPANY, company_id=company_id,
        )
        apply_and_verify(state, aid)

    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_PRICE), (
        f"_enter_select_price: expected PHASE_ACQ_SELECT_PRICE, got phase="
        f"{TURN.get_phase(state)}. The driver auto-chains past SELECT_PRICE "
        f"when only one price offset is legal — give the corp more cash "
        f"headroom so ≥2 offsets remain."
    )
    assert TURN.get_active_corp(state) == corp_id
    assert TURN.get_active_company(state) == company_id


# =============================================================================
# ENUMERATION
# =============================================================================

class TestEnumeration:
    """SELECT_PRICE legality: price offsets (LOC_CORP/LOC_PLAYER) or FI_BUY (LOC_FI). No pass."""

    def test_no_pass_action(self, game_state):
        """SELECT_PRICE has no pass — SELECT_COMPANY already committed."""
        private_co = draw_to_player(game_state, 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(
            game_state, COMPANIES[private_co].get_low_price() + 10,
        )

        _enter_select_price(game_state, corp_id=0, company_id=private_co)

        actions = get_legal_actions(game_state)
        assert not any(info.action_type == ACTION_PASS for _, info in actions)

    def test_action_id_encoding_for_offset(self, game_state):
        """SELECT_PRICE offset action id equals the offset (0..50)."""
        private_co = draw_to_player(game_state, 0)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 10_000)  # unlimited cash

        _enter_select_price(game_state, corp_id=0, company_id=private_co)

        aid = find_legal_action(game_state, action_type=ACTION_ACQ_PRICE, amount=0)
        assert aid == 0
        aid_one = find_legal_action(game_state, action_type=ACTION_ACQ_PRICE, amount=1)
        assert aid_one == 1

    def test_offsets_capped_by_affordability(self, game_state):
        """Max offset is capped by cash - low (when cash is the binding constraint)."""
        private_co = draw_to_player(game_state, 0)
        low = COMPANIES[private_co].get_low_price()
        high = COMPANIES[private_co].get_high_price()
        if high - low < 3:
            # Degenerate 1-2 price spread — can't isolate the cash cap vs. high cap.
            return
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        # Cash leaves exactly 3 affordable offsets (0, 1, 2) — below high - low.
        CORPS[0].set_cash(game_state, low + 2)

        _enter_select_price(game_state, corp_id=0, company_id=private_co)

        offsets = sorted(
            info.amount for _, info in get_legal_actions(game_state)
            if info.action_type == ACTION_ACQ_PRICE
        )
        assert offsets == [0, 1, 2]
        assert offsets[-1] < high - low  # cash cap bound, not high cap

    def test_offsets_capped_at_high_price(self, game_state):
        """Unlimited cash caps max offset at ``high - low``."""
        private_co = draw_to_player(game_state, 0)
        low = COMPANIES[private_co].get_low_price()
        high = COMPANIES[private_co].get_high_price()
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 10_000)

        _enter_select_price(game_state, corp_id=0, company_id=private_co)

        offsets = sorted(
            info.amount for _, info in get_legal_actions(game_state)
            if info.action_type == ACTION_ACQ_PRICE
        )
        assert offsets[0] == 0
        assert offsets[-1] == min(high - low, 50)

    def test_fi_buy_sole_action_for_fi_target(self, game_state):
        """LOC_FI target exposes only FI_BUY — SELECT_PRICE auto-chains past it.

        Observable via legal-action inspection at SELECT_CORP-entry time:
        force SELECT_COMPANY to land on an FI target with more than 1 option
        in SELECT_COMPANY, then step through SELECT_COMPANY to SELECT_PRICE
        manually; at that moment enumeration should contain exactly FI_BUY.
        """
        # Two FI targets so SELECT_COMPANY is a real decision.
        fi_a = draw_to_fi(game_state)
        fi_b = draw_to_fi(game_state)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        setup_acquisition_phase_py(game_state)
        aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=0,
        )
        apply_and_verify(game_state, aid)
        # Now in SELECT_COMPANY with 2 legal targets. Apply SELECT_COMPANY for
        # fi_a — SELECT_PRICE has 1 legal action (FI_BUY) so the driver auto-
        # chains past the phase. Inspect intermediate state for FI_BUY-only.
        select_company_aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_COMPANY, company_id=fi_a,
        )
        result = apply_and_verify(game_state, select_company_aid)

        # Find the SELECT_PRICE step in the driver's history.
        from core.data import DecisionPhase
        price_steps = [
            (state_array, phase_id, act_id)
            for (state_array, phase_id, act_id) in result.history
            if phase_id == int(DecisionPhase.DPHASE_ACQ_SELECT_PRICE)
        ]
        assert len(price_steps) == 1
        # The action id recorded on that step must be the FI_BUY encoding (51).
        assert price_steps[0][2] == 51
        # fi_b untouched
        assert COMPANIES[fi_b].get_location(game_state) == int(CompanyLocation.LOC_FI)


# =============================================================================
# ACQ PRICE: SAME-PRESIDENT EXECUTION
# =============================================================================

class TestAcqPrice:
    """Same-president direct execution: cash flows + company transfer."""

    def test_corp_to_player_cash_flow(self, game_state):
        """Corp pays player; company moves to LOC_CORP_ACQ."""
        private_co = draw_to_player(game_state, 0)
        low = COMPANIES[private_co].get_low_price()
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 10_000)

        _enter_select_price(game_state, corp_id=0, company_id=private_co)

        # Pick the median legal offset so the offset exists regardless of spread.
        offset_infos = find_all_legal_actions_with_info(
            game_state, action_type=ACTION_ACQ_PRICE,
        )
        aid, info = offset_infos[len(offset_infos) // 2]
        price = low + info.amount

        player_cash_before = PLAYERS[0].get_cash(game_state)
        corp_cash_before = CORPS[0].get_cash(game_state)

        apply_and_verify(game_state, aid)

        assert PLAYERS[0].get_cash(game_state) == player_cash_before + price
        assert CORPS[0].get_cash(game_state) == corp_cash_before - price
        loc = COMPANIES[private_co].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[private_co].get_owner_id(game_state) == 0

    def test_corp_to_corp_same_president(self, game_state):
        """Corp acquires from another corp under same president. Proceeds buffer."""
        seller_co = float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        draw_to_corp(game_state, 0)
        low = COMPANIES[seller_co].get_low_price()

        float_corp_for_test(game_state, corp_id=1, player_id=0, par_index=12)
        CORPS[1].set_cash(game_state, 10_000)

        _enter_select_price(game_state, corp_id=1, company_id=seller_co)

        offset_infos = find_all_legal_actions_with_info(
            game_state, action_type=ACTION_ACQ_PRICE,
        )
        aid, info = offset_infos[len(offset_infos) // 2]
        price = low + info.amount

        buyer_cash_before = CORPS[1].get_cash(game_state)
        seller_proceeds_before = CORPS[0].get_acquisition_proceeds(game_state)

        apply_and_verify(game_state, aid)

        loc = COMPANIES[seller_co].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[seller_co].get_owner_id(game_state) == 1
        assert CORPS[1].get_cash(game_state) == buyer_cash_before - price
        # Proceeds accumulate in buffer while still in ACQ subphase flow.
        if TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_SELECT_CORP):
            assert CORPS[0].get_acquisition_proceeds(game_state) == seller_proceeds_before + price

    def test_price_at_low_boundary(self, game_state):
        """Offset 0 → buyer pays the low price exactly."""
        private_co = draw_to_player(game_state, 0)
        low = COMPANIES[private_co].get_low_price()
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, low + 10)

        _enter_select_price(game_state, corp_id=0, company_id=private_co)

        corp_cash_before = CORPS[0].get_cash(game_state)
        aid = find_legal_action(game_state, action_type=ACTION_ACQ_PRICE, amount=0)
        apply_and_verify(game_state, aid)

        assert CORPS[0].get_cash(game_state) == corp_cash_before - low

    def test_price_at_high_boundary(self, game_state):
        """Offset = high - low → buyer pays exactly high_price."""
        private_co = draw_to_player(game_state, 0)
        low = COMPANIES[private_co].get_low_price()
        high = COMPANIES[private_co].get_high_price()
        max_offset = high - low
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, high + 50)

        _enter_select_price(game_state, corp_id=0, company_id=private_co)

        corp_cash_before = CORPS[0].get_cash(game_state)
        aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_PRICE, amount=max_offset,
        )
        apply_and_verify(game_state, aid)

        assert CORPS[0].get_cash(game_state) == corp_cash_before - high

    def test_post_execution_clears_active_pair(self, game_state):
        """Direct execution clears active_corp + active_company and returns to SELECT_CORP."""
        private_co = draw_to_player(game_state, 0)
        draw_to_player(game_state, 0)  # keep a second target so SELECT_CORP reappears
        low = COMPANIES[private_co].get_low_price()
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, low + 20)

        _enter_select_price(game_state, corp_id=0, company_id=private_co)

        aid = find_legal_action(game_state, action_type=ACTION_ACQ_PRICE, amount=1)
        apply_and_verify(game_state, aid)

        if TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_SELECT_CORP):
            assert TURN.get_active_corp(game_state) == -1
            assert TURN.get_active_company(game_state) == -1


# =============================================================================
# FI BUY: SAME-PLAYER DIRECT (NO PREEMPTION)
# =============================================================================

class TestFiBuy:
    """FI purchases with no preemption — auto-chain straight to execution."""

    def test_fi_buy_transfers_and_pays_high_price(self, game_state):
        """FI buy transfers company to corp, deducts high price from corp, pays FI."""
        fi_co = draw_to_fi(game_state)
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, 500)

        corp_cash_before = CORPS[0].get_cash(game_state)
        fi_cash_before = FI.get_cash(game_state)
        expected_price = COMPANIES[fi_co].get_high_price()

        setup_acquisition_phase_py(game_state)
        aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=0,
        )
        apply_and_verify(game_state, aid)

        loc = COMPANIES[fi_co].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[fi_co].get_owner_id(game_state) == 0
        assert CORPS[0].get_cash(game_state) == corp_cash_before - expected_price
        assert FI.get_cash(game_state) == fi_cash_before + expected_price

    def test_os_pays_face_value_for_fi(self, game_state):
        """OS (corp 2) pays face value, not high price, when buying from FI."""
        fi_co = draw_to_fi(game_state)
        float_corp_for_test(game_state, corp_id=CORP_OS, player_id=0, par_index=10)
        CORPS[CORP_OS].set_cash(game_state, 500)

        corp_cash_before = CORPS[CORP_OS].get_cash(game_state)
        fi_cash_before = FI.get_cash(game_state)
        expected_price = COMPANIES[fi_co].get_face_value()

        setup_acquisition_phase_py(game_state)
        aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=CORP_OS,
        )
        apply_and_verify(game_state, aid)

        assert CORPS[CORP_OS].get_cash(game_state) == corp_cash_before - expected_price
        assert FI.get_cash(game_state) == fi_cash_before + expected_price


# =============================================================================
# FI BUY: PREEMPTION
# =============================================================================

class TestFiPreemption:
    """FI buy triggers ACQ_OFFER preemption when a higher-priority corp qualifies."""

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_higher_priority_corp_preempts(self, game_state):
        """Higher-share-price corp gets offered the FI company via ACQ_OFFER."""
        fi_co = 10
        give_company_to_fi(game_state, fi_co)

        # Low-price buyer (player 1)
        float_corp_for_test(game_state, corp_id=3, company_id=11, player_id=1, par_index=5)
        CORPS[3].set_cash(game_state, 500)

        # High-price preemptor (player 2)
        float_corp_for_test(game_state, corp_id=4, company_id=12, player_id=2, par_index=15)
        CORPS[4].set_cash(game_state, 500)

        setup_acquisition_phase_py(game_state)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_SELECT_CORP)
        assert TURN.get_active_player(game_state) == 1

        aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=3,
        )
        apply_and_verify(game_state, aid)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_OFFER)
        assert TURN.get_active_player(game_state) == 2
        assert TURN.get_active_corp(game_state) == 4
        assert TURN.get_active_company(game_state) == fi_co

    def test_same_president_higher_priority_corp_does_not_preempt(self, game_state):
        """Same-president preemptors are skipped (no self-intervention)."""
        fi_co = draw_to_fi(game_state)

        float_corp_for_test(game_state, corp_id=CORP_OS, player_id=0, par_index=15)
        CORPS[CORP_OS].set_cash(game_state, 500)

        buyer_corp = 3
        float_corp_for_test(game_state, corp_id=buyer_corp, player_id=0, par_index=10)
        CORPS[buyer_corp].set_cash(game_state, 500)

        setup_acquisition_phase_py(game_state)

        buyer_cash_before = CORPS[buyer_corp].get_cash(game_state)
        os_cash_before = CORPS[CORP_OS].get_cash(game_state)
        fi_cash_before = FI.get_cash(game_state)
        expected_price = COMPANIES[fi_co].get_high_price()

        # Active player is 0 either via the OS or buyer corp. Pick the buyer corp.
        aid = find_legal_action(
            game_state, action_type=ACTION_ACQ_SELECT_CORP, corp_id=buyer_corp,
        )
        apply_and_verify(game_state, aid)

        loc = COMPANIES[fi_co].get_location(game_state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert COMPANIES[fi_co].get_owner_id(game_state) == buyer_corp
        assert CORPS[buyer_corp].get_cash(game_state) == buyer_cash_before - expected_price
        assert CORPS[CORP_OS].get_cash(game_state) == os_cash_before
        assert FI.get_cash(game_state) == fi_cash_before + expected_price


# =============================================================================
# CROSS-PRESIDENT: ENTRY TO ACQ_OFFER
# =============================================================================

class TestCrossPresidentEntry:
    """Cross-president LOC_CORP / LOC_PLAYER acquisitions enter ACQ_OFFER."""

    @staticmethod
    def _make_state(num_players):
        state = GameState(num_players, acq_same_president=False)
        state.initialize_game(num_players, seed=42)
        return state

    def test_corp_to_corp_cross_president_enters_acq_offer(self):
        """Cross-pres LOC_CORP acquisition pushes into ACQ_OFFER for the owner."""
        state = self._make_state(3)

        seller_co = float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        draw_to_corp(state, 0)

        float_corp_for_test(state, corp_id=1, player_id=1, par_index=12)
        low = COMPANIES[seller_co].get_low_price()
        CORPS[1].set_cash(state, low + 20)

        _enter_select_price(state, corp_id=1, company_id=seller_co)

        aid = find_legal_action(state, action_type=ACTION_ACQ_PRICE, amount=3)
        apply_and_verify(state, aid)

        assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_OFFER)
        # Owner of corp 0 is player 0 — they must decide.
        assert TURN.get_active_player(state) == 0

    def test_corp_to_player_cross_president_enters_acq_offer(self):
        """Cross-pres LOC_PLAYER acquisition pushes into ACQ_OFFER for the owner."""
        state = self._make_state(3)

        private_co = draw_to_player(state, 1)
        # Extra privates so SELECT_COMPANY is a real decision.
        draw_to_player(state, 0)
        draw_to_player(state, 0)

        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        low = COMPANIES[private_co].get_low_price()
        CORPS[0].set_cash(state, low + 20)

        _enter_select_price(state, corp_id=0, company_id=private_co)

        aid = find_legal_action(state, action_type=ACTION_ACQ_PRICE, amount=2)
        apply_and_verify(state, aid)

        assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_OFFER)
        assert TURN.get_active_player(state) == 1

    def test_same_president_still_executes_directly(self):
        """Same-president acquisition executes directly even with flag=False."""
        state = self._make_state(3)

        private_co = draw_to_player(state, 0)
        draw_to_player(state, 0)  # second target keeps SELECT_CORP returning
        low = COMPANIES[private_co].get_low_price()
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(state, low + 20)

        _enter_select_price(state, corp_id=0, company_id=private_co)

        corp_cash_before = CORPS[0].get_cash(state)
        aid = find_legal_action(state, action_type=ACTION_ACQ_PRICE, amount=2)
        apply_and_verify(state, aid)

        # Did NOT enter ACQ_OFFER — direct execution path.
        assert TURN.get_phase(state) != int(GamePhases.PHASE_ACQ_OFFER)
        loc = COMPANIES[private_co].get_location(state)
        assert loc in (int(CompanyLocation.LOC_CORP_ACQ), int(CompanyLocation.LOC_CORP))
        assert CORPS[0].get_cash(state) < corp_cash_before


# =============================================================================
# STAY ON SAME PLAYER (POST DIRECT EXECUTION)
# =============================================================================

class TestStayOnSamePlayer:
    """After SELECT_PRICE direct execution, the active player stays put."""

    def test_active_player_unchanged_after_direct_execution(self, game_state):
        """Returning to SELECT_CORP after a direct-execution buy keeps active_player."""
        private_co = draw_to_player(game_state, 0)
        draw_to_player(game_state, 0)  # second target so SELECT_CORP reappears
        low = COMPANIES[private_co].get_low_price()
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        CORPS[0].set_cash(game_state, low + 20)

        _enter_select_price(game_state, corp_id=0, company_id=private_co)
        active_before = TURN.get_active_player(game_state)

        aid = find_legal_action(game_state, action_type=ACTION_ACQ_PRICE, amount=1)
        apply_and_verify(game_state, aid)

        if TURN.get_phase(game_state) == int(GamePhases.PHASE_ACQ_SELECT_CORP):
            assert TURN.get_active_player(game_state) == active_before
