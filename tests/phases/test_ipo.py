"""Tests for the IPO phase (corp-select half of Form Corporation).

Covers: setup, per-company remaining flags, descending face-value processing
order, pass semantics, corp-select legality (inactive-only, any-affordable-par
gate, market / star-tier / affordability filters applied via ``_any_par_affordable``),
and the IPO→PAR transition (active_corp set, phase=PHASE_PAR). Float execution
and PAR legality live in ``test_par.py``.
"""
from core.actions import (
    ACTION_PASS_PY as ACTION_PASS,
    ACTION_IPO_PY as ACTION_IPO,
)
from core.data import GamePhases, GameConstants
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES, CompanyLocation
from entities.market import MARKET
from phases.ipo import setup_ipo_phase_py

from tests.phases.conftest import (
    apply_and_verify,
    get_legal_actions,
    find_legal_action,
    float_corp_for_test,
)
from tests.phases.helpers.finance import set_player_cashs
from tests.phases.helpers.ownership import give_company_to_player


# =============================================================================
# HELPERS
# =============================================================================

# Companies used across tests:
CO_1S_A = 0    # BME (star 1, FV 1)
CO_1S_B = 1    # BSE (star 1, FV 2)
CO_2S = 6      # first orange (star 2, FV 11)
CO_3S = 14     # first yellow (star 3, FV 20)
CO_4S = 22     # first green  (star 4, FV 30)
CO_5S = 35     # CDG (star 5, FV 60)


def _give_company(state, company_id, player_id=0):
    """Transfer a company into LOC_PLAYER regardless of current location."""
    give_company_to_player(state, company_id, player_id)


def _enter_ipo(state, owners, cash_by_player=None):
    """Prepare player-owned companies and enter PHASE_IPO.

    Args:
        state: GameState
        owners: dict mapping company_id → player_id
        cash_by_player: dict mapping player_id → cash (defaults to $100 each)
    """
    for cid, pid in owners.items():
        _give_company(state, cid, pid)
    if cash_by_player is None:
        cash_by_player = {pid: 100 for pid in set(owners.values())}
    set_player_cashs(state, cash_by_player)
    setup_ipo_phase_py(state)


# =============================================================================
# ENUMERATION
# =============================================================================

class TestEnumeration:
    """Legal actions: pass + (inactive corp × at-least-one-affordable-par)."""

    def test_pass_always_legal(self, game_state):
        """PASS is legal in IPO whenever a company is being offered."""
        _enter_ipo(game_state, {CO_3S: 0})
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        assert pass_id == 0

    def test_all_inactive_corps_offered(self, game_state):
        """Every inactive corp shows up exactly once — par choice is in PAR."""
        _enter_ipo(game_state, {CO_3S: 0}, {0: 1000})
        ipo_actions = [info for _, info in get_legal_actions(game_state)
                       if info.action_type == ACTION_IPO]
        # 8 corps × 1 action each (single corp-select logit).
        assert len(ipo_actions) == int(GameConstants.NUM_CORPS)
        corp_ids = {info.corp_id for info in ipo_actions}
        assert corp_ids == set(range(int(GameConstants.NUM_CORPS)))

    def test_action_id_encoding(self, game_state):
        """Action id for IPO corp-select is ``1 + corp_id``."""
        _enter_ipo(game_state, {CO_3S: 0}, {0: 1000})
        aid = find_legal_action(game_state, action_type=ACTION_IPO, corp_id=3)
        assert aid == 1 + 3

    def test_active_corp_excluded(self, game_state):
        """An already-floated corp is not a valid IPO target."""
        # Float corp 0 with a different company first.
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1S_A, par_index=MARKET.get_index_for_price(10))
        _enter_ipo(game_state, {CO_3S: 0}, {0: 1000})
        ipo_actions = [info for _, info in get_legal_actions(game_state)
                       if info.action_type == ACTION_IPO]
        corp_ids = {info.corp_id for info in ipo_actions}
        assert 0 not in corp_ids
        assert len(corp_ids) == int(GameConstants.NUM_CORPS) - 1

    def test_corp_excluded_when_no_par_affordable(self, game_state):
        """Corp hidden from IPO if player can't afford *any* valid par price."""
        # CO_3S (FV=20): cheapest valid par is 20 (cost 0). With $0, par=20 is
        # affordable (cost 0). Use CO_4S (FV=30): cheapest affordable par costs
        # more than $0 for all pars except equality — par=30 costs 0.
        # So we need a company where every par costs > 0. Use CO_3S but charge
        # $-1 — not possible. Instead, block the $0-cost par market slot.
        _enter_ipo(game_state, {CO_3S: 0}, {0: 0})
        # CO_3S: FV=20 so par=20 has cost 0 (the sole affordable par at $0).
        MARKET.set_space_available(game_state, MARKET.get_index_for_price(20), False)
        actions = get_legal_actions(game_state)
        # No corp-select should be legal — only PASS.
        assert len(actions) == 1
        assert actions[0][1].action_type == ACTION_PASS

    def test_corp_included_when_any_par_affordable(self, game_state):
        """Corp remains legal if at least one par satisfies the triple gate."""
        # Player has $1: CO_3S (FV=20) par=20 is cost 0 (affordable), others
        # cost >= 2 (not affordable) but corp should still be offered.
        _enter_ipo(game_state, {CO_3S: 0}, {0: 1})
        ipo_actions = [info for _, info in get_legal_actions(game_state)
                       if info.action_type == ACTION_IPO]
        assert len(ipo_actions) == int(GameConstants.NUM_CORPS)

    def test_no_inactive_corps_only_pass(self, game_state):
        """If every corp is active, only PASS remains."""
        _enter_ipo(game_state, {CO_3S: 0}, {0: 1000})
        # Float all 8 corps against distinct throwaway companies at distinct
        # market indices (5..12), so each float claims a fresh slot.
        throwaway = [CO_1S_A, CO_1S_B, 2, 3, 4, 5, CO_2S, 7]
        for corp_id, cid in enumerate(throwaway):
            float_corp_for_test(game_state, corp_id=corp_id, player_id=0,
                                company_id=cid, par_index=5 + corp_id)
        # The IPO company CO_3S is still player-owned after all floats.
        assert COMPANIES[CO_3S].get_location(game_state) == CompanyLocation.LOC_PLAYER
        actions = get_legal_actions(game_state)
        assert len(actions) == 1
        assert actions[0][1].action_type == ACTION_PASS


# =============================================================================
# PASS ACTION
# =============================================================================

class TestPassAction:
    """Passing skips IPO for the active company and advances."""

    def test_pass_clears_ipo_remaining(self, game_state):
        """Pass marks the active company as no-longer-remaining."""
        _enter_ipo(game_state, {CO_3S: 0})
        assert TURN.is_ipo_remaining(game_state, CO_3S)

        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert not TURN.is_ipo_remaining(game_state, CO_3S)

    def test_pass_leaves_company_with_player(self, game_state):
        """Pass does not remove the company from the player."""
        _enter_ipo(game_state, {CO_3S: 0})
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert COMPANIES[CO_3S].get_location(game_state) == CompanyLocation.LOC_PLAYER
        assert COMPANIES[CO_3S].get_owner_id(game_state) == 0

    def test_pass_does_not_activate_corp(self, game_state):
        """Pass does not float any corporation."""
        _enter_ipo(game_state, {CO_3S: 0})
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        for corp_id in range(int(GameConstants.NUM_CORPS)):
            assert not CORPS[corp_id].is_active(game_state)

    def test_pass_does_not_change_player_cash(self, game_state):
        """Pass leaves player cash untouched."""
        _enter_ipo(game_state, {CO_3S: 0})
        cash_before = PLAYERS[0].get_cash(game_state)
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert PLAYERS[0].get_cash(game_state) == cash_before


# =============================================================================
# IPO → PAR TRANSITION
# =============================================================================

class TestIpoToParTransition:
    """Corp-select advances to PHASE_PAR with active_corp seeded."""

    def test_corp_select_sets_active_corp(self, game_state):
        """Applying an IPO corp-select seeds active_corp with the chosen id."""
        _enter_ipo(game_state, {CO_3S: 0})
        aid = find_legal_action(game_state, action_type=ACTION_IPO, corp_id=3)
        apply_and_verify(game_state, aid)

        assert TURN.get_active_corp(game_state) == 3

    def test_corp_select_switches_to_par_phase(self, game_state):
        """Applying an IPO corp-select transitions to PHASE_PAR."""
        _enter_ipo(game_state, {CO_3S: 0})
        aid = find_legal_action(game_state, action_type=ACTION_IPO, corp_id=0)
        apply_and_verify(game_state, aid)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_PAR)

    def test_corp_select_keeps_active_company(self, game_state):
        """active_company is preserved across the IPO→PAR handoff."""
        _enter_ipo(game_state, {CO_3S: 0})
        aid = find_legal_action(game_state, action_type=ACTION_IPO, corp_id=0)
        apply_and_verify(game_state, aid)

        assert TURN.get_active_company(game_state) == CO_3S

    def test_corp_select_does_not_float_yet(self, game_state):
        """IPO corp-select is pure state machine — no cash / share movement."""
        _enter_ipo(game_state, {CO_3S: 0}, {0: 100})
        aid = find_legal_action(game_state, action_type=ACTION_IPO, corp_id=0)
        apply_and_verify(game_state, aid)

        # Corp still inactive, player still holds the company, no cash moved.
        assert not CORPS[0].is_active(game_state)
        assert COMPANIES[CO_3S].get_location(game_state) == CompanyLocation.LOC_PLAYER
        assert PLAYERS[0].get_cash(game_state) == 100


# =============================================================================
# PROCESSING ORDER
# =============================================================================

class TestProcessingOrder:
    """Companies are processed in descending face-value order."""

    def test_highest_face_value_first(self, game_state):
        """Among several player-owned companies, the highest FV is offered first."""
        _enter_ipo(game_state, {CO_1S_A: 0, CO_3S: 0, CO_5S: 0}, {0: 200})
        assert TURN.get_active_company(game_state) == CO_5S  # FV=60

    def test_active_player_matches_owner(self, game_state):
        """Active player is always the owner of the currently active company."""
        _enter_ipo(game_state, {CO_5S: 0, CO_3S: 1}, {0: 200, 1: 200})

        # FV=60 first, owned by player 0.
        assert TURN.get_active_company(game_state) == CO_5S
        assert TURN.get_active_player(game_state) == 0

        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        # FV=20 second, owned by player 1.
        assert TURN.get_active_company(game_state) == CO_3S
        assert TURN.get_active_player(game_state) == 1

    def test_descending_order_three_companies(self, game_state):
        """Three companies with distinct face values pop in descending order."""
        # CO_5S (60) > CO_3S (20) > CO_1S_A (1)
        _enter_ipo(game_state, {CO_5S: 0, CO_3S: 0, CO_1S_A: 0}, {0: 200})

        expected_order = [CO_5S, CO_3S, CO_1S_A]
        for expected in expected_order:
            assert TURN.get_phase(game_state) == int(GamePhases.PHASE_IPO)
            assert TURN.get_active_company(game_state) == expected
            pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, pass_id)


# =============================================================================
# PHASE TRANSITIONS (pass-only paths — PAR paths are in test_par.py)
# =============================================================================

class TestPhaseTransitions:
    """IPO ends (→ INVEST) after every player-owned company is resolved."""

    def test_no_companies_transitions_immediately(self, game_state):
        """With no player-owned companies, setup transitions straight out of IPO."""
        initial_turn = TURN.get_turn_number(game_state)
        setup_ipo_phase_py(game_state)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)
        assert TURN.get_turn_number(game_state) == initial_turn + 1

    def test_single_pass_transitions(self, game_state):
        """After passing on the only company, phase leaves IPO."""
        _enter_ipo(game_state, {CO_3S: 0})
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)

    def test_transition_increments_turn_number(self, game_state):
        """End-of-IPO bumps the turn counter."""
        initial_turn = TURN.get_turn_number(game_state)
        _enter_ipo(game_state, {CO_3S: 0})
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert TURN.get_turn_number(game_state) == initial_turn + 1

    def test_transition_resets_active_player_to_position_zero(self, game_state):
        """The new turn's active player is the one at turn-order position 0."""
        # CO_3S owned by player 1 — last active player will be player 1.
        _enter_ipo(game_state, {CO_3S: 1}, {1: 200})

        assert TURN.get_active_player(game_state) == 1
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        # Back in INVEST with active player at position 0.
        expected = TURN.find_player_at_position(game_state, 0)
        assert TURN.get_active_player(game_state) == expected

    def test_transition_clears_active_company(self, game_state):
        """Active company slot is cleared when leaving IPO."""
        _enter_ipo(game_state, {CO_3S: 0})
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert TURN.get_active_company(game_state) == -1


# =============================================================================
# REMAINING FLAGS
# =============================================================================

class TestRemainingFlags:
    """ipo_remaining tracks which player companies still need a decision."""

    def test_only_loc_player_companies_marked(self, game_state):
        """Only LOC_PLAYER companies are flagged as remaining at setup."""
        _enter_ipo(game_state, {CO_3S: 0})

        for cid in range(int(GameConstants.NUM_COMPANIES)):
            loc = COMPANIES[cid].get_location(game_state)
            if loc == CompanyLocation.LOC_PLAYER:
                assert TURN.is_ipo_remaining(game_state, cid)
            else:
                assert not TURN.is_ipo_remaining(game_state, cid)

    def test_multiple_remaining_companies(self, game_state):
        """All player companies flagged remaining at setup, cleared as processed."""
        _enter_ipo(game_state, {CO_3S: 0, CO_1S_A: 0}, {0: 100})

        assert TURN.is_ipo_remaining(game_state, CO_3S)
        assert TURN.is_ipo_remaining(game_state, CO_1S_A)

        # Pass on CO_3S (FV=20, offered first)
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        assert not TURN.is_ipo_remaining(game_state, CO_3S)
        assert TURN.is_ipo_remaining(game_state, CO_1S_A)
