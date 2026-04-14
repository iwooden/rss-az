"""Tests for the IPO phase.

Covers: pass vs IPO actions, share distribution rules (FV > par → 2 shares each,
otherwise 1 share each), player/corp cash flows, market space claim, corp
activation and presidency, processing order (descending face value), legal-action
enumeration (star tier, market availability, affordability), and the phase
transition back to INVEST at the start of a new turn.
"""
from core.actions import (
    ACTION_PASS_PY as ACTION_PASS,
    ACTION_IPO_PY as ACTION_IPO,
)
from core.data import GamePhases, GameConstants, ALL_PAR_PRICES
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
    find_all_legal_actions,
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

# Par-price → par-index shortcuts derived from the engine's table.
# ALL_PAR_PRICES = (10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 27, 30, 33, 37).
PAR_10 = ALL_PAR_PRICES.index(10)
PAR_14 = ALL_PAR_PRICES.index(14)
PAR_16 = ALL_PAR_PRICES.index(16)
PAR_18 = ALL_PAR_PRICES.index(18)
PAR_20 = ALL_PAR_PRICES.index(20)
PAR_22 = ALL_PAR_PRICES.index(22)
PAR_24 = ALL_PAR_PRICES.index(24)
PAR_27 = ALL_PAR_PRICES.index(27)
PAR_30 = ALL_PAR_PRICES.index(30)
PAR_33 = ALL_PAR_PRICES.index(33)
PAR_37 = ALL_PAR_PRICES.index(37)


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
    """Legal actions: pass + (inactive corp × valid par × market available × affordable)."""

    def test_pass_always_legal(self, game_state):
        """PASS is legal in IPO whenever a company is being offered."""
        _enter_ipo(game_state, {CO_3S: 0})
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        assert pass_id == 0

    def test_only_valid_par_prices_for_star_tier(self, game_state):
        """Star-3 company only exposes par indices 5–10 (prices 16–27)."""
        _enter_ipo(game_state, {CO_3S: 0})
        ipo_actions = [info for _, info in get_legal_actions(game_state)
                       if info.action_type == ACTION_IPO]
        par_indices = {info.amount for info in ipo_actions}
        assert par_indices == {PAR_16, PAR_18, PAR_20, PAR_22, PAR_24, PAR_27}

    def test_all_inactive_corps_offered(self, game_state):
        """Every inactive corp shows up with every valid par for the star tier."""
        _enter_ipo(game_state, {CO_3S: 0}, {0: 1000})
        ipo_actions = [info for _, info in get_legal_actions(game_state)
                       if info.action_type == ACTION_IPO]
        # 8 corps × 6 valid par prices each (star 3 tier has 6 valid pars)
        assert len(ipo_actions) == 8 * 6
        corp_ids = {info.corp_id for info in ipo_actions}
        assert corp_ids == set(range(int(GameConstants.NUM_CORPS)))

    def test_action_id_encoding(self, game_state):
        """Action id = 1 + corp_id * 14 + par_index."""
        _enter_ipo(game_state, {CO_3S: 0}, {0: 1000})
        aid = find_legal_action(game_state, action_type=ACTION_IPO, corp_id=3, amount=PAR_20)
        assert aid == 1 + 3 * 14 + PAR_20

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

    def test_occupied_market_space_excludes_par(self, game_state):
        """Par whose market slot is occupied is not offered."""
        _enter_ipo(game_state, {CO_3S: 0}, {0: 1000})
        # Block the market space for par price 20.
        MARKET.set_space_available(game_state, MARKET.get_index_for_price(20), False)
        for _, info in get_legal_actions(game_state):
            if info.action_type == ACTION_IPO:
                assert info.amount != PAR_20

    def test_unaffordable_par_excluded(self, game_state):
        """Par prices the player cannot pay for are filtered."""
        # Company 14 (FV=20): par 16 cost 12, par 20 cost 0, par 22 cost 2.
        _enter_ipo(game_state, {CO_3S: 0}, {0: 1})
        ipo_actions = [info for _, info in get_legal_actions(game_state)
                       if info.action_type == ACTION_IPO]
        par_indices = {info.amount for info in ipo_actions}
        # par 16 (cost 12) and par 22 (cost 2) are excluded; par 20 (cost 0) remains.
        assert PAR_16 not in par_indices
        assert PAR_22 not in par_indices
        assert PAR_20 in par_indices

    def test_zero_cash_keeps_equal_par(self, game_state):
        """With $0 cash the player can still float at par == face (cost 0)."""
        _enter_ipo(game_state, {CO_3S: 0}, {0: 0})
        ipo_actions = [info for _, info in get_legal_actions(game_state)
                       if info.action_type == ACTION_IPO]
        par_indices = {info.amount for info in ipo_actions}
        assert par_indices == {PAR_20}  # FV=20 matches par 20 only

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
# IPO ACTION — SHARE DISTRIBUTION
# =============================================================================

class TestShareDistribution:
    """Share split depends on whether face value exceeds par price."""

    def test_fv_greater_than_par_gives_two_shares_each(self, game_state):
        """FV > par → player and bank each get 2 shares (4 issued)."""
        # CO_3S face=20; par=16 → FV > par.
        _enter_ipo(game_state, {CO_3S: 0})
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_16)
        apply_and_verify(game_state, aid)

        assert PLAYERS[0].get_shares(game_state, 0) == 2
        assert CORPS[0].get_issued_shares(game_state) == 4
        assert CORPS[0].get_bank_shares(game_state) == 2

    def test_fv_equal_to_par_gives_one_share_each(self, game_state):
        """FV == par → player and bank each get 1 share (2 issued)."""
        _enter_ipo(game_state, {CO_3S: 0})
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert PLAYERS[0].get_shares(game_state, 0) == 1
        assert CORPS[0].get_issued_shares(game_state) == 2
        assert CORPS[0].get_bank_shares(game_state) == 1

    def test_fv_less_than_par_gives_one_share_each(self, game_state):
        """FV < par → player and bank each get 1 share."""
        _enter_ipo(game_state, {CO_3S: 0})
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_27)
        apply_and_verify(game_state, aid)

        assert PLAYERS[0].get_shares(game_state, 0) == 1
        assert CORPS[0].get_issued_shares(game_state) == 2
        assert CORPS[0].get_bank_shares(game_state) == 1


# =============================================================================
# IPO ACTION — CASH FLOWS
# =============================================================================

class TestCashFlows:
    """Player pays (shares × par) − face; corp receives player + bank payments."""

    def test_player_payment_fv_greater_than_par(self, game_state):
        """Player pays 2*par − face when FV > par."""
        _enter_ipo(game_state, {CO_3S: 0}, {0: 100})
        # par=16, FV=20: cost = 2*16 - 20 = 12
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_16)
        apply_and_verify(game_state, aid)
        assert PLAYERS[0].get_cash(game_state) == 100 - 12

    def test_player_payment_zero_when_fv_equals_par(self, game_state):
        """Player pays nothing when par == face (1 share at face)."""
        _enter_ipo(game_state, {CO_3S: 0}, {0: 50})
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_20)
        apply_and_verify(game_state, aid)
        assert PLAYERS[0].get_cash(game_state) == 50

    def test_player_payment_fv_less_than_par(self, game_state):
        """Player pays par − face when FV < par (1 share each)."""
        _enter_ipo(game_state, {CO_3S: 0}, {0: 100})
        # par=27, FV=20: cost = 1*27 - 20 = 7
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_27)
        apply_and_verify(game_state, aid)
        assert PLAYERS[0].get_cash(game_state) == 100 - 7

    def test_corp_treasury_sums_both_payments(self, game_state):
        """Corp starts with player_payment + bank_payment cash."""
        _enter_ipo(game_state, {CO_3S: 0}, {0: 100})
        # par=16, FV=20: player=12, bank=32, total=44
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_16)
        apply_and_verify(game_state, aid)
        assert CORPS[0].get_cash(game_state) == 12 + 32

    def test_corp_treasury_five_star_zero_player_payment(self, game_state):
        """5-star FV=60, par=30: player payment=0, bank=60 → corp cash=60."""
        _enter_ipo(game_state, {CO_5S: 0}, {0: 0})
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_30)
        apply_and_verify(game_state, aid)
        assert PLAYERS[0].get_cash(game_state) == 0
        assert CORPS[0].get_cash(game_state) == 60


# =============================================================================
# IPO ACTION — CORP & COMPANY SIDE EFFECTS
# =============================================================================

class TestCorpAndCompanyEffects:
    """IPO activates the corp, moves the company, claims market, sets presidency."""

    def test_corp_is_activated(self, game_state):
        """IPO activates the chosen corporation."""
        _enter_ipo(game_state, {CO_3S: 0})
        assert not CORPS[0].is_active(game_state)

        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert CORPS[0].is_active(game_state)

    def test_company_moves_to_corp(self, game_state):
        """The floating company transfers from player to corp."""
        _enter_ipo(game_state, {CO_3S: 0})
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert COMPANIES[CO_3S].get_location(game_state) == CompanyLocation.LOC_CORP
        assert COMPANIES[CO_3S].get_owner_id(game_state) == 0

    def test_price_index_matches_par_price(self, game_state):
        """Corp's price index matches the selected par price's market slot."""
        _enter_ipo(game_state, {CO_3S: 0})
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_24)
        apply_and_verify(game_state, aid)

        assert CORPS[0].get_price_index(game_state) == MARKET.get_index_for_price(24)
        assert CORPS[0].get_share_price(game_state) == 24

    def test_market_space_claimed(self, game_state):
        """Chosen par price's market slot becomes unavailable."""
        _enter_ipo(game_state, {CO_3S: 0})
        mkt = MARKET.get_index_for_price(18)
        assert MARKET.is_space_available(game_state, mkt)

        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_18)
        apply_and_verify(game_state, aid)

        assert not MARKET.is_space_available(game_state, mkt)

    def test_player_becomes_president(self, game_state):
        """Floating player becomes the corp's president."""
        _enter_ipo(game_state, {CO_3S: 0})
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert CORPS[0].get_president_id(game_state) == 0

    def test_ipo_clears_remaining_flag(self, game_state):
        """After IPO, the company's ipo_remaining flag is cleared."""
        _enter_ipo(game_state, {CO_3S: 0})
        assert TURN.is_ipo_remaining(game_state, CO_3S)

        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert not TURN.is_ipo_remaining(game_state, CO_3S)


# =============================================================================
# PROCESSING ORDER
# =============================================================================

class TestProcessingOrder:
    """Companies are processed in descending face-value order."""

    def test_highest_face_value_first(self, game_state):
        """Among several player-owned companies, the highest FV is offered first."""
        _enter_ipo(game_state, {CO_1S_A: 0, CO_3S: 0, CO_5S: 0}, {0: 200})
        assert TURN.get_ipo_company(game_state) == CO_5S  # FV=60

    def test_active_player_matches_owner(self, game_state):
        """Active player is always the owner of the currently active company."""
        num_players = TURN.get_num_players(game_state)
        if num_players < 2:
            return
        _enter_ipo(game_state, {CO_5S: 0, CO_3S: 1}, {0: 200, 1: 200})

        # FV=60 first, owned by player 0.
        assert TURN.get_ipo_company(game_state) == CO_5S
        assert TURN.get_active_player(game_state) == 0

        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        # FV=20 second, owned by player 1.
        assert TURN.get_ipo_company(game_state) == CO_3S
        assert TURN.get_active_player(game_state) == 1

    def test_descending_order_three_companies(self, game_state):
        """Three companies with distinct face values pop in descending order."""
        # CO_5S (60) > CO_3S (20) > CO_1S_A (1)
        _enter_ipo(game_state, {CO_5S: 0, CO_3S: 0, CO_1S_A: 0}, {0: 200})

        expected_order = [CO_5S, CO_3S, CO_1S_A]
        for expected in expected_order:
            assert TURN.get_phase(game_state) == int(GamePhases.PHASE_IPO)
            assert TURN.get_ipo_company(game_state) == expected
            pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
            apply_and_verify(game_state, pass_id)


# =============================================================================
# PHASE TRANSITIONS
# =============================================================================

class TestPhaseTransitions:
    """IPO ends after every player-owned company is resolved."""

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

    def test_single_ipo_transitions(self, game_state):
        """After IPO on the only company, phase leaves IPO."""
        _enter_ipo(game_state, {CO_3S: 0})
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_20)
        apply_and_verify(game_state, aid)

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
        num_players = TURN.get_num_players(game_state)
        if num_players < 2:
            return
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

        assert TURN.get_ipo_company(game_state) == -1

    def test_mixed_pass_and_ipo_sequence(self, game_state):
        """Multi-company flow with both pass and IPO actions transitions out."""
        _enter_ipo(game_state, {CO_5S: 0, CO_3S: 0, CO_1S_A: 0}, {0: 300})

        # CO_5S (60) — IPO into corp 0 at par=30 (cost 0)
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_30)
        apply_and_verify(game_state, aid)
        assert CORPS[0].is_active(game_state)

        # CO_3S (20) — pass
        assert TURN.get_ipo_company(game_state) == CO_3S
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        # CO_1S_A (1) — IPO into corp 1 at par=10 (cost 9)
        assert TURN.get_ipo_company(game_state) == CO_1S_A
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=1, amount=PAR_10)
        apply_and_verify(game_state, aid)
        assert CORPS[1].is_active(game_state)

        # Out of IPO
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)


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


# =============================================================================
# BOUNDARY
# =============================================================================

class TestBoundary:
    """Exact-threshold affordability and cash checks."""

    def test_cost_exactly_equals_cash(self, game_state):
        """Player with cash equal to cost can afford the IPO."""
        # CO_3S, par=16 costs 12.
        _enter_ipo(game_state, {CO_3S: 0}, {0: 12})
        aid = find_legal_action(game_state, action_type=ACTION_IPO,
                                corp_id=0, amount=PAR_16)
        apply_and_verify(game_state, aid)
        assert PLAYERS[0].get_cash(game_state) == 0

    def test_cost_one_more_than_cash_excluded(self, game_state):
        """Par requiring $1 more than the player has is filtered out."""
        _enter_ipo(game_state, {CO_3S: 0}, {0: 11})
        par_indices = {info.amount for _, info in get_legal_actions(game_state)
                       if info.action_type == ACTION_IPO}
        assert PAR_16 not in par_indices  # cost 12 > 11

    def test_cost_one_less_than_cash_legal(self, game_state):
        """Par costing $1 less than the player's cash is legal."""
        _enter_ipo(game_state, {CO_3S: 0}, {0: 13})
        aids = find_all_legal_actions(game_state, action_type=ACTION_IPO,
                                      corp_id=0, amount=PAR_16)
        assert len(aids) == 1
