"""Tests for the PAR phase (price-select half of Form Corporation).

Entered only via ``apply_ipo_action`` (corp-select) from PHASE_IPO. Covers par
enumeration (star-tier / market-slot / affordability triple gate, no pass),
float execution (share distribution, cash flows, presidency, market claim),
active_corp cleanup, and the PAR→IPO / PAR→INVEST transitions after resolution.

IPO-phase concerns (pass semantics, corp-select legality, processing order)
live in ``test_ipo.py``.
"""
from core.actions import (
    ACTION_PASS_PY as ACTION_PASS,
    ACTION_IPO_PY as ACTION_IPO,
    ACTION_PAR_PY as ACTION_PAR,
)
from core.data import GamePhases, ALL_PAR_PRICES
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
)
from tests.phases.helpers.finance import set_player_cashs
from tests.phases.helpers.ownership import give_company_to_player


# =============================================================================
# HELPERS
# =============================================================================

# Companies used across tests (same mapping as test_ipo.py):
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


def _enter_par(state, company_id, corp_id, *, owners=None, cash_by_player=None):
    """Seed state so the active decision is PHASE_PAR for (company_id, corp_id).

    Routes through PHASE_IPO and applies an IPO corp-select action so the
    PAR handler sees the same active_corp/active_company seeding production
    code would produce.
    """
    if owners is None:
        owners = {company_id: 0}
    for cid, pid in owners.items():
        give_company_to_player(state, cid, pid)
    if cash_by_player is None:
        cash_by_player = {pid: 100 for pid in set(owners.values())}
    set_player_cashs(state, cash_by_player)
    setup_ipo_phase_py(state)

    assert TURN.get_active_company(state) == company_id, (
        f"expected active_company={company_id}, got {TURN.get_active_company(state)}"
    )
    aid = find_legal_action(state, action_type=ACTION_IPO, corp_id=corp_id)
    apply_and_verify(state, aid)
    # If the PAR state has only one legal par price, driver auto-chain resolves
    # it immediately and pushes us out of PHASE_PAR. Callers must pick cash /
    # star-tier leaving ≥2 legal pars so the test observes a real decision.
    assert TURN.get_phase(state) == int(GamePhases.PHASE_PAR), (
        f"_enter_par: expected PHASE_PAR after IPO corp-select, got phase="
        f"{TURN.get_phase(state)}. The driver auto-chains past PAR when only "
        f"one par price is legal — increase cash or pick a company with ≥2 pars."
    )
    assert TURN.get_active_corp(state) == corp_id
    assert TURN.get_active_company(state) == company_id


# =============================================================================
# ENUMERATION
# =============================================================================

class TestEnumeration:
    """PAR legality: valid par × available market × affordable. No pass."""

    def test_no_pass_action(self, game_state):
        """PAR has no pass; every legal action is a par-price choice."""
        _enter_par(game_state, CO_3S, 0)
        actions = get_legal_actions(game_state)
        assert all(info.action_type == ACTION_PAR for _, info in actions)

    def test_only_valid_par_prices_for_star_tier(self, game_state):
        """Star-3 company only exposes par indices 5–10 (prices 16–27)."""
        _enter_par(game_state, CO_3S, 0)
        par_indices = {info.amount for _, info in get_legal_actions(game_state)}
        assert par_indices == {PAR_16, PAR_18, PAR_20, PAR_22, PAR_24, PAR_27}

    def test_action_id_encoding(self, game_state):
        """Action id in PAR is the par_index itself (no pass offset)."""
        _enter_par(game_state, CO_3S, 0, cash_by_player={0: 1000})
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_20)
        assert aid == PAR_20

    def test_occupied_market_space_excludes_par(self, game_state):
        """Par whose market slot is occupied is not offered."""
        # Block the market space for par price 20 BEFORE entering PAR so the
        # IPO gate still offers corp 0 (the other pars are available).
        MARKET.set_space_available(game_state, MARKET.get_index_for_price(20), False)
        _enter_par(game_state, CO_3S, 0, cash_by_player={0: 1000})
        par_indices = {info.amount for _, info in get_legal_actions(game_state)}
        assert PAR_20 not in par_indices

    def test_unaffordable_par_excluded(self, game_state):
        """Par prices the player cannot pay for are filtered."""
        # CO_3S (FV=20): par 16 cost 12, par 18 cost 16, par 20 cost 0,
        # par 22 cost 2, par 24 cost 4, par 27 cost 7. Cash=$11 excludes 16 & 18.
        _enter_par(game_state, CO_3S, 0, cash_by_player={0: 11})
        par_indices = {info.amount for _, info in get_legal_actions(game_state)}
        assert PAR_16 not in par_indices  # cost 12 > 11
        assert PAR_18 not in par_indices  # cost 16 > 11
        assert PAR_20 in par_indices      # cost 0
        assert PAR_22 in par_indices      # cost 2


# =============================================================================
# SHARE DISTRIBUTION
# =============================================================================

class TestShareDistribution:
    """Share split depends on whether face value exceeds par price."""

    def test_fv_greater_than_par_gives_two_shares_each(self, game_state):
        """FV > par → player and bank each get 2 shares (4 issued)."""
        # CO_3S face=20; par=16 → FV > par.
        _enter_par(game_state, CO_3S, 0)
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_16)
        apply_and_verify(game_state, aid)

        assert PLAYERS[0].get_shares(game_state, 0) == 2
        assert CORPS[0].get_issued_shares(game_state) == 4
        assert CORPS[0].get_bank_shares(game_state) == 2

    def test_fv_equal_to_par_gives_one_share_each(self, game_state):
        """FV == par → player and bank each get 1 share (2 issued)."""
        _enter_par(game_state, CO_3S, 0)
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert PLAYERS[0].get_shares(game_state, 0) == 1
        assert CORPS[0].get_issued_shares(game_state) == 2
        assert CORPS[0].get_bank_shares(game_state) == 1

    def test_fv_less_than_par_gives_one_share_each(self, game_state):
        """FV < par → player and bank each get 1 share."""
        _enter_par(game_state, CO_3S, 0)
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_27)
        apply_and_verify(game_state, aid)

        assert PLAYERS[0].get_shares(game_state, 0) == 1
        assert CORPS[0].get_issued_shares(game_state) == 2
        assert CORPS[0].get_bank_shares(game_state) == 1


# =============================================================================
# CASH FLOWS
# =============================================================================

class TestCashFlows:
    """Player pays (shares × par) − face; corp receives player + bank payments."""

    def test_player_payment_fv_greater_than_par(self, game_state):
        """Player pays 2*par − face when FV > par."""
        _enter_par(game_state, CO_3S, 0, cash_by_player={0: 100})
        # par=16, FV=20: cost = 2*16 - 20 = 12
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_16)
        apply_and_verify(game_state, aid)
        assert PLAYERS[0].get_cash(game_state) == 100 - 12

    def test_player_payment_zero_when_fv_equals_par(self, game_state):
        """Player pays nothing when par == face (1 share at face)."""
        _enter_par(game_state, CO_3S, 0, cash_by_player={0: 50})
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_20)
        apply_and_verify(game_state, aid)
        assert PLAYERS[0].get_cash(game_state) == 50

    def test_player_payment_fv_less_than_par(self, game_state):
        """Player pays par − face when FV < par (1 share each)."""
        _enter_par(game_state, CO_3S, 0, cash_by_player={0: 100})
        # par=27, FV=20: cost = 1*27 - 20 = 7
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_27)
        apply_and_verify(game_state, aid)
        assert PLAYERS[0].get_cash(game_state) == 100 - 7

    def test_corp_treasury_sums_both_payments(self, game_state):
        """Corp starts with player_payment + bank_payment cash."""
        _enter_par(game_state, CO_3S, 0, cash_by_player={0: 100})
        # par=16, FV=20: player=12, bank=32, total=44
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_16)
        apply_and_verify(game_state, aid)
        assert CORPS[0].get_cash(game_state) == 12 + 32

    def test_corp_treasury_five_star_zero_player_payment(self, game_state):
        """5-star FV=60, par=30: player payment=0, bank=60 → corp cash=60.

        Uses cash=$14 so pars 30 (cost 0), 33 (cost 6), 37 (cost 14) all
        remain legal — keeps the driver from auto-chaining past PAR on a
        single-legal-action state.
        """
        _enter_par(game_state, CO_5S, 0, cash_by_player={0: 14})
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_30)
        apply_and_verify(game_state, aid)
        assert PLAYERS[0].get_cash(game_state) == 14  # par_30 cost is 0
        assert CORPS[0].get_cash(game_state) == 60


# =============================================================================
# CORP & COMPANY SIDE EFFECTS
# =============================================================================

class TestCorpAndCompanyEffects:
    """PAR activates the corp, moves the company, claims market, sets presidency."""

    def test_corp_is_activated(self, game_state):
        """PAR activates the chosen corporation."""
        _enter_par(game_state, CO_3S, 0)
        assert not CORPS[0].is_active(game_state)

        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert CORPS[0].is_active(game_state)

    def test_company_moves_to_corp(self, game_state):
        """The floating company transfers from player to corp."""
        _enter_par(game_state, CO_3S, 0)
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert COMPANIES[CO_3S].get_location(game_state) == CompanyLocation.LOC_CORP
        assert COMPANIES[CO_3S].get_owner_id(game_state) == 0

    def test_price_index_matches_par_price(self, game_state):
        """Corp's price index matches the selected par price's market slot."""
        _enter_par(game_state, CO_3S, 0)
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_24)
        apply_and_verify(game_state, aid)

        assert CORPS[0].get_price_index(game_state) == MARKET.get_index_for_price(24)
        assert CORPS[0].get_share_price(game_state) == 24

    def test_market_space_claimed(self, game_state):
        """Chosen par price's market slot becomes unavailable."""
        _enter_par(game_state, CO_3S, 0)
        mkt = MARKET.get_index_for_price(18)
        assert MARKET.is_space_available(game_state, mkt)

        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_18)
        apply_and_verify(game_state, aid)

        assert not MARKET.is_space_available(game_state, mkt)

    def test_player_becomes_president(self, game_state):
        """Floating player becomes the corp's president."""
        _enter_par(game_state, CO_3S, 0)
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert CORPS[0].get_president_id(game_state) == 0


# =============================================================================
# POST-RESOLUTION STATE
# =============================================================================

class TestPostResolution:
    """PAR clears active_corp, clears remaining flag, advances to next company."""

    def test_active_corp_cleared(self, game_state):
        """Resolving PAR clears active_corp so IPO starts fresh for next company."""
        _enter_par(game_state, CO_3S, 0, owners={CO_3S: 0, CO_1S_A: 0},
                   cash_by_player={0: 100})
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert TURN.get_active_corp(game_state) == -1

    def test_remaining_flag_cleared(self, game_state):
        """The floated company is dropped from the ipo_remaining set."""
        _enter_par(game_state, CO_3S, 0)
        assert TURN.is_ipo_remaining(game_state, CO_3S)

        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert not TURN.is_ipo_remaining(game_state, CO_3S)


# =============================================================================
# PHASE TRANSITIONS
# =============================================================================

class TestPhaseTransitions:
    """PAR → IPO when more companies remain; PAR → INVEST when last."""

    def test_par_returns_to_ipo_when_more_companies_remain(self, game_state):
        """After resolving PAR, control returns to IPO for the next company."""
        # Two companies: CO_3S (FV=20) processed first, CO_1S_A (FV=1) next.
        _enter_par(game_state, CO_3S, 0,
                   owners={CO_3S: 0, CO_1S_A: 0}, cash_by_player={0: 100})

        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_IPO)
        assert TURN.get_active_company(game_state) == CO_1S_A

    def test_par_transitions_to_invest_when_last_company(self, game_state):
        """After resolving PAR on the only remaining company, phase → INVEST."""
        _enter_par(game_state, CO_3S, 0)
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)

    def test_transition_increments_turn_number(self, game_state):
        """End-of-IPO (via PAR resolving the last company) bumps the turn."""
        initial_turn = TURN.get_turn_number(game_state)
        _enter_par(game_state, CO_3S, 0)
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert TURN.get_turn_number(game_state) == initial_turn + 1

    def test_transition_clears_active_company(self, game_state):
        """active_company clears when PAR transitions out of the IPO flow."""
        _enter_par(game_state, CO_3S, 0)
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_20)
        apply_and_verify(game_state, aid)

        assert TURN.get_active_company(game_state) == -1

    def test_mixed_pass_and_par_sequence(self, game_state):
        """Multi-company flow with both IPO pass and IPO→PAR→IPO transitions."""
        owners = {CO_5S: 0, CO_3S: 0, CO_1S_A: 0}
        for cid, pid in owners.items():
            give_company_to_player(game_state, cid, pid)
        set_player_cashs(game_state, {0: 300})
        setup_ipo_phase_py(game_state)

        # CO_5S (60): IPO corp-select 0 → PAR → par=30 (cost 0)
        ipo_aid = find_legal_action(game_state, action_type=ACTION_IPO, corp_id=0)
        apply_and_verify(game_state, ipo_aid)
        par_aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_30)
        apply_and_verify(game_state, par_aid)
        assert CORPS[0].is_active(game_state)
        assert TURN.get_active_corp(game_state) == -1

        # CO_3S (20): pass
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_IPO)
        assert TURN.get_active_company(game_state) == CO_3S
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        apply_and_verify(game_state, pass_id)

        # CO_1S_A (1): IPO corp-select 1 → PAR → par=10 (cost 9)
        assert TURN.get_active_company(game_state) == CO_1S_A
        ipo_aid = find_legal_action(game_state, action_type=ACTION_IPO, corp_id=1)
        apply_and_verify(game_state, ipo_aid)
        par_aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_10)
        apply_and_verify(game_state, par_aid)
        assert CORPS[1].is_active(game_state)

        # Out of IPO / PAR
        assert TURN.get_phase(game_state) == int(GamePhases.PHASE_INVEST)


# =============================================================================
# BOUNDARY
# =============================================================================

class TestBoundary:
    """Exact-threshold affordability and cash checks at PAR."""

    def test_cost_exactly_equals_cash(self, game_state):
        """Player with cash equal to cost can afford the PAR."""
        # CO_3S, par=16 costs 12.
        _enter_par(game_state, CO_3S, 0, cash_by_player={0: 12})
        aid = find_legal_action(game_state, action_type=ACTION_PAR, amount=PAR_16)
        apply_and_verify(game_state, aid)
        assert PLAYERS[0].get_cash(game_state) == 0

    def test_cost_one_more_than_cash_excluded(self, game_state):
        """Par requiring $1 more than the player has is filtered out."""
        _enter_par(game_state, CO_3S, 0, cash_by_player={0: 11})
        par_indices = {info.amount for _, info in get_legal_actions(game_state)}
        assert PAR_16 not in par_indices  # cost 12 > 11

    def test_cost_one_less_than_cash_legal(self, game_state):
        """Par costing $1 less than the player's cash is legal."""
        _enter_par(game_state, CO_3S, 0, cash_by_player={0: 13})
        aids = find_all_legal_actions(game_state, action_type=ACTION_PAR,
                                      amount=PAR_16)
        assert len(aids) == 1
