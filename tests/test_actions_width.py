"""Enumerator-width vs ACTION_SIZE agreement tests.

Each ``_enumerate_*`` helper in ``core/actions.pyx`` writes phase-local legal
action ids into a sparse buffer. The ids it emits must stay within the
declared ``ACTION_SIZE`` for that phase — otherwise the model's per-phase
policy head silently misaligns.

``_require_action_capacity`` guards against the global ``MAX_ACTION_SIZE``
upper bound (tight across all phases). It does **not** check per-phase
``ACTION_SIZE``. This file supplies the missing invariant test: for every
decision phase, construct a state that drives the enumerator to its maximum
plausible action id, then assert:

  1. ``count <= ACTION_SIZE[phase]``
  2. every emitted id is ``< ACTION_SIZE[phase]``
  3. the maximum emitted id equals ``ACTION_SIZE[phase] - 1``
     (tightness — shrinking the size without updating the encoder fails here)
"""
import numpy as np
import pytest

from core.actions import (
    enumerate_legal_actions_py,
    get_decision_phase_py,
)
from core.data import (
    DecisionPhase,
    GamePhases,
    GameConstants,
    MAX_ACTION_SIZE,
    PHASE_ACTION_SIZES,
)
from core.state import GameState
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES, CompanyLocation

from tests.phases.conftest import float_corp_for_test
from tests.phases.helpers.ownership import (
    give_company_to_player,
    give_company_to_fi,
)


NUM_CORPS = int(GameConstants.NUM_CORPS)  # 8
NUM_COMPANIES = int(GameConstants.NUM_COMPANIES)  # 36


def _fresh_state(num_players=3, seed=42):
    """Fresh initialized state (lands in PHASE_INVEST)."""
    state = GameState(num_players)
    state.initialize_game(num_players, seed=seed)
    return state


def _enumerate(state):
    """Return (phase_id, list_of_legal_ids) for the current decision phase."""
    buf = np.zeros(MAX_ACTION_SIZE, dtype=np.uint16)
    count = enumerate_legal_actions_py(state, buf)
    phase_id = get_decision_phase_py(state)
    return phase_id, [int(buf[i]) for i in range(count)]


# =============================================================================
# PER-PHASE STATE BUILDERS
# =============================================================================

def _state_invest():
    """INVEST at max: sell corp 7 emits id 52 (= ACTION_SIZE_INVEST - 1)."""
    state = _fresh_state()
    # Float corp 7 for the active player so they own a share → sell is legal.
    active = TURN.get_active_player(state)
    float_corp_for_test(state, corp_id=7, player_id=active, par_index=10, float_shares=2)
    TURN.set_active_player(state, active)
    assert TURN.get_phase(state) == int(GamePhases.PHASE_INVEST)
    return state


def _state_bid():
    """BID opening bid: offset 14 produces id 15 (= ACTION_SIZE_BID - 1)."""
    state = _fresh_state()
    active = TURN.get_active_player(state)
    PLAYERS[active].set_cash(state, 200)  # headroom past face + AUCTION_CAP
    # Pick any LOC_AUCTION company and enter BID directly; low face keeps all
    # 15 offsets (0..14) inside cash.
    auction_co = next(
        cid for cid in range(NUM_COMPANIES)
        if COMPANIES[cid].get_location(state) == int(CompanyLocation.LOC_AUCTION)
    )
    TURN.set_phase(state, int(GamePhases.PHASE_BID))
    TURN.set_active_company(state, auction_co)
    TURN.set_auction_price(state, 0)
    TURN.clear_auction_high_bidder(state)  # opening bid: high_bidder == -1
    TURN.set_auction_starter(state, active)
    return state


def _state_acq_select_corp():
    """ACQ_SELECT_CORP: select corp 7 emits id 8 (= ACTION_SIZE_ACQ_SELECT_CORP - 1).

    Same-president acquisition under corp 7 with a legal LOC_PLAYER target
    (company 35). Active player presides corp 7 and owns company 35.
    """
    state = _fresh_state()
    active = TURN.get_active_player(state)
    # Float corp 7 for the active player.
    float_corp_for_test(state, corp_id=7, player_id=active, par_index=10, float_shares=2)
    # Give corp 7 enough cash to afford any target's low price.
    CORPS[7].set_cash(state, 10_000)
    # Give company 35 (CDG, 5-star) to the active player as a legal target.
    give_company_to_player(state, 35, active)
    TURN.set_phase(state, int(GamePhases.PHASE_ACQ_SELECT_CORP))
    TURN.set_active_player(state, active)
    TURN.clear_active_corp(state)
    TURN.clear_active_company(state)
    state.acq_same_president = True
    return state


def _state_acq_select_company():
    """ACQ_SELECT_COMPANY: company 35 emits id 35 (= ACTION_SIZE_ACQ_SELECT_COMPANY - 1)."""
    state = _state_acq_select_corp()
    # Move into SELECT_COMPANY with active_corp = 7.
    TURN.set_phase(state, int(GamePhases.PHASE_ACQ_SELECT_COMPANY))
    TURN.set_active_corp(state, 7)
    return state


def _state_acq_select_price():
    """ACQ_SELECT_PRICE: LOC_FI target → FI_BUY emits id 51 (= ACTION_SIZE_ACQ_SELECT_PRICE - 1)."""
    state = _fresh_state()
    active = TURN.get_active_player(state)
    float_corp_for_test(state, corp_id=7, player_id=active, par_index=10, float_shares=2)
    CORPS[7].set_cash(state, 10_000)
    # Move company 35 to FI — FI_BUY emits exactly id 51 regardless of price.
    give_company_to_fi(state, 35)
    TURN.set_phase(state, int(GamePhases.PHASE_ACQ_SELECT_PRICE))
    TURN.set_active_corp(state, 7)
    TURN.set_active_company(state, 35)
    TURN.set_active_player(state, active)
    state.acq_same_president = True
    return state


def _state_acq_offer():
    """ACQ_OFFER: enumerator is unconditional {PASS=0, ACCEPT=1}."""
    state = _fresh_state()
    TURN.set_phase(state, int(GamePhases.PHASE_ACQ_OFFER))
    return state


def _state_closing():
    """CLOSING: close company 35 emits id 36 (= ACTION_SIZE_CLOSING - 1).

    Blue companies (stars=5) have CoO cost 0 at every CoO level, so their
    adjusted income is always positive. The enumerator width test needs
    the unrestricted legality surface, so we flip the compatibility flag.
    """
    state = _fresh_state()
    state.allow_positive_income_closing = True
    active = TURN.get_active_player(state)
    give_company_to_player(state, 35, active)
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, active)
    TURN.clear_active_corp(state)
    TURN.clear_active_company(state)
    return state


def _state_dividends():
    """DIVIDENDS: corp at $75 with cash headroom emits dividend 25 (= ACTION_SIZE_DIVIDENDS - 1).

    max_div = min(cash // issued, price // 3, 25). Price index 26 is $75,
    so price // 3 = 25.
    """
    state = _fresh_state()
    active = TURN.get_active_player(state)
    float_corp_for_test(state, corp_id=0, player_id=active, par_index=10, float_shares=1)
    # Bump the share price to the top of the market ($75) and give the corp
    # enough cash that the affordability cap does not bind.
    CORPS[0].set_price_index(state, 26)
    CORPS[0].set_cash(state, 10_000)
    TURN.set_phase(state, int(GamePhases.PHASE_DIVIDENDS))
    TURN.set_active_corp(state, 0)
    return state


def _state_issue():
    """ISSUE: enumerator is unconditional {PASS=0, ISSUE=1}."""
    state = _fresh_state()
    TURN.set_phase(state, int(GamePhases.PHASE_ISSUE_SHARES))
    return state


def _state_ipo():
    """IPO: all 8 corps inactive + affordable par → select corp 7 emits id 8."""
    state = _fresh_state()
    active = TURN.get_active_player(state)
    # Put the 5-star company in as the IPO target — every par price is
    # trivially affordable (face $60 exceeds par, so player payment is negative
    # at high pars and always within default starting cash at low pars).
    give_company_to_player(state, 35, active)
    PLAYERS[active].set_cash(state, 10_000)
    TURN.set_phase(state, int(GamePhases.PHASE_IPO))
    TURN.set_active_company(state, 35)
    TURN.set_active_player(state, active)
    return state


def _state_par():
    """PAR: star-5 company with par_index 13 valid + affordable → id 13."""
    state = _fresh_state()
    active = TURN.get_active_player(state)
    give_company_to_player(state, 35, active)
    PLAYERS[active].set_cash(state, 10_000)
    TURN.set_phase(state, int(GamePhases.PHASE_PAR))
    TURN.set_active_company(state, 35)
    TURN.set_active_player(state, active)
    return state


PHASE_BUILDERS = {
    DecisionPhase.DPHASE_INVEST: _state_invest,
    DecisionPhase.DPHASE_BID: _state_bid,
    DecisionPhase.DPHASE_ACQ_SELECT_CORP: _state_acq_select_corp,
    DecisionPhase.DPHASE_ACQ_SELECT_COMPANY: _state_acq_select_company,
    DecisionPhase.DPHASE_ACQ_SELECT_PRICE: _state_acq_select_price,
    DecisionPhase.DPHASE_ACQ_OFFER: _state_acq_offer,
    DecisionPhase.DPHASE_CLOSING: _state_closing,
    DecisionPhase.DPHASE_DIVIDENDS: _state_dividends,
    DecisionPhase.DPHASE_ISSUE: _state_issue,
    DecisionPhase.DPHASE_IPO: _state_ipo,
    DecisionPhase.DPHASE_PAR: _state_par,
}


# =============================================================================
# WIDTH INVARIANT
# =============================================================================

@pytest.mark.parametrize(
    "phase", list(PHASE_BUILDERS.keys()), ids=[p.name for p in PHASE_BUILDERS]
)
def test_enumerator_width_within_action_size(phase):
    """Enumerator at max plausible load stays within ACTION_SIZE for every phase.

    Asserts all three invariants: count cap, per-id cap, and tightness
    (max id reaches size - 1). The tightness check is what catches a silent
    ACTION_SIZE shrink — the other two bounds could pass on a state that
    happens not to exercise the largest id.
    """
    phase_id = int(phase)
    size = PHASE_ACTION_SIZES[phase_id]

    state = PHASE_BUILDERS[phase]()

    actual_phase, ids = _enumerate(state)
    assert actual_phase == phase_id, (
        f"{phase.name}: setup left state in decision phase {actual_phase}, "
        f"expected {phase_id}"
    )

    assert len(ids) <= size, (
        f"{phase.name}: enumerator emitted {len(ids)} ids, "
        f"exceeds ACTION_SIZE={size}"
    )
    assert all(0 <= aid < size for aid in ids), (
        f"{phase.name}: emitted ids outside [0, {size}): {ids}"
    )
    assert max(ids) == size - 1, (
        f"{phase.name}: expected max id {size - 1} (tightness), got {max(ids)}. "
        f"Either the setup is too weak or ACTION_SIZE no longer matches the "
        f"encoder's maximum.\nLegal ids: {ids}"
    )
