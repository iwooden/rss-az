"""Shared fixtures and assertion helpers for phase tests.

Usage:
  - apply_and_verify(state, action_id): Apply action through the driver, check
    invariants on all intermediate states (auto-applied forced actions) and the
    final state. Returns ApplyResult.
  - assert_invariants(state, msg): Check state invariants directly (for tests
    that verify state setup before applying actions).
  - get_legal_actions(state): Enumerate and decode all legal actions. Returns
    list of (action_id, decoded_info) tuples.
  - find_legal_action(state, **kwargs): Find a legal action matching decoded
    properties. Returns action_id or raises AssertionError.
  - float_corp_for_test(...): Float a corporation for testing.
  - setup_receivership_corp(...): Float a corp and put it into receivership.
"""
import pytest
import numpy as np

from core.state import GameState
from core.driver import DRIVER, STATUS_OK_PY as STATUS_OK
from core.actions import (
    enumerate_legal_actions_py,
    decode_action_py,
    get_decision_phase_py,
    MAX_LEGAL_ACTIONS_PY as MAX_LEGAL_ACTIONS,
)
from core.data import GamePhases, GameConstants
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES, CompanyLocation
from entities.market import MARKET
from entities.fi import FI
from entities.deck import DECK


# =============================================================================
# LEGAL ACTION HELPERS
# =============================================================================

def get_legal_actions(state):
    """Enumerate and decode all legal actions for the current state.

    Returns:
        List of (action_id, info) tuples where info is a namedtuple with
        fields: phase, action_type, corp_id, company_id, amount.
    """
    buf = np.zeros(MAX_LEGAL_ACTIONS, dtype=np.uint16)
    count = enumerate_legal_actions_py(state, buf)
    phase_id = get_decision_phase_py(state)
    assert phase_id >= 0, f"not in a decision phase (engine phase={TURN.get_phase(state)})"
    result = []
    for i in range(count):
        action_id = int(buf[i])
        info = decode_action_py(phase_id, action_id)
        result.append((action_id, info))
    return result


def find_legal_action(state, action_type=None, corp_id=None, company_id=None, amount=None):
    """Find a legal action matching the given decoded properties.

    Args:
        state: GameState
        action_type: ACTION_* constant to match (e.g. ACTION_PASS)
        corp_id: Corporation ID to match
        company_id: Company ID to match
        amount: Amount/offset to match

    Returns:
        The action_id of the first matching legal action.

    Raises:
        AssertionError if no matching action is found.
    """
    actions = get_legal_actions(state)
    for action_id, info in actions:
        if action_type is not None and info.action_type != action_type:
            continue
        if corp_id is not None and info.corp_id != corp_id:
            continue
        if company_id is not None and info.company_id != company_id:
            continue
        if amount is not None and info.amount != amount:
            continue
        return action_id

    # Build a helpful error message
    filters = []
    if action_type is not None:
        filters.append(f"action_type={action_type}")
    if corp_id is not None:
        filters.append(f"corp_id={corp_id}")
    if company_id is not None:
        filters.append(f"company_id={company_id}")
    if amount is not None:
        filters.append(f"amount={amount}")
    available = [(aid, f"type={i.action_type} corp={i.corp_id} co={i.company_id} amt={i.amount}")
                 for aid, i in actions]
    assert False, (
        f"No legal action matching {', '.join(filters)}.\n"
        f"Available ({len(available)}): {available[:20]}"
        f"{'...' if len(available) > 20 else ''}"
    )


def find_all_legal_actions(state, action_type=None, corp_id=None, company_id=None, amount=None):
    """Like find_legal_action but returns ALL matching action_ids as a list."""
    actions = get_legal_actions(state)
    result = []
    for action_id, info in actions:
        if action_type is not None and info.action_type != action_type:
            continue
        if corp_id is not None and info.corp_id != corp_id:
            continue
        if company_id is not None and info.company_id != company_id:
            continue
        if amount is not None and info.amount != amount:
            continue
        result.append(action_id)
    return result


# =============================================================================
# APPLY + VERIFY
# =============================================================================

class ApplyResult:
    """Result wrapper for apply_and_verify().

    History entries are 3-tuples: (state_array, phase_id, action_id).
    For decision actions: phase_id is the DecisionPhase, action_id is phase-local.
    For automated phases: phase_id is -1, action_id is the engine phase enum.
    """

    def __init__(self, state, history, status, num_players):
        self.state = state
        self.history = history
        self.status = status
        self.applied_count = len(history)
        self._num_players = num_players

    def get_state_at(self, index):
        """Get state snapshot at position (supports negative indexing)."""
        return GameState.from_array(self.history[index][0], self._num_players)

    def get_action_at(self, index):
        """Get (phase_id, action_id) at position (supports negative indexing)."""
        return self.history[index][1], self.history[index][2]

    @property
    def last_action(self):
        """Last (phase_id, action_id) applied (convenience)."""
        if not self.history:
            return None
        return self.history[-1][1], self.history[-1][2]


def apply_and_verify(state, action_id, msg="", expected_status=STATUS_OK):
    """Apply action through the driver and verify invariants on every state.

    Checks invariants on all intermediate states captured by the driver's
    auto-chain loop, plus the final state after all actions.

    Args:
        state: GameState (modified in place)
        action_id: Phase-local action ID to apply
        msg: Context for assertion failures
        expected_status: Expected return status (STATUS_OK or STATUS_GAME_OVER)

    Returns:
        ApplyResult for history inspection.
    """
    num_players = TURN.get_num_players(state)

    # Verify action is legal before applying
    buf = np.zeros(MAX_LEGAL_ACTIONS, dtype=np.uint16)
    count = enumerate_legal_actions_py(state, buf)
    legal_ids = set(int(buf[i]) for i in range(count))
    assert action_id in legal_ids, (
        f"{msg}\nAction {action_id} not legal. "
        f"Legal actions ({count}): {sorted(legal_ids)[:30]}"
    )

    history = []
    status = DRIVER.apply_action(state, action_id, history=history)
    assert status == expected_status, (
        f"{msg}\nAction {action_id} returned status {status}, expected {expected_status}"
    )

    # Check invariants on every intermediate state (captured BEFORE each action)
    for i, (state_array, phase_id, act_id) in enumerate(history):
        intermediate = GameState.from_array(state_array, num_players)
        assert_invariants(
            intermediate,
            f"{msg}\nIntermediate state {i}/{len(history)}, "
            f"before phase={phase_id} action={act_id}",
        )

    # Check invariants on final state (AFTER all actions)
    assert_invariants(state, f"{msg}\nFinal state after action chain")

    # Verify valid actions exist in non-terminal states
    phase = TURN.get_phase(state)
    if phase != GamePhases.PHASE_GAME_OVER:
        decision_phase = get_decision_phase_py(state)
        if decision_phase >= 0:
            buf2 = np.zeros(MAX_LEGAL_ACTIONS, dtype=np.uint16)
            count2 = enumerate_legal_actions_py(state, buf2)
            assert count2 > 0, f"{msg}\nNo legal actions after action {action_id}"

    return ApplyResult(state, history, status, num_players)


# =============================================================================
# STATE SETUP HELPERS
# =============================================================================

def draw_company(state):
    """Draw a company from the deck (LOC_DECK -> LOC_REVEALED). Returns company_id."""
    cid = DECK.draw(state)
    assert cid >= 0, "Deck is empty"
    return cid


def draw_to_player(state, player_id):
    """Draw a company and transfer it to a player. Returns company_id."""
    cid = draw_company(state)
    COMPANIES[cid].transfer_to_player(state, player_id)
    return cid


def draw_to_fi(state):
    """Draw a company and transfer it to the Foreign Investor. Returns company_id."""
    cid = draw_company(state)
    COMPANIES[cid].transfer_to_fi(state)
    return cid


def draw_to_corp(state, corp_id):
    """Draw a company and transfer it to a corporation. Returns company_id."""
    cid = draw_company(state)
    COMPANIES[cid].transfer_to_corp(state, corp_id)
    return cid


def float_corp_for_test(state, corp_id, company_id=None, player_id=0, par_index=10, float_shares=1):
    """Float a corporation for testing.

    Args:
        state: GameState instance
        corp_id: Corporation ID to float
        company_id: Company ID to use. If None, draws from deck.
        player_id: Player who becomes president (default 0)
        par_index: Market price index for starting share price (default 10)
        float_shares: Shares for player and bank each (default 1)

    Returns:
        The company_id that was used.
    """
    if company_id is None:
        company_id = DECK.draw(state)
        assert company_id >= 0, "Deck is empty, cannot draw company for floating"

    COMPANIES[company_id].transfer_to_player(state, player_id)
    CORPS[corp_id].float_corp(state, player_id, company_id, par_index, float_shares)
    return company_id


def setup_receivership_corp(state, corp_id, company_ids):
    """Float a corp and put it into receivership with the given companies.

    First company is used for floating (transferred to player 0, then floated).
    Player 0's shares are zeroed to trigger receivership. Additional companies
    are transferred directly to the corp.

    Args:
        state: GameState
        corp_id: Corporation ID
        company_ids: List of company IDs (first used for floating)
    """
    float_corp_for_test(state, corp_id=corp_id, company_id=company_ids[0])
    PLAYERS[0].set_shares(state, corp_id, 0)
    for cid in company_ids[1:]:
        COMPANIES[cid].transfer_to_corp(state, corp_id)


# =============================================================================
# INVARIANT ASSERTIONS
# =============================================================================

def assert_invariants(state, msg=""):
    """Assert game state invariants are maintained.

    Checks:
      - Share conservation per corp
      - Player cash, corp cash, FI cash >= 0
      - Active corp has >= 1 company, valid price index
      - President invariants (non-receivership has president with >= 1 share)
      - Receivership has no president
      - Market boundary spaces available
      - Company location validity and ownership consistency
      - Deck count matches LOC_DECK companies
      - Auction row size <= num_players
    """
    num_players = TURN.get_num_players(state)

    # --- Share conservation for active corps ---
    for corp_id in range(int(GameConstants.NUM_CORPS)):
        corp = CORPS[corp_id]
        if not corp.is_active(state):
            continue
        total = corp.get_unissued_shares(state) + corp.get_bank_shares(state)
        for p in range(num_players):
            total += PLAYERS[p].get_shares(state, corp_id)
        # Total must equal the corp's canonical share count
        expected = corp.get_unissued_shares(state) + corp.get_issued_shares(state)
        assert total == expected, (
            f"{msg}\nCorp {corp_id} share conservation: "
            f"unissued({corp.get_unissued_shares(state)}) + "
            f"bank({corp.get_bank_shares(state)}) + "
            f"player_held({total - corp.get_unissued_shares(state) - corp.get_bank_shares(state)}) "
            f"= {total} != expected {expected}"
        )

    # --- Issued = bank + player shares ---
    for corp_id in range(int(GameConstants.NUM_CORPS)):
        corp = CORPS[corp_id]
        if not corp.is_active(state):
            continue
        issued = corp.get_issued_shares(state)
        bank = corp.get_bank_shares(state)
        player_held = sum(PLAYERS[p].get_shares(state, corp_id) for p in range(num_players))
        assert issued == bank + player_held, (
            f"{msg}\nCorp {corp_id} issued shares mismatch: "
            f"issued({issued}) != bank({bank}) + players({player_held})"
        )

    # --- Player cash non-negative ---
    for p in range(num_players):
        cash = PLAYERS[p].get_cash(state)
        assert cash >= 0, f"{msg}\nPlayer {p} cash negative: {cash}"

    # --- Corp cash non-negative for active corps ---
    for corp_id in range(int(GameConstants.NUM_CORPS)):
        if CORPS[corp_id].is_active(state):
            cash = CORPS[corp_id].get_cash(state)
            assert cash >= 0, f"{msg}\nCorp {corp_id} cash negative: {cash}"

    # --- FI cash non-negative ---
    assert FI.get_cash(state) >= 0, f"{msg}\nFI cash negative: {FI.get_cash(state)}"

    # --- Active corp has >= 1 company ---
    for corp_id in range(int(GameConstants.NUM_CORPS)):
        if CORPS[corp_id].is_active(state):
            count = CORPS[corp_id].count_companies(state, include_acquisition=True)
            assert count >= 1, (
                f"{msg}\nCorp {corp_id} active but has {count} companies"
            )

    # --- Corp price index in valid range ---
    for corp_id in range(int(GameConstants.NUM_CORPS)):
        if CORPS[corp_id].is_active(state):
            idx = CORPS[corp_id].get_price_index(state)
            assert 0 <= idx <= int(GameConstants.NUM_MARKET_SPACES) - 1, (
                f"{msg}\nCorp {corp_id} price index out of range: {idx}"
            )

    # --- President invariants ---
    for corp_id in range(int(GameConstants.NUM_CORPS)):
        corp = CORPS[corp_id]
        if not corp.is_active(state):
            continue
        if corp.is_in_receivership(state):
            assert corp.get_president_id(state) == -1, (
                f"{msg}\nCorp {corp_id} in receivership but has president: "
                f"player {corp.get_president_id(state)}"
            )
        else:
            pres_id = corp.get_president_id(state)
            assert pres_id >= 0, (
                f"{msg}\nCorp {corp_id} active, not in receivership, but no president"
            )
            pres_shares = PLAYERS[pres_id].get_shares(state, corp_id)
            assert pres_shares >= 1, (
                f"{msg}\nCorp {corp_id} president (player {pres_id}) holds "
                f"{pres_shares} shares"
            )

    # --- Market boundary spaces always available ---
    assert MARKET.is_space_available(state, 0), (
        f"{msg}\nMarket space 0 ($0 bankruptcy) must always be available"
    )
    assert MARKET.is_space_available(state, int(GameConstants.NUM_MARKET_SPACES) - 1), (
        f"{msg}\nMarket space 26 ($75 max) must always be available"
    )

    # --- Company location validity ---
    for cid in range(int(GameConstants.NUM_COMPANIES)):
        loc = COMPANIES[cid].get_location(state)
        assert 0 <= loc <= int(CompanyLocation.LOC_EXCLUDED), (
            f"{msg}\nCompany {cid} has invalid location: {loc}"
        )

    # --- Deck count matches LOC_DECK companies ---
    deck_remaining = DECK.get_remaining_count(state)
    deck_loc_count = sum(
        1 for cid in range(int(GameConstants.NUM_COMPANIES))
        if COMPANIES[cid].get_location(state) == int(CompanyLocation.LOC_DECK)
    )
    assert deck_remaining == deck_loc_count, (
        f"{msg}\nDeck count mismatch: entity says {deck_remaining}, "
        f"LOC_DECK count is {deck_loc_count}"
    )

    # --- Company ownership consistency ---
    for cid in range(int(GameConstants.NUM_COMPANIES)):
        company = COMPANIES[cid]
        loc = company.get_location(state)
        owner_id = company.get_owner_id(state)

        if loc == int(CompanyLocation.LOC_PLAYER):
            assert 0 <= owner_id < num_players, (
                f"{msg}\nCompany {cid} at LOC_PLAYER has invalid owner_id: {owner_id}"
            )
            assert PLAYERS[owner_id].owns_company(state, cid), (
                f"{msg}\nCompany {cid} at LOC_PLAYER owner={owner_id} "
                f"but player doesn't list it"
            )
        elif loc == int(CompanyLocation.LOC_CORP):
            assert 0 <= owner_id < int(GameConstants.NUM_CORPS), (
                f"{msg}\nCompany {cid} at LOC_CORP has invalid owner_id: {owner_id}"
            )
            assert CORPS[owner_id].owns_company(state, cid), (
                f"{msg}\nCompany {cid} at LOC_CORP owner={owner_id} "
                f"but corp doesn't list it"
            )
        elif loc == int(CompanyLocation.LOC_CORP_ACQ):
            assert 0 <= owner_id < int(GameConstants.NUM_CORPS), (
                f"{msg}\nCompany {cid} at LOC_CORP_ACQ has invalid owner_id: {owner_id}"
            )

    # --- Auction row size <= num_players ---
    auction_count = sum(
        1 for cid in range(int(GameConstants.NUM_COMPANIES))
        if COMPANIES[cid].get_location(state) == int(CompanyLocation.LOC_AUCTION)
    )
    assert auction_count <= num_players, (
        f"{msg}\nAuction row size {auction_count} > {num_players}"
    )

    # --- Ghost deck entries must not have LOC_DECK ---
    for slot_idx, cid in DECK.get_ghost_entries(state):
        loc = COMPANIES[cid].get_location(state)
        assert loc != int(CompanyLocation.LOC_DECK), (
            f"{msg}\nCompany {cid} in ghost deck slot {slot_idx} "
            f"still has LOC_DECK location"
        )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(params=[2, 3, 4, 5, 6])
def game_state(request):
    """Fresh game state in INVEST phase, parameterized across all player counts."""
    num_players = request.param
    state = GameState(num_players)
    state.initialize_game(num_players, seed=42)
    assert TURN.get_phase(state) == GamePhases.PHASE_INVEST
    return state
