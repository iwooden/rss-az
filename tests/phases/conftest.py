"""Shared fixtures and assertion helpers for phase tests.

Usage:
  - apply_and_verify(state, action_id): Apply action through the driver, check
    invariants on all intermediate states (auto-applied forced actions) and the
    final state. Returns ApplyResult.
  - assert_invariants(state, msg): Check state invariants directly (for tests
    that verify state setup before applying actions).
  - get_legal_actions(state): Enumerate and decode all legal actions. Returns
    list of (action_id, decoded_info) tuples.
  - find_legal_action_with_info(state, **kwargs): Find the first matching legal
    action and return (action_id, decoded_info).
  - find_all_legal_actions_with_info(state, **kwargs): Return all matching legal
    actions as (action_id, decoded_info) tuples.
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
from core.data import (
    GamePhases,
    GameConstants,
    CorpIndices,
    ENGINE_TO_DECISION_PHASE,
    PY_PAR_PRICE_VALID,
    PY_CASH_DIVISOR,
    PY_NET_WORTH_DIVISOR,
    PY_SHARE_DIVISOR,
    PY_ENTITY_INCOME_DIVISOR,
    PY_COMPANY_INCOME_DIVISOR,
    PY_COMPANY_PRICE_DIVISOR,
    PY_SHARE_PRICE_DIVISOR,
    PY_CORP_STAR_DIVISOR,
    PY_COMPANY_STAR_DIVISOR,
    PY_IMPACT_DIVISOR,
)
from core.token_data import get_token_data, get_num_tokens, TokenDataSize
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES, CompanyLocation
from entities.market import MARKET
from entities.fi import FI
from entities.deck import DECK

from tests.phases.helpers.ownership import (
    give_company_to_player,
    give_company_to_corp,
)


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


def _legal_action_matches(info, action_type=None, corp_id=None, company_id=None, amount=None):
    if action_type is not None and info.action_type != action_type:
        return False
    if corp_id is not None and info.corp_id != corp_id:
        return False
    if company_id is not None and info.company_id != company_id:
        return False
    if amount is not None and info.amount != amount:
        return False
    return True


def _format_legal_action_filters(action_type=None, corp_id=None, company_id=None, amount=None):
    filters = []
    if action_type is not None:
        filters.append(f"action_type={action_type}")
    if corp_id is not None:
        filters.append(f"corp_id={corp_id}")
    if company_id is not None:
        filters.append(f"company_id={company_id}")
    if amount is not None:
        filters.append(f"amount={amount}")
    return filters


def find_legal_action_with_info(state, action_type=None, corp_id=None, company_id=None, amount=None):
    """Find the first legal action matching the given decoded properties.

    Returns:
        (action_id, info) tuple for the first matching legal action.

    Raises:
        AssertionError if no matching action is found.
    """
    actions = get_legal_actions(state)
    for action_id, info in actions:
        if _legal_action_matches(
            info,
            action_type=action_type,
            corp_id=corp_id,
            company_id=company_id,
            amount=amount,
        ):
            return action_id, info

    filters = _format_legal_action_filters(
        action_type=action_type,
        corp_id=corp_id,
        company_id=company_id,
        amount=amount,
    )
    available = [(aid, f"type={i.action_type} corp={i.corp_id} co={i.company_id} amt={i.amount}")
                 for aid, i in actions]
    assert False, (
        f"No legal action matching {', '.join(filters)}.\n"
        f"Available ({len(available)}): {available[:20]}"
        f"{'...' if len(available) > 20 else ''}"
    )


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
    action_id, _ = find_legal_action_with_info(
        state,
        action_type=action_type,
        corp_id=corp_id,
        company_id=company_id,
        amount=amount,
    )
    return action_id


def find_all_legal_actions_with_info(
    state, action_type=None, corp_id=None, company_id=None, amount=None
):
    """Like find_legal_action_with_info but returns all matching (action_id, info) pairs."""
    actions = get_legal_actions(state)
    result = []
    for action_id, info in actions:
        if _legal_action_matches(
            info,
            action_type=action_type,
            corp_id=corp_id,
            company_id=company_id,
            amount=amount,
        ):
            result.append((action_id, info))
    return result


def find_all_legal_actions(state, action_type=None, corp_id=None, company_id=None, amount=None):
    """Like find_legal_action but returns ALL matching action_ids as a list."""
    return [
        action_id
        for action_id, _ in find_all_legal_actions_with_info(
            state,
            action_type=action_type,
            corp_id=corp_id,
            company_id=company_id,
            amount=amount,
        )
    ]


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
        ctx = (
            f"{msg}\nIntermediate state {i}/{len(history)}, "
            f"before phase={phase_id} action={act_id}"
        )
        assert_invariants(intermediate, ctx)
        assert_token_data_invariants(intermediate, ctx)

    # Check invariants on final state (AFTER all actions)
    assert_invariants(state, f"{msg}\nFinal state after action chain")
    assert_token_data_invariants(state, f"{msg}\nFinal state after action chain")

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
        company_id: Company ID to use. If None, draws the top of the deck.
            If given, the company is routed into player ownership through the
            ownership test helpers, regardless of current location.
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
    else:
        give_company_to_player(state, company_id, player_id)

    CORPS[corp_id].float_corp(state, player_id, company_id, par_index, float_shares)
    return company_id


def setup_receivership_corp(state, corp_id, company_ids, par_index=10):
    """Float a corp and put it into receivership with the given companies.

    First company is used for floating (transferred to player 0, then floated).
    Player 0's shares are zeroed to trigger receivership. Additional companies
    are transferred directly to the active corp through the ownership helpers,
    regardless of whether they started in the deck or were already owned.

    Args:
        state: GameState
        corp_id: Corporation ID
        company_ids: List of company IDs (first used for floating)
        par_index: Market price index for starting share price (default 10)
    """
    float_corp_for_test(state, corp_id=corp_id, company_id=company_ids[0],
                        par_index=par_index)
    PLAYERS[0].set_shares(state, corp_id, 0)
    for cid in company_ids[1:]:
        give_company_to_corp(state, cid, corp_id)


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
# TOKEN DATA INVARIANTS
# =============================================================================
#
# Phase-specific tokens are laid out after the fixed tokens. The token buffer
# carries (num_players + 53, TOKEN_DIM=97) float32 rows in this order:
#
#     [0, num_players)              : player tokens (3-5p)
#     [num_players, num_players+8)  : corp tokens    (8)
#     [num_players+8, num_players+44): company tokens (36)
#     num_players+44                : FI
#     num_players+45                : Market
#     num_players+46                : Global
#     num_players+47                : Invest  (zeroed outside PHASE_INVEST)
#     num_players+48                : Auction (zeroed outside PHASE_BID)
#     num_players+49                : Dividend(zeroed outside PHASE_DIVIDENDS)
#     num_players+50                : Issue   (zeroed outside PHASE_ISSUE_SHARES)
#     num_players+51                : Par/IPO (zeroed outside PHASE_IPO)
#     num_players+52                : AcqOffer(zeroed outside PHASE_ACQ_OFFER)
#
# The 7 per-phase pass anchors live inside the model (concatenated after
# projection) and are not part of the input buffer.
#
# The feature-offset layouts below mirror core/token_data.pyx — treat that
# file as the source of truth if these ever drift.


# Feature offsets within each token type (must match core/token_data.pyx).
_PLAYER_OFF = {
    "ACTIVE":       0,
    "PLAYER_ID":    1,   # 5 slots
    "TURN_ORDER":   6,   # 5 slots
    "HAS_PASSED":   11,
    "CASH":         12,
    "NET_WORTH":    13,
    "LIQUIDITY":    14,
    "INCOME":       15,
    "SHARES":       16,  # 8 slots
    "ROUND_TRIPS":  24,
    "SHARE_BUYS":   25,  # 8 slots
    "SHARE_SELLS":  33,  # 8 slots
    "PRESIDENCIES": 41,  # 8 slots
    "COMPANIES":    49,  # 36 slots
}
_CORP_OFF = {
    "ACTIVE_CORP":   0,
    "CORP_ID":       1,   # 8 slots
    "ACTIVE":        9,
    "IN_RECV":       10,
    "PASSED_ACQ":    11,
    "UNISSUED":      12,
    "ISSUED":        13,
    "BANK":          14,
    "PRICE_IDX":     15,  # 27 slots
    "SHARE_PRICE":   42,
    "PENDING_MOVE":  43,
    "CASH":          44,
    "ACQ_PROCEEDS":  45,
    "INCOME":        46,
    "STARS":         47,
    "RAW_REVENUE":   48,
    "SYNERGY":       49,
    "COO_COST":      50,
    "ABILITY":       51,
    "PRESIDENT":     52,  # 5 slots
    "COMPANIES":     57,  # 36 slots
}
_COMPANY_OFF = {
    "ACTIVE_COMPANY": 0,
    "COMPANY_ID":     1,   # 36 slots
    "CORP_OWNER":     37,  # 8 slots
    "PLAYER_OWNER":   45,  # 5 slots
    "FI_OWNED":       50,
    "LOC_AUCTION":    51,
    "LOC_REVEALED":   52,
    "LOC_ACQ_PILE":   53,
    "LOC_REMOVED":    54,
    "ADJ_INCOME":     55,
    "LOW_PRICE":      56,
    "FACE_VALUE":     57,
    "HIGH_PRICE":     58,
    "BASE_INCOME":    59,
    "STARS":          60,
    "SYNERGIES":      61,  # 36 slots
}


def _assert_close(val, expected, tol, msg):
    """Assert val ≈ expected with absolute tolerance ``tol``."""
    diff = abs(float(val) - float(expected))
    assert diff <= tol, f"{msg} (got={val}, expected={expected}, diff={diff})"


def _assert_zero_row(row, tol, msg):
    """Assert every element of ``row`` is zero within tolerance."""
    mx = float(np.abs(row).max()) if row.size else 0.0
    assert mx <= tol, f"{msg} (max|row|={mx})"


def assert_token_data_invariants(state, msg=""):
    """Assert ``get_token_data`` produces a structurally valid buffer.

    Checks, for every intermediate state visited by the driver:
      - Buffer is the right shape and contains only finite values.
      - Player/corp/company one-hots are correct and well-formed.
      - Cash / share / price normalizations match the entity handles.
      - Ownership flags and location flags match the canonical
        `CompanyLocation` + owner slots.
      - Market availability flags mirror `MARKET.is_space_available`.
      - Global token (num_players, phase, CoO, end_card, cards_remaining)
        agrees with `TURN.get_*`.
      - Each phase-specific token is all-zero outside its phase and carries
        the expected per-phase scalars/flags inside it.
      - Pass token is always all-zero (type embedding only).

    Training is scoped to 3-5p; this function is a no-op for 2p/6p states
    (``get_token_data`` rejects those with an assert).
    """
    num_players = TURN.get_num_players(state)
    if not (3 <= num_players <= 5):
        return

    num_corps = int(GameConstants.NUM_CORPS)
    num_companies = int(GameConstants.NUM_COMPANIES)
    num_market = int(GameConstants.NUM_MARKET_SPACES)
    token_dim = int(TokenDataSize.TOKEN_DIM)

    # Token slot positions (see block comment above).
    player_base = 0
    corp_base = num_players
    company_base = num_players + num_corps
    fi_tok = company_base + num_companies
    market_tok = fi_tok + 1
    global_tok = fi_tok + 2
    invest_tok = fi_tok + 3
    auction_tok = fi_tok + 4
    dividend_tok = fi_tok + 5
    issue_tok = fi_tok + 6
    par_tok = fi_tok + 7
    acq_offer_tok = fi_tok + 8
    num_tokens = get_num_tokens(num_players)
    assert num_tokens == acq_offer_tok + 1, (
        f"{msg}\nget_num_tokens({num_players})={num_tokens} "
        f"inconsistent with layout (expected {acq_offer_tok + 1})"
    )

    buf = np.zeros((num_tokens, token_dim), dtype=np.float32)
    get_token_data(state, buf)

    # --- Basic buffer invariants ---
    assert buf.shape == (num_tokens, token_dim), (
        f"{msg}\nToken buffer shape {buf.shape} != ({num_tokens}, {token_dim})"
    )
    assert np.all(np.isfinite(buf)), (
        f"{msg}\nToken buffer contains NaN/Inf"
    )

    # Tolerance: float32 + divisions up to ~150 (CASH_DIVISOR). Looser
    # tolerance on scaled values; strict-zero for flags/one-hots.
    T_FLAG = 1e-6    # flags, one-hots, zero regions
    T_SCALE = 5e-3   # normalized scalars reconstructed through divisor

    active_player = TURN.get_active_player(state)
    active_corp = TURN.get_active_corp(state)
    active_company = TURN.get_active_company(state)
    phase = TURN.get_phase(state)

    # =========================================================================
    # PLAYER TOKENS
    # =========================================================================
    for p in range(num_players):
        tok = player_base + p
        pm = f"{msg}\nplayer token p={p}"

        # Active player flag
        expected_active = 1.0 if active_player == p else 0.0
        _assert_close(buf[tok, _PLAYER_OFF["ACTIVE"]], expected_active, T_FLAG,
                      f"{pm}: active_player flag")

        # Player ID one-hot (padded to 5; p < num_players ≤ 5)
        player_id_slice = buf[tok, _PLAYER_OFF["PLAYER_ID"]:_PLAYER_OFF["PLAYER_ID"] + 5]
        _assert_close(player_id_slice.sum(), 1.0, T_FLAG,
                      f"{pm}: player_id one-hot must sum to 1")
        _assert_close(player_id_slice[p], 1.0, T_FLAG,
                      f"{pm}: player_id one-hot bit")

        # Turn order one-hot
        to = PLAYERS[p].get_turn_order(state)
        turn_order_slice = buf[tok, _PLAYER_OFF["TURN_ORDER"]:_PLAYER_OFF["TURN_ORDER"] + 5]
        if 0 <= to < 5:
            _assert_close(turn_order_slice.sum(), 1.0, T_FLAG,
                          f"{pm}: turn_order one-hot sum")
            _assert_close(turn_order_slice[to], 1.0, T_FLAG,
                          f"{pm}: turn_order one-hot bit")
        else:
            _assert_close(turn_order_slice.sum(), 0.0, T_FLAG,
                          f"{pm}: turn_order one-hot must be zero for to={to}")

        # has_passed
        expected_passed = 1.0 if PLAYERS[p].has_passed(state) else 0.0
        _assert_close(buf[tok, _PLAYER_OFF["HAS_PASSED"]], expected_passed, T_FLAG,
                      f"{pm}: has_passed")

        # Financial scalars (reconstruct via divisor)
        _assert_close(buf[tok, _PLAYER_OFF["CASH"]] * PY_CASH_DIVISOR,
                      PLAYERS[p].get_cash(state), T_SCALE,
                      f"{pm}: cash")
        _assert_close(buf[tok, _PLAYER_OFF["NET_WORTH"]] * PY_NET_WORTH_DIVISOR,
                      PLAYERS[p].get_net_worth(state), T_SCALE,
                      f"{pm}: net_worth")
        _assert_close(buf[tok, _PLAYER_OFF["LIQUIDITY"]] * PY_NET_WORTH_DIVISOR,
                      PLAYERS[p].get_liquidity(state), T_SCALE,
                      f"{pm}: liquidity")
        _assert_close(buf[tok, _PLAYER_OFF["INCOME"]] * PY_ENTITY_INCOME_DIVISOR,
                      PLAYERS[p].get_income(state), T_SCALE,
                      f"{pm}: income")

        # Per-corp shares / buys / sells / presidencies
        any_roundtrip = False
        for c in range(num_corps):
            shares = PLAYERS[p].get_shares(state, c)
            buys = PLAYERS[p].get_share_buys(state, c)
            sells = PLAYERS[p].get_share_sells(state, c)

            _assert_close(buf[tok, _PLAYER_OFF["SHARES"] + c] * PY_SHARE_DIVISOR,
                          shares, T_SCALE, f"{pm}: shares[{c}]")
            _assert_close(buf[tok, _PLAYER_OFF["SHARE_BUYS"] + c] * PY_SHARE_DIVISOR,
                          buys, T_SCALE, f"{pm}: share_buys[{c}]")
            _assert_close(buf[tok, _PLAYER_OFF["SHARE_SELLS"] + c] * PY_SHARE_DIVISOR,
                          sells, T_SCALE, f"{pm}: share_sells[{c}]")

            expected_pres = 1.0 if CORPS[c].get_president_id(state) == p else 0.0
            _assert_close(buf[tok, _PLAYER_OFF["PRESIDENCIES"] + c], expected_pres, T_FLAG,
                          f"{pm}: presidency[{c}]")

            if buys >= 2 or sells >= 2:
                any_roundtrip = True

        expected_rt = 1.0 if any_roundtrip else 0.0
        _assert_close(buf[tok, _PLAYER_OFF["ROUND_TRIPS"]], expected_rt, T_FLAG,
                      f"{pm}: round_trips flag")

        # Owned-companies bitmap
        for cid in range(num_companies):
            loc = COMPANIES[cid].get_location(state)
            owner = COMPANIES[cid].get_owner_id(state)
            expected = 1.0 if (loc == int(CompanyLocation.LOC_PLAYER) and owner == p) else 0.0
            _assert_close(buf[tok, _PLAYER_OFF["COMPANIES"] + cid], expected, T_FLAG,
                          f"{pm}: owned_company[{cid}]")

    # =========================================================================
    # CORP TOKENS
    # =========================================================================
    for c in range(num_corps):
        tok = corp_base + c
        cm = f"{msg}\ncorp token c={c}"
        corp = CORPS[c]
        active = corp.is_active(state)

        # Active corp flag
        expected_active_corp = 1.0 if active_corp == c else 0.0
        _assert_close(buf[tok, _CORP_OFF["ACTIVE_CORP"]], expected_active_corp, T_FLAG,
                      f"{cm}: active_corp flag")

        # Corp ID one-hot (always set)
        corp_id_slice = buf[tok, _CORP_OFF["CORP_ID"]:_CORP_OFF["CORP_ID"] + num_corps]
        _assert_close(corp_id_slice.sum(), 1.0, T_FLAG, f"{cm}: corp_id one-hot sum")
        _assert_close(corp_id_slice[c], 1.0, T_FLAG, f"{cm}: corp_id one-hot bit")

        # Active / receivership / passed_acq flags
        _assert_close(buf[tok, _CORP_OFF["ACTIVE"]], 1.0 if active else 0.0, T_FLAG,
                      f"{cm}: active flag")
        _assert_close(buf[tok, _CORP_OFF["IN_RECV"]],
                      1.0 if corp.is_in_receivership(state) else 0.0, T_FLAG,
                      f"{cm}: in_receivership flag")
        _assert_close(buf[tok, _CORP_OFF["PASSED_ACQ"]],
                      1.0 if corp.has_passed_acq_offer(state) else 0.0, T_FLAG,
                      f"{cm}: passed_acq_offer flag")

        # Share counts (meaningful regardless of active status)
        _assert_close(buf[tok, _CORP_OFF["UNISSUED"]] * PY_SHARE_DIVISOR,
                      corp.get_unissued_shares(state), T_SCALE, f"{cm}: unissued")
        _assert_close(buf[tok, _CORP_OFF["ISSUED"]] * PY_SHARE_DIVISOR,
                      corp.get_issued_shares(state), T_SCALE, f"{cm}: issued")
        _assert_close(buf[tok, _CORP_OFF["BANK"]] * PY_SHARE_DIVISOR,
                      corp.get_bank_shares(state), T_SCALE, f"{cm}: bank")

        price_slice = buf[tok, _CORP_OFF["PRICE_IDX"]:_CORP_OFF["PRICE_IDX"] + num_market]
        president_slice = buf[tok, _CORP_OFF["PRESIDENT"]:_CORP_OFF["PRESIDENT"] + 5]

        if active:
            # Price index one-hot
            price_idx = corp.get_price_index(state)
            _assert_close(price_slice.sum(), 1.0, T_FLAG, f"{cm}: price_idx one-hot sum")
            _assert_close(price_slice[price_idx], 1.0, T_FLAG,
                          f"{cm}: price_idx bit at {price_idx}")

            # Financial scalars
            _assert_close(buf[tok, _CORP_OFF["SHARE_PRICE"]] * PY_SHARE_PRICE_DIVISOR,
                          corp.get_share_price(state), T_SCALE,
                          f"{cm}: share_price")
            _assert_close(buf[tok, _CORP_OFF["CASH"]] * PY_CASH_DIVISOR,
                          corp.get_cash(state), T_SCALE, f"{cm}: cash")
            _assert_close(buf[tok, _CORP_OFF["ACQ_PROCEEDS"]] * PY_CASH_DIVISOR,
                          corp.get_acquisition_proceeds(state), T_SCALE,
                          f"{cm}: acq_proceeds")
            _assert_close(buf[tok, _CORP_OFF["INCOME"]] * PY_ENTITY_INCOME_DIVISOR,
                          corp.get_income(state), T_SCALE, f"{cm}: income")
            _assert_close(buf[tok, _CORP_OFF["STARS"]] * PY_CORP_STAR_DIVISOR,
                          corp.get_total_stars(state), T_SCALE, f"{cm}: total_stars")

            # President one-hot: inactive OR in-receivership corps leave it zero
            if corp.is_in_receivership(state):
                _assert_close(president_slice.sum(), 0.0, T_FLAG,
                              f"{cm}: president one-hot must be zero in receivership")
            else:
                pres = corp.get_president_id(state)
                _assert_close(president_slice.sum(), 1.0, T_FLAG,
                              f"{cm}: president one-hot sum")
                _assert_close(president_slice[pres], 1.0, T_FLAG,
                              f"{cm}: president bit at {pres}")

            # Owned-companies bitmap = owned OR in acquisition pile
            for cid in range(num_companies):
                loc = COMPANIES[cid].get_location(state)
                owner = COMPANIES[cid].get_owner_id(state)
                is_owned = loc == int(CompanyLocation.LOC_CORP) and owner == c
                is_acq = loc == int(CompanyLocation.LOC_CORP_ACQ) and owner == c
                expected = 1.0 if (is_owned or is_acq) else 0.0
                _assert_close(buf[tok, _CORP_OFF["COMPANIES"] + cid], expected, T_FLAG,
                              f"{cm}: owned_company[{cid}]")
        else:
            # Inactive corps: price/president regions and income-side scalars all zero
            _assert_close(price_slice.sum(), 0.0, T_FLAG,
                          f"{cm}: inactive corp price_idx must be zero")
            _assert_close(president_slice.sum(), 0.0, T_FLAG,
                          f"{cm}: inactive corp president must be zero")
            for key in ("SHARE_PRICE", "PENDING_MOVE", "CASH", "ACQ_PROCEEDS",
                        "INCOME", "STARS", "RAW_REVENUE", "SYNERGY", "COO_COST",
                        "ABILITY"):
                _assert_close(buf[tok, _CORP_OFF[key]], 0.0, T_FLAG,
                              f"{cm}: inactive corp {key} must be zero")

    # =========================================================================
    # COMPANY TOKENS
    # =========================================================================
    for cid in range(num_companies):
        tok = company_base + cid
        km = f"{msg}\ncompany token cid={cid}"
        company = COMPANIES[cid]
        loc = company.get_location(state)
        owner = company.get_owner_id(state)

        # Active company flag
        _assert_close(buf[tok, _COMPANY_OFF["ACTIVE_COMPANY"]],
                      1.0 if active_company == cid else 0.0, T_FLAG,
                      f"{km}: active_company flag")

        # Company ID one-hot (always set)
        id_slice = buf[tok, _COMPANY_OFF["COMPANY_ID"]:_COMPANY_OFF["COMPANY_ID"] + num_companies]
        _assert_close(id_slice.sum(), 1.0, T_FLAG, f"{km}: company_id one-hot sum")
        _assert_close(id_slice[cid], 1.0, T_FLAG, f"{km}: company_id one-hot bit")

        # Ownership one-hot regions (mutually exclusive)
        corp_owner_slice = buf[tok, _COMPANY_OFF["CORP_OWNER"]:_COMPANY_OFF["CORP_OWNER"] + num_corps]
        player_owner_slice = buf[tok, _COMPANY_OFF["PLAYER_OWNER"]:_COMPANY_OFF["PLAYER_OWNER"] + 5]
        fi_owned = buf[tok, _COMPANY_OFF["FI_OWNED"]]

        if loc in (int(CompanyLocation.LOC_CORP), int(CompanyLocation.LOC_CORP_ACQ)):
            _assert_close(corp_owner_slice.sum(), 1.0, T_FLAG,
                          f"{km}: corp_owner one-hot sum")
            _assert_close(corp_owner_slice[owner], 1.0, T_FLAG,
                          f"{km}: corp_owner bit at {owner}")
            _assert_close(player_owner_slice.sum(), 0.0, T_FLAG,
                          f"{km}: player_owner must be zero when corp-owned")
            _assert_close(fi_owned, 0.0, T_FLAG,
                          f"{km}: fi_owned must be zero when corp-owned")
        elif loc == int(CompanyLocation.LOC_PLAYER):
            _assert_close(corp_owner_slice.sum(), 0.0, T_FLAG,
                          f"{km}: corp_owner must be zero when player-owned")
            _assert_close(player_owner_slice.sum(), 1.0, T_FLAG,
                          f"{km}: player_owner one-hot sum")
            _assert_close(player_owner_slice[owner], 1.0, T_FLAG,
                          f"{km}: player_owner bit at {owner}")
            _assert_close(fi_owned, 0.0, T_FLAG,
                          f"{km}: fi_owned must be zero when player-owned")
        elif loc == int(CompanyLocation.LOC_FI):
            _assert_close(fi_owned, 1.0, T_FLAG, f"{km}: fi_owned flag")
            _assert_close(corp_owner_slice.sum(), 0.0, T_FLAG,
                          f"{km}: corp_owner must be zero when FI-owned")
            _assert_close(player_owner_slice.sum(), 0.0, T_FLAG,
                          f"{km}: player_owner must be zero when FI-owned")
        else:
            _assert_close(corp_owner_slice.sum(), 0.0, T_FLAG,
                          f"{km}: corp_owner must be zero for loc={loc}")
            _assert_close(player_owner_slice.sum(), 0.0, T_FLAG,
                          f"{km}: player_owner must be zero for loc={loc}")
            _assert_close(fi_owned, 0.0, T_FLAG,
                          f"{km}: fi_owned must be zero for loc={loc}")

        # Location flags (mutually exclusive)
        loc_flag_map = {
            int(CompanyLocation.LOC_AUCTION):  _COMPANY_OFF["LOC_AUCTION"],
            int(CompanyLocation.LOC_REVEALED): _COMPANY_OFF["LOC_REVEALED"],
            int(CompanyLocation.LOC_CORP_ACQ): _COMPANY_OFF["LOC_ACQ_PILE"],
            int(CompanyLocation.LOC_REMOVED):  _COMPANY_OFF["LOC_REMOVED"],
        }
        loc_flag_slice = buf[tok, _COMPANY_OFF["LOC_AUCTION"]:_COMPANY_OFF["LOC_REMOVED"] + 1]
        expected_loc_flag = loc_flag_map.get(loc)
        if expected_loc_flag is not None:
            _assert_close(loc_flag_slice.sum(), 1.0, T_FLAG,
                          f"{km}: location-flag region must have exactly one bit for loc={loc}")
            _assert_close(buf[tok, expected_loc_flag], 1.0, T_FLAG,
                          f"{km}: location-flag bit for loc={loc}")
        else:
            _assert_close(loc_flag_slice.sum(), 0.0, T_FLAG,
                          f"{km}: location-flag region must be zero for loc={loc}")

        # Static data sanity (normalization scale)
        _assert_close(buf[tok, _COMPANY_OFF["FACE_VALUE"]] * PY_COMPANY_PRICE_DIVISOR,
                      company.get_face_value(), T_SCALE, f"{km}: face_value")
        _assert_close(buf[tok, _COMPANY_OFF["LOW_PRICE"]] * PY_COMPANY_PRICE_DIVISOR,
                      company.get_low_price(), T_SCALE, f"{km}: low_price")
        _assert_close(buf[tok, _COMPANY_OFF["HIGH_PRICE"]] * PY_COMPANY_PRICE_DIVISOR,
                      company.get_high_price(), T_SCALE, f"{km}: high_price")
        _assert_close(buf[tok, _COMPANY_OFF["STARS"]] * PY_COMPANY_STAR_DIVISOR,
                      company.get_stars(), T_SCALE, f"{km}: stars")
        _assert_close(buf[tok, _COMPANY_OFF["ADJ_INCOME"]] * PY_COMPANY_INCOME_DIVISOR,
                      company.get_adjusted_income(state), T_SCALE,
                      f"{km}: adjusted_income")

        # Synergy row: extractor takes max of both directions, so verify
        # against max(get_synergy_with(cid→k), get_synergy_with(k→cid)).
        # Self-slot (cid == k) is always zero.
        syn_base = _COMPANY_OFF["SYNERGIES"]
        _assert_close(buf[tok, syn_base + cid], 0.0, T_FLAG,
                      f"{km}: self-synergy slot must be zero")
        for k in range(num_companies):
            if k == cid:
                continue
            syn = max(company.get_synergy_with(k),
                      COMPANIES[k].get_synergy_with(cid))
            _assert_close(buf[tok, syn_base + k] * PY_COMPANY_INCOME_DIVISOR,
                          syn, T_SCALE, f"{km}: synergy[{k}]")

    # =========================================================================
    # FI TOKEN
    # =========================================================================
    fm = f"{msg}\nFI token"
    _assert_close(buf[fi_tok, 0] * PY_CASH_DIVISOR, FI.get_cash(state), T_SCALE,
                  f"{fm}: cash")
    _assert_close(buf[fi_tok, 1] * PY_ENTITY_INCOME_DIVISOR, FI.get_income(state),
                  T_SCALE, f"{fm}: income")
    # Owned-companies bitmap: 1.0 exactly when company is at LOC_FI
    for cid in range(num_companies):
        expected = 1.0 if COMPANIES[cid].get_location(state) == int(CompanyLocation.LOC_FI) else 0.0
        _assert_close(buf[fi_tok, 2 + cid], expected, T_FLAG,
                      f"{fm}: owned[{cid}]")
    # Tail beyond the 36-flag region must be zero
    _assert_zero_row(buf[fi_tok, 2 + num_companies:], T_FLAG,
                     f"{fm}: tail beyond owned bitmap")

    # =========================================================================
    # MARKET TOKEN
    # =========================================================================
    mm = f"{msg}\nMarket token"
    # Boundary spaces always available
    _assert_close(buf[market_tok, 0], 1.0, T_FLAG,
                  f"{mm}: slot 0 ($0) must always be available")
    _assert_close(buf[market_tok, num_market - 1], 1.0, T_FLAG,
                  f"{mm}: slot {num_market - 1} ($75) must always be available")
    for i in range(num_market):
        expected = 1.0 if MARKET.is_space_available(state, i) else 0.0
        _assert_close(buf[market_tok, i], expected, T_FLAG,
                      f"{mm}: availability slot {i}")
    # Tail beyond the 27 slots must be zero
    _assert_zero_row(buf[market_tok, num_market:], T_FLAG,
                     f"{mm}: tail beyond availability flags")

    # =========================================================================
    # GLOBAL TOKEN
    # =========================================================================
    gm = f"{msg}\nGlobal token"
    OFF_NUM_PLAYERS = 0
    OFF_PHASE = 3
    OFF_COO = 11
    OFF_END_CARD = 18
    OFF_CARDS_REM = 19

    # num_players one-hot (3 slots, 3p→0, 4p→1, 5p→2)
    np_slice = buf[global_tok, OFF_NUM_PLAYERS:OFF_NUM_PLAYERS + 3]
    _assert_close(np_slice.sum(), 1.0, T_FLAG, f"{gm}: num_players one-hot sum")
    _assert_close(np_slice[num_players - 3], 1.0, T_FLAG,
                  f"{gm}: num_players bit at slot {num_players - 3}")

    # Phase one-hot (8 slots). Active only for decision phases.
    phase_slice = buf[global_tok, OFF_PHASE:OFF_PHASE + 8]
    dp = ENGINE_TO_DECISION_PHASE[phase] if 0 <= phase < 12 else -1
    if 0 <= dp < 8:
        _assert_close(phase_slice.sum(), 1.0, T_FLAG, f"{gm}: phase one-hot sum")
        _assert_close(phase_slice[dp], 1.0, T_FLAG,
                      f"{gm}: phase bit at decision slot {dp} (engine phase={phase})")
    else:
        _assert_close(phase_slice.sum(), 0.0, T_FLAG,
                      f"{gm}: phase one-hot must be zero for automated/terminal phase={phase}")

    # CoO one-hot (7 slots; level 1..7 → slots 0..6)
    coo_slice = buf[global_tok, OFF_COO:OFF_COO + 7]
    coo = TURN.get_coo_level(state)
    if 1 <= coo <= 7:
        _assert_close(coo_slice.sum(), 1.0, T_FLAG, f"{gm}: CoO one-hot sum")
        _assert_close(coo_slice[coo - 1], 1.0, T_FLAG,
                      f"{gm}: CoO bit for level {coo}")
    else:
        _assert_close(coo_slice.sum(), 0.0, T_FLAG,
                      f"{gm}: CoO one-hot must be zero for level {coo}")

    _assert_close(buf[global_tok, OFF_END_CARD],
                  1.0 if TURN.is_end_card_flipped(state) else 0.0, T_FLAG,
                  f"{gm}: end_card flag")
    _assert_close(buf[global_tok, OFF_CARDS_REM] * num_companies,
                  TURN.get_cards_remaining(state), T_SCALE,
                  f"{gm}: cards_remaining")

    # Tail beyond the 20 defined slots must be zero
    _assert_zero_row(buf[global_tok, OFF_CARDS_REM + 1:], T_FLAG,
                     f"{gm}: tail beyond global token features")

    # =========================================================================
    # PHASE-SPECIFIC TOKENS
    # =========================================================================
    # Each phase-specific token is zeroed out by ``get_token_data`` when the
    # engine is not in the corresponding phase. When it IS in the phase, the
    # key scalars / bit-flags must match the state.

    # Invest token --------------------------------------------------------
    # The extractor carries its own private ``_find_next_higher/lower_space``
    # copies. Verify each active corp's buy/sell impact matches the canonical
    # ``MARKET.find_next_{higher,lower}_space`` result — precisely the drift
    # an invariant should catch.
    im = f"{msg}\nInvest token"
    if phase != int(GamePhases.PHASE_INVEST):
        _assert_zero_row(buf[invest_tok], T_FLAG,
                         f"{im} must be all-zero outside PHASE_INVEST (phase={phase})")
    else:
        # passes / CONSECUTIVE_PASSES_DIVISOR = 5.0
        _assert_close(buf[invest_tok, 0] * 5.0,
                      TURN.get_consecutive_passes(state), T_SCALE,
                      f"{im}: consecutive_passes")
        for c in range(num_corps):
            if CORPS[c].is_active(state):
                cur = CORPS[c].get_price_index(state)
                exp_buy = MARKET.find_next_higher_space(state, cur) - cur
                exp_sell = MARKET.find_next_lower_space(state, cur) - cur
                _assert_close(buf[invest_tok, 1 + c] * PY_IMPACT_DIVISOR,
                              exp_buy, T_SCALE,
                              f"{im}: corp {c} buy_impact")
                _assert_close(buf[invest_tok, 9 + c] * PY_IMPACT_DIVISOR,
                              exp_sell, T_SCALE,
                              f"{im}: corp {c} sell_impact")
            else:
                _assert_close(buf[invest_tok, 1 + c], 0.0, T_FLAG,
                              f"{im}: inactive corp {c} buy_impact must be zero")
                _assert_close(buf[invest_tok, 9 + c], 0.0, T_FLAG,
                              f"{im}: inactive corp {c} sell_impact must be zero")
        # Tail beyond the 17 defined slots must be zero
        _assert_zero_row(buf[invest_tok, 17:], T_FLAG, f"{im}: tail")

    # Auction token -------------------------------------------------------
    am = f"{msg}\nAuction token"
    if phase != int(GamePhases.PHASE_BID):
        _assert_zero_row(buf[auction_tok], T_FLAG,
                         f"{am} must be all-zero outside PHASE_BID (phase={phase})")
    else:
        # Slots represent the *minimum legal next bid*:
        #   - first bid (high_bidder == -1): min = face_value (offset 0)
        #   - otherwise: min = auction_price + 1
        high = TURN.get_auction_high_bidder(state)
        is_first_bid = high < 0
        if 0 <= active_company < num_companies:
            face = COMPANIES[active_company].get_face_value()
            if is_first_bid:
                min_bid = face
            else:
                min_bid = TURN.get_auction_price(state) + 1
            exp_offset = (min_bid - face) / 15.0
            _assert_close(buf[auction_tok, 0], exp_offset, T_SCALE,
                          f"{am}: min_bid_idx offset (min-face)/15")
            _assert_close(buf[auction_tok, 1] * PY_COMPANY_PRICE_DIVISOR,
                          min_bid, T_SCALE,
                          f"{am}: min_bid_value scalar")
        else:
            _assert_close(buf[auction_tok, 0], 0.0, T_FLAG,
                          f"{am}: min_bid_idx must be zero with no active company")
            _assert_close(buf[auction_tok, 1], 0.0, T_FLAG,
                          f"{am}: min_bid_value must be zero with no active company")

        _assert_close(buf[auction_tok, 2], 1.0 if is_first_bid else 0.0, T_FLAG,
                      f"{am}: is_first_bid flag")

        # high_bidder / starter one-hots (zero when the respective field is -1)
        high_slice = buf[auction_tok, 3:8]
        starter_slice = buf[auction_tok, 8:13]
        starter = TURN.get_auction_starter(state)
        if 0 <= high < 5:
            _assert_close(high_slice.sum(), 1.0, T_FLAG,
                          f"{am}: high_bidder one-hot sum")
            _assert_close(high_slice[high], 1.0, T_FLAG,
                          f"{am}: high_bidder bit at {high}")
        else:
            _assert_close(high_slice.sum(), 0.0, T_FLAG,
                          f"{am}: high_bidder one-hot must be zero for high={high}")
        if 0 <= starter < 5:
            _assert_close(starter_slice.sum(), 1.0, T_FLAG,
                          f"{am}: starter one-hot sum")
            _assert_close(starter_slice[starter], 1.0, T_FLAG,
                          f"{am}: starter bit at {starter}")
        else:
            _assert_close(starter_slice.sum(), 0.0, T_FLAG,
                          f"{am}: starter one-hot must be zero for starter={starter}")
        _assert_zero_row(buf[auction_tok, 13:], T_FLAG, f"{am}: tail")

    # Dividend token ------------------------------------------------------
    dm = f"{msg}\nDividend token"
    if phase != int(GamePhases.PHASE_DIVIDENDS):
        _assert_zero_row(buf[dividend_tok], T_FLAG,
                         f"{dm} must be all-zero outside PHASE_DIVIDENDS (phase={phase})")
    else:
        # Per-amount impact (26 slots, amounts 0..25) verified via the
        # canonical ``simulate_dividend_price_move`` — same helper the
        # extractor now calls, so drift between them is impossible by
        # construction. We still verify the slots to catch off-by-one /
        # scaling bugs in the extractor's indexing.
        if 0 <= active_corp < num_corps and CORPS[active_corp].is_active(state):
            for amount in range(26):
                exp = CORPS[active_corp].simulate_dividend_price_move(state, amount)
                _assert_close(buf[dividend_tok, amount] * PY_IMPACT_DIVISOR,
                              exp, T_SCALE,
                              f"{dm}: impact[amount={amount}] for corp {active_corp}")
        else:
            _assert_zero_row(buf[dividend_tok, 0:26], T_FLAG,
                             f"{dm}: impact region must be zero with no active corp")
        for c in range(num_corps):
            expected = 1.0 if TURN.is_dividend_remaining(state, c) else 0.0
            _assert_close(buf[dividend_tok, 26 + c], expected, T_FLAG,
                          f"{dm}: remaining[{c}]")
        _assert_zero_row(buf[dividend_tok, 26 + num_corps:], T_FLAG, f"{dm}: tail")

    # Issue token ---------------------------------------------------------
    isum = f"{msg}\nIssue token"
    if phase != int(GamePhases.PHASE_ISSUE_SHARES):
        _assert_zero_row(buf[issue_tok], T_FLAG,
                         f"{isum} must be all-zero outside PHASE_ISSUE_SHARES "
                         f"(phase={phase})")
    else:
        # Impact: issuing one share moves the active corp's price one "sell"
        # step lower, except Stock Masters (SM) which has no price change.
        if 0 <= active_corp < num_corps and CORPS[active_corp].is_active(state):
            if active_corp == int(CorpIndices.CORP_SM):
                _assert_close(buf[issue_tok, 0], 0.0, T_FLAG,
                              f"{isum}: SM impact must be zero")
            else:
                cur = CORPS[active_corp].get_price_index(state)
                exp_delta = MARKET.find_next_lower_space(state, cur) - cur
                _assert_close(buf[issue_tok, 0] * PY_IMPACT_DIVISOR, exp_delta,
                              T_SCALE, f"{isum}: issue_impact for corp {active_corp}")
        else:
            _assert_close(buf[issue_tok, 0], 0.0, T_FLAG,
                          f"{isum}: impact must be zero with no active corp")
        for c in range(num_corps):
            expected = 1.0 if TURN.is_issue_remaining(state, c) else 0.0
            _assert_close(buf[issue_tok, 1 + c], expected, T_FLAG,
                          f"{isum}: remaining[{c}]")
        _assert_zero_row(buf[issue_tok, 1 + num_corps:], T_FLAG, f"{isum}: tail")

    # Par / IPO token -----------------------------------------------------
    pm2 = f"{msg}\nPar/IPO token"
    if phase != int(GamePhases.PHASE_IPO):
        _assert_zero_row(buf[par_tok], T_FLAG,
                         f"{pm2} must be all-zero outside PHASE_IPO (phase={phase})")
    else:
        # Per-par-price outputs (player_cash / corp_cash / issued_shares)
        # verified via the canonical ``simulate_float`` helper — same
        # helper the extractor now calls. For invalid par indices (per
        # the company's star tier) the extractor writes zero.
        if 0 <= active_company < num_companies:
            star_tier = COMPANIES[active_company].get_stars()
            # Any corp works — simulate_float is a pure function of
            # (face_value, par_index), so we borrow CORPS[0].
            for par_index in range(14):
                if 1 <= star_tier <= 5 and PY_PAR_PRICE_VALID[star_tier - 1][par_index]:
                    float_result = CORPS[0].simulate_float(active_company, par_index)
                    player_pmt = float_result[2]
                    corp_cash_after = float_result[3]
                    issued = float_result[4]
                    _assert_close(buf[par_tok, 0 + par_index] * PY_CASH_DIVISOR,
                                  player_pmt, T_SCALE,
                                  f"{pm2}: player_cash[par={par_index}]")
                    _assert_close(buf[par_tok, 14 + par_index] * PY_CASH_DIVISOR,
                                  corp_cash_after, T_SCALE,
                                  f"{pm2}: corp_cash[par={par_index}]")
                    # Issued shares normalized by FLOAT_SHARES_MAX = 4.0
                    _assert_close(buf[par_tok, 28 + par_index] * 4.0,
                                  issued, T_SCALE,
                                  f"{pm2}: issued_shares[par={par_index}]")
                else:
                    # Invalid par for this star tier → all three slots zero
                    _assert_close(buf[par_tok, 0 + par_index], 0.0, T_FLAG,
                                  f"{pm2}: player_cash[par={par_index}] "
                                  f"must be zero (invalid for star_tier={star_tier})")
                    _assert_close(buf[par_tok, 14 + par_index], 0.0, T_FLAG,
                                  f"{pm2}: corp_cash[par={par_index}] "
                                  f"must be zero (invalid for star_tier={star_tier})")
                    _assert_close(buf[par_tok, 28 + par_index], 0.0, T_FLAG,
                                  f"{pm2}: issued_shares[par={par_index}] "
                                  f"must be zero (invalid for star_tier={star_tier})")

        # "IPO remaining" in the par token means "corp is not yet floated"
        # (i.e. still inactive and therefore available to be selected).
        for c in range(num_corps):
            expected = 1.0 if not CORPS[c].is_active(state) else 0.0
            _assert_close(buf[par_tok, 42 + c], expected, T_FLAG,
                          f"{pm2}: remaining[{c}]")
        _assert_zero_row(buf[par_tok, 42 + num_corps:], T_FLAG, f"{pm2}: tail")

    # Acq-offer token -----------------------------------------------------
    aom = f"{msg}\nAcq-offer token"
    if phase != int(GamePhases.PHASE_ACQ_OFFER):
        _assert_zero_row(buf[acq_offer_tok], T_FLAG,
                         f"{aom} must be all-zero outside PHASE_ACQ_OFFER "
                         f"(phase={phase})")
    else:
        # Price index scalar: (offer_price - low_price) / 51 (ACQ_PRICE_OFFSETS)
        if 0 <= active_company < num_companies:
            low = COMPANIES[active_company].get_low_price()
            exp_offset = (TURN.get_acq_offer_price(state) - low) / 51.0
            _assert_close(buf[acq_offer_tok, 0], exp_offset, T_SCALE,
                          f"{aom}: price_idx offset (offer-low)/51")
            # FI_COMPANY flag (slot 10): set iff active_company is FI-owned
            exp_fi = 1.0 if COMPANIES[active_company].get_location(state) == int(CompanyLocation.LOC_FI) else 0.0
            _assert_close(buf[acq_offer_tok, 10], exp_fi, T_FLAG,
                          f"{aom}: FI_COMPANY flag")
        else:
            _assert_close(buf[acq_offer_tok, 0], 0.0, T_FLAG,
                          f"{aom}: price_idx must be zero with no active company")
            _assert_close(buf[acq_offer_tok, 10], 0.0, T_FLAG,
                          f"{aom}: FI_COMPANY flag must be zero with no active company")
        _assert_close(buf[acq_offer_tok, 1] * PY_COMPANY_PRICE_DIVISOR,
                      TURN.get_acq_offer_price(state), T_SCALE,
                      f"{aom}: offer_price scalar")
        corp_slice = buf[acq_offer_tok, 2:2 + num_corps]
        offer_corp = TURN.get_acq_offer_corp(state)
        if 0 <= offer_corp < num_corps:
            _assert_close(corp_slice.sum(), 1.0, T_FLAG,
                          f"{aom}: offer_corp one-hot sum")
            _assert_close(corp_slice[offer_corp], 1.0, T_FLAG,
                          f"{aom}: offer_corp bit at {offer_corp}")
        else:
            _assert_close(corp_slice.sum(), 0.0, T_FLAG,
                          f"{aom}: offer_corp one-hot must be zero when no offer")
        _assert_zero_row(buf[acq_offer_tok, 2 + num_corps + 1:], T_FLAG,
                         f"{aom}: tail beyond FI_COMPANY flag")



# =============================================================================
# AUTO-PHASE HELPERS
# =============================================================================

def make_auto_phase_state(num_players, engine_phase, seed=42):
    """Fresh state placed into ``engine_phase`` with no decision-phase context.

    Initializes via ``GameState.initialize_game`` (which lands in
    PHASE_INVEST), then flips the phase and clears the decision-phase
    sentinels that would have been set by the preceding decision phase.
    Caller is responsible for any further setup (floating corps, setting
    FI cash, relocating companies, etc.) specific to the scenario under
    test.
    """
    state = GameState(num_players)
    state.initialize_game(num_players, seed=seed)
    TURN.set_phase(state, engine_phase)
    TURN.clear_active_corp(state)
    TURN.clear_active_company(state)
    TURN.clear_acq_offer_price(state)
    return state


def assert_post_auto(state, expected_phase, msg=""):
    """Wrap the standard invariant pair with an explicit phase check.

    Catches the common bug of "auto phase mutated state but forgot to
    transition" or "transitioned to the wrong next phase".
    """
    actual_phase = TURN.get_phase(state)
    assert actual_phase == expected_phase, (
        f"{msg}\nExpected phase {expected_phase}, got {actual_phase}"
    )
    assert_invariants(state, msg)
    assert_token_data_invariants(state, msg)


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
