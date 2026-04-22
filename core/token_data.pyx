"""
Token data extraction: compact GameState -> transformer eval buffer.

``get_token_data`` is the sole engine→NN interface: it fills a
(num_tokens, TOKEN_DIM) float32 buffer with normalized per-token
features from a compact GameState. It is called once per NN evaluation,
so the trunk + MCTS throughput depends on this being fast.

Token order (matches ``nn/transformer.py``):
    # Static data tokens (pure game-setup data; layout-stable across a game).
    # Placed first so a self-play worker can prefill them once into the shared
    # buffer and skip rewriting them on every extraction (see beads qa4m).
    [market_slot_prices, companies...,
    # Dynamic informational tokens.
     market_availability,
     company_removed, company_auction, company_revealed, company_acq_pile,
     company_adjusted_income, FI,
     active_player, active_corp, active_company,
     phase, num_players, game_progress,
    # Phase-specific tokens (left zero unless the engine is in the matching phase).
     invest, auction, dividend, issue, par, acq_offer, acq_price_info,
    # Corp tokens, then player tokens — players last so the buffer can be padded
    # for higher player counts later with the extra rows masked out in attention.
     corps..., players...]

Total tokens in the buffer = num_players + 65. The model concatenates 7
learned pass-token anchors to the projected trunk sequence internally
(one per pass-using decision phase — INVEST, BID, ACQ_SELECT_CORP,
ACQ_OFFER, CLOSING, ISSUE, IPO — the rest have no pass action), so the
input buffer carries no pass rows.

Per-token feature layouts (sum of widths ≤ TOKEN_DIM = 92 = max width,
currently pinned by the Corp token):

  Player (84):  player_id onehot (5) + turn_order onehot (5) + has_passed
                (1) + cash (1) + net_worth (1) + liquidity (1) + income
                (1) + owned_shares (8) + round_trips (1) + share_buys (8)
                + share_sells (8) + presidencies (8) + owned_companies
                (36)
  Corp   (92):  corp_id onehot (8) + active (1) + in_receivership (1) +
                passed_acq_offer (1) + unissued/issued/bank shares (3) +
                price_index onehot (27) + share_price (1) +
                pending_price_move (1) + cash (1) + acq_proceeds (1) +
                income (1) + stars (1) + raw_revenue (1) + synergy_income
                (1) + coo_cost (1) + ability_income (1) + president_id
                onehot (5) + owned_companies (36)
  Company (78): company_id onehot (36) + static data [low/face/high/
                low_high_diff/income/stars] (6) + synergies (36). This
                token is now pure static game-setup data — ownership,
                location, CoO-adjusted income, and active-entity selection
                all live in dedicated tokens. ``low_high_diff`` is the
                ACQ_SELECT_PRICE offset count (``high - low + 1``), the
                same quantity the price head conditions on, normalized by
                PRICE_RANGE_DIVISOR (max is 51 for CDG).
  FI      (38): cash + income + owned_companies (36)
  MarketAvail (27): availability flags (27)
  MarketPrices (27): static market-space prices normalized by
                SHARE_PRICE_DIVISOR ($0..$75 / 75)
  CompanyLoc* (36, four tokens): 1 per company_id if the company is
                currently at the matching location, else 0. The four
                tokens (in buffer order) cover LOC_REMOVED, LOC_AUCTION,
                LOC_REVEALED, and LOC_CORP_ACQ. The LOC_REMOVED bitmap
                additionally flags setup-excluded companies (LOC_EXCLUDED)
                once CoO has advanced past their star tier — the deck is
                past that colour group, so the exclusion is publicly
                observable and the bit can be set without leaking setup
                randomness.
  CompanyAdjIncome (36): per-company CoO-adjusted income normalized by
                COMPANY_INCOME_DIVISOR (may be negative).
  ActivePlayer (5): one-hot over active player (padded to 5 slots);
                all-zero when no active player is selected.
  ActiveCorp (8): one-hot over active corp; all-zero when unset.
  ActiveCompany (36): one-hot over active company; all-zero when unset.
  Phase   (11): decision phase onehot (11)
  NumPlyr  (3): num_players onehot (3)
  Progress (9): CoO onehot (7) + end_card_flipped + cards_remaining
  Invest  (17): consecutive_passes + buy_impacts (8) + sell_impacts (8)
  Auction (13): min_bid_index + min_bid_value + is_first_bid +
                high_bidder onehot (5) + starter onehot (5)
  Divd    (34): dividend_impacts (26) + dividend_remaining (8)
  Issue    (9): issue_impact + issue_remaining (8)
  IPO     (50): player_cash_required (14) + resulting_corp_cash (14) +
                resulting_issued_shares (14) + ipo_remaining (8)
  AcqOff  (11): offer_price_index + offer_price + offer_corp onehot (8) +
                fi_company
  AcqPrice (3): max_offset (ACQ offset count for target) + fi_flag +
                total_synergies (marginal synergy income the active corp
                would gain by adding the target company)

All values normalized by divisors defined in ``core.data`` (compile-time
floats inlined by the C compiler). Phase-specific tokens are zeroed out
when the current engine phase does not match. The function is designed
so the post-GIL body runs ``nogil``; a small Python-level prologue
forces per-player cache refreshes so the nogil body can read cached
net_worth / liquidity / income slots directly.
"""

import numpy as np

from libc.stdint cimport int16_t
from libc.string cimport memset

from core.state cimport (
    GameState, LAYOUT, TURN_OFFSETS, CORP_FIELDS, PLAYER_FIELDS,
    COMPANY_OFFSETS, FI_OFFSETS,
)
from core.data cimport (
    GameConstants,
    GamePhases,
    DecisionPhase,
    ENGINE_TO_DECISION_PHASE,
    CorpIndices,
    COMPANY_FACE_VALUE,
    COMPANY_LOW_PRICE,
    COMPANY_HIGH_PRICE,
    COMPANY_STARS,
    COMPANY_INCOME,
    COMPANY_SYNERGY,
    MARKET_PRICES,
    PAR_PRICE_VALID,
    CASH_DIVISOR,
    NET_WORTH_DIVISOR,
    COMPANY_INCOME_DIVISOR,
    COMPANY_SYNERGY_DIVISOR,
    ENTITY_INCOME_DIVISOR,
    SHARE_DIVISOR,
    COMPANY_PRICE_DIVISOR,
    SHARE_PRICE_DIVISOR,
    COMPANY_STAR_DIVISOR,
    CORP_STAR_DIVISOR,
    IMPACT_DIVISOR,
    PRICE_RANGE_DIVISOR,
)
from entities.corp cimport (
    corp_is_active,
    corp_cash,
    corp_unissued_shares,
    corp_issued_shares,
    corp_bank_shares,
    corp_price_index,
    corp_share_price,
    corp_acquisition_proceeds,
    corp_is_in_receivership,
    corp_president_id,
    corp_has_passed_acq_offer,
    corp_owns_company,
    corp_has_acquisition_company,
    corp_income,
    corp_raw_revenue,
    corp_synergy_income,
    corp_coo_cost,
    corp_ability_income,
    corp_total_stars,
    corp_pending_price_move,
    corp_candidate_synergy_delta,
    _simulate_dividend_price_move,
    _simulate_float,
)
from entities.company cimport (
    LOC_AUCTION, LOC_REVEALED, LOC_PLAYER, LOC_FI, LOC_CORP, LOC_CORP_ACQ,
    LOC_REMOVED, LOC_EXCLUDED,
    company_location,
    company_owner_id,
    company_adjusted_income,
)
from entities.market cimport (
    market_find_next_higher_space,
    market_find_next_lower_space,
)
from entities.player cimport refresh_player_cache_if_dirty


# =============================================================================
# PER-TOKEN FEATURE COUNTS (single source of truth for offset arithmetic)
# =============================================================================

# Fixed token layout constants (see nn/transformer.py). These match the
# token slicing bookkeeping in ``RSSTransformerNet.__init__`` for 3-5p.
DEF NUM_CORPS = 8
DEF NUM_COMPANIES = 36
DEF NUM_MARKET_SPACES = 27
DEF NUM_DECISION_PHASES = 11
DEF NUM_COO_LEVELS = 7
DEF MAX_MODEL_PLAYERS = 5          # 3-5p supported; one-hots are padded to 5
DEF AUCTION_CAP_INT = 15           # INVEST auction price offsets per company
DEF NUM_PAR_PRICES = 14
DEF MAX_DIVIDEND = 26              # dividend amounts 0..25 (26 slots)
DEF ACQ_PRICE_OFFSETS = 51         # acquisition price offsets (matches action encoding)
DEF FLOAT_SHARES_MAX = 4.0         # max issued shares at float (face>par → 4)
DEF ROUNDTRIP_LIMIT = 2            # share buy+sell limit per corp per turn

# Normalization constant for the invest token's consecutive_passes slot.
# Matches the max training player count (5).
DEF CONSECUTIVE_PASSES_DIVISOR = 5.0


# =============================================================================
# PUBLIC API
# =============================================================================

cpdef int get_num_tokens(int num_players) noexcept nogil:
    """Input-buffer token count for the given player count (num_players + 65).

    The model-side trunk is wider by 7 (per-phase pass anchors concatenated
    inside ``RSSTransformerNet._project_tokens``), but those rows are
    learned anchors with no input features, so the engine-side buffer
    doesn't carry them.
    """
    return num_players + 65


cpdef object get_token_widths(int num_players):
    """Per-position non-padded feature widths matching ``_fill_buffer``.

    Each ``buffer[i]`` row is TOKEN_DIM wide but only the first
    ``widths[i]`` slots carry features — the rest are zero padding. The
    returned array mirrors the buffer layout (static data first, then
    dynamic info, phase-specific, corp, player) so the caller can slice
    ``buffer[i, :widths[i]]`` or group positions by type for per-type
    projection modules without duplicating the layout logic.

    Returns a uint8 ``(num_players + 65,)`` numpy array; all widths fit
    in a byte (max is TW_CORP = 92 < 256).
    """
    assert 3 <= num_players <= 5, \
        f"get_token_widths: num_players must be 3-5, got {num_players}"

    cdef int num_tokens = num_players + 65
    widths = np.empty(num_tokens, dtype=np.uint8)
    cdef unsigned char[::1] w = widths

    cdef int i
    cdef int tok = 0

    # Static data tokens
    w[tok] = <unsigned char>TokenWidth.TW_MARKET_SLOT_PRICES
    tok += 1
    for i in range(NUM_COMPANIES):
        w[tok] = <unsigned char>TokenWidth.TW_COMPANY
        tok += 1

    # Dynamic informational tokens
    w[tok] = <unsigned char>TokenWidth.TW_MARKET_AVAILABILITY
    tok += 1
    for i in range(4):  # REMOVED, AUCTION, REVEALED, CORP_ACQ
        w[tok] = <unsigned char>TokenWidth.TW_COMPANY_LOCATION
        tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_COMPANY_ADJ_INCOME
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_FI
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_ACTIVE_PLAYER
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_ACTIVE_CORP
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_ACTIVE_COMPANY
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_PHASE
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_NUM_PLAYERS
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_GAME_PROGRESS
    tok += 1

    # Phase-specific tokens — the slots exist regardless of the current
    # phase (they're just zero-filled when the phase doesn't match), so
    # the widths here are the per-phase logical widths.
    w[tok] = <unsigned char>TokenWidth.TW_INVEST
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_AUCTION
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_DIVIDEND
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_ISSUE
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_PAR
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_ACQ_OFFER
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_ACQ_PRICE
    tok += 1

    # Corp tokens
    for i in range(NUM_CORPS):
        w[tok] = <unsigned char>TokenWidth.TW_CORP
        tok += 1

    # Player tokens (trailing)
    for i in range(num_players):
        w[tok] = <unsigned char>TokenWidth.TW_PLAYER
        tok += 1

    return widths


cpdef void get_token_data(GameState state, float[:, ::1] buffer):
    """Fill ``buffer`` with per-token NN features for ``state``.

    ``buffer`` must be a writable C-contiguous float32 memoryview at least
    ``(num_players + 65, TOKEN_DIM)`` in size. Training is scoped to
    3-5 players; other player counts are rejected.

    The cache-refresh prologue and ``_fill_buffer`` run in a single nogil
    block — refresh goes through the module-level
    ``refresh_player_cache_if_dirty`` helper rather than the Python-level
    ``PLAYERS[i].get_net_worth(state)`` lookup it used to.
    """
    cdef int num_players = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.num_players]
    cdef int num_tokens = num_players + 65
    cdef int i

    assert 3 <= num_players <= 5, \
        f"get_token_data: num_players must be 3-5, got {num_players}"
    assert buffer.shape[0] >= num_tokens, \
        f"get_token_data: buffer rows {buffer.shape[0]} < num_tokens {num_tokens}"
    # Exact-match on the padded width: the nogil memset in ``_fill_buffer``
    # writes ``num_tokens * TOKEN_DIM * 4`` contiguous bytes, so a wider
    # buffer would silently clobber across rows.
    assert buffer.shape[1] == <int>TokenDataSize.TOKEN_DIM, \
        f"get_token_data: buffer cols {buffer.shape[1]} != TOKEN_DIM {<int>TokenDataSize.TOKEN_DIM}"

    with nogil:
        for i in range(num_players):
            refresh_player_cache_if_dirty(state, i)
        _fill_buffer(state, buffer, num_players, num_tokens)


cpdef void get_token_data_batch(
    list state_arrays,
    int num_players,
    float[:, :, ::1] buffer,
):
    """Batched ``get_token_data``: fill ``buffer[i]`` for each ``state_arrays[i]``.

    Reuses a single scratch ``GameState`` across all rows via ``rebind``
    (zero-copy). Each row's cache refresh + fill runs in a single nogil
    block; the only GIL-held op per iteration is ``rebind`` (Python-level
    validation + ``_array`` attribute write).

    Args:
        state_arrays: List of writable C-contiguous int16 state arrays, one
            per leaf. Every entry must size-match the shared ``num_players``
            layout (same constraint as ``GameState.rebind``).
        num_players: Training player count (3-5). Applies to every state
            array in the batch — mixed-player batches are not supported.
        buffer: ``(n, num_players + 65, TOKEN_DIM)`` float32 output, C-contig.
    """
    cdef int n = len(state_arrays)
    cdef int num_tokens = num_players + 65
    cdef int i, p
    cdef GameState scratch_gs

    assert 3 <= num_players <= 5, \
        f"get_token_data_batch: num_players must be 3-5, got {num_players}"
    if n == 0:
        return
    assert buffer.shape[0] >= n, \
        f"get_token_data_batch: buffer batch {buffer.shape[0]} < n {n}"
    assert buffer.shape[1] >= num_tokens, \
        f"get_token_data_batch: buffer rows {buffer.shape[1]} < num_tokens {num_tokens}"
    # Exact-match on the padded width: see ``get_token_data`` for the same
    # constraint — memset writes assume rows are tightly packed at TOKEN_DIM.
    assert buffer.shape[2] == <int>TokenDataSize.TOKEN_DIM, \
        f"get_token_data_batch: buffer cols {buffer.shape[2]} != TOKEN_DIM {<int>TokenDataSize.TOKEN_DIM}"

    scratch_gs = GameState.from_buffer(state_arrays[0], num_players)

    for i in range(n):
        if i > 0:
            scratch_gs.rebind(state_arrays[i], num_players)

        with nogil:
            for p in range(num_players):
                refresh_player_cache_if_dirty(scratch_gs, p)
            _fill_buffer(scratch_gs, buffer[i], num_players, num_tokens)


# =============================================================================
# INTERNAL: MAIN FILL DRIVER
# =============================================================================

cdef void _fill_buffer(
    GameState state,
    float[:, ::1] buffer,
    int num_players,
    int num_tokens,
) noexcept nogil:
    cdef int i, tok
    cdef int phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]

    # Zero the region we'll write. Phase-specific tokens rely on this to
    # stay at zero when the current phase does not match.
    memset(&buffer[0, 0], 0, num_tokens * <int>TokenDataSize.TOKEN_DIM * sizeof(float))

    tok = 0

    # --- Static data tokens (pre-populatable; see beads qa4m) ---

    # Market slot-prices token (static $0..$75 normalized)
    _fill_market_slot_prices_token(buffer, tok)
    tok += 1

    # Company tokens (static game-setup data — face/low/high/income/stars/synergies)
    for i in range(NUM_COMPANIES):
        _fill_company_token(state, buffer, tok, i, num_players)
        tok += 1

    # --- Dynamic informational tokens ---

    # Market availability token (27 per-space flags)
    _fill_market_availability_token(state, buffer, tok)
    tok += 1

    # Company-location tokens: one 36-wide bitmap per target location, with
    # a 1 at company_id if that company is currently at the location. The
    # REMOVED bitmap also flags LOC_EXCLUDED companies whose tier the CoO
    # has moved past (see _fill_company_removed_token).
    _fill_company_removed_token(state, buffer, tok)
    tok += 1
    _fill_company_location_token(state, buffer, tok, <int>LOC_AUCTION)
    tok += 1
    _fill_company_location_token(state, buffer, tok, <int>LOC_REVEALED)
    tok += 1
    _fill_company_location_token(state, buffer, tok, <int>LOC_CORP_ACQ)
    tok += 1

    # Company-adjusted-income token: per-company CoO-adjusted income, 36 wide.
    _fill_company_adjusted_income_token(state, buffer, tok)
    tok += 1

    # FI token
    _fill_fi_token(state, buffer, tok)
    tok += 1

    # Active-entity tokens: dedicated one-hots for the currently-selected
    # player / corp / company. Factored out so the per-entity tokens can stay
    # permutation-equivariant (each entity's own token no longer carries an
    # "am I active?" bit) and so the model can attend to the selector directly.
    _fill_active_player_token(state, buffer, tok)
    tok += 1
    _fill_active_corp_token(state, buffer, tok)
    tok += 1
    _fill_active_company_token(state, buffer, tok)
    tok += 1

    # Phase token (decision-phase one-hot)
    _fill_phase_token(state, buffer, tok)
    tok += 1

    # Num-players token (3/4/5 one-hot)
    _fill_num_players_token(buffer, tok, num_players)
    tok += 1

    # Game-progress token (CoO + end-card + cards-remaining)
    _fill_game_progress_token(state, buffer, tok)
    tok += 1

    # --- Phase-specific tokens (left zero when the current phase doesn't match) ---

    if phase == <int>GamePhases.PHASE_INVEST:
        _fill_invest_token(state, buffer, tok)
    tok += 1

    if phase == <int>GamePhases.PHASE_BID:
        _fill_auction_token(state, buffer, tok)
    tok += 1

    if phase == <int>GamePhases.PHASE_DIVIDENDS:
        _fill_dividend_token(state, buffer, tok)
    tok += 1

    if phase == <int>GamePhases.PHASE_ISSUE_SHARES:
        _fill_issue_token(state, buffer, tok)
    tok += 1

    if phase == <int>GamePhases.PHASE_IPO or phase == <int>GamePhases.PHASE_PAR:
        _fill_par_token(state, buffer, tok)
    tok += 1

    if phase == <int>GamePhases.PHASE_ACQ_OFFER:
        _fill_acq_offer_token(state, buffer, tok)
    tok += 1

    if phase == <int>GamePhases.PHASE_ACQ_SELECT_PRICE:
        _fill_acq_price_info_token(state, buffer, tok)
    tok += 1

    # --- Corp tokens ---
    for i in range(NUM_CORPS):
        _fill_corp_token(state, buffer, tok, i, num_players)
        tok += 1

    # --- Player tokens (last; trailing slot makes higher-player padding easy) ---
    for i in range(num_players):
        _fill_player_token(state, buffer, tok, i, num_players)
        tok += 1


# =============================================================================
# PLAYER TOKEN
# =============================================================================

cdef void _fill_player_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
    int player_id,
    int num_players,
) noexcept nogil:
    # Feature offsets within the player token. Active-player selection lives
    # in the dedicated ``active_player`` token, not here.
    cdef int OFF_PLAYER_ID    = 0    # 5 slots
    cdef int OFF_TURN_ORDER   = 5    # 5 slots
    cdef int OFF_HAS_PASSED   = 10
    cdef int OFF_CASH         = 11
    cdef int OFF_NET_WORTH    = 12
    cdef int OFF_LIQUIDITY    = 13
    cdef int OFF_INCOME       = 14
    cdef int OFF_SHARES       = 15   # 8 slots
    cdef int OFF_ROUND_TRIPS  = 23
    cdef int OFF_SHARE_BUYS   = 24   # 8 slots
    cdef int OFF_SHARE_SELLS  = 32   # 8 slots
    cdef int OFF_PRESIDENCIES = 40   # 8 slots
    cdef int OFF_COMPANIES    = 48   # 36 slots

    cdef int player_base = LAYOUT.players_offset + player_id * PLAYER_FIELDS.size
    cdef int turn_order = <int>state._data[player_base + PLAYER_FIELDS.turn_order]
    cdef int has_passed = <int>state._data[player_base + PLAYER_FIELDS.has_passed]
    cdef int cash = <int>state._data[player_base + PLAYER_FIELDS.cash]
    cdef int net_worth = <int>state._data[player_base + PLAYER_FIELDS.net_worth]
    cdef int liquidity = <int>state._data[player_base + PLAYER_FIELDS.liquidity]
    cdef int income = <int>state._data[player_base + PLAYER_FIELDS.income]
    cdef int c, shares, buys, sells, roundtrip_flag

    # Player ID one-hot (padded to 5 slots). player_id is always
    # 0 <= player_id < num_players <= MAX_MODEL_PLAYERS — the caller loops
    # over range(num_players) and the entry assert pins num_players to [3,5].
    assert 0 <= player_id < MAX_MODEL_PLAYERS, \
        f"_fill_player_token: player_id {player_id} out of range"
    buffer[tok, OFF_PLAYER_ID + player_id] = 1.0

    # Turn order one-hot (padded to 5 slots)
    assert 0 <= turn_order < MAX_MODEL_PLAYERS, \
        f"_fill_player_token: turn_order {turn_order} out of range for player {player_id}"
    buffer[tok, OFF_TURN_ORDER + turn_order] = 1.0

    # Has passed
    buffer[tok, OFF_HAS_PASSED] = 1.0 if has_passed else 0.0

    # Financials
    buffer[tok, OFF_CASH] = <float>cash / CASH_DIVISOR
    buffer[tok, OFF_NET_WORTH] = <float>net_worth / NET_WORTH_DIVISOR
    buffer[tok, OFF_LIQUIDITY] = <float>liquidity / NET_WORTH_DIVISOR
    buffer[tok, OFF_INCOME] = <float>income / ENTITY_INCOME_DIVISOR

    # Per-corp: shares, buys, sells, presidency, round-trip flag
    roundtrip_flag = 0
    for c in range(NUM_CORPS):
        shares = <int>state._data[player_base + PLAYER_FIELDS.owned_shares + c]
        buys = <int>state._data[player_base + PLAYER_FIELDS.share_buys + c]
        sells = <int>state._data[player_base + PLAYER_FIELDS.share_sells + c]

        buffer[tok, OFF_SHARES + c] = <float>shares / SHARE_DIVISOR
        buffer[tok, OFF_SHARE_BUYS + c] = <float>buys / SHARE_DIVISOR
        buffer[tok, OFF_SHARE_SELLS + c] = <float>sells / SHARE_DIVISOR

        # Presidency: 1.0 if this player is the corp's president
        if corp_president_id(state, c) == player_id:
            buffer[tok, OFF_PRESIDENCIES + c] = 1.0

        # Round-trip threshold: once the player hits the buy+sell cap on
        # any corp, any further buy/sell in that corp is illegal this turn.
        if buys >= ROUNDTRIP_LIMIT or sells >= ROUNDTRIP_LIMIT:
            roundtrip_flag = 1

    buffer[tok, OFF_ROUND_TRIPS] = 1.0 if roundtrip_flag else 0.0

    # Owned companies (36 flags)
    cdef int comp_loc
    cdef int comp_owner
    for c in range(NUM_COMPANIES):
        comp_loc = company_location(state, c)
        if comp_loc != <int>LOC_PLAYER:
            continue
        comp_owner = company_owner_id(state, c)
        if comp_owner == player_id:
            buffer[tok, OFF_COMPANIES + c] = 1.0


# =============================================================================
# CORP TOKEN
# =============================================================================

cdef void _fill_corp_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
    int corp_id,
    int num_players,
) noexcept nogil:
    # Feature offsets within the corp token. Active-corp selection lives in
    # the dedicated ``active_corp`` token, not here.
    cdef int OFF_CORP_ID       = 0    # 8 slots
    cdef int OFF_ACTIVE        = 8
    cdef int OFF_IN_RECV       = 9
    cdef int OFF_PASSED_ACQ    = 10
    cdef int OFF_UNISSUED      = 11
    cdef int OFF_ISSUED        = 12
    cdef int OFF_BANK          = 13
    cdef int OFF_PRICE_IDX     = 14   # 27 slots
    cdef int OFF_SHARE_PRICE   = 41
    cdef int OFF_PENDING_MOVE  = 42
    cdef int OFF_CASH          = 43
    cdef int OFF_ACQ_PROCEEDS  = 44
    cdef int OFF_INCOME        = 45
    cdef int OFF_STARS         = 46
    cdef int OFF_RAW_REVENUE   = 47
    cdef int OFF_SYNERGY       = 48
    cdef int OFF_COO_COST      = 49
    cdef int OFF_ABILITY       = 50
    cdef int OFF_PRESIDENT     = 51   # 5 slots
    cdef int OFF_COMPANIES     = 56   # 36 slots

    cdef bint active = corp_is_active(state, corp_id)
    cdef int price_idx, president, company_id

    # Corp ID one-hot
    buffer[tok, OFF_CORP_ID + corp_id] = 1.0

    buffer[tok, OFF_ACTIVE] = 1.0 if active else 0.0

    # Flags and share counts are meaningful regardless of active status
    buffer[tok, OFF_IN_RECV] = 1.0 if corp_is_in_receivership(state, corp_id) else 0.0
    buffer[tok, OFF_PASSED_ACQ] = 1.0 if corp_has_passed_acq_offer(state, corp_id) else 0.0
    buffer[tok, OFF_UNISSUED] = <float>corp_unissued_shares(state, corp_id) / SHARE_DIVISOR
    buffer[tok, OFF_ISSUED] = <float>corp_issued_shares(state, corp_id) / SHARE_DIVISOR
    buffer[tok, OFF_BANK] = <float>corp_bank_shares(state, corp_id) / SHARE_DIVISOR

    if active:
        price_idx = corp_price_index(state, corp_id)
        assert 0 <= price_idx < NUM_MARKET_SPACES, \
            f"_fill_corp_token: price_idx {price_idx} out of range for active corp {corp_id}"
        buffer[tok, OFF_PRICE_IDX + price_idx] = 1.0
        buffer[tok, OFF_SHARE_PRICE] = <float>corp_share_price(state, corp_id) / SHARE_PRICE_DIVISOR
        buffer[tok, OFF_PENDING_MOVE] = <float>corp_pending_price_move(state, corp_id) / IMPACT_DIVISOR
        buffer[tok, OFF_CASH] = <float>corp_cash(state, corp_id) / CASH_DIVISOR
        buffer[tok, OFF_ACQ_PROCEEDS] = <float>corp_acquisition_proceeds(state, corp_id) / CASH_DIVISOR
        buffer[tok, OFF_INCOME] = <float>corp_income(state, corp_id) / ENTITY_INCOME_DIVISOR
        buffer[tok, OFF_STARS] = <float>corp_total_stars(state, corp_id) / CORP_STAR_DIVISOR
        buffer[tok, OFF_RAW_REVENUE] = <float>corp_raw_revenue(state, corp_id) / ENTITY_INCOME_DIVISOR
        buffer[tok, OFF_SYNERGY] = <float>corp_synergy_income(state, corp_id) / ENTITY_INCOME_DIVISOR
        buffer[tok, OFF_COO_COST] = <float>corp_coo_cost(state, corp_id) / ENTITY_INCOME_DIVISOR
        buffer[tok, OFF_ABILITY] = <float>corp_ability_income(state, corp_id) / ENTITY_INCOME_DIVISOR

        # President one-hot (inactive/receivership corps leave this zero)
        if not corp_is_in_receivership(state, corp_id):
            president = corp_president_id(state, corp_id)
            assert 0 <= president < MAX_MODEL_PLAYERS, \
                f"_fill_corp_token: president {president} out of range for active corp {corp_id}"
            buffer[tok, OFF_PRESIDENT + president] = 1.0

        # Owned companies (36 flags — owned OR in acquisition pile)
        for company_id in range(NUM_COMPANIES):
            if corp_owns_company(state, corp_id, company_id):
                buffer[tok, OFF_COMPANIES + company_id] = 1.0
            elif corp_has_acquisition_company(state, corp_id, company_id):
                buffer[tok, OFF_COMPANIES + company_id] = 1.0


# =============================================================================
# COMPANY TOKEN
# =============================================================================

cdef void _fill_company_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
    int company_id,
    int num_players,
) noexcept nogil:
    # The company token now carries only static (game-setup) data. Active-
    # company selection lives in the ``active_company`` token; ownership lives
    # in the player / corp / FI tokens; location lives in the four per-location
    # tokens; CoO-adjusted income lives in the ``company_adjusted_income`` token.
    cdef int OFF_COMPANY_ID     = 0    # 36 slots
    cdef int OFF_LOW_PRICE      = 36
    cdef int OFF_FACE_VALUE     = 37
    cdef int OFF_HIGH_PRICE     = 38
    cdef int OFF_LOW_HIGH_DIFF  = 39
    cdef int OFF_BASE_INCOME    = 40
    cdef int OFF_STARS          = 41
    cdef int OFF_SYNERGIES      = 42   # 36 slots

    cdef int k

    # Company ID one-hot
    buffer[tok, OFF_COMPANY_ID + company_id] = 1.0

    # Static data
    buffer[tok, OFF_LOW_PRICE] = <float>COMPANY_LOW_PRICE[company_id] / COMPANY_PRICE_DIVISOR
    buffer[tok, OFF_FACE_VALUE] = <float>COMPANY_FACE_VALUE[company_id] / COMPANY_PRICE_DIVISOR
    buffer[tok, OFF_HIGH_PRICE] = <float>COMPANY_HIGH_PRICE[company_id] / COMPANY_PRICE_DIVISOR
    # low_high_diff: count of valid ACQ_SELECT_PRICE offsets for this company
    # (high - low + 1). Matches the ``max_off`` ceiling the acq-price head
    # conditions on. Max 51 (CDG: 80 - 30 + 1).
    buffer[tok, OFF_LOW_HIGH_DIFF] = (
        <float>(COMPANY_HIGH_PRICE[company_id] - COMPANY_LOW_PRICE[company_id] + 1)
        / PRICE_RANGE_DIVISOR
    )
    buffer[tok, OFF_BASE_INCOME] = <float>COMPANY_INCOME[company_id] / COMPANY_INCOME_DIVISOR
    buffer[tok, OFF_STARS] = <float>COMPANY_STARS[company_id] / COMPANY_STAR_DIVISOR

    # Synergies: value of synergy with each other company (in either
    # direction). The matrix is directional, so we take the max-magnitude
    # direction to surface the bonus regardless of which side holds it.
    # Use a dedicated divisor here: some static synergy entries reach 16,
    # which is larger than the base-income normalization range.
    cdef int syn_ab, syn_ba, syn
    for k in range(NUM_COMPANIES):
        if k == company_id:
            continue
        syn_ab = COMPANY_SYNERGY[company_id][k]
        syn_ba = COMPANY_SYNERGY[k][company_id]
        syn = syn_ab if syn_ab >= syn_ba else syn_ba
        if syn != 0:
            buffer[tok, OFF_SYNERGIES + k] = <float>syn / COMPANY_SYNERGY_DIVISOR


# =============================================================================
# FI TOKEN
# =============================================================================

cdef void _fill_fi_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int OFF_CASH      = 0
    cdef int OFF_INCOME    = 1
    cdef int OFF_COMPANIES = 2   # 36 slots

    cdef int cash = <int>state._data[LAYOUT.fi_offset + FI_OFFSETS.cash]
    cdef int income = <int>state._data[LAYOUT.fi_offset + FI_OFFSETS.income]
    cdef int c

    buffer[tok, OFF_CASH] = <float>cash / CASH_DIVISOR
    buffer[tok, OFF_INCOME] = <float>income / ENTITY_INCOME_DIVISOR

    for c in range(NUM_COMPANIES):
        if company_location(state, c) == <int>LOC_FI:
            buffer[tok, OFF_COMPANIES + c] = 1.0


# =============================================================================
# COMPANY-LOCATION TOKENS
# =============================================================================

cdef void _fill_company_location_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
    int target_location,
) noexcept nogil:
    # 36-slot bitmap: buffer[tok, c] = 1.0 iff company c is at ``target_location``.
    cdef int c
    for c in range(NUM_COMPANIES):
        if company_location(state, c) == target_location:
            buffer[tok, c] = 1.0


cdef void _fill_company_removed_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    # Same shape as the other company-location bitmaps, but with two kinds
    # of "removed":
    #   * LOC_REMOVED — closed during play.
    #   * LOC_EXCLUDED — filtered out at setup, BUT only once the deck has
    #     revealed this: CoO level L means the current top-of-deck tier is
    #     L (red=1 … blue=5), so a star-S company is publicly known-gone
    #     iff coo > S. Flagging LOC_EXCLUDED unconditionally would leak
    #     which specific cards were cut at setup — information the players
    #     don't have until the deck has advanced past that tier.
    cdef int c, loc
    cdef int coo = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.coo_level]
    for c in range(NUM_COMPANIES):
        loc = company_location(state, c)
        if loc == <int>LOC_REMOVED:
            buffer[tok, c] = 1.0
        elif loc == <int>LOC_EXCLUDED and coo > COMPANY_STARS[c]:
            buffer[tok, c] = 1.0


cdef void _fill_company_adjusted_income_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    # 36-slot vector: per-company CoO-adjusted income, normalized by
    # COMPANY_INCOME_DIVISOR. Values may be negative (a company's income can
    # be pushed below zero by CoO-dependent penalties).
    cdef int c
    for c in range(NUM_COMPANIES):
        buffer[tok, c] = (
            <float>company_adjusted_income(state, c) / COMPANY_INCOME_DIVISOR
        )


# =============================================================================
# ACTIVE-ENTITY TOKENS
# =============================================================================

cdef void _fill_active_player_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    # 5-slot one-hot (padded to MAX_MODEL_PLAYERS). All zero when no active
    # player is selected (active_player == -1 on automated/terminal phases).
    # Any value outside [-1, MAX_MODEL_PLAYERS) is an engine invariant break.
    cdef int active_player = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_player]
    assert -1 <= active_player < MAX_MODEL_PLAYERS, \
        f"_fill_active_player_token: active_player {active_player} out of [-1, {MAX_MODEL_PLAYERS})"
    if active_player >= 0:
        buffer[tok, active_player] = 1.0


cdef void _fill_active_corp_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    # 8-slot one-hot; all zero when no active corp is selected. Any value
    # outside [-1, NUM_CORPS) is an engine invariant break.
    cdef int active_corp = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_corp]
    assert -1 <= active_corp < NUM_CORPS, \
        f"_fill_active_corp_token: active_corp {active_corp} out of [-1, {NUM_CORPS})"
    if active_corp >= 0:
        buffer[tok, active_corp] = 1.0


cdef void _fill_active_company_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    # 36-slot one-hot; all zero when no active company is selected. Any value
    # outside [-1, NUM_COMPANIES) is an engine invariant break.
    cdef int active_company = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_company]
    assert -1 <= active_company < NUM_COMPANIES, \
        f"_fill_active_company_token: active_company {active_company} out of [-1, {NUM_COMPANIES})"
    if active_company >= 0:
        buffer[tok, active_company] = 1.0


# =============================================================================
# MARKET TOKENS
# =============================================================================

cdef void _fill_market_availability_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int i
    # Market section is a flat run of 27 int16 0/1 availability flags;
    # direct cast is enough (no need for a ternary).
    for i in range(NUM_MARKET_SPACES):
        buffer[tok, i] = <float>state._data[LAYOUT.market_offset + i]


cdef void _fill_market_slot_prices_token(
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    # Static $0..$75 slot prices normalized by SHARE_PRICE_DIVISOR. Written
    # every call — it's a short loop and the token carries no state, so a
    # rotating "static constants" token is simpler than gating on a first-use
    # flag.
    cdef int i
    for i in range(NUM_MARKET_SPACES):
        buffer[tok, i] = <float>MARKET_PRICES[i] / SHARE_PRICE_DIVISOR


# =============================================================================
# PHASE / NUM_PLAYERS / GAME_PROGRESS TOKENS
# =============================================================================

cdef void _fill_phase_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    # Phase one-hot over decision phases — one slot per DecisionPhase.
    # Automated / terminal engine phases map to -1 and leave all slots zero.
    cdef int phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]
    cdef int decision_phase

    assert 0 <= phase < <int>GameConstants.NUM_PHASES, \
        f"_fill_phase_token: corrupt engine phase {phase}"
    decision_phase = ENGINE_TO_DECISION_PHASE[phase]
    # Automated / terminal engine phases map to -1; anything else is a
    # corrupt ENGINE_TO_DECISION_PHASE entry.
    assert -1 <= decision_phase < NUM_DECISION_PHASES, \
        f"_fill_phase_token: decision_phase {decision_phase} out of [-1, {NUM_DECISION_PHASES}) for engine phase {phase}"
    if decision_phase >= 0:
        buffer[tok, decision_phase] = 1.0


cdef void _fill_num_players_token(
    float[:, ::1] buffer,
    int tok,
    int num_players,
) noexcept nogil:
    # num_players one-hot: slot 0 = 3p, slot 1 = 4p, slot 2 = 5p.
    # Training scope is 3-5p; the entry-point assert already rejects
    # anything outside that range, so re-assert here rather than silently
    # leaving the token zero for mis-sized states.
    assert 3 <= num_players <= 5, \
        f"_fill_num_players_token: num_players {num_players} out of [3, 5]"
    buffer[tok, num_players - 3] = 1.0


cdef void _fill_game_progress_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int OFF_COO            = 0   # 7 slots (CoO level 1..7 → slots 0..6)
    cdef int OFF_END_CARD       = 7
    cdef int OFF_CARDS_REM      = 8

    cdef int coo = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.coo_level]
    cdef int end_card = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.end_card_flipped]
    cdef int cards_rem = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.cards_remaining]

    # CoO one-hot: levels 1-7 → slots 0-6
    assert 1 <= coo <= NUM_COO_LEVELS, \
        f"_fill_game_progress_token: coo_level {coo} out of [1, 7]"
    buffer[tok, OFF_COO + (coo - 1)] = 1.0

    buffer[tok, OFF_END_CARD] = 1.0 if end_card else 0.0
    buffer[tok, OFF_CARDS_REM] = <float>cards_rem / <float>NUM_COMPANIES


# =============================================================================
# PHASE-SPECIFIC TOKENS
# =============================================================================

cdef void _fill_invest_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int OFF_PASSES     = 0
    cdef int OFF_BUY_IMPACT = 1   # 8 slots
    cdef int OFF_SELL_IMPACT = 9  # 8 slots

    cdef int passes = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.consecutive_passes]
    buffer[tok, OFF_PASSES] = <float>passes / CONSECUTIVE_PASSES_DIVISOR

    cdef int c, current_idx, new_idx, delta
    for c in range(NUM_CORPS):
        if not corp_is_active(state, c):
            continue
        current_idx = corp_price_index(state, c)

        # Buy impact: delta to the next higher available market space.
        new_idx = market_find_next_higher_space(state, current_idx)
        delta = new_idx - current_idx
        buffer[tok, OFF_BUY_IMPACT + c] = <float>delta / IMPACT_DIVISOR

        # Sell impact: delta to the next lower available market space.
        new_idx = market_find_next_lower_space(state, current_idx)
        delta = new_idx - current_idx
        buffer[tok, OFF_SELL_IMPACT + c] = <float>delta / IMPACT_DIVISOR


cdef void _fill_auction_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    # Price slots carry the *minimum legal next bid* rather than the last
    # bid placed. On the opening bid (is_first_bid == 1) the minimum is
    # face_value (offset 0); afterwards it's current_bid + 1.
    cdef int OFF_MIN_BID_IDX   = 0
    cdef int OFF_MIN_BID_VALUE = 1
    cdef int OFF_IS_FIRST_BID  = 2
    cdef int OFF_HIGH_BIDDER   = 3   # 5 slots
    cdef int OFF_STARTER       = 8   # 5 slots

    cdef int auction_price = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_price]
    cdef int high_bidder = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_high_bidder]
    cdef int starter = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_starter]
    cdef int active_company = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_company]
    cdef bint is_first_bid = high_bidder < 0
    cdef int face_value, min_bid, min_offset

    # active_company is seeded in INVEST before the BID transition and only
    # cleared at auction resolution; any BID state without it is a driver bug.
    assert 0 <= active_company < NUM_COMPANIES, \
        f"_fill_auction_token: active_company {active_company} unset or out of range in BID"

    face_value = COMPANY_FACE_VALUE[active_company]
    if is_first_bid:
        min_bid = face_value
    else:
        min_bid = auction_price + 1
    min_offset = min_bid - face_value
    buffer[tok, OFF_MIN_BID_IDX] = <float>min_offset / <float>AUCTION_CAP_INT
    buffer[tok, OFF_MIN_BID_VALUE] = <float>min_bid / COMPANY_PRICE_DIVISOR

    buffer[tok, OFF_IS_FIRST_BID] = 1.0 if is_first_bid else 0.0

    # high_bidder legitimately == -1 on the opening bid; starter is stamped
    # whenever we reach PHASE_BID and must be a live player slot.
    assert high_bidder < MAX_MODEL_PLAYERS, \
        f"_fill_auction_token: high_bidder {high_bidder} >= MAX_MODEL_PLAYERS"
    if high_bidder >= 0:
        buffer[tok, OFF_HIGH_BIDDER + high_bidder] = 1.0
    assert 0 <= starter < MAX_MODEL_PLAYERS, \
        f"_fill_auction_token: auction_starter {starter} unset or out of range in BID"
    buffer[tok, OFF_STARTER + starter] = 1.0


cdef void _fill_dividend_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int OFF_IMPACT    = 0    # 26 slots (amounts 0..25)
    cdef int OFF_REMAINING = 26   # 8 slots

    cdef int active_corp = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_corp]
    cdef int amount, price_move
    cdef int c

    # Dividend impacts. Driver contract for PHASE_DIVIDENDS: active_corp is
    # always a live, active corp when this token is filled.
    assert 0 <= active_corp < NUM_CORPS, \
        f"_fill_dividend_token: active_corp {active_corp} unset or out of range"
    assert corp_is_active(state, active_corp), \
        f"_fill_dividend_token: active_corp {active_corp} not active"
    for amount in range(MAX_DIVIDEND):
        price_move = _simulate_dividend_price_move(state, active_corp, amount)
        buffer[tok, OFF_IMPACT + amount] = <float>price_move / IMPACT_DIVISOR

    # Per-corp dividend-remaining flags
    for c in range(NUM_CORPS):
        if (<int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.dividend_remaining + c]) != 0:
            buffer[tok, OFF_REMAINING + c] = 1.0


cdef void _fill_issue_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int OFF_IMPACT    = 0
    cdef int OFF_REMAINING = 1    # 8 slots

    cdef int active_corp = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_corp]
    cdef int current_idx, new_idx, delta
    cdef int c

    # Issue impact: issuing one share drops the corp's price like a sell,
    # except Stock Masters (SM) has no price change on issue. Driver contract
    # for PHASE_ISSUE_SHARES: active_corp is always a live, active corp.
    assert 0 <= active_corp < NUM_CORPS, \
        f"_fill_issue_token: active_corp {active_corp} unset or out of range"
    assert corp_is_active(state, active_corp), \
        f"_fill_issue_token: active_corp {active_corp} not active"
    if active_corp == <int>CorpIndices.CORP_SM:
        buffer[tok, OFF_IMPACT] = 0.0
    else:
        current_idx = corp_price_index(state, active_corp)
        new_idx = market_find_next_lower_space(state, current_idx)
        delta = new_idx - current_idx
        buffer[tok, OFF_IMPACT] = <float>delta / IMPACT_DIVISOR

    # Per-corp issue-remaining flags
    for c in range(NUM_CORPS):
        if (<int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.issue_remaining + c]) != 0:
            buffer[tok, OFF_REMAINING + c] = 1.0


cdef void _fill_par_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int OFF_PLAYER_CASH   = 0    # 14 slots
    cdef int OFF_CORP_CASH     = 14   # 14 slots
    cdef int OFF_ISSUED_SHARES = 28   # 14 slots
    cdef int OFF_REMAINING     = 42   # 8 slots (inactive corps)

    cdef int active_company = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_company]
    cdef int face_value, star_tier, par_index
    cdef int float_shares, market_index, player_payment, corp_cash_result, issued
    cdef int c

    # Driver contract for PHASE_IPO / PHASE_PAR: active_company is always
    # stamped to the selected corp's target company, and every company's
    # star tier is 1..5 by static data.
    assert 0 <= active_company < NUM_COMPANIES, \
        f"_fill_par_token: active_company {active_company} unset or out of range"
    face_value = COMPANY_FACE_VALUE[active_company]
    star_tier = COMPANY_STARS[active_company]
    assert 1 <= star_tier <= MAX_MODEL_PLAYERS, \
        f"_fill_par_token: star_tier {star_tier} out of [1, 5] for company {active_company}"
    for par_index in range(NUM_PAR_PRICES):
        if PAR_PRICE_VALID[star_tier - 1][par_index] == 0:
            continue
        # Canonical IPO simulation — same helper used by
        # ``phases/ipo.pyx::_process_ipo``.
        (float_shares, market_index, player_payment,
         corp_cash_result, issued) = _simulate_float(face_value, par_index)

        buffer[tok, OFF_PLAYER_CASH + par_index] = (
            <float>player_payment / CASH_DIVISOR
        )
        buffer[tok, OFF_CORP_CASH + par_index] = (
            <float>corp_cash_result / CASH_DIVISOR
        )
        buffer[tok, OFF_ISSUED_SHARES + par_index] = (
            <float>issued / FLOAT_SHARES_MAX
        )

    # IPO remaining: flag each corp that has not yet been floated (still
    # inactive) and is therefore available to be selected for an IPO.
    for c in range(NUM_CORPS):
        if not corp_is_active(state, c):
            buffer[tok, OFF_REMAINING + c] = 1.0


cdef void _fill_acq_offer_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int OFF_PRICE_IDX   = 0
    cdef int OFF_PRICE_VALUE = 1
    cdef int OFF_OFFER_CORP  = 2   # 8 slots
    cdef int OFF_FI_COMPANY  = 10

    cdef int offer_price = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.acq_offer_price]
    cdef int offer_corp = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.acq_offer_corp]
    cdef int active_company = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_company]
    cdef int low_price, offset

    # Driver contract for PHASE_ACQ_OFFER: both selectors are stamped
    # (active_company by SELECT_COMPANY, offer_corp by SELECT_PRICE).
    assert 0 <= active_company < NUM_COMPANIES, \
        f"_fill_acq_offer_token: active_company {active_company} unset or out of range"
    assert 0 <= offer_corp < NUM_CORPS, \
        f"_fill_acq_offer_token: acq_offer_corp {offer_corp} unset or out of range"

    low_price = COMPANY_LOW_PRICE[active_company]
    offset = offer_price - low_price
    buffer[tok, OFF_PRICE_IDX] = <float>offset / <float>ACQ_PRICE_OFFSETS
    if company_location(state, active_company) == <int>LOC_FI:
        buffer[tok, OFF_FI_COMPANY] = 1.0

    buffer[tok, OFF_PRICE_VALUE] = <float>offer_price / COMPANY_PRICE_DIVISOR
    buffer[tok, OFF_OFFER_CORP + offer_corp] = 1.0


cdef void _fill_acq_price_info_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    """Fill the acq_price_info token during PHASE_ACQ_SELECT_PRICE.

    Kept minimal: every (active_corp, active_company)-level scalar is
    already on the corp / company / active-entity / company-location
    tokens and reaches the price head via attention. The three slots
    here carry only what can't be read off those tokens directly:
      * max_offset     — ACQ price-offset count for the target, same as
                         the company token's ``low_high_diff`` field
                         ((high - low + 1), normalized by PRICE_RANGE_DIVISOR).
      * fi_flag        — 1 if the target company is FI-owned, else 0.
                         A hard discontinuity for the head (FI sale is a
                         single fixed-price action, no offset to pick).
      * total_synergies — marginal synergy income the corp would gain
                         by adding this company to its portfolio,
                         normalized by ENTITY_INCOME_DIVISOR.
    """
    cdef int OFF_MAX_OFFSET      = 0
    cdef int OFF_FI_FLAG         = 1
    cdef int OFF_TOTAL_SYNERGIES = 2

    cdef int active_corp = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_corp]
    cdef int active_company = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_company]

    # Driver contract for PHASE_ACQ_SELECT_PRICE: both selectors stamped,
    # corp active. SELECT_COMPANY's enumerator also rejects receivership
    # sellers, so any LOC_CORP target reaching here has a live seller.
    assert 0 <= active_corp < NUM_CORPS, \
        f"_fill_acq_price_info_token: active_corp {active_corp} unset or out of range"
    assert 0 <= active_company < NUM_COMPANIES, \
        f"_fill_acq_price_info_token: active_company {active_company} unset or out of range"
    assert corp_is_active(state, active_corp), \
        f"_fill_acq_price_info_token: active_corp {active_corp} not active"

    cdef int low_price = COMPANY_LOW_PRICE[active_company]
    cdef int high_price = COMPANY_HIGH_PRICE[active_company]
    cdef int synergy_delta = corp_candidate_synergy_delta(
        state, active_corp, active_company,
    )

    buffer[tok, OFF_MAX_OFFSET] = (
        <float>(high_price - low_price + 1) / PRICE_RANGE_DIVISOR
    )
    if company_location(state, active_company) == <int>LOC_FI:
        buffer[tok, OFF_FI_FLAG] = 1.0
    buffer[tok, OFF_TOTAL_SYNERGIES] = (
        <float>synergy_delta / ENTITY_INCOME_DIVISOR
    )
