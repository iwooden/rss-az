"""
Token data extraction: compact GameState -> transformer eval buffer.

``get_token_data`` is the sole engine→NN interface: it fills a
(num_tokens, TOKEN_DIM) float32 buffer with normalized per-token
features from a compact GameState. It is called once per NN evaluation,
so the trunk + MCTS throughput depends on this being fast.

Token order (matches ``nn/transformer.py``):
    [players..., corps..., companies..., FI, market, global,
     invest, auction, dividend, issue, par, acq_offer, acq_price_info]

Total tokens in the buffer = num_players + 54. The model concatenates 7
learned pass-token anchors to the projected trunk sequence internally
(one per pass-using decision phase — INVEST, BID, ACQ_SELECT_CORP,
ACQ_OFFER, CLOSING, ISSUE, IPO — the rest have no pass action), so the
input buffer carries no pass rows.

Per-token feature layouts (sum of widths ≤ TOKEN_DIM = 97):

  Player (85):  active_player (1) + player_id onehot (5) + turn_order
                onehot (5) + has_passed (1) + cash (1) + net_worth (1)
                + liquidity (1) + income (1) + owned_shares (8) +
                round_trips (1) + share_buys (8) + share_sells (8) +
                presidencies (8) + owned_companies (36)
  Corp   (93):  active_corp (1) + corp_id onehot (8) + active (1) +
                in_receivership (1) + passed_acq_offer (1) +
                unissued/issued/bank shares (3) + price_index onehot (27)
                + share_price (1) + pending_price_move (1) + cash (1) +
                acq_proceeds (1) + income (1) + stars (1) + raw_revenue
                (1) + synergy_income (1) + coo_cost (1) + ability_income
                (1) + president_id onehot (5) + owned_companies (36)
  Company (97): active_company (1) + company_id onehot (36) +
                corp_owner onehot (8) + player_owner onehot (5) +
                fi_owned (1) + location flags (4) + adjusted_income (1)
                + static data [low/face/high/income/stars] (5) +
                synergies (36)
  FI      (38): cash + income + owned_companies (36)
  Market  (27): availability flags (27)
  Global  (23): num_players onehot (3) + phase onehot (11) + CoO onehot
                (7) + end_card_flipped + cards_remaining
  Invest  (17): consecutive_passes + buy_impacts (8) + sell_impacts (8)
  Auction (13): min_bid_index + min_bid_value + is_first_bid +
                high_bidder onehot (5) + starter onehot (5)
  Divd    (34): dividend_impacts (26) + dividend_remaining (8)
  Issue    (9): issue_impact + issue_remaining (8)
  IPO     (50): player_cash_required (14) + resulting_corp_cash (14) +
                resulting_issued_shares (14) + ipo_remaining (8)
  AcqOff  (11): offer_price_index + offer_price + offer_corp onehot (8) +
                fi_company
  AcqPrice (11): corp_cash + corp_share_price + company low/face/high +
                company adjusted_income + company stars + FI flag +
                cross_president flag + receivership_seller flag +
                max_affordable_offset

All values normalized by divisors defined in ``core.data`` (compile-time
floats inlined by the C compiler). Phase-specific tokens are zeroed out
when the current engine phase does not match. The function is designed
so the post-GIL body runs ``nogil``; a small Python-level prologue
forces per-player cache refreshes so the nogil body can read cached
net_worth / liquidity / income slots directly.
"""

from libc.stdint cimport int16_t

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
    PAR_PRICE_VALID,
    CASH_DIVISOR,
    NET_WORTH_DIVISOR,
    COMPANY_INCOME_DIVISOR,
    ENTITY_INCOME_DIVISOR,
    SHARE_DIVISOR,
    COMPANY_PRICE_DIVISOR,
    SHARE_PRICE_DIVISOR,
    COMPANY_STAR_DIVISOR,
    CORP_STAR_DIVISOR,
    IMPACT_DIVISOR,
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
    _simulate_dividend_price_move,
    _simulate_float,
)
from entities.company cimport (
    LOC_AUCTION, LOC_REVEALED, LOC_PLAYER, LOC_FI, LOC_CORP, LOC_CORP_ACQ,
    LOC_REMOVED,
    company_location,
    company_owner_id,
    company_adjusted_income,
)
from entities.market cimport (
    copy_market_availability,
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
    """Input-buffer token count for the given player count (num_players + 54).

    The model-side trunk is wider by 7 (per-phase pass anchors concatenated
    inside ``RSSTransformerNet._project_tokens``), but those rows are
    learned anchors with no input features, so the engine-side buffer
    doesn't carry them.
    """
    return num_players + 54


cpdef void get_token_data(GameState state, float[:, ::1] buffer):
    """Fill ``buffer`` with per-token NN features for ``state``.

    ``buffer`` must be a writable C-contiguous float32 memoryview at least
    ``(num_players + 54, TOKEN_DIM)`` in size. Training is scoped to
    3-5 players; other player counts are rejected.

    The cache-refresh prologue and ``_fill_buffer`` run in a single nogil
    block — refresh goes through the module-level
    ``refresh_player_cache_if_dirty`` helper rather than the Python-level
    ``PLAYERS[i].get_net_worth(state)`` lookup it used to.
    """
    cdef int num_players = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.num_players]
    cdef int num_tokens = num_players + 54
    cdef int i

    assert 3 <= num_players <= 5, \
        f"get_token_data: num_players must be 3-5, got {num_players}"
    assert buffer.shape[0] >= num_tokens, \
        f"get_token_data: buffer rows {buffer.shape[0]} < num_tokens {num_tokens}"
    assert buffer.shape[1] >= <int>TokenDataSize.TOKEN_DIM, \
        f"get_token_data: buffer cols {buffer.shape[1]} < TOKEN_DIM {<int>TokenDataSize.TOKEN_DIM}"

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
        buffer: ``(n, num_players + 54, TOKEN_DIM)`` float32 output, C-contig.
    """
    cdef int n = len(state_arrays)
    cdef int num_tokens = num_players + 54
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
    assert buffer.shape[2] >= <int>TokenDataSize.TOKEN_DIM, \
        f"get_token_data_batch: buffer cols {buffer.shape[2]} < TOKEN_DIM {<int>TokenDataSize.TOKEN_DIM}"

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
    cdef int i, j, tok

    # Zero the region we'll write. Phase-specific tokens rely on this to
    # stay at zero when the current phase does not match.
    for i in range(num_tokens):
        for j in range(<int>TokenDataSize.TOKEN_DIM):
            buffer[i, j] = 0.0

    tok = 0

    # Player tokens
    for i in range(num_players):
        _fill_player_token(state, buffer, tok, i, num_players)
        tok += 1

    # Corp tokens
    for i in range(NUM_CORPS):
        _fill_corp_token(state, buffer, tok, i, num_players)
        tok += 1

    # Company tokens
    for i in range(NUM_COMPANIES):
        _fill_company_token(state, buffer, tok, i, num_players)
        tok += 1

    # FI token
    _fill_fi_token(state, buffer, tok)
    tok += 1

    # Market token
    _fill_market_token(state, buffer, tok)
    tok += 1

    # Global token
    _fill_global_token(state, buffer, tok, num_players)
    tok += 1

    # Phase-specific tokens. Each slot is left at zero when the current
    # engine phase does not match.
    cdef int phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]

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

    # Pass token: no input features — left zero, trunk adds its type embedding.
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
    # Feature offsets within the player token
    cdef int OFF_ACTIVE       = 0
    cdef int OFF_PLAYER_ID    = 1    # 5 slots
    cdef int OFF_TURN_ORDER   = 6    # 5 slots
    cdef int OFF_HAS_PASSED   = 11
    cdef int OFF_CASH         = 12
    cdef int OFF_NET_WORTH    = 13
    cdef int OFF_LIQUIDITY    = 14
    cdef int OFF_INCOME       = 15
    cdef int OFF_SHARES       = 16   # 8 slots
    cdef int OFF_ROUND_TRIPS  = 24
    cdef int OFF_SHARE_BUYS   = 25   # 8 slots
    cdef int OFF_SHARE_SELLS  = 33   # 8 slots
    cdef int OFF_PRESIDENCIES = 41   # 8 slots
    cdef int OFF_COMPANIES    = 49   # 36 slots

    cdef int player_base = LAYOUT.players_offset + player_id * PLAYER_FIELDS.size
    cdef int active_player = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_player]
    cdef int turn_order = <int>state._data[player_base + PLAYER_FIELDS.turn_order]
    cdef int has_passed = <int>state._data[player_base + PLAYER_FIELDS.has_passed]
    cdef int cash = <int>state._data[player_base + PLAYER_FIELDS.cash]
    cdef int net_worth = <int>state._data[player_base + PLAYER_FIELDS.net_worth]
    cdef int liquidity = <int>state._data[player_base + PLAYER_FIELDS.liquidity]
    cdef int income = <int>state._data[player_base + PLAYER_FIELDS.income]
    cdef int c, shares, buys, sells, roundtrip_flag

    # Active player flag
    if active_player == player_id:
        buffer[tok, OFF_ACTIVE] = 1.0

    # Player ID one-hot (padded to 5 slots)
    if player_id < MAX_MODEL_PLAYERS:
        buffer[tok, OFF_PLAYER_ID + player_id] = 1.0

    # Turn order one-hot (padded to 5 slots)
    if 0 <= turn_order < MAX_MODEL_PLAYERS:
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
    cdef int OFF_ACTIVE_CORP   = 0
    cdef int OFF_CORP_ID       = 1    # 8 slots
    cdef int OFF_ACTIVE        = 9
    cdef int OFF_IN_RECV       = 10
    cdef int OFF_PASSED_ACQ    = 11
    cdef int OFF_UNISSUED      = 12
    cdef int OFF_ISSUED        = 13
    cdef int OFF_BANK          = 14
    cdef int OFF_PRICE_IDX     = 15   # 27 slots
    cdef int OFF_SHARE_PRICE   = 42
    cdef int OFF_PENDING_MOVE  = 43
    cdef int OFF_CASH          = 44
    cdef int OFF_ACQ_PROCEEDS  = 45
    cdef int OFF_INCOME        = 46
    cdef int OFF_STARS         = 47
    cdef int OFF_RAW_REVENUE   = 48
    cdef int OFF_SYNERGY       = 49
    cdef int OFF_COO_COST      = 50
    cdef int OFF_ABILITY       = 51
    cdef int OFF_PRESIDENT     = 52   # 5 slots
    cdef int OFF_COMPANIES     = 57   # 36 slots

    cdef int active_corp = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_corp]
    cdef bint active = corp_is_active(state, corp_id)
    cdef int price_idx, president, company_id

    # Active corp selector flag
    if active_corp == corp_id:
        buffer[tok, OFF_ACTIVE_CORP] = 1.0

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
        if 0 <= price_idx < NUM_MARKET_SPACES:
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
            if 0 <= president < MAX_MODEL_PLAYERS:
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
    cdef int OFF_ACTIVE_COMPANY = 0
    cdef int OFF_COMPANY_ID     = 1    # 36 slots
    cdef int OFF_CORP_OWNER     = 37   # 8 slots
    cdef int OFF_PLAYER_OWNER   = 45   # 5 slots
    cdef int OFF_FI_OWNED       = 50
    cdef int OFF_LOC_AUCTION    = 51
    cdef int OFF_LOC_REVEALED   = 52
    cdef int OFF_LOC_ACQ_PILE   = 53
    cdef int OFF_LOC_REMOVED    = 54
    cdef int OFF_ADJ_INCOME     = 55
    cdef int OFF_LOW_PRICE      = 56
    cdef int OFF_FACE_VALUE     = 57
    cdef int OFF_HIGH_PRICE     = 58
    cdef int OFF_BASE_INCOME    = 59
    cdef int OFF_STARS          = 60
    cdef int OFF_SYNERGIES      = 61   # 36 slots

    cdef int active_company = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_company]
    cdef int location = company_location(state, company_id)
    cdef int owner = company_owner_id(state, company_id)
    cdef int k

    # Active company selector flag
    if active_company == company_id:
        buffer[tok, OFF_ACTIVE_COMPANY] = 1.0

    # Company ID one-hot
    buffer[tok, OFF_COMPANY_ID + company_id] = 1.0

    # Ownership one-hots
    if location == <int>LOC_CORP or location == <int>LOC_CORP_ACQ:
        if 0 <= owner < NUM_CORPS:
            buffer[tok, OFF_CORP_OWNER + owner] = 1.0
    elif location == <int>LOC_PLAYER:
        if 0 <= owner < MAX_MODEL_PLAYERS:
            buffer[tok, OFF_PLAYER_OWNER + owner] = 1.0
    elif location == <int>LOC_FI:
        buffer[tok, OFF_FI_OWNED] = 1.0

    # Location flags
    if location == <int>LOC_AUCTION:
        buffer[tok, OFF_LOC_AUCTION] = 1.0
    elif location == <int>LOC_REVEALED:
        buffer[tok, OFF_LOC_REVEALED] = 1.0
    elif location == <int>LOC_CORP_ACQ:
        buffer[tok, OFF_LOC_ACQ_PILE] = 1.0
    elif location == <int>LOC_REMOVED:
        buffer[tok, OFF_LOC_REMOVED] = 1.0

    # Adjusted income (reflects current CoO level; can be negative)
    buffer[tok, OFF_ADJ_INCOME] = (
        <float>company_adjusted_income(state, company_id) / COMPANY_INCOME_DIVISOR
    )

    # Static data
    buffer[tok, OFF_LOW_PRICE] = <float>COMPANY_LOW_PRICE[company_id] / COMPANY_PRICE_DIVISOR
    buffer[tok, OFF_FACE_VALUE] = <float>COMPANY_FACE_VALUE[company_id] / COMPANY_PRICE_DIVISOR
    buffer[tok, OFF_HIGH_PRICE] = <float>COMPANY_HIGH_PRICE[company_id] / COMPANY_PRICE_DIVISOR
    buffer[tok, OFF_BASE_INCOME] = <float>COMPANY_INCOME[company_id] / COMPANY_INCOME_DIVISOR
    buffer[tok, OFF_STARS] = <float>COMPANY_STARS[company_id] / COMPANY_STAR_DIVISOR

    # Synergies: value of synergy with each other company (in either
    # direction). The matrix is directional, so we take the max-magnitude
    # direction to surface the bonus regardless of which side holds it.
    cdef int syn_ab, syn_ba, syn
    for k in range(NUM_COMPANIES):
        if k == company_id:
            continue
        syn_ab = COMPANY_SYNERGY[company_id][k]
        syn_ba = COMPANY_SYNERGY[k][company_id]
        syn = syn_ab if syn_ab >= syn_ba else syn_ba
        if syn != 0:
            buffer[tok, OFF_SYNERGIES + k] = <float>syn / COMPANY_INCOME_DIVISOR


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
# MARKET TOKEN
# =============================================================================

cdef void _fill_market_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int i
    # Market section is a flat run of 27 availability flags.
    for i in range(NUM_MARKET_SPACES):
        buffer[tok, i] = 1.0 if state._data[LAYOUT.market_offset + i] else 0.0


# =============================================================================
# GLOBAL TOKEN
# =============================================================================

cdef void _fill_global_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
    int num_players,
) noexcept nogil:
    cdef int OFF_NUM_PLAYERS    = 0   # 3 slots (3p/4p/5p)
    cdef int OFF_PHASE          = 3   # 11 slots (one per decision phase)
    cdef int OFF_COO            = 14  # 7 slots
    cdef int OFF_END_CARD       = 21
    cdef int OFF_CARDS_REM      = 22

    cdef int phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]
    cdef int coo = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.coo_level]
    cdef int end_card = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.end_card_flipped]
    cdef int cards_rem = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.cards_remaining]
    cdef int decision_phase

    # num_players one-hot: slot 0 = 3p, slot 1 = 4p, slot 2 = 5p
    if 3 <= num_players <= 5:
        buffer[tok, OFF_NUM_PLAYERS + (num_players - 3)] = 1.0

    # Phase one-hot over decision phases — one slot per DecisionPhase.
    # Automated / terminal engine phases map to -1 and leave all slots zero.
    assert 0 <= phase < <int>GameConstants.NUM_PHASES, \
        f"_fill_global_token: corrupt engine phase {phase}"
    decision_phase = ENGINE_TO_DECISION_PHASE[phase]
    if 0 <= decision_phase < NUM_DECISION_PHASES:
        buffer[tok, OFF_PHASE + decision_phase] = 1.0

    # CoO one-hot: levels 1-7 → slots 0-6
    if 1 <= coo <= NUM_COO_LEVELS:
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

    if 0 <= high_bidder < MAX_MODEL_PLAYERS:
        buffer[tok, OFF_HIGH_BIDDER + high_bidder] = 1.0
    if 0 <= starter < MAX_MODEL_PLAYERS:
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

    # Dividend impacts — only meaningful if there is an active corp.
    if 0 <= active_corp < NUM_CORPS and corp_is_active(state, active_corp):
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
    # except Stock Masters (SM) has no price change on issue.
    if 0 <= active_corp < NUM_CORPS and corp_is_active(state, active_corp):
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

    if 0 <= active_company < NUM_COMPANIES:
        face_value = COMPANY_FACE_VALUE[active_company]
        star_tier = COMPANY_STARS[active_company]
        if 1 <= star_tier <= MAX_MODEL_PLAYERS:
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

    if 0 <= active_company < NUM_COMPANIES:
        low_price = COMPANY_LOW_PRICE[active_company]
        offset = offer_price - low_price
        buffer[tok, OFF_PRICE_IDX] = <float>offset / <float>ACQ_PRICE_OFFSETS
        if company_location(state, active_company) == <int>LOC_FI:
            buffer[tok, OFF_FI_COMPANY] = 1.0

    buffer[tok, OFF_PRICE_VALUE] = <float>offer_price / COMPANY_PRICE_DIVISOR

    if 0 <= offer_corp < NUM_CORPS:
        buffer[tok, OFF_OFFER_CORP + offer_corp] = 1.0


cdef void _fill_acq_price_info_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    """Fill the acq_price_info token during PHASE_ACQ_SELECT_PRICE.

    The price head is a readout on this single token, so it needs the
    (active_corp, active_company) context the trunk can't otherwise
    concentrate into one position. Kept compact — token is zero outside
    SELECT_PRICE and the active corp/company rows carry the full entity
    detail via attention.
    """
    cdef int OFF_CORP_CASH           = 0
    cdef int OFF_CORP_SHARE_PRICE    = 1
    cdef int OFF_COMP_LOW            = 2
    cdef int OFF_COMP_FACE           = 3
    cdef int OFF_COMP_HIGH           = 4
    cdef int OFF_COMP_ADJ_INCOME     = 5
    cdef int OFF_COMP_STARS          = 6
    cdef int OFF_FI_FLAG             = 7
    cdef int OFF_CROSS_PRESIDENT     = 8
    cdef int OFF_RECEIVERSHIP_SELLER = 9
    cdef int OFF_MAX_AFFORD          = 10

    cdef int active_corp = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_corp]
    cdef int active_company = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_company]
    cdef int active_player = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_player]

    # SELECT_PRICE requires both flags set (driver contract); bail if not.
    if active_corp < 0 or active_corp >= NUM_CORPS:
        return
    if active_company < 0 or active_company >= NUM_COMPANIES:
        return
    if not corp_is_active(state, active_corp):
        return

    cdef int cash = corp_cash(state, active_corp)
    cdef int share_price = corp_share_price(state, active_corp)
    cdef int low_price = COMPANY_LOW_PRICE[active_company]
    cdef int face_value = COMPANY_FACE_VALUE[active_company]
    cdef int high_price = COMPANY_HIGH_PRICE[active_company]
    cdef int adj_income = company_adjusted_income(state, active_company)
    cdef int stars = COMPANY_STARS[active_company]
    cdef int loc = company_location(state, active_company)
    cdef int owner = company_owner_id(state, active_company)
    cdef bint is_fi = loc == <int>LOC_FI
    cdef bint cross_pres = False
    cdef bint recv_seller = False
    cdef int max_off

    buffer[tok, OFF_CORP_CASH] = <float>cash / CASH_DIVISOR
    buffer[tok, OFF_CORP_SHARE_PRICE] = <float>share_price / SHARE_PRICE_DIVISOR
    buffer[tok, OFF_COMP_LOW] = <float>low_price / COMPANY_PRICE_DIVISOR
    buffer[tok, OFF_COMP_FACE] = <float>face_value / COMPANY_PRICE_DIVISOR
    buffer[tok, OFF_COMP_HIGH] = <float>high_price / COMPANY_PRICE_DIVISOR
    buffer[tok, OFF_COMP_ADJ_INCOME] = <float>adj_income / COMPANY_INCOME_DIVISOR
    buffer[tok, OFF_COMP_STARS] = <float>stars / COMPANY_STAR_DIVISOR

    if is_fi:
        buffer[tok, OFF_FI_FLAG] = 1.0
    else:
        # Cross-president fires only under ``acq_same_president=False`` when
        # the seller is a foreign player (LOC_PLAYER) or a foreign-president
        # corp (LOC_CORP). Receivership corps have no president so they do
        # not branch to ACQ_OFFER even in the cross-president variant.
        if not state.acq_same_president:
            if loc == <int>LOC_PLAYER and owner != active_player:
                cross_pres = True
            elif loc == <int>LOC_CORP and not corp_is_in_receivership(state, owner):
                if corp_president_id(state, owner) != active_player:
                    cross_pres = True
        if loc == <int>LOC_CORP and corp_is_in_receivership(state, owner):
            recv_seller = True

    if cross_pres:
        buffer[tok, OFF_CROSS_PRESIDENT] = 1.0
    if recv_seller:
        buffer[tok, OFF_RECEIVERSHIP_SELLER] = 1.0

    # Max affordable offset: 0..50 range for negotiated buys; leave zero
    # for FI targets (price is fixed, decision is a single FI_BUY action).
    if not is_fi:
        max_off = high_price - low_price
        if cash - low_price < max_off:
            max_off = cash - low_price
        if max_off > ACQ_PRICE_OFFSETS - 1:
            max_off = ACQ_PRICE_OFFSETS - 1
        if max_off < 0:
            max_off = 0
        buffer[tok, OFF_MAX_AFFORD] = <float>max_off / <float>(ACQ_PRICE_OFFSETS - 1)
